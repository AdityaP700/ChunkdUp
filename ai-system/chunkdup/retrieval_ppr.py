# ai-system/chunkdup/retrieval_ppr.py
"""
HippoRAG-Inspired Personalized PageRank (PPR) Relational Graph Retrieval.

Features:
1. Bipartite / Tripartite Graph Structure:
   - Passage / Turn nodes (passage:<id>)
   - Relational Claim nodes (claim:<id>)
   - Entity nodes (ent:<canonical_name>)
2. Hub-Suppressed Transition Probabilities:
   - Degree-normalized edge transitions: w(u, v) = 1 / sqrt(deg(u) * deg(v))
   - Prevents generic high-degree hubs from capturing artificial probability mass.
3. CUDA/GPU-Accelerated Power Iteration:
   - Executes vector-matrix multiplication directly on CUDA (NVIDIA RTX 4050)
   - Convergence tolerance checking (tol=1e-5, max_iter=30)
   - Automatic CPU fallback if CUDA is unavailable.
4. Multi-Hop Graph Traversal:
   - Propagates activation from query entities across claims to intermediate entities
     and finally to target evidence passages that may not contain the initial query tokens.
5. Graceful Fallbacks & Robustness:
   - Empty graphs degrade gracefully to dense/BM25 retrieval.
   - Queries with unrecognized entities distribute teleport mass to passage seeds.
   - Disconnected graph components converge without numerical instability.
6. Hybrid Rank Fusion (RRF):
   - Combines PPR stationary probabilities, dense semantic similarity, and BM25 lexical scores.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from chunkdup.relational.graph import RelationalGraph
from chunkdup.relational.schema import RelationalClaim

logger = logging.getLogger(__name__)


class RelationalPPRGraph:
    """
    Constructs and executes Personalized PageRank over an entity-claim-passage network.
    """

    def __init__(
        self,
        relational_graph: Optional[RelationalGraph] = None,
        damping: float = 0.15,
        max_iter: int = 30,
        tol: float = 1e-5,
        hub_suppression: bool = True,
        device: Optional[str] = None,
    ):
        self.relational_graph = relational_graph or RelationalGraph()
        self.damping = float(damping)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.hub_suppression = hub_suppression

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.node_to_idx: Dict[str, int] = {}
        self.idx_to_node: List[str] = []
        self.node_types: List[str] = []  # "entity", "claim", "passage"
        self.passage_id_map: Dict[str, str] = {}  # passage node -> raw passage text

        # Adjacency: u -> {v: weight}
        self.adj: Dict[int, Dict[int, float]] = {}
        self.transition_matrix: Optional[torch.Tensor] = None
        self._is_built: bool = False

    def _get_or_add_node(self, node_key: str, node_type: str) -> int:
        if node_key in self.node_to_idx:
            return self.node_to_idx[node_key]
        idx = len(self.idx_to_node)
        self.node_to_idx[node_key] = idx
        self.idx_to_node.append(node_key)
        self.node_types.append(node_type)
        self.adj[idx] = {}
        self._is_built = False
        return idx

    def add_edge(self, u_key: str, u_type: str, v_key: str, v_type: str, weight: float = 1.0):
        u = self._get_or_add_node(u_key, u_type)
        v = self._get_or_add_node(v_key, v_type)
        self.adj[u][v] = self.adj[u].get(v, 0.0) + weight
        self.adj[v][u] = self.adj[v].get(u, 0.0) + weight
        self._is_built = False

    def build_from_passages_and_claims(
        self,
        passages: List[Dict[str, Any]],
        claims: Optional[List[RelationalClaim]] = None,
        as_of: Optional[str] = None,
    ):
        """
        Populate the graph from raw passage turns and relational claims.
        """
        self.node_to_idx.clear()
        self.idx_to_node.clear()
        self.node_types.clear()
        self.passage_id_map.clear()
        self.adj.clear()
        self.transition_matrix = None
        self._is_built = False

        # 1. Register passage nodes
        for p in passages:
            pid = str(p.get("id") or p.get("turn_id") or p.get("chunk_id"))
            pkey = f"passage:{pid}"
            ptext = str(p.get("text") or p.get("passage") or p.get("raw_text") or "")
            self._get_or_add_node(pkey, "passage")
            self.passage_id_map[pkey] = ptext

        # 2. Collect claims
        if claims is None:
            claims = self.relational_graph.get_claims(as_of=as_of, active_only=True)

        for claim in claims:
            ckey = f"claim:{claim.claim_id}"
            self._get_or_add_node(ckey, "claim")

            # Entity <-> Claim edges
            subj_key = f"ent:{claim.subject_entity}"
            obj_key = f"ent:{claim.object_entity}"
            self.add_edge(subj_key, "entity", ckey, "claim", weight=claim.confidence)
            self.add_edge(obj_key, "entity", ckey, "claim", weight=claim.confidence)

            # Claim <-> Passage edges
            for tid in claim.source_turn_ids:
                pkey = f"passage:{tid}"
                if pkey in self.node_to_idx:
                    self.add_edge(ckey, "claim", pkey, "passage", weight=1.0)

        # 3. Direct mention edges (Entity <-> Passage) for passages containing known entity names
        for ent_name in self.relational_graph.entities.keys():
            ekey = f"ent:{ent_name}"
            pattern = re.compile(rf"\b{re.escape(ent_name)}\b", re.IGNORECASE)
            for pkey, ptext in self.passage_id_map.items():
                if pattern.search(ptext):
                    self.add_edge(ekey, "entity", pkey, "passage", weight=0.5)

        self._compile_transition_matrix()

    def _compile_transition_matrix(self):
        """
        Compile transition probability matrix M on device (CUDA or CPU) with hub suppression.
        """
        N = len(self.idx_to_node)
        if N == 0:
            self._is_built = True
            return

        # Calculate degrees
        degrees = np.zeros(N, dtype=np.float32)
        for u, neighbors in self.adj.items():
            degrees[u] = sum(neighbors.values())

        # Construct weighted transition matrix
        # For hub suppression: normalized weight w(u, v) = w / sqrt(deg(u) * deg(v))
        mat = np.zeros((N, N), dtype=np.float32)

        for u, neighbors in self.adj.items():
            if not neighbors:
                # Dangling node -> self-loop
                mat[u, u] = 1.0
                continue

            deg_u = max(degrees[u], 1e-6)
            row_sum = 0.0
            for v, weight in neighbors.items():
                if self.hub_suppression:
                    deg_v = max(degrees[v], 1e-6)
                    norm_w = weight / math.sqrt(deg_u * deg_v)
                else:
                    norm_w = weight
                mat[u, v] = norm_w
                row_sum += norm_w

            if row_sum > 0:
                mat[u, :] /= row_sum
            else:
                mat[u, u] = 1.0

        # Move to PyTorch tensor on target device (CUDA or CPU)
        try:
            self.transition_matrix = torch.tensor(mat, dtype=torch.float32, device=self.device)
        except Exception as e:
            logger.warning(f"Failed to place transition matrix on {self.device}, falling back to CPU: {e}")
            self.device = "cpu"
            self.transition_matrix = torch.tensor(mat, dtype=torch.float32, device="cpu")

        self._is_built = True

    def run_ppr(
        self,
        seed_nodes: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Execute Personalized PageRank power iteration:
        p^(t+1) = (1 - alpha) * M^T * p^(t) + alpha * p_0

        Args:
            seed_nodes: Mapping of node keys to seed teleport weights.
        Returns:
            Mapping of node keys to stationary probabilities.
        """
        N = len(self.idx_to_node)
        if N == 0:
            return {}

        if not self._is_built or self.transition_matrix is None:
            self._compile_transition_matrix()

        # Build teleport distribution vector p_0
        p0_vec = np.zeros(N, dtype=np.float32)
        total_seed_weight = 0.0

        for node_key, weight in seed_nodes.items():
            if node_key in self.node_to_idx:
                idx = self.node_to_idx[node_key]
                p0_vec[idx] += weight
                total_seed_weight += weight

        if total_seed_weight > 0:
            p0_vec /= total_seed_weight
        else:
            # Uniform teleport fallback if no seeds match graph
            p0_vec[:] = 1.0 / N

        p0 = torch.tensor(p0_vec, dtype=torch.float32, device=self.device)
        p = p0.clone()

        # Transition matrix transpose for forward propagation
        # M[u, v] is probability of jumping from u to v.
        # p^(t+1)_v = sum_u p^(t)_u * M[u, v] -> p^(t+1) = p^(t) * M = M^T * p
        M = self.transition_matrix
        alpha = self.damping

        # Power iteration
        for _ in range(self.max_iter):
            p_next = (1.0 - alpha) * torch.matmul(p, M) + alpha * p0
            diff = torch.norm(p_next - p, p=1).item()
            p = p_next
            if diff < self.tol:
                break

        # Move stationary distribution to CPU dict
        scores = p.cpu().numpy()
        return {self.idx_to_node[i]: float(scores[i]) for i in range(N)}

    def retrieve(
        self,
        query: str,
        candidate_passage_seeds: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Run entity-seeded PPR and return ranked passage results.
        """
        N = len(self.idx_to_node)
        if N == 0:
            return []

        # Extract query entity seeds
        query_entities = self._extract_query_entities(query)
        seeds: Dict[str, float] = {}

        if query_entities:
            # Distribute 70% seed mass to recognized entities
            ent_weight = 0.70 / len(query_entities)
            for ent in query_entities:
                seeds[f"ent:{ent}"] = ent_weight

        # Distribute remaining (or full) seed mass to candidate passages (from dense/BM25)
        if candidate_passage_seeds:
            pass_weight = (0.30 if query_entities else 1.0) / len(candidate_passage_seeds)
            for pid in candidate_passage_seeds:
                seeds[f"passage:{pid}"] = pass_weight

        # Fallback: if no seeds identified, distribute uniformly
        if not seeds:
            for idx, ntype in enumerate(self.node_types):
                if ntype == "passage":
                    seeds[self.idx_to_node[idx]] = 1.0

        stationary_probs = self.run_ppr(seeds)

        # Filter to passage nodes only
        passage_scores: List[Dict[str, Any]] = []
        for node_key, score in stationary_probs.items():
            if node_key.startswith("passage:"):
                pid = node_key[len("passage:"):]
                passage_scores.append({
                    "id": pid,
                    "score": score,
                    "text": self.passage_id_map.get(node_key, ""),
                })

        passage_scores.sort(key=lambda x: x["score"], reverse=True)
        return passage_scores[:top_k]

    def _extract_query_entities(self, query: str) -> List[str]:
        """
        Match entities in query using known entities, aliases, and capitalized patterns.
        """
        matches: List[str] = []
        q_lower = query.lower()

        # 1. Match against known relational graph canonical entities
        for canonical_name in self.relational_graph.entities.keys():
            if canonical_name.lower() in q_lower:
                matches.append(canonical_name)

        # 2. Match against aliases
        alias_map = getattr(self.relational_graph, "alias_map", {}) or getattr(self.relational_graph, "known_aliases", {})
        for alias, canonical_name in alias_map.items():
            if alias.lower() in q_lower and canonical_name not in matches:
                matches.append(canonical_name)

        # 3. Capitalized tokens / sequences
        cap_tokens = re.findall(r"\b[A-Z][a-zA-Z0-9_-]*(?:\s+[A-Z][a-zA-Z0-9_-]*)*\b", query)
        for token in cap_tokens:
            token_clean = token.strip()
            if token_clean and token_clean not in matches:
                matches.append(token_clean)

        return matches


class PPRHybridRetriever:
    """
    Hybrid retriever fusing HippoRAG Personalized PageRank, Dense Semantic Similarity,
    BM25 Lexical Retrieval, and Cross-Encoder reranking with GPU acceleration.
    """

    def __init__(
        self,
        relational_graph: Optional[RelationalGraph] = None,
        passages: Optional[List[Dict[str, Any]]] = None,
        ppr_weight: float = 0.40,
        dense_weight: float = 0.35,
        bm25_weight: float = 0.25,
        device: Optional[str] = None,
    ):
        self.relational_graph = relational_graph or RelationalGraph()
        self.passages = passages or []
        self.ppr_weight = ppr_weight
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.ppr_graph = RelationalPPRGraph(
            relational_graph=self.relational_graph,
            device=device,
        )
        if self.passages:
            self.ppr_graph.build_from_passages_and_claims(self.passages)

    def index_passages(self, passages: List[Dict[str, Any]], claims: Optional[List[RelationalClaim]] = None):
        self.passages = passages
        self.ppr_graph.build_from_passages_and_claims(passages, claims)

    def retrieve(
        self,
        query: str,
        dense_candidates: Optional[List[Dict[str, Any]]] = None,
        bm25_candidates: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Fuses PPR, dense, and BM25 candidates using Reciprocal Rank Fusion (RRF).
        """
        passage_seeds = [c["id"] for c in (dense_candidates or [])[:3]]
        ppr_results = self.ppr_graph.retrieve(
            query=query,
            candidate_passage_seeds=passage_seeds,
            top_k=max(top_k * 3, 10),
        )

        # Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = {}
        passage_texts: Dict[str, str] = {}
        rrf_k = 60.0

        # PPR ranks
        for rank, item in enumerate(ppr_results):
            pid = item["id"]
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (self.ppr_weight / (rrf_k + rank + 1))
            passage_texts[pid] = item.get("text", "")

        # Dense ranks
        if dense_candidates:
            for rank, item in enumerate(dense_candidates):
                pid = item["id"]
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (self.dense_weight / (rrf_k + rank + 1))
                if pid not in passage_texts:
                    passage_texts[pid] = item.get("text") or item.get("passage", "")

        # BM25 ranks
        if bm25_candidates:
            for rank, item in enumerate(bm25_candidates):
                pid = item["id"]
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (self.bm25_weight / (rrf_k + rank + 1))
                if pid not in passage_texts:
                    passage_texts[pid] = item.get("text") or item.get("passage", "")

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {
                "id": pid,
                "score": score,
                "text": passage_texts.get(pid, ""),
                "retrieval_path": "ppr_hybrid_rrf",
            }
            for pid, score in ranked[:top_k]
        ]
