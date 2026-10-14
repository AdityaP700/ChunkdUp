# ai-system/chunkdup/retrieval_graph_augmented.py
"""
Graph-Augmented Retriever for ChunkdUp.

Implements Controlled Graph-Expansion Retrieval:
1. Candidate Pool Generation:
   - Initial pool of candidate turns retrieved via Hybrid (BM25 + Dense Sentence-Transformers) retrieval.
2. Entity & Relation Extraction:
   - Identifies query entities and extracts entities/claims from top initial candidates.
3. Bounded Graph Expansion (1-hop & 2-hop edges):
   - Relational Claims: follows claim -> entity -> neighbor claim -> source turns.
   - Entity Co-occurrence: links turns sharing canonical entity mentions.
   - Temporal Adjacency: links immediate predecessor/successor turns (t-1, t+1) in the same session.
   - Expansion is strictly bounded (max K neighbors per seed node, max total expanded pool).
4. Cross-Encoder Neural Reranking:
   - Reranks the unified pool (initial + expanded candidates) using ms-marco-MiniLM-L-12-v2.
5. Grounding & Provenance Tracking:
   - Tracks expansion lineage, source turns, and rerank confidence scores.
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from rank_bm25 import BM25Okapi

from chunkdup.relational.graph import RelationalGraph
from chunkdup.relational.schema import RelationalClaim
from chunkdup.retrieval_production import (
    _get_cross_encoder,
    _get_sentence_transformer,
    _index_text,
    _resolve_device,
    _rrf_fuse,
    _safe_encode,
    _safe_predict,
    _tokenize,
)

logger = logging.getLogger(__name__)


def _extract_dia_id(text_or_id: Any) -> Optional[str]:
    """Helper to extract canonical dialogue turn ID (e.g. 'D1:2')."""
    if not text_or_id:
        return None
    s = str(text_or_id).strip()
    match = re.search(r"\b(D\d+:\d+)\b", s)
    if match:
        return match.group(1)
    if s.startswith("D") and ":" in s:
        return s
    return None


class GraphAugmentedRetriever:
    """
    Controlled Graph-Augmented Retriever combining Hybrid BM25/Dense retrieval,
    1-hop/2-hop relational and temporal graph expansion, and Cross-Encoder reranking.
    """

    def __init__(
        self,
        relational_graph: Optional[RelationalGraph] = None,
        device: Optional[str] = None,
        initial_pool_size: int = 15,
        max_expanded_candidates: int = 10,
        max_neighbors_per_seed: int = 3,
        abstention_threshold: float = -10.947404,
    ):
        self.device = _resolve_device(device)
        self.relational_graph = relational_graph or RelationalGraph()
        self.initial_pool_size = initial_pool_size
        self.max_expanded_candidates = max_expanded_candidates
        self.max_neighbors_per_seed = max_neighbors_per_seed
        self.abstention_threshold = abstention_threshold

        self.passages: List[Dict[str, Any]] = []
        self.passage_by_id: Dict[str, Dict[str, Any]] = {}
        self.passage_by_dia_id: Dict[str, Dict[str, Any]] = {}

        # Dense index
        self.dense_matrix: Optional[np.ndarray] = None
        # BM25 index
        self.bm25: Optional[BM25Okapi] = None

        # Graph mappings
        self.entity_to_passages: Dict[str, Set[str]] = {}
        self.passage_to_entities: Dict[str, Set[str]] = {}
        self.claims_by_passage: Dict[str, Set[str]] = {}
        self.session_turn_order: Dict[str, List[str]] = {}  # session_id -> list of dia_ids

        self._is_indexed = False

    def index_passages(
        self,
        passages: List[Dict[str, Any]],
        relational_graph: Optional[RelationalGraph] = None,
    ):
        """
        Index dialogue passages, build BM25 + dense representations, and wire graph links.
        """
        if relational_graph is not None:
            self.relational_graph = relational_graph

        self.passages = list(passages)
        self.passage_by_id.clear()
        self.passage_by_dia_id.clear()
        self.entity_to_passages.clear()
        self.passage_to_entities.clear()
        self.claims_by_passage.clear()
        self.session_turn_order.clear()

        if not self.passages:
            self._is_indexed = True
            return

        tokenized_corpus: List[List[str]] = []
        texts_to_embed: List[str] = []

        for p in self.passages:
            pid = str(p.get("id") or p.get("dia_id") or p.get("turn_id") or "")
            dia_id = _extract_dia_id(pid) or _extract_dia_id(p.get("dia_id")) or _extract_dia_id(p.get("turn_id")) or pid

            # Canonicalize passage dictionary
            p_dict = dict(p)
            p_dict["id"] = pid
            p_dict["dia_id"] = dia_id
            p_dict["source_turn_id"] = dia_id
            p_text = str(p.get("text") or p.get("value") or p.get("passage") or "").strip()
            p_dict["text"] = p_text

            self.passage_by_id[pid] = p_dict
            if dia_id:
                self.passage_by_dia_id[dia_id] = p_dict

            # Track session ordering for temporal adjacency
            sess = str(p.get("session") or p.get("session_id") or "")
            if not sess and dia_id and dia_id.startswith("D") and ":" in dia_id:
                sess = dia_id.split(":")[0]
            if not sess:
                sess = "default_session"

            if sess not in self.session_turn_order:
                self.session_turn_order[sess] = []
            self.session_turn_order[sess].append(dia_id)

            # Representation for retrieval
            rep_text = _index_text(p_dict) if "key" in p_dict else p_text
            texts_to_embed.append(rep_text)
            tokenized_corpus.append(_tokenize(rep_text))

            # Scan entities mentioned in this passage
            p_lower = p_text.lower()
            mentioned_ents: Set[str] = set()

            for canonical_name in self.relational_graph.entities.keys():
                if canonical_name.lower() in p_lower:
                    mentioned_ents.add(canonical_name)

            for alias, canonical_name in self.relational_graph.alias_map.items():
                if alias in p_lower:
                    mentioned_ents.add(canonical_name)

            # Speaker entity
            spk = p.get("speaker")
            if spk and str(spk).strip():
                clean_spk = str(spk).strip()
                if clean_spk.lower() not in {"user", "assistant", "speaker a", "speaker b"}:
                    mentioned_ents.add(clean_spk)

            self.passage_to_entities[dia_id] = mentioned_ents
            for ent in mentioned_ents:
                if ent not in self.entity_to_passages:
                    self.entity_to_passages[ent] = set()
                self.entity_to_passages[ent].add(dia_id)

        # Index claims by passage
        for claim in self.relational_graph.get_claims(active_only=True):
            for tid in claim.source_turn_ids:
                clean_tid = _extract_dia_id(tid) or tid
                if clean_tid not in self.claims_by_passage:
                    self.claims_by_passage[clean_tid] = set()
                self.claims_by_passage[clean_tid].add(claim.claim_id)

        # Build BM25
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Build Dense Embeddings
        st = _get_sentence_transformer(device=self.device)
        embeddings = _safe_encode(st, texts_to_embed, normalize_embeddings=True)
        if hasattr(embeddings, "cpu"):
            embeddings = embeddings.cpu().numpy()
        self.dense_matrix = np.asarray(embeddings, dtype=np.float32)

        self._is_indexed = True

    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract recognized canonical entities and capitalized terms from query."""
        found: List[str] = []
        q_lower = query.lower()

        # 1. Known entities
        for canonical_name in self.relational_graph.entities.keys():
            if canonical_name.lower() in q_lower and canonical_name not in found:
                found.append(canonical_name)

        # 2. Known aliases
        for alias, canonical_name in self.relational_graph.alias_map.items():
            if alias in q_lower and canonical_name not in found:
                found.append(canonical_name)

        # 3. Capitalized named entity tokens
        cap_tokens = re.findall(r"\b[A-Z][a-zA-Z0-9_\-\.]*(?:\s+[A-Z][a-zA-Z0-9_\-\.]*)*\b", query)
        for token in cap_tokens:
            token_clean = token.strip()
            if token_clean and token_clean not in found:
                found.append(token_clean)

        return found

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        initial_pool_size: Optional[int] = None,
        max_expanded: Optional[int] = None,
        as_of: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute controlled graph-expansion retrieval:
        1. Hybrid candidate generation (BM25 + Dense RRF)
        2. Bounded 1-hop and 2-hop graph expansion
        3. Cross-Encoder neural reranking of the combined pool
        """
        if not self._is_indexed or not self.passages:
            return []

        pool_k = initial_pool_size or self.initial_pool_size
        max_exp = max_expanded or self.max_expanded_candidates

        # Step 1: Hybrid candidate generation
        st = _get_sentence_transformer(device=self.device)
        q_vec = _safe_encode(st, query, normalize_embeddings=True)
        if hasattr(q_vec, "cpu"):
            q_vec = q_vec.cpu().numpy()

        sims = self.dense_matrix @ q_vec
        dense_order = np.argsort(-sims)[:pool_k]
        dense_candidates = []
        for idx in dense_order:
            cand = dict(self.passages[int(idx)])
            cand["_similarity"] = float(sims[int(idx)])
            dense_candidates.append(cand)

        bm25_scores = self.bm25.get_scores(_tokenize(query))
        bm25_order = np.argsort(-bm25_scores)[:pool_k]
        bm25_candidates = []
        for idx in bm25_order:
            if bm25_scores[idx] <= 0.0:
                break
            cand = dict(self.passages[int(idx)])
            cand["_bm25_score"] = float(bm25_scores[idx])
            bm25_candidates.append(cand)

        # Fuse with Reciprocal Rank Fusion
        initial_fused = _rrf_fuse([dense_candidates, bm25_candidates], k=60, limit=pool_k)

        # Step 2: Extract entities & claims from query and top initial candidates
        query_entities = self._extract_query_entities(query)
        initial_ids: Set[str] = {
            c.get("dia_id") or _extract_dia_id(c.get("id")) or str(c.get("id"))
            for c in initial_fused
        }

        # Step 3: Bounded Graph Expansion
        expanded_candidates: List[Dict[str, Any]] = []
        seen_expanded_ids: Set[str] = set()

        # Seed candidates: top 3 from initial retrieval
        seeds = initial_fused[:3]

        for seed in seeds:
            if len(expanded_candidates) >= max_exp:
                break

            seed_dia_id = seed.get("dia_id") or _extract_dia_id(seed.get("id")) or str(seed.get("id"))

            # Path A: Relational Claims expansion (1-hop through connected claims)
            claim_ids = self.claims_by_passage.get(seed_dia_id, set())
            for cid in claim_ids:
                if len(expanded_candidates) >= max_exp:
                    break
                claim = self.relational_graph.claims.get(cid)
                if not claim or not claim.is_valid_as_of(as_of):
                    continue

                # Get 1-hop neighbor claims via subject and object entities
                for ent in (claim.subject_entity, claim.object_entity):
                    neighbors = self.relational_graph.get_neighbors(ent, as_of=as_of)
                    for n_ent, pred, n_claim in neighbors[:self.max_neighbors_per_seed]:
                        for neighbor_tid in n_claim.source_turn_ids:
                            clean_ntid = _extract_dia_id(neighbor_tid) or neighbor_tid
                            if clean_ntid not in initial_ids and clean_ntid not in seen_expanded_ids:
                                p_obj = self.passage_by_dia_id.get(clean_ntid) or self.passage_by_id.get(clean_ntid)
                                if p_obj:
                                    cand = dict(p_obj)
                                    cand["_expansion_type"] = "relational_claim"
                                    cand["_expanded_from"] = seed_dia_id
                                    cand["_expansion_relation"] = f"{ent}->{pred}->{n_ent}"
                                    expanded_candidates.append(cand)
                                    seen_expanded_ids.add(clean_ntid)
                                    if len(expanded_candidates) >= max_exp:
                                        break

            # Path B: Entity co-occurrence expansion
            seed_entities = self.passage_to_entities.get(seed_dia_id, set())
            relevant_entities = (seed_entities | set(query_entities))
            for ent in relevant_entities:
                if len(expanded_candidates) >= max_exp:
                    break
                linked_passages = self.entity_to_passages.get(ent, set())
                # Add up to max_neighbors_per_seed passages mentioning this entity
                added_for_ent = 0
                for l_pid in linked_passages:
                    if l_pid not in initial_ids and l_pid not in seen_expanded_ids:
                        p_obj = self.passage_by_dia_id.get(l_pid) or self.passage_by_id.get(l_pid)
                        if p_obj:
                            cand = dict(p_obj)
                            cand["_expansion_type"] = "entity_cooccurrence"
                            cand["_expanded_from"] = seed_dia_id
                            cand["_expansion_entity"] = ent
                            expanded_candidates.append(cand)
                            seen_expanded_ids.add(l_pid)
                            added_for_ent += 1
                            if added_for_ent >= self.max_neighbors_per_seed or len(expanded_candidates) >= max_exp:
                                break

            # Path C: Temporal Adjacency expansion (immediate dialogue turns t-1, t+1)
            sess = str(seed.get("session") or seed.get("session_id") or "")
            if not sess and seed_dia_id.startswith("D") and ":" in seed_dia_id:
                sess = seed_dia_id.split(":")[0]
            if not sess:
                sess = "default_session"

            turn_list = self.session_turn_order.get(sess, [])
            if seed_dia_id in turn_list:
                pos = turn_list.index(seed_dia_id)
                # Check neighbors t-1 and t+1
                for offset in (-1, 1):
                    neighbor_pos = pos + offset
                    if 0 <= neighbor_pos < len(turn_list):
                        adj_tid = turn_list[neighbor_pos]
                        if adj_tid not in initial_ids and adj_tid not in seen_expanded_ids:
                            p_obj = self.passage_by_dia_id.get(adj_tid) or self.passage_by_id.get(adj_tid)
                            if p_obj:
                                cand = dict(p_obj)
                                cand["_expansion_type"] = "temporal_adjacency"
                                cand["_expanded_from"] = seed_dia_id
                                cand["_expansion_offset"] = offset
                                expanded_candidates.append(cand)
                                seen_expanded_ids.add(adj_tid)
                                if len(expanded_candidates) >= max_exp:
                                    break

        # Step 4: Cross-Encoder Neural Reranking over unified pool
        unified_pool: List[Dict[str, Any]] = list(initial_fused) + expanded_candidates

        ce = _get_cross_encoder(device=self.device)
        pairs = []
        for c in unified_pool:
            p_text = c.get("text") or c.get("value") or ""
            rep = _index_text(c) if "key" in c else p_text
            pairs.append([query, rep])

        if pairs:
            scores = _safe_predict(ce, pairs, batch_size=32)
            for c, s in zip(unified_pool, scores):
                c["_rerank_score"] = float(s)
                c["score"] = float(s)
                c["retrieval_path"] = "graph_augmented"
                c["is_expanded"] = c.get("_expansion_type") is not None

            unified_pool.sort(key=lambda x: x["_rerank_score"], reverse=True)

        return unified_pool[:top_k]
