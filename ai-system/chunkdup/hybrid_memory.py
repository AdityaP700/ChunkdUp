# ai-system/chunkdup/hybrid_memory.py
"""
Hybrid Memory System for ChunkdUp.

Unites:
1. Persistent Profile Hot-Path:
   - Compact, bounded tier-1 profile facts served with zero retrieval latency.
2. Context-Aware Relational Extraction:
   - Multi-turn window extraction with pronoun coreference resolution & bi-temporal validity.
3. HippoRAG Personalized PageRank (PPR) Graph Retrieval:
   - GPU-accelerated power iteration over bipartite entity-claim-passage network.
4. Grounded Abstention Gate:
   - Rejects queries lacking grounded evidence with explicit abstention signaling.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from chunkdup.profile.profile_memory import PersistentProfile, ProfileConsolidationEngine
from chunkdup.relational.extractor import RelationalExtractor
from chunkdup.relational.graph import RelationalGraph
from chunkdup.relational.schema import RelationalClaim, current_iso_utc
from chunkdup.retrieval_ppr import PPRHybridRetriever, RelationalPPRGraph

logger = logging.getLogger(__name__)


class HybridMemory:
    """
    Unified Persistent Memory Architecture combining Profile Hot-Path,
    Context-Aware Relational Extraction, and HippoRAG PPR Retrieval.
    """

    def __init__(
        self,
        max_profile_facts: int = 50,
        ppr_damping: float = 0.15,
        ppr_weight: float = 0.40,
        dense_weight: float = 0.35,
        bm25_weight: float = 0.25,
        abstention_threshold: float = 0.0001,
        device: Optional[str] = None,
        extractor_mode: str = "rules",
        groq_client: Optional[Any] = None,
    ):
        self.profile = PersistentProfile(max_facts=max_profile_facts)
        self.profile_engine = ProfileConsolidationEngine(mode=extractor_mode, groq_client=groq_client)
        self.relational_extractor = RelationalExtractor(mode=extractor_mode, groq_client=groq_client)
        self.relational_graph = RelationalGraph()
        self.raw_turns: List[Dict[str, Any]] = []
        self.ppr_graph = RelationalPPRGraph(
            relational_graph=self.relational_graph,
            damping=ppr_damping,
            device=device,
        )
        self.ppr_weight = ppr_weight
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.abstention_threshold = abstention_threshold
        self._graph_dirty = True

    def remember(
        self,
        text: str,
        speaker: str = "User",
        timestamp: Optional[str] = None,
        turn_id: Optional[str] = None,
        context_window: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest conversation turn:
        1. Consolidate profile facts (Tier 1 Hot Path)
        2. Context-aware multi-turn relational extraction (Tier 2 Graphiti)
        3. Store dialogue turn for episodic archive (Tier 3)
        """
        ts = timestamp or current_iso_utc()
        tid = turn_id or f"turn_{len(self.raw_turns)+1}"

        turn_dict = {
            "turn_id": tid,
            "speaker": speaker,
            "text": text,
            "timestamp": ts,
        }
        self.raw_turns.append(turn_dict)

        # 1. Profile consolidation
        profile_decisions = self.profile_engine.evaluate_turn(turn_dict, self.profile)
        for d in profile_decisions:
            action = d.get("action")
            k = d.get("key")
            v = d.get("value")
            if action in ("ADD", "UPDATE") and k and v:
                self.profile.set_fact(k, v, category=d.get("category", "preference"), source_turn_id=tid)
            elif action == "DELETE" and k:
                self.profile.delete_fact(k)

        # 2. Relational extraction
        window = context_window if context_window is not None else self.raw_turns[-4:-1]
        claims = self.relational_extractor.extract_from_window(turn_dict, context_window=window)
        for claim in claims:
            self.relational_graph.add_claim(claim)

        self._graph_dirty = True
        return {
            "turn_id": tid,
            "profile_decisions": profile_decisions,
            "claims_extracted": len(claims),
        }

    def _ensure_ppr_index(self):
        if self._graph_dirty:
            passages = [{"id": t["turn_id"], "text": f"{t['speaker']}: {t['text']}"} for t in self.raw_turns]
            self.ppr_graph.build_from_passages_and_claims(passages)
            self._graph_dirty = False

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        dense_candidates: Optional[List[Dict[str, Any]]] = None,
        bm25_candidates: Optional[List[Dict[str, Any]]] = None,
        as_of: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Unified retrieval path:
        1. Profile Hot-Path Check:
           If query matches canonical profile facts, serve immediately with zero archive latency.
        2. HippoRAG Personalized PageRank Graph Retrieval over relational memory:
           Propagates multi-hop graph activation across entities and bi-temporal claims.
        3. Reciprocal Rank Fusion & Abstention Gating:
           Fuses PPR, dense, and lexical scores; abstains if evidence fails confidence threshold.
        """
        t0 = time.perf_counter()

        # Step 1: Hot-Path check
        profile_matches = self.profile.search_facts(query)
        if profile_matches:
            top_fact = profile_matches[0]
            lat_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "source": "profile_hot_path",
                "archive_searched": False,
                "fact": top_fact.to_dict(),
                "all_matching_facts": [f.to_dict() for f in profile_matches],
                "results": [
                    {
                        "id": f"fact_{top_fact.fact_id}",
                        "text": f"{top_fact.key}: {top_fact.value}",
                        "source_turn_id": top_fact.source_turn_id,
                        "score": 1.0,
                    }
                ],
                "abstained": False,
                "latency_ms": lat_ms,
                "prompt_context": self.profile.to_prompt_context(),
            }

        # Step 2: PPR Relational Graph Retrieval
        self._ensure_ppr_index()

        # Generate lexical candidates if not provided
        if not bm25_candidates and self.raw_turns:
            q_tokens = set(re.findall(r"\w+", query.lower()))
            scored_passages = []
            for t in self.raw_turns:
                t_tokens = set(re.findall(r"\w+", t["text"].lower()))
                overlap = len(q_tokens.intersection(t_tokens))
                if overlap > 0:
                    scored_passages.append({"id": t["turn_id"], "score": float(overlap), "text": t["text"]})
            scored_passages.sort(key=lambda x: x["score"], reverse=True)
            bm25_candidates = scored_passages[:5]

        passage_seeds = [c["id"] for c in (dense_candidates or bm25_candidates or [])[:3]]
        ppr_results = self.ppr_graph.retrieve(
            query=query,
            candidate_passage_seeds=passage_seeds,
            top_k=max(top_k * 3, 10),
        )

        # Step 3: Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = {}
        passage_info: Dict[str, Dict[str, Any]] = {}
        rrf_k = 60.0

        for rank, item in enumerate(ppr_results):
            pid = item["id"]
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (self.ppr_weight / (rrf_k + rank + 1))
            passage_info[pid] = item

        if dense_candidates:
            for rank, item in enumerate(dense_candidates):
                pid = item["id"]
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (self.dense_weight / (rrf_k + rank + 1))
                if pid not in passage_info:
                    passage_info[pid] = item

        if bm25_candidates:
            for rank, item in enumerate(bm25_candidates):
                pid = item["id"]
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (self.bm25_weight / (rrf_k + rank + 1))
                if pid not in passage_info:
                    passage_info[pid] = item

        ranked_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Grounding / Abstention check
        top_score = ranked_items[0][1] if ranked_items else 0.0
        abstained = top_score < self.abstention_threshold or len(ranked_items) == 0

        final_results = []
        if not abstained:
            for pid, score in ranked_items[:top_k]:
                info = passage_info.get(pid, {})
                final_results.append({
                    "id": pid,
                    "text": info.get("text") or info.get("passage", ""),
                    "score": score,
                    "source_turn_id": pid,
                    "retrieval_path": "ppr_hybrid_rrf",
                })

        lat_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "source": "relational_ppr_archive",
            "archive_searched": True,
            "results": final_results,
            "abstained": abstained,
            "top_score": top_score,
            "latency_ms": lat_ms,
            "prompt_context": self._format_prompt_context(final_results),
        }

    def _format_prompt_context(self, passages: List[Dict[str, Any]]) -> str:
        lines = []
        prof_ctx = self.profile.to_prompt_context()
        if prof_ctx:
            lines.append(prof_ctx)
        if passages:
            lines.append("[RELEVANT CONTEXT]")
            for p in passages:
                lines.append(f"- {p.get('text', '')}")
        return "\n\n".join(lines)
