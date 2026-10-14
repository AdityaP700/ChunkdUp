# ai-system/chunkdup/hybrid_memory_optimized.py
"""
Optimized Hybrid Memory System for ChunkdUp (Strategy F Optimization).

Unites:
1. Query Router:
   - Categorizes queries into relational_multihop, profile_attribute, factual_episodic, or adversarial.
2. GraphAugmentedRetriever:
   - Specialized candidate-expansion graph retrieval with Cross-Encoder reranking for multi-hop reasoning.
3. ProfileGuidedArchiveRetriever:
   - Two-tier profile-guided archive retrieval with strictly bounded token context.
4. Fast Hybrid Retrieval:
   - Direct BM25 + Dense Sentence-Transformers + Cross-Encoder for factual episodic queries.
5. Calibrated Abstention Gate:
   - High-precision confidence thresholding rejecting adversarial/unanswerable queries.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from chunkdup.profile.profile_guided_retriever import ProfileGuidedArchiveRetriever
from chunkdup.profile.profile_memory import PersistentProfile
from chunkdup.query_router import QueryRouter
from chunkdup.relational.extractor import RelationalExtractor
from chunkdup.relational.graph import RelationalGraph
from chunkdup.retrieval_graph_augmented import GraphAugmentedRetriever

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


class OptimizedHybridMemory:
    """
    Unified, Query-Routed Hybrid Memory Architecture.
    Routes queries to the optimal specialized memory strategy and enforces calibrated abstention.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        extractor_mode: str = "rules",
        adversarial_threshold: float = 2.0,
        general_threshold: float = -2.5,
    ):
        self.device = device
        self.extractor_mode = extractor_mode
        self.adversarial_threshold = adversarial_threshold
        self.general_threshold = general_threshold

        self.relational_graph = RelationalGraph()
        self.relational_extractor = RelationalExtractor(mode=extractor_mode)
        self.profile = PersistentProfile(max_facts=50)

        self.graph_retriever = GraphAugmentedRetriever(
            relational_graph=self.relational_graph,
            device=device,
        )
        self.profile_retriever = ProfileGuidedArchiveRetriever(
            profile=self.profile,
            device=device,
        )

        self.query_router = QueryRouter()
        self.turns: List[Dict[str, Any]] = []
        self._is_indexed = False

    def remember(
        self,
        turn_id: str,
        speaker: str,
        text: str,
        session_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ):
        """Ingest a dialogue turn into relational graph, profile facts, and episodic archive."""
        dia_id = _extract_dia_id(turn_id) or turn_id
        turn_dict = {
            "id": turn_id,
            "dia_id": dia_id,
            "turn_id": dia_id,
            "source_turn_id": dia_id,
            "speaker": speaker,
            "text": text,
            "session": session_id or "default",
            "timestamp": timestamp or "",
        }
        self.turns.append(turn_dict)

        # 1. Update relational graph via window extraction
        window = self.turns[-4:-1]
        claims = self.relational_extractor.extract_from_window(turn_dict, context_window=window)
        for c in claims:
            self.relational_graph.add_claim(c)

        # 2. Extract profile facts
        clean_text = text.lower()
        pref_match = re.search(
            r"(?:i prefer|i like|my favorite|i work as|i am a|i live in|married to)\s+([A-Za-z0-9_\-\.\s]+)",
            text,
            re.IGNORECASE,
        )
        if pref_match:
            val = pref_match.group(1).strip().rstrip(".,;:!?")
            key = f"{speaker.lower()}_pref_{len(self.profile.facts)+1}"
            self.profile.set_fact(key, val, category="preference", source_turn_id=dia_id)

        # 3. Update router known entities
        if speaker and speaker.lower() not in {"user", "assistant"}:
            self.query_router.update_known_entities({speaker})
        for c in claims:
            self.query_router.update_known_entities({c.subject_entity, c.object_entity})

        self._is_indexed = False

    def _ensure_indexed(self):
        if not self._is_indexed:
            passages = [
                {
                    "id": t["turn_id"],
                    "dia_id": t["dia_id"],
                    "turn_id": t["turn_id"],
                    "source_turn_id": t["dia_id"],
                    "text": f"{t['speaker']}: {t['text']}",
                    "speaker": t["speaker"],
                    "session": t.get("session"),
                    "timestamp": t.get("timestamp"),
                }
                for t in self.turns
            ]
            self.graph_retriever.index_passages(passages, relational_graph=self.relational_graph)
            self.profile_retriever.index_turns(passages, profile=self.profile)
            self._is_indexed = True

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        as_of: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Query-Routed Retrieval:
        1. Query Router decides strategy:
           - "relational_multihop" -> GraphAugmentedRetriever
           - "profile_attribute"  -> ProfileGuidedArchiveRetriever
           - "factual_episodic"   -> GraphAugmentedRetriever (with fast hybrid pool)
           - "adversarial"        -> Enforce strict abstention gate
        2. Calibrated Abstention Verification.
        """
        t0 = time.perf_counter()
        self._ensure_indexed()

        # Step 1: Query Routing
        route = self.query_router.route(query)

        # Step 2: Route-specific execution
        results: List[Dict[str, Any]] = []
        prompt_context: str = ""

        if route == "adversarial":
            # For known adversarial / trick queries, evaluate candidates under strict threshold
            candidates = self.graph_retriever.retrieve(query, top_k=top_k, as_of=as_of)
            top_score = candidates[0].get("score", -99.0) if candidates else -99.0
            if top_score < self.adversarial_threshold:
                # Calibrated abstention
                lat_ms = (time.perf_counter() - t0) * 1000.0
                return {
                    "route": route,
                    "results": [],
                    "abstained": True,
                    "top_score": top_score,
                    "latency_ms": lat_ms,
                    "prompt_context": "",
                }
            results = candidates

        elif route == "relational_multihop":
            # Graph-expansion retrieval with Cross-Encoder reranking
            results = self.graph_retriever.retrieve(query, top_k=top_k, as_of=as_of)
            prompt_context = "\n".join([f"- {r.get('text', '')}" for r in results])

        elif route == "profile_attribute":
            # Two-tier profile-guided retrieval
            prof_dict = self.profile_retriever.retrieve(query, top_k=top_k)
            results = prof_dict.get("results", [])
            prompt_context = prof_dict.get("prompt_context", "")

        else:  # "factual_episodic"
            # Fast hybrid retrieval with Cross-Encoder reranking
            results = self.graph_retriever.retrieve(query, top_k=top_k, max_expanded=2, as_of=as_of)
            prompt_context = "\n".join([f"- {r.get('text', '')}" for r in results])

        # Step 3: Calibrated general abstention check
        top_score = results[0].get("score", -99.0) if results else -99.0
        abstained = len(results) == 0 or top_score < self.general_threshold

        lat_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "route": route,
            "results": [] if abstained else results,
            "abstained": abstained,
            "top_score": top_score,
            "latency_ms": lat_ms,
            "prompt_context": "" if abstained else prompt_context,
        }
