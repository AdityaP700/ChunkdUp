# ai-system/chunkdup/associative/retrievers.py
"""
The 7 Retrieval Variants for the Evidence-Preserving Associative Memory Sprint:
- Variant A: GrepRetriever (transparent exact/substring lexical baseline)
- Variant B: BM25Retriever (pure BM25Okapi baseline)
- Variant C: HybridCEControlRetriever (hardened production hybrid CE control)
- Variant D: BoundedAssociativeRetriever (hybrid seeds + bounded graph expansion)
- Variant E: ConsolidatedSemanticRetriever (raw turns + atomic facts with provenance)
- Variant F: ConsolidatedAssociativeRetriever (consolidation + associative links)
- Variant G: QueryRoutedAssociativeRetriever (intent-aware dynamic router)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import logging
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from chunkdup.in_memory_repository import InMemoryRepository
from chunkdup.relational.extractor import RelationalExtractor
from chunkdup.relational.schema import RelationalClaim
from chunkdup.retrieval_production import (
    ProductionRetriever,
    _get_cross_encoder,
    _get_sentence_transformer,
    _safe_encode,
    _tokenize,
)
from .graph import AssociativeGraph, BoundedGraphExpansion
from .schema import (
    AssociativeLink,
    AssociativeLinkType,
    ExperienceTurn,
    MemoryRepresentationBundle,
    SemanticFact,
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Standardized retrieval output item across all 7 variants."""
    item_id: str
    dia_id: str                      # Authoritative dialogue turn ID
    text: str
    score: float
    source_type: str                 # 'turn', 'fact', 'graph_expansion', 'lexical'
    speaker: str = ""
    timestamp: str = ""
    underlying_dia_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.item_id,
            "dia_id": self.dia_id,
            "text": self.text,
            "score": float(self.score),
            "source_type": self.source_type,
            "speaker": self.speaker,
            "timestamp": self.timestamp,
            "underlying_dia_ids": self.underlying_dia_ids,
            "metadata": self.metadata,
        }


# ============================================================================
# Variant A: GrepRetriever (Clean exact lexical baseline over raw turns)
# ============================================================================
class GrepRetriever:
    """Exact case-normalized substring, phrase, and token matching over raw turns."""

    def __init__(self, turns: List[ExperienceTurn]):
        self.turns = turns

    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        q_clean = query.strip().lower()
        q_tokens = [t for t in re.findall(r"\w+", q_clean) if len(t) > 2]
        if not q_tokens:
            q_tokens = re.findall(r"\w+", q_clean)

        scored: List[Tuple[float, ExperienceTurn]] = []

        for turn in self.turns:
            t_text = turn.text.lower()
            spk = turn.speaker.lower()
            combined = f"{spk}: {t_text}"

            score = 0.0
            # 1. Exact phrase match bonus
            if q_clean in combined:
                score += 10.0
            elif len(q_clean) > 8:
                # Sub-phrase match
                for sz in (5, 4, 3):
                    words = q_clean.split()
                    for idx in range(len(words) - sz + 1):
                        sub = " ".join(words[idx : idx + sz])
                        if sub in combined:
                            score += 3.0 * sz

            # 2. Token overlap and coverage
            matched_tokens = 0
            for tok in q_tokens:
                if tok in combined:
                    matched_tokens += 1
                    # Boundary match bonus
                    if re.search(rf"\b{re.escape(tok)}\b", combined):
                        score += 1.5
                    else:
                        score += 0.8

            coverage = matched_tokens / max(1, len(q_tokens))
            score += coverage * 5.0

            if score > 0.0:
                scored.append((score, turn))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k]

        return [
            RetrievalResult(
                item_id=t.dia_id,
                dia_id=t.dia_id,
                text=f"{t.speaker}: {t.text}",
                score=round(sc, 4),
                source_type="lexical",
                speaker=t.speaker,
                timestamp=t.timestamp,
                underlying_dia_ids=[t.dia_id],
            )
            for sc, t in top
        ]


# ============================================================================
# Variant B: BM25Retriever (Pure BM25Okapi baseline over raw turns)
# ============================================================================
class BM25Retriever:
    """Pure BM25Okapi retrieval over raw conversational turns."""

    def __init__(self, turns: List[ExperienceTurn]):
        from rank_bm25 import BM25Okapi

        self.turns = turns
        self.corpus = [_tokenize(f"{t.speaker}: {t.text}") for t in turns]
        self.bm25 = BM25Okapi(self.corpus) if any(self.corpus) else None

    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        if not self.bm25 or not self.turns:
            return []
        toks = _tokenize(query)
        if not toks:
            toks = query.lower().split()
        scores = self.bm25.get_scores(toks)
        ranked_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in ranked_indices:
            sc = float(scores[idx])
            if sc <= 0.0:
                continue
            t = self.turns[idx]
            results.append(
                RetrievalResult(
                    item_id=t.dia_id,
                    dia_id=t.dia_id,
                    text=f"{t.speaker}: {t.text}",
                    score=round(sc, 4),
                    source_type="bm25",
                    speaker=t.speaker,
                    timestamp=t.timestamp,
                    underlying_dia_ids=[t.dia_id],
                )
            )
        return results


# ============================================================================
# Variant C: HybridCEControlRetriever (Verified Production Hybrid Control)
# ============================================================================
class HybridCEControlRetriever:
    """Wraps ChunkdUp's production ProductionRetriever (hybrid_ce_abstain)."""

    def __init__(
        self,
        turns: List[ExperienceTurn],
        sample_id: str = "sample",
        device: str = "cuda",
    ):
        self.turns = turns
        self.sample_id = sample_id
        self.device = device if (torch.cuda.is_available() and device == "cuda") else "cpu"
        self.repo = InMemoryRepository()

        # Seed repository with raw dialogue turns preserving dia_id
        for t in self.turns:
            self.repo.save({
                "id": f"{self.sample_id}:{t.dia_id}",
                "dia_id": t.dia_id,
                "source_turn_id": t.dia_id,
                "key": f"turn_{t.dia_id}",
                "value": f"{t.speaker}: {t.text}",
                "contextual_text": t.contextual_text or f"In session {t.session_num} ({t.timestamp}), {t.speaker} said: \"{t.text}\"",
                "type": "dialogue_turn",
                "speaker": t.speaker,
                "session": t.session_num,
                "timestamp": t.timestamp,
                "status": "active",
            })

        self.retriever = ProductionRetriever(
            repository=self.repo,
            config_name="hybrid_ce_abstain",
            query_log_path=None,
            device=self.device,
        )
        self.retriever._ensure_index()

    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        hits = self.retriever.retrieve(query, k=k)
        results = []
        for h in hits:
            did = h.get("dia_id") or h.get("source_turn_id")
            if not did and ":" in str(h.get("id", "")):
                did = str(h["id"]).split(":", 1)[1]
            results.append(
                RetrievalResult(
                    item_id=str(h.get("id", did)),
                    dia_id=str(did),
                    text=str(h.get("value", "")),
                    score=float(h.get("_ce_score", h.get("_dense_score", 0.0))),
                    source_type="hybrid_ce",
                    speaker=str(h.get("speaker", "")),
                    timestamp=str(h.get("timestamp", "")),
                    underlying_dia_ids=[str(did)] if did else [],
                )
            )
        return results


# ============================================================================
# Variant D: BoundedAssociativeRetriever (Hybrid seeds + bounded expansion)
# ============================================================================
class BoundedAssociativeRetriever:
    """
    Retrieves initial seeds via hybrid retrieval, applies bounded associative
    expansion (depth <= 2, branching <= 3, max candidates <= 5), resolves
    neighbor nodes to grounding source turns, and reranks with Cross-Encoder.
    """

    def __init__(
        self,
        turns: List[ExperienceTurn],
        graph: AssociativeGraph,
        sample_id: str = "sample",
        device: str = "cuda",
        max_depth: int = 2,
        max_branching: int = 3,
        max_expanded_candidates: int = 5,
    ):
        self.turns = turns
        self.turns_by_id = {t.dia_id: t for t in turns}
        self.graph = graph
        self.sample_id = sample_id
        self.device = device if (torch.cuda.is_available() and device == "cuda") else "cpu"
        self.ce = _get_cross_encoder(device=self.device)

        # Baseline hybrid retriever for seed discovery
        self.base_retriever = HybridCEControlRetriever(turns=turns, sample_id=sample_id, device=self.device)
        self.expander = BoundedGraphExpansion(
            graph=graph,
            max_depth=max_depth,
            max_branching=max_branching,
            max_expanded_candidates=max_expanded_candidates,
        )

    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        # 1. Retrieve initial candidate seeds
        initial_hits = self.base_retriever.retrieve(query, k=k)
        if not initial_hits:
            return []

        candidate_turn_ids: Set[str] = set()
        seed_entities: List[str] = []
        seed_scores: Dict[str, float] = {}

        for h in initial_hits:
            candidate_turn_ids.add(h.dia_id)
            # Find entities linked to this turn in graph
            for node, evidence_set in self.graph.node_evidence.items():
                if h.dia_id in evidence_set:
                    seed_entities.append(node)
                    seed_scores[node] = max(seed_scores.get(node, 0.0), h.score)

        # 2. Bounded graph expansion
        expanded_nodes = self.expander.expand(seed_entities[:3], seed_scores=seed_scores)

        for _, _, evidence_set in expanded_nodes:
            for tid in evidence_set:
                candidate_turn_ids.add(tid)

        # 3. Cross-Encoder reranking over combined candidate pool
        pool_turns: List[ExperienceTurn] = [
            self.turns_by_id[tid] for tid in candidate_turn_ids if tid in self.turns_by_id
        ]
        if not pool_turns:
            return initial_hits[:k]

        pairs = [[query, f"{t.speaker}: {t.text}"] for t in pool_turns]
        try:
            scores = self.ce.predict(pairs)
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
        except Exception:
            scores = [0.0] * len(pool_turns)

        scored_turns = list(zip(scores, pool_turns))
        scored_turns.sort(key=lambda x: x[0], reverse=True)

        results = []
        for sc, t in scored_turns[:k]:
            is_expanded = t.dia_id not in {h.dia_id for h in initial_hits}
            results.append(
                RetrievalResult(
                    item_id=t.dia_id,
                    dia_id=t.dia_id,
                    text=f"{t.speaker}: {t.text}",
                    score=round(float(sc), 4),
                    source_type="graph_expansion" if is_expanded else "hybrid_seed",
                    speaker=t.speaker,
                    timestamp=t.timestamp,
                    underlying_dia_ids=[t.dia_id],
                )
            )
        return results


# ============================================================================
# Variant E: ConsolidatedSemanticRetriever (Raw turns + atomic facts)
# ============================================================================
class ConsolidatedSemanticRetriever:
    """
    Jointly indexes raw conversation turns and atomic semantic facts.
    When a semantic fact matches, its source_turn_ids are preserved and expanded.
    """

    def __init__(
        self,
        turns: List[ExperienceTurn],
        facts: List[SemanticFact],
        sample_id: str = "sample",
        device: str = "cuda",
    ):
        self.turns = turns
        self.facts = facts
        self.turns_by_id = {t.dia_id: t for t in turns}
        self.sample_id = sample_id
        self.device = device if (torch.cuda.is_available() and device == "cuda") else "cpu"
        self.repo = InMemoryRepository()

        # 1. Index raw turns
        for t in self.turns:
            self.repo.save({
                "id": f"{self.sample_id}:turn:{t.dia_id}",
                "dia_id": t.dia_id,
                "source_turn_id": t.dia_id,
                "key": f"turn_{t.dia_id}",
                "value": f"{t.speaker}: {t.text}",
                "contextual_text": t.contextual_text or f"In session {t.session_num} ({t.timestamp}), {t.speaker} said: \"{t.text}\"",
                "type": "dialogue_turn",
                "speaker": t.speaker,
                "session": t.session_num,
                "timestamp": t.timestamp,
                "status": "active",
            })

        # 2. Index atomic semantic facts
        for f in self.facts:
            primary_dia = f.source_turn_ids[0] if f.source_turn_ids else ""
            self.repo.save({
                "id": f"{self.sample_id}:fact:{f.fact_id}",
                "dia_id": primary_dia,
                "source_turn_id": primary_dia,
                "source_turn_ids": f.source_turn_ids,
                "key": f"{f.subject}_{f.predicate}",
                "value": f.statement,
                "contextual_text": f"Extracted fact ({f.validity_status}): {f.statement}",
                "type": "semantic_fact",
                "speaker": "",
                "timestamp": f.timestamp,
                "status": f.validity_status,
            })

        self.retriever = ProductionRetriever(
            repository=self.repo,
            config_name="hybrid_ce_abstain",
            query_log_path=None,
            device=self.device,
        )
        self.retriever._ensure_index()

    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        hits = self.retriever.retrieve(query, k=k * 2)
        results = []
        seen_turn_ids: Set[str] = set()

        for h in hits:
            stype = h.get("type", "turn")
            did = h.get("dia_id") or h.get("source_turn_id")
            underlying = h.get("source_turn_ids") or ([did] if did else [])

            # Prefer mapping fact to its underlying turn evidence
            if did and did not in seen_turn_ids:
                seen_turn_ids.add(did)
                t_obj = self.turns_by_id.get(did)
                results.append(
                    RetrievalResult(
                        item_id=str(h.get("id")),
                        dia_id=did,
                        text=f"{t_obj.speaker}: {t_obj.text}" if t_obj else str(h.get("value", "")),
                        score=float(h.get("_ce_score", h.get("_dense_score", 0.0))),
                        source_type="semantic_fact_provenance" if stype == "semantic_fact" else "turn",
                        speaker=t_obj.speaker if t_obj else "",
                        timestamp=t_obj.timestamp if t_obj else str(h.get("timestamp", "")),
                        underlying_dia_ids=[str(u) for u in underlying],
                    )
                )
            if len(results) >= k:
                break

        return results[:k]


# ============================================================================
# Variant F: ConsolidatedAssociativeRetriever (Consolidation + Bounded Graph)
# ============================================================================
class ConsolidatedAssociativeRetriever:
    """
    Combines atomic semantic consolidation with bounded associative graph expansion.
    Tests H3: Whether consolidation and associative links provide complementary value.
    """

    def __init__(
        self,
        turns: List[ExperienceTurn],
        facts: List[SemanticFact],
        graph: AssociativeGraph,
        sample_id: str = "sample",
        device: str = "cuda",
    ):
        self.turns = turns
        self.facts = facts
        self.graph = graph
        self.sample_id = sample_id
        self.device = device if (torch.cuda.is_available() and device == "cuda") else "cpu"

        # Semantic consolidation base
        self.semantic_retriever = ConsolidatedSemanticRetriever(
            turns=turns, facts=facts, sample_id=sample_id, device=self.device
        )
        # Graph expansion
        self.expander = BoundedGraphExpansion(
            graph=graph,
            max_depth=2,
            max_branching=3,
            max_expanded_candidates=5,
        )
        self.turns_by_id = {t.dia_id: t for t in turns}
        self.ce = _get_cross_encoder(device=self.device)

    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        initial_hits = self.semantic_retriever.retrieve(query, k=k)
        if not initial_hits:
            return []

        candidate_turn_ids: Set[str] = set()
        seed_entities: List[str] = []
        seed_scores: Dict[str, float] = {}

        for h in initial_hits:
            candidate_turn_ids.add(h.dia_id)
            for nid, ev_set in self.graph.node_evidence.items():
                if h.dia_id in ev_set:
                    seed_entities.append(nid)
                    seed_scores[nid] = max(seed_scores.get(nid, 0.0), h.score)

        # Graph expansion
        expanded = self.expander.expand(seed_entities[:3], seed_scores=seed_scores)
        for _, _, ev_set in expanded:
            for tid in ev_set:
                candidate_turn_ids.add(tid)

        # Rerank
        pool = [self.turns_by_id[tid] for tid in candidate_turn_ids if tid in self.turns_by_id]
        if not pool:
            return initial_hits[:k]

        pairs = [[query, f"{t.speaker}: {t.text}"] for t in pool]
        try:
            scores = self.ce.predict(pairs)
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
        except Exception:
            scores = [0.0] * len(pool)

        scored = sorted(zip(scores, pool), key=lambda x: x[0], reverse=True)
        results = []
        for sc, t in scored[:k]:
            results.append(
                RetrievalResult(
                    item_id=t.dia_id,
                    dia_id=t.dia_id,
                    text=f"{t.speaker}: {t.text}",
                    score=round(float(sc), 4),
                    source_type="consolidated_associative",
                    speaker=t.speaker,
                    timestamp=t.timestamp,
                    underlying_dia_ids=[t.dia_id],
                )
            )
        return results


# ============================================================================
# Variant G: QueryRoutedAssociativeRetriever (Intent-aware dynamic router)
# ============================================================================
class QueryRoutedAssociativeRetriever:
    """
    Inspects query features and routes to optimal representation:
    - Adversarial query patterns -> Abstain
    - Multi-hop ("why", "how did", "relationship", "connect", "both") -> Variant D
    - Temporal ("when", "before", "after", "date", "year") -> Variant E
    - Exact lexical quotes -> Variant A / B
    - Default -> Variant C (Hybrid CE Control)
    """

    _TEMPORAL_PATTERNS = re.compile(r"\b(when|before|after|date|month|year|day|time|recent|earlier|latest|first|last|since)\b", re.IGNORECASE)
    _MULTIHOP_PATTERNS = re.compile(r"\b(why|how did|relationship|connect|both|together|reason|caused|led to|influence)\b", re.IGNORECASE)
    _ADVERSARIAL_PATTERNS = re.compile(r"\b(why did .* never|when did .* steal|untrue|fake|unicorn|alien)\b", re.IGNORECASE)

    def __init__(
        self,
        variant_a: GrepRetriever,
        variant_b: BM25Retriever,
        variant_c: HybridCEControlRetriever,
        variant_d: BoundedAssociativeRetriever,
        variant_e: ConsolidatedSemanticRetriever,
        variant_f: ConsolidatedAssociativeRetriever,
    ):
        self.var_a = variant_a
        self.var_b = variant_b
        self.var_c = variant_c
        self.var_d = variant_d
        self.var_e = variant_e
        self.var_f = variant_f

    def route_query(self, query: str) -> str:
        q = query.strip()
        if self._ADVERSARIAL_PATTERNS.search(q):
            return "abstain"
        if '"' in q or "'" in q:
            return "variant_a"  # Exact lexical match
        if self._MULTIHOP_PATTERNS.search(q):
            return "variant_d"  # Bounded associative expansion
        if self._TEMPORAL_PATTERNS.search(q):
            return "variant_e"  # Semantic consolidation with temporal validity
        return "variant_c"      # Hybrid CE control

    def retrieve(self, query: str, k: int = 5) -> Tuple[List[RetrievalResult], str]:
        selected = self.route_query(query)
        if selected == "abstain":
            return [], "abstain"
        elif selected == "variant_a":
            res = self.var_a.retrieve(query, k=k)
            # If grep yields no exact hits, fallback to variant_c
            if not res:
                res = self.var_c.retrieve(query, k=k)
                return res, "variant_a_fallback_c"
            return res, "variant_a"
        elif selected == "variant_d":
            return self.var_d.retrieve(query, k=k), "variant_d"
        elif selected == "variant_e":
            return self.var_e.retrieve(query, k=k), "variant_e"
        else:
            return self.var_c.retrieve(query, k=k), "variant_c"
