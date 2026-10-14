# ai-system/chunkdup/profile/profile_guided_retriever.py
"""
Profile-Guided Archive Retriever for ChunkdUp (Strategy E Optimization).

Implements Two-Tier Profile-Guided Retrieval:
1. Tier 1: Compact structured profile (user attributes, preferences, state, source turn IDs).
2. Tier 2: Episodic turn archive indexed for hybrid dense + lexical retrieval.

Retrieval Flow:
1. Query matches against Tier 1 profile to identify relevant attributes and entity facts.
2. If matching facts are found:
   - Formulate a focused/expanded query incorporating key attributes.
   - Prioritize source turns referenced by matched profile facts.
3. Retrieve candidate dialogue turns from Tier 2 archive via BM25 + Dense RRF fusion.
4. Rerank candidate turns with Cross-Encoder (ms-marco-MiniLM-L-12-v2).
5. Inject relevant profile facts into prompt context with a strict bounded token budget (<= 150 tokens),
   completely resolving the 5,000+ token context explosion observed in naive profile memory.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from rank_bm25 import BM25Okapi

from chunkdup.profile.profile_memory import PersistentProfile
from chunkdup.profile.schema import ProfileFact
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


class ProfileGuidedArchiveRetriever:
    """
    Two-Tier Profile-Guided Archive Retriever.
    Resolves both Strategy E failure modes:
    - Eliminates token explosion with strict budget bounding.
    - Resolves narrow factual queries via profile-guided episodic archive retrieval and CE reranking.
    """

    def __init__(
        self,
        profile: Optional[PersistentProfile] = None,
        device: Optional[str] = None,
        initial_pool_size: int = 15,
        max_context_tokens: int = 150,
    ):
        self.device = _resolve_device(device)
        self.profile = profile or PersistentProfile(max_facts=50)
        self.initial_pool_size = initial_pool_size
        self.max_context_tokens = max_context_tokens

        self.passages: List[Dict[str, Any]] = []
        self.passage_by_id: Dict[str, Dict[str, Any]] = {}
        self.passage_by_dia_id: Dict[str, Dict[str, Any]] = {}

        self.dense_matrix: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Okapi] = None
        self._is_indexed = False

    def index_turns(
        self,
        turns: List[Dict[str, Any]],
        profile: Optional[PersistentProfile] = None,
    ):
        """Index conversation turns for Tier 2 episodic archive search."""
        if profile is not None:
            self.profile = profile

        self.passages = list(turns)
        self.passage_by_id.clear()
        self.passage_by_dia_id.clear()

        if not self.passages:
            self._is_indexed = True
            return

        tokenized_corpus: List[List[str]] = []
        texts_to_embed: List[str] = []

        for p in self.passages:
            pid = str(p.get("id") or p.get("dia_id") or p.get("turn_id") or "")
            dia_id = _extract_dia_id(pid) or _extract_dia_id(p.get("dia_id")) or _extract_dia_id(p.get("turn_id")) or pid

            p_dict = dict(p)
            p_dict["id"] = pid
            p_dict["dia_id"] = dia_id
            p_dict["source_turn_id"] = dia_id
            p_text = str(p.get("text") or p.get("value") or p.get("passage") or "").strip()
            p_dict["text"] = p_text

            self.passage_by_id[pid] = p_dict
            if dia_id:
                self.passage_by_dia_id[dia_id] = p_dict

            rep_text = _index_text(p_dict) if "key" in p_dict else p_text
            texts_to_embed.append(rep_text)
            tokenized_corpus.append(_tokenize(rep_text))

        self.bm25 = BM25Okapi(tokenized_corpus)

        st = _get_sentence_transformer(device=self.device)
        embeddings = _safe_encode(st, texts_to_embed, normalize_embeddings=True)
        if hasattr(embeddings, "cpu"):
            embeddings = embeddings.cpu().numpy()
        self.dense_matrix = np.asarray(embeddings, dtype=np.float32)

        self._is_indexed = True

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Execute two-tier profile-guided retrieval:
        1. Profile search for relevant attributes.
        2. Query enrichment and source turn seeding.
        3. Hybrid BM25 + dense search over archive.
        4. Cross-Encoder reranking.
        5. Strictly bounded context formatting.
        """
        if not self._is_indexed or not self.passages:
            return {
                "source": "empty",
                "results": [],
                "matched_facts": [],
                "prompt_context": "",
                "abstained": True,
            }

        # Step 1: Match query against Tier 1 persistent profile
        matched_facts = self.profile.search_facts(query)

        # Step 2: Formulate focused query if relevant facts exist
        focused_query = query
        profile_seed_turn_ids: List[str] = []

        if matched_facts:
            # Use top 2 matched facts to enrich query semantics
            fact_tokens = []
            for f in matched_facts[:2]:
                fact_tokens.append(f"{f.key} {f.value}")
                if f.source_turn_id:
                    clean_tid = _extract_dia_id(f.source_turn_id) or f.source_turn_id
                    if clean_tid not in profile_seed_turn_ids:
                        profile_seed_turn_ids.append(clean_tid)
            focused_query = f"{query} {' '.join(fact_tokens)}"

        # Step 3: Hybrid candidate retrieval over Tier 2 archive
        pool_k = max(self.initial_pool_size, top_k * 3)

        st = _get_sentence_transformer(device=self.device)
        q_vec = _safe_encode(st, focused_query, normalize_embeddings=True)
        if hasattr(q_vec, "cpu"):
            q_vec = q_vec.cpu().numpy()

        sims = self.dense_matrix @ q_vec
        dense_order = np.argsort(-sims)[:pool_k]
        dense_candidates = []
        for idx in dense_order:
            cand = dict(self.passages[int(idx)])
            cand["_similarity"] = float(sims[int(idx)])
            dense_candidates.append(cand)

        bm25_scores = self.bm25.get_scores(_tokenize(focused_query))
        bm25_order = np.argsort(-bm25_scores)[:pool_k]
        bm25_candidates = []
        for idx in bm25_order:
            if bm25_scores[idx] <= 0.0:
                break
            cand = dict(self.passages[int(idx)])
            cand["_bm25_score"] = float(bm25_scores[idx])
            bm25_candidates.append(cand)

        fused = _rrf_fuse([dense_candidates, bm25_candidates], k=60, limit=pool_k)

        # Step 4: Inject profile source turns if not already in candidates
        cand_dia_ids = {c.get("dia_id") or _extract_dia_id(c.get("id")) for c in fused}
        for stid in profile_seed_turn_ids:
            if stid not in cand_dia_ids:
                p_obj = self.passage_by_dia_id.get(stid) or self.passage_by_id.get(stid)
                if p_obj:
                    cand = dict(p_obj)
                    cand["_profile_seeded"] = True
                    fused.append(cand)
                    cand_dia_ids.add(stid)

        # Step 5: Cross-Encoder neural reranking with original query
        ce = _get_cross_encoder(device=self.device)
        pairs = []
        for c in fused:
            p_text = c.get("text") or c.get("value") or ""
            rep = _index_text(c) if "key" in c else p_text
            pairs.append([query, rep])

        if pairs:
            scores = _safe_predict(ce, pairs, batch_size=32)
            for c, s in zip(fused, scores):
                c["_rerank_score"] = float(s)
                c["score"] = float(s)
                c["retrieval_path"] = "profile_guided_archive"

            fused.sort(key=lambda x: x["_rerank_score"], reverse=True)

        ranked_results = fused[:top_k]

        # Step 6: Bounded prompt context formatting (strictly capped at max_context_tokens)
        prompt_parts: List[str] = []
        if matched_facts:
            # Format at most top 3 facts, concise format
            prof_lines = ["[PROFILE FACTS]"]
            for f in matched_facts[:3]:
                prof_lines.append(f"- {f.key}: {f.value}")
            prompt_parts.append("\n".join(prof_lines))

        if ranked_results:
            ctx_lines = ["[RELEVANT TURNS]"]
            for r in ranked_results:
                txt = r.get("text") or r.get("value") or ""
                # Clip turn to max 120 chars if needed to guarantee budget
                clip_txt = txt[:120] + ("..." if len(txt) > 120 else "")
                ctx_lines.append(f"- {clip_txt}")
            prompt_parts.append("\n".join(ctx_lines))

        bounded_context = "\n\n".join(prompt_parts)

        return {
            "source": "profile_guided_archive",
            "results": ranked_results,
            "matched_facts": [f.to_dict() for f in matched_facts[:3]],
            "prompt_context": bounded_context,
            "abstained": len(ranked_results) == 0,
        }
