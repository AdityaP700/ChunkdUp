# ai-system/chunkdup/retrieval/reranker.py
import math
from typing import List, Dict, Any, Optional

class PluggableReranker:
    """
    Pluggable Reranker supporting:
    - 'cross_encoder': Neural cross-attention re-scoring via sentence-transformers
    - 'llm': Listwise heuristic ranking with relevance explanation simulation
    - 'heuristic': Feature-weighted fast reranker (type boost + score spread)
    """
    def __init__(self, mode: str = "heuristic", model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.mode = mode
        self.model_name = model_name
        self._cross_encoder = None

        if self.mode == "cross_encoder":
            try:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder(self.model_name)
            except Exception as e:
                # Fallback to heuristic mode if cross-encoder package/model fails to load
                print(f"[Reranker Warning] Failed to load CrossEncoder '{model_name}': {e}. Falling back to 'heuristic'.")
                self.mode = "heuristic"

    def rerank(self, query: str, candidates: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        if self.mode == "cross_encoder" and self._cross_encoder:
            return self._rerank_cross_encoder(query, candidates, limit)
        elif self.mode == "llm":
            return self._rerank_llm(query, candidates, limit)
        else:
            return self._rerank_heuristic(query, candidates, limit)

    def _rerank_cross_encoder(self, query: str, candidates: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        pairs = [[query, c.get("contextual_text", f"{c.get('key')} {c.get('value')}")] for c in candidates]
        scores = self._cross_encoder.predict(pairs)

        scored_candidates = []
        for doc, score in zip(candidates, scores):
            doc_copy = dict(doc)
            doc_copy["_rerank_score"] = float(score)
            scored_candidates.append(doc_copy)

        scored_candidates.sort(key=lambda x: x["_rerank_score"], reverse=True)
        return scored_candidates[:limit]

    def _rerank_llm(self, query: str, candidates: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        # Simulate LLM Listwise scoring based on exact token alignment and semantic relevance
        q_words = set(query.lower().split())
        scored = []
        for c in candidates:
            c_copy = dict(c)
            val = str(c.get("value", "")).lower()
            key = str(c.get("key", "")).lower()
            c_words = set(f"{key} {val}".split())
            overlap = len(q_words & c_words)
            llm_score = c.get("_score", 0.0) + (overlap * 0.5)
            c_copy["_rerank_score"] = round(llm_score, 4)
            scored.append(c_copy)

        scored.sort(key=lambda x: x["_rerank_score"], reverse=True)
        return scored[:limit]

    def _rerank_heuristic(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Fast reranker combining semantic similarity with lightweight
        type and lexical relevance signals.
        """
        type_weights = {
            "programming_language": 1.05,
            "project_name": 1.05,
            "environment": 1.05,
            "tool": 1.05,
            "editor": 1.05,
            "preference": 1.0,
            "response_style": 0.95,
        }

        q_words = set(query.lower().split())
        scored = []

        for c in candidates:
            c_copy = dict(c)

            semantic_score = max(
                0.0,
                float(c.get("_similarity", 0.0))
            )

            hybrid_score = float(c.get("_score", 0.0))

            text = f"{c.get('key', '')} {c.get('value', '')}".lower()
            doc_words = set(text.split())

            overlap = len(q_words & doc_words)

            lexical_score = min(
                overlap / max(len(q_words), 1),
                1.0
            )

            type_boost = type_weights.get(
                c.get("type"),
                1.0
            )

            final_score = (
                semantic_score * 0.75
                + lexical_score * 0.15
                + hybrid_score * 0.10
            ) * type_boost

            c_copy["_rerank_score"] = round(
                final_score,
                4
            )

            scored.append(c_copy)

        scored.sort(
            key=lambda x: x["_rerank_score"],
            reverse=True
        )

        return scored[:limit]