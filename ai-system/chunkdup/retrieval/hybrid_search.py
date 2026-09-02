# ai-system/chunkdup/retrieval/hybrid_search.py
from typing import List, Dict, Any

class HybridSearchEngine:
    """
    Implements Hybrid Search with Reciprocal Rank Fusion (RRF),
    Dynamic Intent-based Type Boosting, and Contextual Expansion.
    """
    def __init__(self, keyword_weight: float = 0.4, semantic_weight: float = 0.6, type_boost: Dict[str, float] = None):
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        self.type_boost = type_boost or {
            "programming_language": 2.0,
            "project_name": 1.5,
            "company": 2.5,
            "role": 2.5,
            "environment": 1.3,
            "editor": 1.5,
            "preference": 2.0,
            "response_style": 2.0
        }

    def _get_dynamic_intent_boost(self, query: str, candidate_type: str) -> float:
        q_lower = query.lower()
        # Fix 4: Career intent boost
        if any(w in q_lower for w in ["work", "company", "job", "role", "employer", "title"]):
            if candidate_type in ["company", "role", "employment", "job_title"]:
                return 3.0
        # Fix 4: Preference / style intent boost
        if any(w in q_lower for w in ["prefer", "style", "respond", "communication", "how should"]):
            if candidate_type in ["preference", "response_style"]:
                return 3.0
        # Fix 4: Tool / Editor intent boost
        if any(w in q_lower for w in ["editor", "ide", "tool", "editor or ide"]):
            if candidate_type in ["editor", "tool"]:
                return 2.5
        return 1.0

    def combine(self, query: str, keyword_results: List[Dict[str, Any]], semantic_results: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
        k = 60  # RRF constant
        scores = {}
        metadata = {}

        # 1. Apply static + dynamic intent boost before RRF
        for rank, item in enumerate(keyword_results):
            m_id = item["id"]
            m_type = item.get("type", "")
            base_boost = self.type_boost.get(m_type, 1.0)
            intent_boost = self._get_dynamic_intent_boost(query, m_type)
            total_boost = base_boost * intent_boost

            rrf_score = self.keyword_weight * (1.0 / (k + rank + 1)) * total_boost
            scores[m_id] = scores.get(m_id, 0.0) + rrf_score
            metadata[m_id] = dict(item)

        for rank, item in enumerate(semantic_results):
            m_id = item["id"]
            m_type = item.get("type", "")
            base_boost = self.type_boost.get(m_type, 1.0)
            intent_boost = self._get_dynamic_intent_boost(query, m_type)
            total_boost = base_boost * intent_boost

            rrf_score = self.semantic_weight * (1.0 / (k + rank + 1)) * total_boost
            scores[m_id] = scores.get(m_id, 0.0) + rrf_score
            if m_id not in metadata:
                metadata[m_id] = dict(item)

        for m_id, item in metadata.items():
            metadata[m_id]["_score"] = round(scores[m_id], 6)

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [metadata[m_id] for m_id in sorted_ids[:limit]]
