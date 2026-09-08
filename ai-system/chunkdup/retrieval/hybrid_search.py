# ai-system/chunkdup/retrieval/hybrid_search.py
from typing import List, Dict, Any

class HybridSearchEngine:
    """
    Implements Hybrid Search with Reciprocal Rank Fusion (RRF),
    Dynamic Intent-based Type Boosting, and Contextual Expansion.
    """
    def __init__(self, keyword_weight: float = 0.4, semantic_weight: float = 0.6, entity_weight: float = 0.5, type_boost: Dict[str, float] = None):
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        self.entity_weight = entity_weight
        self.last_diagnostics = []
        self.type_boost = type_boost or {
           "programming_language": 2.0,
    "database": 1.8,
    "runtime_environment": 2.5,   
    "embedding_model": 2.0,
    "environment": 1.8,
    "tool": 1.5,
    "project": 1.5,
    "employment": 1.3,
    "role": 1.3,
    "learning": 1.2,
    "preference": 1.0,
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
        diagnostics = {}

        def add_rrf_result(item, rank, weight, signal):
            m_id = item["id"]
            m_type = item.get("type", "")
            type_score = self.type_boost.get(m_type, 1.0) * self._get_dynamic_intent_boost(query, m_type)
            contribution = weight * (1.0 / (k + rank + 1)) * type_score
            scores[m_id] = scores.get(m_id, 0.0) + contribution
            diagnostics.setdefault(m_id, {"keyword": 0.0, "semantic": 0.0})[signal] += contribution
            diagnostics[m_id]["type"] = type_score
            metadata.setdefault(m_id, dict(item))
            if item.get("_entity_score", 0.0) > metadata[m_id].get("_entity_score", 0.0):
                metadata[m_id]["_entity_score"] = item["_entity_score"]

        # 1. Apply static + dynamic intent boost before RRF
        for rank, item in enumerate(keyword_results):
            add_rrf_result(item, rank, self.keyword_weight, "keyword")

        for rank, item in enumerate(semantic_results):
            add_rrf_result(item, rank, self.semantic_weight, "semantic")

        for m_id, item in metadata.items():
            entity_score = item.get("_entity_score", 0.0)
            rrf_score = scores[m_id]
            final_score = rrf_score + self.entity_weight * entity_score
            metadata[m_id]["_rrf_score"] = round(rrf_score, 6)
            metadata[m_id]["_score"] = round(final_score, 6)
            diagnostics.setdefault(m_id, {}).update({
                "entity": round(entity_score, 6),
                "rrf": round(rrf_score, 6),
                "final": round(final_score, 6),
            })

        sorted_ids = sorted(scores.keys(), key=lambda x: metadata[x]["_score"], reverse=True)
        results = [metadata[m_id] for m_id in sorted_ids[:limit]]
        self.last_diagnostics = [
            {
                "id": item["id"],
                "key": item.get("key"),
                "value": item.get("value"),
                **diagnostics[item["id"]],
            }
            for item in results
        ]
        return results
