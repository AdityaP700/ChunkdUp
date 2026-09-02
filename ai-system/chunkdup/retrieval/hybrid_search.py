# ai-system/chunkdup/retrieval/hybrid_search.py
from typing import List, Dict, Any

class HybridSearchEngine:
    """
    Implements Hybrid Search with Reciprocal Rank Fusion (RRF)
    and configurable weights for Keyword vs Semantic components.
    """
    def __init__(self, keyword_weight: float = 0.4, semantic_weight: float = 0.6, type_boost: Dict[str, float] = None):
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        self.type_boost = type_boost or {
            "programming_language": 2.0,
            "project_name": 1.5,
            "environment": 1.3,
            "editor": 1.0,
            "preference": 0.8,
            "response_style": 0.6
        }

    def combine(self, keyword_results: List[Dict[str, Any]], semantic_results: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
        k = 60  # RRF constant
        scores = {}
        metadata = {}

        # 1. Score keyword results
        for rank, item in enumerate(keyword_results):
            m_id = item["id"]
            rrf_score = self.keyword_weight * (1.0 / (k + rank + 1))
            scores[m_id] = scores.get(m_id, 0.0) + rrf_score
            metadata[m_id] = dict(item)

        # 2. Score semantic results
        for rank, item in enumerate(semantic_results):
            m_id = item["id"]
            rrf_score = self.semantic_weight * (1.0 / (k + rank + 1))
            scores[m_id] = scores.get(m_id, 0.0) + rrf_score
            if m_id not in metadata:
                metadata[m_id] = dict(item)

        # 3. Apply type boost AFTER initial candidates are established
        for m_id, item in metadata.items():
            m_type = item.get("type")
            boost = self.type_boost.get(m_type, 1.0)
            scores[m_id] *= boost
            metadata[m_id]["_score"] = round(scores[m_id], 6)

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [metadata[m_id] for m_id in sorted_ids[:limit]]
