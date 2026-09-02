# ai-system/chunkdup/retrieval/adaptive_k.py
from typing import List, Dict, Any

class AdaptiveKSelector:
    """
    Dynamically calculates candidate retrieval size and final k limit
    based on query ambiguity, token length, and confidence score distribution.
    """
    @staticmethod
    def select_candidate_k(target_k: int = 5) -> int:
        """Candidate retrieval pool should be generous (min 10) to avoid trimming true positives before reranking."""
        return max(10, target_k * 2)

    @staticmethod
    def select_k(query: str, results: List[Dict[str, Any]], target_k: int = 5) -> int:
        if not results:
            return 0

        # 1. Synthesis or broad entity query -> expand target k
        ambiguous_keywords = ["stack", "environment", "all", "overview", "summary", "setup", "tools", "languages", "career"]
        if any(kw in query.lower() for kw in ambiguous_keywords):
            return min(max(5, target_k + 3), len(results))

        # 2. Strict out-of-domain check: if maximum score is trivial, return 0
        max_score = max([r.get("_rerank_score", r.get("_score", 0.0)) for r in results]) if results else 0.0
        if max_score < 0.05:
            return 0

        return min(target_k, len(results))
