# ai-system/chunkdup/retrieval/adaptive_k.py
from typing import List, Dict, Any

class AdaptiveKSelector:
    """
    Dynamically calculates candidate retrieval size and final k limit
    based on query ambiguity, token length, and confidence score distribution.
    """
    @staticmethod
    def select_k(query: str, results: List[Dict[str, Any]], target_k: int = 5) -> int:
        if not results:
            return 0

        # 1. Very precise query (1 or 2 tokens, e.g. "Python") -> tight k
        tokens = query.strip().split()
        if len(tokens) <= 2:
            return min(3, len(results))

        # 2. High confidence spread check (top score significantly outperforms tail)
        scores = [r.get("_rerank_score", r.get("_score", 0.0)) for r in results]
        if len(scores) >= 2:
            max_s = max(scores)
            min_s = min(scores)
            if (max_s - min_s) > 0.5:
                # Top candidates are clearly dominant
                return min(3, len(results))

        # 3. Ambiguous / multi-entity synthesis queries -> larger k
        ambiguous_keywords = ["stack", "environment", "all", "overview", "summary", "setup", "tools"]
        if any(kw in query.lower() for kw in ambiguous_keywords):
            return min(max(5, target_k + 2), len(results))

        return min(target_k, len(results))
