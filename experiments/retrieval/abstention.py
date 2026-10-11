# experiments/retrieval/abstention.py
"""
Relevance-decision (abstention) stage.

Runs AFTER candidate ranking and reranking. Understands the score semantics
of its upstream pipeline via the score_field it is configured with (cosine
similarity, BM25 score, RRF score, or cross-encoder logit are NOT comparable,
so each pipeline carries its own calibrated threshold).

Behavior:
  - If the top candidate's score is below the threshold -> return [] (abstain).
  - Otherwise keep every candidate at or above the threshold, preserving
    low-confidence-but-relevant memories instead of cutting the list at an
    arbitrary k.

An optional top1-top2 gap veto exists but is disabled by default; it is only
enabled when dev-set calibration shows the separation signal helps.
"""
from typing import Dict, List, Optional


class AbstentionGate:
    def __init__(self, score_field: str, threshold: float, min_top_gap: Optional[float] = None):
        self.score_field = score_field
        self.threshold = threshold
        self.min_top_gap = min_top_gap  # None disables the gap signal

    def decide(self, results: List[Dict]) -> Dict:
        """Returns {'abstained': bool, 'top_score': float|None, 'gap': float|None}."""
        if not results:
            return {"abstained": True, "top_score": None, "gap": None}
        top_score = float(results[0].get(self.score_field, 0.0))
        gap = None
        if len(results) > 1:
            second = float(results[1].get(self.score_field, 0.0))
            gap = top_score - second
        abstained = top_score < self.threshold
        if not abstained and self.min_top_gap is not None and gap is not None:
            # Only veto when the top candidate barely leads the runner-up.
            abstained = gap < self.min_top_gap
        return {"abstained": abstained, "top_score": top_score, "gap": gap}

    def filter(self, results: List[Dict]) -> List[Dict]:
        decision = self.decide(results)
        if decision["abstained"]:
            return []
        return [
            r for r in results
            if float(r.get(self.score_field, 0.0)) >= self.threshold
        ]
