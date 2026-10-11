# experiments/retrieval/evaluate.py
"""
Metric computation shared by every experiment run.

Metric conventions (explicit, per the task spec):
  - Precision@k is computed ONLY over the results actually returned. A query
    that correctly returns [] gets precision 1.0 (nothing irrelevant returned);
    a query that returns junk gets precision 0.0.
  - Recall@k: |relevant ∩ returned| / |relevant|. For unanswerable queries,
    recall is 1.0 when [] is returned and 0.0 otherwise (matching the original
    benchmark's convention).
  - HitRate@k: fraction of queries with at least one relevant memory in the
    returned list. Unanswerable queries count as hits when [] is returned
    (a correct rejection is the "hit").
  - MRR@k: 1/rank of the first relevant memory in the returned list; 0 when
    the list is empty or contains no relevant memory. Unanswerable queries
    contribute 1.0 when [] is correctly returned (rank 0 convention).

Abstention-specific metrics are reported separately so correct rejections are
never hidden inside aggregate precision.
"""
from typing import Dict, List

K = 5


def score_query(row: Dict, returned_ids: List[str]) -> Dict:
    expected = set(row["relevant_memory_ids"])
    returned = list(returned_ids)[:K]
    hits = [mid for mid in returned if mid in expected]

    if expected:
        precision = len(hits) / len(returned) if returned else 0.0
        recall = len(hits) / len(expected)
        rr = 1.0 / (returned.index(hits[0]) + 1) if hits else 0.0
        hit = 1.0 if hits else 0.0
    else:
        correct_rejection = not returned
        precision = 1.0 if correct_rejection else 0.0
        recall = 1.0 if correct_rejection else 0.0
        rr = 1.0 if correct_rejection else 0.0
        hit = 1.0 if correct_rejection else 0.0

    return {
        "id": row["id"],
        "precision": precision,
        "recall": recall,
        "reciprocal_rank": rr,
        "hit": hit,
    }


def aggregate(per_query: List[Dict], rows: List[Dict]) -> Dict:
    n = len(per_query)
    answerable = [pq for pq, row in zip(per_query, rows) if row["answerable"]]
    unanswerable = [pq for pq, row in zip(per_query, rows) if not row["answerable"]]

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    # Abstention accounting (explicit booleans recorded per query by the runner)
    def count(pred, subset):
        return sum(1 for pq in subset if pred(pq))

    answerable_fn_rate = (
        count(lambda pq: pq["abstained"], answerable) / len(answerable) if answerable else None
    )
    unanswerable_reject_rate = (
        count(lambda pq: pq["abstained"], unanswerable) / len(unanswerable)
        if unanswerable else None
    )
    unanswerable_fp_rate = (
        count(lambda pq: not pq["abstained"], unanswerable) / len(unanswerable)
        if unanswerable else None
    )

    return {
        "n_queries": n,
        "MRR": round(mean([pq["reciprocal_rank"] for pq in per_query]), 4),
        "Precision@K": round(mean([pq["precision"] for pq in per_query]), 4),
        "Recall@K": round(mean([pq["recall"] for pq in per_query]), 4),
        "HitRate@K": round(mean([pq["hit"] for pq in per_query]), 4),
        "answerable": {
            "n": len(answerable),
            "MRR": round(mean([pq["reciprocal_rank"] for pq in answerable]), 4) if answerable else None,
            "HitRate@K": round(mean([pq["hit"] for pq in answerable]), 4) if answerable else None,
            "false_abstain_rate": round(answerable_fn_rate, 4) if answerable_fn_rate is not None else None,
            "relevant_memories_returned": sum(pq["n_relevant_returned"] for pq in answerable),
        },
        "unanswerable": {
            "n": len(unanswerable),
            "correct_rejection_rate": round(unanswerable_reject_rate, 4) if unanswerable_reject_rate is not None else None,
            "false_positive_rate": round(unanswerable_fp_rate, 4) if unanswerable_fp_rate is not None else None,
            "irrelevant_memories_returned": sum(pq["n_irrelevant_returned"] for pq in unanswerable),
        },
    }
