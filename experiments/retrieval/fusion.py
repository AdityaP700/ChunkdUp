# experiments/retrieval/fusion.py
"""
Standard Reciprocal Rank Fusion (RRF).

    RRF(d) = sum over lists of 1 / (k + rank_1based(d))

A candidate absent from a ranked list contributes nothing from that list.
Candidates are deduplicated by their stable fixture id. No score bonuses:
type/intent/entity bonuses are intentionally NOT part of this stage; the
legacy weighted engine remains available separately as a labeled baseline
configuration so its bonuses can be measured rather than assumed.
"""
from typing import Dict, List


def rrf_fuse(result_lists: List[List[Dict]], k: int = 60, limit: int = 20) -> List[Dict]:
    """Fuse ranked candidate lists with standard RRF.

    Args:
        result_lists: ranked lists, e.g. [dense_results, bm25_results]. Each
            candidate must carry a unique "id".
        k: RRF smoothing constant.
        limit: maximum number of fused candidates to return.
    Returns:
        Fused candidates sorted by _rrf_score (descending). Each candidate
        carries _rrf_score and _rrf_parts {list_name: contribution}.
    """
    if not result_lists:
        return []
    list_names = [f"list{i}" for i in range(len(result_lists))]
    # Prefer semantic names when the caller provides scored fields.
    field_names = []
    for results in result_lists:
        if results and "_similarity" in results[0] and "_bm25_score" not in results[0]:
            field_names.append("dense")
        elif results and "_bm25_score" in results[0] and "_similarity" not in results[0]:
            field_names.append("bm25")
        else:
            field_names.append(None)

    scores: Dict[str, float] = {}
    parts: Dict[str, Dict[str, float]] = {}
    candidates: Dict[str, Dict] = {}

    for results, name, field in zip(result_lists, list_names, field_names):
        leg_name = field or name
        for rank, candidate in enumerate(results, start=1):
            cid = candidate["id"]
            contribution = 1.0 / (k + rank)
            scores[cid] = scores.get(cid, 0.0) + contribution
            parts.setdefault(cid, {})[leg_name] = contribution
            if cid not in candidates:
                candidates[cid] = dict(candidate)
                candidates[cid]["_rrf_parts"] = {}
            # Keep the native score of whichever list supplied the candidate.
            if field:
                candidates[cid][f"_{field}_seen"] = True

    fused = []
    for cid, score in scores.items():
        candidate = candidates[cid]
        candidate["_rrf_score"] = score
        candidate["_rrf_parts"] = parts[cid]
        fused.append(candidate)

    fused.sort(key=lambda c: c["_rrf_score"], reverse=True)
    return fused[:limit]
