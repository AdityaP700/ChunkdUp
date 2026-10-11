# experiments/retrieval/analyze_fp_patterns.py
"""
Round-2 risks 4 + 5: false-abstention accounting and false-positive pattern
analysis, computed from the committed per-query result files (no re-runs).

Risk 4 - false-abstention rate on answerable queries, reported NEXT TO the
         correct-rejection rate on unanswerable queries, per config.
Risk 5 - every false-positive result (irrelevant memory returned in top-5)
         from the leading configs, grouped by query category, with named
         examples, to decide whether a concentrated failure pattern justifies
         a 6th experiment.

Writes experiments/retrieval/results/fp_abstention_analysis.json
"""
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent))

RESULTS = Path(__file__).parent / "results"

CONFIGS_ROUND1 = [
    "baseline_current", "dense", "bm25", "hybrid_rrf", "dense_ce", "hybrid_ce",
    "dense_abstain", "hybrid_abstain", "hybrid_ce_abstain",
]
# The heldout expansion lives in the round-2 tagged files; risk 4/5 use both
# the round-1 heldout (14) and, where present, the expanded round-2 files.
ROUND2_HELDOUT = {
    "baseline_current", "dense_abstain", "hybrid_ce_abstain", "bm25",
}


def load(split, name):
    if split == "heldout_r2":
        return json.loads((RESULTS / f"round2.{name}.heldout.json").read_text(encoding="utf-8"))
    return json.loads((RESULTS / f"{name}.{split}.json").read_text(encoding="utf-8"))


def available_splits(name):
    splits = ["dev", "heldout"]
    if name in ROUND2_HELDOUT:
        splits.append("heldout_r2")
    return splits


def main() -> int:
    analysis: Dict = {"risk4_false_abstention": {}, "risk5_fp_patterns": {}}

    # Authoritative expected-answer map (per-query rows are identical across configs)
    expected_by_id = {}
    for name in ("dense", "baseline_current"):
        for split in available_splits(name):
            for r in load(split, name)["per_query"]:
                expected_by_id[r["id"]] = r["expected"]
    n_answerable = sum(1 for v in expected_by_id.values() if v)
    n_unanswerable = sum(1 for v in expected_by_id.values() if not v)

    # ---- Risk 4: false abstention vs correct rejection, per config ----
    for name in CONFIGS_ROUND1:
        entry = {}
        for split in available_splits(name):
            per_q = load(split, name)["per_query"]
            ans = [r for r in per_q if expected_by_id[r["id"]]]
            unans = [r for r in per_q if not expected_by_id[r["id"]]]
            entry[split] = {
                "n_answerable": len(ans),
                "false_abstentions": sum(1 for r in ans if r["abstained"]),
                "false_abstention_rate": round(sum(1 for r in ans if r["abstained"]) / len(ans), 4) if ans else None,
                "false_abstention_query_ids": [r["id"] for r in ans if r["abstained"]],
                "n_unanswerable": len(unans),
                "correct_rejection_rate": round(sum(1 for r in unans if r["abstained"]) / len(unans), 4) if unans else None,
                "irrelevant_memories_returned": sum(
                    len([mid for mid in r["returned"] if mid not in set(expected_by_id[r["id"]])])
                    for r in unans
                ),
            }
        analysis["risk4_false_abstention"][name] = entry

    # ---- Risk 5: FP pattern analysis for the leading configs ----
    for name in ("dense_abstain", "hybrid_ce_abstain", "bm25"):
        cat_fp_queries = Counter()
        cat_fp_memories = Counter()
        total_fp_queries = 0
        examples = []
        for split in available_splits(name):
            for r in load(split, name)["per_query"]:
                expected = set(expected_by_id[r["id"]])
                if not r["expected"]:
                    continue  # risk 5 covers answerable queries only
                fp_ids = [mid for mid in r["returned"] if mid not in expected]
                if fp_ids:
                    total_fp_queries += 1
                    cat_fp_queries[r["category"]] += 1
                    cat_fp_memories[r["category"]] += len(fp_ids)
                    examples.append({
                        "id": r["id"], "category": r["category"], "query": r["query"],
                        "expected": sorted(expected), "irrelevant_returned": fp_ids,
                        "returned": r["returned"],
                    })
        analysis["risk5_fp_patterns"][name] = {
            "n_answerable_total": n_answerable,
            "total_answerable_queries_with_fp": total_fp_queries,
            "fp_queries_by_category": dict(cat_fp_queries),
            "fp_memories_by_category": dict(cat_fp_memories),
            "examples": examples,
        }

    out_path = RESULTS / "fp_abstention_analysis.json"
    out_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"saved {out_path}  (answerable={n_answerable}, unanswerable={n_unanswerable} across splits used)")

    print("\n=== Risk 4: false abstention (answerable) vs correct rejection (unanswerable) ===")
    for name in CONFIGS_ROUND1:
        parts = []
        for split, label in (("dev", "dev"), ("heldout", "ho"), ("heldout_r2", "ho36")):
            e = analysis["risk4_false_abstention"][name].get(split)
            if e:
                parts.append(f"{label}: abstain={e['false_abstention_rate']} rej={e['correct_rejection_rate']}")
        print(f"{name:<20} " + "  ".join(parts))

    print("\n=== Risk 5: FP queries by category (answerable queries only) ===")
    for name in ("dense_abstain", "hybrid_ce_abstain", "bm25"):
        p = analysis["risk5_fp_patterns"][name]
        print(f"{name}: {p['total_answerable_queries_with_fp']}/{p['n_answerable_total']} answerable queries returned >=1 irrelevant memory")
        print(f"   fp queries by category: {p['fp_queries_by_category']}")
        print(f"   fp memories by category: {p['fp_memories_by_category']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
