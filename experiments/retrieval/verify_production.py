# experiments/retrieval/verify_production.py
"""
Closeout Part 1: verify that the PRODUCTION module (retrieval_production.py,
the code the SDK actually calls) reproduces the round-3 experiment-pipeline
numbers on the full 92-query combined set.

Two runs:
  1. default config WITH entity demotion  -> expect pooled MRR 0.8424, P@5 0.6475
     (round-3 hybrid_ce_abstain_ed experiment numbers)
  2. demotion disabled                    -> expect pooled MRR 0.8397, P@5 0.6366
     (round-3 hybrid_ce_abstain before numbers)

A mismatch means the experiment logic did not transfer into the production
module; the script reports the exact deltas instead of passing silently.

Usage: python experiments/retrieval/verify_production.py
Writes experiments/retrieval/results/production_verification.json
"""
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

from ai_system_path import ai_system_path  # noqa: F401  (sys.path setup)
from evaluate import aggregate, score_query
from fixture import all_memories

RESULTS_DIR = Path(__file__).parent / "results"

EXPECTED = {
    "with_demotion": {"MRR": 0.8424, "Precision@K": 0.6475},
    "without_demotion": {"MRR": 0.8397, "Precision@K": 0.6366},
}


def load_rows() -> List[Dict]:
    base = Path(__file__).parent / "dataset.jsonl"
    r2 = Path(__file__).parent / "dataset_round2.jsonl"
    rows = [json.loads(l) for l in base.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows += [json.loads(l) for l in r2.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows


def build_repo():
    from chunkdup.in_memory_repository import InMemoryRepository

    repo = InMemoryRepository()
    for memory in all_memories():
        if memory.get("status") == "active":
            repo.save(dict(memory))
        else:
            # InMemoryRepository.save() forces status="active"; historical
            # (inactive) memories must be stored as the repository would hold
            # them, invisible to get_all(), to mirror the experiment fixture.
            repo._memories[memory["id"]] = dict(memory)
            repo._key_index[memory["key"]] = memory["id"]
    return repo


def run_production(rows: List[Dict], demote: bool) -> Dict:
    from chunkdup.retrieval_production import ProductionRetriever

    retriever = ProductionRetriever(build_repo(), config_name="hybrid_ce_abstain", query_log_path=None)
    retriever.demote_enabled = demote

    per_query = []
    for row in rows:
        results = retriever.retrieve(row["query"], k=5)
        returned_ids = [r["id"] for r in results]
        sq = score_query(row, returned_ids)
        expected = set(row["relevant_memory_ids"])
        sq.update({
            "n_relevant_returned": len([i for i in returned_ids if i in expected]),
            "n_irrelevant_returned": len([i for i in returned_ids if i not in expected]),
            "abstained": not returned_ids,
        })
        per_query.append(sq)
    return aggregate(per_query, rows)


def main() -> int:
    rows = load_rows()
    print(f"92-query combined set: {len(rows)} rows")

    outcomes = {}
    all_match = True
    for variant, demote in (("with_demotion", True), ("without_demotion", False)):
        summary = run_production(rows, demote)
        expected = EXPECTED[variant]
        mrr_ok = abs(summary["MRR"] - expected["MRR"]) < 5e-4
        p_ok = abs(summary["Precision@K"] - expected["Precision@K"]) < 5e-4
        match = mrr_ok and p_ok
        all_match = all_match and match
        outcomes[variant] = {
            "expected": expected,
            "actual": {"MRR": summary["MRR"], "Precision@K": summary["Precision@K"]},
            "match": match,
        }
        print(f"  {variant:<18} MRR {summary['MRR']} (expected {expected['MRR']})  "
              f"P@5 {summary['Precision@K']} (expected {expected['Precision@K']})  "
              f"-> {'MATCH' if match else 'MISMATCH'}")

    out = {"all_match": all_match, "variants": outcomes}
    out_path = RESULTS_DIR / "production_verification.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved {out_path}")
    print("PRODUCTION VERIFICATION:", "PASS" if all_match else "FAIL")
    return 0 if all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
