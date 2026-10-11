# experiments/retrieval/audit_ce_jargon.py
"""
Round-2 risk 3: cross-encoder jargon audit.

The round-1 report logged one CE anomaly: the correct PostgreSQL memory scored
-8.06 for "What database does ChunkdUp use?" while irrelevant candidates scored
positive. This audit checks whether that is systemic.

Method:
  1. Select every query that names a specific technical term / product /
     project (Python, Go, Neovim, pgvector, PostgreSQL, Docker, AWS,
     Google DeepMind, all-MiniLM-L6-v2, ChunkdUp, Ubuntu, MiniLM). Generic
     English ("database", "editor", "cloud") is deliberately excluded.
  2. Run each query through the hybrid_ce pipeline; record the CE's raw score
     for every expected (correct) memory and for the top-scoring incorrect
     candidate.
  3. Miscalibration = the CE itself ranks an incorrect memory above a correct
     one (rank inversion), regardless of what the gate does afterwards.
     Also recorded: correct-memory CE score < 0 while an incorrect one is > 0
     (the strict round-1 anomaly pattern), and whether the gate filtered the
     correct memory out entirely.

Usage: python experiments/retrieval/audit_ce_jargon.py
Writes experiments/retrieval/results/ce_jargon_audit.json
"""
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

from run_experiment import ExperimentPipeline, load_calibration, load_dataset
from config import CONFIGS

RESULTS_DIR = Path(__file__).parent / "results"

# Case-sensitive word-boundary matching; "Go" must not match "Google".
TECH_TERMS = [
    "Python", "Go", "Neovim", "pgvector", "PostgreSQL", "Docker", "AWS",
    "Google DeepMind", "all-MiniLM-L6-v2", "ChunkdUp", "Ubuntu", "MiniLM",
]


def is_technical_query(query: str) -> bool:
    for term in TECH_TERMS:
        pattern = r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])"
        if re.search(pattern, query):
            return True
    return False


def main() -> int:
    calibration = load_calibration()
    pipeline = ExperimentPipeline(CONFIGS["hybrid_ce"], calibration)
    gated = ExperimentPipeline(CONFIGS["hybrid_ce_abstain"], calibration)
    gate_threshold = gated.gate.threshold

    rows = load_dataset("both", str(Path(__file__).parent / "dataset_round2.jsonl"))
    audit_rows = [r for r in rows if is_technical_query(r["query"]) and r["answerable"]]
    print(f"Auditing {len(audit_rows)} technical-term queries "
          f"(of {len(rows)} total; gate threshold={gate_threshold})")

    records = []
    miscalibrated = 0
    strict_negative = 0
    gate_filtered = 0

    for row in audit_rows:
        results, _timings, ranked = pipeline.retrieve(row["query"], k=5)
        expected = set(row["relevant_memory_ids"])

        ce_scored = [(c["id"], c["_rerank_score"]) for c in ranked if "_rerank_score" in c]
        if not ce_scored:
            continue
        incorrect = [(cid, s) for cid, s in ce_scored if cid not in expected]
        expected_scored = [(cid, s) for cid, s in ce_scored if cid in expected]

        top_incorrect = max(incorrect, key=lambda x: x[1]) if incorrect else None
        best_expected = max(expected_scored, key=lambda x: x[1]) if expected_scored else None

        # CE rank of the best expected memory among all CE-scored candidates
        ce_order = sorted(ce_scored, key=lambda x: x[1], reverse=True)
        ce_rank_of_expected = next(
            (i + 1 for i, (cid, _s) in enumerate(ce_order) if cid in expected), None
        )
        # Dense-only rank for comparison (what would happen without the CE)
        dense_results = pipeline.dense.search(row["query"], 20)
        dense_rank_of_expected = next(
            (i + 1 for i, c in enumerate(dense_results) if c["id"] in expected), None
        )

        miscal = (
            best_expected is not None
            and top_incorrect is not None
            and best_expected[1] < top_incorrect[1]
        )
        strict = (
            best_expected is not None
            and top_incorrect is not None
            and best_expected[1] < 0 < top_incorrect[1]
        )
        # Would the calibrated gate on the CE score admit the correct memory?
        gate_passes = best_expected is not None and best_expected[1] >= gate_threshold
        miscalibrated += miscal
        strict_negative += strict
        gate_filtered += not gate_passes

        records.append({
            "id": row["id"],
            "query": row["query"],
            "category": row["category"],
            "expected": sorted(expected),
            "best_expected_ce": round(best_expected[1], 4) if best_expected else None,
            "top_incorrect_id": top_incorrect[0] if top_incorrect else None,
            "top_incorrect_ce": round(top_incorrect[1], 4) if top_incorrect else None,
            "ce_rank_of_expected": ce_rank_of_expected,
            "dense_rank_of_expected": dense_rank_of_expected,
            "gate_filtered_correct_memory": not gate_passes,
            "miscalibrated": miscal,
            "strict_negative_pattern": strict,
        })

    n = len(records)
    summary = {
        "n_technical_queries": n,
        "ce_rank_inversion": miscalibrated,
        "ce_rank_inversion_pct": round(100 * miscalibrated / n, 1) if n else None,
        "strict_negative_pattern": strict_negative,
        "strict_negative_pct": round(100 * strict_negative / n, 1) if n else None,
        "gate_filtered_correct_memory": gate_filtered,
        "threshold": gate_threshold,
        "verdict_rule": "miscalibrated = CE ranks an incorrect memory above a correct one",
    }
    out = {"summary": summary, "records": records}
    out_path = RESULTS_DIR / "ce_jargon_audit.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("\nPer-query (miscalibrated only):")
    for r in records:
        if r["miscalibrated"]:
            print(f"  {r['id']} {r['query']!r}: expected CE={r['best_expected_ce']} vs "
                  f"top-incorrect {r['top_incorrect_id']} CE={r['top_incorrect_ce']} "
                  f"(CE rank {r['ce_rank_of_expected']}, dense rank {r['dense_rank_of_expected']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
