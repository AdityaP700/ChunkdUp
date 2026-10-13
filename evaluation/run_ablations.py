# evaluation/run_ablations.py
"""
Unified Benchmark Runner & Ablation Harness for ChunkdUp.

Evaluates configurations across:
1. Frozen 92-query regression suite (MRR, P@5, abstention rate, latency).
2. LoCoMo Benchmark Track A on the dev split (conv-26, conv-30) / eval split:
   - Single-hop factual recall
   - Temporal reasoning recall
   - Multi-hop reasoning recall
   - Adversarial rejection rate

Configurations evaluated:
  - legacy_baseline: Lexical / dense baseline without calibrated gate or entity expansion.
  - dense_abstain_ed: Fast dense path with scale-aware abstention and entity demotion.
  - hybrid_ce_abstain: Full production pipeline with hybrid retrieval, cross-encoder,
                       scale-aware calibrated abstention tau(N), and bounded 1-hop expansion.

Writes results to evaluation/results/ablations_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Path setup
REPO_ROOT = Path(__file__).resolve().parent.parent
AI_SYSTEM_DIR = REPO_ROOT / "ai-system"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(AI_SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SYSTEM_DIR))

from evaluation.adapters.locomo_adapter import LoCoMoLoader, TrackAEvaluator
from experiments.retrieval.evaluate import aggregate, score_query
from experiments.retrieval.fixture import all_memories
from chunkdup.in_memory_repository import InMemoryRepository
from chunkdup.retrieval_production import ProductionRetriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("chunkdup.ablations")

RESULTS_DIR = Path(__file__).parent / "results"
EXPERIMENTS_DIR = REPO_ROOT / "experiments" / "retrieval"


def get_git_commit_hash() -> str:
    """Get current git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def load_frozen_rows() -> List[Dict[str, Any]]:
    """Load the 92 queries from the frozen regression suite."""
    base_file = EXPERIMENTS_DIR / "dataset.jsonl"
    r2_file = EXPERIMENTS_DIR / "dataset_round2.jsonl"
    rows: List[Dict[str, Any]] = []
    if base_file.exists():
        rows += [json.loads(line) for line in base_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if r2_file.exists():
        rows += [json.loads(line) for line in r2_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def build_frozen_repo() -> InMemoryRepository:
    """Construct InMemoryRepository seeded with the 12 frozen regression memories."""
    repo = InMemoryRepository()
    for mem in all_memories():
        if mem.get("status") == "active":
            repo.save(dict(mem))
        else:
            repo._memories[mem["id"]] = dict(mem)
            repo._key_index[mem["key"]] = mem["id"]
    return repo


def evaluate_frozen_suite_config(config_key: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate a specific retriever configuration on the frozen 92-query regression suite."""
    repo = build_frozen_repo()

    if config_key == "legacy_baseline":
        retriever = ProductionRetriever(repo, config_name="baseline_current", query_log_path=None)
        retriever.demote_enabled = False
        retriever.entity_expansion_enabled = False
    elif config_key == "dense_abstain_ed":
        retriever = ProductionRetriever(repo, config_name="dense_abstain_ed", query_log_path=None, fast_mode=True)
        retriever.demote_enabled = True
        retriever.entity_expansion_enabled = False
    elif config_key == "hybrid_ce_abstain":
        retriever = ProductionRetriever(repo, config_name="hybrid_ce_abstain", query_log_path=None)
        retriever.demote_enabled = True
        retriever.entity_expansion_enabled = True
    else:
        raise ValueError(f"Unknown config_key: {config_key}")

    # Warm-up phase: run warm-up queries to isolate steady-state query latency from cold-start weight loading
    cold_t0 = time.perf_counter()
    retriever.retrieve("warmup initialization query", k=5)
    retriever.retrieve("second warmup query", k=5)
    cold_start_latency_ms = (time.perf_counter() - cold_t0) * 1000

    per_query = []
    t_start = time.perf_counter()
    latencies = []

    for row in rows:
        q_t0 = time.perf_counter()
        results = retriever.retrieve(row["query"], k=5)
        latencies.append((time.perf_counter() - q_t0) * 1000)

        returned_ids = [r["id"] for r in results]
        sq = score_query(row, returned_ids)
        expected = set(row.get("relevant_memory_ids", []))
        sq.update({
            "n_relevant_returned": len([i for i in returned_ids if i in expected]),
            "n_irrelevant_returned": len([i for i in returned_ids if i not in expected]),
            "abstained": not returned_ids,
        })
        per_query.append(sq)

    total_duration = time.perf_counter() - t_start
    summary = aggregate(per_query, rows)

    sorted_lats = sorted(latencies)
    p50_latency_ms = round(sorted_lats[int(len(sorted_lats) * 0.50)], 2) if sorted_lats else 0.0
    p95_latency_ms = round(sorted_lats[int(len(sorted_lats) * 0.95)], 2) if sorted_lats else 0.0
    avg_latency_ms = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    return {
        "config_key": config_key,
        "n_queries": len(rows),
        "MRR": summary.get("MRR", 0.0),
        "P@5": summary.get("Precision@K", 0.0),
        "recall_at_5": summary.get("Recall@K", 0.0),
        "abstain_count": summary.get("abstain_count", 0),
        "total_duration_seconds": round(total_duration, 2),
        "cold_start_latency_ms": round(cold_start_latency_ms, 2),
        "warm_p50_latency_ms": p50_latency_ms,
        "warm_p95_latency_ms": p95_latency_ms,
        "warm_avg_latency_ms": avg_latency_ms,
        "avg_latency_ms": avg_latency_ms,
    }


def evaluate_locomo_track_a(
    data_path: Path,
    split: str = "dev",
    k_values: List[int] = [5],
) -> Dict[str, Any]:
    """Run LoCoMo Track A evaluation across ablation configurations."""
    samples, _ = LoCoMoLoader.load(data_path, split=split)

    configs_to_run = ["bm25", "dense_abstain_ed", "hybrid_ce_abstain"]
    locomo_results = {}

    for cfg in configs_to_run:
        logger.info(f"Running LoCoMo Track A for config: {cfg} (split={split}, samples={len(samples)})...")
        evaluator = TrackAEvaluator(
            config_name=cfg,
            k_values=tuple(k_values),
        )
        summary = evaluator.evaluate(samples=samples)
        overall = summary.get("overall_answerable", summary.get("overall", {}))
        cat_breakdown = summary.get("categories", summary.get("category_breakdown", {}))
        adv_overall = summary.get("overall_adversarial", {})
        adv_rejection = adv_overall.get(
            "correct_rejection_rate",
            cat_breakdown.get("Adversarial / Unanswerable", {}).get("correct_rejection_rate", 0.0)
        )

        locomo_results[cfg] = {
            "config_name": cfg,
            "MRR@5": overall.get("MRR@5", 0.0),
            "Recall@5": overall.get("Recall@5", 0.0),
            "Precision@5": overall.get("Precision@5", 0.0),
            "abstain_rate": overall.get("abstain_rate", 0.0),
            "duration_seconds": summary.get("total_duration_seconds", 0.0),
            "adversarial_rejection_rate": adv_rejection,
            "category_breakdown": cat_breakdown,
        }

    return locomo_results


def run_ablations(
    split: str = "dev",
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute end-to-end ablations and generate comparison scoreboard."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = output_path or (RESULTS_DIR / "ablations_report.json")

    commit_hash = get_git_commit_hash()
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    # 1. Frozen 92-query regression suite
    logger.info("Evaluating Frozen 92-Query Regression Suite across 3 configurations...")
    frozen_rows = load_frozen_rows()
    frozen_results = {}
    for cfg in ["legacy_baseline", "dense_abstain_ed", "hybrid_ce_abstain"]:
        frozen_results[cfg] = evaluate_frozen_suite_config(cfg, frozen_rows)

    # 2. LoCoMo Benchmark Track A
    locomo_path = Path(__file__).parent / "data" / "locomo10.json"
    locomo_results = {}
    if locomo_path.exists():
        logger.info(f"Evaluating LoCoMo Benchmark Track A (split={split})...")
        locomo_results = evaluate_locomo_track_a(locomo_path, split=split, k_values=[5])
    else:
        logger.warning(f"LoCoMo dataset not found at {locomo_path}, skipping LoCoMo ablation.")

    # 3. Construct Comparative Report
    report = {
        "timestamp": timestamp,
        "commit_hash": commit_hash,
        "frozen_92_query_suite": frozen_results,
        "locomo_track_a": {
            "split": split,
            "results": locomo_results,
        },
        "scoreboard": {
            "frozen_92": {
                "legacy_baseline": {
                    "MRR": frozen_results["legacy_baseline"]["MRR"],
                    "P@5": frozen_results["legacy_baseline"]["P@5"],
                    "warm_p50_ms": frozen_results["legacy_baseline"]["warm_p50_latency_ms"],
                    "warm_p95_ms": frozen_results["legacy_baseline"]["warm_p95_latency_ms"],
                    "avg_latency_ms": frozen_results["legacy_baseline"]["warm_avg_latency_ms"],
                },
                "dense_abstain_ed": {
                    "MRR": frozen_results["dense_abstain_ed"]["MRR"],
                    "P@5": frozen_results["dense_abstain_ed"]["P@5"],
                    "warm_p50_ms": frozen_results["dense_abstain_ed"]["warm_p50_latency_ms"],
                    "warm_p95_ms": frozen_results["dense_abstain_ed"]["warm_p95_latency_ms"],
                    "avg_latency_ms": frozen_results["dense_abstain_ed"]["warm_avg_latency_ms"],
                },
                "hybrid_ce_abstain": {
                    "MRR": frozen_results["hybrid_ce_abstain"]["MRR"],
                    "P@5": frozen_results["hybrid_ce_abstain"]["P@5"],
                    "warm_p50_ms": frozen_results["hybrid_ce_abstain"]["warm_p50_latency_ms"],
                    "warm_p95_ms": frozen_results["hybrid_ce_abstain"]["warm_p95_latency_ms"],
                    "avg_latency_ms": frozen_results["hybrid_ce_abstain"]["warm_avg_latency_ms"],
                },
            },
        },
    }

    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"Saved ablation report to {out_file}")

    # 4. Print Scoreboard Table
    print("\n" + "=" * 90)
    print("CHUNKDUP SYSTEM ABLATION SCOREBOARD")
    print("=" * 90)
    print(f"Commit: {commit_hash} | Timestamp: {timestamp}\n")
    print("1. FROZEN 92-QUERY SUITE (Fixed 12-Memory Regression Fixture)")
    print("-" * 90)
    print(f"{'Configuration':<20} | {'MRR':<8} | {'P@5':<8} | {'Warm p50':<10} | {'Warm p95':<10} | {'Cold Start':<10}")
    print("-" * 90)
    for cfg, r in frozen_results.items():
        print(f"{cfg:<20} | {r['MRR']:<8.4f} | {r['P@5']:<8.4f} | {r['warm_p50_latency_ms']:<7.1f} ms | {r['warm_p95_latency_ms']:<7.1f} ms | {r['cold_start_latency_ms']:<7.1f} ms")

    if locomo_results:
        print("\n2. LOCOMO BENCHMARK TRACK A (Split: " + split + ")")
        print("-" * 80)
        print(f"{'Configuration':<22} | {'MRR@5':<8} | {'Recall@5':<10} | {'Prec@5':<8} | {'Adv Rejection':<14}")
        print("-" * 80)
        for cfg, r in locomo_results.items():
            print(f"{cfg:<22} | {r['MRR@5']:<8.4f} | {r['Recall@5']:<10.4f} | {r['Precision@5']:<8.4f} | {r['adversarial_rejection_rate']:<14.4f}")
    print("=" * 80 + "\n")

    return report


def main():
    parser = argparse.ArgumentParser(description="Run ChunkdUp unified ablations")
    parser.add_argument("--split", choices=["dev", "eval", "all"], default="dev", help="LoCoMo split to evaluate")
    parser.add_argument("--out", type=Path, default=None, help="Output path for report JSON")
    args = parser.parse_args()

    run_ablations(split=args.split, output_path=args.out)


if __name__ == "__main__":
    main()
