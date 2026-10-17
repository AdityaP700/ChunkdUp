# evaluation/run_associative_sprint.py
"""
LoCoMo Associative Memory Research Benchmark Runner (Phases 1, 4, 5).
Evaluates the 7 retrieval variants and controlled ablations across all 5 LoCoMo question categories:
- Cat 1: Single-hop / Factual
- Cat 2: Temporal reasoning
- Cat 3: Multi-hop reasoning
- Cat 4: Open-domain / Inference
- Cat 5: Adversarial / Unanswerable

Metrics:
- Hit@1, Hit@3, Hit@5, Hit@8
- MRR
- Evidence Recall@k
- Multi-hop full recall
- Temporal accuracy
- Adversarial rejection & false positive rates
- Steady-state latency (p50, p95 ms)
- Context token consumption
- Failure taxonomy (Categories A through J)
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
AI_SYSTEM_DIR = REPO_ROOT / "ai-system"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(AI_SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SYSTEM_DIR))

import numpy as np
import torch

from chunkdup.associative.builder import AssociativeMemoryBuilder
from chunkdup.associative.graph import BoundedGraphExpansion
from chunkdup.associative.retrievers import (
    BM25Retriever,
    BoundedAssociativeRetriever,
    ConsolidatedAssociativeRetriever,
    ConsolidatedSemanticRetriever,
    GrepRetriever,
    HybridCEControlRetriever,
    QueryRoutedAssociativeRetriever,
    RetrievalResult,
)
from evaluation.adapters.locomo_adapter import (
    CATEGORY_NAMES,
    ConversationSample,
    LoCoMoLoader,
    QuestionItem,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("chunkdup.associative_sprint")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def count_tokens(text: str) -> int:
    return len(re.findall(r"\w+", text))


def classify_failure(
    q: QuestionItem,
    retrieved_dia_ids: List[str],
    retriever_name: str,
    abstained: bool,
    top_score: float,
) -> Optional[str]:
    """Assign error to Categories A through J."""
    gold = set(q.evidence_dia_ids)

    if not q.is_answerable:
        if not abstained and len(retrieved_dia_ids) > 0:
            return "J: Incorrect abstention (hallucinated on unanswerable)"
        return None  # Correct rejection

    if abstained or len(retrieved_dia_ids) == 0:
        return "J: Incorrect abstention (false refusal)"

    hits = [d for d in retrieved_dia_ids if d in gold]

    if not hits:
        # Zero evidence retrieved
        if q.category == 3:
            return "F: Missing multi-hop evidence"
        elif q.category == 2:
            return "G: Temporal conflict / stale state"
        else:
            return "C: Semantic retrieval miss"

    elif len(hits) < len(gold):
        # Partial retrieval on multi-hop or multi-evidence
        if q.category == 3:
            return "F: Missing multi-hop evidence (partial recall only)"
        elif q.category == 2:
            return "G: Temporal conflict (partial/older turn retrieved)"
        else:
            return "I: Ranking failure (remaining evidence displaced)"

    return None  # Full success


class AssociativeBenchmarkRunner:
    """Executes comparative evaluation across all 7 variants and ablations."""

    def __init__(self, device: str = "cuda", run_ablations: bool = True):
        self.device = device if (torch.cuda.is_available() and device == "cuda") else "cpu"
        self.run_ablations = run_ablations
        self.builder = AssociativeMemoryBuilder(extractor_mode="rules")

    def evaluate_sample(
        self, sample: ConversationSample
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        logger.info(
            "Building representation bundle for %s (%d turns, %d questions)...",
            sample.sample_id,
            len(sample.turns),
            len(sample.questions),
        )
        bundle, graph = self.builder.build_bundle(sample)

        # 1. Instantiate the 7 variants
        var_a = GrepRetriever(turns=bundle.experience_turns)
        var_b = BM25Retriever(turns=bundle.experience_turns)
        var_c = HybridCEControlRetriever(
            turns=bundle.experience_turns, sample_id=sample.sample_id, device=self.device
        )
        var_d = BoundedAssociativeRetriever(
            turns=bundle.experience_turns,
            graph=graph,
            sample_id=sample.sample_id,
            device=self.device,
            max_depth=2,
            max_branching=3,
            max_expanded_candidates=5,
        )
        var_e = ConsolidatedSemanticRetriever(
            turns=bundle.experience_turns,
            facts=bundle.semantic_facts,
            sample_id=sample.sample_id,
            device=self.device,
        )
        var_f = ConsolidatedAssociativeRetriever(
            turns=bundle.experience_turns,
            facts=bundle.semantic_facts,
            graph=graph,
            sample_id=sample.sample_id,
            device=self.device,
        )
        var_g = QueryRoutedAssociativeRetriever(var_a, var_b, var_c, var_d, var_e, var_f)

        strategies: Dict[str, Any] = {
            "Variant_A_Grep": var_a,
            "Variant_B_BM25": var_b,
            "Variant_C_HybridCE": var_c,
            "Variant_D_BoundedAssociative": var_d,
            "Variant_E_ConsolidatedSemantic": var_e,
            "Variant_F_ConsolidatedAssociative": var_f,
            "Variant_G_QueryRouted": var_g,
        }

        # 2. Add ablations if requested
        if self.run_ablations:
            strategies["Ablation_D_Depth1"] = BoundedAssociativeRetriever(
                turns=bundle.experience_turns,
                graph=graph,
                sample_id=sample.sample_id,
                device=self.device,
                max_depth=1,
                max_branching=3,
                max_expanded_candidates=5,
            )
            strategies["Ablation_D_Branch1"] = BoundedAssociativeRetriever(
                turns=bundle.experience_turns,
                graph=graph,
                sample_id=sample.sample_id,
                device=self.device,
                max_depth=2,
                max_branching=1,
                max_expanded_candidates=5,
            )

        # Warmup GPU
        var_c.retrieve("warmup query", k=1)

        per_strategy_records: Dict[str, List[Dict[str, Any]]] = {s: [] for s in strategies}
        per_question_rows: List[Dict[str, Any]] = []

        for q_idx, q in enumerate(sample.questions):
            gold_dia_ids = set(q.evidence_dia_ids)

            for s_name, strat in strategies.items():
                t0 = time.perf_counter()
                if s_name == "Variant_G_QueryRouted":
                    hits, route_mode = strat.retrieve(q.question, k=5)
                else:
                    hits = strat.retrieve(q.question, k=5)
                    route_mode = "fixed"
                latency_ms = (time.perf_counter() - t0) * 1000.0

                retrieved_dia_ids = [h.dia_id for h in hits]
                abstained = len(hits) == 0
                prompt_ctx = "\n".join([f"- {h.text}" for h in hits])
                tokens = count_tokens(prompt_ctx)

                rec: Dict[str, Any] = {
                    "sample_id": sample.sample_id,
                    "question_idx": q.question_idx,
                    "category": q.category,
                    "category_name": q.category_name,
                    "is_answerable": q.is_answerable,
                    "strategy": s_name,
                    "latency_ms": latency_ms,
                    "tokens": tokens,
                    "abstained": abstained,
                    "route_mode": route_mode,
                    "retrieved_dia_ids": retrieved_dia_ids,
                    "gold_dia_ids": list(gold_dia_ids),
                }

                if q.is_answerable and gold_dia_ids:
                    hits_in_gold = [did for did in retrieved_dia_ids if did in gold_dia_ids]
                    unique_hits = set(hits_in_gold)
                    rec["hit@1"] = 1.0 if retrieved_dia_ids and retrieved_dia_ids[0] in gold_dia_ids else 0.0
                    rec["hit@3"] = 1.0 if any(did in gold_dia_ids for did in retrieved_dia_ids[:3]) else 0.0
                    rec["hit@5"] = 1.0 if unique_hits else 0.0
                    rec["recall@5"] = len(unique_hits) / len(gold_dia_ids)
                    rec["full_recall@5"] = 1.0 if gold_dia_ids.issubset(unique_hits) else 0.0
                    rec["precision@5"] = len(hits_in_gold) / len(retrieved_dia_ids) if retrieved_dia_ids else 0.0

                    if hits_in_gold:
                        first_rank = min(retrieved_dia_ids.index(did) for did in hits_in_gold)
                        rec["mrr@5"] = 1.0 / (first_rank + 1)
                    else:
                        rec["mrr@5"] = 0.0
                elif not q.is_answerable:
                    rec["correct_rejection"] = 1.0 if abstained else 0.0
                    rec["false_positive"] = 0.0 if abstained else 1.0

                # Taxonomy
                top_sc = hits[0].score if hits else 0.0
                fail_cat = classify_failure(q, retrieved_dia_ids, s_name, abstained, top_sc)
                rec["failure_category"] = fail_cat

                per_strategy_records[s_name].append(rec)
                per_question_rows.append(rec)

        return per_strategy_records, per_question_rows


def aggregate_strategy_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    def mean(vals: List[float]) -> float:
        return float(np.mean(vals)) if vals else 0.0

    ans = [r for r in records if r["is_answerable"]]
    adv = [r for r in records if not r["is_answerable"]]

    latencies = [r["latency_ms"] for r in records]
    p50_lat = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0

    summary: Dict[str, Any] = {
        "n_total": len(records),
        "n_answerable": len(ans),
        "n_adversarial": len(adv),
        "hit_at_1": round(mean([r.get("hit@1", 0.0) for r in ans]), 4),
        "hit_at_3": round(mean([r.get("hit@3", 0.0) for r in ans]), 4),
        "hit_at_5": round(mean([r.get("hit@5", 0.0) for r in ans]), 4),
        "mrr_at_5": round(mean([r.get("mrr@5", 0.0) for r in ans]), 4),
        "recall_at_5": round(mean([r.get("recall@5", 0.0) for r in ans]), 4),
        "full_recall_at_5": round(mean([r.get("full_recall@5", 0.0) for r in ans]), 4),
        "precision_at_5": round(mean([r.get("precision@5", 0.0) for r in ans]), 4),
        "mean_tokens": round(mean([r.get("tokens", 0) for r in records]), 1),
        "latency_p50_ms": round(p50_lat, 2),
        "latency_p95_ms": round(p95_lat, 2),
        "adversarial_rejection_rate": round(
            mean([r.get("correct_rejection", 0.0) for r in adv]), 4
        ) if adv else 0.0,
    }

    # Categories breakdown
    cat_breakdown: Dict[str, Any] = {}
    for cat in sorted(CATEGORY_NAMES.keys()):
        cat_name = CATEGORY_NAMES[cat]
        cat_recs = [r for r in records if r["category"] == cat]
        if not cat_recs:
            continue
        if cat == 5:
            cat_breakdown[cat_name] = {
                "n": len(cat_recs),
                "rejection_rate": round(mean([r.get("correct_rejection", 0.0) for r in cat_recs]), 4),
                "fp_rate": round(mean([r.get("false_positive", 0.0) for r in cat_recs]), 4),
            }
        else:
            cat_breakdown[cat_name] = {
                "n": len(cat_recs),
                "hit_at_5": round(mean([r.get("hit@5", 0.0) for r in cat_recs]), 4),
                "recall_at_5": round(mean([r.get("recall@5", 0.0) for r in cat_recs]), 4),
                "full_recall_at_5": round(mean([r.get("full_recall@5", 0.0) for r in cat_recs]), 4),
                "mrr_at_5": round(mean([r.get("mrr@5", 0.0) for r in cat_recs]), 4),
            }
    summary["category_breakdown"] = cat_breakdown

    # Failure Taxonomy Breakdown
    fails = [r["failure_category"] for r in records if r.get("failure_category")]
    fail_counts = Counter(fails)
    summary["failure_taxonomy"] = {k: fail_counts[k] for k in sorted(fail_counts.keys())}

    return summary


def run_benchmark(split: str = "dev", device: str = "cuda", run_ablations: bool = True):
    data_path = REPO_ROOT / "evaluation" / "data" / "locomo10.json"
    logger.info("Loading LoCoMo dataset split=%s...", split)
    samples, val_report = LoCoMoLoader.load(str(data_path), split=split)
    logger.info("Loaded %d conversations, %d questions.", len(samples), val_report["total_questions"])

    runner = AssociativeBenchmarkRunner(device=device, run_ablations=run_ablations)

    all_strat_recs: Dict[str, List[Dict[str, Any]]] = {}
    all_question_rows: List[Dict[str, Any]] = []

    for sample in samples:
        strat_recs, q_rows = runner.evaluate_sample(sample)
        all_question_rows.extend(q_rows)
        for s_name, recs in strat_recs.items():
            if s_name not in all_strat_recs:
                all_strat_recs[s_name] = []
            all_strat_recs[s_name].extend(recs)

    # Summarize metrics
    summary_report: Dict[str, Any] = {
        "benchmark": "LoCoMo",
        "split": split,
        "n_samples": len(samples),
        "total_questions": len(all_question_rows) // len(all_strat_recs),
        "device": device,
        "strategies": {},
    }

    for s_name, recs in all_strat_recs.items():
        summary_report["strategies"][s_name] = aggregate_strategy_metrics(recs)

    # Save JSON report
    out_json = RESULTS_DIR / f"associative_sprint_{split}.json"
    out_json.write_text(json.dumps(summary_report, indent=2), encoding="utf-8")
    logger.info("Saved summary report to %s", out_json)

    # Save CSV per-question report
    out_csv = RESULTS_DIR / f"associative_per_question_{split}.csv"
    if all_question_rows:
        fieldnames = [
            "sample_id",
            "question_idx",
            "category",
            "category_name",
            "is_answerable",
            "strategy",
            "hit@1",
            "hit@3",
            "hit@5",
            "mrr@5",
            "recall@5",
            "full_recall@5",
            "precision@5",
            "tokens",
            "latency_ms",
            "abstained",
            "route_mode",
            "failure_category",
        ]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_question_rows)
        logger.info("Saved per-question CSV to %s", out_csv)

    # Print Comparative Scoreboard
    print("\n" + "=" * 115)
    print(f"  CHUNKDUP ASSOCIATIVE MEMORY RESEARCH SPRINT SCOREBOARD (Split: {split.upper()})")
    print("=" * 115)
    print(
        f"{'Strategy':<34} | {'Hit@5':<8} | {'MRR@5':<8} | {'FullR@5':<8} | {'Multi-Hop':<9} | {'Adv Rej':<8} | {'p50(ms)':<8} | {'Tokens':<7}"
    )
    print("-" * 115)

    for s_name, s_data in summary_report["strategies"].items():
        mh_hit = s_data["category_breakdown"].get("Multi-hop reasoning", {}).get("hit_at_5", 0.0) * 100
        print(
            f"{s_name:<34} | "
            f"{s_data['hit_at_5']*100:>6.2f}% | "
            f"{s_data['mrr_at_5']:>7.4f} | "
            f"{s_data['full_recall_at_5']*100:>6.2f}% | "
            f"{mh_hit:>7.2f}% | "
            f"{s_data['adversarial_rejection_rate']*100:>6.2f}% | "
            f"{s_data['latency_p50_ms']:>6.1f}ms | "
            f"{s_data['mean_tokens']:>6.1f}"
        )
    print("=" * 115 + "\n")
    return summary_report


def main():
    parser = argparse.ArgumentParser(description="LoCoMo Associative Memory Sprint Benchmark")
    parser.add_argument("--split", choices=["dev", "eval", "all"], default="dev", help="Dataset split")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Inference device")
    parser.add_argument("--no-ablations", action="store_true", help="Skip graph expansion ablations")
    args = parser.parse_args()

    run_benchmark(split=args.split, device=args.device, run_ablations=not args.no_ablations)


if __name__ == "__main__":
    main()
