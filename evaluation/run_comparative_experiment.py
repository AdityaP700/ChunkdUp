# evaluation/run_comparative_experiment.py
"""
ChunkdUp Research Evaluation: Comparative Benchmark Across 6 Memory Strategies.

Strategies Evaluated:
1. Strategy A: Existing ChunkdUp pipeline (Control: existing extraction + existing hybrid CE retrieval)
2. Strategy B: Context-aware relational extraction + existing hybrid retrieval
3. Strategy C: Existing extraction + HippoRAG PPR graph retrieval
4. Strategy D: Full Relational Pipeline (Relational extraction + HippoRAG PPR retrieval)
5. Strategy E: GPT-Inspired 3-Tier Profile, Summary, and Archive Memory
6. Strategy F: Hybrid (Durable profile hot-path + relational PPR archive + abstention gating)

Dataset: LoCoMo Development Split (conv-26, conv-30: 96 turns, 304 questions)
Held-out split (conv-42 through conv-50) is strictly preserved and excluded.

Saves report to: evaluation/results/comparative_experiment_report.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

# Path setup
REPO_ROOT = Path(__file__).resolve().parent.parent
AI_SYSTEM_DIR = REPO_ROOT / "ai-system"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(AI_SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SYSTEM_DIR))

import numpy as np
import torch

from chunkdup.hybrid_memory import HybridMemory
from chunkdup.memory import Memory
from chunkdup.profile.profile_memory import ProfileMemory
from chunkdup.relational.extractor import RelationalExtractor
from chunkdup.relational.graph import RelationalGraph
from chunkdup.relational.schema import RelationalClaim
from chunkdup.retrieval_ppr import PPRHybridRetriever, RelationalPPRGraph
from chunkdup.retrieval_production import ProductionRetriever
from evaluation.adapters.locomo_adapter import ConversationSample, LoCoMoLoader, QuestionItem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("chunkdup.comparative_eval")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


import tempfile

logging.getLogger("chunkdup.memory").setLevel(logging.WARNING)
logging.getLogger("chunkdup.retrieval_production").setLevel(logging.WARNING)
logging.getLogger("chunkdup.policies").setLevel(logging.WARNING)


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score between prediction string and ground truth."""
    pred_tokens = re.findall(r"\w+", str(prediction).lower())
    gt_tokens = re.findall(r"\w+", str(ground_truth).lower())
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(pred_tokens)
    rec = num_same / len(gt_tokens)
    return (2.0 * prec * rec) / (prec + rec)


class ExperimentHarness:
    """Runs and profiles all 6 memory strategies on a conversation sample."""

    def __init__(self, sample: ConversationSample, max_turns: Optional[int] = None):
        self.sample = sample
        self.turns = sample.turns[:max_turns] if max_turns else sample.turns
        self.questions = sample.questions

    def setup_strategy_a(self) -> Tuple[Memory, ProductionRetriever]:
        """Strategy A: Control (Existing extraction + Existing Production hybrid CE retrieval)."""
        temp_dir = tempfile.mkdtemp()
        mem = Memory(store="memory", data_dir=temp_dir, extractor="rules")
        for t in self.turns:
            mem.remember(
                text=f"{t.speaker}: {t.text}",
                session_id=f"session_{t.session_num}",
                speaker=t.speaker,
                timestamp=t.date_time,
                turn_id=t.dia_id,
            )
        retriever = ProductionRetriever(
            repository=mem.repository,
            config_name="hybrid_ce_abstain",
            entity_graph=mem.entity_graph,
            query_log_path=None,
        )
        retriever._ensure_index()
        return mem, retriever

    def setup_strategy_b(self) -> Tuple[Memory, ProductionRetriever]:
        """Strategy B: Relational extraction + Existing hybrid retrieval."""
        temp_dir = tempfile.mkdtemp()
        mem = Memory(store="memory", data_dir=temp_dir, extractor="rules")
        rel_extractor = RelationalExtractor(mode="rules")

        turn_dicts = [
            {"turn_id": t.dia_id, "speaker": t.speaker, "text": t.text, "timestamp": t.date_time}
            for t in self.turns
        ]

        for idx, t in enumerate(turn_dicts):
            ctx_window = turn_dicts[max(0, idx - 3):idx]
            claims = rel_extractor.extract_from_window(t, context_window=ctx_window)
            for c in claims:
                mem.repository.save({
                    "id": c.claim_id,
                    "key": f"{c.subject_entity}_{c.predicate}",
                    "value": f"{c.subject_entity} {c.predicate} {c.object_entity}",
                    "category": "relation",
                    "source_turn_id": t["turn_id"],
                    "session_id": f"session_{self.turns[idx].session_num}",
                    "speaker": t["speaker"],
                    "timestamp": t["timestamp"],
                    "decision": "STORE",
                })

        retriever = ProductionRetriever(
            repository=mem.repository,
            config_name="hybrid_ce_abstain",
            entity_graph=mem.entity_graph,
            query_log_path=None,
        )
        retriever._ensure_index()
        return mem, retriever

    def setup_strategy_c(self) -> RelationalPPRGraph:
        """Strategy C: Existing extraction + HippoRAG PPR graph retrieval."""
        temp_dir = tempfile.mkdtemp()
        mem = Memory(store="memory", data_dir=temp_dir, extractor="rules")
        for t in self.turns:
            mem.remember(
                text=f"{t.speaker}: {t.text}",
                session_id=f"session_{t.session_num}",
                speaker=t.speaker,
                timestamp=t.date_time,
                turn_id=t.dia_id,
            )
        rel_graph = RelationalGraph()
        # Build claims from stored memories
        for m in mem.get_all():
            val = str(m.get("value", ""))
            k = str(m.get("key", ""))
            tid = str(m.get("source_turn_id", ""))
            rel_graph.add_claim(RelationalClaim(
                claim_id=str(m.get("id")),
                subject_entity=k.split("_")[0] if "_" in k else k,
                predicate="related_to",
                object_entity=val[:40],
                source_turn_ids=[tid] if tid else [],
                context_text=val,
            ))
        passages = [{"id": t.dia_id, "text": f"{t.speaker}: {t.text}"} for t in self.turns]
        ppr_graph = RelationalPPRGraph(relational_graph=rel_graph)
        ppr_graph.build_from_passages_and_claims(passages)
        return ppr_graph

    def setup_strategy_d(self) -> RelationalPPRGraph:
        """Strategy D: Full Relational Pipeline (Relational extraction + HippoRAG PPR retrieval)."""
        rel_extractor = RelationalExtractor(mode="rules")
        rel_graph = RelationalGraph()

        turn_dicts = [
            {"turn_id": t.dia_id, "speaker": t.speaker, "text": t.text, "timestamp": t.date_time}
            for t in self.turns
        ]

        for idx, t in enumerate(turn_dicts):
            ctx_window = turn_dicts[max(0, idx - 3):idx]
            claims = rel_extractor.extract_from_window(t, context_window=ctx_window)
            for c in claims:
                rel_graph.add_claim(c)

        passages = [{"id": t.dia_id, "text": f"{t.speaker}: {t.text}"} for t in self.turns]
        ppr_graph = RelationalPPRGraph(relational_graph=rel_graph)
        ppr_graph.build_from_passages_and_claims(passages)
        return ppr_graph

    def setup_strategy_e(self) -> ProfileMemory:
        """Strategy E: GPT-Inspired Profile, Summary, and Archive Memory."""
        prof_mem = ProfileMemory(consolidation_mode="rules")
        for t in self.turns:
            prof_mem.remember({
                "turn_id": t.dia_id,
                "speaker": t.speaker,
                "text": t.text,
                "timestamp": t.date_time,
            })
        return prof_mem

    def setup_strategy_f(self) -> HybridMemory:
        """Strategy F: Hybrid (Profile hot-path + relational PPR + abstention)."""
        hybrid = HybridMemory(extractor_mode="rules")
        for t in self.turns:
            hybrid.remember(
                text=t.text,
                speaker=t.speaker,
                timestamp=t.date_time,
                turn_id=t.dia_id,
            )
        return hybrid


def evaluate_strategy(
    strategy_id: str,
    strategy_obj: Any,
    questions: List[QuestionItem],
    max_questions: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluates a single strategy across questions and records comprehensive metrics."""
    eval_qs = questions[:max_questions] if max_questions else questions
    total_q = len(eval_qs)
    if total_q == 0:
        return {}

    hits_1 = 0
    hits_3 = 0
    hits_5 = 0
    rr_sum = 0.0

    multihop_total = 0
    multihop_hits = 0

    temporal_total = 0
    temporal_hits = 0

    adversarial_total = 0
    adversarial_abstained = 0

    warm_latencies: List[float] = []
    cold_latencies: List[float] = []
    prompt_token_counts: List[int] = []

    error_breakdown = {
        "EXTRACTION_FAILURE": 0,
        "RETRIEVAL_FAILURE": 0,
        "RANKING_FAILURE": 0,
        "FALSE_ABSTENTION": 0,
        "FALSE_RETRIEVAL": 0,
    }

    # Cold run warm-up
    t_cold_start = time.perf_counter()
    _warm_query(strategy_id, strategy_obj, "warmup query about user preference")
    cold_latencies.append((time.perf_counter() - t_cold_start) * 1000.0)

    for q in eval_qs:
        gt_turns = set(q.evidence_dia_ids)
        is_answerable = q.is_answerable
        cat = q.category

        t_start = time.perf_counter()
        retrieved_items, abstained, prompt_ctx = _execute_retrieval(strategy_id, strategy_obj, q.question)
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        warm_latencies.append(latency_ms)

        tokens = len(prompt_ctx.split())
        prompt_token_counts.append(tokens)

        # Map retrieved items to turn IDs
        retrieved_turn_ids: List[str] = []
        for item in retrieved_items:
            tid = item.get("source_turn_id") or item.get("id")
            if tid:
                retrieved_turn_ids.append(str(tid))

        # Check hits
        ranks = []
        for rank, item in enumerate(retrieved_items[:5]):
            tid = str(item.get("source_turn_id") or item.get("id") or "")
            cand_text = f"{item.get('text', '')} {item.get('value', '')} {item.get('contextual_text', '')}"
            f1 = compute_token_f1(cand_text, str(q.answer))
            if (tid and tid in gt_turns) or f1 >= 0.5:
                ranks.append(rank + 1)

        hit1 = bool(ranks and ranks[0] == 1)
        hit3 = bool(ranks and ranks[0] <= 3)
        hit5 = bool(ranks and ranks[0] <= 5)

        if hit1:
            hits_1 += 1
        if hit3:
            hits_3 += 1
        if hit5:
            hits_5 += 1

        recip_rank = (1.0 / ranks[0]) if ranks else 0.0
        rr_sum += recip_rank

        # Category breakdowns
        if cat == 3:  # Multi-hop
            multihop_total += 1
            if hit3:
                multihop_hits += 1

        if cat == 2:  # Temporal / invalidation
            temporal_total += 1
            if hit3:
                temporal_hits += 1

        if cat == 5:  # Adversarial / unanswerable
            adversarial_total += 1
            if abstained:
                adversarial_abstained += 1
            else:
                error_breakdown["FALSE_RETRIEVAL"] += 1
        else:
            # Failure categorization for answerable questions
            if not hit5:
                if abstained:
                    error_breakdown["FALSE_ABSTENTION"] += 1
                elif len(ranks) > 0:
                    error_breakdown["RANKING_FAILURE"] += 1
                else:
                    error_breakdown["RETRIEVAL_FAILURE"] += 1

    p50_lat = float(np.percentile(warm_latencies, 50)) if warm_latencies else 0.0
    p95_lat = float(np.percentile(warm_latencies, 95)) if warm_latencies else 0.0
    avg_tokens = float(np.mean(prompt_token_counts)) if prompt_token_counts else 0.0

    return {
        "strategy_id": strategy_id,
        "total_questions": total_q,
        "hit_at_1": hits_1 / total_q,
        "hit_at_3": hits_3 / total_q,
        "hit_at_5": hits_5 / total_q,
        "mrr": rr_sum / total_q,
        "multihop_accuracy": (multihop_hits / multihop_total) if multihop_total > 0 else 0.0,
        "multihop_queries_count": multihop_total,
        "temporal_invalidation_accuracy": (temporal_hits / temporal_total) if temporal_total > 0 else 0.0,
        "temporal_queries_count": temporal_total,
        "adversarial_abstention_accuracy": (adversarial_abstained / adversarial_total) if adversarial_total > 0 else 0.0,
        "adversarial_queries_count": adversarial_total,
        "latency_p50_ms": p50_lat,
        "latency_p95_ms": p95_lat,
        "cold_start_latency_ms": cold_latencies[0] if cold_latencies else 0.0,
        "avg_context_tokens": avg_tokens,
        "failure_breakdown": error_breakdown,
    }


def _warm_query(strategy_id: str, strategy_obj: Any, query: str):
    """Executes single query without recording metrics to warm up index / GPU."""
    try:
        _execute_retrieval(strategy_id, strategy_obj, query)
    except Exception:
        pass


def _execute_retrieval(
    strategy_id: str,
    strategy_obj: Any,
    query: str,
) -> Tuple[List[Dict[str, Any]], bool, str]:
    """Standardizes retrieval across the different strategy interfaces."""
    if strategy_id in ("Strategy_A", "Strategy_B"):
        mem, retriever = strategy_obj
        res = retriever.retrieve(query, k=5)
        abstained = len(res) == 0
        prompt_ctx = "\n".join([f"- {r.get('value', '')}" for r in res])
        return res, abstained, prompt_ctx

    elif strategy_id in ("Strategy_C", "Strategy_D"):
        ppr_graph: RelationalPPRGraph = strategy_obj
        res = ppr_graph.retrieve(query=query, top_k=5)
        abstained = len(res) == 0
        prompt_ctx = "\n".join([f"- {r.get('text', '')}" for r in res])
        return res, abstained, prompt_ctx

    elif strategy_id == "Strategy_E":
        prof_mem: ProfileMemory = strategy_obj
        res_dict = prof_mem.retrieve(query=query, top_k=5)
        if res_dict["source"] == "profile_hot_path":
            items = [res_dict["fact"]]
            abstained = False
        else:
            items = res_dict.get("passages", [])
            abstained = len(items) == 0
        return items, abstained, res_dict.get("context_injection", "")

    elif strategy_id == "Strategy_F":
        hybrid: HybridMemory = strategy_obj
        res_dict = hybrid.retrieve(query=query, top_k=5)
        items = res_dict.get("results", [])
        abstained = res_dict.get("abstained", False)
        return items, abstained, res_dict.get("prompt_context", "")

    return [], True, ""


def run_experiment(
    max_turns: Optional[int] = None,
    max_questions: Optional[int] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute complete comparative evaluation over the LoCoMo development split."""
    locomo_path = REPO_ROOT / "evaluation" / "data" / "locomo10.json"
    if not locomo_path.exists():
        raise FileNotFoundError(f"Dataset not found: {locomo_path}")

    samples, _ = LoCoMoLoader.load(locomo_path, split="dev")
    sample_ids = [s.sample_id for s in samples]
    logger.info(f"Loaded LoCoMo development split: {sample_ids}")

    strategies = [
        "Strategy_A",  # Control: Existing ChunkdUp pipeline
        "Strategy_B",  # Relational extraction + existing retrieval
        "Strategy_C",  # Existing extraction + PPR retrieval
        "Strategy_D",  # Relational extraction + PPR retrieval (full relational)
        "Strategy_E",  # Profile, summary, and archive memory (GPT-inspired)
        "Strategy_F",  # Hybrid (profile + relational PPR + abstention)
    ]

    all_results: Dict[str, List[Dict[str, Any]]] = {s: [] for s in strategies}

    for sample in samples:
        logger.info(f"--- Setting up strategies for {sample.sample_id} ({len(sample.turns)} turns, {len(sample.questions)} questions) ---")
        harness = ExperimentHarness(sample, max_turns=max_turns)

        strat_objs = {
            "Strategy_A": harness.setup_strategy_a(),
            "Strategy_B": harness.setup_strategy_b(),
            "Strategy_C": harness.setup_strategy_c(),
            "Strategy_D": harness.setup_strategy_d(),
            "Strategy_E": harness.setup_strategy_e(),
            "Strategy_F": harness.setup_strategy_f(),
        }

        for strat_id in strategies:
            logger.info(f"Evaluating {strat_id} on {sample.sample_id}...")
            strat_eval = evaluate_strategy(
                strategy_id=strat_id,
                strategy_obj=strat_objs[strat_id],
                questions=sample.questions,
                max_questions=max_questions,
            )
            strat_eval["sample_id"] = sample.sample_id
            all_results[strat_id].append(strat_eval)

    # Aggregate across dev split samples
    aggregated: Dict[str, Dict[str, Any]] = {}
    for strat_id in strategies:
        sample_metrics = all_results[strat_id]
        total_q = sum(m["total_questions"] for m in sample_metrics)
        agg_mrr = sum(m["mrr"] * m["total_questions"] for m in sample_metrics) / total_q if total_q > 0 else 0.0
        agg_hit1 = sum(m["hit_at_1"] * m["total_questions"] for m in sample_metrics) / total_q if total_q > 0 else 0.0
        agg_hit3 = sum(m["hit_at_3"] * m["total_questions"] for m in sample_metrics) / total_q if total_q > 0 else 0.0
        agg_hit5 = sum(m["hit_at_5"] * m["total_questions"] for m in sample_metrics) / total_q if total_q > 0 else 0.0

        mh_q = sum(m["multihop_queries_count"] for m in sample_metrics)
        agg_mh = sum(m["multihop_accuracy"] * m["multihop_queries_count"] for m in sample_metrics) / mh_q if mh_q > 0 else 0.0

        temp_q = sum(m["temporal_queries_count"] for m in sample_metrics)
        agg_temp = sum(m["temporal_invalidation_accuracy"] * m["temporal_queries_count"] for m in sample_metrics) / temp_q if temp_q > 0 else 0.0

        adv_q = sum(m["adversarial_queries_count"] for m in sample_metrics)
        agg_adv = sum(m["adversarial_abstention_accuracy"] * m["adversarial_queries_count"] for m in sample_metrics) / adv_q if adv_q > 0 else 0.0

        agg_p50 = float(np.mean([m["latency_p50_ms"] for m in sample_metrics]))
        agg_p95 = float(np.mean([m["latency_p95_ms"] for m in sample_metrics]))
        agg_tokens = float(np.mean([m["avg_context_tokens"] for m in sample_metrics]))

        combined_errors: Dict[str, int] = {}
        for m in sample_metrics:
            for ek, ev in m["failure_breakdown"].items():
                combined_errors[ek] = combined_errors.get(ek, 0) + ev

        aggregated[strat_id] = {
            "strategy_id": strat_id,
            "total_questions_evaluated": total_q,
            "hit_at_1": round(agg_hit1, 4),
            "hit_at_3": round(agg_hit3, 4),
            "hit_at_5": round(agg_hit5, 4),
            "mrr": round(agg_mrr, 4),
            "multihop_accuracy": round(agg_mh, 4),
            "temporal_invalidation_accuracy": round(agg_temp, 4),
            "adversarial_abstention_accuracy": round(agg_adv, 4),
            "latency_p50_ms": round(agg_p50, 2),
            "latency_p95_ms": round(agg_p95, 2),
            "avg_context_tokens": round(agg_tokens, 1),
            "failure_breakdown": combined_errors,
        }

    final_report = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dev_samples": sample_ids,
            "held_out_samples_excluded": ["conv-42", "conv-43", "conv-44", "conv-45", "conv-47", "conv-48", "conv-49", "conv-50"],
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
        "aggregated_results": aggregated,
        "per_sample_results": all_results,
    }

    target_file = output_path or (RESULTS_DIR / "comparative_experiment_report.json")
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    logger.info(f"Comparative report successfully generated at {target_file}")
    return final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ChunkdUp 6-Strategy Comparative Benchmark")
    parser.add_argument("--max-turns", type=int, default=None, help="Cap turns per conversation")
    parser.add_argument("--max-questions", type=int, default=None, help="Cap questions per conversation")
    args = parser.parse_args()

    report = run_experiment(max_turns=args.max_turns, max_questions=args.max_questions)
    print("\n" + "=" * 80)
    print("CHUNKDUP 6-STRATEGY COMPARATIVE BENCHMARK SUMMARY (DEV SPLIT: conv-26, conv-30)")
    print("=" * 80)
    print(f"{'Strategy':<14} | {'Hit@1':<7} | {'Hit@3':<7} | {'Hit@5':<7} | {'MRR':<7} | {'Multi-Hop':<9} | {'Temporal':<8} | {'p50 (ms)':<8}")
    print("-" * 80)
    for s_id, s_data in report["aggregated_results"].items():
        print(f"{s_id:<14} | {s_data['hit_at_1']:<7.4f} | {s_data['hit_at_3']:<7.4f} | {s_data['hit_at_5']:<7.4f} | {s_data['mrr']:<7.4f} | {s_data['multihop_accuracy']:<9.4f} | {s_data['temporal_invalidation_accuracy']:<8.4f} | {s_data['latency_p50_ms']:<8.2f}")
    print("=" * 80)
