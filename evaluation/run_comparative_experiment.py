# evaluation/run_comparative_experiment.py
"""
ChunkdUp Research Evaluation: Full-Scale Comparative Benchmark Across 6 Memory Strategies.

Strategies Evaluated:
1. Strategy A: Existing ChunkdUp pipeline (Control: dialogue turns + rules extraction + hybrid CE retrieval)
2. Strategy B: Context-aware relational extraction + existing hybrid retrieval
3. Strategy C: Existing extraction + HippoRAG PPR graph retrieval
4. Strategy D: Full Relational Pipeline (Relational extraction + HippoRAG PPR retrieval)
5. Strategy E: GPT-Inspired 3-Tier Profile, Summary, and Archive Memory
6. Strategy F: Hybrid (Durable profile hot-path + relational PPR archive + abstention gating)

Dataset: LoCoMo Development Split (conv-26, conv-30: 788 turns, 304 questions)
Held-out split (conv-42 through conv-50) is strictly preserved and excluded.

Saves reports to:
- evaluation/results/comparative_experiment_report.json
- evaluation/results/full_locomo_dev_evaluation.json
- evaluation/results/full_locomo_dev_evaluation.csv
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

# Path setup
REPO_ROOT = Path(__file__).resolve().parent.parent
AI_SYSTEM_DIR = REPO_ROOT / "ai-system"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(AI_SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SYSTEM_DIR))

import numpy as np
import psutil
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

logging.getLogger("chunkdup.memory").setLevel(logging.WARNING)
logging.getLogger("chunkdup.retrieval_production").setLevel(logging.WARNING)
logging.getLogger("chunkdup.policies").setLevel(logging.WARNING)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CATEGORY_NAMES = {
    1: "Single-hop / Factual",
    2: "Temporal reasoning",
    3: "Multi-hop reasoning",
    4: "Open-domain / Inference",
    5: "Adversarial / Unanswerable",
}


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


def extract_item_turn_ids(item: Dict[str, Any]) -> List[str]:
    """Extract canonical dialogue turn IDs (e.g. 'D1:2') from a retrieved item."""
    turn_ids: List[str] = []
    candidates: List[Any] = [
        item.get("dia_id"),
        item.get("source_turn_id"),
        item.get("turn_id"),
        item.get("id"),
    ]
    if isinstance(item.get("source_turn_ids"), list):
        candidates.extend(item.get("source_turn_ids"))
    if isinstance(item.get("_source_turns"), (list, set, tuple)):
        candidates.extend(item.get("_source_turns"))

    for c in candidates:
        if not c:
            continue
        c_str = str(c).strip()
        # Handle prefixes like "conv-26:D1:2" -> "D1:2"
        if ":" in c_str and ("conv-" in c_str or "sample_" in c_str):
            parts = c_str.split(":", 1)
            c_str = parts[1]
        # Match D<session>:<turn> format
        match = re.search(r"\b(D\d+:\d+)\b", c_str)
        if match:
            tid = match.group(1)
            if tid not in turn_ids:
                turn_ids.append(tid)
        elif c_str.startswith("D") and ":" in c_str and c_str not in turn_ids:
            turn_ids.append(c_str)

    # Check contextual_text or value
    text_content = f"{item.get('contextual_text', '')} {item.get('source', '')} {item.get('value', '')}"
    for match in re.finditer(r"\b(D\d+:\d+)\b", text_content):
        tid = match.group(1)
        if tid not in turn_ids:
            turn_ids.append(tid)

    return turn_ids


class ExperimentHarness:
    """Runs and profiles memory strategies on a conversation sample."""

    def __init__(self, sample: ConversationSample, max_turns: Optional[int] = None):
        self.sample = sample
        self.turns = sample.turns[:max_turns] if max_turns else sample.turns
        self.questions = sample.questions

    def setup_strategy_a(self) -> Tuple[Memory, ProductionRetriever]:
        """Strategy A: Control (Existing ChunkdUp pipeline: turns indexed + rules extraction + hybrid CE retrieval)."""
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
            # Index dialogue turns into repository so ProductionRetriever has conversation evidence
            mem.repository.save({
                "id": f"{self.sample.sample_id}:{t.dia_id}",
                "dia_id": t.dia_id,
                "source_turn_id": t.dia_id,
                "key": f"turn_{t.dia_id}",
                "value": f"{t.speaker}: {t.text}",
                "contextual_text": f"In session {t.session_num} ({t.date_time}), {t.speaker} said: \"{t.text}\"",
                "type": "dialogue_turn",
                "speaker": t.speaker,
                "session": t.session_num,
                "timestamp": t.date_time,
                "status": "active",
            })
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

        # Index dialogue turns
        for t in self.turns:
            mem.repository.save({
                "id": f"{self.sample.sample_id}:{t.dia_id}",
                "dia_id": t.dia_id,
                "source_turn_id": t.dia_id,
                "key": f"turn_{t.dia_id}",
                "value": f"{t.speaker}: {t.text}",
                "contextual_text": f"In session {t.session_num} ({t.date_time}), {t.speaker} said: \"{t.text}\"",
                "type": "dialogue_turn",
                "speaker": t.speaker,
                "session": t.session_num,
                "timestamp": t.date_time,
                "status": "active",
            })

        for idx, t in enumerate(turn_dicts):
            ctx_window = turn_dicts[max(0, idx - 3):idx]
            claims = rel_extractor.extract_from_window(t, context_window=ctx_window)
            for c in claims:
                mem.repository.save({
                    "id": c.claim_id,
                    "dia_id": t["turn_id"],
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


def classify_failure(
    category: int,
    is_answerable: bool,
    gt_turns: Set[str],
    retrieved_turns: Set[str],
    ranks: List[int],
    abstained: bool,
    best_f1: float,
) -> Optional[str]:
    """
    Classify failures into standard taxonomy (A through H):
    A. Required fact was never extracted or stored
    B. Fact was stored but lost during consolidation or profile compression
    C. Fact exists but retrieval missed it
    D. Only part of the required multi-hop evidence was retrieved
    E. Temporal ordering, validity or fact updates were handled incorrectly
    F. Correct evidence was retrieved but the answer was wrong (or token F1 < 0.5)
    G. The system answered despite insufficient evidence (adversarial false retrieval)
    H. Other or unexplained failure
    """
    if category == 5 or not is_answerable:
        # Category 5: Adversarial / Unanswerable
        if abstained:
            return None  # Correct abstention
        return "G"  # Answered despite insufficient evidence

    # Answerable queries (Categories 1-4)
    overlap = retrieved_turns & gt_turns

    if gt_turns and gt_turns.issubset(retrieved_turns):
        # All required evidence was retrieved in top 5
        if best_f1 >= 0.5:
            return None  # Full success
        return "F"  # Correct evidence retrieved but answer token F1 < 0.5

    if 0 < len(overlap) < len(gt_turns):
        # Partial evidence retrieved
        return "D"  # Partial multi-hop evidence retrieved

    if len(overlap) == 0:
        # Zero evidence retrieved
        if abstained:
            return "C"  # Fact exists in store but retriever abstained
        if category == 2:
            return "E"  # Temporal ordering or validity issue
        return "C"  # Fact exists but retrieval missed it / ranked below top 5

    return "H"  # Other / unexplained


def evaluate_strategy(
    strategy_id: str,
    strategy_obj: Any,
    questions: List[QuestionItem],
    max_questions: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluates a single strategy across questions and records comprehensive metrics + per-question logs."""
    eval_qs = questions[:max_questions] if max_questions else questions
    total_q = len(eval_qs)
    if total_q == 0:
        return {}, []

    hits_1 = 0
    hits_3 = 0
    hits_5 = 0
    rr_sum = 0.0
    recalls_5 = []
    full_recalls_5 = []

    # Category breakdowns: 1=Single-hop, 2=Temporal, 3=Multi-hop, 4=Open-domain, 5=Adversarial
    category_stats = {
        cat: {
            "total": 0,
            "hits_1": 0,
            "hits_3": 0,
            "hits_5": 0,
            "rr_sum": 0.0,
            "recalls_5": [],
            "full_recalls_5": [],
            "abstained": 0,
            "latencies": [],
            "tokens": [],
        }
        for cat in range(1, 6)
    }

    warm_latencies: List[float] = []
    cold_latencies: List[float] = []
    prompt_token_counts: List[int] = []

    error_breakdown: Dict[str, int] = {
        "A": 0,  # Never extracted/stored
        "B": 0,  # Lost in profile compression
        "C": 0,  # Retrieval missed it / false abstention
        "D": 0,  # Partial multi-hop evidence
        "E": 0,  # Temporal invalidation error
        "F": 0,  # Correct evidence, low F1
        "G": 0,  # Adversarial false retrieval
        "H": 0,  # Other
    }

    per_question_records: List[Dict[str, Any]] = []

    # Hardware measurement start
    process = psutil.Process()
    ram_start_mb = process.memory_info().rss / (1024 * 1024)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Cold run warm-up
    t_cold_start = time.perf_counter()
    _warm_query(strategy_id, strategy_obj, "warmup query about user preference")
    cold_latencies.append((time.perf_counter() - t_cold_start) * 1000.0)

    t_eval_start = time.perf_counter()

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

        # Collect retrieved turn IDs across items
        retrieved_turn_ids: List[str] = []
        ranks: List[int] = []
        best_f1 = 0.0

        for rank, item in enumerate(retrieved_items[:5]):
            item_tids = extract_item_turn_ids(item)
            for tid in item_tids:
                if tid not in retrieved_turn_ids:
                    retrieved_turn_ids.append(tid)

            cand_text = f"{item.get('text', '')} {item.get('value', '')} {item.get('contextual_text', '')} {item.get('fact', '')}"
            f1 = compute_token_f1(cand_text, str(q.answer))
            if f1 > best_f1:
                best_f1 = f1

            is_evidence_hit = bool(set(item_tids) & gt_turns) or (f1 >= 0.5)
            if is_evidence_hit:
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

        overlap = set(retrieved_turn_ids) & gt_turns
        rec5 = (len(overlap) / len(gt_turns)) if gt_turns else (1.0 if not is_answerable and abstained else 0.0)
        full_rec5 = 1.0 if (gt_turns and gt_turns.issubset(set(retrieved_turn_ids))) else 0.0

        recalls_5.append(rec5)
        full_recalls_5.append(full_rec5)

        # Category accumulation
        if cat in category_stats:
            cs = category_stats[cat]
            cs["total"] += 1
            if hit1:
                cs["hits_1"] += 1
            if hit3:
                cs["hits_3"] += 1
            if hit5:
                cs["hits_5"] += 1
            cs["rr_sum"] += recip_rank
            cs["recalls_5"].append(rec5)
            cs["full_recalls_5"].append(full_rec5)
            if abstained:
                cs["abstained"] += 1
            cs["latencies"].append(latency_ms)
            cs["tokens"].append(tokens)

        # Failure classification
        failure_class = classify_failure(
            category=cat,
            is_answerable=is_answerable,
            gt_turns=gt_turns,
            retrieved_turns=set(retrieved_turn_ids),
            ranks=ranks,
            abstained=abstained,
            best_f1=best_f1,
        )

        if failure_class:
            error_breakdown[failure_class] = error_breakdown.get(failure_class, 0) + 1

        # Per-question structured log
        q_record = {
            "sample_id": q.sample_id,
            "question_idx": q.question_idx,
            "strategy_id": strategy_id,
            "category": cat,
            "category_name": CATEGORY_NAMES.get(cat, "Unknown"),
            "question": q.question,
            "answer": str(q.answer),
            "is_answerable": is_answerable,
            "expected_evidence_ids": sorted(gt_turns),
            "retrieved_evidence_ids": retrieved_turn_ids[:5],
            "hit_at_1": 1 if hit1 else 0,
            "hit_at_3": 1 if hit3 else 0,
            "hit_at_5": 1 if hit5 else 0,
            "mrr": round(recip_rank, 4),
            "recall_at_5": round(rec5, 4),
            "full_recall_at_5": round(full_rec5, 4),
            "best_answer_token_f1": round(best_f1, 4),
            "abstained": abstained,
            "context_tokens": tokens,
            "latency_ms": round(latency_ms, 2),
            "failure_class": failure_class,
        }
        per_question_records.append(q_record)

    total_eval_time_s = time.perf_counter() - t_eval_start
    ram_end_mb = process.memory_info().rss / (1024 * 1024)
    peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0

    p50_lat = float(np.percentile(warm_latencies, 50)) if warm_latencies else 0.0
    p95_lat = float(np.percentile(warm_latencies, 95)) if warm_latencies else 0.0
    avg_tokens = float(np.mean(prompt_token_counts)) if prompt_token_counts else 0.0

    # Process category summaries
    per_category_summary = {}
    for cat_id, stats in category_stats.items():
        c_tot = stats["total"]
        if c_tot == 0:
            continue
        per_category_summary[CATEGORY_NAMES[cat_id]] = {
            "category_id": cat_id,
            "total_questions": c_tot,
            "hit_at_1": round(stats["hits_1"] / c_tot, 4),
            "hit_at_3": round(stats["hits_3"] / c_tot, 4),
            "hit_at_5": round(stats["hits_5"] / c_tot, 4),
            "mrr": round(stats["rr_sum"] / c_tot, 4),
            "mean_recall_at_5": round(float(np.mean(stats["recalls_5"])), 4),
            "full_recall_at_5": round(float(np.mean(stats["full_recalls_5"])), 4),
            "abstention_rate": round(stats["abstained"] / c_tot, 4),
            "p50_latency_ms": round(float(np.percentile(stats["latencies"], 50)), 2) if stats["latencies"] else 0.0,
            "mean_context_tokens": round(float(np.mean(stats["tokens"])), 1) if stats["tokens"] else 0.0,
        }

    mh_stats = category_stats[3]
    mh_tot = mh_stats["total"]
    multihop_full_recall = float(np.mean(mh_stats["full_recalls_5"])) if mh_tot > 0 else 0.0

    temp_stats = category_stats[2]
    temp_tot = temp_stats["total"]
    temp_accuracy = (temp_stats["hits_3"] / temp_tot) if temp_tot > 0 else 0.0

    adv_stats = category_stats[5]
    adv_tot = adv_stats["total"]
    adv_abstention_rate = (adv_stats["abstained"] / adv_tot) if adv_tot > 0 else 0.0

    summary_result = {
        "strategy_id": strategy_id,
        "total_questions": total_q,
        "hit_at_1": round(hits_1 / total_q, 4),
        "hit_at_3": round(hits_3 / total_q, 4),
        "hit_at_5": round(hits_5 / total_q, 4),
        "mrr": round(rr_sum / total_q, 4),
        "mean_recall_at_5": round(float(np.mean(recalls_5)), 4),
        "full_recall_at_5": round(float(np.mean(full_recalls_5)), 4),
        "multihop_full_recall_at_5": round(multihop_full_recall, 4),
        "temporal_accuracy_hit3": round(temp_accuracy, 4),
        "adversarial_abstention_accuracy": round(adv_abstention_rate, 4),
        "latency_p50_ms": round(p50_lat, 2),
        "latency_p95_ms": round(p95_lat, 2),
        "cold_start_latency_ms": round(cold_latencies[0], 2) if cold_latencies else 0.0,
        "avg_context_tokens": round(avg_tokens, 1),
        "eval_duration_seconds": round(total_eval_time_s, 2),
        "ram_usage_mb": round(ram_end_mb, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "failure_breakdown": error_breakdown,
        "category_breakdown": per_category_summary,
    }

    return summary_result, per_question_records


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
    total_dev_turns = sum(len(s.turns) for s in samples)
    total_dev_questions = sum(len(s.questions) for s in samples)
    logger.info(f"Loaded LoCoMo development split: {sample_ids} ({total_dev_turns} turns, {total_dev_questions} questions)")

    strategies = [
        "Strategy_A",  # Control: Existing ChunkdUp pipeline
        "Strategy_B",  # Relational extraction + existing retrieval
        "Strategy_C",  # Existing extraction + PPR retrieval
        "Strategy_D",  # Relational extraction + PPR retrieval (full relational)
        "Strategy_E",  # Profile, summary, and archive memory (GPT-inspired)
        "Strategy_F",  # Hybrid (profile + relational PPR + abstention)
    ]

    all_results: Dict[str, List[Dict[str, Any]]] = {s: [] for s in strategies}
    all_question_records: List[Dict[str, Any]] = []

    for sample in samples:
        sample_turns_cnt = len(sample.turns[:max_turns]) if max_turns else len(sample.turns)
        sample_q_cnt = len(sample.questions[:max_questions]) if max_questions else len(sample.questions)
        logger.info(f"--- Setting up strategies for {sample.sample_id} ({sample_turns_cnt} turns, {sample_q_cnt} questions) ---")
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
            strat_eval, q_records = evaluate_strategy(
                strategy_id=strat_id,
                strategy_obj=strat_objs[strat_id],
                questions=sample.questions,
                max_questions=max_questions,
            )
            strat_eval["sample_id"] = sample.sample_id
            all_results[strat_id].append(strat_eval)
            all_question_records.extend(q_records)

    # Aggregate across dev split samples
    aggregated: Dict[str, Dict[str, Any]] = {}
    for strat_id in strategies:
        sample_metrics = all_results[strat_id]
        total_q = sum(m["total_questions"] for m in sample_metrics)
        agg_mrr = sum(m["mrr"] * m["total_questions"] for m in sample_metrics) / total_q if total_q > 0 else 0.0
        agg_hit1 = sum(m["hit_at_1"] * m["total_questions"] for m in sample_metrics) / total_q if total_q > 0 else 0.0
        agg_hit3 = sum(m["hit_at_3"] * m["total_questions"] for m in sample_metrics) / total_q if total_q > 0 else 0.0
        agg_hit5 = sum(m["hit_at_5"] * m["total_questions"] for m in sample_metrics) / total_q if total_q > 0 else 0.0
        agg_rec5 = sum(m["mean_recall_at_5"] * m["total_questions"] for m in sample_metrics) / total_q if total_q > 0 else 0.0
        agg_frec5 = sum(m["full_recall_at_5"] * m["total_questions"] for m in sample_metrics) / total_q if total_q > 0 else 0.0

        agg_mh = float(np.mean([m["multihop_full_recall_at_5"] for m in sample_metrics]))
        agg_temp = float(np.mean([m["temporal_accuracy_hit3"] for m in sample_metrics]))
        agg_adv = float(np.mean([m["adversarial_abstention_accuracy"] for m in sample_metrics]))

        agg_p50 = float(np.mean([m["latency_p50_ms"] for m in sample_metrics]))
        agg_p95 = float(np.mean([m["latency_p95_ms"] for m in sample_metrics]))
        agg_tokens = float(np.mean([m["avg_context_tokens"] for m in sample_metrics]))
        agg_duration = float(sum(m["eval_duration_seconds"] for m in sample_metrics))
        agg_ram = float(max(m["ram_usage_mb"] for m in sample_metrics))
        agg_vram = float(max(m["peak_vram_mb"] for m in sample_metrics))

        combined_errors: Dict[str, int] = {}
        for m in sample_metrics:
            for ek, ev in m["failure_breakdown"].items():
                combined_errors[ek] = combined_errors.get(ek, 0) + ev

        # Aggregate per category
        combined_cat_summary: Dict[str, Dict[str, Any]] = {}
        for cat_name in CATEGORY_NAMES.values():
            cat_entries = [m["category_breakdown"][cat_name] for m in sample_metrics if cat_name in m["category_breakdown"]]
            if not cat_entries:
                continue
            cat_q_tot = sum(e["total_questions"] for e in cat_entries)
            combined_cat_summary[cat_name] = {
                "total_questions": cat_q_tot,
                "hit_at_1": round(sum(e["hit_at_1"] * e["total_questions"] for e in cat_entries) / cat_q_tot, 4),
                "hit_at_3": round(sum(e["hit_at_3"] * e["total_questions"] for e in cat_entries) / cat_q_tot, 4),
                "hit_at_5": round(sum(e["hit_at_5"] * e["total_questions"] for e in cat_entries) / cat_q_tot, 4),
                "mrr": round(sum(e["mrr"] * e["total_questions"] for e in cat_entries) / cat_q_tot, 4),
                "mean_recall_at_5": round(sum(e["mean_recall_at_5"] * e["total_questions"] for e in cat_entries) / cat_q_tot, 4),
                "full_recall_at_5": round(sum(e["full_recall_at_5"] * e["total_questions"] for e in cat_entries) / cat_q_tot, 4),
                "abstention_rate": round(sum(e["abstention_rate"] * e["total_questions"] for e in cat_entries) / cat_q_tot, 4),
                "p50_latency_ms": round(float(np.mean([e["p50_latency_ms"] for e in cat_entries])), 2),
                "mean_context_tokens": round(float(np.mean([e["mean_context_tokens"] for e in cat_entries])), 1),
            }

        aggregated[strat_id] = {
            "strategy_id": strat_id,
            "total_questions_evaluated": total_q,
            "hit_at_1": round(agg_hit1, 4),
            "hit_at_3": round(agg_hit3, 4),
            "hit_at_5": round(agg_hit5, 4),
            "mrr": round(agg_mrr, 4),
            "mean_recall_at_5": round(agg_rec5, 4),
            "full_recall_at_5": round(agg_frec5, 4),
            "multihop_full_recall_at_5": round(agg_mh, 4),
            "temporal_accuracy_hit3": round(agg_temp, 4),
            "adversarial_abstention_accuracy": round(agg_adv, 4),
            "latency_p50_ms": round(agg_p50, 2),
            "latency_p95_ms": round(agg_p95, 2),
            "avg_context_tokens": round(agg_tokens, 1),
            "eval_duration_seconds": round(agg_duration, 2),
            "ram_usage_mb": round(agg_ram, 2),
            "peak_vram_mb": round(agg_vram, 2),
            "failure_breakdown": combined_errors,
            "category_breakdown": combined_cat_summary,
        }

    final_report = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dev_samples": sample_ids,
            "total_turns": total_dev_turns,
            "total_questions": total_dev_questions,
            "held_out_samples_excluded": ["conv-42", "conv-43", "conv-44", "conv-45", "conv-47", "conv-48", "conv-49", "conv-50"],
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
        "aggregated_results": aggregated,
        "per_sample_results": all_results,
    }

    # Save summary JSON
    target_file = output_path or (RESULTS_DIR / "comparative_experiment_report.json")
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    logger.info(f"Summary report written to {target_file}")

    # Save per-question JSON
    per_q_json_path = RESULTS_DIR / "full_locomo_dev_evaluation.json"
    with open(per_q_json_path, "w", encoding="utf-8") as f:
        json.dump(all_question_records, f, indent=2)
    logger.info(f"Per-question JSON written to {per_q_json_path} ({len(all_question_records)} records)")

    # Save per-question CSV
    per_q_csv_path = RESULTS_DIR / "full_locomo_dev_evaluation.csv"
    if all_question_records:
        csv_headers = [
            "sample_id",
            "question_idx",
            "strategy_id",
            "category",
            "category_name",
            "is_answerable",
            "hit_at_1",
            "hit_at_3",
            "hit_at_5",
            "mrr",
            "recall_at_5",
            "full_recall_at_5",
            "best_answer_token_f1",
            "abstained",
            "context_tokens",
            "latency_ms",
            "failure_class",
            "question",
            "answer",
            "expected_evidence_ids",
            "retrieved_evidence_ids",
        ]
        with open(per_q_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers, extrasaction="ignore")
            writer.writeheader()
            for r in all_question_records:
                row = dict(r)
                row["expected_evidence_ids"] = ";".join(r["expected_evidence_ids"])
                row["retrieved_evidence_ids"] = ";".join(r["retrieved_evidence_ids"])
                writer.writerow(row)
        logger.info(f"Per-question CSV written to {per_q_csv_path}")

    return final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ChunkdUp 6-Strategy Full-Scale Comparative Benchmark")
    parser.add_argument("--max-turns", type=int, default=None, help="Cap turns per conversation")
    parser.add_argument("--max-questions", type=int, default=None, help="Cap questions per conversation")
    parser.add_argument("--output", type=str, default=None, help="Custom output JSON path")
    args = parser.parse_args()

    out_p = Path(args.output) if args.output else None
    report = run_experiment(max_turns=args.max_turns, max_questions=args.max_questions, output_path=out_p)

    print("\n" + "=" * 105)
    print("CHUNKDUP 6-STRATEGY COMPARATIVE BENCHMARK: FULL LOCOMO DEV EVALUATION (conv-26, conv-30)")
    print(f"Total Turns: {report['metadata']['total_turns']} | Total Questions: {report['metadata']['total_questions']} | Device: {report['metadata']['device']}")
    print("=" * 105)
    print(f"{'Strategy':<14} | {'Hit@1':<7} | {'Hit@3':<7} | {'Hit@5':<7} | {'MRR':<7} | {'Recall@5':<8} | {'Multi-Hop':<9} | {'Temporal':<8} | {'Adv Abst':<8} | {'p50(ms)':<7}")
    print("-" * 105)
    for s_id, s_data in report["aggregated_results"].items():
        print(f"{s_id:<14} | {s_data['hit_at_1']:<7.4f} | {s_data['hit_at_3']:<7.4f} | {s_data['hit_at_5']:<7.4f} | {s_data['mrr']:<7.4f} | {s_data['mean_recall_at_5']:<8.4f} | {s_data['multihop_full_recall_at_5']:<9.4f} | {s_data['temporal_accuracy_hit3']:<8.4f} | {s_data['adversarial_abstention_accuracy']:<8.4f} | {s_data['latency_p50_ms']:<7.2f}")
    print("=" * 105)
