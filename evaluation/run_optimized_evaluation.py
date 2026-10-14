# evaluation/run_optimized_evaluation.py
"""
Full-Scale Comparative Evaluation of Baseline vs Optimized Memory Strategies.

Strategies Evaluated:
1. Strategy_A: Baseline Hybrid CE Control (Unchanged)
2. Strategy_B: Relational Extraction + Hybrid CE Control (Unchanged)
3. Strategy_C: Baseline Pure PPR with Existing Extraction (Unchanged reference)
4. Strategy_Opt_C: Graph-Augmented Retrieval with Existing Extraction (Phase 2)
5. Strategy_D: Baseline Relational PPR (Unchanged reference)
6. Strategy_Opt_D: Graph-Augmented Retrieval with Relational Extraction (Phase 2)
7. Strategy_E: Baseline Profile Memory (Unchanged reference)
8. Strategy_Opt_E: Two-Tier Profile-Guided Archive Retriever (Phase 3)
9. Strategy_F: Baseline Hybrid Memory (Unchanged reference)
10. Strategy_Opt_F: Query-Routed Hybrid Memory with Calibrated Abstention (Phase 4)

Datasets:
- Development Split: conv-26, conv-30 (788 turns, 304 questions)
- Held-Out Split: conv-41 through conv-50 (5,094 turns, 1,682 questions)
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
import psutil
import torch

from chunkdup.hybrid_memory import HybridMemory
from chunkdup.hybrid_memory_optimized import OptimizedHybridMemory
from chunkdup.memory import Memory
from chunkdup.profile.profile_guided_retriever import ProfileGuidedArchiveRetriever
from chunkdup.profile.profile_memory import ProfileMemory
from chunkdup.relational.extractor import RelationalExtractor
from chunkdup.relational.graph import RelationalGraph
from chunkdup.relational.schema import RelationalClaim
from chunkdup.retrieval_graph_augmented import GraphAugmentedRetriever
from chunkdup.retrieval_ppr import PPRHybridRetriever, RelationalPPRGraph
from chunkdup.retrieval_production import ProductionRetriever
from evaluation.adapters.locomo_adapter import ConversationSample, LoCoMoLoader, QuestionItem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("chunkdup.optimized_eval")

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
        if ":" in c_str and ("conv-" in c_str or "sample_" in c_str):
            parts = c_str.split(":", 1)
            c_str = parts[1]
        match = re.search(r"\b(D\d+:\d+)\b", c_str)
        if match:
            tid = match.group(1)
            if tid not in turn_ids:
                turn_ids.append(tid)
        elif c_str.startswith("D") and ":" in c_str and c_str not in turn_ids:
            turn_ids.append(c_str)

    text_content = f"{item.get('contextual_text', '')} {item.get('source', '')} {item.get('value', '')} {item.get('text', '')}"
    for match in re.finditer(r"\b(D\d+:\d+)\b", text_content):
        tid = match.group(1)
        if tid not in turn_ids:
            turn_ids.append(tid)

    return turn_ids


class OptimizedHarness:
    """Sets up and runs memory strategies on a conversation sample."""

    def __init__(self, sample: ConversationSample, device: str = "cuda"):
        self.sample = sample
        self.turns = sample.turns
        self.questions = sample.questions
        self.device = device if (torch.cuda.is_available() and device == "cuda") else "cpu"

    # -- Strategy A (Baseline Control) ----------------------------------------
    def setup_strategy_a(self) -> Tuple[Memory, ProductionRetriever]:
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
            device=self.device,
        )
        retriever._ensure_index()
        return mem, retriever

    # -- Strategy B (Baseline Relational + Hybrid) -----------------------------
    def setup_strategy_b(self) -> Tuple[Memory, ProductionRetriever]:
        temp_dir = tempfile.mkdtemp()
        mem = Memory(store="memory", data_dir=temp_dir, extractor="rules")
        rel_extractor = RelationalExtractor(mode="rules")
        turn_dicts = [
            {"turn_id": t.dia_id, "speaker": t.speaker, "text": t.text, "timestamp": t.date_time}
            for t in self.turns
        ]
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
            device=self.device,
        )
        retriever._ensure_index()
        return mem, retriever

    # -- Strategy C (Baseline Pure PPR) ----------------------------------------
    def setup_strategy_c(self) -> RelationalPPRGraph:
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
        ppr_graph = RelationalPPRGraph(relational_graph=rel_graph, device=self.device)
        ppr_graph.build_from_passages_and_claims(passages)
        return ppr_graph

    # -- Strategy Opt_C (Optimized Graph-Augmented with Existing Extraction) ---
    def setup_strategy_opt_c(self) -> GraphAugmentedRetriever:
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
        passages = [
            {
                "id": t.dia_id,
                "dia_id": t.dia_id,
                "text": f"{t.speaker}: {t.text}",
                "speaker": t.speaker,
                "session": f"session_{t.session_num}",
                "timestamp": t.date_time,
            }
            for t in self.turns
        ]
        retriever = GraphAugmentedRetriever(relational_graph=rel_graph, device=self.device)
        retriever.index_passages(passages)
        return retriever

    # -- Strategy D (Baseline Relational PPR) -----------------------------------
    def setup_strategy_d(self) -> RelationalPPRGraph:
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
        ppr_graph = RelationalPPRGraph(relational_graph=rel_graph, device=self.device)
        ppr_graph.build_from_passages_and_claims(passages)
        return ppr_graph

    # -- Strategy Opt_D (Optimized Graph-Augmented with Relational Extraction) -
    def setup_strategy_opt_d(self) -> GraphAugmentedRetriever:
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
        passages = [
            {
                "id": t.dia_id,
                "dia_id": t.dia_id,
                "text": f"{t.speaker}: {t.text}",
                "speaker": t.speaker,
                "session": f"session_{t.session_num}",
                "timestamp": t.date_time,
            }
            for t in self.turns
        ]
        retriever = GraphAugmentedRetriever(relational_graph=rel_graph, device=self.device)
        retriever.index_passages(passages)
        return retriever

    # -- Strategy E (Baseline Profile Memory) ----------------------------------
    def setup_strategy_e(self) -> ProfileMemory:
        prof_mem = ProfileMemory(consolidation_mode="rules")
        for t in self.turns:
            prof_mem.remember({
                "turn_id": t.dia_id,
                "speaker": t.speaker,
                "text": t.text,
                "timestamp": t.date_time,
            })
        return prof_mem

    # -- Strategy Opt_E (Two-Tier Profile-Guided Archive Retriever) -----------
    def setup_strategy_opt_e(self) -> ProfileGuidedArchiveRetriever:
        prof_mem = ProfileMemory(consolidation_mode="rules")
        passages = []
        for t in self.turns:
            prof_mem.remember({
                "turn_id": t.dia_id,
                "speaker": t.speaker,
                "text": t.text,
                "timestamp": t.date_time,
            })
            passages.append({
                "id": t.dia_id,
                "dia_id": t.dia_id,
                "text": f"{t.speaker}: {t.text}",
                "speaker": t.speaker,
                "session": f"session_{t.session_num}",
                "timestamp": t.date_time,
            })
        retriever = ProfileGuidedArchiveRetriever(profile=prof_mem.profile, device=self.device)
        retriever.index_turns(passages)
        return retriever

    # -- Strategy F (Baseline Hybrid Memory) -----------------------------------
    def setup_strategy_f(self) -> HybridMemory:
        hybrid = HybridMemory(extractor_mode="rules", device=self.device)
        for t in self.turns:
            hybrid.remember(
                text=t.text,
                speaker=t.speaker,
                timestamp=t.date_time,
                turn_id=t.dia_id,
            )
        return hybrid

    # -- Strategy Opt_F (Query-Routed Hybrid Memory with Calibrated Abstention)
    def setup_strategy_opt_f(self) -> OptimizedHybridMemory:
        hybrid = OptimizedHybridMemory(device=self.device, extractor_mode="rules", adversarial_threshold=2.0)
        for t in self.turns:
            hybrid.remember(
                turn_id=t.dia_id,
                speaker=t.speaker,
                text=t.text,
                session_id=f"session_{t.session_num}",
                timestamp=t.date_time,
            )
        return hybrid


def execute_strategy_query(
    strategy_id: str,
    strategy_obj: Any,
    query: str,
) -> Tuple[List[Dict[str, Any]], bool, str]:
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

    elif strategy_id in ("Strategy_Opt_C", "Strategy_Opt_D"):
        retriever: GraphAugmentedRetriever = strategy_obj
        res = retriever.retrieve(query=query, top_k=5)
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

    elif strategy_id == "Strategy_Opt_E":
        retriever: ProfileGuidedArchiveRetriever = strategy_obj
        res_dict = retriever.retrieve(query=query, top_k=5)
        items = res_dict.get("results", [])
        abstained = res_dict.get("abstained", False)
        return items, abstained, res_dict.get("prompt_context", "")

    elif strategy_id == "Strategy_F":
        hybrid: HybridMemory = strategy_obj
        res_dict = hybrid.retrieve(query=query, top_k=5)
        items = res_dict.get("results", [])
        abstained = res_dict.get("abstained", False)
        return items, abstained, res_dict.get("prompt_context", "")

    elif strategy_id == "Strategy_Opt_F":
        hybrid: OptimizedHybridMemory = strategy_obj
        res_dict = hybrid.retrieve(query=query, top_k=5)
        items = res_dict.get("results", [])
        abstained = res_dict.get("abstained", False)
        return items, abstained, res_dict.get("prompt_context", "")

    return [], True, ""


def evaluate_single_strategy(
    strategy_id: str,
    strategy_obj: Any,
    questions: List[QuestionItem],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    total_q = len(questions)
    if total_q == 0:
        return {}, []

    hits_1 = 0
    hits_3 = 0
    hits_5 = 0
    rr_sum = 0.0
    recalls_5 = []
    full_recalls_5 = []
    p_at_5_sum = 0.0

    category_stats = {
        cat: {
            "total": 0, "hits_1": 0, "hits_3": 0, "hits_5": 0,
            "rr_sum": 0.0, "recalls_5": [], "full_recalls_5": [],
            "abstained": 0, "latencies": [], "tokens": [],
        }
        for cat in range(1, 6)
    }

    warm_latencies: List[float] = []
    prompt_token_counts: List[int] = []
    per_question_records: List[Dict[str, Any]] = []

    # Warmup
    try:
        execute_strategy_query(strategy_id, strategy_obj, "warmup query about user preference")
    except Exception:
        pass

    t_eval_start = time.perf_counter()

    for q in questions:
        gt_turns = set(q.evidence_dia_ids)
        is_answerable = q.is_answerable
        cat = q.category

        t_start = time.perf_counter()
        retrieved_items, abstained, prompt_ctx = execute_strategy_query(strategy_id, strategy_obj, q.question)
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        warm_latencies.append(latency_ms)

        tokens = len(prompt_ctx.split())
        prompt_token_counts.append(tokens)

        retrieved_turn_ids: List[str] = []
        ranks: List[int] = []
        best_f1 = 0.0

        for rank, item in enumerate(retrieved_items[:5]):
            item_tids = extract_item_turn_ids(item)
            for tid in item_tids:
                if tid not in retrieved_turn_ids:
                    retrieved_turn_ids.append(tid)

            cand_text = f"{item.get('text', '')} {item.get('value', '')} {item.get('contextual_text', '')}"
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

        p_at_5 = (len(overlap) / 5.0) if gt_turns else (1.0 if not is_answerable and abstained else 0.0)
        p_at_5_sum += p_at_5

        if cat in category_stats:
            cs = category_stats[cat]
            cs["total"] += 1
            if hit1: cs["hits_1"] += 1
            if hit3: cs["hits_3"] += 1
            if hit5: cs["hits_5"] += 1
            cs["rr_sum"] += recip_rank
            cs["recalls_5"].append(rec5)
            cs["full_recalls_5"].append(full_rec5)
            if abstained: cs["abstained"] += 1
            cs["latencies"].append(latency_ms)
            cs["tokens"].append(tokens)

        per_question_records.append({
            "question_id": getattr(q, "question_idx", getattr(q, "question_id", "")),
            "category": cat,
            "category_name": CATEGORY_NAMES.get(cat, "Unknown"),
            "question": q.question,
            "gold_evidence": sorted(gt_turns),
            "retrieved_turns": retrieved_turn_ids[:5],
            "ranks": ranks,
            "hit_1": hit1,
            "hit_3": hit3,
            "hit_5": hit5,
            "reciprocal_rank": recip_rank,
            "recall_at_5": rec5,
            "p_at_5": p_at_5,
            "abstained": abstained,
            "best_f1": best_f1,
            "latency_ms": latency_ms,
            "prompt_tokens": tokens,
        })

    duration_s = time.perf_counter() - t_eval_start
    p50_lat = float(np.percentile(warm_latencies, 50)) if warm_latencies else 0.0
    p95_lat = float(np.percentile(warm_latencies, 95)) if warm_latencies else 0.0
    avg_tokens = float(np.mean(prompt_token_counts)) if prompt_token_counts else 0.0

    cat_breakdown = {}
    for cat, cs in category_stats.items():
        n = cs["total"]
        if n > 0:
            cat_breakdown[cat] = {
                "name": CATEGORY_NAMES.get(cat, str(cat)),
                "count": n,
                "hit_at_1": float(cs["hits_1"] / n),
                "hit_at_3": float(cs["hits_3"] / n),
                "hit_at_5": float(cs["hits_5"] / n),
                "mrr": float(cs["rr_sum"] / n),
                "mean_recall_at_5": float(np.mean(cs["recalls_5"])),
                "full_recall_at_5": float(np.mean(cs["full_recalls_5"])),
                "abstention_rate": float(cs["abstained"] / n),
                "latency_p50_ms": float(np.percentile(cs["latencies"], 50)) if cs["latencies"] else 0.0,
                "avg_tokens": float(np.mean(cs["tokens"])) if cs["tokens"] else 0.0,
            }

    temporal_acc = cat_breakdown.get(2, {}).get("hit_at_3", 0.0)
    adv_acc = cat_breakdown.get(5, {}).get("abstention_rate", 0.0)
    multihop_full_rec = cat_breakdown.get(3, {}).get("full_recall_at_5", 0.0)

    summary = {
        "strategy_id": strategy_id,
        "total_questions_evaluated": total_q,
        "hit_at_1": float(hits_1 / total_q),
        "hit_at_3": float(hits_3 / total_q),
        "hit_at_5": float(hits_5 / total_q),
        "mrr": float(rr_sum / total_q),
        "p_at_5": float(p_at_5_sum / total_q),
        "mean_recall_at_5": float(np.mean(recalls_5)) if recalls_5 else 0.0,
        "full_recall_at_5": float(np.mean(full_recalls_5)) if full_recalls_5 else 0.0,
        "multihop_full_recall_at_5": float(multihop_full_rec),
        "temporal_accuracy_hit3": float(temporal_acc),
        "adversarial_abstention_accuracy": float(adv_acc),
        "latency_p50_ms": float(p50_lat),
        "latency_p95_ms": float(p95_lat),
        "avg_context_tokens": float(avg_tokens),
        "eval_duration_seconds": float(duration_s),
        "category_breakdown": cat_breakdown,
    }

    return summary, per_question_records


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline vs optimized ChunkdUp memory strategies")
    parser.add_argument("--held-out", action="store_true", help="Run on untouched held-out split (conv-41..conv-50)")
    parser.add_argument("--sample-ids", nargs="+", default=None, help="Explicit conversation sample IDs to evaluate")
    parser.add_argument("--max-questions", type=int, default=None, help="Limit questions per conversation")
    parser.add_argument("--strategies", nargs="+", default=None, help="Subset of strategies to evaluate")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (cuda or cpu)")
    args = parser.parse_args()

    locomo_path = REPO_ROOT / "evaluation" / "data" / "locomo10.json"
    if not locomo_path.exists():
        raise FileNotFoundError(f"Dataset not found: {locomo_path}")

    split_name = "eval" if args.held_out else "dev"
    loaded_samples, _ = LoCoMoLoader.load(locomo_path, split=split_name)

    if args.sample_ids:
        target_ids = set(args.sample_ids)
        samples = [s for s in loaded_samples if s.sample_id in target_ids]
    else:
        samples = loaded_samples

    sample_ids = [s.sample_id for s in samples]
    logger.info("Evaluating on samples: %s", sample_ids)

    all_strategies = [
        "Strategy_A",
        "Strategy_B",
        "Strategy_C",
        "Strategy_Opt_C",
        "Strategy_D",
        "Strategy_Opt_D",
        "Strategy_E",
        "Strategy_Opt_E",
        "Strategy_F",
        "Strategy_Opt_F",
    ]
    eval_strategies = args.strategies or all_strategies

    strategy_aggregated: Dict[str, Dict[str, Any]] = {}
    strategy_per_question: Dict[str, List[Dict[str, Any]]] = {s: [] for s in eval_strategies}

    for sample in samples:
        sample_id = sample.sample_id
        qs = sample.questions[:args.max_questions] if args.max_questions else sample.questions
        logger.info("Loaded sample %s: %d turns, %d questions", sample_id, len(sample.turns), len(qs))
        harness = OptimizedHarness(sample, device=args.device)

        for strat_id in eval_strategies:
            logger.info("Initializing %s for %s...", strat_id, sample_id)
            if strat_id == "Strategy_A":
                obj = harness.setup_strategy_a()
            elif strat_id == "Strategy_B":
                obj = harness.setup_strategy_b()
            elif strat_id == "Strategy_C":
                obj = harness.setup_strategy_c()
            elif strat_id == "Strategy_Opt_C":
                obj = harness.setup_strategy_opt_c()
            elif strat_id == "Strategy_D":
                obj = harness.setup_strategy_d()
            elif strat_id == "Strategy_Opt_D":
                obj = harness.setup_strategy_opt_d()
            elif strat_id == "Strategy_E":
                obj = harness.setup_strategy_e()
            elif strat_id == "Strategy_Opt_E":
                obj = harness.setup_strategy_opt_e()
            elif strat_id == "Strategy_F":
                obj = harness.setup_strategy_f()
            elif strat_id == "Strategy_Opt_F":
                obj = harness.setup_strategy_opt_f()
            else:
                logger.warning("Unknown strategy %s, skipping", strat_id)
                continue

            summary, q_records = evaluate_single_strategy(strat_id, obj, qs)
            for rec in q_records:
                rec["sample_id"] = sample_id
                strategy_per_question[strat_id].append(rec)

            logger.info(
                "[%s | %s] Hit@5: %.2f%% | MRR: %.4f | P@5: %.4f | Recall@5: %.2f%% | Tokens: %.1f | p50: %.1f ms",
                sample_id, strat_id,
                summary["hit_at_5"] * 100.0,
                summary["mrr"],
                summary["p_at_5"],
                summary["mean_recall_at_5"] * 100.0,
                summary["avg_context_tokens"],
                summary["latency_p50_ms"],
            )

    # Compute macro-aggregated results across all evaluated questions
    print("\n" + "=" * 100)
    print("FINAL COMPARATIVE EVALUATION RESULTS")
    print("=" * 100)
    print(f"{'Strategy':16} | {'Hit@1':7} | {'Hit@3':7} | {'Hit@5':7} | {'MRR':7} | {'P@5':7} | {'Rec@5':7} | {'Tokens':7} | {'p50(ms)':7} | {'AdvAbs':7}")
    print("-" * 100)

    final_report_data: Dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "samples": sample_ids,
            "total_samples": len(sample_ids),
            "split": "held_out" if args.held_out else "development",
            "device": args.device,
        },
        "results": {},
    }

    for strat_id in eval_strategies:
        recs = strategy_per_question[strat_id]
        if not recs:
            continue
        tot = len(recs)
        h1 = sum(1 for r in recs if r["hit_1"]) / tot
        h3 = sum(1 for r in recs if r["hit_3"]) / tot
        h5 = sum(1 for r in recs if r["hit_5"]) / tot
        mrr = sum(r["reciprocal_rank"] for r in recs) / tot
        p5 = sum(r["p_at_5"] for r in recs) / tot
        rec5 = sum(r["recall_at_5"] for r in recs) / tot
        tokens = float(np.mean([r["prompt_tokens"] for r in recs]))
        p50 = float(np.percentile([r["latency_ms"] for r in recs], 50))
        p95 = float(np.percentile([r["latency_ms"] for r in recs], 95))

        adv_recs = [r for r in recs if r["category"] == 5]
        adv_acc = (sum(1 for r in adv_recs if r["abstained"]) / len(adv_recs)) if adv_recs else 0.0

        # Category breakdowns
        cat_data = {}
        for cat in range(1, 6):
            c_recs = [r for r in recs if r["category"] == cat]
            if c_recs:
                c_tot = len(c_recs)
                cat_data[cat] = {
                    "name": CATEGORY_NAMES.get(cat, str(cat)),
                    "count": c_tot,
                    "hit_at_1": float(sum(1 for r in c_recs if r["hit_1"]) / c_tot),
                    "hit_at_3": float(sum(1 for r in c_recs if r["hit_3"]) / c_tot),
                    "hit_at_5": float(sum(1 for r in c_recs if r["hit_5"]) / c_tot),
                    "mrr": float(sum(r["reciprocal_rank"] for r in c_recs) / c_tot),
                    "p_at_5": float(sum(r["p_at_5"] for r in c_recs) / c_tot),
                    "mean_recall_at_5": float(sum(r["recall_at_5"] for r in c_recs) / c_tot),
                    "abstention_rate": float(sum(1 for r in c_recs if r["abstained"]) / c_tot),
                }

        strat_summary = {
            "strategy_id": strat_id,
            "total_questions": tot,
            "hit_at_1": float(h1),
            "hit_at_3": float(h3),
            "hit_at_5": float(h5),
            "mrr": float(mrr),
            "p_at_5": float(p5),
            "mean_recall_at_5": float(rec5),
            "avg_context_tokens": float(tokens),
            "latency_p50_ms": float(p50),
            "latency_p95_ms": float(p95),
            "adversarial_abstention_accuracy": float(adv_acc),
            "category_breakdown": cat_data,
        }
        final_report_data["results"][strat_id] = strat_summary

        print(
            f"{strat_id:16} | {h1*100:6.2f}% | {h3*100:6.2f}% | {h5*100:6.2f}% | {mrr:7.4f} | {p5:7.4f} | {rec5*100:6.2f}% | {tokens:7.1f} | {p50:7.1f} | {adv_acc*100:6.2f}%"
        )

    print("=" * 100)

    # Save results
    prefix = "held_out" if args.held_out else "dev"
    json_path = RESULTS_DIR / f"optimized_evaluation_{prefix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_report_data, f, indent=2)
    logger.info("Saved summary report to %s", json_path)

    # Save per-question records CSV
    csv_path = RESULTS_DIR / f"optimized_per_question_{prefix}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "question_id", "category", "category_name", "strategy_id",
            "hit_1", "hit_3", "hit_5", "mrr", "recall_at_5", "p_at_5", "abstained", "latency_ms", "tokens"
        ])
        for strat_id, recs in strategy_per_question.items():
            for r in recs:
                qid = r.get("question_id") or r.get("question_idx") or ""
                writer.writerow([
                    r["sample_id"], qid, r["category"], r["category_name"], strat_id,
                    r["hit_1"], r["hit_3"], r["hit_5"], r["reciprocal_rank"], r["recall_at_5"], r["p_at_5"],
                    r["abstained"], f"{r['latency_ms']:.2f}", r["prompt_tokens"]
                ])
    logger.info("Saved per-question CSV to %s", csv_path)


if __name__ == "__main__":
    main()
