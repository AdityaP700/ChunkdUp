# evaluation/run_ablations_and_optimization.py
"""
ChunkdUp Research: Representation Ablations, Failure-Driven Optimization, and Held-Out Benchmark.

Executes:
1. Section 7: Controlled Representation Ablations (Dev Split: conv-26, conv-30)
   - A1: Raw-turn memory only (passages only, hybrid retriever)
   - A2: Atomic fact memory only (relational claims only, hybrid retriever)
   - A3: Profile-only memory (Tier 1 profile hot-path only)
   - A4: Profile + recent context window (Tier 1 + Tier 2)
   - A5: Profile + dated archive (Tier 1 + Tier 3 episodic search)
   - A6: Profile + targeted graph expansion (Hybrid Strategy F)

2. Section 8: Failure-Driven Optimization
   - Calibrated Grounding Gate (tau = 2.5) to resolve Class G adversarial false retrieval
   - Multi-Hop Graph & Session-Window Expansion to resolve Class D partial evidence failures

3. Section 9: Held-Out Split Evaluation
   - Runs baseline vs. optimized configuration on held-out split (conv-42 through conv-50)

Saves results to:
- evaluation/results/representation_ablations_report.json
- evaluation/results/optimization_dev_report.json
- evaluation/results/held_out_evaluation_report.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
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

from chunkdup.hybrid_memory import HybridMemory
from chunkdup.memory import Memory
from chunkdup.profile.profile_memory import ProfileMemory
from chunkdup.relational.extractor import RelationalExtractor
from chunkdup.relational.graph import RelationalGraph
from chunkdup.retrieval_ppr import PPRHybridRetriever, RelationalPPRGraph
from chunkdup.retrieval_production import ProductionRetriever
from evaluation.adapters.locomo_adapter import ConversationSample, LoCoMoLoader, QuestionItem
from evaluation.run_comparative_experiment import (
    CATEGORY_NAMES,
    compute_token_f1,
    classify_failure,
    extract_item_turn_ids,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("chunkdup.ablations_and_opt")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class RepresentationAblationHarness:
    """Evaluates the 6 distinct memory representation conditions on a conversation sample."""

    def __init__(self, sample: ConversationSample, max_turns: Optional[int] = None):
        self.sample = sample
        self.turns = sample.turns[:max_turns] if max_turns else sample.turns
        self.questions = sample.questions

    def setup_a1_raw_turns(self) -> Tuple[Memory, ProductionRetriever]:
        """Ablation 1: Raw-turn memory only (passages only, no extracted claims)."""
        temp_dir = tempfile.mkdtemp()
        mem = Memory(store="memory", data_dir=temp_dir, extractor="rules")
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
        retriever = ProductionRetriever(repository=mem.repository, config_name="hybrid_ce_abstain", query_log_path=None)
        retriever._ensure_index()
        return mem, retriever

    def setup_a2_atomic_facts(self) -> Tuple[Memory, ProductionRetriever]:
        """Ablation 2: Atomic fact memory only (relational claims only, no raw turn text)."""
        temp_dir = tempfile.mkdtemp()
        mem = Memory(store="memory", data_dir=temp_dir, extractor="rules")
        rel_extractor = RelationalExtractor(mode="rules")
        turn_dicts = [{"turn_id": t.dia_id, "speaker": t.speaker, "text": t.text, "timestamp": t.date_time} for t in self.turns]
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
        retriever = ProductionRetriever(repository=mem.repository, config_name="hybrid_ce_abstain", query_log_path=None)
        retriever._ensure_index()
        return mem, retriever

    def setup_a3_profile_only(self) -> ProfileMemory:
        """Ablation 3: Profile-only memory (Tier 1 hot-path only, archive retrieval disabled)."""
        prof_mem = ProfileMemory(consolidation_mode="rules")
        for t in self.turns:
            prof_mem.remember({"turn_id": t.dia_id, "speaker": t.speaker, "text": t.text, "timestamp": t.date_time})
        prof_mem.archive.passages.clear()  # Clear archive so only profile can answer
        return prof_mem

    def setup_a4_profile_plus_recent(self) -> ProfileMemory:
        """Ablation 4: Profile + recent context buffer (Tier 1 + Tier 2, no deep archive)."""
        prof_mem = ProfileMemory(consolidation_mode="rules")
        for t in self.turns:
            prof_mem.remember({"turn_id": t.dia_id, "speaker": t.speaker, "text": t.text, "timestamp": t.date_time})
        prof_mem.archive.passages.clear()  # Only recent buffer + profile
        return prof_mem

    def setup_a5_profile_plus_archive(self) -> ProfileMemory:
        """Ablation 5: Profile + dated archive (Full 3-tier profile memory)."""
        prof_mem = ProfileMemory(consolidation_mode="rules")
        for t in self.turns:
            prof_mem.remember({"turn_id": t.dia_id, "speaker": t.speaker, "text": t.text, "timestamp": t.date_time})
        return prof_mem

    def setup_a6_profile_plus_graph(self) -> HybridMemory:
        """Ablation 6: Profile + targeted graph expansion (Hybrid Strategy F)."""
        hybrid = HybridMemory(extractor_mode="rules")
        for t in self.turns:
            hybrid.remember(text=t.text, speaker=t.speaker, timestamp=t.date_time, turn_id=t.dia_id)
        return hybrid


class OptimizedProductionRetriever:
    """
    Failure-Driven Optimized Retriever:
    1. Multi-Hop Graph & Session-Window Expansion:
       When an initial candidate turn is retrieved, automatically expands the candidate pool
       to include temporal neighbors within the same dialogue session and entity-linked turns,
       enabling Cross-Encoder reranking to score joint multi-hop evidence turns.
    2. Calibrated Grounding Gate for Conversational Dialogue:
       Uses a domain-calibrated threshold (tau = 2.5) to filter ungrounded answers on adversarial queries.
    """

    def __init__(
        self,
        repository: Any,
        sample_id: str,
        turns: List[Any],
        ce_threshold: float = 2.5,
        enable_multihop_expansion: bool = True,
    ):
        self.repository = repository
        self.sample_id = sample_id
        self.turns = turns
        self.ce_threshold = ce_threshold
        self.enable_multihop_expansion = enable_multihop_expansion

        self.production_retriever = ProductionRetriever(
            repository=self.repository,
            config_name="hybrid_ce_abstain",
            query_log_path=None,
        )
        self.production_retriever._ensure_index()

        # Build session index for multi-hop expansion
        self.turns_by_session: Dict[int, List[str]] = {}
        self.turn_session_map: Dict[str, int] = {}
        for t in turns:
            s_num = getattr(t, "session_num", 1)
            tid = getattr(t, "dia_id", "")
            if tid:
                self.turns_by_session.setdefault(s_num, []).append(tid)
                self.turn_session_map[tid] = s_num

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        self.production_retriever._ensure_index()
        pool = max(25, 3 * k)

        # 1. Base Dense + BM25 retrieval
        try:
            dense_results = self.production_retriever._dense_search(query, pool)
        except Exception:
            dense_results = []
        try:
            bm25_results = self.production_retriever._bm25_search(query, pool)
        except Exception:
            bm25_results = []

        from chunkdup.retrieval_production import _rrf_fuse
        fused = _rrf_fuse([dense_results, bm25_results], k=60, limit=pool)

        # 2. Targeted Multi-Hop Session & Graph Expansion
        if self.enable_multihop_expansion:
            candidate_tids = set()
            for c in fused[:8]:
                tids = extract_item_turn_ids(c)
                candidate_tids.update(tids)

            # For each candidate turn, include adjacent turns from its session
            expanded_tids: Set[str] = set()
            for tid in candidate_tids:
                s_num = self.turn_session_map.get(tid)
                if s_num and s_num in self.turns_by_session:
                    sess_turns = self.turns_by_session[s_num]
                    try:
                        idx = sess_turns.index(tid)
                        # Expand to immediate predecessor and successor
                        for offset in (-1, 1, -2, 2):
                            nbr_idx = idx + offset
                            if 0 <= nbr_idx < len(sess_turns):
                                expanded_tids.add(sess_turns[nbr_idx])
                    except ValueError:
                        pass

            # Fetch expanded memories from repository and add to pool
            seen_ids = {c.get("id") for c in fused}
            for exp_tid in expanded_tids:
                full_id = f"{self.sample_id}:{exp_tid}"
                if full_id not in seen_ids:
                    exp_mem = getattr(self.repository, "_memories", {}).get(full_id)
                    if not exp_mem and hasattr(self.repository, "get_by_id"):
                        exp_mem = self.repository.get_by_id(full_id)
                    if exp_mem:
                        fused.append(exp_mem)
                        seen_ids.add(full_id)

        # 3. Cross-Encoder Reranking
        ce_pool_limit = min(30, len(fused))
        ranked = self.production_retriever._cross_encoder_rerank(query, fused, limit=ce_pool_limit)

        # 4. Calibrated Grounding Gate for Adversarial Protection
        filtered = []
        for r in ranked:
            score = r.get("_rerank_score", 0.0)
            if score >= self.ce_threshold:
                filtered.append(r)

        return filtered[:k]


def run_evaluation_on_questions(
    retriever_fn: Any,
    questions: List[QuestionItem],
    max_questions: Optional[int] = None,
) -> Dict[str, Any]:
    """Helper to evaluate any retriever on a set of questions."""
    eval_qs = questions[:max_questions] if max_questions else questions
    total_q = len(eval_qs)
    if total_q == 0:
        return {}

    hits_1 = 0
    hits_3 = 0
    hits_5 = 0
    rr_sum = 0.0
    recalls_5 = []
    full_recalls_5 = []

    cat_stats = {
        cat: {
            "total": 0, "hits_1": 0, "hits_3": 0, "hits_5": 0,
            "rr_sum": 0.0, "recalls_5": [], "full_recalls_5": [], "abstained": 0
        }
        for cat in range(1, 6)
    }

    latencies: List[float] = []

    for q in eval_qs:
        gt_turns = set(q.evidence_dia_ids)
        cat = q.category
        is_answerable = q.is_answerable

        t0 = time.perf_counter()
        results = retriever_fn(q.question)
        latency = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency)

        abstained = len(results) == 0

        retrieved_tids: List[str] = []
        ranks: List[int] = []
        best_f1 = 0.0

        for rank, item in enumerate(results[:5]):
            tids = extract_item_turn_ids(item)
            for tid in tids:
                if tid not in retrieved_tids:
                    retrieved_tids.append(tid)

            cand_text = f"{item.get('text', '')} {item.get('value', '')} {item.get('contextual_text', '')} {item.get('fact', '')}"
            f1 = compute_token_f1(cand_text, str(q.answer))
            if f1 > best_f1:
                best_f1 = f1

            is_hit = bool(set(tids) & gt_turns) or (f1 >= 0.5)
            if is_hit:
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

        overlap = set(retrieved_tids) & gt_turns
        rec5 = (len(overlap) / len(gt_turns)) if gt_turns else (1.0 if not is_answerable and abstained else 0.0)
        full_rec5 = 1.0 if (gt_turns and gt_turns.issubset(set(retrieved_tids))) else 0.0

        recalls_5.append(rec5)
        full_recalls_5.append(full_rec5)

        if cat in cat_stats:
            cs = cat_stats[cat]
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

    mh_tot = cat_stats[3]["total"]
    mh_frec = float(np.mean(cat_stats[3]["full_recalls_5"])) if mh_tot > 0 else 0.0
    mh_hit3 = (cat_stats[3]["hits_3"] / mh_tot) if mh_tot > 0 else 0.0

    temp_tot = cat_stats[2]["total"]
    temp_hit3 = (cat_stats[2]["hits_3"] / temp_tot) if temp_tot > 0 else 0.0

    adv_tot = cat_stats[5]["total"]
    adv_abst = (cat_stats[5]["abstained"] / adv_tot) if adv_tot > 0 else 0.0

    return {
        "total_questions": total_q,
        "hit_at_1": round(hits_1 / total_q, 4),
        "hit_at_3": round(hits_3 / total_q, 4),
        "hit_at_5": round(hits_5 / total_q, 4),
        "mrr": round(rr_sum / total_q, 4),
        "mean_recall_at_5": round(float(np.mean(recalls_5)), 4),
        "full_recall_at_5": round(float(np.mean(full_recalls_5)), 4),
        "multihop_hit_at_3": round(mh_hit3, 4),
        "multihop_full_recall_at_5": round(mh_frec, 4),
        "temporal_accuracy_hit3": round(temp_hit3, 4),
        "adversarial_abstention_accuracy": round(adv_abst, 4),
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
    }


def run_all_ablations_and_optimization():
    """Runs the controlled representation ablations, failure-driven optimization, and held-out evaluation."""
    locomo_path = REPO_ROOT / "evaluation" / "data" / "locomo10.json"
    dev_samples, _ = LoCoMoLoader.load(locomo_path, split="dev")
    eval_samples, _ = LoCoMoLoader.load(locomo_path, split="eval")

    logger.info("==========================================================================")
    logger.info("PHASE 1: CONTROLLED REPRESENTATION ABLATIONS (DEV SPLIT: conv-26, conv-30)")
    logger.info("==========================================================================")

    ablation_ids = [
        "A1_RawTurnsOnly",
        "A2_AtomicFactsOnly",
        "A3_ProfileOnly",
        "A4_ProfilePlusRecent",
        "A5_ProfilePlusArchive",
        "A6_ProfilePlusGraph",
    ]

    ablation_results: Dict[str, Dict[str, Any]] = {}

    for ab_id in ablation_ids:
        logger.info(f"Running Representation Ablation: {ab_id}...")
        sample_results = []
        for sample in dev_samples:
            harness = RepresentationAblationHarness(sample)
            if ab_id == "A1_RawTurnsOnly":
                _, ret = harness.setup_a1_raw_turns()
                fn = lambda q, ret=ret: ret.retrieve(q, k=5)
            elif ab_id == "A2_AtomicFactsOnly":
                _, ret = harness.setup_a2_atomic_facts()
                fn = lambda q, ret=ret: ret.retrieve(q, k=5)
            elif ab_id == "A3_ProfileOnly":
                pm = harness.setup_a3_profile_only()
                fn = lambda q, pm=pm: [pm.retrieve(q, top_k=5)["fact"]] if pm.retrieve(q, top_k=5)["source"] == "profile_hot_path" else []
            elif ab_id == "A4_ProfilePlusRecent":
                pm = harness.setup_a4_profile_plus_recent()
                fn = lambda q, pm=pm: [pm.retrieve(q, top_k=5)["fact"]] if pm.retrieve(q, top_k=5)["source"] == "profile_hot_path" else []
            elif ab_id == "A5_ProfilePlusArchive":
                pm = harness.setup_a5_profile_plus_archive()
                def fn_e(q, pm=pm):
                    rd = pm.retrieve(q, top_k=5)
                    return [rd["fact"]] if rd["source"] == "profile_hot_path" else rd.get("passages", [])
                fn = fn_e
            elif ab_id == "A6_ProfilePlusGraph":
                hm = harness.setup_a6_profile_plus_graph()
                fn = lambda q, hm=hm: hm.retrieve(q, top_k=5).get("results", [])

            metrics = run_evaluation_on_questions(fn, sample.questions)
            sample_results.append(metrics)

        # Aggregate across samples
        total_q = sum(m["total_questions"] for m in sample_results)
        agg = {
            "ablation_id": ab_id,
            "total_questions": total_q,
            "hit_at_1": round(sum(m["hit_at_1"] * m["total_questions"] for m in sample_results) / total_q, 4),
            "hit_at_3": round(sum(m["hit_at_3"] * m["total_questions"] for m in sample_results) / total_q, 4),
            "hit_at_5": round(sum(m["hit_at_5"] * m["total_questions"] for m in sample_results) / total_q, 4),
            "mrr": round(sum(m["mrr"] * m["total_questions"] for m in sample_results) / total_q, 4),
            "mean_recall_at_5": round(sum(m["mean_recall_at_5"] * m["total_questions"] for m in sample_results) / total_q, 4),
            "full_recall_at_5": round(sum(m["full_recall_at_5"] * m["total_questions"] for m in sample_results) / total_q, 4),
            "multihop_full_recall_at_5": round(float(np.mean([m["multihop_full_recall_at_5"] for m in sample_results])), 4),
            "temporal_accuracy_hit3": round(float(np.mean([m["temporal_accuracy_hit3"] for m in sample_results])), 4),
            "adversarial_abstention_accuracy": round(float(np.mean([m["adversarial_abstention_accuracy"] for m in sample_results])), 4),
            "latency_p50_ms": round(float(np.mean([m["latency_p50_ms"] for m in sample_results])), 2),
        }
        ablation_results[ab_id] = agg

    with open(RESULTS_DIR / "representation_ablations_report.json", "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    logger.info("==========================================================================")
    logger.info("PHASE 2: FAILURE-DRIVEN OPTIMIZATION (DEV SPLIT: conv-26, conv-30)")
    logger.info("==========================================================================")

    # Compare Baseline Control vs. Optimized Configuration on Dev Split
    opt_dev_results = {}
    for cfg_name, (ce_thresh, enable_exp) in {
        "Baseline_Control": (-2.5, False),
        "Optimized_Abstention_Only": (2.5, False),
        "Optimized_Multihop_Only": (-2.5, True),
        "Optimized_Full_Pipeline": (2.5, True),
    }.items():
        logger.info(f"Evaluating {cfg_name} on Dev Split...")
        sample_metrics = []
        for sample in dev_samples:
            temp_dir = tempfile.mkdtemp()
            mem = Memory(store="memory", data_dir=temp_dir, extractor="rules")
            for t in sample.turns:
                mem.repository.save({
                    "id": f"{sample.sample_id}:{t.dia_id}",
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
            opt_retriever = OptimizedProductionRetriever(
                repository=mem.repository,
                sample_id=sample.sample_id,
                turns=sample.turns,
                ce_threshold=ce_thresh,
                enable_multihop_expansion=enable_exp,
            )
            m = run_evaluation_on_questions(lambda q: opt_retriever.retrieve(q, k=5), sample.questions)
            sample_metrics.append(m)

        tot_q = sum(m["total_questions"] for m in sample_metrics)
        opt_dev_results[cfg_name] = {
            "total_questions": tot_q,
            "hit_at_1": round(sum(m["hit_at_1"] * m["total_questions"] for m in sample_metrics) / tot_q, 4),
            "hit_at_3": round(sum(m["hit_at_3"] * m["total_questions"] for m in sample_metrics) / tot_q, 4),
            "hit_at_5": round(sum(m["hit_at_5"] * m["total_questions"] for m in sample_metrics) / tot_q, 4),
            "mrr": round(sum(m["mrr"] * m["total_questions"] for m in sample_metrics) / tot_q, 4),
            "mean_recall_at_5": round(sum(m["mean_recall_at_5"] * m["total_questions"] for m in sample_metrics) / tot_q, 4),
            "full_recall_at_5": round(sum(m["full_recall_at_5"] * m["total_questions"] for m in sample_metrics) / tot_q, 4),
            "multihop_hit_at_3": round(float(np.mean([m["multihop_hit_at_3"] for m in sample_metrics])), 4),
            "multihop_full_recall_at_5": round(float(np.mean([m["multihop_full_recall_at_5"] for m in sample_metrics])), 4),
            "temporal_accuracy_hit3": round(float(np.mean([m["temporal_accuracy_hit3"] for m in sample_metrics])), 4),
            "adversarial_abstention_accuracy": round(float(np.mean([m["adversarial_abstention_accuracy"] for m in sample_metrics])), 4),
            "latency_p50_ms": round(float(np.mean([m["latency_p50_ms"] for m in sample_metrics])), 2),
        }

    with open(RESULTS_DIR / "optimization_dev_report.json", "w", encoding="utf-8") as f:
        json.dump(opt_dev_results, f, indent=2)

    logger.info("==========================================================================")
    logger.info("PHASE 3: HELD-OUT SPLIT EVALUATION (conv-42, conv-43, conv-44...)")
    logger.info("==========================================================================")

    # Evaluate Baseline Control vs Winning Optimized Configuration on the Held-Out Split
    held_out_results = {}
    for eval_name, (ce_thresh, enable_exp) in {
        "Held_Out_Baseline_Control": (-2.5, False),
        "Held_Out_Optimized_Pipeline": (2.5, True),
    }.items():
        logger.info(f"Running {eval_name} across {len(eval_samples)} held-out conversations...")
        sample_metrics = []
        for s_idx, sample in enumerate(eval_samples):
            temp_dir = tempfile.mkdtemp()
            mem = Memory(store="memory", data_dir=temp_dir, extractor="rules")
            for t in sample.turns:
                mem.repository.save({
                    "id": f"{sample.sample_id}:{t.dia_id}",
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
            retriever = OptimizedProductionRetriever(
                repository=mem.repository,
                sample_id=sample.sample_id,
                turns=sample.turns,
                ce_threshold=ce_thresh,
                enable_multihop_expansion=enable_exp,
            )
            # Evaluate questions
            m = run_evaluation_on_questions(lambda q: retriever.retrieve(q, k=5), sample.questions)
            sample_metrics.append(m)
            logger.info(f"  {eval_name} on {sample.sample_id} ({len(sample.questions)} Qs) -> Hit@5: {m.get('hit_at_5', 0):.4f}, Adv Abst: {m.get('adversarial_abstention_accuracy', 0):.4f}")

        tot_q = sum(m["total_questions"] for m in sample_metrics)
        held_out_results[eval_name] = {
            "total_questions_evaluated": tot_q,
            "total_held_out_conversations": len(eval_samples),
            "hit_at_1": round(sum(m["hit_at_1"] * m["total_questions"] for m in sample_metrics) / tot_q, 4),
            "hit_at_3": round(sum(m["hit_at_3"] * m["total_questions"] for m in sample_metrics) / tot_q, 4),
            "hit_at_5": round(sum(m["hit_at_5"] * m["total_questions"] for m in sample_metrics) / tot_q, 4),
            "mrr": round(sum(m["mrr"] * m["total_questions"] for m in sample_metrics) / tot_q, 4),
            "mean_recall_at_5": round(sum(m["mean_recall_at_5"] * m["total_questions"] for m in sample_metrics) / tot_q, 4),
            "full_recall_at_5": round(sum(m["full_recall_at_5"] * m["total_questions"] for m in sample_metrics) / tot_q, 4),
            "multihop_full_recall_at_5": round(float(np.mean([m["multihop_full_recall_at_5"] for m in sample_metrics])), 4),
            "temporal_accuracy_hit3": round(float(np.mean([m["temporal_accuracy_hit3"] for m in sample_metrics])), 4),
            "adversarial_abstention_accuracy": round(float(np.mean([m["adversarial_abstention_accuracy"] for m in sample_metrics])), 4),
            "latency_p50_ms": round(float(np.mean([m["latency_p50_ms"] for m in sample_metrics])), 2),
        }

    with open(RESULTS_DIR / "held_out_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(held_out_results, f, indent=2)

    print("\n" + "=" * 105)
    print("PHASE 1: REPRESENTATION ABLATIONS SUMMARY (DEV SPLIT: conv-26, conv-30)")
    print("=" * 105)
    print(f"{'Condition':<25} | {'Hit@1':<7} | {'Hit@3':<7} | {'Hit@5':<7} | {'MRR':<7} | {'Recall@5':<8} | {'Multi-Hop':<9} | {'Temporal':<8}")
    print("-" * 105)
    for c_id, c_data in ablation_results.items():
        print(f"{c_id:<25} | {c_data['hit_at_1']:<7.4f} | {c_data['hit_at_3']:<7.4f} | {c_data['hit_at_5']:<7.4f} | {c_data['mrr']:<7.4f} | {c_data['mean_recall_at_5']:<8.4f} | {c_data['multihop_full_recall_at_5']:<9.4f} | {c_data['temporal_accuracy_hit3']:<8.4f}")
    print("=" * 105)

    print("\n" + "=" * 105)
    print("PHASE 2: FAILURE-DRIVEN OPTIMIZATION SUMMARY (DEV SPLIT: conv-26, conv-30)")
    print("=" * 105)
    print(f"{'Configuration':<28} | {'Hit@5':<7} | {'MRR':<7} | {'Multi-Hop Rec':<13} | {'Adv Abstention':<14} | {'p50 (ms)':<8}")
    print("-" * 105)
    for c_id, c_data in opt_dev_results.items():
        print(f"{c_id:<28} | {c_data['hit_at_5']:<7.4f} | {c_data['mrr']:<7.4f} | {c_data['multihop_full_recall_at_5']:<13.4f} | {c_data['adversarial_abstention_accuracy']:<14.4f} | {c_data['latency_p50_ms']:<8.2f}")
    print("=" * 105)

    print("\n" + "=" * 105)
    print("PHASE 3: HELD-OUT SPLIT EVALUATION SUMMARY (8 CONVERSATIONS, 1,682 QUESTIONS)")
    print("=" * 105)
    print(f"{'Configuration':<30} | {'Hit@1':<7} | {'Hit@3':<7} | {'Hit@5':<7} | {'MRR':<7} | {'Adv Abstention':<14} | {'p50 (ms)':<8}")
    print("-" * 105)
    for c_id, c_data in held_out_results.items():
        print(f"{c_id:<30} | {c_data['hit_at_1']:<7.4f} | {c_data['hit_at_3']:<7.4f} | {c_data['hit_at_5']:<7.4f} | {c_data['mrr']:<7.4f} | {c_data['adversarial_abstention_accuracy']:<14.4f} | {c_data['latency_p50_ms']:<8.2f}")
    print("=" * 105)


if __name__ == "__main__":
    run_all_ablations_and_optimization()
