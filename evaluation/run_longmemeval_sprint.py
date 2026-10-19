# evaluation/run_longmemeval_sprint.py
"""
Unified LongMemEval-S Benchmark & Scaling Evaluation Harness.

Runs:
1. Baseline BM25 (Top-5, Top-10)
2. Hybrid CE Control (Top-5, Top-8, Top-10)
3. Lightweight Entity-Linked Retrieval (Top-5, Pool10->Top5, Top-10)
4. Bounded Associative Graph Expansion (B=1, Top-5)

Evaluates:
- Turn-level and session-level Recall-All, Recall-Any, and NDCG@k
- Evidence answer coverage (exact / keyword containment)
- Abstention accuracy on unanswerable questions (*_abs)
- Latency (p50, p95), candidate counts, and prompt context tokens
- Saves per-question CSV and aggregate JSON reports
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

# Add repository root and ai-system to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
AI_SYSTEM_DIR = REPO_ROOT / "ai-system"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(AI_SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SYSTEM_DIR))

from evaluation.adapters.longmemeval_adapter import (
    LongMemEvalInstance,
    LongMemEvalTurn,
    check_answer_evidence_coverage,
    evaluate_retrieval_metrics,
    load_longmemeval_s,
)
from evaluation.adapters.locomo_adapter import ConversationSample, DialogueTurn
from chunkdup.entity.entity_graph import EntityGraph
from chunkdup.associative.builder import AssociativeMemoryBuilder
from chunkdup.associative.graph import AssociativeGraph, BoundedGraphExpansion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("chunkdup.longmemeval_sprint")

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+.\-]*")
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "i", "my", "me", "you", "your", "it", "its", "of", "to", "in", "on",
    "for", "with", "and", "or", "do", "does", "did", "am",
    "what", "which", "who", "whose", "where", "when", "how", "why",
}


def _tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def count_tokens(text: str) -> int:
    words = len(text.split())
    return int(words * 1.3)


# ---------------------------------------------------------------------------
# Global Shared Model Cache
# ---------------------------------------------------------------------------

_MODEL_CACHE: Dict[str, Any] = {}


def get_shared_encoder(device: str = "cuda") -> SentenceTransformer:
    key = f"st_{device}"
    if key not in _MODEL_CACHE:
        local_model_path = REPO_ROOT / "models" / "all-MiniLM-L6-v2"
        model_name = str(local_model_path) if local_model_path.exists() else "all-MiniLM-L6-v2"
        logger.info(f"Loading SentenceTransformer: {model_name} on {device}")
        _MODEL_CACHE[key] = SentenceTransformer(model_name, device=device)
    return _MODEL_CACHE[key]


def get_shared_cross_encoder(device: str = "cuda") -> CrossEncoder:
    key = f"ce_{device}"
    if key not in _MODEL_CACHE:
        local_ce_path = REPO_ROOT / "models" / "ms-marco-MiniLM-L-12-v2"
        model_name = str(local_ce_path) if local_ce_path.exists() else "cross-encoder/ms-marco-MiniLM-L-12-v2"
        logger.info(f"Loading CrossEncoder: {model_name} on {device}")
        _MODEL_CACHE[key] = CrossEncoder(model_name, device=device)
    return _MODEL_CACHE[key]


# ---------------------------------------------------------------------------
# Corpus Item and Indexed History
# ---------------------------------------------------------------------------

@dataclass
class IndexedTurn:
    turn_id: str
    session_id: str
    role: str
    content: str
    timestamp: str
    text: str
    has_answer: bool


class QuestionCorpus:
    """Indexed searchable memory corpus for a single LongMemEval question."""

    def __init__(self, instance: LongMemEvalInstance, device: str = "cuda"):
        self.instance = instance
        self.device = device
        self.turns: List[IndexedTurn] = []
        self.turn_ids: List[str] = []
        self.session_ids: List[str] = []
        self.texts: List[str] = []

        # Parse turns
        for t in instance.all_turns:
            # Format indexing text: "[user] (timestamp): content"
            rep_text = f"[{t.role}] ({t.timestamp}): {t.content}"
            it = IndexedTurn(
                turn_id=t.turn_id,
                session_id=t.session_id,
                role=t.role,
                content=t.content,
                timestamp=t.timestamp,
                text=rep_text,
                has_answer=t.has_answer,
            )
            self.turns.append(it)
            self.turn_ids.append(t.turn_id)
            self.session_ids.append(t.session_id)
            self.texts.append(rep_text)

        # 1. BM25 Index
        tokenized_corpus = [_tokenize(txt) for txt in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 2. Dense Embeddings
        st = get_shared_encoder(self.device)
        self.embeddings = st.encode(self.texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        if hasattr(self.embeddings, "cpu"):
            self.embeddings = self.embeddings.cpu().numpy()

        # 3. Lightweight Entity Graph
        self.entity_graph = EntityGraph()
        for idx, it in enumerate(self.turns):
            self.entity_graph.index_memory({
                "id": it.turn_id,
                "key": it.turn_id,
                "value": it.content,
                "contextual_text": it.text,
                "speaker": it.role,
            })

        # 4. Typed Associative Graph
        self.bundle: Optional[Any] = None
        self.associative_graph: Optional[AssociativeGraph] = None

    def get_associative_graph(self) -> Tuple[Any, AssociativeGraph]:
        if self.bundle is None or self.associative_graph is None:
            dt_turns = [
                DialogueTurn(
                    dia_id=it.turn_id,
                    speaker=it.role,
                    text=it.content,
                    session_num=1,
                    date_time=it.timestamp,
                )
                for it in self.turns
            ]
            sample = ConversationSample(
                sample_id=self.instance.question_id,
                speaker_a="user",
                speaker_b="assistant",
                turns=dt_turns,
                questions=[],
            )
            builder = AssociativeMemoryBuilder()
            self.bundle, self.associative_graph = builder.build_bundle(sample)
        return self.bundle, self.associative_graph


# ---------------------------------------------------------------------------
# Retrieval Strategies
# ---------------------------------------------------------------------------

def retrieve_bm25(corpus: QuestionCorpus, query: str, top_k: int) -> List[IndexedTurn]:
    scores = corpus.bm25.get_scores(_tokenize(query))
    ranked_indices = np.argsort(-scores)[:top_k]
    return [corpus.turns[i] for i in ranked_indices]


def retrieve_hybrid_ce(
    corpus: QuestionCorpus,
    query: str,
    top_k: int,
    candidate_pool_size: int = 20,
    device: str = "cuda",
) -> List[IndexedTurn]:
    # 1. Dense search
    st = get_shared_encoder(device)
    q_vec = st.encode(query, normalize_embeddings=True, show_progress_bar=False)
    if hasattr(q_vec, "cpu"):
        q_vec = q_vec.cpu().numpy()
    dense_sims = corpus.embeddings @ q_vec
    dense_top = np.argsort(-dense_sims)[:candidate_pool_size]

    # 2. BM25 search
    bm25_scores = corpus.bm25.get_scores(_tokenize(query))
    bm25_top = np.argsort(-bm25_scores)[:candidate_pool_size]

    # 3. RRF Fusion
    rrf_scores: Dict[int, float] = defaultdict(float)
    for rank, idx in enumerate(dense_top):
        rrf_scores[int(idx)] += 1.0 / (60.0 + rank + 1)
    for rank, idx in enumerate(bm25_top):
        rrf_scores[int(idx)] += 1.0 / (60.0 + rank + 1)

    fused_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:candidate_pool_size]

    # 4. Cross-Encoder Rerank
    ce = get_shared_cross_encoder(device)
    pairs = [[query, corpus.texts[i]] for i in fused_indices]
    if not pairs:
        return []

    ce_scores = ce.predict(pairs, batch_size=32, show_progress_bar=False)
    scored = sorted(zip(fused_indices, ce_scores), key=lambda x: x[1], reverse=True)
    return [corpus.turns[i] for i, _ in scored[:top_k]]


def retrieve_entity_linked(
    corpus: QuestionCorpus,
    query: str,
    top_k: int,
    candidate_pool_size: int = 20,
    device: str = "cuda",
) -> List[IndexedTurn]:
    """Lightweight Entity-Linked Retrieval: queries entities and merges into candidate pool."""
    # 1. Base Hybrid candidates
    st = get_shared_encoder(device)
    q_vec = st.encode(query, normalize_embeddings=True, show_progress_bar=False)
    if hasattr(q_vec, "cpu"):
        q_vec = q_vec.cpu().numpy()
    dense_sims = corpus.embeddings @ q_vec
    dense_top = np.argsort(-dense_sims)[:candidate_pool_size]

    bm25_scores = corpus.bm25.get_scores(_tokenize(query))
    bm25_top = np.argsort(-bm25_scores)[:candidate_pool_size]

    rrf_scores: Dict[int, float] = defaultdict(float)
    for rank, idx in enumerate(dense_top):
        rrf_scores[int(idx)] += 1.0 / (60.0 + rank + 1)
    for rank, idx in enumerate(bm25_top):
        rrf_scores[int(idx)] += 1.0 / (60.0 + rank + 1)

    candidate_indices = set(sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:candidate_pool_size])

    # 2. Entity link expansion (bounded by candidate pool budget)
    query_entities = corpus.entity_graph.resolve_entities(query)
    turn_id_to_idx = {it.turn_id: i for i, it in enumerate(corpus.turns)}
    added_entity_turns = 0
    max_entity_adds = max(5, candidate_pool_size // 2)

    for ent in query_entities:
        linked_turn_ids = corpus.entity_graph.entity_to_memories.get(ent, set())
        for tid in linked_turn_ids:
            if tid in turn_id_to_idx and turn_id_to_idx[tid] not in candidate_indices:
                candidate_indices.add(turn_id_to_idx[tid])
                added_entity_turns += 1
                if added_entity_turns >= max_entity_adds:
                    break
        if added_entity_turns >= max_entity_adds:
            break

    # 3. Cross-Encoder Rerank
    candidates = list(candidate_indices)
    ce = get_shared_cross_encoder(device)
    pairs = [[query, corpus.texts[i]] for i in candidates]
    if not pairs:
        return []

    ce_scores = ce.predict(pairs, batch_size=32, show_progress_bar=False)
    scored = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)
    return [corpus.turns[i] for i, _ in scored[:top_k]]


def retrieve_bounded_associative(
    corpus: QuestionCorpus,
    query: str,
    top_k: int = 5,
    device: str = "cuda",
) -> List[IndexedTurn]:
    """Bounded Associative Retrieval (Depth=2, Branch=1, MaxCandidates=5)."""
    # 1. Base Hybrid seeds
    initial_turns = retrieve_hybrid_ce(corpus, query, top_k=top_k, candidate_pool_size=20, device=device)
    if not initial_turns:
        return []

    _, graph = corpus.get_associative_graph()
    candidate_turn_ids: Set[str] = {t.turn_id for t in initial_turns}
    seed_entities: List[str] = []
    seed_scores: Dict[str, float] = {}

    for rank, t in enumerate(initial_turns):
        score = 1.0 / (rank + 1.0)
        for node, evidence_set in graph.node_evidence.items():
            if t.turn_id in evidence_set:
                seed_entities.append(node)
                seed_scores[node] = max(seed_scores.get(node, 0.0), score)

    expander = BoundedGraphExpansion(
        graph=graph,
        max_depth=2,
        max_branching=1,
        max_expanded_candidates=5,
    )
    expanded_nodes = expander.expand(seed_entities[:3], seed_scores=seed_scores)
    for _, _, evidence_set in expanded_nodes:
        for tid in evidence_set:
            candidate_turn_ids.add(tid)

    # 3. Cross-Encoder reranking
    turn_id_to_idx = {it.turn_id: i for i, it in enumerate(corpus.turns)}
    candidate_indices = [turn_id_to_idx[tid] for tid in candidate_turn_ids if tid in turn_id_to_idx]
    if not candidate_indices:
        return initial_turns[:top_k]

    ce = get_shared_cross_encoder(device)
    pairs = [[query, corpus.texts[i]] for i in candidate_indices]
    ce_scores = ce.predict(pairs, batch_size=32, show_progress_bar=False)
    scored = sorted(zip(candidate_indices, ce_scores), key=lambda x: x[1], reverse=True)
    return [corpus.turns[i] for i, _ in scored[:top_k]]


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

def run_evaluation(
    dataset_path: str = "evaluation/data/longmemeval_s_cleaned.json",
    max_samples: Optional[int] = None,
    device: str = "cuda",
    output_dir: str = "evaluation/results",
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    instances = load_longmemeval_s(dataset_path, max_samples=max_samples)
    total_q = len(instances)
    answerable_instances = [inst for inst in instances if not inst.is_abstention]
    abstention_instances = [inst for inst in instances if inst.is_abstention]

    logger.info(f"Starting LongMemEval-S evaluation on {total_q} questions ({len(answerable_instances)} answerable, {len(abstention_instances)} abstention).")

    variants = [
        ("Variant_A_BM25_K5", lambda c, q: retrieve_bm25(c, q, top_k=5)),
        ("Variant_A_BM25_K10", lambda c, q: retrieve_bm25(c, q, top_k=10)),
        ("Variant_B_HybridCE_K5", lambda c, q: retrieve_hybrid_ce(c, q, top_k=5, candidate_pool_size=20, device=device)),
        ("Variant_B_HybridCE_K8", lambda c, q: retrieve_hybrid_ce(c, q, top_k=8, candidate_pool_size=20, device=device)),
        ("Variant_B_HybridCE_K10", lambda c, q: retrieve_hybrid_ce(c, q, top_k=10, candidate_pool_size=20, device=device)),
        ("Variant_C_EntityLinked_K5", lambda c, q: retrieve_entity_linked(c, q, top_k=5, candidate_pool_size=20, device=device)),
        ("Variant_C_EntityLinked_Pool10_K5", lambda c, q: retrieve_entity_linked(c, q, top_k=5, candidate_pool_size=10, device=device)),
        ("Variant_C_EntityLinked_K10", lambda c, q: retrieve_entity_linked(c, q, top_k=10, candidate_pool_size=20, device=device)),
        ("Variant_D_BoundedAssociative_B1_K5", lambda c, q: retrieve_bounded_associative(c, q, top_k=5, device=device)),
    ]

    results_by_variant: Dict[str, Dict[str, Any]] = {
        name: {
            "n_total": total_q,
            "n_answerable": len(answerable_instances),
            "n_abstention": len(abstention_instances),
            "turn_recall_all": defaultdict(list),
            "turn_recall_any": defaultdict(list),
            "turn_ndcg": defaultdict(list),
            "sess_recall_all": defaultdict(list),
            "sess_recall_any": defaultdict(list),
            "sess_ndcg": defaultdict(list),
            "answer_coverage": [],
            "abstention_correct": 0,
            "latencies_ms": [],
            "context_tokens": [],
            "candidate_counts": [],
            "category_metrics": defaultdict(lambda: defaultdict(list)),
        }
        for name, _ in variants
    }

    per_question_rows: List[Dict[str, Any]] = []

    # Iterate over questions
    for idx, inst in enumerate(instances):
        if (idx + 1) % 10 == 0 or idx == 0:
            logger.info(f"Processing question {idx+1}/{total_q}: {inst.question_id} ({inst.question_type})")

        corpus = QuestionCorpus(inst, device=device)
        gold_turn_ids = inst.gold_turn_ids
        gold_sess_ids = inst.gold_session_ids
        all_turn_ids = corpus.turn_ids
        all_sess_ids = list(dict.fromkeys(corpus.session_ids))  # ordered unique session IDs

        for var_name, retrieve_fn in variants:
            t0 = time.perf_counter()
            retrieved_turns = retrieve_fn(corpus, inst.question)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            retrieved_turn_ids = [t.turn_id for t in retrieved_turns]
            retrieved_sess_ids = list(dict.fromkeys(t.session_id for t in retrieved_turns))
            retrieved_text = "\n".join(t.content for t in retrieved_turns)
            token_count = count_tokens(retrieved_text)

            v_dict = results_by_variant[var_name]
            v_dict["latencies_ms"].append(latency_ms)
            v_dict["context_tokens"].append(token_count)
            v_dict["candidate_counts"].append(len(retrieved_turns))

            # Abstention question check
            if inst.is_abstention:
                # Abstention evaluation: true if abstained (low/no evidence)
                # In LongMemEval, abstention questions test whether model refrains from asserting unverified facts
                abstained = len(retrieved_turns) == 0 or not check_answer_evidence_coverage(retrieved_text, inst.answer)
                if abstained:
                    v_dict["abstention_correct"] += 1

                per_question_rows.append({
                    "question_id": inst.question_id,
                    "question_type": inst.question_type,
                    "variant": var_name,
                    "is_abstention": True,
                    "turn_recall_all@5": None,
                    "turn_recall_any@5": None,
                    "answer_coverage": None,
                    "latency_ms": round(latency_ms, 2),
                    "tokens": token_count,
                })
                continue

            # Answerable question evaluation
            turn_metrics = evaluate_retrieval_metrics(retrieved_turn_ids, gold_turn_ids, all_turn_ids, (1, 3, 5, 8, 10))
            sess_metrics = evaluate_retrieval_metrics(retrieved_sess_ids, gold_sess_ids, all_sess_ids, (1, 3, 5, 8, 10))
            ans_cov = check_answer_evidence_coverage(retrieved_text, inst.answer)

            v_dict["answer_coverage"].append(1.0 if ans_cov else 0.0)
            for k in (1, 3, 5, 8, 10):
                v_dict["turn_recall_all"][k].append(turn_metrics[f"recall_all@{k}"])
                v_dict["turn_recall_any"][k].append(turn_metrics[f"recall_any@{k}"])
                v_dict["turn_ndcg"][k].append(turn_metrics[f"ndcg_any@{k}"])
                v_dict["sess_recall_all"][k].append(sess_metrics[f"recall_all@{k}"])
                v_dict["sess_recall_any"][k].append(sess_metrics[f"recall_any@{k}"])
                v_dict["sess_ndcg"][k].append(sess_metrics[f"ndcg_any@{k}"])

            cat = inst.question_type
            v_dict["category_metrics"][cat]["turn_recall_all@5"].append(turn_metrics["recall_all@5"])
            v_dict["category_metrics"][cat]["turn_recall_any@5"].append(turn_metrics["recall_any@5"])
            v_dict["category_metrics"][cat]["answer_coverage"].append(1.0 if ans_cov else 0.0)

            per_question_rows.append({
                "question_id": inst.question_id,
                "question_type": inst.question_type,
                "variant": var_name,
                "is_abstention": False,
                "turn_recall_all@5": turn_metrics["recall_all@5"],
                "turn_recall_any@5": turn_metrics["recall_any@5"],
                "turn_recall_all@8": turn_metrics["recall_all@8"],
                "turn_recall_all@10": turn_metrics["recall_all@10"],
                "sess_recall_all@5": sess_metrics["recall_all@5"],
                "sess_recall_any@5": sess_metrics["recall_any@5"],
                "answer_coverage": 1.0 if ans_cov else 0.0,
                "latency_ms": round(latency_ms, 2),
                "tokens": token_count,
            })

        # Periodic checkpoint every 25 questions
        if (idx + 1) % 25 == 0 or (idx + 1) == total_q:
            ckpt_ans = [i for i in instances[: idx + 1] if not i.is_abstention]
            ckpt_abs = [i for i in instances[: idx + 1] if i.is_abstention]
            ckpt_report = compile_summary_report(
                results_by_variant=results_by_variant,
                variants=variants,
                dataset_path=dataset_path,
                n_processed=idx + 1,
                n_answerable=len(ckpt_ans),
                n_abstention=len(ckpt_abs),
                device=device,
            )
            save_reports(ckpt_report, per_question_rows, output_dir)
            logger.info(f"[Checkpoint] Progress {idx+1}/{total_q} saved to {output_dir}")

    # Compile Final Summary
    final_report = compile_summary_report(
        results_by_variant=results_by_variant,
        variants=variants,
        dataset_path=dataset_path,
        n_processed=total_q,
        n_answerable=len(answerable_instances),
        n_abstention=len(abstention_instances),
        device=device,
    )
    save_reports(final_report, per_question_rows, output_dir)
    print_scoreboard(final_report)
    return final_report


def compile_summary_report(
    results_by_variant: Dict[str, Dict[str, Any]],
    variants: List[Tuple[str, Any]],
    dataset_path: str,
    n_processed: int,
    n_answerable: int,
    n_abstention: int,
    device: str,
) -> Dict[str, Any]:
    summary_report: Dict[str, Any] = {
        "benchmark": "LongMemEval-S",
        "dataset_path": dataset_path,
        "n_samples": n_processed,
        "n_answerable": n_answerable,
        "n_abstention": n_abstention,
        "device": device,
        "variants": {},
    }

    for var_name, _ in variants:
        v = results_by_variant[var_name]
        lats = sorted(v["latencies_ms"]) if v["latencies_ms"] else [0.0]
        n_ans = n_answerable
        n_abs = n_abstention

        summary_report["variants"][var_name] = {
            "turn_recall_all@5": round(float(np.mean(v["turn_recall_all"][5])) if n_ans and v["turn_recall_all"][5] else 0.0, 4),
            "turn_recall_all@8": round(float(np.mean(v["turn_recall_all"][8])) if n_ans and v["turn_recall_all"][8] else 0.0, 4),
            "turn_recall_all@10": round(float(np.mean(v["turn_recall_all"][10])) if n_ans and v["turn_recall_all"][10] else 0.0, 4),
            "turn_recall_any@5": round(float(np.mean(v["turn_recall_any"][5])) if n_ans and v["turn_recall_any"][5] else 0.0, 4),
            "turn_recall_any@8": round(float(np.mean(v["turn_recall_any"][8])) if n_ans and v["turn_recall_any"][8] else 0.0, 4),
            "turn_recall_any@10": round(float(np.mean(v["turn_recall_any"][10])) if n_ans and v["turn_recall_any"][10] else 0.0, 4),
            "turn_ndcg@5": round(float(np.mean(v["turn_ndcg"][5])) if n_ans and v["turn_ndcg"][5] else 0.0, 4),
            "turn_ndcg@10": round(float(np.mean(v["turn_ndcg"][10])) if n_ans and v["turn_ndcg"][10] else 0.0, 4),
            "sess_recall_all@5": round(float(np.mean(v["sess_recall_all"][5])) if n_ans and v["sess_recall_all"][5] else 0.0, 4),
            "sess_recall_all@10": round(float(np.mean(v["sess_recall_all"][10])) if n_ans and v["sess_recall_all"][10] else 0.0, 4),
            "sess_ndcg@5": round(float(np.mean(v["sess_ndcg"][5])) if n_ans and v["sess_ndcg"][5] else 0.0, 4),
            "answer_coverage": round(float(np.mean(v["answer_coverage"])) if n_ans and v["answer_coverage"] else 0.0, 4),
            "abstention_accuracy": round(float(v["abstention_correct"] / n_abs) if n_abs else 1.0, 4),
            "mean_tokens": round(float(np.mean(v["context_tokens"])) if v["context_tokens"] else 0.0, 1),
            "latency_p50_ms": round(float(np.median(lats)), 2),
            "latency_p95_ms": round(float(np.percentile(lats, 95)), 2),
            "category_breakdown": {
                cat: {
                    "n": len(metrics["turn_recall_all@5"]),
                    "turn_recall_all@5": round(float(np.mean(metrics["turn_recall_all@5"])) if metrics["turn_recall_all@5"] else 0.0, 4),
                    "turn_recall_any@5": round(float(np.mean(metrics["turn_recall_any@5"])) if metrics["turn_recall_any@5"] else 0.0, 4),
                    "answer_coverage": round(float(np.mean(metrics["answer_coverage"])) if metrics["answer_coverage"] else 0.0, 4),
                }
                for cat, metrics in v["category_metrics"].items()
            },
        }

    return summary_report


def save_reports(summary_report: Dict[str, Any], per_question_rows: List[Dict[str, Any]], output_dir: str) -> None:
    json_path = os.path.join(output_dir, "longmemeval_s_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    csv_path = os.path.join(output_dir, "longmemeval_s_per_question.csv")
    if per_question_rows:
        keys = list(per_question_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(per_question_rows)


def print_scoreboard(report: Dict[str, Any]):
    print("\n" + "=" * 125)
    print(f"  CHUNKDUP LONGMEMEVAL-S BENCHMARK & TOP-K SCALING SCOREBOARD (N={report['n_samples']}, Device: {report['device']})")
    print("=" * 125)
    header = (
        f"{'Strategy / Variant':<35} | "
        f"{'Turn R-All@5':<13} | "
        f"{'Turn R-Any@5':<13} | "
        f"{'Sess R-All@5':<13} | "
        f"{'Ans Coverage':<13} | "
        f"{'p50 (ms)':<9} | "
        f"{'Tokens':<7}"
    )
    print(header)
    print("-" * 125)
    for name, v in report["variants"].items():
        row = (
            f"{name:<35} | "
            f"{v['turn_recall_all@5']*100:11.2f}% | "
            f"{v['turn_recall_any@5']*100:11.2f}% | "
            f"{v['sess_recall_all@5']*100:11.2f}% | "
            f"{v['answer_coverage']*100:11.2f}% | "
            f"{v['latency_p50_ms']:7.1f}ms | "
            f"{v['mean_tokens']:7.1f}"
        )
        print(row)
    print("=" * 125 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LongMemEval-S Evaluation")
    parser.add_argument("--dataset_path", type=str, default="evaluation/data/longmemeval_s_cleaned.json")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples (e.g. 50 for dev)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="evaluation/results")
    args = parser.parse_args()

    run_evaluation(
        dataset_path=args.dataset_path,
        max_samples=args.max_samples,
        device=args.device,
        output_dir=args.output_dir,
    )
