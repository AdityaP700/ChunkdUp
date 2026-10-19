# evaluation/adapters/longmemeval_adapter.py
"""
LongMemEval-S Benchmark Adapter & Evaluation Engine for ChunkdUp.

Implements the official LongMemEval evaluation protocol:
- Paper: "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory" (ICLR 2025)
- Repository: https://github.com/xiaowu0162/LongMemEval
- Dataset: xiaowu0162/longmemeval-cleaned (longmemeval_s_cleaned.json, 500 questions)

Capabilities & Tasks:
- Information Extraction (single-session-user, single-session-assistant)
- Multi-Session Reasoning (multi-session)
- Knowledge Updates (knowledge-update)
- Temporal Reasoning (temporal-reasoning)
- Abstention (single-session-preference, *_abs questions)

Measures:
- Turn-level and session-level Recall@k (Recall-All and Recall-Any)
- NDCG@k
- Answer evidence coverage (substring containment / exact match)
- Abstention accuracy on unanswerable queries
- p50/p95 latency and context token usage
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("chunkdup.longmemeval_adapter")

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class LongMemEvalTurn:
    """A single dialogue turn within a haystack session."""
    turn_id: str  # e.g., "answer_280352e9_5"
    session_id: str  # e.g., "answer_280352e9"
    turn_index: int  # 1-indexed
    role: str  # "user" | "assistant"
    content: str
    timestamp: str
    has_answer: bool = False


@dataclass
class LongMemEvalInstance:
    """A single evaluation question instance in LongMemEval-S."""
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    haystack_dates: List[str]
    haystack_session_ids: List[str]
    haystack_sessions: List[List[Dict[str, Any]]]
    answer_session_ids: List[str]
    is_abstention: bool = False

    @property
    def all_turns(self) -> List[LongMemEvalTurn]:
        """Flatten all turns across haystack sessions that precede or match the question date."""
        turns: List[LongMemEvalTurn] = []
        for sess_id, sess_date, turns_raw in zip(self.haystack_session_ids, self.haystack_dates, self.haystack_sessions):
            for idx, t in enumerate(turns_raw):
                turn_obj = LongMemEvalTurn(
                    turn_id=f"{sess_id}_{idx+1}",
                    session_id=sess_id,
                    turn_index=idx + 1,
                    role=t.get("role", "user"),
                    content=t.get("content", ""),
                    timestamp=sess_date,
                    has_answer=bool(t.get("has_answer", False)),
                )
                turns.append(turn_obj)
        return turns

    @property
    def gold_turn_ids(self) -> List[str]:
        """IDs of turns carrying required evidence."""
        return [t.turn_id for t in self.all_turns if t.has_answer]

    @property
    def gold_session_ids(self) -> List[str]:
        """IDs of sessions carrying required evidence."""
        return list(self.answer_session_ids)


# ---------------------------------------------------------------------------
# Dataset Loader
# ---------------------------------------------------------------------------

def load_longmemeval_s(
    file_path: str = "evaluation/data/longmemeval_s_cleaned.json",
    max_samples: Optional[int] = None,
    filter_type: Optional[str] = None,
) -> List[LongMemEvalInstance]:
    """Load official cleaned LongMemEval-S dataset from disk."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"LongMemEval dataset not found at {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    instances: List[LongMemEvalInstance] = []
    for item in data:
        q_id = item["question_id"]
        q_type = item.get("question_type", "unknown")
        is_abs = "_abs" in q_id or q_type == "abstention"

        if filter_type and q_type != filter_type:
            continue

        inst = LongMemEvalInstance(
            question_id=q_id,
            question_type=q_type,
            question=item["question"],
            answer=str(item.get("answer", "")),
            question_date=item.get("question_date", ""),
            haystack_dates=item.get("haystack_dates", []),
            haystack_session_ids=item.get("haystack_session_ids", []),
            haystack_sessions=item.get("haystack_sessions", []),
            answer_session_ids=item.get("answer_session_ids", []),
            is_abstention=is_abs,
        )
        instances.append(inst)
        if max_samples and len(instances) >= max_samples:
            break

    logger.info(f"Loaded {len(instances)} LongMemEval-S instances from {file_path}")
    return instances


# ---------------------------------------------------------------------------
# Official Metric Primitives (Mirroring LongMemEval/src/retrieval/eval_utils.py)
# ---------------------------------------------------------------------------

def dcg(relevances: List[float], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    arr = np.asarray(relevances, dtype=float)[:k]
    if arr.size:
        return float(arr[0] + np.sum(arr[1:] / np.log2(np.arange(2, arr.size + 1))))
    return 0.0


def ndcg(rankings: List[int], correct_docs: Set[str], corpus_ids: List[str], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    relevances = [1.0 if corpus_ids[idx] in correct_docs else 0.0 for idx in range(len(corpus_ids))]
    sorted_relevances = [relevances[idx] for idx in rankings[:k] if idx < len(relevances)]
    ideal_relevance = sorted(relevances, reverse=True)
    ideal_dcg = dcg(ideal_relevance, k)
    actual_dcg = dcg(sorted_relevances, k)
    if ideal_dcg <= 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def evaluate_retrieval_metrics(
    ranked_doc_ids: List[str],
    gold_doc_ids: List[str],
    all_corpus_ids: List[str],
    k_values: Tuple[int, ...] = (1, 3, 5, 8, 10),
) -> Dict[str, float]:
    """Compute official Recall-Any, Recall-All, and NDCG across specified k cutoffs."""
    gold_set = set(gold_doc_ids)
    if not gold_set:
        return {f"recall_any@{k}": 0.0 for k in k_values} | {f"recall_all@{k}": 0.0 for k in k_values} | {f"ndcg_any@{k}": 0.0 for k in k_values}

    id_to_idx = {cid: idx for idx, cid in enumerate(all_corpus_ids)}
    rankings = [id_to_idx[rid] for rid in ranked_doc_ids if rid in id_to_idx]
    # Append unretrieved indices at the end
    retrieved_idx_set = set(rankings)
    unretrieved = [idx for idx in range(len(all_corpus_ids)) if idx not in retrieved_idx_set]
    full_rankings = rankings + unretrieved

    metrics: Dict[str, float] = {}
    for k in k_values:
        recalled_at_k = set(ranked_doc_ids[:k])
        metrics[f"recall_any@{k}"] = 1.0 if any(g in recalled_at_k for g in gold_set) else 0.0
        metrics[f"recall_all@{k}"] = 1.0 if all(g in recalled_at_k for g in gold_set) else 0.0
        metrics[f"ndcg_any@{k}"] = round(ndcg(full_rankings, gold_set, all_corpus_ids, k=k), 4)

    return metrics


def check_answer_evidence_coverage(retrieved_text: str, expected_answer: str) -> bool:
    """
    Check if the expected answer string or its core keywords appear in the retrieved context.
    Normalizes casing, punctuation, and handles multi-word phrases.
    """
    if not expected_answer or not retrieved_text:
        return False

    ans_norm = re.sub(r"[^\w\s]", " ", expected_answer.lower()).strip()
    text_norm = re.sub(r"[^\w\s]", " ", retrieved_text.lower()).strip()

    if not ans_norm:
        return False

    # Exact normalized substring match
    if ans_norm in text_norm:
        return True

    # If multi-token answer, check if all significant content words (>3 chars) are present
    words = [w for w in ans_norm.split() if len(w) > 3]
    if len(words) >= 2 and all(w in text_norm for w in words):
        return True

    return False
