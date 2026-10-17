# evaluation/adapters/locomo_adapter.py
"""
LoCoMo Benchmark Adapter & Baseline Evaluation Runner for ChunkdUp.

LoCoMo (Long Conversation Memory) Benchmark:
- Paper: "Evaluating Very Long-Term Conversational Memory of LLM Agents" (ACL 2024)
- Authors: Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, Yuwei Fang
- Repository: https://github.com/snap-research/locomo
- License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
- Dataset File: data/locomo10.json (10 conversations, 1,986 QA pairs)

Question Categories:
  1: Single-hop / Factual (within a single dialogue session)
  2: Temporal reasoning (dates, event sequences, duration across sessions)
  3: Multi-hop reasoning (synthesizing evidence across multiple sessions)
  4: Open-domain / Inference (speaker statements combined with external/common-sense context)
  5: Adversarial / Unanswerable (trick queries with false premises; target answer is "None")

Evaluation Tracks:
  Track A: Retrieval-Only
    Dialogue turns are indexed in ChunkdUp's memory store preserving provenance
    (dia_id, session, timestamp, speaker). ChunkdUp's production retrieval engine
    (ProductionRetriever) is queried directly. Evaluates ranking, recall, MRR,
    and abstention against ground-truth evidence turns.

  Track B: End-to-End Memory Lifecycle
    Dialogue turns are streamed chronologically through ChunkdUp's ingestion path
    (Memory.remember). Discloses whether facts failed because they were never
    extracted/stored (retention failure) versus ranked too low (retrieval failure).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add repository root and ai-system to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
AI_SYSTEM_DIR = REPO_ROOT / "ai-system"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(AI_SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SYSTEM_DIR))

logger = logging.getLogger("chunkdup.locomo_adapter")

# ---------------------------------------------------------------------------
# Category Metadata
# ---------------------------------------------------------------------------

CATEGORY_NAMES: Dict[int, str] = {
    1: "Single-hop / Factual",
    2: "Temporal reasoning",
    3: "Multi-hop reasoning",
    4: "Open-domain / Inference",
    5: "Adversarial / Unanswerable",
}

DATASET_METADATA = {
    "name": "LoCoMo",
    "paper": "Evaluating Very Long-Term Conversational Memory of LLM Agents (ACL 2024)",
    "authors": "Maharana, Lee, Tulyakov, Bansal, Barbieri, Fang",
    "source_url": "https://github.com/snap-research/locomo",
    "license": "CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International)",
    "default_file": "evaluation/data/locomo10.json",
}

# ---------------------------------------------------------------------------
# Splits and Configurations
# ---------------------------------------------------------------------------

DEFAULT_DEV_SPLIT: Tuple[str, ...] = ("conv-26", "conv-30")
DEFAULT_EVAL_SPLIT: Tuple[str, ...] = (
    "conv-42", "conv-43", "conv-44", "conv-45", "conv-47", "conv-48", "conv-49", "conv-50"
)
TRACK_A_CONFIGS: Tuple[str, ...] = ("hybrid_ce_abstain", "dense_abstain", "bm25", "oracle")

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class DialogueTurn:
    dia_id: str
    speaker: str
    text: str
    session_num: int
    date_time: str
    blip_caption: Optional[str] = None


@dataclass
class QuestionItem:
    sample_id: str
    question_idx: int
    question: str
    answer: str
    category: int
    category_name: str
    raw_evidence: List[str]
    evidence_dia_ids: List[str]
    is_answerable: bool
    is_malformed: bool = False
    malformed_reason: Optional[str] = None


@dataclass
class ConversationSample:
    sample_id: str
    speaker_a: str
    speaker_b: str
    turns: List[DialogueTurn]
    questions: List[QuestionItem]
    observations: Dict[str, Any] = field(default_factory=dict)
    session_summaries: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LoCoMo Loader & Validator
# ---------------------------------------------------------------------------


class LoCoMoLoader:
    """Loads and validates the LoCoMo benchmark dataset with deterministic ordering."""

    _EVIDENCE_PATTERN = re.compile(r"D\s*:?\s*(\d+)\s*:\s*(\d+)")

    @classmethod
    def parse_evidence_ids(cls, raw_list: Any) -> Tuple[List[str], bool, Optional[str]]:
        """Normalize raw evidence annotations into a list of canonical turn IDs ('D<sess>:<turn>').

        Returns (cleaned_ids, is_malformed, reason).
        """
        if not raw_list:
            return [], False, None

        if isinstance(raw_list, str):
            raw_list = [raw_list]

        cleaned: List[str] = []
        is_malformed = False
        malformed_reason = None

        for item in raw_list:
            item_str = str(item).strip()
            # Clean parentheses if present
            item_str = item_str.replace("(", "").replace(")", "").strip()
            if not item_str:
                continue

            matches = cls._EVIDENCE_PATTERN.findall(item_str)
            if matches:
                for sess_str, turn_str in matches:
                    canon_id = f"D{int(sess_str)}:{int(turn_str)}"
                    if canon_id not in cleaned:
                        cleaned.append(canon_id)
            else:
                # Malformed token that couldn't be parsed into any D<sess>:<turn>
                is_malformed = True
                malformed_reason = f"Unparseable evidence token: '{item}'"

        return cleaned, is_malformed, malformed_reason

    @classmethod
    def load(
        cls,
        path: str | Path,
        smoke: bool = False,
        smoke_sample_ids: Optional[List[str]] = None,
        smoke_max_questions: Optional[int] = 10,
        split: str = "all",
        dev_sample_ids: Tuple[str, ...] = DEFAULT_DEV_SPLIT,
    ) -> Tuple[List[ConversationSample], Dict[str, Any]]:
        """Load dataset from disk, validate records, and return structured samples."""
        if split not in ("all", "dev", "eval"):
            raise ValueError(f"Unknown split: {split}. Expected 'all', 'dev', or 'eval'.")

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"LoCoMo dataset file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, list):
            raise ValueError(f"Expected list of conversation samples, got {type(raw_data)}")

        samples: List[ConversationSample] = []
        validation_report = {
            "total_conversations_in_file": len(raw_data),
            "split": split,
            "loaded_conversations": 0,
            "total_turns": 0,
            "total_questions": 0,
            "questions_by_category": {cat: 0 for cat in CATEGORY_NAMES},
            "malformed_evidence_records": [],
            "unsupported_category_records": [],
            "duplicate_sample_ids": [],
            "missing_field_records": [],
        }

        seen_sample_ids: Set[str] = set()

        # Deterministic sorting of conversations by sample_id
        sorted_raw = sorted(raw_data, key=lambda x: str(x.get("sample_id", "")))

        for conv in sorted_raw:
            sample_id = conv.get("sample_id")
            if not sample_id:
                validation_report["missing_field_records"].append("Missing sample_id")
                continue

            if sample_id in seen_sample_ids:
                validation_report["duplicate_sample_ids"].append(sample_id)
                continue
            seen_sample_ids.add(sample_id)

            # Filter by split if requested
            if split == "dev" and sample_id not in dev_sample_ids:
                continue
            elif split == "eval" and sample_id in dev_sample_ids:
                continue

            # Filter for smoke test if requested
            if smoke and smoke_sample_ids and sample_id not in smoke_sample_ids:
                continue

            conv_dict = conv.get("conversation", {})
            speaker_a = conv_dict.get("speaker_a", "Speaker A")
            speaker_b = conv_dict.get("speaker_b", "Speaker B")

            # Extract sessions in chronological order (session_1, session_2, ...)
            session_keys = []
            for k in conv_dict.keys():
                if k.startswith("session_") and not k.endswith("_date_time"):
                    try:
                        num = int(k.split("_")[1])
                        session_keys.append((num, k))
                    except ValueError:
                        pass
            session_keys.sort(key=lambda x: x[0])

            turns: List[DialogueTurn] = []
            for sess_num, sess_key in session_keys:
                sess_turns = conv_dict.get(sess_key, [])
                dt = conv_dict.get(f"{sess_key}_date_time", "")
                if not isinstance(sess_turns, list):
                    continue
                for t in sess_turns:
                    dia_id = t.get("dia_id", "")
                    speaker = t.get("speaker", "")
                    text = t.get("text", "")
                    caption = t.get("blip_caption")
                    if not dia_id or not text:
                        validation_report["missing_field_records"].append(
                            f"{sample_id}:{sess_key} missing dia_id or text"
                        )
                        continue
                    turns.append(
                        DialogueTurn(
                            dia_id=dia_id,
                            speaker=speaker,
                            text=text,
                            session_num=sess_num,
                            date_time=dt,
                            blip_caption=caption,
                        )
                    )

            # Extract and validate QA pairs
            questions: List[QuestionItem] = []
            raw_qa = conv.get("qa", [])
            for q_idx, qa in enumerate(raw_qa):
                q_text = qa.get("question", "").strip()
                ans_text = str(qa.get("answer", "")).strip()
                category = qa.get("category")
                raw_ev = qa.get("evidence", [])

                if not q_text:
                    validation_report["missing_field_records"].append(
                        f"{sample_id} Q#{q_idx} missing question text"
                    )
                    continue

                if category not in CATEGORY_NAMES:
                    validation_report["unsupported_category_records"].append({
                        "sample_id": sample_id,
                        "question_idx": q_idx,
                        "category": category,
                        "question": q_text,
                    })
                    continue

                cleaned_ev, is_malformed, mal_reason = cls.parse_evidence_ids(raw_ev)
                if is_malformed:
                    validation_report["malformed_evidence_records"].append({
                        "sample_id": sample_id,
                        "question_idx": q_idx,
                        "raw_evidence": raw_ev,
                        "reason": mal_reason,
                    })

                is_answerable = category != 5 and ans_text.lower() != "none"

                questions.append(
                    QuestionItem(
                        sample_id=sample_id,
                        question_idx=q_idx,
                        question=q_text,
                        answer=ans_text,
                        category=category,
                        category_name=CATEGORY_NAMES[category],
                        raw_evidence=raw_ev if isinstance(raw_ev, list) else [raw_ev],
                        evidence_dia_ids=cleaned_ev,
                        is_answerable=is_answerable,
                        is_malformed=is_malformed,
                        malformed_reason=mal_reason,
                    )
                )
                validation_report["questions_by_category"][category] += 1

            if smoke and smoke_max_questions and len(questions) > smoke_max_questions:
                questions = questions[:smoke_max_questions]

            validation_report["total_turns"] += len(turns)
            validation_report["total_questions"] += len(questions)

            samples.append(
                ConversationSample(
                    sample_id=sample_id,
                    speaker_a=speaker_a,
                    speaker_b=speaker_b,
                    turns=turns,
                    questions=questions,
                    observations=conv.get("observation", {}),
                    session_summaries=conv.get("session_summary", {}),
                )
            )

            if smoke and smoke_sample_ids is None:
                # Default smoke test uses just 1 conversation
                break

        validation_report["loaded_conversations"] = len(samples)
        return samples, validation_report


# ---------------------------------------------------------------------------
# Track A: Retrieval-Only Evaluator
# ---------------------------------------------------------------------------


class TrackAEvaluator:
    """Evaluates ChunkdUp's ProductionRetriever directly on dialogue turn memory stores."""

    def __init__(
        self,
        config_name: str = "hybrid_ce_abstain",
        ce_timeout_s: float = 10.0,
        demote_enabled: bool = True,
        k_values: Tuple[int, ...] = (1, 3, 5, 10),
    ):
        self.config_name = config_name
        self.ce_timeout_s = ce_timeout_s
        self.demote_enabled = demote_enabled
        self.k_values = k_values

    def build_sample_repository(self, sample: ConversationSample):
        """Prepare an InMemoryRepository containing all dialogue turns for a sample."""
        from chunkdup.in_memory_repository import InMemoryRepository

        repo = InMemoryRepository()
        for turn in sample.turns:
            mem = {
                "id": f"{sample.sample_id}:{turn.dia_id}",
                "dia_id": turn.dia_id,
                "sample_id": sample.sample_id,
                "key": f"turn_{turn.dia_id}",
                "value": f"{turn.speaker}: {turn.text}",
                "type": "dialogue_turn",
                "speaker": turn.speaker,
                "session": turn.session_num,
                "timestamp": turn.date_time,
                "status": "active",
                "source": f"LoCoMo conversation {sample.sample_id} session {turn.session_num}",
                "contextual_text": f"In session {turn.session_num} ({turn.date_time}), {turn.speaker} said: \"{turn.text}\"",
            }
            repo.save(mem)
        return repo

    def evaluate(
        self,
        samples: List[ConversationSample],
        progress_callback=None,
    ) -> Dict[str, Any]:
        """Run Track A retrieval-only evaluation across all samples."""
        from chunkdup.retrieval_production import ProductionRetriever

        max_k = max(self.k_values)
        per_query_results: List[Dict[str, Any]] = []
        unmappable_queries: List[Dict[str, Any]] = []

        t_start = time.perf_counter()

        for s_idx, sample in enumerate(samples):
            repo = self.build_sample_repository(sample)
            if self.config_name != "oracle":
                effective_cfg = "hybrid_ce_abstain" if self.config_name == "bm25" else self.config_name
                retriever = ProductionRetriever(
                    repo,
                    config_name=effective_cfg,
                    ce_timeout_s=self.ce_timeout_s,
                    query_log_path=None,
                )
                retriever.demote_enabled = self.demote_enabled
                # Ensure indexing / warmup
                retriever._ensure_index()
            else:
                retriever = None

            for q_idx, q in enumerate(sample.questions):
                t0 = time.perf_counter()
                ground_truth = set(q.evidence_dia_ids)

                # Extract retrieved dia_ids based on evaluation config
                retrieved_dia_ids: List[str] = []
                if self.config_name == "oracle":
                    if q.is_answerable and ground_truth:
                        retrieved_dia_ids = [d for d in q.evidence_dia_ids][:max_k]
                    else:
                        retrieved_dia_ids = []
                elif self.config_name == "bm25":
                    bm25_hits = retriever._bm25_search(q.question, limit=max_k)
                    for r in bm25_hits:
                        did = r.get("dia_id")
                        if not did and ":" in str(r.get("id", "")):
                            did = str(r["id"]).split(":", 1)[1]
                        if did:
                            retrieved_dia_ids.append(did)
                else:
                    results = retriever.retrieve(q.question, k=max_k)
                    for r in results:
                        did = r.get("dia_id")
                        if not did and ":" in str(r.get("id", "")):
                            did = str(r["id"]).split(":", 1)[1]
                        if did:
                            retrieved_dia_ids.append(did)

                latency_ms = (time.perf_counter() - t0) * 1000.0

                # Track unmappable evidence records (e.g. malformed or zero evidence on answerable queries)
                if q.is_answerable and not ground_truth:
                    unmappable_queries.append({
                        "sample_id": q.sample_id,
                        "question_idx": q.question_idx,
                        "category": q.category,
                        "question": q.question,
                        "raw_evidence": q.raw_evidence,
                    })

                record: Dict[str, Any] = {
                    "sample_id": q.sample_id,
                    "question_idx": q.question_idx,
                    "category": q.category,
                    "category_name": q.category_name,
                    "is_answerable": q.is_answerable,
                    "n_evidence_required": len(ground_truth),
                    "n_retrieved": len(retrieved_dia_ids),
                    "abstained": len(retrieved_dia_ids) == 0,
                    "latency_ms": latency_ms,
                    "retrieved_dia_ids": retrieved_dia_ids,
                    "ground_truth_dia_ids": list(ground_truth),
                }

                if q.is_answerable and ground_truth:
                    for k in self.k_values:
                        top_k = retrieved_dia_ids[:k]
                        hits = [did for did in top_k if did in ground_truth]
                        unique_hits = set(hits)

                        record[f"recall@{k}"] = len(unique_hits) / len(ground_truth)
                        record[f"partial_recall@{k}"] = 1.0 if unique_hits else 0.0
                        record[f"full_recall@{k}"] = 1.0 if ground_truth.issubset(unique_hits) else 0.0
                        record[f"precision@{k}"] = (len(hits) / len(top_k)) if top_k else 0.0
                        if hits:
                            first_idx = min(top_k.index(h) for h in hits)
                            record[f"mrr@{k}"] = 1.0 / (first_idx + 1)
                        else:
                            record[f"mrr@{k}"] = 0.0

                elif not q.is_answerable:
                    # Adversarial / Unanswerable queries
                    correct_rejection = len(retrieved_dia_ids) == 0
                    record["correct_rejection"] = 1.0 if correct_rejection else 0.0
                    record["false_positive"] = 0.0 if correct_rejection else 1.0

                per_query_results.append(record)

                if progress_callback:
                    progress_callback(s_idx, len(samples), q_idx, len(sample.questions))

        total_duration = time.perf_counter() - t_start
        summary = self._aggregate_metrics(per_query_results, unmappable_queries)
        summary["total_duration_seconds"] = round(total_duration, 2)
        summary["config_name"] = self.config_name
        return summary

    def _aggregate_metrics(
        self,
        per_query: List[Dict[str, Any]],
        unmappable: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate per-query results by category and overall."""
        def mean(nums: List[float]) -> float:
            return sum(nums) / len(nums) if nums else 0.0

        categories = sorted({q["category"] for q in per_query})
        category_breakdown: Dict[str, Any] = {}

        for cat in categories:
            cat_name = CATEGORY_NAMES.get(cat, f"Category {cat}")
            cat_queries = [q for q in per_query if q["category"] == cat]
            n_cat = len(cat_queries)

            if cat == 5:
                # Adversarial
                rej_rate = mean([q["correct_rejection"] for q in cat_queries])
                fp_rate = mean([q["false_positive"] for q in cat_queries])
                avg_ret = mean([float(q["n_retrieved"]) for q in cat_queries])
                category_breakdown[cat_name] = {
                    "category_id": cat,
                    "n_queries": n_cat,
                    "correct_rejection_rate": round(rej_rate, 4),
                    "false_positive_rate": round(fp_rate, 4),
                    "avg_returned_candidates": round(avg_ret, 3),
                }
            else:
                cat_summary: Dict[str, Any] = {
                    "category_id": cat,
                    "n_queries": n_cat,
                    "abstain_rate": round(mean([1.0 if q["abstained"] else 0.0 for q in cat_queries]), 4),
                }
                for k in self.k_values:
                    cat_summary[f"Recall@{k}"] = round(mean([q.get(f"recall@{k}", 0.0) for q in cat_queries]), 4)
                    cat_summary[f"PartialRecall@{k}"] = round(mean([q.get(f"partial_recall@{k}", 0.0) for q in cat_queries]), 4)
                    cat_summary[f"FullRecall@{k}"] = round(mean([q.get(f"full_recall@{k}", 0.0) for q in cat_queries]), 4)
                    cat_summary[f"Precision@{k}"] = round(mean([q.get(f"precision@{k}", 0.0) for q in cat_queries]), 4)
                    cat_summary[f"MRR@{k}"] = round(mean([q.get(f"mrr@{k}", 0.0) for q in cat_queries]), 4)
                category_breakdown[cat_name] = cat_summary

        # Overall Answerable (Cats 1-4)
        answerable = [q for q in per_query if q["is_answerable"]]
        overall_answerable: Dict[str, Any] = {
            "n_queries": len(answerable),
            "abstain_rate": round(mean([1.0 if q["abstained"] else 0.0 for q in answerable]), 4),
        }
        for k in self.k_values:
            overall_answerable[f"Recall@{k}"] = round(mean([q.get(f"recall@{k}", 0.0) for q in answerable]), 4)
            overall_answerable[f"PartialRecall@{k}"] = round(mean([q.get(f"partial_recall@{k}", 0.0) for q in answerable]), 4)
            overall_answerable[f"FullRecall@{k}"] = round(mean([q.get(f"full_recall@{k}", 0.0) for q in answerable]), 4)
            overall_answerable[f"Precision@{k}"] = round(mean([q.get(f"precision@{k}", 0.0) for q in answerable]), 4)
            overall_answerable[f"MRR@{k}"] = round(mean([q.get(f"mrr@{k}", 0.0) for q in answerable]), 4)

        # Overall Adversarial (Cat 5)
        adversarial = [q for q in per_query if not q["is_answerable"]]
        overall_adversarial = {
            "n_queries": len(adversarial),
            "correct_rejection_rate": round(mean([q["correct_rejection"] for q in adversarial]), 4) if adversarial else 0.0,
            "false_positive_rate": round(mean([q["false_positive"] for q in adversarial]), 4) if adversarial else 0.0,
        }

        return {
            "total_queries": len(per_query),
            "overall_answerable": overall_answerable,
            "overall_adversarial": overall_adversarial,
            "categories": category_breakdown,
            "unmappable_queries_count": len(unmappable),
            "unmappable_queries": unmappable,
        }


# ---------------------------------------------------------------------------
# Track B: End-to-End Memory Lifecycle Evaluator
# ---------------------------------------------------------------------------


class TrackBEvaluator:
    """Evaluates the end-to-end memory lifecycle via Memory.remember() and Memory.retrieve()."""

    def __init__(self, k: int = 5):
        self.k = k

    def evaluate(
        self,
        samples: List[ConversationSample],
        progress_callback=None,
    ) -> Dict[str, Any]:
        """Feed dialogue turns through Memory.remember() and evaluate QA retention."""
        from chunkdup.memory import Memory

        t_start = time.perf_counter()

        ingestion_stats = {
            "total_turns_fed": 0,
            "decisions": {"STORE": 0, "MERGE": 0, "UPDATE": 0, "IGNORE": 0},
            "total_stored_keys": 0,
        }

        lifecycle_breakdown = {
            "RETENTION_NEVER_EXTRACTED": 0,
            "RETENTION_IGNORED_LOW_SCORE": 0,
            "RETENTION_MERGED": 0,
            "RETENTION_OVERWRITTEN": 0,
            "RETRIEVAL_RANKED_TOO_LOW": 0,
            "RETRIEVAL_ABSTAINED": 0,
            "SUCCESS": 0,
        }

        per_question_results: List[Dict[str, Any]] = []

        for s_idx, sample in enumerate(samples):
            # Create fresh in-memory instance
            mem = Memory(store="memory", data_dir=tempfile.mkdtemp())

            # Track which turns produced which memories
            # dia_id -> list of decisions made
            turn_decisions: Dict[str, List[str]] = {}
            turn_stored_keys: Dict[str, List[str]] = {}

            # Feed turns chronologically
            for turn in sample.turns:
                text = f"{turn.speaker}: {turn.text}"
                res = mem.remember(text)
                ingestion_stats["total_turns_fed"] += 1

                decisions_for_turn = []
                keys_for_turn = []
                for m in res.get("memories", []):
                    dec = m.get("decision", "IGNORE").upper()
                    ingestion_stats["decisions"][dec] = ingestion_stats["decisions"].get(dec, 0) + 1
                    decisions_for_turn.append(dec)
                    if dec in ("STORE", "MERGE", "UPDATE") and m.get("key"):
                        keys_for_turn.append(m["key"])

                turn_decisions[turn.dia_id] = decisions_for_turn
                turn_stored_keys[turn.dia_id] = keys_for_turn

            stored_memories = mem.repository.get_all()
            ingestion_stats["total_stored_keys"] += len(stored_memories)
            active_keys = {m["key"] for m in stored_memories if m.get("key")}

            # Evaluate each question
            for q_idx, q in enumerate(sample.questions):
                if not q.is_answerable or not q.evidence_dia_ids:
                    continue

                # Run retrieval through Memory.retrieve
                retrieved = mem.retrieve(q.question, limit=self.k)
                retrieved_keys = {r.get("key") for r in retrieved if r.get("key")}

                # Analyze lifecycle for each required evidence turn
                for ev_dia_id in q.evidence_dia_ids:
                    decisions = turn_decisions.get(ev_dia_id, ["IGNORE"])
                    keys = turn_stored_keys.get(ev_dia_id, [])

                    if "STORE" not in decisions and "MERGE" not in decisions and "UPDATE" not in decisions:
                        status = "RETENTION_NEVER_EXTRACTED"
                    elif keys and not any(k in active_keys for k in keys):
                        status = "RETENTION_OVERWRITTEN"
                    elif keys and any(k in retrieved_keys for k in keys):
                        status = "SUCCESS"
                    elif keys and any(k in active_keys for k in keys):
                        if not retrieved:
                            status = "RETRIEVAL_ABSTAINED"
                        else:
                            status = "RETRIEVAL_RANKED_TOO_LOW"
                    else:
                        status = "RETENTION_NEVER_EXTRACTED"

                    lifecycle_breakdown[status] = lifecycle_breakdown.get(status, 0) + 1

                    per_question_results.append({
                        "sample_id": q.sample_id,
                        "question_idx": q.question_idx,
                        "category": q.category,
                        "category_name": q.category_name,
                        "evidence_dia_id": ev_dia_id,
                        "lifecycle_status": status,
                    })

                if progress_callback:
                    progress_callback(s_idx, len(samples), q_idx, len(sample.questions))

        total_eval_evidence = sum(lifecycle_breakdown.values())
        retention_failure_count = (
            lifecycle_breakdown["RETENTION_NEVER_EXTRACTED"]
            + lifecycle_breakdown["RETENTION_IGNORED_LOW_SCORE"]
            + lifecycle_breakdown["RETENTION_MERGED"]
            + lifecycle_breakdown["RETENTION_OVERWRITTEN"]
        )
        retrieval_failure_count = (
            lifecycle_breakdown["RETRIEVAL_RANKED_TOO_LOW"]
            + lifecycle_breakdown["RETRIEVAL_ABSTAINED"]
        )

        return {
            "duration_seconds": round(time.perf_counter() - t_start, 2),
            "ingestion_stats": ingestion_stats,
            "lifecycle_breakdown": lifecycle_breakdown,
            "total_evaluated_evidence_turns": total_eval_evidence,
            "retention_failure_count": retention_failure_count,
            "retention_failure_rate": round(retention_failure_count / total_eval_evidence, 4) if total_eval_evidence else 0.0,
            "retrieval_failure_count": retrieval_failure_count,
            "retrieval_failure_rate": round(retrieval_failure_count / total_eval_evidence, 4) if total_eval_evidence else 0.0,
            "success_count": lifecycle_breakdown["SUCCESS"],
            "success_rate": round(lifecycle_breakdown["SUCCESS"] / total_eval_evidence, 4) if total_eval_evidence else 0.0,
        }


# ---------------------------------------------------------------------------
# Runner & CLI Interface
# ---------------------------------------------------------------------------


def get_git_commit_hash() -> str:
    """Get current git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def print_scorecard(report: Dict[str, Any]):
    """Print an aligned ASCII scorecard of the LoCoMo benchmark results."""
    print("\n" + "=" * 78)
    print("  CHUNKDUP LoCoMo BENCHMARK SCORECARD")
    print("=" * 78)
    print(f"Dataset:       {DATASET_METADATA['name']} ({DATASET_METADATA['paper']})")
    print(f"Attribution:   {DATASET_METADATA['authors']}")
    print(f"License:       {DATASET_METADATA['license']}")
    print(f"Git Commit:    {report.get('commit_hash', 'unknown')}")
    print(f"Execution Date:{report.get('timestamp', 'unknown')}")
    print(f"Mode:          {'SMOKE TEST' if report.get('is_smoke') else 'FULL BENCHMARK'}")
    print("-" * 78)

    # Track A Table
    if "track_a" in report:
        ta = report["track_a"]
        print("\n[TRACK A: RETRIEVAL-ONLY PRODUCTION PIPELINE]")
        print(f"Config: {ta.get('config_name')} | Total Queries: {ta.get('total_queries')} | Duration: {ta.get('total_duration_seconds')}s")
        print("-" * 78)
        header = f"{'Category':<28} | {'N':<5} | {'P@5':<7} | {'R@5':<7} | {'Hit@5':<7} | {'FullR@5':<7} | {'MRR@5':<7}"
        print(header)
        print("-" * 78)
        for cat_name, metrics in ta.get("categories", {}).items():
            if metrics.get("category_id") == 5:
                # Adversarial
                print(f"{cat_name:<28} | {metrics['n_queries']:<5} | Rej: {metrics['correct_rejection_rate']:<5.4f} | FP: {metrics['false_positive_rate']:<5.4f} | AvgCandidates: {metrics['avg_returned_candidates']}")
            else:
                p5 = metrics.get("Precision@5", 0.0)
                r5 = metrics.get("Recall@5", 0.0)
                h5 = metrics.get("PartialRecall@5", 0.0)
                fr5 = metrics.get("FullRecall@5", 0.0)
                mrr5 = metrics.get("MRR@5", 0.0)
                print(f"{cat_name:<28} | {metrics['n_queries']:<5} | {p5:<7.4f} | {r5:<7.4f} | {h5:<7.4f} | {fr5:<7.4f} | {mrr5:<7.4f}")
        print("-" * 78)
        ans = ta.get("overall_answerable", {})
        print(f"{'OVERALL ANSWERABLE (Cats 1-4)':<28} | {ans.get('n_queries', 0):<5} | {ans.get('Precision@5', 0.0):<7.4f} | {ans.get('Recall@5', 0.0):<7.4f} | {ans.get('PartialRecall@5', 0.0):<7.4f} | {ans.get('FullRecall@5', 0.0):<7.4f} | {ans.get('MRR@5', 0.0):<7.4f}")
        adv = ta.get("overall_adversarial", {})
        print(f"{'OVERALL ADVERSARIAL (Cat 5)':<28} | {adv.get('n_queries', 0):<5} | Rej: {adv.get('correct_rejection_rate', 0.0):<5.4f} | FP: {adv.get('false_positive_rate', 0.0):<5.4f}")

    # Track B Table
    if "track_b" in report:
        tb = report["track_b"]
        print("\n[TRACK B: END-TO-END MEMORY LIFECYCLE]")
        print(f"Total Turns Fed: {tb['ingestion_stats']['total_turns_fed']} | Stored Keys: {tb['ingestion_stats']['total_stored_keys']}")
        print(f"Decisions Breakdown: {tb['ingestion_stats']['decisions']}")
        print("-" * 78)
        print(f"{'Failure / Outcome Type':<35} | {'Count':<8} | {'Percentage':<10}")
        print("-" * 78)
        tot = tb["total_evaluated_evidence_turns"]
        for k, count in tb["lifecycle_breakdown"].items():
            pct = (count / tot * 100.0) if tot else 0.0
            print(f"{k:<35} | {count:<8} | {pct:>6.2f}%")
        print("-" * 78)
        print(f"RETENTION FAILURE (Never stored):      {tb['retention_failure_count']} ({tb['retention_failure_rate']*100:.2f}%)")
        print(f"RETRIEVAL FAILURE (Ranked too low):    {tb['retrieval_failure_count']} ({tb['retrieval_failure_rate']*100:.2f}%)")
        print(f"END-TO-END SUCCESS (Stored & Found):   {tb['success_count']} ({tb['success_rate']*100:.2f}%)")
    print("=" * 78 + "\n")


def run_benchmark(
    data_path: str | Path = "evaluation/data/locomo10.json",
    out_path: str | Path = "evaluation/results/locomo_baseline.json",
    track: str = "all",
    smoke: bool = False,
    split: str = "all",
    config_name: str = "hybrid_ce_abstain",
    ce_timeout_s: float = 10.0,
) -> Dict[str, Any]:
    """Execute LoCoMo evaluation and save machine-readable results."""
    # Ensure offline mode if models cached
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("CHUNKDUP_CE_TIMEOUT_S", str(ce_timeout_s))

    data_path = Path(data_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading LoCoMo dataset from: {data_path} (smoke={smoke}, split={split})...")
    samples, validation_report = LoCoMoLoader.load(
        data_path,
        smoke=smoke,
        smoke_max_questions=10 if smoke else None,
        split=split,
    )
    print(f"Loaded {len(samples)} conversations, {validation_report['total_questions']} questions.")

    report: Dict[str, Any] = {
        "benchmark": "LoCoMo",
        "dataset_metadata": DATASET_METADATA,
        "commit_hash": get_git_commit_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "is_smoke": smoke,
        "split": split,
        "config_name": config_name,
        "validation_report": validation_report,
    }

    if track in ("A", "all"):
        print("\nRunning Track A (Retrieval-Only)...")
        evaluator_a = TrackAEvaluator(
            config_name=config_name,
            ce_timeout_s=ce_timeout_s,
        )
        report["track_a"] = evaluator_a.evaluate(samples)

    if track in ("B", "all"):
        print("\nRunning Track B (End-to-End Lifecycle)...")
        evaluator_b = TrackBEvaluator(k=5)
        report["track_b"] = evaluator_b.evaluate(samples)

    # Save results
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nMachine-readable baseline saved to: {out_path}")

    # Display console scorecard
    print_scorecard(report)
    return report


def main():
    parser = argparse.ArgumentParser(description="ChunkdUp LoCoMo Benchmark Adapter & Baseline Runner")
    parser.add_argument("--data", default="evaluation/data/locomo10.json", help="Path to locomo10.json")
    parser.add_argument("--out", default="evaluation/results/locomo_baseline.json", help="Output results path")
    parser.add_argument("--track", choices=["A", "B", "all"], default="all", help="Evaluation track to run")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test on 1 sample conversation")
    parser.add_argument("--split", choices=["all", "dev", "eval"], default="all", help="Dataset split (all, dev [2 convs], eval [8 convs])")
    parser.add_argument("--config", default="hybrid_ce_abstain", help="Retrieval config (hybrid_ce_abstain, dense_abstain, bm25, oracle)")
    parser.add_argument("--ce-timeout", type=float, default=10.0, help="Cross-encoder timeout in seconds")
    args = parser.parse_args()

    run_benchmark(
        data_path=args.data,
        out_path=args.out,
        track=args.track,
        smoke=args.smoke,
        split=args.split,
        config_name=args.config,
        ce_timeout_s=args.ce_timeout,
    )


if __name__ == "__main__":
    main()
