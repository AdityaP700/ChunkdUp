# tests/test_locomo_adapter.py
"""Unit tests for LoCoMo adapter, validation, and evaluation tracks."""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import pytest
from pathlib import Path

from evaluation.adapters.locomo_adapter import (
    LoCoMoLoader,
    DialogueTurn,
    QuestionItem,
    ConversationSample,
    TrackAEvaluator,
    TrackBEvaluator,
    CATEGORY_NAMES,
)


def test_evidence_id_normalization():
    """Verify evidence normalization handles clean, space-separated, semicolon-separated, and typo formats."""
    # Standard format
    evs, is_mal, _ = LoCoMoLoader.parse_evidence_ids(["D1:5"])
    assert evs == ["D1:5"]
    assert not is_mal

    # Semicolon separated
    evs, is_mal, _ = LoCoMoLoader.parse_evidence_ids(["D8:6; D9:17"])
    assert evs == ["D8:6", "D9:17"]
    assert not is_mal

    # Space separated
    evs, is_mal, _ = LoCoMoLoader.parse_evidence_ids(["D9:1 D4:4 D4:6"])
    assert evs == ["D9:1", "D4:4", "D4:6"]
    assert not is_mal

    # Typo format (extra colon)
    evs, is_mal, _ = LoCoMoLoader.parse_evidence_ids(["D:11:26"])
    assert evs == ["D11:26"]
    assert not is_mal

    # Parentheses format
    evs, is_mal, _ = LoCoMoLoader.parse_evidence_ids(["(D2:3)"])
    assert evs == ["D2:3"]
    assert not is_mal

    # Unparseable malformed token
    evs, is_mal, reason = LoCoMoLoader.parse_evidence_ids(["D"])
    assert evs == []
    assert is_mal is True
    assert "Unparseable" in reason


def test_loader_validation(tmp_path):
    """Verify LoCoMoLoader validates missing fields, duplicates, and unsupported categories."""
    synthetic_data = [
        {
            "sample_id": "conv-test-1",
            "conversation": {
                "speaker_a": "Alice",
                "speaker_b": "Bob",
                "session_1_date_time": "10:00 AM on 1 Jan 2024",
                "session_1": [
                    {"dia_id": "D1:1", "speaker": "Alice", "text": "Hello Bob"},
                    {"dia_id": "D1:2", "speaker": "Bob", "text": "I like Neovim"},
                ],
            },
            "qa": [
                {
                    "question": "What editor does Bob like?",
                    "answer": "Neovim",
                    "category": 1,
                    "evidence": ["D1:2"],
                },
                {
                    "question": "Unsupported category question",
                    "answer": "Foo",
                    "category": 99,
                    "evidence": ["D1:1"],
                },
                {
                    "question": "Adversarial question",
                    "answer": "None",
                    "category": 5,
                    "evidence": ["D1:1"],
                },
            ],
        },
        {
            "sample_id": "conv-test-1",  # duplicate
            "conversation": {},
            "qa": [],
        },
    ]

    json_file = tmp_path / "test_locomo.json"
    json_file.write_text(json.dumps(synthetic_data), encoding="utf-8")

    samples, report = LoCoMoLoader.load(json_file)
    assert len(samples) == 1
    assert samples[0].sample_id == "conv-test-1"
    assert len(samples[0].turns) == 2
    # Question 2 had unsupported category 99, so only 2 valid questions kept
    assert len(samples[0].questions) == 2
    assert report["loaded_conversations"] == 1
    assert "conv-test-1" in report["duplicate_sample_ids"]
    assert len(report["unsupported_category_records"]) == 1
    assert report["unsupported_category_records"][0]["category"] == 99


def test_track_a_metrics_aggregation():
    """Verify Track A metric calculation for single-hop, multi-hop, and adversarial queries."""
    evaluator = TrackAEvaluator(k_values=(1, 5))

    # Mock per-query results
    mock_per_query = [
        # Cat 1: Single-hop answerable, hit at rank 1
        {
            "category": 1,
            "is_answerable": True,
            "abstained": False,
            "recall@1": 1.0, "partial_recall@1": 1.0, "full_recall@1": 1.0, "precision@1": 1.0, "mrr@1": 1.0,
            "recall@5": 1.0, "partial_recall@5": 1.0, "full_recall@5": 1.0, "precision@5": 0.2, "mrr@5": 1.0,
        },
        # Cat 3: Multi-hop answerable (requires 2 turns), 1 hit at rank 2
        {
            "category": 3,
            "is_answerable": True,
            "abstained": False,
            "recall@1": 0.0, "partial_recall@1": 0.0, "full_recall@1": 0.0, "precision@1": 0.0, "mrr@1": 0.0,
            "recall@5": 0.5, "partial_recall@5": 1.0, "full_recall@5": 0.0, "precision@5": 0.2, "mrr@5": 0.5,
        },
        # Cat 5: Adversarial, correctly abstained
        {
            "category": 5,
            "is_answerable": False,
            "n_retrieved": 0,
            "correct_rejection": 1.0,
            "false_positive": 0.0,
        },
        # Cat 5: Adversarial, false positive (returned 2 candidates)
        {
            "category": 5,
            "is_answerable": False,
            "n_retrieved": 2,
            "correct_rejection": 0.0,
            "false_positive": 1.0,
        },
    ]

    summary = evaluator._aggregate_metrics(mock_per_query, unmappable=[])
    assert summary["total_queries"] == 4

    # Cat 1 checks
    cat1 = summary["categories"][CATEGORY_NAMES[1]]
    assert cat1["Recall@5"] == 1.0
    assert cat1["MRR@5"] == 1.0

    # Cat 3 multi-hop checks
    cat3 = summary["categories"][CATEGORY_NAMES[3]]
    assert cat3["PartialRecall@5"] == 1.0  # partial recall succeeded
    assert cat3["FullRecall@5"] == 0.0     # full recall failed (only 1 of 2 retrieved)
    assert cat3["Recall@5"] == 0.5
    assert cat3["MRR@5"] == 0.5

    # Cat 5 adversarial checks
    cat5 = summary["categories"][CATEGORY_NAMES[5]]
    assert cat5["correct_rejection_rate"] == 0.5
    assert cat5["false_positive_rate"] == 0.5

    # Overall answerable vs adversarial separation
    ans = summary["overall_answerable"]
    assert ans["n_queries"] == 2
    assert ans["PartialRecall@5"] == 1.0
    assert ans["FullRecall@5"] == 0.5

    adv = summary["overall_adversarial"]
    assert adv["n_queries"] == 2
    assert adv["correct_rejection_rate"] == 0.5


def test_track_b_lifecycle_failure_separation():
    """Verify Track B separates ingestion/retention failures from retrieval ranking failures."""
    evaluator = TrackBEvaluator(k=5)

    sample = ConversationSample(
        sample_id="test-conv",
        speaker_a="Alice",
        speaker_b="Bob",
        turns=[
            DialogueTurn(
                dia_id="D1:1",
                speaker="Alice",
                text="Good morning Bob, lovely weather today.",
                session_num=1,
                date_time="morning",
            ),
            DialogueTurn(
                dia_id="D1:2",
                speaker="Bob",
                text="I prefer Neovim.",
                session_num=1,
                date_time="morning",
            ),
        ],
        questions=[
            # Question 1 asks about weather turn D1:1 (which rule extractor ignores)
            QuestionItem(
                sample_id="test-conv",
                question_idx=0,
                question="How was the weather?",
                answer="Lovely",
                category=1,
                category_name="Single-hop / Factual",
                raw_evidence=["D1:1"],
                evidence_dia_ids=["D1:1"],
                is_answerable=True,
            ),
            # Question 2 asks about editor turn D1:2 (which rule extractor stores)
            QuestionItem(
                sample_id="test-conv",
                question_idx=1,
                question="What editor does Bob prefer?",
                answer="Neovim",
                category=1,
                category_name="Single-hop / Factual",
                raw_evidence=["D1:2"],
                evidence_dia_ids=["D1:2"],
                is_answerable=True,
            ),
        ],
    )

    results = evaluator.evaluate([sample])
    breakdown = results["lifecycle_breakdown"]

    # D1:1 was chit-chat -> never extracted
    assert breakdown["RETENTION_NEVER_EXTRACTED"] >= 1
    # D1:2 matched "I prefer Neovim" -> stored in repo
    assert results["ingestion_stats"]["decisions"]["STORE"] >= 1
    assert results["total_evaluated_evidence_turns"] == 2


def test_locomo_splits(tmp_path):
    """Verify LoCoMoLoader splits dataset into dev (tuning) and eval (held-out) subsets."""
    sample_records = [
        {
            "sample_id": f"conv-{cid}",
            "conversation": {
                "session_1": [{"dia_id": "D1:1", "speaker": "A", "text": "Hi"}]
            },
            "qa": [
                {"question": "Q?", "answer": "A", "category": 1, "evidence": ["D1:1"]}
            ],
        }
        for cid in ["26", "30", "42", "43"]
    ]
    data_file = tmp_path / "split_test.json"
    data_file.write_text(json.dumps(sample_records), encoding="utf-8")

    # All split
    all_samples, _ = LoCoMoLoader.load(data_file, split="all")
    assert len(all_samples) == 4

    # Dev split (conv-26, conv-30)
    dev_samples, _ = LoCoMoLoader.load(data_file, split="dev")
    assert len(dev_samples) == 2
    assert {s.sample_id for s in dev_samples} == {"conv-26", "conv-30"}

    # Eval split (conv-42, conv-43)
    eval_samples, _ = LoCoMoLoader.load(data_file, split="eval")
    assert len(eval_samples) == 2
    assert {s.sample_id for s in eval_samples} == {"conv-42", "conv-43"}


def test_track_a_oracle_and_bm25():
    """Verify Track A execution with oracle (upper-bound ceiling) and bm25 configurations."""
    sample = ConversationSample(
        sample_id="conv-bench",
        speaker_a="Alice",
        speaker_b="Bob",
        turns=[
            DialogueTurn(
                dia_id="D1:1",
                speaker="Alice",
                text="Our database is PostgreSQL with pgvector.",
                session_num=1,
                date_time="morning",
            ),
            DialogueTurn(
                dia_id="D1:2",
                speaker="Bob",
                text="The embedding model is all-MiniLM-L6-v2.",
                session_num=1,
                date_time="morning",
            ),
            DialogueTurn(
                dia_id="D1:3",
                speaker="Alice",
                text="Let's grab some coffee and sandwiches.",
                session_num=1,
                date_time="morning",
            ),
            DialogueTurn(
                dia_id="D1:4",
                speaker="Bob",
                text="Sounds great, I will see you soon.",
                session_num=1,
                date_time="morning",
            ),
        ],
        questions=[
            QuestionItem(
                sample_id="conv-bench",
                question_idx=0,
                question="What database are we using?",
                answer="PostgreSQL",
                category=1,
                category_name="Single-hop / Factual",
                raw_evidence=["D1:1"],
                evidence_dia_ids=["D1:1"],
                is_answerable=True,
            ),
            QuestionItem(
                sample_id="conv-bench",
                question_idx=1,
                question="Are we using Redis?",
                answer="None",
                category=5,
                category_name="Adversarial / Unanswerable",
                raw_evidence=[],
                evidence_dia_ids=[],
                is_answerable=False,
            ),
        ],
    )

    # Test Oracle ceiling: full recall on answerable, correct rejection on adversarial
    oracle_eval = TrackAEvaluator(config_name="oracle", k_values=(1, 5))
    oracle_res = oracle_eval.evaluate([sample])
    assert oracle_res["overall_answerable"]["FullRecall@5"] == 1.0
    assert oracle_res["overall_answerable"]["Recall@5"] == 1.0
    assert oracle_res["overall_adversarial"]["correct_rejection_rate"] == 1.0
    assert oracle_res["overall_adversarial"]["false_positive_rate"] == 0.0

    # Test BM25 retrieval
    bm25_eval = TrackAEvaluator(config_name="bm25", k_values=(1, 5))
    bm25_res = bm25_eval.evaluate([sample])
    assert bm25_res["overall_answerable"]["Recall@5"] == 1.0

