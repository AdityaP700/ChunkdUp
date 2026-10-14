# tests/test_comparative_eval.py
"""
Automated regression tests for the 6-strategy comparative evaluation,
failure classification, and failure-driven optimization.
"""

from pathlib import Path
import tempfile
import pytest

from chunkdup.memory import Memory
from chunkdup.profile.profile_memory import ProfileMemory
from chunkdup.relational.extractor import RelationalExtractor
from chunkdup.retrieval_production import ProductionRetriever
from evaluation.adapters.locomo_adapter import ConversationSample, DialogueTurn, QuestionItem
from evaluation.run_comparative_experiment import (
    ExperimentHarness,
    evaluate_strategy,
    classify_failure,
    extract_item_turn_ids,
)
from evaluation.run_ablations_and_optimization import OptimizedProductionRetriever


@pytest.fixture
def dummy_sample():
    turns = [
        DialogueTurn(dia_id="D1:1", speaker="Alice", text="I'm planning a vacation to Japan in April.", session_num=1, date_time="2026-04-01T10:00:00Z"),
        DialogueTurn(dia_id="D1:2", speaker="Bob", text="That sounds amazing! Have you booked flights?", session_num=1, date_time="2026-04-01T10:01:00Z"),
        DialogueTurn(dia_id="D2:1", speaker="Alice", text="Yes, I'm staying in Kyoto for two weeks.", session_num=2, date_time="2026-04-02T12:00:00Z"),
        DialogueTurn(dia_id="D2:2", speaker="Bob", text="Kyoto is wonderful in the spring.", session_num=2, date_time="2026-04-02T12:05:00Z"),
    ]
    questions = [
        QuestionItem(
            sample_id="dummy-1",
            question_idx=0,
            question="Where is Alice planning to vacation?",
            answer="Japan",
            category=1,
            category_name="Single-hop / Factual",
            raw_evidence=["D1:1"],
            evidence_dia_ids=["D1:1"],
            is_answerable=True,
        ),
        QuestionItem(
            sample_id="dummy-1",
            question_idx=1,
            question="How long is Alice staying in Kyoto?",
            answer="Two weeks",
            category=2,
            category_name="Temporal reasoning",
            raw_evidence=["D2:1"],
            evidence_dia_ids=["D2:1"],
            is_answerable=True,
        ),
        QuestionItem(
            sample_id="dummy-1",
            question_idx=2,
            question="Did Alice buy a nuclear submarine?",
            answer="None",
            category=5,
            category_name="Adversarial / Unanswerable",
            raw_evidence=[],
            evidence_dia_ids=[],
            is_answerable=False,
        ),
    ]
    return ConversationSample(
        sample_id="dummy-1",
        speaker_a="Alice",
        speaker_b="Bob",
        turns=turns,
        questions=questions,
    )


def test_turn_id_extraction():
    """Verify robust extraction of dialogue turn IDs across different item representations."""
    assert "D1:1" in extract_item_turn_ids({"dia_id": "D1:1"})
    assert "D2:5" in extract_item_turn_ids({"id": "conv-26:D2:5"})
    assert "D3:4" in extract_item_turn_ids({"source_turn_ids": ["D3:4"]})
    assert "D1:10" in extract_item_turn_ids({"contextual_text": 'In session 1, Alice said: "D1:10 text"'})


def test_failure_classification():
    """Verify failure taxonomy classification A through H."""
    # Class G: Adversarial query answered without abstention
    assert classify_failure(
        category=5,
        is_answerable=False,
        gt_turns=set(),
        retrieved_turns={"D1:1"},
        ranks=[1],
        abstained=False,
        best_f1=0.2,
    ) == "G"

    # Adversarial correct abstention -> None (Success)
    assert classify_failure(
        category=5,
        is_answerable=False,
        gt_turns=set(),
        retrieved_turns=set(),
        ranks=[],
        abstained=True,
        best_f1=0.0,
    ) is None

    # Class D: Partial multi-hop evidence retrieved
    assert classify_failure(
        category=3,
        is_answerable=True,
        gt_turns={"D1:1", "D2:1"},
        retrieved_turns={"D1:1"},
        ranks=[1],
        abstained=False,
        best_f1=0.8,
    ) == "D"

    # Full success: all evidence retrieved + F1 >= 0.5
    assert classify_failure(
        category=1,
        is_answerable=True,
        gt_turns={"D1:1"},
        retrieved_turns={"D1:1"},
        ranks=[1],
        abstained=False,
        best_f1=0.9,
    ) is None


def test_control_pipeline_non_zero_retrieval(dummy_sample):
    """Verify that Strategy A indexes dialogue turns and retrieves successfully (not scoring 0)."""
    harness = ExperimentHarness(dummy_sample)
    mem, retriever = harness.setup_strategy_a()

    assert len(mem.repository.get_all()) == 4
    metrics, records = evaluate_strategy("Strategy_A", (mem, retriever), dummy_sample.questions)

    assert metrics["hit_at_5"] > 0.0
    assert metrics["mrr"] > 0.0
    assert len(records) == 3


def test_optimized_retriever_adversarial_abstention(dummy_sample):
    """Verify that OptimizedProductionRetriever with calibrated threshold filters adversarial query."""
    temp_dir = tempfile.mkdtemp()
    mem = Memory(store="memory", data_dir=temp_dir, extractor="rules")
    for t in dummy_sample.turns:
        mem.repository.save({
            "id": f"{dummy_sample.sample_id}:{t.dia_id}",
            "dia_id": t.dia_id,
            "source_turn_id": t.dia_id,
            "key": f"turn_{t.dia_id}",
            "value": f"{t.speaker}: {t.text}",
            "contextual_text": f"In session {t.session_num}, {t.speaker}: {t.text}",
            "type": "dialogue_turn",
            "speaker": t.speaker,
            "session": t.session_num,
            "timestamp": t.date_time,
            "status": "active",
        })

    # High threshold (e.g. 5.0) forces abstention on unrelated query
    opt = OptimizedProductionRetriever(
        repository=mem.repository,
        sample_id=dummy_sample.sample_id,
        turns=dummy_sample.turns,
        ce_threshold=5.0,
        enable_multihop_expansion=True,
    )
    res = opt.retrieve("Did Alice buy a nuclear submarine?", k=5)
    # Should abstain (return empty results) on irrelevant query with strict threshold
    assert len(res) == 0
