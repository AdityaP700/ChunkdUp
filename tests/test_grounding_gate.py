# tests/test_grounding_gate.py
"""Unit tests for GroundingGate, calibrated abstention, and typed retrieval results."""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest
from chunkdup.memory import Memory
from chunkdup.in_memory_repository import InMemoryRepository
from chunkdup.retrieval_production import (
    ProductionRetriever,
    FROZEN_CE_THRESHOLD,
    FROZEN_COSINE_THRESHOLD,
)
from chunkdup.retrieval.grounding_gate import (
    GroundingGate,
    GroundedRetrievalResult,
)


def test_grounded_retrieval_result_dict_interface():
    """Verify GroundedRetrievalResult provides typed fields and 100% dict backward-compatibility."""
    memory_dict = {
        "id": "mem-1",
        "key": "editor",
        "value": "Neovim",
        "type": "tool",
        "speaker": "Alice",
    }
    result = GroundedRetrievalResult(
        id="mem-1",
        memory=memory_dict,
        score=2.5,
        score_field="_rerank_score",
        source_turns=["D1:2"],
        confidence=0.95,
        is_sufficient=True,
        contradictions=[],
        speaker_aligned=True,
    )

    # Attribute access
    assert result.id == "mem-1"
    assert result.score == 2.5
    assert result.source_turns == ["D1:2"]
    assert result.is_sufficient is True

    # Dict-like access
    assert result["key"] == "editor"
    assert result["value"] == "Neovim"
    assert result.get("speaker") == "Alice"
    assert result.get("nonexistent", "fallback") == "fallback"
    assert "key" in result
    assert "source_turns" in result

    # Serialization
    d = result.to_dict()
    assert isinstance(d, dict)
    assert d["key"] == "editor"
    assert d["_source_turns"] == ["D1:2"]
    assert d["_is_sufficient"] is True
    assert d["_confidence"] == 0.95


def test_grounding_gate_scale_aware_threshold():
    """Verify GroundingGate scales abstention threshold with corpus size N to prevent false acceptance."""
    gate_ce = GroundingGate(
        score_field="_rerank_score",
        threshold=FROZEN_CE_THRESHOLD,
        enable_scale_calibration=True,
    )

    # For small stores (N <= 20), preserves exact frozen base threshold
    assert gate_ce.get_calibrated_threshold(store_size=12) == FROZEN_CE_THRESHOLD
    assert gate_ce.get_calibrated_threshold(store_size=20) == FROZEN_CE_THRESHOLD

    # For larger stores (N = 100, N = 500), increases threshold logarithmically
    tau_100 = gate_ce.get_calibrated_threshold(store_size=100)
    tau_500 = gate_ce.get_calibrated_threshold(store_size=500)
    assert tau_100 > FROZEN_CE_THRESHOLD
    assert tau_500 > tau_100
    assert tau_500 <= -2.5

    # Dense similarity gate scaling
    gate_dense = GroundingGate(
        score_field="_similarity",
        threshold=FROZEN_COSINE_THRESHOLD,
        enable_scale_calibration=True,
    )
    assert gate_dense.get_calibrated_threshold(store_size=12) == FROZEN_COSINE_THRESHOLD
    assert gate_dense.get_calibrated_threshold(store_size=500) > FROZEN_COSINE_THRESHOLD


def test_grounding_gate_speaker_alignment_and_contradictions():
    """Verify speaker alignment verification and contradiction veto."""
    gate = GroundingGate(
        score_field="_rerank_score",
        threshold=-10.0,
        enable_scale_calibration=False,
    )

    # Speaker target extraction
    assert gate.extract_query_speaker("What did Alice say about databases?") == "Alice"
    assert gate.extract_query_speaker("What is Bob's favorite editor?") == "Bob"
    assert gate.extract_query_speaker("What database is used?") is None

    # Candidate provenance extraction
    cand1 = {
        "id": "conv-42:D2:15",
        "key": "editor",
        "value": "Neovim",
        "speaker": "Alice",
        "_rerank_score": 1.2,
    }
    cand2 = {
        "id": "conv-42:D2:18",
        "key": "editor",
        "value": "VS Code",
        "speaker": "Bob",
        "_rerank_score": 0.8,
        "status": "superseded",
        "superseded_by": "conv-42:D2:15",
    }

    # Query specifically asking about Alice
    results = gate.filter_and_ground("What did Alice say?", [cand1, cand2], store_size=20)
    assert len(results) == 2

    # cand1 matches Alice, sufficient
    r1 = results[0]
    assert r1.id == "conv-42:D2:15"
    assert r1.source_turns == ["D2:15"]
    assert r1.speaker_aligned is True
    assert r1.is_sufficient is True

    # cand2 is Bob, superseded, contradictory
    r2 = results[1]
    assert r2.id == "conv-42:D2:18"
    assert r2.is_sufficient is False
    assert len(r2.contradictions) >= 1


def test_grounding_gate_abstention_on_unanswerable():
    """Verify GroundingGate rejects candidates when top candidate fails calibrated threshold."""
    gate = GroundingGate(
        score_field="_rerank_score",
        threshold=-5.0,
        enable_scale_calibration=False,
    )
    weak_candidates = [
        {"id": "m1", "value": "unrelated statement", "_rerank_score": -8.5},
        {"id": "m2", "value": "another unrelated memory", "_rerank_score": -9.2},
    ]

    filtered = gate.filter_and_ground("Is Redis used?", weak_candidates, store_size=50)
    assert filtered == []


def test_production_retriever_grounded_integration():
    """Verify ProductionRetriever produces enriched dicts and typed GroundedRetrievalResults."""
    repo = InMemoryRepository()
    repo.save({
        "id": "conv-1:D1:1",
        "key": "database",
        "value": "PostgreSQL with pgvector",
        "type": "environment",
        "speaker": "Alice",
        "source": "Session 1 D1:1",
        "contextual_text": "In session 1, Alice said: PostgreSQL with pgvector",
        "status": "active",
    })

    retriever = ProductionRetriever(repo, config_name="hybrid_ce_abstain", ce_timeout_s=10.0)

    # Standard retrieve() returns list of enriched dicts
    dict_results = retriever.retrieve("What database?", k=1)
    assert len(dict_results) == 1
    top = dict_results[0]
    assert isinstance(top, dict)
    assert top["key"] == "database"
    assert "_source_turns" in top
    assert "_is_sufficient" in top
    assert top["_is_sufficient"] is True
    assert "D1:1" in top["_source_turns"]

    # retrieve_grounded() returns GroundedRetrievalResult
    grounded_results = retriever.retrieve_grounded("What database?", k=1)
    assert len(grounded_results) == 1
    g_top = grounded_results[0]
    assert isinstance(g_top, GroundedRetrievalResult)
    assert g_top.id == "conv-1:D1:1"
    assert g_top.is_sufficient is True
    assert "D1:1" in g_top.source_turns


def test_memory_sdk_grounded_retrieval():
    """Verify public Memory SDK supports retrieve_grounded seamlessly."""
    mem = Memory(store="memory")
    mem.remember("I prefer Neovim")

    # retrieve() returns dicts
    results = mem.retrieve("What editor?")
    assert len(results) >= 1
    assert results[0]["value"] == "Neovim"
    assert "_is_sufficient" in results[0]

    # retrieve_grounded() returns GroundedRetrievalResult
    grounded = mem.retrieve_grounded("What editor?")
    assert len(grounded) >= 1
    assert isinstance(grounded[0], GroundedRetrievalResult)
    assert grounded[0]["value"] == "Neovim"
    assert grounded[0].is_sufficient is True
