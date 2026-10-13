# tests/test_audit_lifecycle.py
"""
Audit verification tests for end-to-end memory lifecycle invariants:
- Raw turn persistence on IGNORE
- Source-turn ID provenance linkage
- Rejection of invalid/malformed provider outputs
- Idempotent reprocessing (MERGE behavior)
- Temporal supersession and point-in-time querying
- Non-blocking ingestion when optional external providers fail
"""

import os
import uuid
import pytest
from unittest.mock import MagicMock
from pydantic import ValidationError

from chunkdup.memory import Memory
from chunkdup.extraction.schema import ExtractedFact, is_turn_eligible
from chunkdup.extraction.groq_extractor import GroqFactExtractor
from chunkdup.extraction.turn_buffer import TurnBuffer
from chunkdup.decision_provider import ExternalDecisionProvider, Decision


def test_audit_lifecycle_trace_and_invariants():
    """
    Traces a complete multi-turn conversation and validates invariants at each stage:
    Turn 1 (Chit-chat -> IGNORE): Raw turn preserved; 0 active memories.
    Turn 2 (Fact statement -> STORE): Active memory stored with valid source_turn_id.
    Turn 3 (Fact update -> UPDATE): Prior fact superseded; history chain unbroken.
    Turn 4 (Repeat statement -> MERGE): Idempotent reprocessing; frequency incremented.
    """
    memory = Memory(store="memory")
    t0 = "2026-02-01T10:00:00Z"
    t1 = "2026-02-01T11:00:00Z"
    t2 = "2026-02-01T12:00:00Z"

    # --- Turn 1: Chit-chat ---
    turn1_text = "Good morning! Hope you're doing great."
    res1 = memory.remember(turn1_text, speaker="user", session_id="audit-sess-1", timestamp=t0)
    assert res1["count"] == 1
    assert res1["memories"][0]["decision"] == "IGNORE"

    # Invariant: Raw turn must survive IGNORE
    saved_turn1 = memory.get_turn(res1["turn_id"])
    assert saved_turn1 is not None
    assert saved_turn1["raw_text"] == turn1_text
    assert len(memory.get_all()) == 0

    # --- Turn 2: Fact statement ---
    turn2_text = "I use Python for backend microservices."
    res2 = memory.remember(turn2_text, speaker="user", session_id="audit-sess-1", timestamp=t1)
    assert res2["memories"][0]["decision"] in ("STORE", "PROCESSED")

    # Invariant: Extracted fact must have valid source_turn_id referencing turn 2
    mem2 = memory.repository.get_by_key("programming_language")
    assert mem2 is not None
    assert mem2["value"] == "Python"
    assert mem2["source_turn_id"] == res2["turn_id"]
    assert mem2["valid_from"] == t1
    assert mem2["valid_to"] is None
    assert mem2["status"] == "active"
    mem2_id = mem2["id"]

    # --- Turn 3: Fact update ---
    turn3_text = "I now use Rust for backend microservices."
    res3 = memory.remember(turn3_text, speaker="user", session_id="audit-sess-1", timestamp=t2)
    assert res3["memories"][0]["decision"] == "UPDATE"

    # Invariant: Prior memory superseded, new memory references supersedes_id
    mem3 = memory.repository.get_by_key("programming_language")
    assert mem3 is not None
    assert mem3["value"] == "Rust"
    assert mem3["id"] != mem2_id
    assert mem3["supersedes_id"] == mem2_id
    assert mem3["source_turn_id"] == res3["turn_id"]

    old_mem2 = memory.repository._memories[mem2_id]
    assert old_mem2["status"] == "superseded"
    assert old_mem2["superseded_by"] == mem3["id"]
    assert old_mem2["valid_to"] == t2

    # --- Turn 4: Idempotent reprocessing (Repeat statement -> MERGE) ---
    res4 = memory.remember(turn3_text, speaker="user", session_id="audit-sess-1")
    assert res4["memories"][0]["decision"] == "MERGE"

    # Stored memory still has value Rust, frequency incremented to 2
    mem4 = memory.repository.get_by_key("programming_language")
    assert mem4["value"] == "Rust"
    assert mem4["id"] == mem3["id"]
    assert mem4["frequency"] == 2

    # --- Point-in-time Querying ---
    # Query as of t1 (between turn 2 and turn 3) -> should reconstruct Python
    pit_result = memory.retrieve("What programming language do I use?", as_of="2026-02-01T11:30:00Z")
    assert len(pit_result) > 0
    assert pit_result[0]["value"] == "Python"

    # Query without as_of -> returns Rust
    curr_result = memory.retrieve("What programming language do I use?")
    assert len(curr_result) > 0
    assert curr_result[0]["value"] == "Rust"


def test_invalid_provider_output_rejected():
    """Verify malformed or invalid provider outputs cannot become stored facts."""
    # 1. Missing source_turn_id
    with pytest.raises(ValidationError):
        ExtractedFact(
            key="editor",
            value="Neovim",
            type="tool",
            source_turn_id="",  # Empty
        )

    # 2. Invalid memory type
    with pytest.raises(ValidationError):
        ExtractedFact(
            key="editor",
            value="Neovim",
            type="non_existent_type",
            source_turn_id="turn-xyz",
        )

    # 3. Invalid key with whitespace
    with pytest.raises(ValidationError):
        ExtractedFact(
            key="my editor",
            value="Neovim",
            type="tool",
            source_turn_id="turn-xyz",
        )


def test_optional_provider_failure_does_not_block_raw_turn():
    """Verify that when an external decision provider errors/times out, raw turns are still saved."""
    memory = Memory(store="memory")

    # Mock external provider that always raises an error
    failing_provider = ExternalDecisionProvider(
        callback=MagicMock(side_effect=RuntimeError("Provider connection timed out"))
    )
    memory.manager.decision_provider = failing_provider

    # Ingest turn
    res = memory.remember("I use Go for network tools.", speaker="user")
    assert "turn_id" in res

    # Raw turn must still be safely stored in the turn log
    saved_turn = memory.get_turn(res["turn_id"])
    assert saved_turn is not None
    assert saved_turn["raw_text"] == "I use Go for network tools."
