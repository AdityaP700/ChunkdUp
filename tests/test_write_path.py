# tests/test_write_path.py
"""
Deterministic test suite for the complete ChunkdUp write path invariants:
1. Raw turn persistence before extraction
2. Resilience against failed/skipped/malformed extraction
3. Input-batch source provenance validation (reject fabricated/unknown turn IDs)
4. Rejection of empty, unknown, or fabricated provenance references
5. Rejection of speculative, hypothetical, and unsupported statements
6. Temporal history preservation on supersession
7. Idempotent reprocessing with identical turn_id
8. Partial batch failure recovery without duplication
9. Bounded retries and rate limit respect
10. Privacy-safe failure logging without content leakage
11. Raw persistence operational when Groq is unavailable
12. Raw turn preservation even under IGNORE policy
13. Entity linking synchronization in write path
"""

import time
import uuid
import pytest
from unittest.mock import MagicMock, patch
from pydantic import ValidationError

from chunkdup.memory import Memory
from chunkdup.extraction.schema import ExtractedFact, is_turn_eligible, is_speculative
from chunkdup.extraction.groq_extractor import GroqFactExtractor
from chunkdup.extraction.turn_buffer import TurnBuffer
from chunkdup.policies import Decision


def test_req1_raw_turn_persisted_before_extraction():
    """Invariant 1: Raw turn is persisted to turn repository before extraction occurs."""
    memory = Memory(store="memory")
    res = memory.remember("I am a distributed systems architect.", speaker="user")
    turn_id = res["turn_id"]
    saved_turn = memory.get_turn(turn_id)
    assert saved_turn is not None
    assert saved_turn["turn_id"] == turn_id
    assert saved_turn["raw_text"] == "I am a distributed systems architect."


def test_req2_failed_extraction_preserves_raw_turn():
    """Invariant 2: When extractor raises an exception, raw turn remains durably preserved."""
    memory = Memory(store="memory")
    # Force extractor to fail
    memory.extractor.extract = MagicMock(side_effect=RuntimeError("Extraction neural model crashed"))
    res = memory.remember("I use Go for network tools.", speaker="user")
    turn_id = res["turn_id"]
    saved_turn = memory.get_turn(turn_id)
    assert saved_turn is not None
    assert saved_turn["raw_text"] == "I use Go for network tools."


def test_req3_and_4_reject_fabricated_or_unknown_provenance():
    """Invariants 3 & 4: Facts referencing turn IDs not in the input batch are rejected."""
    extractor = GroqFactExtractor(api_key="mock-key")
    # Mock LLM returning a hallucinated turn_id not in the input batch
    mock_response = '''{
        "facts": [
            {
                "turn_id": "hallucinated-turn-999",
                "key": "cloud_provider",
                "value": "AWS",
                "type": "environment",
                "confidence": 0.95,
                "entities": ["AWS"]
            },
            {
                "turn_id": "valid-turn-1",
                "key": "editor",
                "value": "Neovim",
                "type": "tool",
                "confidence": 0.95,
                "entities": ["Neovim"]
            }
        ]
    }'''

    # Parse with valid_turn_ids restricted to valid-turn-1
    facts = extractor._parse_and_validate_response(
        mock_response,
        valid_turn_ids={"valid-turn-1"},
    )

    # Hallucinated turn must be rejected; valid turn must be accepted
    assert len(facts) == 1
    assert facts[0].source_turn_id == "valid-turn-1"
    assert facts[0].key == "editor"


def test_req4_reject_empty_provenance():
    """Invariant 4: Empty provenance reference raises ValidationError."""
    with pytest.raises(ValidationError):
        ExtractedFact(
            key="os",
            value="Linux",
            type="environment",
            source_turn_id="   ",  # Blank
        )


def test_req5_distinguish_explicit_facts_from_speculation():
    """Invariant 5: Speculative and hypothetical statements are filtered or rejected."""
    # 1. is_speculative helper correctly detects speculative triggers
    assert is_speculative("I might move to Seattle someday")
    assert is_speculative("Maybe I will learn Zig next year")
    assert is_speculative("If I ever switch from Python to Julia")
    assert is_speculative("Hypothetically speaking, I would use Arch")
    assert not is_speculative("I use Python")
    assert not is_speculative("I work at Stripe")

    # 2. is_turn_eligible rejects purely speculative turns
    assert not is_turn_eligible({"raw_text": "Maybe I will switch to Windows someday"})
    assert not is_turn_eligible({"raw_text": "I wonder if I should buy a Mac"})

    # 3. ExtractedFact validation rejects speculative values
    with pytest.raises(ValidationError):
        ExtractedFact(
            key="location",
            value="might move to London",
            type="general",
            source_turn_id="turn-1",
        )


def test_req6_preserve_temporal_supersession_history():
    """Invariant 6: Superseding a fact preserves historical records with valid_to closed."""
    memory = Memory(store="memory")
    t1 = "2026-01-01T00:00:00Z"
    t2 = "2026-06-01T00:00:00Z"

    memory.remember("I use Python.", timestamp=t1)
    orig_mem = memory.repository.get_by_key("programming_language")
    orig_id = orig_mem["id"]
    assert orig_mem["status"] == "active"
    assert orig_mem["valid_to"] is None

    memory.remember("I now use Rust.", timestamp=t2)
    new_mem = memory.repository.get_by_key("programming_language")
    assert new_mem["value"] == "Rust"
    assert new_mem["supersedes_id"] == orig_id

    # Check original memory in repository history
    old_mem = memory.repository._memories[orig_id]
    assert old_mem["status"] == "superseded"
    assert old_mem["valid_to"] == t2
    assert old_mem["superseded_by"] == new_mem["id"]


def test_req7_reprocessing_same_turn_is_idempotent():
    """Invariant 7: Reprocessing the exact same turn with same turn_id does not create duplicates."""
    memory = Memory(store="memory")
    turn_id = "deterministic-turn-1"

    res1 = memory.remember("I use Linux.", turn_id=turn_id)
    assert res1["turn_id"] == turn_id
    assert res1["memories"][0]["decision"] in ("STORE", "PROCESSED")
    first_mem = memory.repository.get_by_key("os")
    assert first_mem["frequency"] == 1

    # Re-process with identical turn_id
    res2 = memory.remember("I use Linux.", turn_id=turn_id)
    assert res2["turn_id"] == turn_id
    assert res2["memories"][0]["decision"] == "MERGE"

    # Only 1 active memory in store
    all_active = memory.get_all()
    assert len(all_active) == 1
    assert all_active[0]["frequency"] == 2


def test_req8_partial_batch_failures_recoverable_without_duplication():
    """Invariant 8: When one fact in a batch errors, other facts still persist cleanly."""
    memory = Memory(store="memory")
    buffer = TurnBuffer(
        turn_log=memory.turn_log,
        extractor=memory.groq_extractor,
        memory_manager=memory.manager,
        batch_size=2,
    )

    fact1 = ExtractedFact(
        key="editor",
        value="Neovim",
        type="tool",
        source_turn_id="turn-batch-1",
    )
    fact2 = ExtractedFact(
        key="failing_key",
        value="Error Value",
        type="general",
        source_turn_id="turn-batch-1",
    )
    fact3 = ExtractedFact(
        key="os",
        value="Linux",
        type="environment",
        source_turn_id="turn-batch-2",
    )

    # Mock extract_batch to return 3 facts
    buffer.extractor.extract_batch = MagicMock(return_value=[fact1, fact2, fact3])

    # Make memory_manager fail ONLY on failing_key
    orig_process = memory.manager.process
    def mock_process(m):
        if m.get("key") == "failing_key":
            raise ValueError("Database serialization glitch")
        return orig_process(m)

    memory.manager.process = mock_process

    buffer._buffer = [
        {"turn_id": "turn-batch-1", "raw_text": "I use Neovim."},
        {"turn_id": "turn-batch-2", "raw_text": "I use Linux."},
    ]

    results = buffer.flush()
    assert len(results) == 3
    assert results[0]["status"] == "success"
    assert results[0]["key"] == "editor"
    assert results[1]["status"] == "failed"
    assert results[1]["error"] == "ValueError"
    assert results[2]["status"] == "success"
    assert results[2]["key"] == "os"

    # Neovim and Linux are stored; failing_key is not
    assert memory.repository.get_by_key("editor") is not None
    assert memory.repository.get_by_key("os") is not None
    assert memory.repository.get_by_key("failing_key") is None


def test_req9_and_10_bounded_retries_and_privacy_safe_logging(caplog):
    """Invariants 9 & 10: Retries are bounded and sensitive turn text is not dumped in error logs."""
    client = MagicMock()
    # Mock persistent rate limit
    client.chat.completions.create.side_effect = RuntimeError("RateLimitError: 429 Too Many Requests")

    extractor = GroqFactExtractor(client=client, max_retries=2, backoff_factor=0.01)
    private_turn = {
        "turn_id": "turn-secret-1",
        "speaker": "user",
        "raw_text": "My highly confidential password or medical condition statement",
    }

    t_start = time.perf_counter()
    facts = extractor.extract_turn(private_turn)
    duration = time.perf_counter() - t_start

    # Retries must be bounded and return empty without hanging
    assert duration < 1.0
    assert len(facts) == 0

    # Privacy check: Private text must NOT appear in logged messages
    log_text = " ".join(record.message for record in caplog.records)
    assert "confidential password" not in log_text
    assert "medical condition" not in log_text


def test_req11_and_12_raw_persistence_when_groq_unavailable_and_ignore():
    """Invariants 11 & 12: Raw turn is persisted even when Groq is unavailable and policy ignores."""
    memory = Memory(store="memory", llm_provider="none")
    memory.groq_extractor._client = None
    memory.groq_extractor._api_key = None
    assert not memory.groq_extractor.is_available()

    # Ingest greeting that results in IGNORE
    res = memory.remember("Good morning! Lovely sunny day.", speaker="user")
    assert res["memories"][0]["decision"] == "IGNORE"

    # Turn must be saved
    saved_turn = memory.get_turn(res["turn_id"])
    assert saved_turn is not None
    assert saved_turn["raw_text"] == "Good morning! Lovely sunny day."
    assert len(memory.get_all()) == 0


def test_req13_entity_linking_synchronized_in_write_path():
    """Invariant 13: Stored facts are automatically indexed in Memory's EntityGraph."""
    memory = Memory(store="memory")
    assert memory.entity_graph is not None

    res = memory.remember("I use Docker with PostgreSQL.", speaker="user")
    assert res["memories"][0]["decision"] in ("STORE", "PROCESSED")

    # EntityGraph must have indexed the memory
    assert "PostgreSQL" in memory.entity_graph.entities or "Docker" in memory.entity_graph.entities
    # Both Memory and its ProductionRetriever share the same entity_graph
    assert memory.retriever.entity_graph is memory.entity_graph
