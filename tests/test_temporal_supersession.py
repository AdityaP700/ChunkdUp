# tests/test_temporal_supersession.py
import pytest
from datetime import datetime, timezone
from chunkdup.memory import Memory
from chunkdup.in_memory_repository import InMemoryRepository
from chunkdup.turn_repository import InMemoryTurnRepository, SourceTurnLog


def test_raw_turn_saved_on_ignore():
    """Verify raw dialogue turns are permanently preserved even when extraction yields IGNORE."""
    memory = Memory(store="memory")

    # This chit-chat does not match any pattern or NER rule -> IGNORE
    raw_text = "Good morning! Hope you are having a wonderful day."
    res = memory.remember(
        raw_text,
        session_id="session-101",
        speaker="user",
        metadata={"client": "web-chat"},
    )

    # 1. Verification of remember result
    assert "turn_id" in res
    assert res["count"] == 1
    assert res["memories"][0]["decision"] == "IGNORE"

    # 2. Verification that turn is preserved in turn log
    turn = memory.get_turn(res["turn_id"])
    assert turn is not None
    assert turn["turn_id"] == res["turn_id"]
    assert turn["raw_text"] == raw_text
    assert turn["session_id"] == "session-101"
    assert turn["speaker"] == "user"
    assert turn["metadata"] == {"client": "web-chat"}

    # 3. Verification of session querying
    session_turns = memory.get_turns(session_id="session-101")
    assert len(session_turns) == 1
    assert session_turns[0]["turn_id"] == res["turn_id"]

    # 4. Invariant: 0 facts stored in active memory, but turn is safely kept
    assert len(memory.get_all()) == 0


def test_fact_update_creates_supersession_chain():
    """Verify that updating a fact creates an append-only new record and supersedes the old one."""
    memory = Memory(store="memory")

    # Step 1: Initial fact
    res1 = memory.remember("I use Python", speaker="user", session_id="dev-sess")
    assert res1["count"] == 1
    assert res1["memories"][0]["decision"] in ("STORE", "PROCESSED")

    fact1 = memory.repository.get_by_key("programming_language")
    assert fact1 is not None
    assert fact1["value"] == "Python"
    assert fact1["version"] == 1
    assert fact1["status"] == "active"
    assert fact1["valid_to"] is None
    assert fact1["supersedes_id"] is None
    assert fact1["source_turn_id"] == res1["turn_id"]
    fact1_id = fact1["id"]

    # Step 2: Update fact via new dialogue turn
    res2 = memory.remember("I now use Go", speaker="user", session_id="dev-sess")
    assert res2["count"] == 1
    assert res2["memories"][0]["decision"] == "UPDATE"

    # Active memory is now Go
    fact2 = memory.repository.get_by_key("programming_language")
    assert fact2 is not None
    assert fact2["value"] == "Go"
    assert fact2["version"] == 2
    assert fact2["status"] == "active"
    assert fact2["valid_to"] is None
    assert fact2["supersedes_id"] == fact1_id
    assert fact2["source_turn_id"] == res2["turn_id"]
    assert fact2["id"] != fact1_id

    # Inspect old row directly
    old_row = memory.repository._memories[fact1_id]
    assert old_row["status"] == "superseded"
    assert old_row["valid_to"] == fact2["valid_from"]
    assert old_row["superseded_by"] == fact2["id"]

    # History retrieval returns full chronologically ordered supersession chain
    history = memory.get_history("programming_language")
    assert len(history) == 2
    assert history[0]["version"] == 1
    assert history[0]["value"] == "Python"
    assert history[0]["superseded_by"] == fact2["id"]
    assert history[1]["version"] == 2
    assert history[1]["value"] == "Go"
    assert history[1]["supersedes_id"] == fact1_id


def test_point_in_time_querying():
    """Verify point-in-time querying accurately reconstructs historical state."""
    repo = InMemoryRepository()

    t0 = "2026-01-01T10:00:00Z"
    t_mid = "2026-01-01T15:00:00Z"
    t1 = "2026-01-02T10:00:00Z"
    t_after = "2026-01-02T15:00:00Z"

    # Store initial fact valid from t0
    repo.save({
        "id": "mem-1",
        "key": "editor",
        "value": "Neovim",
        "type": "tool",
        "valid_from": t0,
        "created_at": t0,
    })

    # Update fact valid from t1
    repo.update({
        "id": "mem-2",
        "key": "editor",
        "value": "VS Code",
        "type": "tool",
        "supersedes_id": "mem-1",
        "valid_from": t1,
    })

    # Query at t_mid (between t0 and t1) -> should see Neovim
    mem_mid = repo.get_by_key("editor", as_of=t_mid)
    assert mem_mid is not None
    assert mem_mid["value"] == "Neovim"
    assert mem_mid["id"] == "mem-1"

    # Query at t_after (after t1) -> should see VS Code
    mem_after = repo.get_by_key("editor", as_of=t_after)
    assert mem_after is not None
    assert mem_after["value"] == "VS Code"
    assert mem_after["id"] == "mem-2"

    # Query without as_of -> should see current active (VS Code)
    mem_now = repo.get_by_key("editor")
    assert mem_now is not None
    assert mem_now["value"] == "VS Code"

    # Query get_all point-in-time
    all_mid = repo.get_all(as_of=t_mid)
    assert len(all_mid) == 1
    assert all_mid[0]["value"] == "Neovim"

    all_after = repo.get_all(as_of=t_after)
    assert len(all_after) == 1
    assert all_after[0]["value"] == "VS Code"


def test_memory_api_point_in_time_retrieval():
    """Verify Memory SDK retrieve() respects as_of parameter."""
    memory = Memory(store="memory")

    t0 = "2026-01-01T10:00:00Z"
    t_mid = "2026-01-01T15:00:00Z"
    t1 = "2026-01-02T10:00:00Z"
    t_after = "2026-01-02T15:00:00Z"

    # Remember initial with timestamp t0
    memory.remember("I prefer Neovim", timestamp=t0)
    # Update with timestamp t1
    memory.update("editor", "VS Code", valid_from=t1)

    # Retrieval at t_mid
    results_mid = memory.retrieve("What is my editor?", as_of=t_mid)
    assert len(results_mid) > 0
    assert results_mid[0]["value"] == "Neovim"

    # Retrieval at t_after
    results_after = memory.retrieve("What is my editor?", as_of=t_after)
    assert len(results_after) > 0
    assert results_after[0]["value"] == "VS Code"

    # Grounded retrieval at t_mid
    grounded_mid = memory.retrieve_grounded("What is my editor?", as_of=t_mid)
    assert len(grounded_mid) > 0
    assert grounded_mid[0].memory["value"] == "Neovim"


def test_multi_version_supersession_lineage():
    """Verify 3 consecutive updates maintain an unbroken supersession chain."""
    memory = Memory(store="memory")

    memory.remember("I use Python")
    memory.remember("I now use Go")
    memory.remember("I now use Rust")

    history = memory.get_history("programming_language")
    assert len(history) == 3

    v1, v2, v3 = history[0], history[1], history[2]

    # v1 checks
    assert v1["version"] == 1
    assert v1["value"] == "Python"
    assert v1["status"] == "superseded"
    assert v1["supersedes_id"] is None
    assert v1["superseded_by"] == v2["id"]
    assert v1["valid_to"] == v2["valid_from"]

    # v2 checks
    assert v2["version"] == 2
    assert v2["value"] == "Go"
    assert v2["status"] == "superseded"
    assert v2["supersedes_id"] == v1["id"]
    assert v2["superseded_by"] == v3["id"]
    assert v2["valid_to"] == v3["valid_from"]

    # v3 checks
    assert v3["version"] == 3
    assert v3["value"] == "Rust"
    assert v3["status"] == "active"
    assert v3["supersedes_id"] == v2["id"]
    assert v3["superseded_by"] is None
    assert v3["valid_to"] is None

    # Current active fact
    current = memory.repository.get_by_key("programming_language")
    assert current["id"] == v3["id"]
    assert current["value"] == "Rust"
