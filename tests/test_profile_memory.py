# tests/test_profile_memory.py
"""
Tests for GPT-Inspired 3-Tier Profile, Recent Context, and Archive Memory.
Verifies 4-way consolidation (ADD, UPDATE, DELETE, NOOP), capacity bounding,
hot-path query serving vs archive fallbacks, and fact invalidation.
"""

import pytest

from chunkdup.profile.profile_memory import (
    PersistentProfile,
    ProfileConsolidationEngine,
    ProfileMemory,
    RecentContextBuffer,
)


def test_profile_consolidation_decisions():
    """Verify 4-way consolidation decisions: ADD, UPDATE, DELETE, NOOP."""
    memory = ProfileMemory(consolidation_mode="rules")

    # 1. ADD: Expressed preference
    t1 = {"turn_id": "t1", "speaker": "User", "text": "I prefer Neovim for daily coding."}
    d1 = memory.remember(t1)
    assert any(item["action"] == "ADD" for item in d1)
    fact = memory.profile.get_fact("editor_preference")
    assert fact is not None
    assert "Neovim" in fact.value

    # 2. UPDATE: Switched preference / migration
    t2 = {"turn_id": "t2", "speaker": "User", "text": "I switched from Neovim to VS Code."}
    d2 = memory.remember(t2)
    assert any(item["action"] in ("ADD", "UPDATE") for item in d2)

    # 3. NOOP: Ephemeral chat / questions
    t3 = {"turn_id": "t3", "speaker": "User", "text": "Hello there! Can you explain how quicksort works?"}
    d3 = memory.remember(t3)
    assert any(item["action"] == "NOOP" for item in d3)

    # 4. DELETE: Explicit invalidation
    t4 = {"turn_id": "t4", "speaker": "User", "text": "I stopped using VS Code."}
    d4 = memory.remember(t4)
    assert any(item["action"] == "DELETE" for item in d4)
    # The invalidated preference must no longer be present
    assert memory.profile.get_fact("tech_stack") is None


def test_profile_capacity_bounding_and_eviction():
    """Verify that persistent profile strictly bounds size and evicts oldest entries."""
    max_facts = 3
    profile = PersistentProfile(max_facts=max_facts)

    profile.set_fact("fact_1", "Value 1")
    profile.set_fact("fact_2", "Value 2")
    profile.set_fact("fact_3", "Value 3")
    assert len(profile.facts) == 3

    # Add 4th fact: must evict the first fact (LRU)
    profile.set_fact("fact_4", "Value 4")
    assert len(profile.facts) == 3
    assert profile.get_fact("fact_4") is not None
    assert profile.get_fact("fact_1") is None  # Evicted


def test_hot_path_query_serving_vs_archive_fallback():
    """
    Verify hot-path serving:
    - Query answerable from profile is served immediately WITHOUT searching deep archive.
    - Query absent from profile falls back to deep episodic archive.
    """
    memory = ProfileMemory()

    # Seed profile with core user facts
    memory.remember({"turn_id": "t1", "speaker": "User", "text": "I am building ChunkdUp."})
    memory.remember({"turn_id": "t2", "speaker": "User", "text": "I prefer Python."})

    # Episodic detail stored only in dialogue archive
    memory.remember({
        "turn_id": "t3",
        "speaker": "User",
        "text": "Yesterday afternoon our customer deployment in Frankfurt experienced high network latency."
    })

    # 1. Profile Hot-Path Query: "What language do I prefer?"
    res_hot = memory.retrieve("What language do I prefer?")
    assert res_hot["source"] == "profile_hot_path"
    assert res_hot["archive_searched"] is False
    assert "Python" in res_hot["fact"]["value"]

    # 2. Profile Hot-Path Query: "What is my current project?"
    res_proj = memory.retrieve("What is my current project?")
    assert res_proj["source"] == "profile_hot_path"
    assert res_proj["archive_searched"] is False
    assert "ChunkdUp" in res_proj["fact"]["value"]

    # 3. Archive Fallback Query: "What happened with the customer deployment in Frankfurt?"
    res_archive = memory.retrieve("What happened with the customer deployment in Frankfurt?")
    assert res_archive["source"] == "archive"
    assert res_archive["archive_searched"] is True
    assert len(res_archive["passages"]) >= 1
    assert "Frankfurt" in res_archive["passages"][0]["text"]


def test_recent_context_rolling_summary():
    """Verify rolling context window and session summary."""
    buffer = RecentContextBuffer(window_size=2)

    buffer.add_turn({"speaker": "User", "text": "Let's review the memory design."})
    buffer.add_turn({"speaker": "Assistant", "text": "I have prepared the schema."})
    assert len(buffer.turns) == 2
    assert buffer.rolling_summary == ""

    # Adding 3rd turn rolls the 1st turn into summary
    buffer.add_turn({"speaker": "User", "text": "Can we test it now?"})
    assert len(buffer.turns) == 2
    assert "review the memory design" in buffer.rolling_summary

    ctx = buffer.to_prompt_context()
    assert "[CONVERSATION SUMMARY]" in ctx
    assert "[RECENT TURNS]" in ctx
