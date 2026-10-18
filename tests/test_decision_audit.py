# tests/test_decision_audit.py
"""
Unit and integration tests for ChunkdUp persistent decision audit.
Verifies STORE, MERGE, UPDATE, IGNORE, process-restart durability,
duplicate retry safety, failed persistence tracking, and source traceability.
"""

import os
import json
import uuid
import tempfile
import pytest
from unittest.mock import MagicMock

from chunkdup.memory import Memory, MemoryManager, DecisionEngine, MemoryScorer
from chunkdup.in_memory_repository import InMemoryRepository
from chunkdup.policies import Decision
from chunkdup.audit import DecisionEvent


def test_decision_types_recorded():
    """Verify STORE, MERGE, UPDATE, and IGNORE are all recorded with full details."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mem = Memory(store="memory", data_dir=tmp_dir)

        # 1. STORE decision
        r1 = mem.remember("I use Python for backend services.", turn_id="turn-101")
        assert len(r1["memories"]) >= 1
        decisions_t1 = mem.repository.get_decisions_by_turn("turn-101")
        assert len(decisions_t1) >= 1
        d_store = decisions_t1[0]
        assert d_store["decision_type"] == "STORE"
        assert d_store["source_turn_id"] == "turn-101"
        assert d_store["status"] == "success"
        assert d_store["new_value"] is not None
        mem_id = d_store["target_memory_id"]
        assert mem_id is not None

        # 2. MERGE decision (same value mentions increase frequency)
        r2 = mem.remember("I use Python for backend services.", turn_id="turn-102")
        decisions_t2 = mem.repository.get_decisions_by_turn("turn-102")
        assert len(decisions_t2) >= 1
        d_merge = decisions_t2[0]
        assert d_merge["decision_type"] == "MERGE"
        assert d_merge["target_memory_id"] == mem_id
        assert d_merge["status"] == "success"

        # 3. UPDATE decision (value changed)
        r3 = mem.remember("I now use Go for backend services.", turn_id="turn-103")
        decisions_t3 = mem.repository.get_decisions_by_turn("turn-103")
        assert len(decisions_t3) >= 1
        d_update = decisions_t3[0]
        assert d_update["decision_type"] == "UPDATE"
        assert d_update["status"] == "success"
        assert d_update["previous_value"] == "Python"
        assert d_update["new_value"] == "Go"
        assert d_update["version"] >= 2

        # 4. IGNORE decision (non-fact / casual chat)
        r4 = mem.remember("Good morning! How are you?", turn_id="turn-104")
        decisions_t4 = mem.repository.get_decisions_by_turn("turn-104")
        assert len(decisions_t4) >= 1
        d_ignore = decisions_t4[0]
        assert d_ignore["decision_type"] == "IGNORE"
        assert d_ignore["source_turn_id"] == "turn-104"
        assert d_ignore["status"] == "success"
        assert d_ignore["rationale"] is not None


def test_ignore_distinguishable_from_unprocessed():
    """Verify an ignored message has a durable audit record unlike an unprocessed message."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = InMemoryRepository(data_dir=tmp_dir)
        engine = DecisionEngine()
        scorer = MemoryScorer()
        manager = MemoryManager(repo, engine, scorer, threshold=0.5)

        # Process a memory below threshold
        low_importance_mem = {
            "key": "casual_greeting",
            "value": "hello there",
            "type": "question",  # low weight in scorer
            "source_turn_id": "turn-ignored-1",
        }
        dec = manager.process(low_importance_mem)
        assert dec == Decision.IGNORE

        # Turn 1 was processed and IGNORED
        t1_decisions = repo.get_decisions_by_turn("turn-ignored-1")
        assert len(t1_decisions) == 1
        assert t1_decisions[0]["decision_type"] == "IGNORE"
        assert t1_decisions[0]["rationale"] == "below_importance_threshold"

        # Turn 2 was NEVER processed
        t2_decisions = repo.get_decisions_by_turn("turn-never-processed")
        assert len(t2_decisions) == 0


def test_process_restart_durability():
    """Verify decision audit log survives process restarts when backed by storage."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Session 1: Create repository and record decisions
        repo1 = InMemoryRepository(data_dir=tmp_dir)
        repo1.record_decision({
            "decision_id": "dec-uuid-1",
            "source_turn_id": "turn-restart-1",
            "decision_type": "STORE",
            "target_memory_id": "mem-1",
            "memory_key": "editor",
            "new_value": "Neovim",
            "version": 1,
            "status": "success",
        })
        repo1.record_decision({
            "decision_id": "dec-uuid-2",
            "source_turn_id": "turn-restart-2",
            "decision_type": "IGNORE",
            "memory_key": "chitchat",
            "status": "success",
            "rationale": "below_importance_threshold",
        })

        # Session 2: Fresh process / instance pointing to same directory
        repo2 = InMemoryRepository(data_dir=tmp_dir)
        d1 = repo2.get_decision("dec-uuid-1")
        assert d1 is not None
        assert d1["decision_type"] == "STORE"
        assert d1["new_value"] == "Neovim"

        d2 = repo2.get_decision("dec-uuid-2")
        assert d2 is not None
        assert d2["decision_type"] == "IGNORE"

        turn_recs = repo2.get_decisions_by_turn("turn-restart-1")
        assert len(turn_recs) == 1
        assert turn_recs[0]["decision_id"] == "dec-uuid-1"


def test_duplicate_retries_safe():
    """Verify duplicate turns with same turn_id do not create duplicate memory mutations."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mem = Memory(store="memory", data_dir=tmp_dir)

        # Call remember once
        r1 = mem.remember("I use Linux", turn_id="turn-idempotent-1")
        count_memories_1 = len(mem.repository.get_all())
        all_decisions_1 = mem.repository.get_all_decisions()

        # Call remember again with same turn_id
        r2 = mem.remember("I use Linux", turn_id="turn-idempotent-1")
        count_memories_2 = len(mem.repository.get_all())
        all_decisions_2 = mem.repository.get_all_decisions()

        # Should not duplicate memory records in storage
        assert count_memories_1 == count_memories_2
        assert len(all_decisions_2) >= len(all_decisions_1)


def test_failed_persistence_recorded():
    """Verify failed writes record a DecisionEvent with status='failed' and error info."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = InMemoryRepository(data_dir=tmp_dir)
        # Mock repository.save to raise an intentional disk / write failure
        repo.save = MagicMock(side_effect=IOError("Disk write simulated failure"))

        engine = DecisionEngine()
        scorer = MemoryScorer()
        manager = MemoryManager(repo, engine, scorer, threshold=0.1)

        fail_mem = {
            "key": "os",
            "value": "Linux",
            "type": "environment",
            "source_turn_id": "turn-fail-1",
        }

        with pytest.raises(IOError):
            manager.process(fail_mem)

        # Verify failure was durably recorded
        fail_decisions = repo.get_decisions_by_turn("turn-fail-1")
        assert len(fail_decisions) == 1
        assert fail_decisions[0]["status"] == "failed"
        assert fail_decisions[0]["decision_type"] == "STORE"
        assert "Disk write simulated failure" in fail_decisions[0]["error_message"]


def test_source_traceability():
    """Verify decisions can be looked up by decision_id, turn_id, and memory_id."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mem = Memory(store="memory", data_dir=tmp_dir)
        r = mem.remember("I prefer Neovim", turn_id="turn-trace-1")

        decisions = mem.repository.get_decisions_by_turn("turn-trace-1")
        assert len(decisions) >= 1
        d = decisions[0]
        decision_id = d["decision_id"]
        memory_id = d["target_memory_id"]

        # 1. Lookup by decision_id
        fetched_by_id = mem.repository.get_decision(decision_id)
        assert fetched_by_id is not None
        assert fetched_by_id["decision_id"] == decision_id

        # 2. Lookup by memory_id
        fetched_by_mem = mem.repository.get_decisions_by_memory(memory_id)
        assert len(fetched_by_mem) >= 1
        assert fetched_by_mem[0]["decision_id"] == decision_id
