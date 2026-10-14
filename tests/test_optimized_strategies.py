# tests/test_optimized_strategies.py
"""
Unit tests for ChunkdUp's Optimized Memory and Retrieval Components:
1. GraphAugmentedRetriever (Phase 2):
   - Tests candidate generation, 1-hop claim expansion, co-occurrence, and CE reranking.
2. ProfileGuidedArchiveRetriever (Phase 3):
   - Tests two-tier profile query expansion, episodic archive search, and bounded token context.
3. QueryRouter & OptimizedHybridMemory (Phase 4):
   - Tests deterministic routing rules and calibrated abstention gating.
"""

import tempfile
import pytest

from chunkdup.profile.profile_guided_retriever import ProfileGuidedArchiveRetriever
from chunkdup.profile.profile_memory import PersistentProfile
from chunkdup.query_router import QueryRouter
from chunkdup.relational.graph import RelationalGraph
from chunkdup.relational.schema import RelationalClaim
from chunkdup.retrieval_graph_augmented import GraphAugmentedRetriever
from chunkdup.hybrid_memory_optimized import OptimizedHybridMemory


def test_query_router_classification():
    router = QueryRouter()
    router.update_known_entities({"Alice", "Bob", "ChunkdUp"})

    # Adversarial
    assert router.route("What is the secret password for the spaceship?") == "adversarial"
    assert router.route("Does Alice have telekinesis superpowers?") == "adversarial"

    # Relational / Multi-hop
    assert router.route("What do Alice and Bob both have in common?") == "relational_multihop"
    assert router.route("What is the relationship between Alice and Bob?") == "relational_multihop"
    assert router.route("Who introduced Alice to Bob?") == "relational_multihop"

    # Profile / Attribute
    assert router.route("What is Alice's favorite hobby?") == "profile_attribute"
    assert router.route("Where does Bob live in?") == "profile_attribute"
    assert router.route("What is Alice's profession?") == "profile_attribute"

    # Factual / Episodic
    assert router.route("What did Alice say yesterday morning?") == "factual_episodic"
    assert router.route("When was the meeting scheduled?") == "factual_episodic"


def test_graph_augmented_retriever():
    retriever = GraphAugmentedRetriever(device="cpu")
    rel_graph = RelationalGraph()

    # Add relational claim
    rel_graph.add_claim(RelationalClaim(
        claim_id="claim_1",
        subject_entity="Gina",
        predicate="opened",
        object_entity="clothing store",
        source_turn_ids=["D1:5"],
        context_text="Gina opened her clothing store last month.",
    ))

    passages = [
        {"id": "D1:1", "dia_id": "D1:1", "text": "Gina: Hi Jon! Good to see you.", "speaker": "Gina", "session": "D1"},
        {"id": "D1:2", "dia_id": "D1:2", "text": "Jon: Hey Gina, how are things going?", "speaker": "Jon", "session": "D1"},
        {"id": "D1:5", "dia_id": "D1:5", "text": "Gina: I finally opened my own clothing boutique downtown!", "speaker": "Gina", "session": "D1"},
        {"id": "D1:6", "dia_id": "D1:6", "text": "Jon: Congratulations, that's a huge milestone!", "speaker": "Jon", "session": "D1"},
        {"id": "D2:1", "dia_id": "D2:1", "text": "Jon: What's the weather today?", "speaker": "Jon", "session": "D2"},
    ]

    retriever.index_passages(passages, relational_graph=rel_graph)

    # Retrieval for store
    results = retriever.retrieve("Where did Gina open her boutique?", top_k=3)
    assert len(results) > 0
    top_dia_ids = [r.get("dia_id") for r in results]
    assert "D1:5" in top_dia_ids


def test_profile_guided_archive_retriever():
    prof = PersistentProfile(max_facts=10)
    prof.set_fact("favorite_tool", "Neovim", category="preference", source_turn_id="D1:2")

    retriever = ProfileGuidedArchiveRetriever(profile=prof, device="cpu", max_context_tokens=150)
    turns = [
        {"id": "D1:1", "dia_id": "D1:1", "text": "User: What text editor do you recommend?"},
        {"id": "D1:2", "dia_id": "D1:2", "text": "User: I love using Neovim for Python programming."},
        {"id": "D1:3", "dia_id": "D1:3", "text": "User: Sometimes I also try VS Code."},
    ]
    retriever.index_turns(turns)

    res = retriever.retrieve("What is the preferred text editor?", top_k=2)
    assert not res["abstained"]
    assert len(res["results"]) > 0
    # Bounded prompt context should contain the profile fact and be short
    assert "[PROFILE FACTS]" in res["prompt_context"]
    assert len(res["prompt_context"]) < 500


def test_optimized_hybrid_memory():
    hybrid = OptimizedHybridMemory(device="cpu", adversarial_threshold=2.0)

    hybrid.remember(
        turn_id="D1:1",
        speaker="Gina",
        text="I decided to start my clothing store because I love fashion.",
        session_id="D1",
    )
    hybrid.remember(
        turn_id="D1:2",
        speaker="Jon",
        text="I also started my own design consultancy recently.",
        session_id="D1",
    )

    # Factual query
    res = hybrid.retrieve("Why did Gina start her clothing store?", top_k=2)
    assert not res["abstained"]
    assert len(res["results"]) > 0
    assert "clothing store" in res["results"][0]["text"].lower()

    # Adversarial query should abstain
    adv_res = hybrid.retrieve("Did Gina build a secret time machine in 1840?", top_k=2)
    assert adv_res["abstained"]
    assert len(adv_res["results"]) == 0
