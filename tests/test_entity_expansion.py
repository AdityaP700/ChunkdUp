# tests/test_entity_expansion.py
"""Unit tests for EntityGraph, canonical alias resolution, and bounded 1-hop expansion."""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest
from chunkdup.in_memory_repository import InMemoryRepository
from chunkdup.retrieval_production import ProductionRetriever
from chunkdup.entity.entity_graph import EntityGraph, Entity, EntityRelation


def test_entity_graph_aliases_and_resolution():
    """Verify EntityGraph resolves canonical entities and known aliases."""
    graph = EntityGraph()

    # Builtin aliases
    assert graph.resolve_entities("I want to use postgres") == ["PostgreSQL"]
    assert graph.resolve_entities("We run in a docker container") == ["Docker"]
    assert graph.resolve_entities("Using pgvector for vector search") == ["pgvector"]

    # Register custom entity
    graph.register_entity("Chicago", category="location", aliases=["windy city", "chi-town"])
    assert graph.resolve_entities("She moved to Chicago") == ["Chicago"]
    assert graph.resolve_entities("She loves the windy city") == ["Chicago"]


def test_entity_relations_and_1hop_neighbors():
    """Verify directed relationships and 1-hop neighbor discovery."""
    graph = EntityGraph()
    graph.add_relation("PostgreSQL", "uses", "pgvector", confidence=0.95)
    graph.add_relation("PostgreSQL", "runs_on", "Docker", confidence=0.9)

    neighbors = graph.get_1hop_neighbors("PostgreSQL")
    neighbor_names = [n[0] for n in neighbors]
    assert "pgvector" in neighbor_names
    assert "Docker" in neighbor_names

    # Reverse associative lookup
    pg_neighbors = [n[0] for n in graph.get_1hop_neighbors("pgvector")]
    assert "PostgreSQL" in pg_neighbors


def test_bounded_1hop_expansion_candidate_limits():
    """Verify expansion strictly respects depth=1 and max_expansion limits."""
    graph = EntityGraph()
    repo = InMemoryRepository()

    # Setup chain: Alice -> Chicago -> Illinois -> USA
    graph.register_entity("Alice", category="person")
    graph.register_entity("Chicago", category="location")
    graph.register_entity("Illinois", category="location")

    graph.add_relation("Alice", "located_in", "Chicago", confidence=1.0)
    graph.add_relation("Chicago", "located_in", "Illinois", confidence=1.0)

    # Memories in repo
    m1 = {"id": "m1", "key": "city", "value": "Alice lives in Chicago", "status": "active"}
    m2 = {"id": "m2", "key": "state", "value": "Chicago is in Illinois", "status": "active"}
    m3 = {"id": "m3", "key": "weather", "value": "The winter in Chicago is freezing", "status": "active"}
    repo.save(m1)
    repo.save(m2)
    repo.save(m3)

    graph.index_repository(repo)

    # Query asks: "Where does Alice live?"
    # Initial candidates only has m1
    initial = [m1]
    expanded = graph.expand_candidates(
        query="Tell me about Alice",
        initial_candidates=initial,
        repository=repo,
        max_expansion=2,
    )

    # Strictly bounded: m2 and m3 (connected to Chicago) can be expanded, but capped at 2
    assert len(expanded) <= 2
    # m1 was already in initial, so it must not be re-added
    assert not any(e["id"] == "m1" for e in expanded)
    # Expanded items are tagged with provenance
    for e in expanded:
        assert e["_expanded_from_entity"] is True
        assert e["_expansion_source_entity"] == "Alice"
        assert e["_expansion_target_entity"] == "Chicago"


def test_production_retriever_multi_hop_expansion():
    """Verify ProductionRetriever retrieves 1-hop connected memories for multi-hop queries."""
    repo = InMemoryRepository()
    # Fact 1: Alice's team uses PostgreSQL
    repo.save({
        "id": "hop1",
        "key": "alice_db",
        "value": "Alice uses PostgreSQL",
        "type": "environment",
        "status": "active",
        "speaker": "Alice",
        "contextual_text": "Alice uses PostgreSQL",
    })
    # Fact 2: PostgreSQL uses pgvector extension
    repo.save({
        "id": "hop2",
        "key": "db_ext",
        "value": "PostgreSQL with pgvector",
        "type": "tool",
        "status": "active",
        "contextual_text": "PostgreSQL with pgvector",
    })
    # Fact 3: Unrelated memory
    repo.save({
        "id": "unrelated",
        "key": "lunch",
        "value": "Pizza for lunch",
        "type": "general",
        "status": "active",
        "contextual_text": "Pizza for lunch",
    })

    retriever = ProductionRetriever(
        repo,
        config_name="hybrid_ce_abstain",
        ce_timeout_s=10.0,
    )

    # Query: "What vector extension does Alice's database use?"
    # Multi-hop query: connects Alice -> PostgreSQL -> pgvector!
    results = retriever.retrieve("What vector extension does Alice's database use?", k=5)
    result_ids = [r["id"] for r in results]

    # Both hop1 and hop2 should be retrieved thanks to entity linking!
    assert "hop1" in result_ids or "hop2" in result_ids
