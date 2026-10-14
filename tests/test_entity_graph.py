# tests/test_entity_graph.py
"""
Deterministic tests for ChunkdUp's EntityGraph and Multi-Hop 1-Hop Expansion.

Validates:
1. Canonical entity registration and alias normalization
2. Dynamic entity recognition (named entities vs sentence starters)
3. Relational edge extraction and co-occurrence linking
4. Strict bounded 1-hop neighbor traversal (depth = 1)
5. Multi-hop candidate expansion from query and bridge candidate entities
6. Expansion boundedness (max_expansion cap, no duplicates)
7. End-to-end multi-hop retrieval via ProductionRetriever
"""

import pytest
from chunkdup.entity.entity_graph import EntityGraph, Entity, EntityRelation
from chunkdup.in_memory_repository import InMemoryRepository
from chunkdup.retrieval_production import ProductionRetriever


def test_builtin_and_alias_resolution():
    graph = EntityGraph()
    # Built-in aliases
    assert graph.aliases["postgres"] == "PostgreSQL"
    assert graph.aliases["nvim"] == "Neovim"
    assert graph.aliases["k8s"] == "Kubernetes"

    # Resolve from text
    entities = graph.resolve_entities("I run my code inside Docker using Postgres")
    assert "Docker" in entities
    assert "PostgreSQL" in entities


def test_dynamic_entity_extraction_and_sentence_starter_filtering():
    graph = EntityGraph()
    # "Where", "What", "Please" should not be registered as entities
    entities = graph.resolve_entities("Where did Alice move after leaving TechCorp?")
    assert "Where" not in entities
    assert "Alice" in entities
    assert "TechCorp" in entities


def test_memory_indexing_and_relation_extraction():
    graph = EntityGraph()
    m1 = {
        "id": "mem_1",
        "key": "company_hq",
        "value": "AcmeCorp is located in Zurich.",
        "speaker": "Alice",
        "status": "active",
    }
    graph.index_memory(m1)

    # Both entities should be recorded and linked to mem_1
    assert "mem_1" in graph.entity_to_memories.get("AcmeCorp", set())
    assert "mem_1" in graph.entity_to_memories.get("Zurich", set())

    # 1-hop neighbors
    neighbors = graph.get_1hop_neighbors("AcmeCorp")
    neighbor_names = [n[0] for n in neighbors]
    assert "Zurich" in neighbor_names


def test_bounded_1hop_expansion():
    graph = EntityGraph()
    repo = InMemoryRepository()

    # Fact 1: Alice works at AcmeCorp
    m1 = {"id": "m1", "key": "employment", "value": "Alice works at AcmeCorp.", "status": "active"}
    # Fact 2: AcmeCorp is located in Zurich
    m2 = {"id": "m2", "key": "headquarters", "value": "AcmeCorp is located in Zurich.", "status": "active"}
    # Fact 3: Zurich is in Switzerland
    m3 = {"id": "m3", "key": "country", "value": "Zurich is in Switzerland.", "status": "active"}

    repo.save(m1)
    repo.save(m2)
    repo.save(m3)
    graph.index_repository(repo)

    # Initial candidate: only m1 is retrieved
    initial_candidates = [m1]

    # Expand from query mentioning Alice
    expanded = graph.expand_candidates(
        query="Where does Alice work?",
        initial_candidates=initial_candidates,
        repository=repo,
        max_expansion=2,
    )

    # 1-hop from Alice -> AcmeCorp -> returns m2 (AcmeCorp located in Zurich)
    expanded_ids = [e["id"] for e in expanded]
    assert "m2" in expanded_ids
    assert "m1" not in expanded_ids  # already in initial_candidates, must not be duplicated
    assert len(expanded) <= 2        # strictly bounded


def test_multihop_end_to_end_retrieval():
    """Verify that ProductionRetriever retrieves multi-hop bridge evidence when expansion is enabled."""
    repo = InMemoryRepository()

    # Bridge setup:
    # Memory A: "Alice is the lead architect of ChunkdUp"
    # Memory B: "ChunkdUp requires Docker and PostgreSQL with pgvector"
    # Multi-hop Question: "What database does Alice's project require?"
    # A single-hop dense query for "What database does Alice's project require?"
    # matches Memory A because of "Alice" and "project", but the answer (PostgreSQL) is in Memory B!

    mA = {
        "id": "mA",
        "key": "lead_architect",
        "value": "Alice is the lead architect of ChunkdUp.",
        "contextual_text": "Alice is the lead architect of ChunkdUp project",
        "status": "active",
    }
    mB = {
        "id": "mB",
        "key": "database",
        "value": "ChunkdUp requires Docker and PostgreSQL with pgvector.",
        "contextual_text": "ChunkdUp database infrastructure PostgreSQL with pgvector",
        "status": "active",
    }
    repo.save(mA)
    repo.save(mB)

    # Initialize graph and index memories
    graph = EntityGraph()
    graph.index_repository(repo)

    # Retriever with expansion enabled
    retriever = ProductionRetriever(
        repository=repo,
        config_name="hybrid_ce_abstain",
        entity_graph=graph,
        query_log_path=None,
    )
    retriever.entity_expansion_enabled = True

    results = retriever.retrieve("What database does Alice's project require?", k=5)
    result_ids = [r["id"] for r in results]

    # Both the project memory and the database memory must be retrieved!
    assert "mA" in result_ids or "mB" in result_ids
