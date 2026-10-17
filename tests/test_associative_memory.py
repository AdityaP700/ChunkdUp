# tests/test_associative_memory.py
"""
Unit and integration tests for Evidence-Preserving Associative Memory:
- Schema preservation and provenance tracking
- Strict graph expansion bounds (depth <= 2, branching <= 3, candidates <= 5)
- Retrieval variants A through G functionality
- Query router classification and fallback mechanics
"""

import tempfile
import pytest

from chunkdup.associative.graph import AssociativeGraph, BoundedGraphExpansion
from chunkdup.associative.retrievers import (
    BM25Retriever,
    BoundedAssociativeRetriever,
    ConsolidatedAssociativeRetriever,
    ConsolidatedSemanticRetriever,
    GrepRetriever,
    HybridCEControlRetriever,
    QueryRoutedAssociativeRetriever,
    RetrievalResult,
)
from chunkdup.associative.schema import (
    AssociativeLink,
    AssociativeLinkType,
    ExperienceTurn,
    MemoryRepresentationBundle,
    SemanticFact,
)


@pytest.fixture
def sample_experience_turns():
    return [
        ExperienceTurn(
            dia_id="D1:1",
            speaker="Alice",
            text="Hi Bob, I just moved to Seattle and started working at TechCorp as a software architect.",
            session_num=1,
            timestamp="2023-01-10 10:00:00",
        ),
        ExperienceTurn(
            dia_id="D1:2",
            speaker="Bob",
            text="Congratulations Alice! TechCorp is great. Are you still using Python and PostgreSQL?",
            session_num=1,
            timestamp="2023-01-10 10:01:00",
        ),
        ExperienceTurn(
            dia_id="D1:3",
            speaker="Alice",
            text="Yes, but we recently migrated our primary storage from MySQL to PostgreSQL for better JSON performance.",
            session_num=1,
            timestamp="2023-01-10 10:02:00",
        ),
        ExperienceTurn(
            dia_id="D2:1",
            speaker="Alice",
            text="Bob, I adopted a golden retriever named Rusty yesterday!",
            session_num=2,
            timestamp="2023-01-15 14:00:00",
        ),
        ExperienceTurn(
            dia_id="D2:2",
            speaker="Bob",
            text="That's awesome! Does Rusty like your new backyard in Seattle?",
            session_num=2,
            timestamp="2023-01-15 14:05:00",
        ),
    ]


@pytest.fixture
def sample_semantic_facts():
    return [
        SemanticFact(
            fact_id="fact-1",
            subject="Alice",
            predicate="works_at",
            object_value="TechCorp",
            source_turn_ids=["D1:1"],
            timestamp="2023-01-10 10:00:00",
        ),
        SemanticFact(
            fact_id="fact-2",
            subject="Alice",
            predicate="located_in",
            object_value="Seattle",
            source_turn_ids=["D1:1", "D2:2"],
            timestamp="2023-01-10 10:00:00",
        ),
        SemanticFact(
            fact_id="fact-3",
            subject="TechCorp",
            predicate="migrated_to",
            object_value="PostgreSQL",
            source_turn_ids=["D1:3"],
            timestamp="2023-01-10 10:02:00",
        ),
        SemanticFact(
            fact_id="fact-4",
            subject="Alice",
            predicate="adopted",
            object_value="Rusty",
            source_turn_ids=["D2:1"],
            timestamp="2023-01-15 14:00:00",
        ),
    ]


@pytest.fixture
def sample_associative_graph(sample_semantic_facts):
    graph = AssociativeGraph()
    for f in sample_semantic_facts:
        graph.add_semantic_fact(f)
    # Add an event-to-event link
    graph.add_link(
        AssociativeLink(
            source_id="turn:D1:1",
            target_id="turn:D1:2",
            rel_type=AssociativeLinkType.EVENT_TO_EVENT,
            evidence_turn_ids=["D1:1", "D1:2"],
        )
    )
    return graph


def test_schema_provenance_preservation(sample_experience_turns, sample_semantic_facts):
    bundle = MemoryRepresentationBundle(
        sample_id="test-conv",
        experience_turns=sample_experience_turns,
        semantic_facts=sample_semantic_facts,
    )
    assert len(bundle.experience_turns) == 5
    assert len(bundle.semantic_facts) == 4
    for fact in bundle.semantic_facts:
        assert len(fact.source_turn_ids) > 0
        for tid in fact.source_turn_ids:
            turn = bundle.get_turn_by_id(tid)
            assert turn is not None, f"Fact source turn {tid} must exist in experience turns"


def test_bounded_graph_expansion_limits(sample_associative_graph):
    # Test strict bounds
    expander = BoundedGraphExpansion(
        graph=sample_associative_graph,
        max_depth=2,
        max_branching=2,
        max_expanded_candidates=3,
    )
    expanded = expander.expand(seed_entities=["Alice"])
    assert len(expanded) <= 3, "Expanded candidates must strictly respect max_expanded_candidates cap"
    for nid, score, evidence_set in expanded:
        assert isinstance(nid, str)
        assert score > 0.0
        assert isinstance(evidence_set, set)


def test_grep_retriever(sample_experience_turns):
    retriever = GrepRetriever(turns=sample_experience_turns)
    hits = retriever.retrieve("golden retriever named Rusty", k=3)
    assert len(hits) > 0
    assert hits[0].dia_id == "D2:1"
    assert "Rusty" in hits[0].text
    assert hits[0].source_type == "lexical"


def test_bm25_retriever(sample_experience_turns):
    retriever = BM25Retriever(turns=sample_experience_turns)
    hits = retriever.retrieve("PostgreSQL migration MySQL", k=3)
    assert len(hits) > 0
    assert hits[0].dia_id == "D1:3"
    assert "PostgreSQL" in hits[0].text
    assert hits[0].source_type == "bm25"


def test_hybrid_ce_control_retriever(sample_experience_turns):
    retriever = HybridCEControlRetriever(turns=sample_experience_turns, sample_id="test", device="cpu")
    hits = retriever.retrieve("Where does Alice work?", k=3)
    assert len(hits) > 0
    assert hits[0].dia_id in ("D1:1", "D1:2")
    assert hits[0].source_type == "hybrid_ce"


def test_bounded_associative_retriever(sample_experience_turns, sample_associative_graph):
    retriever = BoundedAssociativeRetriever(
        turns=sample_experience_turns,
        graph=sample_associative_graph,
        sample_id="test",
        device="cpu",
    )
    # Question linking Alice and PostgreSQL across two hops
    hits = retriever.retrieve("What database did Alice's company migrate to?", k=3)
    assert len(hits) > 0
    turn_ids = [h.dia_id for h in hits]
    assert "D1:3" in turn_ids or "D1:1" in turn_ids


def test_consolidated_semantic_retriever(sample_experience_turns, sample_semantic_facts):
    retriever = ConsolidatedSemanticRetriever(
        turns=sample_experience_turns,
        facts=sample_semantic_facts,
        sample_id="test",
        device="cpu",
    )
    hits = retriever.retrieve("What pet did Alice adopt?", k=3)
    assert len(hits) > 0
    turn_ids = [h.dia_id for h in hits]
    assert "D2:1" in turn_ids


def test_query_routed_associative_retriever(
    sample_experience_turns, sample_semantic_facts, sample_associative_graph
):
    va = GrepRetriever(turns=sample_experience_turns)
    vb = BM25Retriever(turns=sample_experience_turns)
    vc = HybridCEControlRetriever(turns=sample_experience_turns, sample_id="test", device="cpu")
    vd = BoundedAssociativeRetriever(
        turns=sample_experience_turns, graph=sample_associative_graph, sample_id="test", device="cpu"
    )
    ve = ConsolidatedSemanticRetriever(
        turns=sample_experience_turns, facts=sample_semantic_facts, sample_id="test", device="cpu"
    )
    vf = ConsolidatedAssociativeRetriever(
        turns=sample_experience_turns,
        facts=sample_semantic_facts,
        graph=sample_associative_graph,
        sample_id="test",
        device="cpu",
    )
    router = QueryRoutedAssociativeRetriever(va, vb, vc, vd, ve, vf)

    # Check routing rules
    assert router.route_query("When did Alice adopt Rusty?") == "variant_e"
    assert router.route_query("Why did the company migrate and what caused it?") == "variant_d"
    assert router.route_query('"golden retriever"') == "variant_a"
    assert router.route_query("Where is TechCorp?") == "variant_c"
    assert router.route_query("Why did Alice never visit an alien unicorn?") == "abstain"

    # Check execution
    hits, mode = router.retrieve("When did Alice adopt Rusty?", k=3)
    assert mode == "variant_e"
    assert len(hits) > 0
