# tests/test_ppr_retrieval.py
"""
Tests for HippoRAG-inspired Personalized PageRank (PPR) Relational Graph Retrieval.
Verifies multi-hop reasoning, hub suppression, disconnected components,
empty graph fallbacks, and hybrid RRF integration.
"""

import pytest
import torch

from chunkdup.relational.graph import RelationalGraph
from chunkdup.relational.schema import RelationalClaim
from chunkdup.retrieval_ppr import PPRHybridRetriever, RelationalPPRGraph


def test_ppr_1hop_retrieval():
    """Verify single-hop retrieval directly activates connected passage."""
    graph = RelationalGraph()
    claim = RelationalClaim(
        claim_id="c1",
        subject_entity="Alice",
        predicate="uses",
        object_entity="PostgreSQL",
        source_turn_ids=["p1"],
        context_text="Alice uses PostgreSQL",
    )
    graph.add_claim(claim)

    passages = [
        {"id": "p1", "text": "I set up PostgreSQL for our primary store."},
        {"id": "p2", "text": "Unrelated passage about cooking dinner."},
    ]

    ppr_graph = RelationalPPRGraph(relational_graph=graph)
    ppr_graph.build_from_passages_and_claims(passages)

    results = ppr_graph.retrieve(query="What database does Alice use?", top_k=2)
    assert len(results) >= 1
    assert results[0]["id"] == "p1"
    assert results[0]["score"] > results[1]["score"]


def test_ppr_multihop_2hop_and_3hop_reasoning():
    """
    Demonstrate multi-hop retrieval success where standard lexical BM25/keyword matching
    would fail because the final target passage contains zero query tokens.

    Query: "What language does the founder of TechCorp use?"
    Chain:
      Query mentions 'TechCorp'
      Hop 1: TechCorp -> Claim1 (Alice founded TechCorp) -> Alice
      Hop 2: Alice -> Claim2 (Alice uses Python) -> Python
      Hop 3: Claim2 -> Passage p3: "She chose the language for quick prototyping."
             (Notice passage p3 contains neither 'TechCorp' nor 'founder'!)
    """
    graph = RelationalGraph()

    # Claim 1: Alice founded TechCorp (source turn p1)
    c1 = RelationalClaim(
        claim_id="c1",
        subject_entity="Alice",
        predicate="founded",
        object_entity="TechCorp",
        source_turn_ids=["p1"],
        context_text="Alice founded TechCorp in 2020",
    )
    # Claim 2: Alice uses Python (source turn p3)
    c2 = RelationalClaim(
        claim_id="c2",
        subject_entity="Alice",
        predicate="uses",
        object_entity="Python",
        source_turn_ids=["p3"],
        context_text="Alice uses Python for engineering",
    )
    graph.add_claim(c1)
    graph.add_claim(c2)

    passages = [
        {"id": "p1", "text": "TechCorp was founded a few years ago."},
        {"id": "p2", "text": "Bob moved to Seattle to work at Amazon."},
        {"id": "p3", "text": "She chose the language for quick prototyping and services."},
    ]

    ppr_graph = RelationalPPRGraph(relational_graph=graph, damping=0.15, max_iter=25)
    ppr_graph.build_from_passages_and_claims(passages)

    # Query with seed 'TechCorp'
    results = ppr_graph.retrieve(query="What language does the founder of TechCorp use?", top_k=3)
    p_ids = [r["id"] for r in results]

    # Both p1 (founding) and p3 (language usage via Alice) must be ranked above unrelated p2!
    assert "p1" in p_ids[:2]
    assert "p3" in p_ids[:2]
    # Unrelated Bob passage must be last
    assert results[-1]["id"] == "p2"


def test_ppr_high_degree_hub_suppression():
    """
    Verify that degree-normalized transition weights prevent generic high-degree nodes
    (e.g., 'Python' connected to dozens of entities) from capturing disproportionate probability mass.
    """
    graph = RelationalGraph()

    # Generic hub 'Python' used by many people
    for i in range(15):
        c = RelationalClaim(
            claim_id=f"hub_{i}",
            subject_entity=f"Person_{i}",
            predicate="uses",
            object_entity="Python",
            source_turn_ids=[f"p_hub_{i}"],
            context_text=f"Person {i} uses Python",
        )
        graph.add_claim(c)

    # Specific focal claim: David built AuroraDB
    focal_c = RelationalClaim(
        claim_id="focal",
        subject_entity="David",
        predicate="built",
        object_entity="AuroraDB",
        source_turn_ids=["p_focal"],
        context_text="David built AuroraDB from scratch",
    )
    graph.add_claim(focal_c)

    passages = [{"id": f"p_hub_{i}", "text": f"Person {i} scripting"} for i in range(15)]
    passages.append({"id": "p_focal", "text": "AuroraDB is an in-memory storage engine."})

    # With hub suppression enabled
    ppr_graph = RelationalPPRGraph(relational_graph=graph, hub_suppression=True)
    ppr_graph.build_from_passages_and_claims(passages)

    results = ppr_graph.retrieve(query="Who built AuroraDB?", top_k=3)
    # The focal passage directly linked to AuroraDB should decisively win over generic Python hub passages
    assert results[0]["id"] == "p_focal"
    assert results[0]["score"] > 3 * results[1]["score"]


def test_ppr_disconnected_components():
    """Verify that disconnected subgraphs converge stably without numerical errors."""
    graph = RelationalGraph()

    # Component A: Alice and TechCorp
    graph.add_claim(RelationalClaim(
        claim_id="cA",
        subject_entity="Alice",
        predicate="founded",
        object_entity="TechCorp",
        source_turn_ids=["pA"],
        context_text="Alice founded TechCorp",
    ))

    # Component B (completely disconnected): Charlie in London
    graph.add_claim(RelationalClaim(
        claim_id="cB",
        subject_entity="Charlie",
        predicate="located_in",
        object_entity="London",
        source_turn_ids=["pB"],
        context_text="Charlie moved to London",
    ))

    passages = [
        {"id": "pA", "text": "Alice founded TechCorp."},
        {"id": "pB", "text": "Charlie moved to London."},
    ]

    ppr_graph = RelationalPPRGraph(relational_graph=graph)
    ppr_graph.build_from_passages_and_claims(passages)

    # Query targeting component A
    results_a = ppr_graph.retrieve(query="Where is TechCorp?", top_k=2)
    assert results_a[0]["id"] == "pA"
    assert results_a[0]["score"] > results_a[1]["score"]

    # Query targeting component B
    results_b = ppr_graph.retrieve(query="Where is Charlie?", top_k=2)
    assert results_b[0]["id"] == "pB"
    assert results_b[0]["score"] > results_b[1]["score"]


def test_ppr_empty_graph_and_unrecognized_entities():
    """Verify graceful handling when graph is empty or no entities match query."""
    empty_graph = RelationalGraph()
    passages = [
        {"id": "p1", "text": "First passage"},
        {"id": "p2", "text": "Second passage"},
    ]

    ppr_empty = RelationalPPRGraph(relational_graph=empty_graph)
    ppr_empty.build_from_passages_and_claims(passages)

    # Should not crash, returns valid ranked results with fallback
    results = ppr_empty.retrieve(query="completely unknown query tokens", top_k=2)
    assert len(results) == 2
    assert all(r["score"] > 0 for r in results)


def test_ppr_hybrid_rrf_retriever():
    """Verify PPRHybridRetriever integrates PPR ranks with dense and BM25 candidates."""
    graph = RelationalGraph()
    graph.add_claim(RelationalClaim(
        claim_id="c1",
        subject_entity="Samantha",
        predicate="uses",
        object_entity="Rust",
        source_turn_ids=["p_rust"],
        context_text="Samantha uses Rust",
    ))

    passages = [
        {"id": "p_rust", "text": "Systems programming in memory-safe languages."},
        {"id": "p_dense", "text": "Samantha loves performance and low-level code."},
        {"id": "p_lexical", "text": "Rust compiler error diagnostics."},
    ]

    retriever = PPRHybridRetriever(relational_graph=graph, passages=passages)

    dense_candidates = [
        {"id": "p_dense", "score": 0.85, "text": passages[1]["text"]},
        {"id": "p_rust", "score": 0.70, "text": passages[0]["text"]},
    ]
    bm25_candidates = [
        {"id": "p_lexical", "score": 12.5, "text": passages[2]["text"]},
        {"id": "p_rust", "score": 8.0, "text": passages[0]["text"]},
    ]

    fused = retriever.retrieve(
        query="What tools does Samantha use?",
        dense_candidates=dense_candidates,
        bm25_candidates=bm25_candidates,
        top_k=3,
    )

    assert len(fused) == 3
    # p_rust appears across PPR, dense, and BM25, so should achieve top reciprocal rank
    assert fused[0]["id"] == "p_rust"
    assert fused[0]["retrieval_path"] == "ppr_hybrid_rrf"
