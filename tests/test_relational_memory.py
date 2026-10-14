# tests/test_relational_memory.py
"""
Deterministic unit tests for Relational Memory (Phase 1):
1. Pronoun coreference resolution ("I", "my", "she")
2. Cross-turn reference resolution
3. Alias canonicalization ("postgres" -> "PostgreSQL")
4. Ambiguous pronoun handling (explicit unresolved reference, no false merge)
5. Strict source turn provenance
6. Contradictory facts and bi-temporal supersession (valid_from, valid_to)
7. Point-in-time (as_of) historical reconstruction
8. Idempotent duplicate ingestion
9. Extraction resilience
"""

import pytest
from chunkdup.relational.schema import EntityNode, RelationalClaim
from chunkdup.relational.extractor import RelationalExtractor
from chunkdup.relational.graph import RelationalGraph


def test_pronoun_coreference_resolution():
    extractor = RelationalExtractor(mode="rules")

    # Turn 1: Speaker is Caroline
    turn1 = {
        "turn_id": "T1",
        "speaker": "Caroline",
        "text": "I work at Stripe.",
        "timestamp": "2026-04-01T10:00:00Z",
    }
    claims = extractor.extract_from_window(turn1)
    assert len(claims) >= 1
    # Subject must be resolved to Caroline, not raw "I"
    assert claims[0].subject_entity == "Caroline"
    assert claims[0].predicate == "works_at"
    assert claims[0].object_entity == "Stripe"
    assert "T1" in claims[0].source_turn_ids


def test_cross_turn_reference_resolution():
    extractor = RelationalExtractor(mode="rules")

    # Turn 1: Introduces Alice
    turn1 = {
        "turn_id": "T1",
        "speaker": "Bob",
        "text": "Alice founded TechCorp.",
        "timestamp": "2026-04-01T10:00:00Z",
    }
    # Turn 2: Uses 3rd-person pronoun "She"
    turn2 = {
        "turn_id": "T2",
        "speaker": "Bob",
        "text": "She uses Python.",
        "timestamp": "2026-04-01T10:01:00Z",
    }

    # Extract with context window
    claims = extractor.extract_from_window(turn2, context_window=[turn1])
    assert len(claims) >= 1
    assert claims[0].subject_entity == "Alice"
    assert claims[0].predicate == "uses"
    assert claims[0].object_entity == "Python"
    assert "T2" in claims[0].source_turn_ids


def test_ambiguous_pronoun_explicit_representation():
    extractor = RelationalExtractor(mode="rules")

    # Turn with multiple female entities
    turn1 = {
        "turn_id": "T1",
        "speaker": "Observer",
        "text": "Alice and Emma arrived at the station.",
        "timestamp": "2026-04-01T10:00:00Z",
    }
    # Turn 2 uses ambiguous "She"
    turn2 = {
        "turn_id": "T2",
        "speaker": "Observer",
        "text": "She moved to Zurich.",
        "timestamp": "2026-04-01T10:01:00Z",
    }

    claims = extractor.extract_from_window(turn2, context_window=[turn1])
    assert len(claims) >= 1
    # Must explicitly mark unresolved reference rather than guessing
    assert len(claims[0].unresolved_references) > 0
    assert "Ambiguous" in claims[0].unresolved_references[0]


def test_alias_canonicalization_and_conservative_linking():
    graph = RelationalGraph()

    claim1 = RelationalClaim(
        claim_id="c1",
        subject_entity="Jon",
        predicate="uses",
        object_entity="postgres",
        source_turn_ids=["t1"],
        context_text="Jon uses postgres",
    )
    graph.add_claim(claim1)

    # Alias "postgres" must canonicalize to "PostgreSQL"
    assert graph.claims["c1"].object_entity == "PostgreSQL"

    # Distinct individuals must not be merged
    p1 = graph.register_entity("Jon", category="person")
    p2 = graph.register_entity("Jonathan", category="person")
    assert p1.canonical_name == "Jon"
    assert p2.canonical_name == "Jonathan"
    assert p1.id != p2.id


def test_bitemporal_supersession():
    graph = RelationalGraph()

    # Claim 1 at T1: Jon works at Stripe
    c1 = RelationalClaim(
        claim_id="c1",
        subject_entity="Jon",
        predicate="works_at",
        object_entity="Stripe",
        source_turn_ids=["t1"],
        context_text="Jon works at Stripe",
        created_at="2026-01-01T00:00:00Z",
        valid_from="2026-01-01T00:00:00Z",
    )
    graph.add_claim(c1)

    assert c1.status == "active"
    assert c1.valid_to is None

    # Claim 2 at T2: Jon works at OpenAI (migration/supersession)
    c2 = RelationalClaim(
        claim_id="c2",
        subject_entity="Jon",
        predicate="works_at",
        object_entity="OpenAI",
        source_turn_ids=["t2"],
        context_text="Jon works at OpenAI",
        created_at="2026-03-01T00:00:00Z",
        valid_from="2026-03-01T00:00:00Z",
    )
    graph.add_claim(c2)

    # c1 is superseded, valid_to set to T2
    assert c1.status == "superseded"
    assert c1.valid_to == "2026-03-01T00:00:00Z"

    # c2 is active
    assert c2.status == "active"
    assert c2.supersedes_id == "c1"


def test_point_in_time_querying():
    graph = RelationalGraph()

    c1 = RelationalClaim(
        claim_id="c1",
        subject_entity="Alice",
        predicate="uses",
        object_entity="AWS",
        source_turn_ids=["t1"],
        context_text="Alice uses AWS",
        created_at="2026-01-01T00:00:00Z",
        valid_from="2026-01-01T00:00:00Z",
    )
    graph.add_claim(c1)

    c2 = RelationalClaim(
        claim_id="c2",
        subject_entity="Alice",
        predicate="uses",
        object_entity="GCP",
        source_turn_ids=["t2"],
        context_text="Alice uses GCP",
        created_at="2026-02-01T00:00:00Z",
        valid_from="2026-02-01T00:00:00Z",
    )
    graph.add_claim(c2)

    # Historical query in January: Alice used AWS
    jan_claims = graph.get_claims(as_of="2026-01-15T00:00:00Z")
    assert len(jan_claims) == 1
    assert jan_claims[0].object_entity == "AWS"

    # Current query in March: Alice uses GCP
    current_claims = graph.get_claims(as_of="2026-03-01T00:00:00Z")
    assert len(current_claims) == 1
    assert current_claims[0].object_entity == "GCP"


def test_duplicate_ingestion_idempotency():
    graph = RelationalGraph()

    c1 = RelationalClaim(
        claim_id="c1",
        subject_entity="Bob",
        predicate="prefers",
        object_entity="Neovim",
        source_turn_ids=["t1"],
        context_text="Bob prefers Neovim",
        created_at="2026-01-01T00:00:00Z",
    )
    c1_dup = RelationalClaim(
        claim_id="c2",
        subject_entity="Bob",
        predicate="prefers",
        object_entity="Neovim",
        source_turn_ids=["t2"],
        context_text="Bob prefers Neovim",
        created_at="2026-01-02T00:00:00Z",
    )

    graph.add_claim(c1)
    graph.add_claim(c1_dup)

    active = graph.get_claims(active_only=True)
    assert len(active) == 1
    # Both turn IDs should be merged into provenance
    assert "t1" in active[0].source_turn_ids
    assert "t2" in active[0].source_turn_ids
