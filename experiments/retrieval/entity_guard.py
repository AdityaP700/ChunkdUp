# experiments/retrieval/entity_guard.py
"""
Round-3 experiments 6 and 7: entity-aware scoring adjustments.

Experiment 6 - type-aware entity scoring (apply_entity_demotion):
    Round-2 risk 5 found a concentrated FP pattern: attribute-seeking queries
    that mention the project name ("What database does ChunkdUp use?") pull
    the project memory (m3, value "ChunkdUp") into the results, because its
    ONLY overlap with the query is the project-name token itself.

    Rule: when the query names a stored identity entity (proper-name
    vocabulary: ChunkdUp, Docker, PostgreSQL, pgvector, WSL2,
    all-MiniLM-L6-v2, sentence-transformers) AND asks about something beyond
    that entity, demote candidates of type "project_name" whose value is
    itself the named entity (the memory matched purely because the name
    appears in both).

    Type-awareness is load-bearing, not decoration: for "Where does
    ChunkdUp run?" the CORRECT Docker memory also shares only the
    "ChunkdUp" token (its contextual text says "ChunkdUp runtime
    environment") - what distinguishes it from the FP project memory is
    that the project memory IS the entity. The rule therefore demotes
    project_name-typed candidates only; the smallest change that targets
    the diagnosed pattern.

Experiment 7 - CE entity-guard (apply_entity_guard):
    Round-2 risk 3 found the cross-encoder miscalibrates on jargon (2/15
    rank inversions, both on the pgvector memory). Guard: on queries that
    name a stored identity entity, the CE may not demote a candidate out
    of the fused (dense+BM25) top-3 - candidates the fusion promoted stay
    promotable. Tested against q46 and q87.
"""
from typing import Dict, List, Set

from ai_system_path import ai_system_path  # noqa: F401  (sys.path setup)
from chunkdup.entity_extractor import EntityExtractor
from retrievers import tokenize

# Proper-name entity vocabulary (EntityExtractor.CUSTOM_ENTITIES values),
# tokenized and casefolded. Generic TECHNICAL_TERMS ("database", "backend",
# ...) are deliberately EXCLUDED: they are domain words, not identities, and
# treating them as identity tokens would demote correct attribute answers.
_IDENTITY_VOCAB: Set[str] = set()
for _value in EntityExtractor.CUSTOM_ENTITIES.values():
    _IDENTITY_VOCAB.update(tokenize(_value))


def identity_tokens_in_query(query: str) -> Set[str]:
    """Identity-entity tokens the query mentions (e.g. {'chunkdup'})."""
    return set(tokenize(query)) & _IDENTITY_VOCAB


def _candidate_text(candidate: Dict) -> str:
    return (
        f"{candidate.get('key', '')} {candidate.get('value', '')} "
        f"{candidate.get('type', '')} {candidate.get('contextual_text', '')}"
    )


def apply_entity_demotion(query: str, ranked: List[Dict], score_field: str,
                          demote_factor: float = 0.25) -> List[Dict]:
    """Experiment 6: demote project-name memories that matched an
    attribute-seeking query purely on the entity token.

    No-op when the query names no stored identity entity, or when the query
    is about the entity itself (no non-entity tokens left), or on candidates
    that carry any substantive (non-identity) overlap.

    Demoted candidates SINK below every non-demoted candidate rather than
    taking a multiplicative penalty: CE logits are unbounded (~+0.3 for the
    entity match vs ~-10 for irrelevant candidates), so a multiplicative
    factor cannot cross that gap (observed in the first round-3 run). The
    first non-demoted candidate sits 1.0 above the sink floor.
    """
    query_tokens = set(tokenize(query))
    identity = identity_tokens_in_query(query)
    non_entity = query_tokens - identity
    if not identity or not non_entity:
        return ranked

    demoted_flags = []
    for candidate in ranked:
        is_demoted = False
        if candidate.get("type") == "project_name":
            candidate_tokens = set(tokenize(_candidate_text(candidate)))
            substantive = (query_tokens & candidate_tokens) - identity
            value_tokens = set(tokenize(str(candidate.get("value", "")).lower()))
            if not substantive and value_tokens and value_tokens <= identity:
                is_demoted = True
        demoted_flags.append(is_demoted)

    live_scores = [float(c.get(score_field, 0.0)) for c, d in zip(ranked, demoted_flags) if not d]
    sink_floor = (min(live_scores) - 1.0) if live_scores else 0.0

    adjusted = []
    sink_offset = 0
    for candidate, is_demoted in zip(ranked, demoted_flags):
        candidate = dict(candidate)
        if is_demoted:
            candidate[f"_pre_demotion_{score_field}"] = candidate.get(score_field)
            candidate[score_field] = sink_floor - 0.001 * sink_offset
            candidate["_entity_demoted"] = True
            sink_offset += 1
        adjusted.append(candidate)

    adjusted.sort(key=lambda c: c.get(score_field, 0.0), reverse=True)
    return adjusted


def apply_entity_guard(query: str, ranked_ce: List[Dict], fused: List[Dict],
                       protect_top: int = 3) -> List[Dict]:
    """Experiment 7: on identity-entity queries, the cross-encoder may not
    demote a candidate out of the fused top-N. Candidates the fusion promoted
    to rank <= protect_top keep a position <= their fused rank."""
    if not identity_tokens_in_query(query) or not fused:
        return ranked_ce

def apply_entity_guard(query: str, ranked_ce: List[Dict], fused: List[Dict],
                       protect_top: int = 3) -> List[Dict]:
    """Experiment 7: on identity-entity queries, fall back to the fused
    ranking for the fused top-N: those candidates lead in fused order, and
    the cross-encoder only orders the remainder. (A naive per-candidate
    restore cascades - restoring one candidate demotes the next - so the
    guard is expressed as one deterministic reordering.)"""
    if not identity_tokens_in_query(query) or not fused:
        return ranked_ce

    protected = []
    protected_ids = set()
    for fused_rank, candidate in enumerate(fused[:protect_top], start=1):
        guarded = dict(candidate)
        positions = [i for i, c in enumerate(ranked_ce) if c["id"] == candidate["id"]]
        if positions and positions[0] + 1 > fused_rank:
            guarded["_entity_guard_restored"] = True
        protected.append(guarded)
        protected_ids.add(candidate["id"])

    rest = [dict(c) for c in ranked_ce if c["id"] not in protected_ids]
    return protected + rest
