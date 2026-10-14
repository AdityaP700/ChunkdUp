# ai-system/chunkdup/relational/graph.py
"""
Relational Knowledge Graph with Bi-Temporal Invalidation & Conservative Entity Resolution.

Implements:
1. Canonical Entity Registry with aliases and conservative disambiguation.
2. Immutable claim log with bi-temporal supersession (valid_from, valid_to, supersedes_id).
3. Bipartite entity-claim-passage adjacency for Personalized PageRank (PPR) traversal.
4. Point-in-time state reconstruction (as_of timestamp).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from chunkdup.relational.schema import EntityNode, RelationalClaim, current_iso_utc

logger = logging.getLogger(__name__)


class RelationalGraph:
    """
    Bi-temporal relational knowledge graph connecting entities, claims, and dialogue passages.
    """

    BUILTIN_ALIASES = {
        "postgres": "PostgreSQL",
        "postgresql database": "PostgreSQL",
        "pgvector": "pgvector",
        "docker": "Docker",
        "wsl": "WSL2",
        "wsl2": "WSL2",
        "k8s": "Kubernetes",
        "nvim": "Neovim",
        "vim": "Neovim",
        "vscode": "VS Code",
        "visual studio code": "VS Code",
        "py": "Python",
        "python3": "Python",
        "golang": "Go",
        "chunkdup": "ChunkdUp",
        "aws": "AWS",
        "gcp": "GCP",
    }

    def __init__(self):
        self.entities: Dict[str, EntityNode] = {}          # canonical_name -> EntityNode
        self.alias_map: Dict[str, str] = {}               # alias_lower -> canonical_name
        self.claims: Dict[str, RelationalClaim] = {}       # claim_id -> RelationalClaim
        self.passage_to_claims: Dict[str, Set[str]] = {}  # turn_id -> set(claim_id)
        self.claim_to_passages: Dict[str, Set[str]] = {}  # claim_id -> set(turn_id)
        self.entity_to_claims: Dict[str, Set[str]] = {}   # canonical_name -> set(claim_id)

        # Initialize builtin aliases
        for alias, canonical in self.BUILTIN_ALIASES.items():
            self.register_alias(canonical, alias)

    def register_entity(self, canonical_name: str, category: str = "concept", aliases: Optional[List[str]] = None) -> EntityNode:
        name = canonical_name.strip()
        if name in self.entities:
            node = self.entities[name]
            if aliases:
                for a in aliases:
                    self.register_alias(name, a)
            return node

        node = EntityNode(
            id=str(uuid.uuid4()),
            canonical_name=name,
            category=category,
            aliases=set(aliases or []),
        )
        self.entities[name] = node
        self.alias_map[name.lower()] = name
        if aliases:
            for a in aliases:
                self.register_alias(name, a)
        return node

    def register_alias(self, canonical_name: str, alias: str) -> None:
        c_name = canonical_name.strip()
        if c_name not in self.entities:
            self.entities[c_name] = EntityNode(
                id=str(uuid.uuid4()),
                canonical_name=c_name,
                category="concept",
                aliases={alias},
            )
        self.entities[c_name].aliases.add(alias)
        self.alias_map[alias.strip().lower()] = c_name

    def resolve_entity_name(self, name: str) -> str:
        """Conservative entity resolution: resolves known aliases, never merges distinct people."""
        clean = name.strip()
        clean_lower = clean.lower()
        if clean_lower in self.alias_map:
            return self.alias_map[clean_lower]
        return clean

    def add_claim(self, claim: RelationalClaim) -> RelationalClaim:
        """
        Store a relational claim with bi-temporal invalidation of superseded facts.
        """
        # Canonicalize subject and object entities
        subj = self.resolve_entity_name(claim.subject_entity)
        obj = self.resolve_entity_name(claim.object_entity)

        self.register_entity(subj)
        self.register_entity(obj)

        claim.subject_entity = subj
        claim.object_entity = obj

        # Check for existing active claims with the exact same (subject, predicate)
        # to apply Graphiti-style bi-temporal supersession
        existing_active = [
            c for c in self.claims.values()
            if c.subject_entity == subj and c.predicate == claim.predicate and c.status == "active"
        ]

        for old_claim in existing_active:
            if old_claim.object_entity.lower() != obj.lower():
                # Value changed: bi-temporally close the old claim's validity
                inv_time = claim.created_at
                old_claim.invalidate(invalid_at=inv_time, reason="superseded")
                claim.supersedes_id = old_claim.claim_id
                logger.info(f"Bi-temporal supersession: {subj} {claim.predicate}: {old_claim.object_entity} -> {obj}")
            elif old_claim.object_entity.lower() == obj.lower():
                # Duplicate fact: preserve without double-creating
                old_claim.confidence = max(old_claim.confidence, claim.confidence)
                for tid in claim.source_turn_ids:
                    if tid not in old_claim.source_turn_ids:
                        old_claim.source_turn_ids.append(tid)
                return old_claim

        # Persist new claim
        self.claims[claim.claim_id] = claim

        # Index bipartite links
        for tid in claim.source_turn_ids:
            if tid not in self.passage_to_claims:
                self.passage_to_claims[tid] = set()
            self.passage_to_claims[tid].add(claim.claim_id)

            if claim.claim_id not in self.claim_to_passages:
                self.claim_to_passages[claim.claim_id] = set()
            self.claim_to_passages[claim.claim_id].add(tid)

        for ent in (subj, obj):
            if ent not in self.entity_to_claims:
                self.entity_to_claims[ent] = set()
            self.entity_to_claims[ent].add(claim.claim_id)

        return claim

    def get_claims(self, as_of: Optional[str] = None, active_only: bool = True) -> List[RelationalClaim]:
        """Query claims, optionally point-in-time as_of an ISO timestamp."""
        res: List[RelationalClaim] = []
        for claim in self.claims.values():
            if as_of:
                if claim.is_valid_as_of(as_of):
                    res.append(claim)
            elif active_only:
                if claim.status == "active":
                    res.append(claim)
            else:
                res.append(claim)
        return res

    def get_neighbors(self, entity_name: str, as_of: Optional[str] = None) -> List[Tuple[str, str, RelationalClaim]]:
        """
        Get connected neighbor entities: [(neighbor_entity, predicate, claim)]
        """
        canonical = self.resolve_entity_name(entity_name)
        claim_ids = self.entity_to_claims.get(canonical, set())
        neighbors: List[Tuple[str, str, RelationalClaim]] = []

        for cid in claim_ids:
            claim = self.claims.get(cid)
            if not claim or not claim.is_valid_as_of(as_of):
                continue
            if claim.subject_entity == canonical:
                neighbors.append((claim.object_entity, claim.predicate, claim))
            elif claim.object_entity == canonical:
                neighbors.append((claim.subject_entity, f"rev_{claim.predicate}", claim))

        return neighbors

    def get_supporting_turn_ids(self, claim_id: str) -> List[str]:
        claim = self.claims.get(claim_id)
        if claim:
            return list(claim.source_turn_ids)
        return []

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_entities": len(self.entities),
            "total_claims": len(self.claims),
            "active_claims": sum(1 for c in self.claims.values() if c.status == "active"),
            "superseded_claims": sum(1 for c in self.claims.values() if c.status == "superseded"),
            "total_indexed_passages": len(self.passage_to_claims),
        }
