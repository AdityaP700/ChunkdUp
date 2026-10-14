# ai-system/chunkdup/relational/schema.py
"""
Data models for Relational Memory in ChunkdUp.

Implements:
1. Canonical EntityNode with alias sets and categories.
2. Typed RelationalClaim with bi-temporal validity (valid_from, valid_to),
   strict source turn provenance, and supersession lineage.
3. Explicit representation of unresolved or ambiguous references.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set


def current_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class EntityNode:
    """Canonical entity representation in the relational graph."""

    id: str
    canonical_name: str
    category: str  # "person" | "tool" | "organization" | "location" | "project" | "concept"
    aliases: Set[str] = field(default_factory=set)
    created_at: str = field(default_factory=current_iso_utc)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches(self, name: str) -> bool:
        clean = name.strip().lower()
        if self.canonical_name.lower() == clean:
            return True
        return clean in {a.lower() for a in self.aliases}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "canonical_name": self.canonical_name,
            "category": self.category,
            "aliases": sorted(list(self.aliases)),
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EntityNode:
        return cls(
            id=data["id"],
            canonical_name=data["canonical_name"],
            category=data.get("category", "concept"),
            aliases=set(data.get("aliases", [])),
            created_at=data.get("created_at", current_iso_utc()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RelationalClaim:
    """
    Typed relational claim connecting subject and object entities.
    Features bi-temporal invalidation, strict provenance, and auditability.
    """

    claim_id: str
    subject_entity: str        # Canonical name or entity ID
    predicate: str             # e.g. "works_at", "uses", "located_in", "prefers", "created", "migrated_to"
    object_entity: str         # Canonical name or entity ID / attribute value
    source_turn_ids: List[str] # Strict source dialogue turns providing evidence
    context_text: str          # Contextual verbalization of the claim
    created_at: str = field(default_factory=current_iso_utc)
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None   # None if active; timestamp when superseded/invalidated
    status: str = "active"           # "active" | "superseded" | "invalidated"
    confidence: float = 1.0
    supersedes_id: Optional[str] = None
    unresolved_references: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid_as_of(self, timestamp: Optional[str] = None) -> bool:
        """
        Check bi-temporal validity at a given point in time.
        If timestamp is None, returns current active status.
        """
        if not timestamp:
            return self.status == "active" and self.valid_to is None

        # Point in time query
        vf = self.valid_from or self.created_at
        if timestamp < vf:
            return False
        if self.valid_to and timestamp >= self.valid_to:
            return False
        return True

    def invalidate(self, invalid_at: str, reason: str = "superseded") -> None:
        """Bi-temporally close the validity of this claim without deleting historical evidence."""
        self.valid_to = invalid_at
        self.status = "superseded" if reason == "superseded" else "invalidated"

    def to_memory_dict(self) -> Dict[str, Any]:
        """Convert to standard ChunkdUp memory dictionary format for compatibility."""
        return {
            "id": self.claim_id,
            "type": "relational_claim",
            "key": f"{self.subject_entity}:{self.predicate}",
            "value": f"{self.subject_entity} {self.predicate} {self.object_entity}",
            "subject": self.subject_entity,
            "predicate": self.predicate,
            "object": self.object_entity,
            "contextual_text": self.context_text,
            "source_turn_id": self.source_turn_ids[0] if self.source_turn_ids else None,
            "source_turn_ids": self.source_turn_ids,
            "status": self.status,
            "valid_from": self.valid_from or self.created_at,
            "valid_to": self.valid_to,
            "supersedes_id": self.supersedes_id,
            "confidence": self.confidence,
            "unresolved_references": self.unresolved_references,
            "created_at": self.created_at,
            "updated_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RelationalClaim:
        return cls(
            claim_id=data.get("id") or data.get("claim_id") or str(uuid.uuid4()),
            subject_entity=data["subject_entity"],
            predicate=data["predicate"],
            object_entity=data["object_entity"],
            source_turn_ids=data.get("source_turn_ids") or ([data["source_turn_id"]] if data.get("source_turn_id") else []),
            context_text=data.get("context_text", f"{data['subject_entity']} {data['predicate']} {data['object_entity']}"),
            created_at=data.get("created_at", current_iso_utc()),
            valid_from=data.get("valid_from"),
            valid_to=data.get("valid_to"),
            status=data.get("status", "active"),
            confidence=float(data.get("confidence", 1.0)),
            supersedes_id=data.get("supersedes_id"),
            unresolved_references=data.get("unresolved_references", []),
            metadata=data.get("metadata", {}),
        )
