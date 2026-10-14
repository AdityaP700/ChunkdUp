# ai-system/chunkdup/profile/schema.py
"""
Data models for Persistent Profile memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def current_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ProfileFact:
    """
    An atomic user profile fact, preference, constraint, or state entry.
    """
    fact_id: str
    key: str
    value: str
    category: str = "preference"  # preference, project_state, constraint, bio, tool
    confidence: float = 1.0
    created_at: str = field(default_factory=current_iso_utc)
    updated_at: str = field(default_factory=current_iso_utc)
    source_turn_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "key": self.key,
            "value": self.value,
            "category": self.category,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source_turn_id": self.source_turn_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProfileFact:
        return cls(
            fact_id=data["fact_id"],
            key=data["key"],
            value=data["value"],
            category=data.get("category", "preference"),
            confidence=data.get("confidence", 1.0),
            created_at=data.get("created_at", current_iso_utc()),
            updated_at=data.get("updated_at", current_iso_utc()),
            source_turn_id=data.get("source_turn_id"),
            metadata=data.get("metadata", {}),
        )
