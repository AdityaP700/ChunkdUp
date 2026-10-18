# ai-system/chunkdup/audit.py
"""
Persistent, append-only decision audit event schema and utilities for ChunkdUp.
Tracks every policy decision (STORE, MERGE, UPDATE, IGNORE), outcome, and provenance.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class DecisionEvent:
    """
    Immutable structured record of a single policy decision and its execution outcome.
    """

    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_turn_id: Optional[str] = None
    decision_type: str = "STORE"  # STORE, MERGE, UPDATE, IGNORE
    target_memory_id: Optional[str] = None
    memory_key: Optional[str] = None
    previous_value: Optional[str] = None
    new_value: Optional[str] = None
    version: int = 1
    rationale: Optional[str] = None
    confidence: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    status: str = "success"  # "success" | "failed"
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DecisionEvent:
        clean = dict(data)
        # Ensure decision_id is present
        if not clean.get("decision_id"):
            clean["decision_id"] = clean.get("id") or str(uuid.uuid4())
        return cls(**{k: v for k, v in clean.items() if k in cls.__dataclass_fields__})
