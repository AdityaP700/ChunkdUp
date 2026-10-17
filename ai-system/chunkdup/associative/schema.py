# ai-system/chunkdup/associative/schema.py
"""
Data structures and relationship vocabulary for Evidence-Preserving Associative Memory.
Strictly preserves source turn identifiers (dia_id) for end-to-end evidence recall.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class AssociativeLinkType(str, Enum):
    """Documented relationship vocabulary for associative graph links."""
    ENTITY_TO_EVENT = "entity_to_event"
    EVENT_TO_EVENT = "event_to_event"
    FACT_TO_SOURCE = "fact_to_source"
    STATE_TRANSITION = "state_transition"
    ENTITY_RELATION = "entity_relation"
    CAUSAL_EXPLANATION = "causal_explanation"


@dataclass
class ExperienceTurn:
    """A. Experience Memory: Original dialogue turn as the ground-truth authoritative evidence."""
    dia_id: str                      # e.g., 'D1:3'
    speaker: str                     # e.g., 'Caroline'
    text: str                        # exact original speech
    session_num: int                 # dialogue session index
    timestamp: str                   # recorded timestamp / session time
    contextual_text: str = ""        # synthesized context for indexing
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticFact:
    """B. Semantic Memory: Extracted atomic fact or event with strict source provenance."""
    fact_id: str
    subject: str
    predicate: str
    object_value: str
    source_turn_ids: List[str]       # Grounding dia_ids (must not be empty)
    timestamp: str = ""              # Time fact was recorded
    event_time: Optional[str] = None # When the event actually took place
    validity_status: str = "active"  # 'active', 'superseded', 'contradicted'
    confidence: float = 1.0          # Extraction confidence [0.0, 1.0]
    superseded_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def statement(self) -> str:
        """Normalized textual representation for semantic indexing."""
        if self.predicate == "related_to":
            return f"{self.subject} is related to {self.object_value}."
        return f"{self.subject} {self.predicate} {self.object_value}."

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AssociativeLink:
    """C. Associative Memory: Typed link connecting entities, events, and facts."""
    source_id: str                   # entity, event, or fact ID
    target_id: str                   # entity, event, or fact ID
    rel_type: AssociativeLinkType
    weight: float = 1.0
    evidence_turn_ids: List[str] = field(default_factory=list) # Supporting dia_ids
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["rel_type"] = self.rel_type.value
        return d


@dataclass
class ProfileAttribute:
    """D. Optional Consolidated Profile Item (traceable view, not canonical store)."""
    key: str
    value: str
    entity: str
    source_fact_ids: List[str]
    source_turn_ids: List[str]
    last_updated: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryRepresentationBundle:
    """Container grouping all representations for a conversation."""
    sample_id: str
    experience_turns: List[ExperienceTurn] = field(default_factory=list)
    semantic_facts: List[SemanticFact] = field(default_factory=list)
    associative_links: List[AssociativeLink] = field(default_factory=list)
    profile_attributes: List[ProfileAttribute] = field(default_factory=list)

    def get_turn_by_id(self, dia_id: str) -> Optional[ExperienceTurn]:
        for t in self.experience_turns:
            if t.dia_id == dia_id:
                return t
        return None
