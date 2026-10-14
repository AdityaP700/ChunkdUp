# ai-system/chunkdup/extraction/schema.py
"""
Strict schema validation and eligibility checking for extracted facts.
"""

import re
import uuid
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator


ALLOWED_MEMORY_TYPES: Set[str] = {
    "environment",
    "project",
    "tool",
    "preference",
    "employment",
    "role",
    "learning",
    "general",
}

KEY_REGEX = re.compile(r"^[a-z0-9_.-]+$")

SPECULATIVE_PATTERNS = [
    re.compile(r"\b(?:might|maybe|perhaps|possibly|probably|hypothetically)\b", re.IGNORECASE),
    re.compile(r"\bif\s+(?:i|we)\s+(?:ever|could|had|were|decided)\b", re.IGNORECASE),
    re.compile(r"\b(?:i\s+wonder\s+if|not\s+sure\s+if|thinking\s+about\s+(?:maybe)?)\b", re.IGNORECASE),
    re.compile(r"\b(?:someday|one\s+day|in\s+the\s+future\s+i\s+might)\b", re.IGNORECASE),
    re.compile(r"\b(?:speculati(?:ve|on)|unconfirmed|hypothetical)\b", re.IGNORECASE),
]

CHIT_CHAT_PATTERNS = [
    r"^(?:hi|hello|hey|greetings|howdy|good\s+(?:morning|afternoon|evening|day))[\s!.]*$",
    r"^(?:thanks|thank\s+you|thx|ty|cheers)[\s!.]*$",
    r"^(?:ok|okay|cool|nice|sure|got\s+it|understood|alright|fine)(?:\s+(?:ok|okay|cool|nice|sure|got\s+it|understood|alright|fine|thanks|bye))*[\s!.]*$",
    r"^(?:yes|no|yep|nope|yeah|nah)[\s!.]*$",
    r"^(?:bye|goodbye|see\s+ya|cya|take\s+care)[\s!.]*$",
]

QUESTION_PREFIXES = (
    "what",
    "where",
    "how",
    "why",
    "when",
    "who",
    "which",
    "can you",
    "could you",
    "would you",
    "do you",
    "is it",
    "are you",
)


def is_speculative(text: str) -> bool:
    """Check if text contains speculation, hypothetical conditions, or unconfirmed inferences."""
    if not text:
        return False
    return any(p.search(text) for p in SPECULATIVE_PATTERNS)


def is_turn_eligible(turn: Dict[str, Any]) -> bool:
    """
    Level 1 Eligibility Check:
    Fast rule-based filter to determine if a dialogue turn contains extractable facts.
    Filters out noise, greetings, pure questions, and purely speculative statements
    before invoking LLM extraction.
    """
    raw_text = (turn.get("raw_text") or "").strip()
    if not raw_text or len(raw_text) < 4:
        return False

    # Check against known chit-chat / greeting patterns
    for pat in CHIT_CHAT_PATTERNS:
        if re.match(pat, raw_text, re.IGNORECASE):
            return False

    # Pure interrogative questions rarely contain user personal facts
    lower = raw_text.lower()
    if lower.endswith("?"):
        if any(lower.startswith(p) for p in QUESTION_PREFIXES):
            # Exception: e.g. "I use Linux, but what do you use?"
            if not ("i use" in lower or "i prefer" in lower or "i'm " in lower or "my " in lower):
                return False

    # Reject purely speculative or hypothetical statements without ground-truth assertions
    if is_speculative(raw_text):
        grounding_markers = ("i work at", "i use", "i am a", "my favorite", "we use", "we migrated", "i painted")
        if not any(gm in lower for gm in grounding_markers):
            return False

    return True


class ExtractedFact(BaseModel):
    """
    Level 3 Strict Schema for facts extracted from raw dialogue turns.
    Guarantees facts conform to system invariants before reaching the policy engine.
    """
    key: str
    value: str
    type: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    entities: List[str] = Field(default_factory=list)
    source_turn_id: str
    temporal_validity: Optional[Dict[str, Any]] = None

    @field_validator("source_turn_id")
    @classmethod
    def validate_source_turn_id(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("source_turn_id cannot be empty")
        return cleaned

    @field_validator("key")
    @classmethod
    def validate_and_normalize_key(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if not cleaned:
            raise ValueError("Fact key cannot be empty")
        if not KEY_REGEX.match(cleaned):
            raise ValueError(f"Invalid characters in fact key: {cleaned}")
        return cleaned

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: str) -> str:
        cleaned = v.strip().rstrip(".,;:")
        if not cleaned:
            raise ValueError("Fact value cannot be empty")
        if is_speculative(cleaned):
            raise ValueError(f"Fact value contains speculation or hypothetical statement: '{cleaned}'")
        return cleaned

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        norm = v.strip().lower()
        if norm not in ALLOWED_MEMORY_TYPES:
            raise ValueError(
                f"Memory type '{v}' not in allowed types: {sorted(ALLOWED_MEMORY_TYPES)}"
            )
        return norm

    def to_memory_dict(self) -> Dict[str, Any]:
        """Convert validated fact to standard ChunkdUp memory dictionary."""
        now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        valid_from = now_iso
        if self.temporal_validity and self.temporal_validity.get("valid_from"):
            valid_from = self.temporal_validity["valid_from"]

        return {
            "id": str(uuid.uuid4()),
            "key": self.key,
            "value": self.value,
            "type": self.type,
            "frequency": 1,
            "created_at": now_iso,
            "updated_at": now_iso,
            "importance": round(self.confidence, 3),
            "confidence": round(self.confidence, 3),
            "entities": list(self.entities),
            "source_turn_id": self.source_turn_id,
            "valid_from": valid_from,
            "valid_to": None,
            "status": "active",
            "version": 1,
            "source": "groq_extractor",
        }
