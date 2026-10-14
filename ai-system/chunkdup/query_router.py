# ai-system/chunkdup/query_router.py
"""
Lightweight, Rule-Based Query Router for ChunkdUp (Phase 4).

Routes incoming queries to the optimal specialized memory strategy:
- "relational_multihop" -> GraphAugmentedRetriever (graph candidate expansion + CE reranking)
- "profile_attribute"  -> ProfileGuidedArchiveRetriever (two-tier profile expansion + archive CE)
- "factual_episodic"   -> ProductionRetriever / Hybrid CE (fast direct hybrid)
- "adversarial"        -> Calibrated Abstention Verification

Design:
- Zero heavy dependencies (no external LLM API calls).
- Sub-millisecond routing latency (<0.1 ms).
- Pattern and entity-based deterministic feature scoring.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class QueryRouter:
    """
    Fast rule-based router that classifies queries into retrieval routes.
    """

    RELATIONAL_KEYWORDS: Set[str] = {
        "both", "in common", "common", "shared", "together", "between",
        "connect", "connected", "connection", "relationship", "relation",
        "compare", "comparison", "difference", "neither", "either",
        "introduced", "mutual", "co-worker", "colleague", "associate",
    }

    PROFILE_KEYWORDS: Set[str] = {
        "prefer", "preference", "favorite", "favourite", "hobby", "hobbies",
        "habit", "occupation", "profession", "job", "career", "work as",
        "live in", "lives in", "reside", "hometown", "born", "birthday",
        "pet", "dog", "cat", "married", "husband", "wife", "spouse",
        "children", "daughter", "son", "kids", "favorite color", "favorite food",
        "allergies", "diet", "vegetarian", "vegan",
    }

    ADVERSARIAL_KEYWORDS: Set[str] = {
        "secret password", "alien", "spaceship", "time machine", "mars",
        "quantum teleport", "conspiracy", "fictional", "imaginary",
        "superpower", "telekinesis", "cryptocurrency wallet seed",
    }

    def __init__(self, known_entities: Optional[Set[str]] = None):
        self.known_entities = {e.lower() for e in (known_entities or set())}

    def update_known_entities(self, entities: Set[str]):
        self.known_entities.update({e.lower() for e in entities})

    def route(self, query: str) -> str:
        """
        Classifies query into one of:
        - "adversarial"
        - "relational_multihop"
        - "profile_attribute"
        - "factual_episodic"
        """
        q_lower = query.lower()
        words = set(re.findall(r"\w+", q_lower))

        # 1. Adversarial check
        for adv in self.ADVERSARIAL_KEYWORDS:
            if adv in q_lower:
                return "adversarial"

        # 2. Relational / Multi-Hop check
        # Check explicit keywords
        for rk in self.RELATIONAL_KEYWORDS:
            if rk in q_lower:
                return "relational_multihop"

        # Check for multiple recognized named entities or proper nouns
        cap_tokens = re.findall(r"\b[A-Z][a-z0-9_-]+\b", query)
        # Filter common question words
        filtered_caps = [
            c for c in cap_tokens
            if c.lower() not in {"what", "when", "where", "which", "who", "whom", "whose", "why", "how", "did", "does", "do", "is", "was", "are", "were"}
        ]
        if len(filtered_caps) >= 2:
            return "relational_multihop"

        # Check known entities overlap
        known_overlap = [e for e in self.known_entities if e in q_lower]
        if len(known_overlap) >= 2:
            return "relational_multihop"

        # 3. Profile / Attribute check
        for pk in self.PROFILE_KEYWORDS:
            if pk in q_lower:
                return "profile_attribute"

        # 4. Default: Factual / Episodic
        return "factual_episodic"
