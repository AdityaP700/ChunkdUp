# ai-system/chunkdup/entity/entity_graph.py
"""
Lightweight Entity and Relation Graph with Bounded One-Hop Expansion.

Enables ChunkdUp to resolve entity aliases, track multi-hop relationships,
and perform strictly bounded 1-hop memory expansion to address multi-hop
reasoning bottlenecks (LoCoMo Category 3) without runaway graph traversals.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Canonical entity representation."""

    canonical_name: str
    category: str  # "tool" | "environment" | "concept" | "person" | "location" | "project"
    aliases: Set[str] = field(default_factory=set)

    def matches(self, term: str) -> bool:
        """Check if term matches canonical name or any alias (case-insensitive)."""
        term_clean = term.strip().lower()
        if self.canonical_name.lower() == term_clean:
            return True
        return term_clean in {a.lower() for a in self.aliases}


@dataclass
class EntityRelation:
    """Directed or associative edge between two canonical entities."""

    source_entity: str
    relation_type: str  # "uses", "depends_on", "runs_on", "located_in", "prefers", "co_occurs_with"
    target_entity: str
    memory_id: Optional[str] = None
    confidence: float = 1.0


class EntityGraph:
    """
    In-memory entity-relationship graph for conservative linking and 1-hop retrieval expansion.
    """

    # Seed canonical entities and recognized aliases
    BUILTIN_ENTITIES: List[Tuple[str, str, List[str]]] = [
        ("ChunkdUp", "project", ["chunkdup", "memory sdk", "memory system"]),
        ("PostgreSQL", "environment", ["postgres", "postgresql database"]),
        ("pgvector", "tool", ["pg vector", "pgvector extension", "vector extension"]),
        ("Docker", "environment", ["docker container", "containers"]),
        ("WSL2", "environment", ["wsl", "windows subsystem for linux"]),
        ("all-MiniLM-L6-v2", "tool", ["all-minilm", "minilm", "embedding model"]),
        ("sentence-transformers", "tool", ["sentence transformers", "sbert"]),
        ("Neovim", "tool", ["nvim", "vim"]),
        ("VS Code", "tool", ["vscode", "visual studio code"]),
        ("Python", "environment", ["python3", "py"]),
        ("Go", "environment", ["golang"]),
        ("Kubernetes", "environment", ["k8s"]),
        ("Redis", "environment", ["redis cache"]),
        ("MySQL", "environment", ["mysql database"]),
        ("MongoDB", "environment", ["mongo"]),
    ]

    RELATION_EXTRACTION_RULES = [
        (re.compile(r"([A-Za-z0-9_\-\.]+)\s+with\s+([A-Za-z0-9_\-\.]+)", re.IGNORECASE), "uses"),
        (re.compile(r"([A-Za-z0-9_\-\.]+)\s+(?:runs? in|runs? inside|deployed on)\s+([A-Za-z0-9_\-\.]+)", re.IGNORECASE), "runs_on"),
        (re.compile(r"([A-Za-z0-9_\-\.]+)\s+(?:depends? on|requires?)\s+([A-Za-z0-9_\-\.]+)", re.IGNORECASE), "depends_on"),
        (re.compile(r"([A-Za-z0-9_\-\.]+)\s+(?:lives? in|moved to|visited|located in)\s+([A-Za-z0-9_\-\.]+)", re.IGNORECASE), "located_in"),
        (re.compile(r"([A-Za-z0-9_\-\.]+)\s+(?:prefers?|likes?|uses?)\s+([A-Za-z0-9_\-\.]+)", re.IGNORECASE), "prefers"),
    ]

    COMMON_SENTENCE_STARTERS: Set[str] = {
        "The", "This", "That", "These", "Those", "What", "Which", "Where", "When",
        "How", "Why", "Who", "Whom", "Whose", "Using", "Use", "Used", "Uses",
        "We", "They", "You", "Our", "My", "His", "Her", "Their", "Please", "Let",
        "Tell", "Give", "Show", "Find", "Get", "Is", "Are", "Was", "Were", "Do",
        "Does", "Did", "Can", "Could", "Would", "Should", "Shall", "Will", "May",
        "Might", "Must", "Have", "Has", "Had", "Hello", "Hi", "Hey", "Good", "Yes",
        "No", "Not", "Just", "Only", "Also", "Then", "So", "And", "Or", "But", "If",
        "He", "She", "It", "Him", "I", "Me", "Some", "Any", "All", "There", "Here",
    }

    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.aliases: Dict[str, str] = {}  # alias_lower -> canonical_name
        self.relations: List[EntityRelation] = []
        self.entity_to_memories: Dict[str, Set[str]] = {}
        self.memory_to_entities: Dict[str, Set[str]] = {}
        self.adjacency: Dict[str, Dict[str, List[EntityRelation]]] = {}

        # Initialize builtin entities
        for canonical, cat, aliases in self.BUILTIN_ENTITIES:
            self.register_entity(canonical, cat, aliases)

    def register_entity(self, canonical_name: str, category: str = "concept", aliases: Optional[List[str]] = None) -> Entity:
        """Register or update an entity in the graph."""
        if canonical_name in self.entities:
            entity = self.entities[canonical_name]
            if aliases:
                for a in aliases:
                    entity.aliases.add(a)
                    self.aliases[a.lower()] = canonical_name
            return entity

        alias_set = set(aliases or [])
        entity = Entity(canonical_name=canonical_name, category=category, aliases=alias_set)
        self.entities[canonical_name] = entity
        self.aliases[canonical_name.lower()] = canonical_name
        for a in alias_set:
            self.aliases[a.lower()] = canonical_name

        if canonical_name not in self.adjacency:
            self.adjacency[canonical_name] = {}
        if canonical_name not in self.entity_to_memories:
            self.entity_to_memories[canonical_name] = set()
        return entity

    def add_relation(
        self,
        source: str,
        relation_type: str,
        target: str,
        memory_id: Optional[str] = None,
        confidence: float = 1.0,
    ) -> EntityRelation:
        """Add a directed relationship between two entities."""
        # Ensure entities registered
        if source not in self.entities:
            self.register_entity(source)
        if target not in self.entities:
            self.register_entity(target)

        rel = EntityRelation(
            source_entity=source,
            relation_type=relation_type,
            target_entity=target,
            memory_id=memory_id,
            confidence=confidence,
        )
        self.relations.append(rel)

        # Update adjacency
        if target not in self.adjacency[source]:
            self.adjacency[source][target] = []
        self.adjacency[source][target].append(rel)

        # Bidirectional associative indexing
        if source not in self.adjacency[target]:
            self.adjacency[target][source] = []
        rev_rel = EntityRelation(
            source_entity=target,
            relation_type=f"rev_{relation_type}",
            target_entity=source,
            memory_id=memory_id,
            confidence=confidence,
        )
        self.adjacency[target][source].append(rev_rel)
        return rel

    def resolve_entities(self, text: str) -> List[str]:
        """Recognize and return canonical entities mentioned in text."""
        if not text:
            return []

        matched: Set[str] = set()
        text_lower = text.lower()

        # Check registered aliases via fast token n-grams (up to 4 words)
        words = re.findall(r"[a-z0-9_+\.\-]+", text_lower)
        n = len(words)
        for i in range(n):
            for k in range(1, min(5, n - i + 1)):
                phrase = " ".join(words[i : i + k])
                if phrase in self.aliases:
                    matched.add(self.aliases[phrase])

        # Extract capitalized multi-word or single-word proper nouns (including CamelCase/PascalCase)
        capitalized_tokens = re.findall(r"\b[A-Z][A-Za-z0-9_-]+(?:\s+[A-Z][A-Za-z0-9_-]+)*\b", text)
        for cap in capitalized_tokens:
            cap_clean = cap.strip().rstrip(".,;:!?")
            if not cap_clean:
                continue
            if cap_clean.lower() in self.aliases:
                matched.add(self.aliases[cap_clean.lower()])
            elif len(cap_clean) > 2 and cap_clean not in self.COMMON_SENTENCE_STARTERS:
                # Register person or named entity dynamically
                self.register_entity(cap_clean, category="named_entity")
                matched.add(cap_clean)

        return sorted(matched)

    def index_memory(self, memory: Dict[str, Any]) -> List[str]:
        """Index entities and discover relations within a memory item."""
        mid = str(memory.get("id") or memory.get("key") or "")
        if not mid:
            return []

        text_parts = [
            str(memory.get("key", "")),
            str(memory.get("value", "")),
            str(memory.get("contextual_text", "")),
            str(memory.get("speaker", "")),
        ]
        full_text = " ".join(p for p in text_parts if p)

        # Resolve entities in memory
        found_entities = self.resolve_entities(full_text)
        if memory.get("speaker"):
            spk = str(memory["speaker"]).strip()
            if spk and spk.lower() not in {"speaker a", "speaker b"}:
                self.register_entity(spk, category="person")
                if spk not in found_entities:
                    found_entities.append(spk)

        # Associate memory with entities
        self.memory_to_entities[mid] = set(found_entities)
        for ent in found_entities:
            if ent not in self.entity_to_memories:
                self.entity_to_memories[ent] = set()
            self.entity_to_memories[ent].add(mid)

        # Discover relations from syntactic patterns
        for pattern, rel_type in self.RELATION_EXTRACTION_RULES:
            for match in pattern.finditer(full_text):
                g1 = match.group(1).strip()
                g2 = match.group(2).strip()
                c1 = self.aliases.get(g1.lower(), g1)
                c2 = self.aliases.get(g2.lower(), g2)
                if c1 in self.entities and c2 in self.entities and c1 != c2:
                    self.add_relation(c1, rel_type, c2, memory_id=mid, confidence=0.9)

        # Co-occurrence relations bounded to top 6 entities to prevent O(M^2) graph blowup
        if len(found_entities) >= 2:
            top_ents = found_entities[:6]
            for i in range(len(top_ents)):
                for j in range(i + 1, len(top_ents)):
                    e1, e2 = top_ents[i], top_ents[j]
                    self.add_relation(e1, "co_occurs_with", e2, memory_id=mid, confidence=0.7)

        return found_entities

    def index_repository(self, repository) -> int:
        """Scan and index all memories in the repository."""
        memories = repository.get_all()
        for mem in memories:
            self.index_memory(mem)
        return len(memories)

    def get_1hop_neighbors(self, entity_name: str) -> List[Tuple[str, str, float]]:
        """Return 1-hop connected neighbor entities: (neighbor_canonical, relation_type, confidence)."""
        canonical = self.aliases.get(entity_name.lower(), entity_name)
        if canonical not in self.adjacency:
            return []

        neighbors: List[Tuple[str, str, float]] = []
        for neighbor, rel_list in self.adjacency[canonical].items():
            if neighbor == canonical:
                continue
            max_conf = max(r.confidence for r in rel_list)
            rel_type = rel_list[0].relation_type
            neighbors.append((neighbor, rel_type, max_conf))

        neighbors.sort(key=lambda x: x[2], reverse=True)
        return neighbors

    def expand_candidates(
        self,
        query: str,
        initial_candidates: List[Dict[str, Any]],
        repository,
        max_expansion: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Perform conservative, bounded 1-hop expansion for multi-hop evidence retrieval.

        Rules:
        1. Expansion depth is strictly 1 (no multi-hop recursion).
        2. At most `max_expansion` candidates are returned.
        3. Only activates if query mentions recognized entities that link to neighbor entities.
        4. Candidates already in initial_candidates are not duplicated.
        """
        query_entities = self.resolve_entities(query)
        initial_ids = {str(c.get("id") or c.get("key") or "") for c in initial_candidates}

        # Collect entities from top initial candidates to enable bridge expansion
        top_cand_entities: List[str] = []
        for c in initial_candidates[:3]:
            cid = str(c.get("id") or c.get("key") or "")
            if cid in self.memory_to_entities:
                for ent in self.memory_to_entities[cid]:
                    if ent not in top_cand_entities and ent not in query_entities:
                        top_cand_entities.append(ent)

        expansion_sources = list(query_entities) + top_cand_entities
        if not expansion_sources:
            return []

        expanded_candidates: List[Dict[str, Any]] = []
        seen_expanded_ids: Set[str] = set()

        for q_ent in expansion_sources:
            neighbors = self.get_1hop_neighbors(q_ent)
            for neighbor_ent, rel_type, confidence in neighbors:
                if len(expanded_candidates) >= max_expansion:
                    break

                # Retrieve memories linked to neighbor entity
                linked_mids = self.entity_to_memories.get(neighbor_ent, set())
                for mid in linked_mids:
                    if mid in initial_ids or mid in seen_expanded_ids:
                        continue
                    if len(expanded_candidates) >= max_expansion:
                        break

                    # Fetch memory object from repository
                    mem = None
                    if hasattr(repository, "get_by_id"):
                        mem = repository.get_by_id(mid)
                    if not mem and hasattr(repository, "get_by_key"):
                        mem = repository.get_by_key(mid)
                    if not mem and hasattr(repository, "get_all"):
                        # Scan repository list if get_by_id failed
                        for m in repository.get_all():
                            if str(m.get("id") or m.get("key")) == mid:
                                mem = m
                                break

                    if mem:
                        mem_copy = dict(mem)
                        mem_copy["_expanded_from_entity"] = True
                        mem_copy["_expansion_source_entity"] = q_ent
                        mem_copy["_expansion_target_entity"] = neighbor_ent
                        mem_copy["_expansion_relation"] = rel_type
                        mem_copy["_expansion_confidence"] = confidence
                        # Provide default base score so reranker or gate can compare
                        mem_copy.setdefault("_similarity", 0.35 * confidence)
                        mem_copy.setdefault("_rerank_score", -5.0 + 2.0 * confidence)
                        expanded_candidates.append(mem_copy)
                        seen_expanded_ids.add(mid)

        return expanded_candidates
