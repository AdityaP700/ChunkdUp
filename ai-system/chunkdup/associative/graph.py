# ai-system/chunkdup/associative/graph.py
"""
Bounded Associative Graph Implementation for ChunkdUp.
Provides typed link indexing and strictly bounded graph expansion (H1 testing).
Guarantees:
- Strict depth bound (depth <= 2)
- Strict branching bound (branching <= 3 per node)
- Strict candidate cap (max_expanded <= 5)
- Explicit evidence turn resolution
"""

from __future__ import annotations

from collections import defaultdict, deque
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from .schema import AssociativeLink, AssociativeLinkType, SemanticFact

logger = logging.getLogger(__name__)


class AssociativeGraph:
    """In-memory associative knowledge graph supporting typed multi-relational edges."""

    def __init__(self):
        # node_id -> list of (target_id, link)
        self.adj: Dict[str, List[Tuple[str, AssociativeLink]]] = defaultdict(list)
        # Reverse adjacency for bidirectional traversal
        self.rev_adj: Dict[str, List[Tuple[str, AssociativeLink]]] = defaultdict(list)
        # node_id -> set of source turn IDs (dia_ids) that ground this node
        self.node_evidence: Dict[str, Set[str]] = defaultdict(set)
        # node_id -> metadata
        self.node_meta: Dict[str, Dict[str, Any]] = {}
        # Track all unique links
        self.links: List[AssociativeLink] = []

    def add_node(
        self,
        node_id: str,
        node_type: str = "entity",
        evidence_turn_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a node with grounding evidence."""
        node_id = str(node_id).strip()
        if not node_id:
            return
        if evidence_turn_ids:
            for tid in evidence_turn_ids:
                if tid:
                    self.node_evidence[node_id].add(str(tid).strip())
        if node_id not in self.node_meta:
            self.node_meta[node_id] = {
                "node_type": node_type,
                **(metadata or {}),
            }

    def add_link(self, link: AssociativeLink) -> None:
        """Add a typed associative link with evidence turn preservation."""
        src = str(link.source_id).strip()
        tgt = str(link.target_id).strip()
        if not src or not tgt or src == tgt:
            return

        # Ensure nodes exist
        self.add_node(src, evidence_turn_ids=link.evidence_turn_ids)
        self.add_node(tgt, evidence_turn_ids=link.evidence_turn_ids)

        # Record link
        self.links.append(link)
        self.adj[src].append((tgt, link))
        self.rev_adj[tgt].append((src, link))

        for tid in link.evidence_turn_ids:
            if tid:
                self.node_evidence[src].add(str(tid).strip())
                self.node_evidence[tgt].add(str(tid).strip())

    def add_semantic_fact(self, fact: SemanticFact) -> None:
        """Index an atomic semantic fact as nodes and associative links."""
        sub = fact.subject.strip()
        obj = fact.object_value.strip()
        fid = f"fact:{fact.fact_id}"

        # Register nodes
        self.add_node(sub, node_type="entity", evidence_turn_ids=fact.source_turn_ids)
        self.add_node(obj, node_type="entity_or_value", evidence_turn_ids=fact.source_turn_ids)
        self.add_node(fid, node_type="fact", evidence_turn_ids=fact.source_turn_ids, metadata={"statement": fact.statement})

        # Register relationship link
        rel_type = (
            AssociativeLinkType.STATE_TRANSITION
            if fact.validity_status in ("superseded", "contradicted")
            else AssociativeLinkType.ENTITY_RELATION
        )
        self.add_link(
            AssociativeLink(
                source_id=sub,
                target_id=obj,
                rel_type=rel_type,
                weight=fact.confidence,
                evidence_turn_ids=fact.source_turn_ids,
                metadata={"predicate": fact.predicate, "fact_id": fact.fact_id},
            )
        )
        # Register fact-to-source link
        self.add_link(
            AssociativeLink(
                source_id=fid,
                target_id=sub,
                rel_type=AssociativeLinkType.FACT_TO_SOURCE,
                weight=fact.confidence,
                evidence_turn_ids=fact.source_turn_ids,
            )
        )


class BoundedGraphExpansion:
    """Executes strictly bounded graph expansion from retrieval seeds."""

    def __init__(
        self,
        graph: AssociativeGraph,
        max_depth: int = 2,
        max_branching: int = 3,
        max_expanded_candidates: int = 5,
        link_type_weights: Optional[Dict[AssociativeLinkType, float]] = None,
    ):
        self.graph = graph
        self.max_depth = max_depth
        self.max_branching = max_branching
        self.max_expanded_candidates = max_expanded_candidates
        self.link_type_weights = link_type_weights or {
            AssociativeLinkType.ENTITY_RELATION: 1.0,
            AssociativeLinkType.EVENT_TO_EVENT: 1.1,
            AssociativeLinkType.ENTITY_TO_EVENT: 1.0,
            AssociativeLinkType.STATE_TRANSITION: 1.2,
            AssociativeLinkType.CAUSAL_EXPLANATION: 1.2,
            AssociativeLinkType.FACT_TO_SOURCE: 0.9,
        }

    def expand(
        self,
        seed_entities: List[str],
        seed_scores: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, float, Set[str]]]:
        """
        Expand from seed entities up to max_depth and max_branching.
        Returns sorted list of (node_id, relevance_score, evidence_turn_ids).
        """
        if not seed_entities or not self.graph.adj:
            return []

        scores: Dict[str, float] = {}
        visited: Set[str] = set()
        seed_scores = seed_scores or {}

        # Queue items: (node_id, current_depth, path_score)
        queue: deque[Tuple[str, int, float]] = deque()

        for s in seed_entities:
            s_clean = str(s).strip()
            if s_clean in self.graph.adj or s_clean in self.graph.rev_adj or s_clean in self.graph.node_meta:
                init_score = seed_scores.get(s_clean, 1.0)
                queue.append((s_clean, 0, init_score))
                visited.add(s_clean)
                scores[s_clean] = init_score

        expanded_nodes: Dict[str, float] = {}

        while queue:
            curr_node, depth, curr_score = queue.popleft()

            if depth >= self.max_depth:
                continue

            # Gather forward and reverse neighbors
            neighbors: List[Tuple[str, AssociativeLink, bool]] = []
            for nxt, link in self.graph.adj.get(curr_node, []):
                neighbors.append((nxt, link, True))
            for prv, link in self.graph.rev_adj.get(curr_node, []):
                neighbors.append((prv, link, False))

            # Rank neighbors by link weight * link type multiplier
            def neighbor_priority(item: Tuple[str, AssociativeLink, bool]) -> float:
                _, l, _ = item
                type_w = self.link_type_weights.get(l.rel_type, 1.0)
                return l.weight * type_w

            neighbors.sort(key=neighbor_priority, reverse=True)

            # Apply branching limit
            branch_count = 0
            for neighbor_id, link, _ in neighbors:
                if neighbor_id in visited:
                    continue
                if branch_count >= self.max_branching:
                    break

                type_w = self.link_type_weights.get(link.rel_type, 1.0)
                decay = 0.7  # Depth discount factor
                next_score = curr_score * link.weight * type_w * decay

                visited.add(neighbor_id)
                scores[neighbor_id] = next_score
                expanded_nodes[neighbor_id] = next_score
                branch_count += 1

                queue.append((neighbor_id, depth + 1, next_score))

                if len(expanded_nodes) >= self.max_expanded_candidates:
                    break

            if len(expanded_nodes) >= self.max_expanded_candidates:
                break

        # Collect results with evidence turn IDs
        results: List[Tuple[str, float, Set[str]]] = []
        for nid, sc in expanded_nodes.items():
            ev = self.graph.node_evidence.get(nid, set())
            results.append((nid, sc, ev))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:self.max_expanded_candidates]
