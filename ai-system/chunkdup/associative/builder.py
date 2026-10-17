# ai-system/chunkdup/associative/builder.py
"""
Memory Representation Bundle Builder.
Constructs Experience Turns, Semantic Facts, and Typed Associative Graph Links
from LoCoMo conversation samples with strict source turn provenance.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from chunkdup.relational.extractor import RelationalExtractor
from chunkdup.relational.schema import RelationalClaim
from evaluation.adapters.locomo_adapter import ConversationSample, DialogueTurn
from .graph import AssociativeGraph
from .schema import (
    AssociativeLink,
    AssociativeLinkType,
    ExperienceTurn,
    MemoryRepresentationBundle,
    ProfileAttribute,
    SemanticFact,
)

logger = logging.getLogger(__name__)


class AssociativeMemoryBuilder:
    """Builds a verified MemoryRepresentationBundle from a conversation sample."""

    def __init__(self, extractor_mode: str = "rules"):
        self.extractor = RelationalExtractor(mode=extractor_mode)

    def build_bundle(self, sample: ConversationSample) -> Tuple[MemoryRepresentationBundle, AssociativeGraph]:
        bundle = MemoryRepresentationBundle(sample_id=sample.sample_id)
        graph = AssociativeGraph()

        # 1. Experience Memory (Raw Turns)
        exp_turns: List[ExperienceTurn] = []
        for t in sample.turns:
            exp_turn = ExperienceTurn(
                dia_id=t.dia_id,
                speaker=t.speaker,
                text=t.text,
                session_num=t.session_num,
                timestamp=t.date_time,
                contextual_text=f"In session {t.session_num} ({t.date_time}), {t.speaker} said: \"{t.text}\"",
                metadata={"blip_caption": t.blip_caption} if t.blip_caption else {},
            )
            exp_turns.append(exp_turn)
            # Add dialogue turn node to graph
            turn_node = f"turn:{t.dia_id}"
            graph.add_node(turn_node, node_type="turn", evidence_turn_ids=[t.dia_id])
            graph.add_node(t.speaker, node_type="speaker", evidence_turn_ids=[t.dia_id])
            graph.add_link(
                AssociativeLink(
                    source_id=t.speaker,
                    target_id=turn_node,
                    rel_type=AssociativeLinkType.ENTITY_TO_EVENT,
                    evidence_turn_ids=[t.dia_id],
                )
            )

        bundle.experience_turns = exp_turns

        # Event-to-event succession within sessions
        for i in range(len(exp_turns) - 1):
            t_curr = exp_turns[i]
            t_next = exp_turns[i + 1]
            if t_curr.session_num == t_next.session_num:
                graph.add_link(
                    AssociativeLink(
                        source_id=f"turn:{t_curr.dia_id}",
                        target_id=f"turn:{t_next.dia_id}",
                        rel_type=AssociativeLinkType.EVENT_TO_EVENT,
                        evidence_turn_ids=[t_curr.dia_id, t_next.dia_id],
                    )
                )

        # 2. Semantic Memory Extraction
        turn_dicts = [
            {"turn_id": t.dia_id, "speaker": t.speaker, "text": t.text, "timestamp": t.date_time}
            for t in sample.turns
        ]
        semantic_facts: List[SemanticFact] = []

        for idx, t_dict in enumerate(turn_dicts):
            ctx_window = turn_dicts[max(0, idx - 3) : idx]
            claims: List[RelationalClaim] = self.extractor.extract_from_window(t_dict, context_window=ctx_window)
            for c in claims:
                fact = SemanticFact(
                    fact_id=c.claim_id,
                    subject=c.subject_entity,
                    predicate=c.predicate,
                    object_value=c.object_entity,
                    source_turn_ids=c.source_turn_ids or [t_dict["turn_id"]],
                    timestamp=c.valid_from or t_dict["timestamp"],
                    confidence=c.confidence,
                )
                semantic_facts.append(fact)
                # Add fact to graph
                graph.add_semantic_fact(fact)

        bundle.semantic_facts = semantic_facts

        # 3. Optional Consolidated Profile Attributes
        profile_attrs: List[ProfileAttribute] = []
        speaker_facts: Dict[str, Dict[str, SemanticFact]] = {
            sample.speaker_a: {},
            sample.speaker_b: {},
        }
        for f in semantic_facts:
            if f.subject in speaker_facts:
                speaker_facts[f.subject][f.predicate] = f

        for spk, p_dict in speaker_facts.items():
            for pred, f_obj in p_dict.items():
                profile_attrs.append(
                    ProfileAttribute(
                        key=pred,
                        value=f_obj.object_value,
                        entity=spk,
                        source_fact_ids=[f_obj.fact_id],
                        source_turn_ids=f_obj.source_turn_ids,
                        last_updated=f_obj.timestamp,
                    )
                )
        bundle.profile_attributes = profile_attrs

        return bundle, graph
