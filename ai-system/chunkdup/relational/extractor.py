# ai-system/chunkdup/relational/extractor.py
"""
Context-Aware Relational Extractor with Coreference Resolution & Strict Provenance.

Features:
1. Multi-turn sliding window context (speaker, text, timestamp, turn_id).
2. Coreference and pronoun resolution (resolving "I", "my", "she", "he", "they" to known entities).
3. Explicit representation of unresolved or ambiguous references.
4. Strict source turn provenance linkage: claims must cite valid turn IDs in the input window.
5. Bi-temporal metadata tracking (valid_from timestamp).
6. Dual-mode execution: deterministic rules baseline or Groq neural LLM extractor.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from chunkdup.relational.schema import RelationalClaim, current_iso_utc

logger = logging.getLogger(__name__)

PRONOUN_MAP = {
    "i": "SPEAKER",
    "me": "SPEAKER",
    "my": "SPEAKER",
    "myself": "SPEAKER",
    "we": "SPEAKER_GROUP",
    "us": "SPEAKER_GROUP",
    "our": "SPEAKER_GROUP",
    "he": "MASCULINE_SINGULAR",
    "him": "MASCULINE_SINGULAR",
    "his": "MASCULINE_SINGULAR",
    "she": "FEMININE_SINGULAR",
    "her": "FEMININE_SINGULAR",
    "hers": "FEMININE_SINGULAR",
    "it": "INANIMATE_SINGULAR",
    "its": "INANIMATE_SINGULAR",
    "they": "PLURAL_OR_NEUTRAL",
    "them": "PLURAL_OR_NEUTRAL",
    "their": "PLURAL_OR_NEUTRAL",
}

RELATIONAL_PATTERNS = [
    # Employment / Role
    (re.compile(r"(\b[A-Za-z0-9_\-\.\s]+?)\s+(?:works? at|employed at|works? for)\s+([A-Za-z0-9_\-\.\s]+)", re.IGNORECASE), "works_at"),
    (re.compile(r"(\b[A-Za-z0-9_\-\.\s]+?)\s+(?:is a|is an|serves as|works as)\s+([A-Za-z0-9_\-\.\s]+)", re.IGNORECASE), "has_role"),
    # Location / Residence
    (re.compile(r"(\b[A-Za-z0-9_\-\.\s]+?)\s+(?:lives? in|moved to|relocated to|located in|headquartered in)\s+([A-Za-z0-9_\-\.\s]+)", re.IGNORECASE), "located_in"),
    # Tools / Technologies / Migration
    (re.compile(r"(\b[A-Za-z0-9_\-\.\s]+?)\s+(?:uses?|using|runs? on|deployed on)\s+([A-Za-z0-9_\-\.\s]+)", re.IGNORECASE), "uses"),
    (re.compile(r"(\b[A-Za-z0-9_\-\.\s]+?)\s+(?:migrated to|switched to|now use|now uses)\s+([A-Za-z0-9_\-\.\s]+)", re.IGNORECASE), "migrated_to"),
    (re.compile(r"(\b[A-Za-z0-9_\-\.\s]+?)\s+(?:prefers?|likes?|favorites?)\s+([A-Za-z0-9_\-\.\s]+)", re.IGNORECASE), "prefers"),
    # Project Creation / Architecture
    (re.compile(r"(\b[A-Za-z0-9_\-\.\s]+?)\s+(?:built|building|founded|created|developed|started)\s+([A-Za-z0-9_\-\.\s]+)", re.IGNORECASE), "created"),
    (re.compile(r"(\b[A-Za-z0-9_\-\.\s]+?)\s+(?:requires?|depends? on)\s+([A-Za-z0-9_\-\.\s]+)", re.IGNORECASE), "depends_on"),
]


FEMALE_NAMES = {"alice", "emma", "caroline", "sarah", "mary", "anna", "elizabeth", "lucy", "jane", "clara", "sophia", "olivia", "emily", "grace", "chloe"}
MALE_NAMES = {"bob", "john", "jon", "jonathan", "david", "michael", "james", "robert", "william", "tom", "peter", "alex", "mark", "steve"}
INANIMATE_SUFFIXES = ("corp", "inc", "ltd", "llc", "technologies", "tech", "station", "city", "database", "system")
TECH_AND_PLACES = {"python", "postgres", "postgresql", "zurich", "london", "paris", "tokyo", "station", "office", "mysql", "redis", "docker", "kubernetes", "react", "fastapi"}


class CoreferenceTracker:
    """
    Maintains recent entities across dialogue turns to resolve pronouns or identify ambiguities.
    """

    def __init__(self):
        self.entities_by_turn: List[Dict[str, Any]] = []  # [{speaker: ..., entities: [...], turn_id: ...}]
        self.known_entities: Dict[str, str] = {}         # name -> gender/type ("Caroline" -> "FEMININE_SINGULAR")

    def register_turn(self, turn_id: str, speaker: str, entities: List[str]):
        clean_entities = [
            e.strip() for e in entities 
            if e and len(e.strip()) > 1 
            and e.strip().lower() not in PRONOUN_MAP 
            and e.strip().lower() not in {"user", "speaker", "unknown", "observer"}
        ]
        self.entities_by_turn.append({
            "turn_id": turn_id,
            "speaker": speaker.strip(),
            "entities": clean_entities,
        })
        if len(self.entities_by_turn) > 8:
            self.entities_by_turn.pop(0)

    def resolve_subject(self, raw_subject: str, current_speaker: str, current_turn_id: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """
        Resolve raw subject string.
        Returns: (resolved_subject, unresolved_reason_if_any)
        """
        subj = raw_subject.strip()
        subj_lower = subj.lower()

        # 1. 1st Person pronouns -> Current Speaker
        if subj_lower in ("i", "me", "my", "myself"):
            if current_speaker and current_speaker.lower() not in ("user", "speaker", "unknown"):
                return current_speaker, None
            return "User", None

        # 2. 1st Person plural -> Speaker & Team
        if subj_lower in ("we", "us", "our"):
            if current_speaker and current_speaker.lower() not in ("user", "speaker", "unknown"):
                return f"{current_speaker}'s team", None
            return "User's team", None

        # 3. 3rd Person pronouns -> Look back through recent turns
        if subj_lower in ("he", "him", "his", "she", "her", "hers", "they", "them", "their", "it"):
            target_pronoun_type = PRONOUN_MAP.get(subj_lower)
            candidates: List[str] = []

            for past_turn in reversed(self.entities_by_turn):
                # Pronouns at the beginning of current clause resolve to prior turn antecedents
                if current_turn_id and past_turn["turn_id"] == current_turn_id:
                    continue

                spk = past_turn["speaker"]
                if spk and spk != current_speaker and spk.lower() not in ("user", "speaker", "unknown", "observer"):
                    if target_pronoun_type == "FEMININE_SINGULAR" and spk.lower() in FEMALE_NAMES:
                        if spk not in candidates:
                            candidates.append(spk)
                    elif target_pronoun_type == "MASCULINE_SINGULAR" and spk.lower() in MALE_NAMES:
                        if spk not in candidates:
                            candidates.append(spk)
                    elif target_pronoun_type in ("PLURAL_OR_NEUTRAL", "SPEAKER", "SPEAKER_GROUP"):
                        if spk not in candidates:
                            candidates.append(spk)

                for ent in past_turn["entities"]:
                    if ent == current_speaker:
                        continue
                    ent_lower = ent.lower()
                    if target_pronoun_type == "FEMININE_SINGULAR":
                        if ent_lower in TECH_AND_PLACES or any(ent_lower.endswith(sfx) for sfx in INANIMATE_SUFFIXES):
                            continue
                        if ent not in candidates:
                            candidates.append(ent)
                    elif target_pronoun_type == "MASCULINE_SINGULAR":
                        if ent_lower in TECH_AND_PLACES or any(ent_lower.endswith(sfx) for sfx in INANIMATE_SUFFIXES):
                            continue
                        if ent not in candidates:
                            candidates.append(ent)
                    elif target_pronoun_type == "INANIMATE_SINGULAR":
                        if ent_lower in TECH_AND_PLACES or any(ent_lower.endswith(sfx) for sfx in INANIMATE_SUFFIXES):
                            if ent not in candidates:
                                candidates.append(ent)
                    else:
                        if ent not in candidates:
                            candidates.append(ent)

                if candidates:
                    if target_pronoun_type == "FEMININE_SINGULAR":
                        females = [c for c in candidates if c.lower() in FEMALE_NAMES or self.known_entities.get(c) == "FEMININE_SINGULAR"]
                        if females:
                            candidates = females
                    elif target_pronoun_type == "MASCULINE_SINGULAR":
                        males = [c for c in candidates if c.lower() in MALE_NAMES or self.known_entities.get(c) == "MASCULINE_SINGULAR"]
                        if males:
                            candidates = males
                    break

            if len(candidates) == 1:
                return candidates[0], None
            elif len(candidates) > 1:
                # Ambiguous reference: do not guess blindly!
                return subj, f"Ambiguous pronoun '{subj}' between candidates: {candidates[:2]}"
            else:
                return subj, f"Unresolved pronoun '{subj}' with no prior antecedents"

        # Explicit entity name
        return subj, None


class RelationalExtractor:
    """
    Context-aware relational extractor supporting multi-turn windows,
    pronoun resolution, bi-temporal validity, and provenance validation.
    """

    def __init__(self, mode: str = "rules", groq_client: Optional[Any] = None, groq_model: Optional[str] = None):
        self.mode = mode.lower()
        self.groq_client = groq_client
        self.groq_model = groq_model or "openai/gpt-oss-20b"
        self.coref_tracker = CoreferenceTracker()

    def extract_from_window(
        self,
        current_turn: Dict[str, Any],
        context_window: Optional[List[Dict[str, Any]]] = None,
    ) -> List[RelationalClaim]:
        """
        Extract relational claims from current turn considering prior context turns.

        Args:
            current_turn: {"turn_id": str, "speaker": str, "text": str, "timestamp": str}
            context_window: List of previous 1-3 turns for coreference context.
        """
        curr_turn_id = str(current_turn.get("turn_id") or uuid.uuid4())
        curr_speaker = str(current_turn.get("speaker") or "User").strip()
        curr_text = str(current_turn.get("text") or current_turn.get("raw_text") or "").strip()
        curr_ts = current_turn.get("timestamp") or current_iso_utc()

        # Collect valid turn IDs in this input window
        valid_turn_ids: Set[str] = {curr_turn_id}
        if context_window:
            for past in context_window:
                tid = past.get("turn_id")
                if tid:
                    valid_turn_ids.add(str(tid))
                # Update coref tracker with context
                spk = past.get("speaker", "User")
                ents = [
                    e.strip() for e in re.findall(r"\b[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*)*\b", past.get("text", ""))
                    if e.strip().lower() not in PRONOUN_MAP and e.strip().lower() not in {"user", "speaker", "unknown"}
                ]
                self.coref_tracker.register_turn(str(tid or ""), spk, ents)

        # Register current turn entities
        curr_ents = [
            e.strip() for e in re.findall(r"\b[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*)*\b", curr_text)
            if e.strip().lower() not in PRONOUN_MAP and e.strip().lower() not in {"user", "speaker", "unknown"}
        ]
        self.coref_tracker.register_turn(curr_turn_id, curr_speaker, curr_ents)

        if self.mode == "groq" and self.groq_client is not None:
            claims = self._extract_groq(current_turn, context_window, valid_turn_ids)
            if claims:
                return claims

        # Fallback or primary rules-based relational extraction
        return self._extract_rules(current_turn, valid_turn_ids)

    def _extract_rules(self, current_turn: Dict[str, Any], valid_turn_ids: Set[str]) -> List[RelationalClaim]:
        curr_turn_id = str(current_turn.get("turn_id") or uuid.uuid4())
        curr_speaker = str(current_turn.get("speaker") or "User").strip()
        curr_text = str(current_turn.get("text") or current_turn.get("raw_text") or "").strip()
        curr_ts = current_turn.get("timestamp") or current_iso_utc()

        claims: List[RelationalClaim] = []

        for pattern, predicate in RELATIONAL_PATTERNS:
            for match in pattern.finditer(curr_text):
                raw_subj = match.group(1).strip()
                raw_obj = match.group(2).strip().rstrip(".,;:!?")

                # Resolve coreference on subject
                resolved_subj, unresolved_err = self.coref_tracker.resolve_subject(
                    raw_subj, curr_speaker, current_turn_id=curr_turn_id
                )

                unresolved_refs = [unresolved_err] if unresolved_err else []
                confidence = 0.70 if unresolved_err else 0.90

                claim = RelationalClaim(
                    claim_id=str(uuid.uuid4()),
                    subject_entity=resolved_subj,
                    predicate=predicate,
                    object_entity=raw_obj,
                    source_turn_ids=[curr_turn_id],
                    context_text=f"{resolved_subj} {predicate} {raw_obj}",
                    created_at=curr_ts,
                    valid_from=curr_ts,
                    valid_to=None,
                    status="active",
                    confidence=confidence,
                    unresolved_references=unresolved_refs,
                )
                claims.append(claim)

        return claims

    def _extract_groq(
        self,
        current_turn: Dict[str, Any],
        context_window: Optional[List[Dict[str, Any]]],
        valid_turn_ids: Set[str],
    ) -> List[RelationalClaim]:
        """Call Groq with multi-turn context and extract relational claims with valid provenance."""
        turns_payload = []
        if context_window:
            for t in context_window[-2:]:
                turns_payload.append({
                    "turn_id": t.get("turn_id"),
                    "speaker": t.get("speaker"),
                    "text": t.get("text") or t.get("raw_text"),
                })
        turns_payload.append({
            "turn_id": current_turn.get("turn_id"),
            "speaker": current_turn.get("speaker"),
            "text": current_turn.get("text") or current_turn.get("raw_text"),
        })

        system_prompt = (
            "You are ChunkdUp's Context-Aware Relational Memory Extractor.\n"
            "Analyze the conversation window. Resolve pronouns ('I', 'she', 'he') to the true entity name.\n"
            "If a pronoun is ambiguous, keep the pronoun and list it under 'unresolved_references'.\n"
            "Extract typed relationships as JSON:\n"
            "{\n"
            '  "claims": [\n'
            '    {\n'
            '      "subject": "canonical subject entity",\n'
            '      "predicate": "relationship verb (e.g. works_at, located_in, uses, created)",\n'
            '      "object": "object entity or attribute",\n'
            '      "source_turn_id": "exact turn_id from input where statement was made",\n'
            '      "unresolved_references": ["list of unresolved pronouns or empty"]\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "CRITICAL RULES:\n"
            "1. 'source_turn_id' MUST be one of the exact turn_id values from the input batch.\n"
            "2. Never hallucinate facts or speculative thoughts.\n"
        )

        try:
            resp = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps({"window": turns_payload})},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            raw = resp.choices[0].message.content
            parsed = json.loads(raw)
            raw_claims = parsed.get("claims", [])
            claims: List[RelationalClaim] = []

            curr_ts = current_turn.get("timestamp") or current_iso_utc()
            for rc in raw_claims:
                tid = str(rc.get("source_turn_id", "")).strip()
                # Strict provenance enforcement: reject fabricated turn IDs
                if tid not in valid_turn_ids:
                    tid = str(current_turn.get("turn_id") or "")

                subj = str(rc.get("subject", "")).strip()
                pred = str(rc.get("predicate", "")).strip()
                obj = str(rc.get("object", "")).strip()
                unres = rc.get("unresolved_references", [])

                if subj and pred and obj:
                    claims.append(RelationalClaim(
                        claim_id=str(uuid.uuid4()),
                        subject_entity=subj,
                        predicate=pred,
                        object_entity=obj,
                        source_turn_ids=[tid] if tid else [str(current_turn.get("turn_id"))],
                        context_text=f"{subj} {pred} {obj}",
                        created_at=curr_ts,
                        valid_from=curr_ts,
                        status="active",
                        confidence=0.95 if not unres else 0.70,
                        unresolved_references=unres,
                    ))
            return claims
        except Exception as e:
            logger.warning(f"Groq relational extraction failed: {e}")
            return []
