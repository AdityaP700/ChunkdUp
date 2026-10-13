# ai-system/chunkdup/retrieval/grounding_gate.py
"""
Grounding Gate and Evidence Provenance for ChunkdUp Retrieval.

Provides:
- GroundedRetrievalResult: Typed retrieval result with source turn provenance,
  speaker alignment, contradiction tracking, and sufficiency flags.
- GroundingGate: Calibrated relevance gate that adjusts abstention thresholds
  based on memory store scale (pool size N), verifies speaker/entity alignment,
  and handles contradictions/supersession.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class GroundedRetrievalResult:
    """Typed retrieval candidate with provenance and sufficiency attribution."""

    id: str
    memory: Dict[str, Any]
    score: float
    score_field: str
    source_turns: List[str] = field(default_factory=list)
    confidence: float = 0.0
    is_sufficient: bool = True
    contradictions: List[str] = field(default_factory=list)
    speaker_aligned: bool = True

    def __getitem__(self, item: str) -> Any:
        """Backward-compatibility with dictionary indexing."""
        if item in self.memory:
            return self.memory[item]
        if hasattr(self, item):
            return getattr(self, item)
        if item == "_source_turns":
            return self.source_turns
        if item == "_is_sufficient":
            return self.is_sufficient
        if item == "_confidence":
            return self.confidence
        if item == "_contradictions":
            return self.contradictions
        if item == "_speaker_aligned":
            return self.speaker_aligned
        raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        """Backward-compatibility with dictionary get()."""
        if item in self.memory:
            return self.memory[item]
        if hasattr(self, item):
            return getattr(self, item)
        if item == "_source_turns":
            return self.source_turns
        if item == "_is_sufficient":
            return self.is_sufficient
        if item == "_confidence":
            return self.confidence
        if item == "_contradictions":
            return self.contradictions
        if item == "_speaker_aligned":
            return self.speaker_aligned
        return default

    def __contains__(self, item: str) -> bool:
        return item in self.memory or hasattr(self, item)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to enriched dictionary for legacy consumers."""
        d = dict(self.memory)
        d["id"] = self.id
        d[self.score_field] = self.score
        d["_score"] = self.score
        d["_score_field"] = self.score_field
        d["_source_turns"] = list(self.source_turns)
        d["_confidence"] = self.confidence
        d["_is_sufficient"] = self.is_sufficient
        d["_contradictions"] = list(self.contradictions)
        d["_speaker_aligned"] = self.speaker_aligned
        return d


class GroundingGate:
    """Scale-aware relevance and grounding verification gate."""

    NON_SPEAKER_WORDS: Set[str] = {
        "what", "which", "where", "when", "why", "how", "who", "whom", "whose",
        "is", "was", "are", "were", "be", "been", "being",
        "do", "does", "did", "have", "has", "had", "used", "made", "created",
        "the", "a", "an", "this", "that", "these", "those",
    }

    # Patterns to extract targeted speaker from query (e.g. "What did Alice say?", "Bob's favorite editor")
    SPEAKER_PATTERNS = [
        re.compile(r"\b(?:what did|did|has|is|was|does)\s+([A-Z][a-z]+)\b"),
        re.compile(r"\b([A-Z][a-z]+)'s\b"),
        re.compile(r"\b([A-Z][a-z]+)\s+(?:said|thinks|prefers|likes|uses|works|lives)\b"),
    ]

    def __init__(
        self,
        score_field: str,
        threshold: float,
        enable_scale_calibration: bool = True,
        calibration_reference_size: int = 12,
    ):
        self.score_field = score_field
        self.threshold = threshold
        self.base_threshold = threshold
        self.enable_scale_calibration = enable_scale_calibration
        self.calibration_reference_size = calibration_reference_size

    def get_calibrated_threshold(self, store_size: Optional[int] = None) -> float:
        """Compute the scale-adjusted threshold based on corpus size N."""
        if not self.enable_scale_calibration or store_size is None or store_size <= 20:
            return self.threshold

        # For cross-encoder logits:
        # At N=12, tau ~ -10.95. As N grows to ~500, random candidate logit
        # maximum rises to around -4.0 .. -5.0. Scaling logarithmic slope
        # keeps false acceptance bounded while maintaining recall.
        if self.score_field == "_rerank_score":
            scale_offset = 3.5 * math.log10(max(1.0, store_size / self.calibration_reference_size))
            return min(-2.5, self.threshold + scale_offset)

        # For dense cosine similarity:
        if self.score_field == "_similarity":
            scale_offset = 0.05 * math.log10(max(1.0, store_size / self.calibration_reference_size))
            return min(0.45, self.threshold + scale_offset)

        return self.threshold

    def extract_query_speaker(self, query: str) -> Optional[str]:
        """Attempt to extract target speaker from query if specifically mentioned."""
        for pattern in self.SPEAKER_PATTERNS:
            match = pattern.search(query)
            if match:
                candidate = match.group(1).strip()
                if candidate.lower() not in self.NON_SPEAKER_WORDS:
                    return candidate
        return None

    def extract_source_turns(self, candidate: Dict[str, Any]) -> List[str]:
        """Extract source turn IDs (provenance) from candidate memory."""
        turns: Set[str] = set()
        if candidate.get("dia_id"):
            turns.add(str(candidate["dia_id"]))
        if candidate.get("source_turns") and isinstance(candidate["source_turns"], (list, tuple)):
            for t in candidate["source_turns"]:
                turns.add(str(t))

        # Check id format like "conv-26:D1:5" or "D1:5"
        cid = str(candidate.get("id", ""))
        if ":" in cid:
            parts = cid.split(":")
            if len(parts) == 2 and parts[1].startswith("D"):
                turns.add(parts[1])
            elif len(parts) == 3 and parts[1].startswith("D"):
                turns.add(f"{parts[1]}:{parts[2]}")

        # Check contextual_text or source
        context = candidate.get("contextual_text") or candidate.get("source") or ""
        match = re.search(r"\b(D\d+:\d+)\b", str(context))
        if match:
            turns.add(match.group(1))

        return sorted(turns)

    def verify_speaker_alignment(self, query_speaker: Optional[str], candidate: Dict[str, Any]) -> bool:
        """Verify candidate speaker aligns with query target speaker if specified."""
        if not query_speaker:
            return True

        target = query_speaker.lower()
        cand_speaker = str(candidate.get("speaker", "")).lower()

        # If candidate has explicit speaker tag
        if cand_speaker:
            if cand_speaker == target:
                return True
            # Check if target speaker is mentioned in value or contextual text
            val = str(candidate.get("value", "")).lower()
            context = str(candidate.get("contextual_text", "")).lower()
            if target in val or target in context:
                return True
            return False

        # If candidate has no explicit speaker tag, check if contextual text references speaker
        context = str(candidate.get("contextual_text", "")).lower()
        if f"{target} said" in context or f"by {target}" in context:
            return True

        return True

    def filter_and_ground(
        self,
        query: str,
        results: List[Dict[str, Any]],
        store_size: Optional[int] = None,
        k: int = 5,
    ) -> List[GroundedRetrievalResult]:
        """Filter candidates by calibrated threshold and enrich with grounding provenance."""
        if not results:
            return []

        effective_threshold = self.get_calibrated_threshold(store_size)
        top_score = float(results[0].get(self.score_field, 0.0))

        # Hard abstention when top candidate fails calibrated threshold
        if top_score < effective_threshold:
            return []

        query_speaker = self.extract_query_speaker(query)
        grounded: List[GroundedRetrievalResult] = []

        seen_keys: Dict[str, GroundedRetrievalResult] = {}

        for cand in results:
            score = float(cand.get(self.score_field, 0.0))
            if score < effective_threshold:
                continue

            cid = str(cand.get("id") or cand.get("key") or "")
            source_turns = self.extract_source_turns(cand)
            speaker_aligned = self.verify_speaker_alignment(query_speaker, cand)

            contradictions: List[str] = []
            if cand.get("status") in ("superseded", "inactive"):
                contradictions.append(f"Memory status is {cand.get('status')}")
            if cand.get("superseded_by"):
                contradictions.append(f"Superseded by memory {cand.get('superseded_by')}")

            # Check duplicate / contradictory keys in the candidate pool
            key = cand.get("key")
            if key:
                if key in seen_keys:
                    prior = seen_keys[key]
                    if prior.memory.get("value") != cand.get("value"):
                        contradictions.append(
                            f"Contradicts higher-ranked memory {prior.id} for key '{key}'"
                        )
                else:
                    pass

            is_sufficient = (
                (score >= effective_threshold)
                and speaker_aligned
                and not bool(cand.get("superseded_by"))
                and cand.get("status") != "superseded"
            )

            # Normalized confidence: ratio of score margin above threshold
            denom = max(1.0, abs(top_score - effective_threshold) + 1.0)
            confidence = max(0.0, min(1.0, (score - effective_threshold + 1.0) / denom))

            item = GroundedRetrievalResult(
                id=cid,
                memory=dict(cand),
                score=score,
                score_field=self.score_field,
                source_turns=source_turns,
                confidence=round(confidence, 4),
                is_sufficient=is_sufficient,
                contradictions=contradictions,
                speaker_aligned=speaker_aligned,
            )

            if key and key not in seen_keys:
                seen_keys[key] = item

            grounded.append(item)

        # Sort: sufficient candidates first, then by score
        grounded.sort(
            key=lambda r: (1 if r.is_sufficient else 0, r.score),
            reverse=True,
        )
        return grounded[:k]

    def filter(self, results: List[Dict[str, Any]], store_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Legacy filter interface returning dicts with calibrated threshold."""
        if not results:
            return []
        effective_threshold = self.get_calibrated_threshold(store_size)
        top = float(results[0].get(self.score_field, 0.0))
        if top < effective_threshold:
            return []
        return [r for r in results if float(r.get(self.score_field, 0.0)) >= effective_threshold]
