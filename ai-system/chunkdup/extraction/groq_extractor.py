# ai-system/chunkdup/extraction/groq_extractor.py
"""
Batched, schema-validated fact extractor using Groq with graceful fallback.
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional

from .schema import ExtractedFact, is_turn_eligible

logger = logging.getLogger(__name__)


EXTRACTION_SYSTEM_PROMPT = """You are a precision fact extraction engine for an AI persistent memory system.
Your job is to extract durable personal facts and preferences about the user from dialogue turns.

CRITICAL INSTRUCTIONS:
1. ONLY extract enduring facts about the user (e.g. tools, programming languages, operating systems, roles, projects, employers, preferences).
2. DO NOT extract transient status, greetings, conversation meta, opinions on external topics, or questions.
3. Every extracted fact MUST include:
   - "key": normalized snake_case attribute name (e.g. "programming_language", "editor", "os", "company", "project_name")
   - "value": concise factual value statement (e.g. "Python", "Neovim", "Linux", "ChunkdUp")
   - "type": EXACTLY one of: ["environment", "project", "tool", "preference", "employment", "role", "learning", "general"]
   - "confidence": float between 0.0 and 1.0 reflecting confidence
   - "entities": list of canonical entity names mentioned (e.g. ["Python"])
   - "turn_id": the exact turn_id of the dialogue turn containing this fact.

Return STRICT JSON ONLY in the following format:
{
  "facts": [
    {
      "turn_id": "string",
      "key": "string",
      "value": "string",
      "type": "string",
      "confidence": 0.95,
      "entities": ["string"]
    }
  ]
}
If no facts are present, return: {"facts": []}"""


class GroqFactExtractor:
    """
    Level 2 & Level 3 Fact Extractor.
    Extracts structured facts from dialogue turns via Groq API with validation and fallback.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        client: Optional[Any] = None,
        fallback_extractor: Optional[Any] = None,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
    ):
        self.model = model
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._client = client
        self._api_key = api_key or os.getenv("GROQ_API_KEY")

        # Lazy fallback extractor
        if fallback_extractor is not None:
            self.fallback_extractor = fallback_extractor
        else:
            try:
                from ..memory import MemoryExtractor
                self.fallback_extractor = MemoryExtractor()
            except Exception:
                self.fallback_extractor = None

    @property
    def client(self):
        """Lazy-initialize Groq client."""
        if self._client is not None:
            return self._client
        if self._api_key:
            try:
                import groq
                self._client = groq.Groq(api_key=self._api_key)
                return self._client
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")
                return None
        return None

    def is_available(self) -> bool:
        """Check if Groq extraction is configured and available."""
        return self.client is not None

    def _call_groq_with_retry(self, user_content: str) -> Optional[str]:
        """Execute chat completion with retry and exponential backoff on 429."""
        client = self.client
        if not client:
            return None

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                return response.choices[0].message.content
            except Exception as exc:
                is_rate_limit = "429" in str(exc) or "RateLimit" in exc.__class__.__name__
                if is_rate_limit and attempt < self.max_retries - 1:
                    sleep_time = self.backoff_factor * (2 ** attempt)
                    logger.warning(
                        f"Groq 429 rate limit reached. Backing off for {sleep_time:.2f}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(sleep_time)
                else:
                    logger.warning(f"Groq API call failed: {exc}")
                    if attempt == self.max_retries - 1:
                        return None
        return None

    def _parse_and_validate_response(
        self,
        raw_response: str,
        default_turn_id: Optional[str] = None,
    ) -> List[ExtractedFact]:
        """Parse JSON response and validate all facts against ExtractedFact schema."""
        if not raw_response:
            return []

        try:
            start = raw_response.find("{")
            end = raw_response.rfind("}") + 1
            if start < 0 or end <= start:
                return []
            data = json.loads(raw_response[start:end])
        except Exception as e:
            logger.warning(f"Failed to parse Groq JSON response: {e}")
            return []

        facts_raw = data.get("facts", [])
        validated_facts = []

        for item in facts_raw:
            try:
                turn_id = item.get("turn_id") or default_turn_id
                if not turn_id:
                    continue

                fact = ExtractedFact(
                    key=item.get("key", ""),
                    value=item.get("value", ""),
                    type=item.get("type", "general"),
                    confidence=float(item.get("confidence", 0.9)),
                    entities=item.get("entities", []),
                    source_turn_id=turn_id,
                )
                validated_facts.append(fact)
            except Exception as val_err:
                logger.debug(f"Fact failed schema validation: {val_err} (raw: {item})")

        return validated_facts

    def _fallback_extract(self, turn: Dict[str, Any]) -> List[ExtractedFact]:
        """Execute regex/spaCy fallback extraction and validate into ExtractedFact."""
        if not self.fallback_extractor:
            return []

        raw_text = turn.get("raw_text", "")
        turn_id = turn.get("turn_id", "")
        try:
            extracted_list = self.fallback_extractor.extract(raw_text)
            validated = []
            for item in extracted_list:
                try:
                    fact = ExtractedFact(
                        key=item.get("key", ""),
                        value=item.get("value", ""),
                        type=item.get("type", "general"),
                        confidence=0.85,
                        entities=item.get("entities", []),
                        source_turn_id=turn_id,
                    )
                    validated.append(fact)
                except Exception:
                    continue
            return validated
        except Exception as e:
            logger.warning(f"Fallback extraction failed: {e}")
            return []

    def extract_turn(self, turn: Dict[str, Any]) -> List[ExtractedFact]:
        """
        Extract facts from a single dialogue turn.
        Checks eligibility -> calls Groq if available -> falls back if unavailable.
        """
        if not is_turn_eligible(turn):
            return []

        turn_id = turn.get("turn_id", "")
        raw_text = turn.get("raw_text", "")
        speaker = turn.get("speaker", "user")

        if self.is_available():
            user_content = json.dumps({
                "turns": [{"turn_id": turn_id, "speaker": speaker, "text": raw_text}]
            })
            resp = self._call_groq_with_retry(user_content)
            if resp:
                facts = self._parse_and_validate_response(resp, default_turn_id=turn_id)
                if facts:
                    return facts

        # Fallback path if Groq is unavailable or returned 0 facts
        return self._fallback_extract(turn)

    def extract_batch(self, turns: List[Dict[str, Any]]) -> List[ExtractedFact]:
        """
        Extract facts from a batch of dialogue turns.
        Preserves provenance by attaching the correct source_turn_id to each fact.
        """
        eligible = [t for t in turns if is_turn_eligible(t)]
        if not eligible:
            return []

        if self.is_available():
            batch_payload = [
                {
                    "turn_id": t.get("turn_id"),
                    "speaker": t.get("speaker", "user"),
                    "text": t.get("raw_text", ""),
                }
                for t in eligible
            ]
            user_content = json.dumps({"turns": batch_payload})
            resp = self._call_groq_with_retry(user_content)
            if resp:
                facts = self._parse_and_validate_response(resp)
                if facts:
                    return facts

        # Fallback path
        all_facts = []
        for t in eligible:
            all_facts.extend(self._fallback_extract(t))
        return all_facts
