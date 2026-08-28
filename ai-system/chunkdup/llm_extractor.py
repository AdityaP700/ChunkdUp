# ai-system/chunkdup/llm_extractor.py
"""
LLM Fallback Extractor for complex memory statements.
"""

import json
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class LLMExtractor:
    """Extracts structured memories using LLM fallback for complex or ambiguous statements."""

    def __init__(self, llm_caller=None):
        self.llm = llm_caller

    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Fallback extraction using LLM when available."""
        if not self.llm:
            return []

        prompt = (
            "Extract personal user facts/memories from the following text as JSON.\n"
            "Only extract clear facts about the user (e.g. tech stack, preferences, role, company, learning, projects).\n"
            "Ignore general statements, opinions, weather, or temporary facts.\n\n"
            f"Text: \"{text}\"\n\n"
            "Return JSON format:\n"
            "{\n"
            '  "memories": [\n'
            '    {"type": "environment|project|preference|tool|employment|role|learning", "key": "key_name", "value": "extracted_value"}\n'
            "  ]\n"
            "}"
        )

        try:
            raw_response = self.llm.generate(prompt)
            # Find JSON block
            start = raw_response.find("{")
            end = raw_response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw_response[start:end])
                memories = []
                for item in data.get("memories", []):
                    memories.append({
                        "id": str(uuid.uuid4()),
                        "type": item.get("type", "general"),
                        "key": item.get("key"),
                        "value": item.get("value"),
                        "frequency": 1,
                        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "source": "llm_fallback",
                        "status": "active"
                    })
                return memories
        except Exception as e:
            logger.warning(f"LLM fallback extraction error: {e}")

        return []
