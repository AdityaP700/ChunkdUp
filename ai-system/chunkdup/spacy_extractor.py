# ai-system/chunkdup/spacy_extractor.py
"""
spaCy Named Entity Recognition & Linguistic Extractor for ChunkdUp.
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    import spacy
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    spacy = None


class SpacyExtractor:
    """Extracts structured memories using spaCy NER and linguistic analysis."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.nlp = None
        if NLP_AVAILABLE:
            try:
                self.nlp = spacy.load(model_name)
                logger.info(f"Loaded spaCy model: {model_name}")
            except Exception as e:
                logger.warning(f"Could not load spaCy model {model_name}: {e}")

    def is_available(self) -> bool:
        return self.nlp is not None

    def extract(self, text: str) -> List[Dict[str, Any]]:
        if not self.is_available():
            return []

        memories = []
        doc = self.nlp(text)

        # Linguistic / NER heuristics
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_lower = sent_text.lower()

            # 1. Employment detection (e.g. "I work at Google", "employed by Microsoft")
            if "work at" in sent_lower or "working at" in sent_lower or "employed at" in sent_lower:
                for ent in sent.ents:
                    if ent.label_ in ("ORG", "PRODUCT", "GPE"):
                        memories.append(self._create_memory("employment", "company", ent.text))
                        break

            # 2. Role detection (e.g. "I'm a software engineer", "I work as a designer")
            if "i'm a " in sent_lower or "i am a " in sent_lower or "work as a " in sent_lower:
                for chunk in sent.noun_chunks:
                    chunk_text = chunk.text.strip()
                    chunk_lower = chunk_text.lower()
                    if chunk_lower.startswith(("a ", "an ")):
                        role_val = chunk_text[chunk_text.find(" ") + 1:].strip()
                        if role_val and role_val.lower() not in ("user", "person", "human"):
                            memories.append(self._create_memory("role", "role", role_val))
                            break

            # 3. Learning detection (e.g. "I'm learning Rust", "studying Python")
            if "learning " in sent_lower or "studying " in sent_lower:
                for token in sent:
                    if token.dep_ in ("dobj", "pobj") and token.head.lemma_ in ("learn", "study"):
                        val = token.text.strip()
                        memories.append(self._create_memory("learning", "learning", val))
                        break

            # 4. Project detection (e.g. "I'm building a web app called ChunkdUp")
            if "building " in sent_lower or "working on " in sent_lower or "developing " in sent_lower:
                # Check for named entities or capitalized terms after "called" / "named"
                if "called " in sent_lower or "named " in sent_lower:
                    idx = sent_lower.find("called ") if "called " in sent_lower else sent_lower.find("named ")
                    parts = sent_text[idx:].split()
                    if len(parts) > 1:
                        proj_val = parts[1].strip(".,!?\"'")
                        memories.append(self._create_memory("project", "project_name", proj_val))

            # 5. Preference / Tool detection (e.g. "I prefer Neovim", "My favorite editor is Neovim")
            if "prefer " in sent_lower:
                for token in sent:
                    if token.dep_ == "dobj" and token.head.lemma_ == "prefer":
                        val = token.text.strip(".,!?\"'")
                        memories.append(self._create_memory("tool", "editor", val))
                        break

        return memories

    def _create_memory(self, mem_type: str, key: str, value: str) -> Dict[str, Any]:
        return {
            "id": str(uuid.uuid4()),
            "type": mem_type,
            "key": key,
            "value": value,
            "frequency": 1,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source": "spacy_ner",
            "status": "active"
        }
