"""Entity extraction used by memory indexing and retrieval."""

import re
import os
from typing import Any, Dict, List

class EntityExtractor:
    """Extract named technical entities with an optional spaCy model."""

    CUSTOM_ENTITIES = {
        "chunkdup": "ChunkdUp",
        "pgvector": "pgvector",
        "docker": "Docker",
        "postgresql": "PostgreSQL",
        "wsl2": "WSL2",
        "all-minilm-l6-v2": "all-MiniLM-L6-v2",
        "sentence-transformers": "sentence-transformers",
    }
    TECHNICAL_TERMS = {
        "vector search": "vector search",
        "embedding model": "embedding model",
        "embeddings": "embeddings",
        "backend": "backend",
        "database": "database",
        "runtime environment": "runtime environment",
    }

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.nlp = None
        spacy = None
        if os.getenv("CHUNKDUP_ENABLE_SPACY_NER") == "1":
            try:
                import spacy
            except ImportError:
                spacy = None
        if spacy is not None:
            try:
                self.nlp = spacy.load(model_name)
                ruler = self.nlp.add_pipe("entity_ruler", before="ner")
                ruler.add_patterns([
                    {"label": "TECH", "pattern": value}
                    for value in self.CUSTOM_ENTITIES.values()
                ])
            except Exception:
                self.nlp = None

    def extract(self, text: str) -> List[str]:
        if not text:
            return []

        entities: Dict[str, str] = {}

        # Deterministic domain vocabulary is stable even when spaCy is absent.
        lowered = text.casefold()
        for normalized, display in {**self.CUSTOM_ENTITIES, **self.TECHNICAL_TERMS}.items():
            if re.search(r"(?<![\w-])" + re.escape(normalized) + r"(?![\w-])", lowered):
                entities[normalized] = display

        if self.nlp is not None:
            for entity in self.nlp(text).ents:
                value = entity.text.strip()
                if value:
                    entities[value.casefold()] = value

        return list(entities.values())

    def annotate(self, memories: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        entities = self.extract(text)
        for memory in memories:
            memory.setdefault("entities", list(entities))
            memory.setdefault("contextual_text", f"Conversation source: {text.strip()}")
        return memories