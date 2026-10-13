# ai-system/chunkdup/extraction/__init__.py
"""
ChunkdUp Fact Extraction Module.
"""

from .schema import ExtractedFact, ALLOWED_MEMORY_TYPES, is_turn_eligible
from .groq_extractor import GroqFactExtractor
from .turn_buffer import TurnBuffer

__all__ = [
    "ExtractedFact",
    "ALLOWED_MEMORY_TYPES",
    "is_turn_eligible",
    "GroqFactExtractor",
    "TurnBuffer",
]
