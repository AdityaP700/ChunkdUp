# ai-system/chunkdup/profile/__init__.py
"""
GPT-Inspired 3-Tier Profile, Recent Context, and Searchable Archive Memory.
"""

from chunkdup.profile.schema import ProfileFact
from chunkdup.profile.profile_memory import (
    PersistentProfile,
    RecentContextBuffer,
    ProfileConsolidationEngine,
    ProfileMemory,
)

__all__ = [
    "ProfileFact",
    "PersistentProfile",
    "RecentContextBuffer",
    "ProfileConsolidationEngine",
    "ProfileMemory",
]
