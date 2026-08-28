"""
ChunkdUp — Persistent memory for AI agents.
"""

from .memory import (
    Memory,
    MemoryExtractor,
    MemoryScorer,
    MemoryRanker,
    MemoryRepository,
    MemoryManager,
    DecisionEngine,
    PromptBuilder,
    MemoryRetriever
)
from .async_memory import AsyncMemory
from .types import (
    MemoryDict,
    SearchResult,
    RememberResult,
    UpdateResult,
    DeleteResult,
    StatsDict
)

try:
    from .policies import (
        Decision,
        BasePolicy,
        EnvironmentPolicy,
        PreferencePolicy,
        ToolPolicy,
        ProjectPolicy,
        PolicyFactory
    )
except ImportError:
    from policies import (
        Decision,
        BasePolicy,
        EnvironmentPolicy,
        PreferencePolicy,
        ToolPolicy,
        ProjectPolicy,
        PolicyFactory
    )

__version__ = "0.2.0"

__all__ = [
    "Memory",
    "AsyncMemory",
    "MemoryDict",
    "SearchResult",
    "RememberResult",
    "UpdateResult",
    "DeleteResult",
    "StatsDict",
    "MemoryExtractor",
    "MemoryScorer",
    "MemoryRanker",
    "MemoryRepository",
    "MemoryManager",
    "DecisionEngine",
    "PromptBuilder",
    "MemoryRetriever",
    "Decision",
    "BasePolicy",
    "EnvironmentPolicy",
    "PreferencePolicy",
    "ToolPolicy",
    "ProjectPolicy",
    "PolicyFactory",
]