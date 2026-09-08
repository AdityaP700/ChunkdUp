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
from .in_memory_repository import InMemoryRepository
from .entity_extractor import EntityExtractor
try:
    from .postgres_repository import PostgresRepository
except ImportError:
    PostgresRepository = None
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
    "EntityExtractor",
    "InMemoryRepository",
    "PostgresRepository",
    "Decision",
    "BasePolicy",
    "EmbeddingGenerator",
    "EnvironmentPolicy",
    "PreferencePolicy",
    "ToolPolicy",
    "ProjectPolicy",
    "PolicyFactory",
]


def __getattr__(name):
    if name == "EmbeddingGenerator":
        from .embeddings import EmbeddingGenerator
        return EmbeddingGenerator
    raise AttributeError(name)