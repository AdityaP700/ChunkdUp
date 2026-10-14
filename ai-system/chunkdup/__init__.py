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
from .turn_repository import TurnRepository, InMemoryTurnRepository, PostgresTurnRepository, SourceTurnLog
from .extraction import ExtractedFact, GroqFactExtractor, TurnBuffer
from .decision_provider import (
    DecisionProvider,
    DefaultDecisionProvider,
    ExternalDecisionProvider,
    LayaDecisionProvider,
    get_decision_provider,
)
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
    "TurnRepository",
    "InMemoryTurnRepository",
    "PostgresTurnRepository",
    "SourceTurnLog",
    "ExtractedFact",
    "GroqFactExtractor",
    "TurnBuffer",
    "DecisionProvider",
    "DefaultDecisionProvider",
    "ExternalDecisionProvider",
    "LayaDecisionProvider",
    "get_decision_provider",
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
    if name == "GraphAugmentedRetriever":
        from .retrieval_graph_augmented import GraphAugmentedRetriever
        return GraphAugmentedRetriever
    if name == "ProfileGuidedArchiveRetriever":
        from .profile.profile_guided_retriever import ProfileGuidedArchiveRetriever
        return ProfileGuidedArchiveRetriever
    if name == "OptimizedHybridMemory":
        from .hybrid_memory_optimized import OptimizedHybridMemory
        return OptimizedHybridMemory
    if name == "QueryRouter":
        from .query_router import QueryRouter
        return QueryRouter
    raise AttributeError(name)