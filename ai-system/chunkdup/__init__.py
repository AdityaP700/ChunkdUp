"""
ChunkDup - A memory management system for AI applications.
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

# Try to import policies, but handle case where they might not be in the package
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

__version__ = "1.0.0"

__all__ = [
    "Memory",
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