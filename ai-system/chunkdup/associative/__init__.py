# ai-system/chunkdup/associative/__init__.py
"""
Evidence-Preserving Associative Memory Module for ChunkdUp.
Integrates Experience Memory (raw turns), Semantic Memory (atomic facts with provenance),
Associative Memory (typed entity/event/fact links), and bounded expansion retrieval.
"""

from .schema import (
    ExperienceTurn,
    SemanticFact,
    AssociativeLink,
    AssociativeLinkType,
    MemoryRepresentationBundle,
)
from .builder import AssociativeMemoryBuilder
from .graph import AssociativeGraph, BoundedGraphExpansion
from .retrievers import (
    GrepRetriever,
    BM25Retriever,
    HybridCEControlRetriever,
    BoundedAssociativeRetriever,
    ConsolidatedSemanticRetriever,
    ConsolidatedAssociativeRetriever,
    QueryRoutedAssociativeRetriever,
    RetrievalResult,
)

__all__ = [
    "ExperienceTurn",
    "SemanticFact",
    "AssociativeLink",
    "AssociativeLinkType",
    "MemoryRepresentationBundle",
    "AssociativeGraph",
    "BoundedGraphExpansion",
    "AssociativeMemoryBuilder",
    "GrepRetriever",
    "BM25Retriever",
    "HybridCEControlRetriever",
    "BoundedAssociativeRetriever",
    "ConsolidatedSemanticRetriever",
    "ConsolidatedAssociativeRetriever",
    "QueryRoutedAssociativeRetriever",
    "RetrievalResult",
]
