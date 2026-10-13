# ai-system/chunkdup/entity/__init__.py
"""
Entity linking and graph-assisted memory expansion for ChunkdUp.
"""

from .entity_graph import Entity, EntityRelation, EntityGraph

__all__ = [
    "Entity",
    "EntityRelation",
    "EntityGraph",
]
