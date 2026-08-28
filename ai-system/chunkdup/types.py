# ai-system/chunkdup/types.py
"""
Type definitions for ChunkdUp SDK.
"""

from typing import TypedDict, Optional, List, Dict, Any


class MemoryDict(TypedDict, total=False):
    id: str
    key: str
    value: str
    type: str  # "project" | "preference" | "environment" | "tool" | etc.
    frequency: int
    importance: float
    status: str  # "active" | "inactive"
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]]


class SearchResult(TypedDict):
    memory: MemoryDict
    score: float


class RememberResult(TypedDict, total=False):
    decision: str  # "STORE" | "UPDATE" | "MERGE" | "IGNORE" | "processed"
    key: Optional[str]
    value: Optional[str]
    memories: Optional[List[Dict[str, Any]]]
    count: Optional[int]


class UpdateResult(TypedDict):
    decision: str
    key: str
    value: str


class DeleteResult(TypedDict):
    success: bool
    id: str


class StatsDict(TypedDict, total=False):
    total_memories: int
    store_type: str
    keys: List[str]
    active_memories: Optional[int]
    memories_by_type: Optional[Dict[str, int]]
    conversation_lines: Optional[int]
