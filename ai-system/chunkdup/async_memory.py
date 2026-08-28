# ai-system/chunkdup/async_memory.py
"""
Async wrapper for ChunkdUp Memory SDK.
"""

import asyncio
from typing import List, Dict, Any, Optional
from .memory import Memory as SyncMemory


class AsyncMemory:
    """
    Async wrapper for ChunkdUp Memory SDK.

    Usage:
        memory = AsyncMemory(store="memory")
        await memory.remember("I use Python")
        results = await memory.retrieve("What language?")
    """

    def __init__(self, *args, **kwargs):
        self._sync = SyncMemory(*args, **kwargs)

    async def remember(self, text: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self._sync.remember, text)

    async def retrieve(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self._sync.retrieve, query, limit=limit)

    async def update(self, key: str, new_value: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self._sync.update, key, new_value)

    async def delete(self, memory_id: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self._sync.delete, memory_id)

    async def search(self, query: str, limit: int = 5, semantic: bool = True) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self._sync.search, query, limit=limit, semantic=semantic)

    async def get_all(self) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self._sync.get_all)

    async def get_history(self, key: str) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self._sync.get_history, key)

    async def get_stats(self) -> Dict[str, Any]:
        return await asyncio.to_thread(self._sync.get_stats)

    async def clear(self) -> None:
        return await asyncio.to_thread(self._sync.clear)
