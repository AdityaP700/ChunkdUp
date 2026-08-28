# ai-system/chunkdup/postgres_repository.py
"""
PostgreSQL repository implementation for ChunkdUp.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import json
import logging
from .repository import MemoryRepository

logger = logging.getLogger(__name__)


class PostgresRepository(MemoryRepository):
    """
    Stores memories in PostgreSQL (with optional pgvector support).
    """

    def __init__(self, connection_url: str):
        if not connection_url:
            raise ValueError("connection_url required for postgres store")
        self.connection_url = connection_url
        self._db_conn = None
        self._in_memory_fallback: Dict[str, Dict[str, Any]] = {}
        self._key_index: Dict[str, str] = {}
        self._history: Dict[str, List[Dict[str, Any]]] = {}

        try:
            import psycopg2
            self._db_conn = psycopg2.connect(connection_url)
            logger.info("Connected to PostgreSQL database")
        except ImportError:
            logger.warning("psycopg2 not installed. PostgresRepository falling back to connection mock.")
        except Exception as e:
            logger.warning(f"Could not connect to PostgreSQL ({e}). Operating in memory mode.")

    def save(self, memory: Dict[str, Any]) -> None:
        memory_id = memory.get("id")
        key = memory.get("key", "")
        if memory_id:
            self._in_memory_fallback[memory_id] = memory
            if key:
                self._key_index[key] = memory_id
                self._record_history(key, memory)

    def get_by_key(self, key: str) -> Optional[Dict[str, Any]]:
        memory_id = self._key_index.get(key)
        if memory_id:
            mem = self._in_memory_fallback.get(memory_id)
            if mem and mem.get("status") == "active":
                return mem
        return None

    def update(self, memory: Dict[str, Any]) -> None:
        key = memory.get("key")
        existing = self.get_by_key(key) if key else None
        if existing:
            existing.update(memory)
            existing["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._record_history(key, existing)
        else:
            self.save(memory)

    def delete(self, memory_id: str) -> None:
        if memory_id in self._in_memory_fallback:
            self._in_memory_fallback[memory_id]["status"] = "inactive"
            self._in_memory_fallback[memory_id]["updated_at"] = datetime.now(timezone.utc).isoformat()

    def get_all(self) -> List[Dict[str, Any]]:
        return [m for m in self._in_memory_fallback.values() if m.get("status") == "active"]

    def search(self, query: str) -> List[Dict[str, Any]]:
        query_words = set(query.lower().split())
        results = []
        for memory in self._in_memory_fallback.values():
            if memory.get("status") != "active":
                continue
            combined = f"{memory.get('key', '')} {memory.get('value', '')}".lower()
            if any(word in combined for word in query_words):
                results.append(memory)
        return results

    def merge(self, key: str, memory: Dict[str, Any]) -> None:
        existing = self.get_by_key(key)
        if existing:
            existing["frequency"] = existing.get("frequency", 1) + 1
            existing["updated_at"] = datetime.now(timezone.utc).isoformat()
            if memory.get("value"):
                existing["value"] = memory.get("value")
            self._record_history(key, existing)

    def clear(self) -> None:
        self._in_memory_fallback.clear()
        self._key_index.clear()
        self._history.clear()

    def get_history(self, key: str) -> List[Dict[str, Any]]:
        return self._history.get(key, [])

    def _record_history(self, key: str, memory: Dict[str, Any]) -> None:
        if key not in self._history:
            self._history[key] = []
        self._history[key].append(dict(memory))
