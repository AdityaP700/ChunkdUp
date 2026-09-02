# chunkdup/in_memory_repository.py
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import uuid
from .repository import MemoryRepository

class InMemoryRepository(MemoryRepository):
    """Stores memories in RAM (great for testing)."""

    def __init__(self):
        self._memories: Dict[str, Dict] = {}  # id -> memory
        self._key_index: Dict[str, str] = {}  # key -> id

    def save(self, memory: Dict[str, Any]) -> None:
        memory_id = memory.get("id", str(uuid.uuid4()))
        memory["id"] = memory_id
        memory["created_at"] = datetime.now(timezone.utc).isoformat()
        memory["updated_at"] = memory["created_at"]
        memory["status"] = "active"

        self._memories[memory_id] = memory
        self._key_index[memory["key"]] = memory_id

    def get_by_key(self, key: str) -> Optional[Dict[str, Any]]:
        memory_id = self._key_index.get(key)
        if memory_id:
            memory = self._memories.get(memory_id)
            if memory and memory.get("status") == "active":
                return memory
        return None

    def update(self, memory: Dict[str, Any]) -> None:
        memory_id = memory.get("id")
        if not memory_id:
            raise ValueError("Memory ID required")

        # Preserve immutable fields
        existing = self._memories.get(memory_id)
        if not existing:
            raise ValueError(f"Memory {memory_id} not found")

        memory["created_at"] = existing["created_at"]
        memory["frequency"] = existing.get("frequency", 1)
        memory["updated_at"] = datetime.now(timezone.utc).isoformat()

        self._memories[memory_id] = memory

    def delete(self, memory_id: str) -> None:
        if memory_id in self._memories:
            self._memories[memory_id]["status"] = "inactive"
            self._memories[memory_id]["updated_at"] = datetime.now(timezone.utc).isoformat()

    def get_all(self) -> List[Dict[str, Any]]:
        return [m for m in self._memories.values() if m.get("status") == "active"]

    def search(self, query: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        query_words = set(query.lower().split())
        results = []

        for memory in self._memories.values():
            if memory.get("status") != "active":
                continue

            contextual = memory.get("contextual_text", "")
            combined = f"{memory['key']} {memory['value']} {memory.get('type', '')} {contextual}".lower()
            if any(word in combined for word in query_words):
                results.append(memory)

        if limit:
            return results[:limit]
        return results

    def merge(self, key: str, memory: Dict[str, Any]) -> None:
        existing = self.get_by_key(key)
        if existing:
            existing["frequency"] = existing.get("frequency", 1) + 1
            existing["updated_at"] = datetime.now(timezone.utc).isoformat()
            if existing["value"] != memory.get("value"):
                existing["value"] = memory.get("value")

    def clear(self) -> None:
        self._memories.clear()
        self._key_index.clear()