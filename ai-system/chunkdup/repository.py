# ai-system/chunkdup/repository.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class MemoryRepository(ABC):
    """The CONTRACT that all repositories must follow."""

    @abstractmethod
    def save(self, memory: Dict[str, Any]) -> None:
        """Store a new memory."""
        pass

    @abstractmethod
    def get_by_key(self, key: str, as_of: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """Get memory by its key, optionally as of a specific point in time."""
        pass

    def get_by_id(self, memory_id: str, as_of: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """Get memory by ID, optionally as of a specific point in time."""
        return None

    @abstractmethod
    def update(self, memory: Dict[str, Any]) -> None:
        """Update an existing memory with append-only supersession."""
        pass

    @abstractmethod
    def delete(self, memory_id: str) -> None:
        """Soft delete a memory."""
        pass

    @abstractmethod
    def get_all(self, as_of: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Get memories, optionally as of a specific point in time."""
        pass

    @abstractmethod
    def search(self, query: str, limit: Optional[int] = None, as_of: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Search for memories matching a query."""
        pass

    @abstractmethod
    def merge(self, key: str, memory: Dict[str, Any]) -> None:
        """Merge a new memory with existing one (increment frequency)."""
        pass

    def get_history(self, key: str) -> List[Dict[str, Any]]:
        """Get chronologically ordered history of a memory key."""
        return []

    def record_decision(self, event: Dict[str, Any]) -> None:
        """Record an append-only policy decision event (STORE, MERGE, UPDATE, IGNORE)."""
        pass

    def get_decision(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a decision event by its unique ID."""
        return None

    def get_decisions_by_turn(self, source_turn_id: str) -> List[Dict[str, Any]]:
        """Retrieve all decision events originating from a specific source dialogue turn."""
        return []

    def get_decisions_by_memory(self, memory_id: str) -> List[Dict[str, Any]]:
        """Retrieve all decision events affecting a specific target memory ID."""
        return []

    def get_all_decisions(self) -> List[Dict[str, Any]]:
        """Retrieve all recorded decision events in chronological order."""
        return []

    @abstractmethod
    def clear(self) -> None:
        """Clear all memories."""
        pass