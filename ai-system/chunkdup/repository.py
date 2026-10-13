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

    @abstractmethod
    def clear(self) -> None:
        """Clear all memories."""
        pass