# Implement the Contract
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

class MemoryRepository(ABC):
    """The CONTRACT that all repositories must follow."""

    @abstractmethod
    def save(self, memory: Dict[str, Any]) -> None:
        """Store a new memory."""
        pass

    @abstractmethod
    def get_by_key(self, key: str) -> Optional[Dict[str, Any]]:
        """Get active memory by its key."""
        pass

    @abstractmethod
    def update(self, memory: Dict[str, Any]) -> None:
        """Update an existing memory."""
        pass

    @abstractmethod
    def delete(self, memory_id: str) -> None:
        """Soft delete a memory."""
        pass

    @abstractmethod
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all active memories."""
        pass

    @abstractmethod
    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search for memories matching a query."""
        pass

    @abstractmethod
    def merge(self, key: str, memory: Dict[str, Any]) -> None:
        """Merge a new memory with existing one (increment frequency)."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all memories."""
        pass