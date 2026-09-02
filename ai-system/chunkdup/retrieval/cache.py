# ai-system/chunkdup/retrieval/cache.py
import time
from typing import List, Dict, Any, Optional

class RetrievalCache:
    """
    In-memory TTL cache for frequent / repeated search queries.
    """
    def __init__(self, ttl: int = 300, max_size: int = 1000):
        self.ttl = ttl
        self.max_size = max_size
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get(self, query: str) -> Optional[List[Dict[str, Any]]]:
        key = query.strip().lower()
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                return [dict(item) for item in entry["results"]]
            else:
                del self._cache[key]
        return None

    def set(self, query: str, results: List[Dict[str, Any]]) -> None:
        key = query.strip().lower()
        if len(self._cache) >= self.max_size:
            # Simple eviction: remove oldest key
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["timestamp"])
            del self._cache[oldest_key]

        self._cache[key] = {
            "timestamp": time.time(),
            "results": [dict(item) for item in results]
        }

    def clear(self) -> None:
        self._cache.clear()
