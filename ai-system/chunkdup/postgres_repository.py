# ai-system/chunkdup/repositories/postgres_repository.py
from typing import Dict, List, Optional, Any
from sqlalchemy import create_engine, select, update, delete, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from datetime import datetime, timezone
from .embeddings import EmbeddingGenerator
import uuid
import os

try:
    from .repository import MemoryRepository
except ImportError:
    from repository import MemoryRepository
try:
    from .models import MemoryModel, MemoryHistoryModel, Base
except ImportError:
    from models import MemoryModel, MemoryHistoryModel, Base

class PostgresRepository(MemoryRepository):
    def __init__(self, connection_url: Optional[str] = None):
        self.connection_url = connection_url or os.getenv("DATABASE_URL")
        if not self.connection_url:
            raise ValueError("DATABASE_URL must be set")

        self.engine = create_engine(
            self.connection_url,
            pool_size=10,          # Number of connections to keep open
            max_overflow=20,       # Extra connections if needed
            pool_timeout=30,       # Seconds to wait for connection
            pool_recycle=3600      # Recycle connections after 1 hour
        )
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Create tables (if not exists)
        self._init_db()
        self._embedding_generator = None
    @property
    def embedding_generator(self):
        """Lazy load embedding generator."""
        if self._embedding_generator is None:
            self._embedding_generator = EmbeddingGenerator()
        return self._embedding_generator

    def _init_db(self):
        """Initialize database schema and enable pgvector extension."""
        with self.engine.connect() as conn:
            # Enable pgvector extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()

        # Create tables
        Base.metadata.create_all(self.engine)

    def _to_dict(self, model: MemoryModel) -> Dict[str, Any]:
        """Convert SQLAlchemy model to dict."""
        return {
            "id": str(model.id),
            "key": model.key,
            "value": model.value,
            "type": model.memory_type,
            "frequency": model.frequency,
            "importance": model.importance,
            "status": model.status,
            "version": model.version,
            "created_at": model.created_at.isoformat() if model.created_at else None,
            "updated_at": model.updated_at.isoformat() if model.updated_at else None,
            "metadata": model.metadata_ or {},
        }

    def save(self, memory: Dict[str, Any]) -> None:
        """Store a new memory."""
        embedding = None
        if memory.get("value"):
            embedding = self.embedding_generator.generate(memory["value"])
        with self.SessionLocal() as session:
            db_memory = MemoryModel(
                id=memory.get("id", uuid.uuid4()),
                key=memory["key"],
                value=memory["value"],
                memory_type=memory["type"],
                frequency=memory.get("frequency", 1),
                importance=memory.get("importance", 0.0),
                status=memory.get("status", "active"),
                version=1,
                metadata_=memory.get("metadata", {}),
                embedding=embedding,
            )
            session.add(db_memory)

            # Add history entry
            history = MemoryHistoryModel(
                memory_id=db_memory.id,
                key=memory["key"],
                value=memory["value"],
                memory_type=memory["type"],
                operation="STORE",
                version=1,
            )
            session.add(history)
            session.commit()

    def get_by_key(self, key: str) -> Optional[Dict[str, Any]]:
        """Get active memory by key."""
        with self.SessionLocal() as session:
            result = session.execute(
                select(MemoryModel).where(
                    MemoryModel.key == key,
                    MemoryModel.status == "active"
                )
            ).scalar_one_or_none()
            return self._to_dict(result) if result else None

    def update(self, memory: Dict[str, Any]) -> None:
        """Update memory with optimistic locking."""
        memory_id = memory.get("id")
        if not memory_id:
            raise ValueError("Memory ID required for update")
        embedding = None
        if memory.get("value"):
            embedding = self.embedding_generator.generate(memory["value"])

        expected_version = memory.get("version", 1)

        with self.SessionLocal() as session:
            # Optimistic locking: only update if version matches
            result = session.execute(
                update(MemoryModel)
                .where(
                    MemoryModel.id == memory_id,
                    MemoryModel.version == expected_version,  # ← Lock check!
                    MemoryModel.status == "active"
                )
                .values(
                    key=memory["key"],
                    value=memory["value"],
                    memory_type=memory["type"],
                    importance=memory.get("importance", 0.0),
                    status=memory.get("status", "active"),
                    version=MemoryModel.version + 1,
                    updated_at=datetime.now(timezone.utc),
                    metadata_=memory.get("metadata", {}),
                    embedding=embedding,
                )
            )

            if result.rowcount == 0:
                raise ValueError(
                    f"Memory {memory_id} was updated by another process (version mismatch)"
                )

            # Add history entry
            history = MemoryHistoryModel(
                memory_id=memory_id,
                key=memory["key"],
                value=memory["value"],
                memory_type=memory["type"],
                operation="UPDATE",
                version=expected_version + 1,
            )
            session.add(history)
            session.commit()

    def delete(self, memory_id: str) -> None:
        """Soft delete a memory."""
        with self.SessionLocal() as session:
            # Get current version
            memory = session.execute(
                select(MemoryModel).where(MemoryModel.id == memory_id)
            ).scalar_one_or_none()
            if not memory:
                return

            # Update status and increment version
            memory.status = "inactive"
            memory.version += 1
            memory.updated_at = datetime.now(timezone.utc)

            # Add history
            history = MemoryHistoryModel(
                memory_id=memory_id,
                key=memory.key,
                value=memory.value,
                memory_type=memory.memory_type,
                operation="DELETE",
                version=memory.version,
            )
            session.add(history)
            session.commit()

    def get_all(self, only_active: bool = True) -> List[Dict[str, Any]]:
        """Get all memories."""
        with self.SessionLocal() as session:
            query = select(MemoryModel)
            if only_active:
                query = query.where(MemoryModel.status == "active")
            results = session.execute(query).scalars().all()
            return [self._to_dict(r) for r in results]

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memories by keyword (simple ILIKE)."""
        with self.SessionLocal() as session:
            search_pattern = f"%{query}%"
            stmt = select(MemoryModel).where(
                MemoryModel.status == "active",
                (MemoryModel.key.ilike(search_pattern) |
                 MemoryModel.value.ilike(search_pattern))
            ).limit(limit)
            results = session.execute(stmt).scalars().all()
            return [self._to_dict(r) for r in results]

    def merge(self, key: str, new_memory: Dict[str, Any]) -> None:
        """Merge a new memory with existing (increment frequency)."""
        with self.SessionLocal() as session:
            existing = session.execute(
                select(MemoryModel).where(
                    MemoryModel.key == key,
                    MemoryModel.status == "active"
                )
            ).scalar_one_or_none()

            if existing:
                # Increment frequency
                existing.frequency += 1
                existing.updated_at = datetime.now(timezone.utc)
                # If value changed, update it
                if existing.value != new_memory.get("value"):
                    existing.value = new_memory.get("value")
                existing.version += 1

                # Add history
                history = MemoryHistoryModel(
                    memory_id=existing.id,
                    key=existing.key,
                    value=existing.value,
                    memory_type=existing.memory_type,
                    operation="MERGE",
                    version=existing.version,
                )
                session.add(history)
                session.commit()

    def clear(self) -> None:
        """Clear all memories (hard delete)."""
        with self.SessionLocal() as session:
            session.execute(delete(MemoryModel))
            session.execute(delete(MemoryHistoryModel))
            session.commit()

    def get_history(self, key: str) -> List[Dict[str, Any]]:
        """Get history of a memory by key."""
        with self.SessionLocal() as session:
            results = session.execute(
                select(MemoryHistoryModel).where(
                    MemoryHistoryModel.key == key
                ).order_by(MemoryHistoryModel.created_at.asc())
            ).scalars().all()
            return [
                {
                    "operation": r.operation,
                    "value": r.value,
                    "version": r.version,
                    "created_at": r.created_at.isoformat(),
                }
                for r in results
            ]

    def search_semantic(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search memories by semantic similarity using pgvector.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of memories sorted by similarity (highest first)
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate(query)
        if query_embedding is None:
            return []

        with self.SessionLocal() as session:
            # Use pgvector's cosine_distance operator (<->)
            results = session.execute(
                select(MemoryModel)
                .where(MemoryModel.status == "active")
                .where(MemoryModel.embedding.is_not(None))
                .order_by(MemoryModel.embedding.cosine_distance(query_embedding))
                .limit(limit)
            ).scalars().all()

            return [self._to_dict(r) for r in results]

    def search_hybrid(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Hybrid search: combines keyword and semantic search.

        Uses Reciprocal Rank Fusion (RRF) to combine results.
        """
        # 1. Keyword search (get 2x results for fusion)
        keyword_results = self.search(query, limit=limit * 2)

        # 2. Semantic search (get 2x results for fusion)
        semantic_results = self.search_semantic(query, limit=limit * 2)

        # 3. Reciprocal Rank Fusion
        return self._rrf(keyword_results, semantic_results, limit)

    def _rrf(self, results_a: List[Dict], results_b: List[Dict], limit: int) -> List[Dict]:
        """
        Reciprocal Rank Fusion.

        Combines two ranked lists using RRF formula:
        score = sum(1 / (k + rank)) for each result in each list
        """
        k = 60  # Standard RRF constant
        scores = {}
        metadata = {}

        # Score results from list A
        for rank, result in enumerate(results_a):
            memory_id = result["id"]
            scores[memory_id] = scores.get(memory_id, 0) + 1 / (k + rank + 1)
            metadata[memory_id] = result

        # Score results from list B
        for rank, result in enumerate(results_b):
            memory_id = result["id"]
            scores[memory_id] = scores.get(memory_id, 0) + 1 / (k + rank + 1)
            if memory_id not in metadata:
                metadata[memory_id] = result

        # Sort by score (highest first)
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Return top results
        return [metadata[memory_id] for memory_id in sorted_ids[:limit]]