# ai-system/chunkdup/repositories/postgres_repository.py
from typing import Dict, List, Optional, Any, Union
from sqlalchemy import create_engine, select, update, delete, text, or_
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from datetime import datetime, timezone
import uuid
import os

try:
    from .repository import MemoryRepository
except ImportError:
    from repository import MemoryRepository
try:
    from .models import MemoryModel, MemoryHistoryModel, TurnModel, Base
except ImportError:
    from models import MemoryModel, MemoryHistoryModel, TurnModel, Base
try:
    from .turn_repository import PostgresTurnRepository
except ImportError:
    from turn_repository import PostgresTurnRepository


def _parse_dt(dt_val: Union[datetime, str]) -> datetime:
    """Normalize datetime or ISO string to UTC datetime."""
    if isinstance(dt_val, datetime):
        if dt_val.tzinfo is None:
            return dt_val.replace(tzinfo=timezone.utc)
        return dt_val
    dt = datetime.fromisoformat(dt_val.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


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
        self.turn_repository = PostgresTurnRepository(engine=self.engine)
        self._embedding_generator = None

    @property
    def embedding_generator(self):
        """Lazy load embedding generator."""
        if self._embedding_generator is None:
            from .embeddings import EmbeddingGenerator
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
        with self.engine.begin() as conn:
            conn.execute(text("ALTER TABLE memories ADD COLUMN IF NOT EXISTS entities JSON"))
            conn.execute(text("ALTER TABLE memories ADD COLUMN IF NOT EXISTS contextual_text TEXT"))
            conn.execute(text("ALTER TABLE memories ADD COLUMN IF NOT EXISTS source_turn_id UUID"))
            conn.execute(text("ALTER TABLE memories ADD COLUMN IF NOT EXISTS valid_from TIMESTAMPTZ"))
            conn.execute(text("ALTER TABLE memories ADD COLUMN IF NOT EXISTS valid_to TIMESTAMPTZ"))
            conn.execute(text("ALTER TABLE memories ADD COLUMN IF NOT EXISTS supersedes_id UUID"))
            conn.execute(text("ALTER TABLE memories ADD COLUMN IF NOT EXISTS superseded_by UUID"))

    def save_turn(self, turn: Dict[str, Any]) -> None:
        self.turn_repository.save_turn(turn)

    def get_turn(self, turn_id: str) -> Optional[Dict[str, Any]]:
        return self.turn_repository.get_turn(turn_id)

    def get_turns(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.turn_repository.get_turns(session_id=session_id)

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
            "entities": model.entities or [],
            "contextual_text": model.contextual_text or "",
            "source_turn_id": str(model.source_turn_id) if model.source_turn_id else None,
            "valid_from": model.valid_from.isoformat() if model.valid_from else (model.created_at.isoformat() if model.created_at else None),
            "valid_to": model.valid_to.isoformat() if model.valid_to else None,
            "supersedes_id": str(model.supersedes_id) if model.supersedes_id else None,
            "superseded_by": str(model.superseded_by) if model.superseded_by else None,
        }

    @staticmethod
    def _embedding_text(memory: Dict[str, Any]) -> str:
        source = memory.get("contextual_text") or memory.get("source_text") or "Stored memory record"
        return (
            f"key: {memory.get('key', '')}\n"
            f"type: {memory.get('type', '')}\n"
            f"value: {memory.get('value', '')}\n"
            f"contextualized source: {source}"
        ).strip()

    def reindex_embeddings(self) -> int:
        """Regenerate all active embeddings using the v3 representation."""
        updated = 0
        with self.SessionLocal() as session:
            memories = session.execute(
                select(MemoryModel).where(MemoryModel.status == "active", MemoryModel.valid_to.is_(None))
            ).scalars().all()
            for memory in memories:
                memory_dict = self._to_dict(memory)
                memory.embedding = self.embedding_generator.generate(self._embedding_text(memory_dict))
                updated += 1
            session.commit()
        return updated

    def save(self, memory: Dict[str, Any]) -> None:
        """Store a new memory."""
        embedding = None
        if memory.get("value"):
            embedding_text = self._embedding_text(memory)
            embedding = self.embedding_generator.generate(embedding_text)

        valid_from = memory.get("valid_from")
        if isinstance(valid_from, str):
            valid_from = _parse_dt(valid_from)
        elif valid_from is None:
            valid_from = datetime.now(timezone.utc)

        valid_to = memory.get("valid_to")
        if isinstance(valid_to, str):
            valid_to = _parse_dt(valid_to)

        source_turn_id = memory.get("source_turn_id")
        if isinstance(source_turn_id, str):
            source_turn_id = uuid.UUID(source_turn_id)

        supersedes_id = memory.get("supersedes_id")
        if isinstance(supersedes_id, str):
            supersedes_id = uuid.UUID(supersedes_id)

        mem_id = memory.get("id")
        if isinstance(mem_id, str):
            mem_id = uuid.UUID(mem_id)
        elif not mem_id:
            mem_id = uuid.uuid4()

        with self.SessionLocal() as session:
            db_memory = MemoryModel(
                id=mem_id,
                key=memory["key"],
                value=memory["value"],
                memory_type=memory["type"],
                frequency=memory.get("frequency", 1),
                importance=memory.get("importance", 0.0),
                status=memory.get("status", "active"),
                version=memory.get("version", 1),
                metadata_=memory.get("metadata", {}),
                entities=memory.get("entities", []),
                contextual_text=memory.get("contextual_text", ""),
                embedding=embedding,
                source_turn_id=source_turn_id,
                valid_from=valid_from,
                valid_to=valid_to,
                supersedes_id=supersedes_id,
            )
            session.add(db_memory)

            # Add history entry
            history = MemoryHistoryModel(
                memory_id=db_memory.id,
                key=memory["key"],
                value=memory["value"],
                memory_type=memory["type"],
                operation="STORE",
                version=memory.get("version", 1),
            )
            session.add(history)
            session.commit()

            memory["id"] = str(db_memory.id)

    def get_by_key(self, key: str, as_of: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """Get active memory by key, optionally as of a point in time."""
        with self.SessionLocal() as session:
            stmt = select(MemoryModel).where(MemoryModel.key == key)
            if as_of is None:
                stmt = stmt.where(MemoryModel.status == "active", MemoryModel.valid_to.is_(None))
            else:
                t = _parse_dt(as_of)
                stmt = stmt.where(
                    MemoryModel.valid_from <= t,
                    (MemoryModel.valid_to.is_(None) | (MemoryModel.valid_to > t)),
                    (MemoryModel.status != "inactive") | (MemoryModel.updated_at > t)
                ).order_by(MemoryModel.valid_from.desc())

            result = session.execute(stmt).scalars().first()
            return self._to_dict(result) if result else None

    def get_by_id(self, memory_id: str, as_of: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """Get memory by ID, optionally as of a point in time."""
        uid = uuid.UUID(memory_id) if isinstance(memory_id, str) else memory_id
        with self.SessionLocal() as session:
            stmt = select(MemoryModel).where(MemoryModel.id == uid)
            if as_of is None:
                stmt = stmt.where(MemoryModel.status == "active", MemoryModel.valid_to.is_(None))
            else:
                t = _parse_dt(as_of)
                stmt = stmt.where(
                    MemoryModel.valid_from <= t,
                    (MemoryModel.valid_to.is_(None) | (MemoryModel.valid_to > t)),
                    (MemoryModel.status != "inactive") | (MemoryModel.updated_at > t)
                )
            result = session.execute(stmt).scalar_one_or_none()
            return self._to_dict(result) if result else None

    def update(self, memory: Dict[str, Any]) -> None:
        """Append-only update with supersession."""
        memory_id = memory.get("id")
        key = memory.get("key")
        supersedes_id = memory.get("supersedes_id")

        embedding = None
        if memory.get("value"):
            embedding_text = self._embedding_text(memory)
            embedding = self.embedding_generator.generate(embedding_text)

        now_utc = datetime.now(timezone.utc)
        valid_from = memory.get("valid_from")
        if isinstance(valid_from, str):
            valid_from = _parse_dt(valid_from)
        elif valid_from is None:
            valid_from = now_utc

        source_turn_id = memory.get("source_turn_id")
        if isinstance(source_turn_id, str):
            source_turn_id = uuid.UUID(source_turn_id)

        with self.SessionLocal() as session:
            # 1. Locate the existing active record
            old_memory = None
            if supersedes_id:
                sid = uuid.UUID(supersedes_id) if isinstance(supersedes_id, str) else supersedes_id
                old_memory = session.execute(
                    select(MemoryModel).where(MemoryModel.id == sid)
                ).scalar_one_or_none()
            if not old_memory and memory_id:
                mid = uuid.UUID(memory_id) if isinstance(memory_id, str) else memory_id
                old_memory = session.execute(
                    select(MemoryModel).where(MemoryModel.id == mid, MemoryModel.status == "active")
                ).scalar_one_or_none()
            if not old_memory and key:
                old_memory = session.execute(
                    select(MemoryModel).where(MemoryModel.key == key, MemoryModel.status == "active", MemoryModel.valid_to.is_(None))
                ).scalar_one_or_none()

            if not old_memory:
                raise ValueError(f"Memory not found for update: id={memory_id}, key={key}")

            # New record id
            new_id = uuid.uuid4()
            old_id = old_memory.id

            # 2. Supersede old record
            old_memory.valid_to = valid_from
            old_memory.superseded_by = new_id
            old_memory.status = "superseded"
            old_memory.updated_at = now_utc

            # 3. Append new record
            new_version = old_memory.version + 1
            new_record = MemoryModel(
                id=new_id,
                key=memory["key"],
                value=memory["value"],
                memory_type=memory.get("type", old_memory.memory_type),
                frequency=old_memory.frequency,
                importance=memory.get("importance", old_memory.importance),
                status="active",
                version=new_version,
                valid_from=valid_from,
                valid_to=None,
                supersedes_id=old_id,
                source_turn_id=source_turn_id or old_memory.source_turn_id,
                metadata_=memory.get("metadata", old_memory.metadata_),
                entities=memory.get("entities", old_memory.entities),
                contextual_text=memory.get("contextual_text", old_memory.contextual_text),
                embedding=embedding,
            )
            session.add(new_record)

            # Add history entry
            history = MemoryHistoryModel(
                memory_id=new_id,
                key=memory["key"],
                value=memory["value"],
                memory_type=new_record.memory_type,
                operation="UPDATE",
                version=new_version,
            )
            session.add(history)
            session.commit()

            memory["id"] = str(new_id)
            memory["version"] = new_version
            memory["supersedes_id"] = str(old_id)

    def delete(self, memory_id: str) -> None:
        """Soft delete a memory."""
        uid = uuid.UUID(memory_id) if isinstance(memory_id, str) else memory_id
        now_utc = datetime.now(timezone.utc)
        with self.SessionLocal() as session:
            memory = session.execute(
                select(MemoryModel).where(MemoryModel.id == uid)
            ).scalar_one_or_none()
            if not memory:
                return

            memory.status = "inactive"
            memory.valid_to = now_utc
            memory.version += 1
            memory.updated_at = now_utc

            history = MemoryHistoryModel(
                memory_id=memory.id,
                key=memory.key,
                value=memory.value,
                memory_type=memory.memory_type,
                operation="DELETE",
                version=memory.version,
            )
            session.add(history)
            session.commit()

    def get_all(self, only_active: bool = True, as_of: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Get all memories, optionally as of a specific point in time."""
        with self.SessionLocal() as session:
            query = select(MemoryModel)
            if as_of is None:
                if only_active:
                    query = query.where(MemoryModel.status == "active", MemoryModel.valid_to.is_(None))
            else:
                t = _parse_dt(as_of)
                query = query.where(
                    MemoryModel.valid_from <= t,
                    (MemoryModel.valid_to.is_(None) | (MemoryModel.valid_to > t))
                )
                if only_active:
                    query = query.where(
                        (MemoryModel.status != "inactive") | (MemoryModel.updated_at > t)
                    )
            results = session.execute(query).scalars().all()
            return [self._to_dict(r) for r in results]

    def search(self, query: str, limit: Optional[int] = 10, as_of: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Search memories by keyword, optionally as of a point in time."""
        limit = limit or 10
        with self.SessionLocal() as session:
            search_pattern = f"%{query}%"
            stmt = select(MemoryModel).where(
                (MemoryModel.key.ilike(search_pattern) | MemoryModel.value.ilike(search_pattern))
            )
            if as_of is None:
                stmt = stmt.where(MemoryModel.status == "active", MemoryModel.valid_to.is_(None))
            else:
                t = _parse_dt(as_of)
                stmt = stmt.where(
                    MemoryModel.valid_from <= t,
                    (MemoryModel.valid_to.is_(None) | (MemoryModel.valid_to > t)),
                    (MemoryModel.status != "inactive") | (MemoryModel.updated_at > t)
                )
            stmt = stmt.limit(limit)
            results = session.execute(stmt).scalars().all()
            return [self._to_dict(r) for r in results]

    def merge(self, key: str, new_memory: Dict[str, Any]) -> None:
        """Merge a new memory with existing (increment frequency)."""
        with self.SessionLocal() as session:
            existing = session.execute(
                select(MemoryModel).where(
                    MemoryModel.key == key,
                    MemoryModel.status == "active",
                    MemoryModel.valid_to.is_(None)
                )
            ).scalar_one_or_none()

            if existing:
                existing.frequency += 1
                existing.updated_at = datetime.now(timezone.utc)
                if existing.value != new_memory.get("value"):
                    existing.value = new_memory.get("value")
                existing.version += 1

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
        """Clear all memories and turns."""
        with self.SessionLocal() as session:
            session.execute(delete(MemoryModel))
            session.execute(delete(MemoryHistoryModel))
            session.execute(delete(TurnModel))
            session.commit()

    def get_history(self, key: str) -> List[Dict[str, Any]]:
        """Get history of a memory by key."""
        with self.SessionLocal() as session:
            results = session.execute(
                select(MemoryModel).where(
                    MemoryModel.key == key
                ).order_by(MemoryModel.valid_from.asc(), MemoryModel.version.asc())
            ).scalars().all()
            return [self._to_dict(r) for r in results]

    def search_semantic(
        self,
        query: str,
        limit: int = 5,
        threshold: float = 0.15,
        as_of: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_generator.generate(query)
        if query_embedding is None:
            return []

        with self.SessionLocal() as session:
            stmt = (
                select(
                    MemoryModel,
                    MemoryModel.embedding.cosine_distance(query_embedding).label("distance")
                )
                .where(MemoryModel.embedding.is_not(None))
            )
            if as_of is None:
                stmt = stmt.where(MemoryModel.status == "active", MemoryModel.valid_to.is_(None))
            else:
                t = _parse_dt(as_of)
                stmt = stmt.where(
                    MemoryModel.valid_from <= t,
                    (MemoryModel.valid_to.is_(None) | (MemoryModel.valid_to > t)),
                    (MemoryModel.status != "inactive") | (MemoryModel.updated_at > t)
                )
            stmt = stmt.order_by(MemoryModel.embedding.cosine_distance(query_embedding)).limit(limit)
            results = session.execute(stmt).all()
            output = []
            for memory, distance in results:
                item = self._to_dict(memory)
                item["_similarity"] = 1.0 - float(distance)
                output.append(item)
        return output

    def search_hybrid(self, query: str, limit: int = 5, as_of: Optional[Any] = None) -> List[Dict[str, Any]]:
        keyword_results = self.search(query, limit=limit * 2, as_of=as_of)
        semantic_results = self.search_semantic(query, limit=limit * 2, as_of=as_of)
        return self._rrf(keyword_results, semantic_results, limit)

    def _rrf(self, results_a: List[Dict], results_b: List[Dict], limit: int) -> List[Dict]:
        k = 60
        scores = {}
        metadata = {}

        for rank, result in enumerate(results_a):
            memory_id = result["id"]
            scores[memory_id] = scores.get(memory_id, 0) + 1 / (k + rank + 1)
            metadata[memory_id] = result

        for rank, result in enumerate(results_b):
            memory_id = result["id"]
            scores[memory_id] = scores.get(memory_id, 0) + 1 / (k + rank + 1)
            if memory_id not in metadata:
                metadata[memory_id] = result

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [metadata[memory_id] for memory_id in sorted_ids[:limit]]