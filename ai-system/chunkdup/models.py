# ai-system/chunkdup/models.py
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, JSON, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector
from datetime import datetime, timezone
import uuid

Base = declarative_base()

class MemoryModel(Base):
    __tablename__ = "memories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key = Column(String(255), nullable=False)
    value = Column(Text, nullable=False)
    memory_type = Column(String(50), nullable=False)
    frequency = Column(Integer, default=1)
    importance = Column(Float, default=0.0)
    status = Column(String(20), default="active")
    version = Column(Integer, default=1)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    metadata_ = Column("metadata",JSON)

    embedding = Column(Vector(384), nullable=True)  # 384-dimensional embeddings

    __table_args__ = (
        Index('idx_memories_key_status', 'key', 'status'),
        Index('idx_memories_created', 'created_at'),
        Index('idx_memories_version', 'version'),
        # ← NEW: HNSW index for fast vector search
        Index('idx_memories_embedding', 'embedding', postgresql_using='hnsw', postgresql_ops={'embedding': 'vector_cosine_ops'}),
    )

class MemoryHistoryModel(Base):
    __tablename__ = "memory_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    memory_id = Column(UUID(as_uuid=True), nullable=False)
    key = Column(String(255), nullable=False)
    value = Column(Text, nullable=False)
    memory_type = Column(String(50), nullable=False)
    operation = Column(String(20), nullable=False)  # STORE, UPDATE, MERGE, DELETE
    version = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))