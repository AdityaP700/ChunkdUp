# ai-system/chunkdup/models.py
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, JSON, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector
from datetime import datetime, timezone
import uuid

Base = declarative_base()


class TurnModel(Base):
    __tablename__ = "source_turns"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(255), nullable=True, default="default")
    speaker = Column(String(50), nullable=False, default="user")
    raw_text = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    metadata_ = Column("metadata", JSON, default=dict)

    __table_args__ = (
        Index('idx_source_turns_session', 'session_id'),
        Index('idx_source_turns_timestamp', 'timestamp'),
    )


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
    metadata_ = Column("metadata", JSON)
    entities = Column(JSON, default=list)
    contextual_text = Column(Text)

    embedding = Column(Vector(384), nullable=True)  # 384-dimensional embeddings

    # Temporal & provenance tracking
    source_turn_id = Column(UUID(as_uuid=True), nullable=True)
    valid_from = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    valid_to = Column(DateTime(timezone=True), nullable=True)
    supersedes_id = Column(UUID(as_uuid=True), nullable=True)
    superseded_by = Column(UUID(as_uuid=True), nullable=True)

    __table_args__ = (
        Index('idx_memories_key_status', 'key', 'status'),
        Index('idx_memories_created', 'created_at'),
        Index('idx_memories_version', 'version'),
        Index('idx_memories_valid_range', 'valid_from', 'valid_to'),
        Index('idx_memories_supersedes', 'supersedes_id'),
        # HNSW index for fast vector search
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


class DecisionModel(Base):
    __tablename__ = "decision_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_turn_id = Column(UUID(as_uuid=True), nullable=True)
    decision_type = Column(String(20), nullable=False)  # STORE, MERGE, UPDATE, IGNORE
    target_memory_id = Column(UUID(as_uuid=True), nullable=True)
    memory_key = Column(String(255), nullable=True)
    previous_value = Column(Text, nullable=True)
    new_value = Column(Text, nullable=True)
    version = Column(Integer, default=1)
    rationale = Column(Text, nullable=True)
    confidence = Column(Float, default=1.0)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    status = Column(String(20), default="success")  # success, failed
    error_message = Column(Text, nullable=True)
    metadata_ = Column("metadata", JSON, default=dict)

    __table_args__ = (
        Index('idx_decision_turn', 'source_turn_id'),
        Index('idx_decision_memory', 'target_memory_id'),
        Index('idx_decision_type', 'decision_type'),
        Index('idx_decision_timestamp', 'timestamp'),
    )