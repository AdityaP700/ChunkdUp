# ai-system/chunkdup/turn_repository.py
"""
Repository and log abstraction for raw dialogue turns.
Guarantees raw turns are permanently preserved regardless of extraction outcomes.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import uuid
import logging

logger = logging.getLogger(__name__)


class TurnRepository(ABC):
    """Abstract contract for dialogue turn persistence."""

    @abstractmethod
    def save_turn(self, turn: Dict[str, Any]) -> None:
        """Store a dialogue turn."""
        pass

    @abstractmethod
    def get_turn(self, turn_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a dialogue turn by ID."""
        pass

    @abstractmethod
    def get_turns(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve dialogue turns, optionally filtered by session."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all dialogue turns."""
        pass


class InMemoryTurnRepository(TurnRepository):
    """In-memory turn repository for testing and lightweight sessions."""

    def __init__(self):
        self._turns: Dict[str, Dict[str, Any]] = {}

    def save_turn(self, turn: Dict[str, Any]) -> None:
        turn_id = turn.get("turn_id") or str(uuid.uuid4())
        turn["turn_id"] = turn_id
        if "timestamp" not in turn or not turn["timestamp"]:
            turn["timestamp"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        if "speaker" not in turn:
            turn["speaker"] = "user"
        if "session_id" not in turn:
            turn["session_id"] = "default"
        if "metadata" not in turn:
            turn["metadata"] = {}
        self._turns[turn_id] = dict(turn)

    def get_turn(self, turn_id: str) -> Optional[Dict[str, Any]]:
        turn = self._turns.get(turn_id)
        return dict(turn) if turn else None

    def get_turns(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        turns = list(self._turns.values())
        if session_id is not None:
            turns = [t for t in turns if t.get("session_id") == session_id]
        turns.sort(key=lambda t: t.get("timestamp", ""))
        return [dict(t) for t in turns]

    def clear(self) -> None:
        self._turns.clear()


class PostgresTurnRepository(TurnRepository):
    """PostgreSQL-backed turn repository."""

    def __init__(self, connection_url: Optional[str] = None, engine: Optional[Any] = None):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        import os

        if engine is not None:
            self.engine = engine
        else:
            url = connection_url or os.getenv("DATABASE_URL")
            if not url:
                raise ValueError("DATABASE_URL must be set for PostgresTurnRepository")
            self.engine = create_engine(url)

        self.SessionLocal = sessionmaker(bind=self.engine)
        self._init_db()

    def _init_db(self):
        try:
            from .models import TurnModel, Base
        except ImportError:
            from models import TurnModel, Base
        Base.metadata.create_all(self.engine)

    def save_turn(self, turn: Dict[str, Any]) -> None:
        try:
            from .models import TurnModel
        except ImportError:
            from models import TurnModel

        turn_id = turn.get("turn_id")
        uid = uuid.UUID(turn_id) if isinstance(turn_id, str) else (turn_id or uuid.uuid4())

        ts = turn.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        elif ts is None:
            ts = datetime.now(timezone.utc)

        with self.SessionLocal() as session:
            db_turn = TurnModel(
                id=uid,
                session_id=turn.get("session_id", "default"),
                speaker=turn.get("speaker", "user"),
                raw_text=turn.get("raw_text", ""),
                timestamp=ts,
                metadata_=turn.get("metadata", {}),
            )
            session.merge(db_turn)
            session.commit()

    def get_turn(self, turn_id: str) -> Optional[Dict[str, Any]]:
        from sqlalchemy import select
        try:
            from .models import TurnModel
        except ImportError:
            from models import TurnModel

        uid = uuid.UUID(turn_id) if isinstance(turn_id, str) else turn_id
        with self.SessionLocal() as session:
            res = session.execute(
                select(TurnModel).where(TurnModel.id == uid)
            ).scalar_one_or_none()
            if not res:
                return None
            return {
                "turn_id": str(res.id),
                "session_id": res.session_id,
                "speaker": res.speaker,
                "raw_text": res.raw_text,
                "timestamp": res.timestamp.isoformat() if res.timestamp else None,
                "metadata": res.metadata_ or {},
            }

    def get_turns(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        from sqlalchemy import select
        try:
            from .models import TurnModel
        except ImportError:
            from models import TurnModel

        with self.SessionLocal() as session:
            stmt = select(TurnModel)
            if session_id is not None:
                stmt = stmt.where(TurnModel.session_id == session_id)
            stmt = stmt.order_by(TurnModel.timestamp.asc())
            results = session.execute(stmt).scalars().all()
            return [
                {
                    "turn_id": str(r.id),
                    "session_id": r.session_id,
                    "speaker": r.speaker,
                    "raw_text": r.raw_text,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                    "metadata": r.metadata_ or {},
                }
                for r in results
            ]

    def clear(self) -> None:
        from sqlalchemy import delete
        try:
            from .models import TurnModel
        except ImportError:
            from models import TurnModel

        with self.SessionLocal() as session:
            session.execute(delete(TurnModel))
            session.commit()


class SourceTurnLog:
    """
    High-level dialogue turn logger.
    Ensures every raw turn is captured with provenance.
    """

    def __init__(self, repository: TurnRepository):
        self.repository = repository

    def append(
        self,
        raw_text: str,
        session_id: str = "default",
        speaker: str = "user",
        metadata: Optional[Dict[str, Any]] = None,
        turn_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist a raw dialogue turn."""
        turn = {
            "turn_id": turn_id or str(uuid.uuid4()),
            "session_id": session_id or "default",
            "speaker": speaker,
            "raw_text": raw_text,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "metadata": metadata or {},
        }
        self.repository.save_turn(turn)
        return turn

    def get(self, turn_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a turn by its ID."""
        return self.repository.get_turn(turn_id)

    def get_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all turns for a specific session."""
        return self.repository.get_turns(session_id=session_id)

    def get_all(self) -> List[Dict[str, Any]]:
        """Retrieve all dialogue turns."""
        return self.repository.get_turns()

    def clear(self) -> None:
        """Clear all dialogue turns."""
        self.repository.clear()
