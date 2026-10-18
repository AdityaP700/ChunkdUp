# chunkdup/in_memory_repository.py
import os
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
import uuid
from .repository import MemoryRepository
from .turn_repository import InMemoryTurnRepository


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


def _is_valid_at(memory: Dict[str, Any], target_dt: datetime) -> bool:
    """Check if memory was valid at a specific point in time."""
    vf_raw = memory.get("valid_from") or memory.get("created_at")
    if not vf_raw:
        return False
    try:
        vf = _parse_dt(vf_raw)
    except Exception:
        return False
    if vf > target_dt:
        return False

    vt_raw = memory.get("valid_to")
    if vt_raw:
        try:
            vt = _parse_dt(vt_raw)
            if target_dt >= vt:
                return False
        except Exception:
            pass

    if memory.get("status") == "inactive":
        up_raw = memory.get("updated_at")
        if up_raw:
            try:
                up = _parse_dt(up_raw)
                if target_dt >= up:
                    return False
            except Exception:
                pass

    return True


class InMemoryRepository(MemoryRepository):
    """Stores memories in RAM with append-only temporal history, supersession, and durable decision audit."""

    def __init__(self, data_dir: Optional[str] = None):
        self._memories: Dict[str, Dict[str, Any]] = {}  # id -> memory
        self._key_index: Dict[str, str] = {}            # key -> active id
        self.turn_repository = InMemoryTurnRepository()
        self._decisions: Dict[str, Dict[str, Any]] = {}
        self._decisions_by_turn: Dict[str, List[str]] = {}
        self._decisions_by_memory: Dict[str, List[str]] = {}
        self._data_dir = data_dir
        self._audit_file = os.path.join(data_dir, "decisions.jsonl") if data_dir else None
        if self._audit_file:
            self._load_decisions()

    def _index_decision(self, event: Dict[str, Any]) -> None:
        d_id = str(event.get("decision_id") or event.get("id"))
        self._decisions[d_id] = event
        t_id = event.get("source_turn_id")
        if t_id:
            d_list = self._decisions_by_turn.setdefault(str(t_id), [])
            if d_id not in d_list:
                d_list.append(d_id)
        m_id = event.get("target_memory_id")
        if m_id:
            m_list = self._decisions_by_memory.setdefault(str(m_id), [])
            if d_id not in m_list:
                m_list.append(d_id)

    def _load_decisions(self) -> None:
        if self._audit_file and os.path.exists(self._audit_file):
            try:
                with open(self._audit_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            event = json.loads(line)
                            self._index_decision(event)
            except Exception:
                pass

    def save_turn(self, turn: Dict[str, Any]) -> None:
        self.turn_repository.save_turn(turn)

    def get_turn(self, turn_id: str) -> Optional[Dict[str, Any]]:
        return self.turn_repository.get_turn(turn_id)

    def get_turns(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.turn_repository.get_turns(session_id=session_id)

    def save(self, memory: Dict[str, Any]) -> None:
        memory_id = memory.get("id") or str(uuid.uuid4())
        memory["id"] = memory_id
        now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        memory["created_at"] = memory.get("created_at") or now_iso
        memory["updated_at"] = memory.get("updated_at") or memory["created_at"]
        memory["status"] = memory.get("status", "active")
        memory["version"] = memory.get("version", 1)
        memory["valid_from"] = memory.get("valid_from") or memory["created_at"]
        memory["valid_to"] = memory.get("valid_to", None)
        memory["supersedes_id"] = memory.get("supersedes_id", None)
        memory["superseded_by"] = memory.get("superseded_by", None)
        memory["source_turn_id"] = memory.get("source_turn_id", None)

        self._memories[memory_id] = memory
        if memory["status"] == "active" and memory.get("valid_to") is None:
            self._key_index[memory["key"]] = memory_id

    def get_by_key(self, key: str, as_of: Optional[Union[datetime, str]] = None) -> Optional[Dict[str, Any]]:
        if as_of is None:
            memory_id = self._key_index.get(key)
            if memory_id:
                memory = self._memories.get(memory_id)
                if memory and memory.get("status") == "active" and memory.get("valid_to") is None:
                    return memory
            return None

        target_dt = _parse_dt(as_of)
        candidates = [
            m for m in self._memories.values()
            if m.get("key") == key and _is_valid_at(m, target_dt)
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda x: _parse_dt(x.get("valid_from") or x.get("created_at")), reverse=True)
        return candidates[0]

    def get_by_id(self, memory_id: str, as_of: Optional[Union[datetime, str]] = None) -> Optional[Dict[str, Any]]:
        memory = self._memories.get(memory_id)
        if not memory:
            return None
        if as_of is None:
            if memory.get("status") == "active" and memory.get("valid_to") is None:
                return memory
            return None
        target_dt = _parse_dt(as_of)
        if _is_valid_at(memory, target_dt):
            return memory
        return None

    def update(self, memory: Dict[str, Any]) -> None:
        """
        Append-only update with supersession.
        The existing record is stamped with valid_to and superseded_by.
        A new record is inserted with supersedes_id and valid_to=None.
        """
        memory_id = memory.get("id")
        key = memory.get("key")
        supersedes_id = memory.get("supersedes_id")

        # Find the existing active record to supersede
        existing = None
        if supersedes_id and supersedes_id in self._memories:
            existing = self._memories[supersedes_id]
        elif memory_id and memory_id in self._memories:
            existing = self._memories[memory_id]
        elif key:
            existing = self.get_by_key(key)

        if not existing:
            raise ValueError(f"Memory not found for update: id={memory_id}, key={key}")

        now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        new_valid_from = memory.get("valid_from") or now_iso
        existing_id = existing["id"]

        # Ensure a new distinct id for the appended row
        if memory_id == existing_id or not memory_id:
            new_id = str(uuid.uuid4())
        else:
            new_id = memory_id

        # 1. Supersede existing row
        existing["valid_to"] = new_valid_from
        existing["superseded_by"] = new_id
        existing["status"] = "superseded"
        existing["updated_at"] = now_iso

        # 2. Append new row
        new_record = dict(memory)
        new_record["id"] = new_id
        new_record["supersedes_id"] = existing_id
        new_record["superseded_by"] = None
        new_record["valid_from"] = new_valid_from
        new_record["valid_to"] = None
        new_record["status"] = "active"
        new_record["version"] = existing.get("version", 1) + 1
        new_record["frequency"] = existing.get("frequency", 1)
        new_record["created_at"] = new_valid_from
        new_record["updated_at"] = now_iso
        new_record["source_turn_id"] = memory.get("source_turn_id") or existing.get("source_turn_id")

        self._memories[new_id] = new_record
        self._key_index[new_record["key"]] = new_id

        # Mutate passed-in memory in place so caller receives new identity and version
        memory["id"] = new_id
        memory["version"] = new_record["version"]
        memory["supersedes_id"] = existing_id
        memory["valid_from"] = new_valid_from
        memory["valid_to"] = None

    def delete(self, memory_id: str) -> None:
        if memory_id in self._memories:
            now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            m = self._memories[memory_id]
            m["status"] = "inactive"
            m["valid_to"] = now_iso
            m["updated_at"] = now_iso
            if self._key_index.get(m["key"]) == memory_id:
                del self._key_index[m["key"]]

    def get_all(self, as_of: Optional[Union[datetime, str]] = None) -> List[Dict[str, Any]]:
        if as_of is None:
            return [m for m in self._memories.values() if m.get("status") == "active" and m.get("valid_to") is None]

        target_dt = _parse_dt(as_of)
        valid_memories = [m for m in self._memories.values() if _is_valid_at(m, target_dt)]
        valid_memories.sort(key=lambda x: _parse_dt(x.get("valid_from") or x.get("created_at")), reverse=True)
        seen_keys = set()
        result = []
        for m in valid_memories:
            key = m.get("key")
            if key not in seen_keys:
                seen_keys.add(key)
                result.append(m)
        return result

    def get_history(self, key: str) -> List[Dict[str, Any]]:
        history = [m for m in self._memories.values() if m.get("key") == key]
        history.sort(key=lambda x: (x.get("version", 1), str(x.get("valid_from") or x.get("created_at") or "")))
        return history

    def search(self, query: str, limit: Optional[int] = None, as_of: Optional[Union[datetime, str]] = None) -> List[Dict[str, Any]]:
        query_words = set(query.lower().split())
        results = []

        pool = self.get_all(as_of=as_of)
        for memory in pool:
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
            existing["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            if existing["value"] != memory.get("value"):
                existing["value"] = memory.get("value")

    def record_decision(self, event: Dict[str, Any]) -> None:
        """Record an append-only decision audit event."""
        d_id = str(event.get("decision_id") or event.get("id") or uuid.uuid4())
        event["decision_id"] = d_id
        if "timestamp" not in event:
            event["timestamp"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self._index_decision(event)
        if self._audit_file:
            try:
                os.makedirs(os.path.dirname(self._audit_file), exist_ok=True)
                with open(self._audit_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event) + "\n")
            except Exception:
                pass

    def get_decision(self, decision_id: str) -> Optional[Dict[str, Any]]:
        return self._decisions.get(str(decision_id))

    def get_decisions_by_turn(self, source_turn_id: str) -> List[Dict[str, Any]]:
        d_ids = self._decisions_by_turn.get(str(source_turn_id), [])
        return [self._decisions[did] for did in d_ids if did in self._decisions]

    def get_decisions_by_memory(self, memory_id: str) -> List[Dict[str, Any]]:
        d_ids = self._decisions_by_memory.get(str(memory_id), [])
        return [self._decisions[did] for did in d_ids if did in self._decisions]

    def get_all_decisions(self) -> List[Dict[str, Any]]:
        return list(self._decisions.values())

    def clear(self) -> None:
        self._memories.clear()
        self._key_index.clear()
        self.turn_repository.clear()
        self._decisions.clear()
        self._decisions_by_turn.clear()
        self._decisions_by_memory.clear()
        if self._audit_file and os.path.exists(self._audit_file):
            try:
                os.remove(self._audit_file)
            except Exception:
                pass