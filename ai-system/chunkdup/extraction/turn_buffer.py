# ai-system/chunkdup/extraction/turn_buffer.py
"""
Batching and buffer coordinator for layered dialogue turn ingestion and fact extraction.
"""

from typing import List, Dict, Any, Optional
import logging

from .groq_extractor import GroqFactExtractor

logger = logging.getLogger(__name__)


class TurnBuffer:
    """
    Coordinates layered ingestion:
      Level 0: Raw turn log append (instant, <5ms).
      Level 1 & 2: Batched Groq extraction when buffer fills or flush() is invoked.
      Level 3: Strict schema validation.
      Level 4: Policy engine proposal (STORE/MERGE/UPDATE/IGNORE).
    """

    def __init__(
        self,
        turn_log: Any,
        extractor: GroqFactExtractor,
        memory_manager: Any,
        batch_size: int = 5,
    ):
        self.turn_log = turn_log
        self.extractor = extractor
        self.memory_manager = memory_manager
        self.batch_size = batch_size
        self._buffer: List[Dict[str, Any]] = []

    def add_turn(
        self,
        raw_text: str,
        session_id: str = "default",
        speaker: str = "user",
        metadata: Optional[Dict[str, Any]] = None,
        turn_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Level 0: Instantly persist raw turn.
        Buffers turn for batched extraction. If buffer size is reached, flushes automatically.
        """
        turn = self.turn_log.append(
            raw_text=raw_text,
            session_id=session_id,
            speaker=speaker,
            metadata=metadata,
            turn_id=turn_id,
            timestamp=timestamp,
        )

        self._buffer.append(turn)

        # Trigger batch extraction when buffer reaches threshold
        flushed_results = []
        if len(self._buffer) >= self.batch_size:
            flushed_results = self.flush()

        return {
            "turn": turn,
            "buffered_count": len(self._buffer),
            "processed_memories": flushed_results,
        }

    def flush(self) -> List[Dict[str, Any]]:
        """
        Process all currently buffered dialogue turns in a single batched extraction call.
        Partial batch failures are isolated and recoverable without duplicating successfully persisted facts.
        """
        if not self._buffer:
            return []

        turns_to_process = list(self._buffer)
        self._buffer.clear()

        # Level 2 & 3: Batched extraction and schema validation
        try:
            extracted_facts = self.extractor.extract_batch(turns_to_process)
        except Exception as exc:
            logger.warning(f"Batch extraction failed ({exc.__class__.__name__}). Falling back to per-turn extraction.")
            extracted_facts = []
            for t in turns_to_process:
                try:
                    extracted_facts.extend(self.extractor._fallback_extract(t))
                except Exception:
                    continue

        # Level 4: Propose to policy engine with per-fact isolation
        results = []
        for fact in extracted_facts:
            try:
                mem_dict = fact.to_memory_dict() if hasattr(fact, "to_memory_dict") else dict(fact)
                decision = self.memory_manager.process(mem_dict)
                results.append({
                    "decision": decision.value.upper(),
                    "key": mem_dict.get("key"),
                    "value": mem_dict.get("value"),
                    "type": mem_dict.get("type"),
                    "source_turn_id": mem_dict.get("source_turn_id"),
                    "status": "success",
                })
            except Exception as proc_exc:
                logger.warning(f"Failed to process fact {getattr(fact, 'key', 'unknown')}: {proc_exc.__class__.__name__}")
                results.append({
                    "decision": "ERROR",
                    "key": getattr(fact, "key", None),
                    "source_turn_id": getattr(fact, "source_turn_id", None),
                    "status": "failed",
                    "error": proc_exc.__class__.__name__,
                })

        return results

    def clear(self) -> None:
        """Clear buffered turns without extracting."""
        self._buffer.clear()

    @property
    def pending_count(self) -> int:
        return len(self._buffer)
