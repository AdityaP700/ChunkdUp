# ai-system/chunkdup/memory.py
"""
Core Memory management class and supporting modules for ChunkdUp SDK.
"""

import os
import json
import math
import re
import uuid
import sys
import importlib.util
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union
import logging
try:
    from .postgres_repository import PostgresRepository
except ImportError:
    PostgresRepository = None

from .repository import MemoryRepository
from .in_memory_repository import InMemoryRepository
from .types import MemoryDict, SearchResult, RememberResult, UpdateResult, DeleteResult, StatsDict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import local modules
try:
    from .policies import Decision, PolicyFactory
    from .prompt_builder import PromptBuilder
    from .retrieval_class import MemoryRetriever
    from .spacy_extractor import SpacyExtractor
    from .llm_extractor import LLMExtractor
except ImportError:
    from policies import Decision, PolicyFactory
    from prompt_builder import PromptBuilder
    from retrieval_class import MemoryRetriever
    from spacy_extractor import SpacyExtractor
    from llm_extractor import LLMExtractor


class MemoryExtractor:
    """Extracts structured memories from text using pattern rules, spaCy NER, and LLM fallback."""

    def __init__(self, llm_caller=None):
        self.spacy_extractor = SpacyExtractor()
        self.llm_extractor = LLMExtractor(llm_caller=llm_caller)

        self.rules = [
            {
                "pattern": r"I'm building (?:a\s+[\w\s]+?\s+called\s+)?([\w\d\-\s]+)",
                "type": "project",
                "key": "project_name",
                "value_group": 1
            },
            {
                "pattern": r"I'm working on a project called ([\w\d\-\s]+)",
                "type": "project",
                "key": "project_name",
                "value_group": 1
            },
            {
                "pattern": r"I prefer (Neovim|Vim|VS Code|VSCode|Emacs|Sublime)(?:\.|$)",
                "type": "tool",
                "key": "editor",
                "value_group": 1
            },
            {
                "pattern": r"I prefer (dark mode|light mode)",
                "type": "preference",
                "key": "preference",
                "value_group": 1
            },
            {
                "pattern": r"I prefer ([\w\s]+?)(?:\.|$)",
                "type": "preference",
                "key": "response_style",
                "value_group": 1
            },
            {
                "pattern": r"I (?:now\s+)?use (Windows|Mac|Linux)",
                "type": "environment",
                "key": "os",
                "value_group": 1
            },
            {
                "pattern": r"I (?:now\s+)?use (AWS|Azure|GCP)",
                "type": "environment",
                "key": "cloud_provider",
                "value_group": 1
            },
            {
                "pattern": r"I (?:now\s+)?use (Docker|Kubernetes)",
                "type": "environment",
                "key": "environment",
                "value_group": 1
            },
            {
                "pattern": r"I (?:now\s+)?use (Python|Java|C\+\+|Rust|Go|TypeScript|JavaScript)",
                "type": "environment",
                "key": "programming_language",
                "value_group": 1
            },
            {
                "pattern": r"My favorite editor is ([\w\s]+?)(?:\.|$)",
                "type": "tool",
                "key": "editor",
                "value_group": 1
            },
            {
                "pattern": r"I (?:now\s+)?work at ([\w\s]+?)(?:\.|$)",
                "type": "employment",
                "key": "company",
                "value_group": 1
            },
            {
                "pattern": r"I'm (?:now\s+)?a ([\w\s]+?)(?:\.|$)",
                "type": "role",
                "key": "role",
                "value_group": 1
            },
            {
                "pattern": r"My favorite ([\w\s]+) is ([\w\s]+?)(?:\.|$)",
                "type": "preference",
                "key": "favorite_{1}",
                "value_group": 2
            },
            {
                "pattern": r"I'm learning ([\w\d\+#\s]+?)(?:\.|$)",
                "type": "learning",
                "key": "learning",
                "value_group": 1
            }
        ]

    def extract(self, conversation: str) -> List[Dict[str, Any]]:
        # Tier 1: Pattern Rule Extraction
        memories = self._extract_with_rules(conversation)
        if memories:
            return memories

        # Tier 2: spaCy NER Extraction
        if self.spacy_extractor.is_available():
            memories = self.spacy_extractor.extract(conversation)
            if memories:
                return memories

        # Tier 3: LLM Fallback Extraction
        memories = self.llm_extractor.extract(conversation)
        return memories

    def _extract_with_rules(self, conversation: str) -> List[Dict[str, Any]]:
        memories = []
        for rule in self.rules:
            matches = re.finditer(rule["pattern"], conversation, re.IGNORECASE)
            for match in matches:
                value = match.group(rule["value_group"]).strip()
                key = rule["key"]
                for i in range(1, len(match.groups()) + 1):
                    placeholder = f"{{{i}}}"
                    if placeholder in key and match.group(i):
                        key = key.replace(placeholder, match.group(i).strip().lower().replace(" ", "_"))

                memory = {
                    "id": str(uuid.uuid4()),
                    "type": rule["type"],
                    "key": key,
                    "value": value,
                    "frequency": 1,
                    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "source": "pattern_rule",
                    "status": "active"
                }
                if "meta" in rule:
                    for meta_key, group_idx in rule["meta"].items():
                        if match.group(group_idx):
                            memory[meta_key] = match.group(group_idx).strip()
                memories.append(memory)
        return memories


class MemoryScorer:
    """Scores memories based on type importance, frequency, and recency."""

    def __init__(
        self,
        type_weight: float = 0.5,
        freq_weight: float = 0.3,
        recency_weight: float = 0.2,
        type_weights: Optional[Dict[str, float]] = None
    ):
        self.type_weight = type_weight
        self.freq_weight = freq_weight
        self.recency_weight = recency_weight

        self.type_weights = type_weights or {
            "project": 1.0,
            "preference": 0.9,
            "environment": 0.8,
            "tool": 0.7,
            "employment": 0.9,
            "role": 0.85,
            "learning": 0.8,
            "favorite": 0.75,
            "company": 0.9,
            "os": 0.8,
            "editor": 0.7,
            "question": 0.3,
        }

    def score(self, memory: Dict[str, Any]) -> float:
        """
        Calculate composite importance score.

        Factors:
        - Type importance (what kind of memory)
        - Frequency (how often mentioned)
        - Recency (how recent)
        """
        # 1. Type importance (0.3 - 1.0)
        type_score = self.type_weights.get(memory.get("type"), 0.5)

        # 2. Frequency (log scale: 1->0, 10->0.5, 100->1.0)
        freq = memory.get("frequency", 1)
        freq_score = min(1.0, math.log10(freq + 1) / 2)

        # 3. Recency (1 / (days_old + 1))
        updated_at = memory.get("updated_at")
        if updated_at:
            if isinstance(updated_at, str):
                try:
                    updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                except Exception:
                    updated_at = datetime.now(timezone.utc)
            days_old = max(0, (datetime.now(timezone.utc) - updated_at).days)
        else:
            days_old = 30
        recency_score = 1.0 / (days_old + 1.0)

        # 4. Composite
        composite = (
            (type_score * self.type_weight) +
            (freq_score * self.freq_weight) +
            (recency_score * self.recency_weight)
        )

        return min(1.0, composite)


class MemoryRanker:
    """Ranks memories by composite score (semantic, frequency, recency)."""

    def __init__(self, scorer: MemoryScorer):
        self.scorer = scorer

    def rank(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        now = datetime.now(timezone.utc)
        ranked = []
        for mem in memories:
            semantic = self.scorer.score(mem)
            freq = mem.get("frequency", 1)
            freq_score = math.log10(freq + 9) - 1

            updated_str = mem.get("updated_at")
            if updated_str:
                updated_at = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
                days_old = max(0, (now - updated_at).total_seconds() / 86400)
            else:
                days_old = 30

            recency_score = 1.0 / (days_old + 1.0)
            composite = semantic + freq_score + recency_score

            mem["_ranking_debug"] = {
                "semantic": round(semantic, 3),
                "frequency_bonus": round(freq_score, 3),
                "recency_bonus": round(recency_score, 3),
                "composite_score": round(composite, 3),
                "days_old": round(days_old, 1)
            }
            ranked.append((composite, mem))

        ranked.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in ranked]


class DecisionEngine:
    """Evaluates policy decisions for existing vs new memories."""

    def decide(self, existing_memory: Optional[Dict[str, Any]], new_memory: Dict[str, Any]) -> Decision:
        policy = PolicyFactory.get(new_memory["type"])
        if not policy:
            return Decision.STORE
        return policy.decide(existing_memory, new_memory)


class MemoryManager:
    """Manages memory persistence pipeline based on scores and policy decisions."""

    def __init__(self, repository: MemoryRepository, decision_engine: DecisionEngine, scorer: MemoryScorer, threshold: float = 0.5):
        self.repository = repository
        self.engine = decision_engine
        self.scorer = scorer
        self.threshold = threshold

    def process(self, memory: Dict[str, Any]) -> Decision:
        importance = self.scorer.score(memory)
        if importance < self.threshold:
            logger.info(f"Discarded memory: {memory.get('value')} (type: {memory.get('type')}, score: {importance})")
            return Decision.IGNORE

        memory["importance"] = importance
        existing = self.repository.get_by_key(memory["key"])
        if existing:
            memory["version"] = existing.get("version", 1)  
            memory["id"] = existing.get("id", memory["id"])

        decision = self.engine.decide(existing, memory)

        if decision == Decision.STORE:
            self.repository.save(memory)
            logger.info(f"Stored new memory: {memory['key']} = {memory['value']}")
        elif decision == Decision.IGNORE:
            logger.info(f"Ignored duplicate memory: {memory['key']}")
        elif decision == Decision.MERGE:
            self.repository.merge(memory["key"], memory)
            logger.info(f"Merged duplicate memory (increased frequency): {memory['key']}")
        elif decision == Decision.UPDATE:
            self.repository.update(memory)
            logger.info(f"Updated memory: {memory['key']} = {memory['value']}")

        return decision

    def get_all_memories(self) -> List[Dict[str, Any]]:
        return self.repository.get_all()


class Memory:
    """
    ChunkdUp — Persistent memory for AI agents.

    Usage:
        memory = Memory(store="memory")
        memory.remember("I use Python")
        results = memory.retrieve("What language?")
    """

    def __init__(
        self,
        store: str = "memory",
        connection_url: Optional[str] = None,
        data_dir: Optional[str] = None,
        llm_provider: str = "gemini",
        labs_dir: Optional[str] = None,
        repository: Optional[MemoryRepository] = None,
        **kwargs
    ):
        """
        Initialize the Memory system.

        Args:
            store: Storage backend ("memory" or "postgres")
            connection_url: PostgreSQL connection string (required for "postgres")
            data_dir: Directory to store conversation/file data (defaults to ./data)
            llm_provider: LLM provider to use ("gemini", "openai", etc.)
            labs_dir: Path to the labs directory containing LLM implementations
            repository: Custom MemoryRepository instance
        """
        self._store_type = store

        # Initialize repository based on store configuration
        if repository is not None:
            self.repository = repository
        elif store == "memory":
            self.repository = InMemoryRepository()
        elif store == "postgres":
            if not connection_url:
                raise ValueError("connection_url required for postgres store")
            from .postgres_repository import PostgresRepository
            self.repository = PostgresRepository(connection_url)
        else:
            raise ValueError(f"Unknown store: {store}")

        # Set up paths
        if data_dir is None:
            self.base_dir = os.getcwd()
            data_dir = os.path.join(self.base_dir, "data")

        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # Find labs directory for LLM imports
        if labs_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = current_dir
            while root_dir and not os.path.exists(os.path.join(root_dir, "labs")):
                parent = os.path.dirname(root_dir)
                if parent == root_dir:
                    break
                root_dir = parent
            if root_dir and os.path.exists(os.path.join(root_dir, "labs")):
                self.labs_dir = os.path.join(root_dir, "labs")
            else:
                self.labs_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "labs")
        else:
            self.labs_dir = labs_dir

        # Memory storage path
        self.memory_path = os.path.join(data_dir, "memory.json")

        # Initialize internal components
        self.decision_engine = DecisionEngine()
        self.scorer = MemoryScorer()
        self.manager = MemoryManager(self.repository, self.decision_engine, self.scorer)
        self.extractor = MemoryExtractor()
        self.retriever = MemoryRetriever(self.repository)
        self.prompt_builder = PromptBuilder()

        # Lazy-load LLM components
        self._llm = None
        self._parser = None
        self._validator = None
        self._llm_provider = llm_provider
        self._llm_loaded = False

        # Initialize conversation storage
        self.conversation_path = os.path.join(data_dir, "conversation.json")
        self._initialize_conversation()

        logger.info(f"Memory initialized with store: {store}")

    # ──────────────────────────────────────────────────────
    # Conversation Storage Helpers
    # ──────────────────────────────────────────────────────

    def _initialize_conversation(self) -> None:
        """Initialize conversation file if it doesn't exist."""
        if not os.path.exists(self.conversation_path):
            try:
                with open(self.conversation_path, "w", encoding="utf-8") as f:
                    json.dump([], f)
            except Exception as e:
                logger.warning(f"Could not create conversation file: {e}")

    def _load_conversation(self) -> List[str]:
        """Load conversation lines from file."""
        if not os.path.exists(self.conversation_path):
            return []
        try:
            with open(self.conversation_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    def _save_conversation(self, lines: List[str]) -> None:
        """Save conversation lines to file."""
        try:
            os.makedirs(os.path.dirname(self.conversation_path), exist_ok=True)
            with open(self.conversation_path, "w", encoding="utf-8") as f:
                json.dump(lines, f, indent=4)
        except Exception as e:
            logger.warning(f"Could not save conversation: {e}")

    def _get_conversation_text(self) -> str:
        """Get full conversation as a single string."""
        lines = self._load_conversation()
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────
    # LLM Helpers
    # ──────────────────────────────────────────────────────

    def _load_llm_components(self) -> bool:
        """Load LLM components from lab2 implementation if available."""
        if self._llm_loaded:
            return True

        try:
            lab2_path = os.path.join(self.labs_dir, "002-structured-outputs", "experiments", "experiment_001.py")
            if not os.path.exists(lab2_path):
                logger.warning(f"Lab2 experiment file not found at: {lab2_path}")
                return False

            spec = importlib.util.spec_from_file_location("lab2_experiment", lab2_path)
            if not spec or not spec.loader:
                return False
            lab2_module = importlib.util.module_from_spec(spec)
            sys.modules["lab2_experiment"] = lab2_module
            spec.loader.exec_module(lab2_module)

            self.LLMCaller = lab2_module.LLMCaller
            self.OutputParser = lab2_module.OutputParser
            self.OutputValidator = lab2_module.OutputValidator

            self._llm = self.LLMCaller(provider=self._llm_provider)
            self._parser = self.OutputParser()
            self._validator = self.OutputValidator(raise_on_fail=False)

            self._llm_loaded = True
            logger.info("LLM components loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"Could not initialize LLM components: {e}")
            self._llm_loaded = False
            self._llm = None
            self._parser = None
            self._validator = None
            return False

    def _get_llm(self):
        if not self._llm_loaded:
            self._load_llm_components()
        return self._llm

    def _get_parser(self):
        if not self._llm_loaded:
            self._load_llm_components()
        return self._parser

    def _get_validator(self):
        if not self._llm_loaded:
            self._load_llm_components()
        return self._validator

    # ──────────────────────────────────────────────────────
    # Core Operations
    # ──────────────────────────────────────────────────────

    def remember(self, text: str) -> Dict[str, Any]:
        """
        Store a memory from conversation text.

        Args:
            text: The conversation text to remember

        Returns:
            Dict with memory details and processed count

        Example:
            >>> memory.remember("I use Python")
            {"memories": [{"decision": "processed", "key": "programming_language", "value": "Python"}], "count": 1}
        """
        logger.info(f"Remembering: {text[:50]}...")

        # Add to conversation history
        lines = self._load_conversation()
        lines.append(text)
        self._save_conversation(lines)

        # Extract memories from current input text
        extracted = self.extractor.extract(text)

        results = []
        for mem in extracted:
            existing = self.repository.get_by_key(mem["key"])
            if existing:
                mem["id"] = existing.get("id", mem["id"])
            decision = self.manager.process(mem)
            results.append({
                "decision": decision.value.upper(),
                "key": mem.get("key"),
                "value": mem.get("value")
            })

        if not results:
            results.append({
                "decision": "IGNORE",
                "key": None,
                "value": None
            })

        return {"memories": results, "count": len(results)}

    def retrieve(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve memories matching a query.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching memories

        Example:
            >>> memory.retrieve("What language?")
            [{"key": "programming_language", "value": "Python", ...}]
        """
        logger.info(f"Retrieving: {query}")
        return self.retriever.retrieve(query, k=limit)

    def update(self, key: str, new_value: str) -> Dict[str, Any]:
        """
        Update a memory by key.

        Args:
            key: The memory key to update
            new_value: The new value

        Returns:
            Dict with update result

        Raises:
            ValueError: If memory with key is not found

        Example:
            >>> memory.update("programming_language", "I now use Go")
            {"decision": "UPDATE", "key": "programming_language", "value": "Go"}
        """
        logger.info(f"Updating {key} -> {new_value[:50]}...")

        existing = self.repository.get_by_key(key)
        if not existing:
            raise ValueError(f"Memory with key '{key}' not found")

        updated = dict(existing)
        updated["value"] = new_value
        updated["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        self.repository.update(updated)
        return {"decision": "UPDATE", "key": key, "value": new_value}

    def delete(self, memory_id: str) -> Dict[str, Any]:
        """
        Delete a memory by ID.

        Args:
            memory_id: The memory ID to delete

        Returns:
            Dict with deletion result

        Example:
            >>> memory.delete("abc-123")
            {"success": True, "id": "abc-123"}
        """
        logger.info(f"Deleting memory: {memory_id}")
        self.repository.delete(memory_id)
        return {"success": True, "id": memory_id}

    def search(self, query: str, limit: int = 5, semantic: bool = True) -> List[Dict[str, Any]]:
        """
        Search memories with optional semantic search.

        Args:
            query: Search query
            limit: Maximum results
            semantic: Use semantic search if available

        Returns:
            List of matching memories
        """
        logger.info(f"Searching: {query}")
        return self.retriever.retrieve(query, k=limit)

    # ──────────────────────────────────────────────────────
    # Utility Operations
    # ──────────────────────────────────────────────────────

    def get_all(self) -> List[Dict[str, Any]]:
        """Get all active memories."""
        return self.repository.get_all()

    def get_history(self, key: str) -> List[Dict[str, Any]]:
        """
        Get history of a memory by key.
        Requires repository supporting get_history.
        """
        if hasattr(self.repository, "get_history"):
            return getattr(self.repository, "get_history")(key)
        return []

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        all_memories = self.get_all()
        return {
            "total_memories": len(all_memories),
            "store_type": self._store_type,
            "keys": [m.get("key") for m in all_memories],
            "active_memories": len(all_memories),
            "conversation_lines": len(self._load_conversation())
        }

    def clear(self) -> None:
        """Clear all memories and conversation history."""
        self.repository.clear()
        self._save_conversation([])
        logger.info("Memory cleared")

    # ──────────────────────────────────────────────────────
    # Legacy & Compatibility Aliases
    # ──────────────────────────────────────────────────────

    def add(self, text: str) -> None:
        """Alias for remember()."""
        self.remember(text)

    def add_conversation_line(self, text: str) -> None:
        """Alias for remember()."""
        self.remember(text)

    def add_conversation(self, conversation: List[str]) -> None:
        """Add multiple conversation lines."""
        for line in conversation:
            self.remember(line)

    def query(self, question: str, k: int = 3, use_llm: bool = True) -> Dict[str, Any]:
        """Query system with answer generation."""
        memories = self.retriever.retrieve(question, k=k)
        prompt = self.prompt_builder.build(
            query=question,
            contexts={"memories": memories, "documents": []},
            variant="expert"
        )
        result = {
            "question": question,
            "memories_used": memories,
            "prompt": prompt,
            "use_llm": use_llm
        }
        if use_llm:
            llm = self._get_llm()
            if llm:
                try:
                    raw_response = llm.generate(prompt)
                    result["raw_response"] = raw_response
                    parser = self._get_parser()
                    validator = self._get_validator()
                    if parser and validator:
                        parsed = parser.parse(raw_response)
                        validated = validator.validate(parsed)
                        result["structured_answer"] = validated
                        result["answer"] = validated.get("answer", raw_response) if validated else raw_response
                        result["confidence"] = validated.get("confidence", 0.0) if validated else 0.0
                        result["citations"] = validated.get("citations", []) if validated else []
                    else:
                        result["answer"] = raw_response
                        result["confidence"] = 0.5
                        result["citations"] = []
                except Exception as e:
                    logger.error(f"Error getting LLM response: {e}")
                    result["error"] = str(e)
                    result["answer"] = self._simple_answer(question, memories)
                    result["confidence"] = 0.0
                    result["citations"] = []
            else:
                result["answer"] = self._simple_answer(question, memories)
                result["confidence"] = 0.0
                result["citations"] = []
                result["warning"] = "LLM not available, using simple answer"
        else:
            result["answer"] = self._simple_answer(question, memories)
            result["confidence"] = 0.0
            result["citations"] = []

        return result

    def _simple_answer(self, question: str, memories: List[Dict[str, Any]]) -> str:
        if not memories:
            return "I don't have any relevant memories about that."
        question_lower = question.lower()
        answers = []
        for mem in memories:
            key = mem.get("key", "").lower()
            value = mem.get("value", "")
            if key in question_lower or question_lower in key:
                answers.append(f"Your {mem['key']} is {value}")
            else:
                answers.append(f"Found memory: {mem['key']} = {value}")
        return "\n".join(answers[:3]) if answers else "I found some memories but they don't directly answer your question."

    def get_all_memories(self) -> List[Dict[str, Any]]:
        """Alias for get_all()."""
        return self.get_all()

    def get_conversation_history(self) -> List[str]:
        """Get conversation lines."""
        return self._load_conversation()

    def get_memories_by_type(self, memory_type: str) -> List[Dict[str, Any]]:
        """Get memories filtered by type."""
        return [m for m in self.get_all() if m.get("type") == memory_type]

    def reset(self) -> None:
        """Alias for clear()."""
        self.clear()

    def delete_memory(self, memory_id: str) -> bool:
        """Legacy delete memory method returning boolean."""
        self.delete(memory_id)
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Legacy alias for get_stats()."""
        return self.get_stats()