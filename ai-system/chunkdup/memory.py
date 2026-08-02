import os
import json
import math
import re
import uuid
import sys
import importlib.util
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the local modules (policies, prompt_builder, retrieval_class)
# These should be in the same directory or available in sys.path
try:
    from .policies import Decision, PolicyFactory
    from .prompt_builder import PromptBuilder
    from .retrieval_class import MemoryRetriever
except ImportError:
    # Fallback for when running as a standalone script
    from policies import Decision, PolicyFactory
    from prompt_builder import PromptBuilder
    from retrieval_class import MemoryRetriever


class MemoryExtractor:
    def __init__(self):
        self.rules = [
            {
                "pattern": r"I'm building ([\w\d\-]+) in ([\w\d\+#]+)",
                "type": "project",
                "key": "project_name",
                "value_group": 1,
                "meta": {"language": 2}
            },
            {
                "pattern": r"I prefer ([\w\s]+?)\.",
                "type": "preference",
                "key": "response_style",
                "value_group": 1
            },
            {
                "pattern": r"I use (Windows|Mac|Linux)",
                "type": "environment",
                "key": "os",
                "value_group": 1
            },
            {
                "pattern": r"I use (Python|Java|C\+\+|Rust|Go)",
                "type": "environment",
                "key": "programming_language",
                "value_group": 1
            },
            {
                "pattern": r"My favorite editor is ([\w\s]+?)\.",
                "type": "tool",
                "key": "editor",
                "value_group": 1
            }
        ]

    def extract(self, conversation: str) -> List[Dict[str, Any]]:
        memories = []
        for rule in self.rules:
            matches = re.finditer(rule["pattern"], conversation, re.IGNORECASE)
            for match in matches:
                value = match.group(rule["value_group"]).strip()
                memory = {
                    "id": str(uuid.uuid4()),
                    "type": rule["type"],
                    "key": rule["key"],
                    "value": value,
                    "frequency": 1,
                    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "source": "conversation",
                    "status": "active"
                }
                if "meta" in rule:
                    for meta_key, group_idx in rule["meta"].items():
                        memory[meta_key] = match.group(group_idx).strip()
                memories.append(memory)
        return memories


class MemoryScorer:
    def score(self, memory: Dict[str, Any]) -> float:
        scores = {
            "project": 1.0,
            "preference": 0.9,
            "environment": 0.8,
            "tool": 0.7,
            "question": 0.3
        }
        return scores.get(memory.get("type"), 0.5)


class MemoryRanker:
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


class MemoryRepository:
    def __init__(self, path: str):
        self.path = path

    def load(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []

    def save(self, memories: List[Dict[str, Any]]) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(memories, f, indent=4)

    def add(self, memory: Dict[str, Any]) -> None:
        memories = self.load()
        memories.append(memory)
        self.save(memories)

    def update(self, new_memory: Dict[str, Any]) -> None:
        memories = self.load()
        for mem in memories:
            if mem.get("key") == new_memory.get("key") and mem.get("status") == "active":
                for k, v in new_memory.items():
                    if k not in ["id", "created_at", "frequency"]:
                        mem[k] = v
                mem["frequency"] = 1
                mem["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                break
        self.save(memories)

    def merge(self, existing_memory: Dict[str, Any], new_memory: Dict[str, Any]) -> None:
        memories = self.load()
        for mem in memories:
            if mem.get("key") == existing_memory.get("key") and mem.get("status") == "active":
                mem["frequency"] = mem.get("frequency", 1) + 1
                mem["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                break
        self.save(memories)

    def find_active_by_key(self, key: str) -> Optional[Dict[str, Any]]:
        memories = self.load()
        for memory in memories:
            if memory.get("key") == key and memory.get("status") == "active":
                return memory
        return None

    def get_all(self) -> List[Dict[str, Any]]:
        return self.load()


class DecisionEngine:
    def decide(self, existing_memory: Optional[Dict[str, Any]], new_memory: Dict[str, Any]) -> Decision:
        policy = PolicyFactory.get(new_memory["type"])
        if not policy:
            return Decision.STORE
        return policy.decide(existing_memory, new_memory)


class MemoryManager:
    def __init__(self, repository: MemoryRepository, decision_engine: DecisionEngine, scorer: MemoryScorer):
        self.repository = repository
        self.engine = decision_engine
        self.scorer = scorer
        self.threshold = 0.6

    def process(self, memory: Dict[str, Any]) -> None:
        importance = self.scorer.score(memory)
        if importance < self.threshold:
            logger.info(f"Discarded memory: {memory.get('value')} (type: {memory.get('type')}, score: {importance})")
            return

        memory["importance"] = importance
        existing = self.repository.find_active_by_key(memory["key"])
        decision = self.engine.decide(existing, memory)

        if decision == Decision.STORE:
            self.repository.add(memory)
            logger.info(f"Stored new memory: {memory['key']} = {memory['value']}")
        elif decision == Decision.IGNORE:
            logger.info(f"Ignored duplicate memory: {memory['key']}")
        elif decision == Decision.MERGE:
            self.repository.merge(existing, memory)
            logger.info(f"Merged duplicate memory (increased frequency): {memory['key']}")
        elif decision == Decision.UPDATE:
            self.repository.update(memory)
            logger.info(f"Updated memory: {memory['key']} = {memory['value']}")

    def get_all_memories(self) -> List[Dict[str, Any]]:
        return self.repository.get_all()


class Memory:
    """
    Main entry point for memory operations.
    Users only need to interact with this class.
    """

    def __init__(self, data_dir: Optional[str] = None, llm_provider: str = "gemini",
                 labs_dir: Optional[str] = None):
        """
        Initialize the Memory system with all internal components.

        Args:
            data_dir: Directory to store memory data (defaults to ./data)
            llm_provider: LLM provider to use ("gemini", "openai", etc.)
            labs_dir: Path to the labs directory containing LLM implementations
        """
        # Set up paths
        if data_dir is None:
            self.base_dir = os.getcwd()
            data_dir = os.path.join(self.base_dir, "data")

        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # Find labs directory for LLM imports
        if labs_dir is None:
            # Try to find labs directory relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up to find the root (where labs/ is located)
            root_dir = current_dir
            while root_dir and not os.path.exists(os.path.join(root_dir, "labs")):
                root_dir = os.path.dirname(root_dir)
            if root_dir:
                self.labs_dir = os.path.join(root_dir, "labs")
            else:
                # Fallback: assume labs is in the parent directory
                self.labs_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "labs")
        else:
            self.labs_dir = labs_dir

        # Memory storage path
        self.memory_path = os.path.join(data_dir, "memory.json")

        # Initialize all internal components
        self.repository = MemoryRepository(self.memory_path)
        self.decision_engine = DecisionEngine()
        self.scorer = MemoryScorer()
        self.manager = MemoryManager(self.repository, self.decision_engine, self.scorer)
        self.extractor = MemoryExtractor()

        # Use the imported MemoryRetriever and PromptBuilder
        self.retriever = MemoryRetriever(self.repository)
        self.prompt_builder = PromptBuilder()

        # Lazy-load LLM components
        #A restaurant menu - you don't cook
        # all dishes upfront,
        # you cook them when a customer orders.
        self._llm = None
        self._parser = None
        self._validator = None
        self._llm_provider = llm_provider
        self._llm_loaded = False

        # Initialize conversation storage
        self.conversation_path = os.path.join(data_dir, "conversation.json")
        self._initialize_conversation()
    #Conversations are stored persistently on disk
    #Even if your program crashes, the conversation history is saved
    #You can restart your program and continue from where you left off
    def _initialize_conversation(self):
        """Initialize conversation file if it doesn't exist."""
        if not os.path.exists(self.conversation_path):
            with open(self.conversation_path, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _load_conversation(self) -> List[str]:
        """Load conversation lines from file."""
        if not os.path.exists(self.conversation_path):
            return []
        with open(self.conversation_path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []

    def _save_conversation(self, lines: List[str]) -> None:
        """Save conversation lines to file."""
        os.makedirs(os.path.dirname(self.conversation_path), exist_ok=True)
        with open(self.conversation_path, "w", encoding="utf-8") as f:
            json.dump(lines, f, indent=4)

    def _get_conversation_text(self) -> str:
        """Get full conversation as a single string."""
        lines = self._load_conversation()
        return "\n".join(lines)

    def _load_llm_components(self):
        """Load LLM components from lab2 implementation."""
        if self._llm_loaded:
            return True

        try:
            # Try to import from lab2 experiment file
            lab2_path = os.path.join(self.labs_dir, "002-structured-outputs", "experiments", "experiment_001.py")

            if not os.path.exists(lab2_path):
                logger.warning(f"Lab2 experiment file not found at: {lab2_path}")
                return False

            # Use importlib to load the lab2 module
            spec = importlib.util.spec_from_file_location("lab2_experiment", lab2_path)
            lab2_module = importlib.util.module_from_spec(spec)
            sys.modules["lab2_experiment"] = lab2_module
            spec.loader.exec_module(lab2_module)

            # Get the classes from the module
            self.LLMCaller = lab2_module.LLMCaller
            self.OutputParser = lab2_module.OutputParser
            self.OutputValidator = lab2_module.OutputValidator

            # Initialize instances
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
        """Get LLM instance, loading if necessary."""
        if not self._llm_loaded:
            self._load_llm_components()
        return self._llm

    def _get_parser(self):
        """Get parser instance, loading if necessary."""
        if not self._llm_loaded:
            self._load_llm_components()
        return self._parser

    def _get_validator(self):
        """Get validator instance, loading if necessary."""
        if not self._llm_loaded:
            self._load_llm_components()
        return self._validator

    def add_conversation_line(self, text: str) -> None:
        """
        Add a conversation line and extract memories from it.

        Args:
            text: The conversation text to process
        """
        # Save to conversation history
        lines = self._load_conversation()
        lines.append(text)
        self._save_conversation(lines)

        # Extract memories from full conversation
        conversation_text = self._get_conversation_text()
        extracted_memories = self.extractor.extract(conversation_text)

        # Process each extracted memory
        for memory in extracted_memories:
            self.manager.process(memory)

    def add(self, text: str) -> None:
        """
        Alias for add_conversation_line.
        """
        self.add_conversation_line(text)

    def add_conversation(self, conversation: List[str]) -> None:
        """
        Add multiple conversation lines at once.

        Args:
            conversation: List of conversation lines
        """
        for line in conversation:
            self.add_conversation_line(line)

    def query(self, question: str, k: int = 3, use_llm: bool = True) -> Dict[str, Any]:
        """
        Query the memory system.

        Args:
            question: The question to ask
            k: Number of memories to retrieve
            use_llm: Whether to use LLM for answer generation

        Returns:
            Dict containing answer, retrieved memories, and raw response
        """
        # Retrieve relevant memories
        memories = self.retriever.retrieve(question, k=k)

        # Build prompt
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

        # Get LLM response if requested and available
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
        """Simple fallback answer when LLM is not available."""
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

        if answers:
            return "\n".join(answers[:3])
        else:
            return "I found some memories but they don't directly answer your question."

    def get_all_memories(self) -> List[Dict[str, Any]]:
        """Get all stored memories."""
        return self.manager.get_all_memories()

    def get_conversation_history(self) -> List[str]:
        """Get the conversation history."""
        return self._load_conversation()

    def get_memories_by_type(self, memory_type: str) -> List[Dict[str, Any]]:
        """Get memories filtered by type."""
        all_memories = self.get_all_memories()
        return [m for m in all_memories if m.get("type") == memory_type]

    def clear(self) -> None:
        """Clear all data (memories and conversation)."""
        self.repository.save([])
        logger.info("Cleared all memories")
        self._save_conversation([])
        logger.info("Cleared conversation history")

    def reset(self) -> None:
        """Alias for clear()."""
        self.clear()

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a specific memory by ID (mark as inactive).

        Args:
            memory_id: The ID of the memory to delete

        Returns:
            bool: True if deleted, False if not found
        """
        memories = self.repository.load()
        for mem in memories:
            if mem.get("id") == memory_id:
                mem["status"] = "inactive"
                mem["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                self.repository.save(memories)
                logger.info(f"Deleted memory: {mem.get('key')} = {mem.get('value')}")
                return True
        return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the memory system."""
        memories = self.get_all_memories()
        active_memories = [m for m in memories if m.get("status") == "active"]

        stats = {
            "total_memories": len(memories),
            "active_memories": len(active_memories),
            "memories_by_type": {},
            "conversation_lines": len(self._load_conversation())
        }

        for mem in active_memories:
            mem_type = mem.get("type", "unknown")
            stats["memories_by_type"][mem_type] = stats["memories_by_type"].get(mem_type, 0) + 1

        return stats


# For backward compatibility, expose individual components
__all__ = [
    "Memory",
    "MemoryExtractor",
    "MemoryScorer",
    "MemoryRanker",
    "MemoryRepository",
    "MemoryManager",
    "DecisionEngine",
    "PromptBuilder",
    "MemoryRetriever"
]