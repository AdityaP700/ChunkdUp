# ai-system/chunkdup/retrieval/contextual.py
from typing import Dict, Any

class ContextualRetriever:
    """
    Implements Anthropic-style Contextual Retrieval by enriching memories
    at INDEXING time before vector embedding and BM25 indexing.
    """
    @staticmethod
    def build_contextual_text(memory: Dict[str, Any]) -> str:
        """
        Prepend memory-specific explanatory context at indexing time.
        """
        memory_type = memory.get("type", "unknown")
        key = memory.get("key", "attribute")
        value = memory.get("value", "")
        topic = memory.get("conversation_topic", "general user context")
        source = memory.get("source", "user preference")

        context_template = (
            f"This memory is a {memory_type} regarding '{key}' from a conversation about {topic}. "
            f"Source domain: {source}. The recorded user context state is: \"{value}\"."
        )
        return context_template.strip()
