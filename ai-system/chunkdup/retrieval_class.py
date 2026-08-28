from .repository import MemoryRepository
class MemoryRetriever:
    def __init__(self, repository:MemoryRepository):
        self.repository = repository

    # our job is to retrieve ,hence
    # retrieve the top k elements ,lets say 3
    def retrieve(self, query: str, k: int = 3):
        # our job is to load active memories
        # active in the sense which are there
        # in the codebase
        # memories = self.repository.get_all()
        # active = [
        #     m for m in memories
        #     if m.get("status") == "active"
        # ]

        # # now the thing is once we got the active
        # # memories then how come we know the
        # # keyword overlap or not
        # query_words = set(query.lower().split())

        # for memory in active:
        #     # combine value, key, and type for a broader abstraction search
        #     val = str(memory.get("value", "")).lower()
        #     key = str(memory.get("key", "")).lower()
        #     m_type = str(memory.get("type", "")).lower()

        #     combined_text = f"{val} {key} {m_type}"
        #     memory_words = set(combined_text.split())

        #     # use & for set intersection, not &&
        #     overlap = len(memory_words & query_words)
        #     memory["retrieval_score"] = overlap

        # # filter out zero score memories if desired, but we'll just sort them
        # ranked = sorted(
        #     active,
        #     key=lambda x: x.get("retrieval_score", 0),
        #     reverse=True
        # )
        results = self.repository.search(query)
        return results[:k]
