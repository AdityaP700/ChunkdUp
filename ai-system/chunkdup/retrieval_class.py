from typing import List, Dict, Any, Optional
from .retrieval import (
    ContextualRetriever,
    HybridSearchEngine,
    PluggableReranker,
    AdaptiveKSelector,
    RetrievalCache,
    QueryRewriter
)

class MemoryRetriever:
    """
    Orchestrates Contextual Indexing, Query Rewriting, Hybrid RRF Search,
    Pluggable Reranking, Adaptive K Selection, and TTL Caching.
    """
    def __init__(
        self,
        repository,
        reranker_mode: str = "heuristic",
        enable_cache: bool = True,
        enable_rewriter: bool = True
    ):
        self.repository = repository
        self.contextual_builder = ContextualRetriever()
        self.hybrid_engine = HybridSearchEngine()
        self.reranker = PluggableReranker(mode=reranker_mode)
        self.adaptive_k_selector = AdaptiveKSelector()
        self.cache = RetrievalCache() if enable_cache else None
        self.rewriter = QueryRewriter() if enable_rewriter else None

    def index_memory(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enriches memory with Anthropic-style contextual prefix before persistent save.
        """
        memory_copy = dict(memory)
        memory_copy["contextual_text"] = self.contextual_builder.build_contextual_text(memory_copy)
        return memory_copy

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        # 1. Cache hit lookup
        if self.cache:
            cached_results = self.cache.get(query)
            if cached_results is not None:
                return cached_results[:k]

        # 2. Query Rewriting (normalize abbreviations, clean filler words)
        search_query = self.rewriter.rewrite(query) if self.rewriter else query

        # 3. Candidate Generation via Hybrid Search (pgvector + keyword)
        keyword_candidates = self.repository.search(search_query, limit=k * 3) if hasattr(self.repository, "search") else []
        semantic_candidates = self.repository.search_semantic(search_query, limit=k * 3) if hasattr(self.repository, "search_semantic") else []

        # If repo doesn't support vector search directly (e.g. basic memory repo), fallback to get_all match
        if not keyword_candidates and not semantic_candidates and hasattr(self.repository, "get_all"):
            all_memories = self.repository.get_all()
            active_mems = [m for m in all_memories if m.get("status") == "active"]
            q_words = set(search_query.lower().split())
            for m in active_mems:
                c_text = m.get("contextual_text", f"{m.get('key')} {m.get('value')} {m.get('type')}")
                overlap = len(set(c_text.lower().split()) & q_words)
                m["_score"] = float(overlap)
            keyword_candidates = sorted(active_mems, key=lambda x: x.get("_score", 0), reverse=True)[:k*3]

        # 4. RRF Fusion + Type Boost
        candidates = self.hybrid_engine.combine(keyword_candidates, semantic_candidates, limit=k * 3)

        # 5. Pluggable Reranking (Cross-Encoder / LLM / Heuristic)
        reranked_results = self.reranker.rerank(search_query, candidates, limit=k * 2)

        # 6. Adaptive K Selection
        final_k = self.adaptive_k_selector.select_k(search_query, reranked_results, target_k=k)
        final_results = reranked_results[:final_k]

        # 7. Store in Cache
        if self.cache:
            self.cache.set(query, final_results)

        return final_results
