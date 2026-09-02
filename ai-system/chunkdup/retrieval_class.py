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

    def retrieve(self, query: str, k: int = 5, threshold: float = 0.15) -> List[Dict[str, Any]]:
        # 1. Cache hit lookup
        if self.cache:
            cached_results = self.cache.get(query)
            if cached_results is not None:
                return cached_results[:k]

        # 2. Fix 2: Fast Path for Exact Lookup (e.g. single/two word queries like "Python")
        if len(query.strip().split()) <= 2 and hasattr(self.repository, "get_by_key"):
            exact = self.repository.get_by_key(query.strip().lower())
            if exact:
                return [exact]

        # 3. Candidate pool calculation (Generous candidate pool = min 10)
        candidate_k = self.adaptive_k_selector.select_candidate_k(target_k=k)

        # 4. Query Rewriting (normalize abbreviations without stripping critical intent)
        search_query = self.rewriter.rewrite(query) if self.rewriter else query

        # 5. Candidate Generation via Hybrid Search (pgvector + keyword)
        keyword_candidates = self.repository.search(search_query, limit=candidate_k) if hasattr(self.repository, "search") else []
        semantic_candidates = self.repository.search_semantic(search_query, limit=candidate_k, threshold=threshold) if hasattr(self.repository, "search_semantic") else []

        # If repo doesn't support vector search directly (e.g. basic memory repo), perform contextual token match
        if not keyword_candidates and not semantic_candidates and hasattr(self.repository, "get_all"):
            all_memories = self.repository.get_all()
            active_mems = [m for m in all_memories if m.get("status") == "active"]
            q_words = set(search_query.lower().split())
            
            matched = []
            for m in active_mems:
                c_text = m.get("contextual_text", f"{m.get('key')} {m.get('value')} {m.get('type')} {m.get('conversation_topic', '')}")
                c_words = set(c_text.lower().split())
                overlap = len(c_words & q_words)
                
                # Check for direct value/key containment or multi-word match
                val_str = str(m.get("value", "")).lower()
                key_str = str(m.get("key", "")).lower()
                type_str = str(m.get("type", "")).lower()
                
                is_hit = overlap > 0 or any(w in val_str or w in key_str or w in type_str for w in q_words)
                if is_hit:
                    score = float(overlap) + (2.0 if val_str in search_query.lower() or search_query.lower() in val_str else 0.5)
                    m_copy = dict(m)
                    m_copy["_score"] = score
                    matched.append(m_copy)

            matched.sort(key=lambda x: x.get("_score", 0), reverse=True)
            keyword_candidates = matched[:candidate_k]

        # Filter out low relevance candidates (Similarity Threshold Enforcement)
        keyword_candidates = [r for r in keyword_candidates if r.get("_score", 1.0) >= threshold]
        semantic_candidates = [r for r in semantic_candidates if r.get("_similarity", 1.0) >= threshold]

        if not keyword_candidates and not semantic_candidates:
            return []

        # 6. RRF Fusion + Intent Type Boost
        candidates = self.hybrid_engine.combine(query, keyword_candidates, semantic_candidates, limit=candidate_k)

        # 7. Pluggable Reranking (Cross-Encoder / LLM / Heuristic)
        reranked_results = self.reranker.rerank(search_query, candidates, limit=candidate_k)

        # 8. Adaptive K Selection
        final_k = self.adaptive_k_selector.select_k(search_query, reranked_results, target_k=k)
        final_results = reranked_results[:final_k]

        # 9. Store in Cache
        if self.cache:
            self.cache.set(query, final_results)

        return final_results
