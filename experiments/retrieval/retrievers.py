# experiments/retrieval/retrievers.py
"""
Dense and BM25 candidate retrievers for the retrieval experiments.

Both operate on the shared fixture (see fixture.py), index only active
memories, and return candidate dicts carrying their native score field:
  - DenseRetriever  -> _similarity  (cosine of normalized vectors, [-1, 1])
  - BM25Retriever   -> _bm25_score  (raw BM25Okapi score, unbounded)

The dense retriever mirrors the embedding text representation used by
postgres_repository._embedding_text so the experiment matches the production
pgvector path as closely as possible.
"""
import math
import os
import re
from pathlib import Path
from typing import Dict, List

from ai_system_path import ai_system_path  # noqa: F401  (sys.path setup)
from chunkdup.retrieval import ContextualRetriever
from chunkdup.entity_extractor import EntityExtractor

from fixture import all_memories

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def index_text(memory: Dict) -> str:
    """Text representation used for both embedding and BM25 indexing."""
    contextual = memory.get("contextual_text") or ""
    return (
        f"key: {memory.get('key', '')}\n"
        f"type: {memory.get('type', '')}\n"
        f"value: {memory.get('value', '')}\n"
        f"contextualized source: {contextual}"
    ).strip()


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+.\-]*")

# Minimal English stopword set. With only ~12 corpus documents BM25's IDF
# cannot reliably downweight function words, and without this set queries like
# "What is the weather forecast today?" leak matches via "is"/"the".
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "i", "my", "me", "you", "your", "it", "its", "of", "to", "in", "on",
    "for", "with", "and", "or", "do", "does", "did", "am",
    "what", "which", "who", "whose", "where", "when", "how", "why",
}


def tokenize(text: str) -> List[str]:
    """Single tokenizer shared by BM25 indexing and querying.

    Keeps internal '.', '-', '+', '_' so identifiers like all-MiniLM-L6-v2,
    C++, pgvector and 22.04 survive as single tokens; drops stopwords.
    """
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class DenseRetriever:
    """Dense retrieval with the local all-MiniLM-L6-v2 embedding model."""

    def __init__(self, memories: List[Dict] = None):
        from sentence_transformers import SentenceTransformer

        self._source_memories = memories if memories is not None else all_memories()
        local_path = _REPO_ROOT / "models" / "all-MiniLM-L6-v2"
        if local_path.exists():
            self.model_source = str(local_path)
        else:
            # Fall back to the package cache; recorded so runs are traceable.
            self.model_source = os.getenv("CHUNKDUP_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.model = SentenceTransformer(self.model_source)
        self._contextual = ContextualRetriever()
        self._embeddings: List[List[float]] = []
        self._memories: List[Dict] = []
        self.index()

    def index(self) -> None:
        self._memories = []
        texts = []
        for memory in self._source_memories:
            if memory.get("status") != "active":
                continue
            memory = dict(memory)
            memory.setdefault("contextual_text", self._contextual.build_contextual_text(memory))
            texts.append(index_text(memory))
            self._memories.append(memory)
        # Batched encode (same normalization as before; one call instead of N).
        vectors = self.model.encode(texts, normalize_embeddings=True).tolist() if texts else []
        self._embeddings = vectors

    def search(self, query: str, limit: int = 20) -> List[Dict]:
        if not self._memories or not query.strip():
            return []
        q_vec = self.model.encode(query, normalize_embeddings=True).tolist()
        scored = []
        for memory, m_vec in zip(self._memories, self._embeddings):
            similarity = sum(a * b for a, b in zip(q_vec, m_vec))
            candidate = dict(memory)
            candidate["_similarity"] = similarity
            scored.append(candidate)
        scored.sort(key=lambda c: c["_similarity"], reverse=True)
        return scored[:limit]


class BM25Retriever:
    """BM25 keyword retrieval (rank_bm25.BM25Okapi) over index_text()."""

    def __init__(self, memories: List[Dict] = None):
        from rank_bm25 import BM25Okapi

        self._contextual = ContextualRetriever()
        corpus = []
        self._memories = []
        for memory in (memories if memories is not None else all_memories()):
            if memory.get("status") != "active":
                continue
            memory = dict(memory)
            memory.setdefault("contextual_text", self._contextual.build_contextual_text(memory))
            corpus.append(tokenize(index_text(memory)))
            self._memories.append(memory)
        if not corpus:
            self._bm25 = None
        else:
            self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, limit: int = 20) -> List[Dict]:
        if self._bm25 is None or not query.strip():
            return []
        scores = self._bm25.get_scores(tokenize(query))
        scored = []
        for memory, score in zip(self._memories, scores):
            candidate = dict(memory)
            candidate["_bm25_score"] = float(score)
            scored.append(candidate)
        scored.sort(key=lambda c: c["_bm25_score"], reverse=True)
        # BM25 gives every document a (possibly zero) score; drop the zeros so
        # the fusion sees only documents sharing at least one term.
        scored = [c for c in scored if c["_bm25_score"] > 0.0]
        return scored[:limit]


class EntityScorer:
    """Entity-overlap scorer reused from the existing EntityExtractor.

    Produces _entity_score = |query_entities ∩ memory_entities| / |query_entities|,
    matching the overlap computation in retrieval_class.py.
    """

    def __init__(self):
        self._extractor = EntityExtractor()
        self._memory_entities: List[Dict] = []
        for memory in all_memories():
            if memory.get("status") != "active":
                continue
            text = f"{memory.get('key', '')} {memory.get('value', '')} " \
                   f"{memory.get('contextual_text', '')}"
            self._memory_entities.append((memory["id"], {e.casefold() for e in self._extractor.extract(text)}))

    def score_memory(self, memory: Dict, query_entities: set) -> float:
        if not query_entities:
            return 0.0
        memory_entities = {e.casefold() for e in memory.get("entities", [])} or next(
            (ents for mid, ents in self._memory_entities if mid == memory["id"]), set()
        )
        overlap = len(query_entities & memory_entities)
        return overlap / len(query_entities)

    def extract(self, query: str) -> set:
        return {e.casefold() for e in self._extractor.extract(query)}


def cosine_check() -> float:
    """Sanity helper used by tests: two identical texts must score ~1.0."""
    dense = DenseRetriever()
    top = dense.search("Python", limit=1)
    return top[0]["_similarity"] if top else float("nan")
