# Retrieval Baseline Report (Phase A)

> Frozen reference for all retrieval experiments. Written after full inspection;
> nothing in this report reflects behavior changes — it records what existed and what it measured.
> Companion: [CONTEXT-CHECK.md](CONTEXT-CHECK.md) · [retrieval-experiments.md](retrieval-experiments.md)

## 1. Architecture and Execution Flow

`Memory.retrieve(query, limit)` (`ai-system/chunkdup/memory.py`) delegates to
`MemoryRetriever.retrieve(query, k, threshold=0.15)` (`ai-system/chunkdup/retrieval_class.py`):

1. **TTL cache** (`retrieval/cache.py`) — 300 s, 1000 entries, no invalidation on writes.
2. **Exact-lookup fast path** — queries ≤ 2 words go to `repository.get_by_key()` and return a single memory, bypassing ranking.
3. **Candidate pool** = `max(10, 2k)` (`retrieval/adaptive_k.py`).
4. **Query rewriting** (`retrieval/query_rewriter.py`) — filler removal (suppressed for preference/career queries), abbreviation expansion, hardcoded expansions for synthesis and preference queries.
5. **Candidate generation**:
   - keyword leg: `repository.search()` — substring matching for `InMemoryRepository`; plus entity-overlap candidates (`EntityExtractor`, `_entity_score`).
   - semantic leg: `repository.search_semantic()` — exists only on `PostgresRepository` (pgvector). **The in-memory store has no dense leg**, so the benchmark never exercised embeddings.
6. **Threshold filters** — keyword `_score ≥ 0.15`; semantic filter `_similarity > -1.0` (no-op).
7. **Fusion** (`retrieval/hybrid_search.py`) — weighted RRF (keyword 0.4 / semantic 0.6, k=60) × static `type_boost` × dynamic intent boost (hand-written career/preference/editor rules) + additive entity bonus (`0.5 × _entity_score`).
8. **Reranking** (`retrieval/reranker.py`) — `heuristic` (0.75·similarity + 0.15·lexical + 0.10·hybrid, × type weights), `llm` (token-overlap simulation, not an LLM), `cross_encoder` (`cross-encoder/ms-marco-MiniLM-L-12-v2`, silently falls back to heuristic on load failure).
9. **Adaptive k** — expands k for synthesis keywords; returns 1 result (not 0) when `max_score < 0.01`.
10. **Cache store**, return.

## 2. Files Responsible

| Concern | File |
|---|---|
| Orchestration | `ai-system/chunkdup/retrieval_class.py` |
| Fusion + boosts | `ai-system/chunkdup/retrieval/hybrid_search.py` |
| Reranking | `ai-system/chunkdup/retrieval/reranker.py` |
| Pool sizing | `ai-system/chunkdup/retrieval/adaptive_k.py` |
| Query rewriting | `ai-system/chunkdup/retrieval/query_rewriter.py` |
| Contextual indexing | `ai-system/chunkdup/retrieval/contextual.py` |
| Caching | `ai-system/chunkdup/retrieval/cache.py` |
| Entities | `ai-system/chunkdup/entity_extractor.py` |
| Embeddings | `ai-system/chunkdup/embeddings.py` |
| Storage | `in_memory_repository.py`, `postgres_repository.py`, `repository.py` |
| Evaluation (old) | `evaluation/retrieval_benchmarks.py`, `evaluation/retrieval_results.json` |
| Policy evaluation | `evaluation/runner.py`, `evaluation/scenarios.json` |

## 3. Frozen Benchmark Results (as found, then reproduced)

`evaluation/retrieval_benchmarks.py` — 7 queries, 10-memory fixture, k=5:

| Reranker mode | MRR | Mean P@K | Mean R@K |
|---|---|---|---|
| `heuristic` | 0.7857 | 0.3821 | 0.8571 |
| `llm` (simulated) | 0.7857 | 0.3821 | 0.8571 |

- As found in `evaluation/retrieval_results.json` (prior sessions).
- **Reproduced exactly** on 2026-10-12 in the rebuilt Windows Python 3.11 environment — byte-identical metrics, confirming the baseline is deterministic and correctly frozen.

Known per-query behavior in the frozen set:
- `"What is the weather forecast today?"` → returns `["Python"]` (heuristic mode) — out-of-domain false positive.
- Language/editor/preference queries: recall 1.0 but precision 0.2–0.25 (many irrelevant co-returns).
- `"Where do I work and what is my role?"` → RR 0.5 (irrelevant memory outranks a relevant one).

Historical milestones (self-reported in `learnings/current_imp.md`, not re-verified): MRR 0.4571 → 0.5714 → 0.8333 across fix rounds; policy decision runner 54/54 at 100 % decision and value accuracy.

## 4. Known Limitations (found by inspection)

1. **No dense retrieval for the in-memory store** — `search_semantic` absent; "semantic" leg always empty in the benchmark.
2. **No BM25** — keyword leg is naive substring matching.
3. **Non-standard RRF** with hardcoded type/intent/entity bonuses, no experimental support.
4. **`llm` reranker mode is a simulation** (token overlap); both saved modes produce identical metrics.
5. **Cross-encoder mode never evaluated**; silent fallback to heuristic would invalidate an evaluation.
6. **Abstention is partial**: threshold on substring-match counts only; semantic filter is a no-op; adaptive-k floors at 1 result; the weather query leaks.
7. **Overfitting risk**: rewriter expansions, intent boosts, and adaptive-k keyword lists are tuned to the 7 benchmark queries.
8. **Fast path** masks precision problems on short queries.
9. **Cache staleness** within TTL after memory writes.
10. **Test gaps**: `tests/test_ranking.py`, `test_semantic_scope.py`, `test_semantic_similarity.py` are expectation data files without assertions; queries like "Where does ChunkdUp run?" reference memories absent from the benchmark fixture.

## 5. Environment Failures (documented, not silently worked around)

- The prior WSL virtualenv (Python 3.12.3) was **empty**; `wheels/` held orphaned cp311 Linux wheels that could not install into it. Root cause matches commit `575fdc7 "fix: isolate dependency mismatch"` — the runnable environment no longer existed.
- Resolution: built a fresh Windows Python 3.11 environment (`torch` CPU wheel, `sentence-transformers`, `rank-bm25`). `requirements.txt` was NOT modified. The cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-12-v2`, ~90 MB) is a one-time HuggingFace download, cached locally.
- `tests/test_postgres.py` excluded (requires a live PostgreSQL); all other tests pass.

## 6. Baseline Freeze Statement

`evaluation/retrieval_benchmarks.py`, `evaluation/retrieval_results.json`, and the production retrieval path were not modified. The baseline remains runnable via `python evaluation/retrieval_benchmarks.py` and serves as the reference point for all experiment configurations in [retrieval-experiments.md](retrieval-experiments.md).
