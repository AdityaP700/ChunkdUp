# Implementation Plan — Retrieval Pipeline Experiments (ChunkdUp)

> Companion documents: [CONTEXT-CHECK.md](CONTEXT-CHECK.md) · [REMAINING-WORK.md](REMAINING-WORK.md)
> Principle: small, evidence-driven retrieval pipeline. Reuse existing components; preserve the
> public `Memory` SDK API; every design choice below must be justified by a measurement, not an assumption.

---

## 1. Target Architecture

One retrieval engine with independently toggleable stages, plus one shared evaluation harness.
Five approaches are **configurations of this engine**, not five separate systems.

```
Memory.retrieve(query, limit)
  └─ MemoryRetriever.retrieve(query, k, config)
       1. (existing) TTL cache                — kept, untouched
       2. (existing) exact-lookup fast path   — kept for SDK behavior, bypassed in harness
       3. Candidate generation (pluggable)    — dense | bm25 | both
       4. Fusion (pluggable)                  — none | standard RRF (k configurable, pool size configurable)
       5. Reranking (pluggable)               — none | cross-encoder | existing heuristic
       6. Relevance decision (pluggable)      — none | calibrated threshold (per-pipeline semantics)
       7. (existing) adaptive k               — kept, evaluated as-is
```

- The default configuration reproduces today's behavior (keyword leg + heuristic reranker) so the baseline stays available.
- Score components (`_similarity`, `_bm25_score`, `_rrf_score`, `_rerank_score`, `_score`) are preserved on every candidate for Phase E tracing.

## 2. File Layout

```text
experiments/retrieval/
  dataset.jsonl              # ~60 reviewed queries (Phase B) — dev + held-out split flagged per row
  fixture.py                 # single memory fixture (10 baseline + project-fact memories), used by every run
  retrievers.py              # DenseRetriever, BM25Retriever (new, Phase C1/C2)
  fusion.py                  # standard RRF fusion (Phase C3)
  abstention.py              # relevance-decision stage (Phase C5)
  config.py                  # ExperimentConfig dataclass + the 8 named configurations
  run_experiment.py          # CLI: python experiments/retrieval/run_experiment.py --config hybrid_ce_abstain
  evaluate.py                # metric computation (shared by run_experiment)
  results/                   # machine-readable per-config JSON + trace JSONL
  calibrate.py               # threshold calibration on the DEV split only

docs/
  retrieval-baseline.md      # Phase A report (architecture, execution flow, frozen metrics)
  retrieval-experiments.md   # final comparison, per-query analysis, decision (Phases D/E)

tests/
  test_bm25_retriever.py     # unit: tokenizer consistency, exact technical terms, empty store
  test_dense_retriever.py    # unit: cosine scoring, missing embeddings, model-load failure
  test_rrf_fusion.py         # unit: standard formula, dedup by id, absent-list contribution
  test_abstention.py         # unit: per-pipeline semantics, returns [] vs preserves low-confidence hits
  test_retrieval_integration.py  # integration: full path incl. unanswerable, updated memories, duplicates
```

Existing files are **not modified** except: `ai-system/chunkdup/retrieval_class.py` (wire pluggable
stages behind defaults that preserve current behavior), `ai-system/chunkdup/in_memory_repository.py`
(add `search_semantic` so the dense leg exists for the in-memory store), and `retrieval/__init__.py`
(export new modules). `Memory` SDK signatures stay identical.

## 3. Dependencies & Models

| Dependency | Purpose | Notes |
|---|---|---|
| `torch` (CPU wheel) | required by sentence-transformers | installed into Windows Python 3.11; no `requirements.txt` change yet |
| `sentence-transformers` | dense leg + cross-encoder | loads **local** `models/all-MiniLM-L6-v2` (no download) |
| `rank-bm25` | BM25 leg | pure-Python, no service/infra dependency |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | reranking | one-time HF download (~90 MB); if download fails → config marked `model_unavailable`, results report it rather than silently falling back |

No hosted APIs, no paid inference, no new infra services.

## 4. Phase B — Dataset (before any new implementation is measured)

**Fixture.** Extend the 10-memory benchmark fixture with the project-fact memories the spec's
required queries reference (verified against the repo's actual facts):
`runtime_environment`="Docker" ("ChunkdUp runs inside Docker"), `embedding_model`="all-MiniLM-L6-v2",
`database`="PostgreSQL with pgvector" (already present as m9), `programming_language`="Python"
(project itself). Fixturized once in `fixture.py`; every config indexes the identical fixture in
identical order.

**Queries.** ~60 rows in `dataset.jsonl`, each:
`{id, query, relevant_memory_ids, answerable, expected_behavior (empty_result | results), category, explanation}`.
Categories and rough counts:

| Category | ~Count | Example |
|---|---|---|
| exact technical lookup | 8 | `"Neovim"`, `"pgvector"` |
| semantic paraphrase | 10 | `"What do I write code in?"` → Python |
| short / ambiguous | 6 | `"editor"`, `"database"` |
| indirect / contextual | 6 | `"How should you respond to me?"` |
| multi-memory synthesis | 8 | `"What is my tech stack?"` |
| updated / conflicting facts | 6 | store Python → update to Go; query must surface Go |
| project-specific | 8 | `"Where does ChunkdUp run?"`, `"What database does ChunkdUp use?"`, `"What technology does it use for vector search?"`, `"What embedding model does ChunkdUp use?"` |
| out-of-domain | 8 | `"What is the weather forecast today?"` → `[]` |
| absent answers | 8 | asks about facts never stored → `[]` |

**Labeling discipline.** Every row hand-checked against the fixture; expected answers never edited to
make a pipeline look better. Known failure queries from the spec are included verbatim. Split: ~45 dev
(for threshold calibration and debugging) / ~15 held-out (touched once, at final comparison).

## 5. Phase C — Component Design

### C1. `DenseRetriever`
- Embeds query and all fixture memories with local `all-MiniLM-L6-v2` (normalized), cosine similarity, returns top-`pool` with `_similarity`.
- Falls within the existing repository abstraction: exposed as `InMemoryRepository.search_semantic(query, limit, threshold)` so the existing retrieval path picks it up naturally; embeddings computed once at index time (mirrors `postgres_repository._embedding_text` representation).
- Empty store / missing embedding → returns `[]`; model-load failure raises a documented exception (no silent fallback).

### C2. `BM25Retriever`
- `rank_bm25.BM25Okapi` over `key + value + type + contextual_text`, one tokenizer shared by index and query (casefold, split on non-alphanumerics but keep `+`/`.`/`-` inside tokens so `all-MiniLM-L6-v2`, `C++`, `pgvector` survive).
- Returns `_bm25_score` (raw) + rank; indexes only `status == "active"` memories.

### C3. `fusion.py` — standard RRF
- `RRF(d) = Σ_lists 1/(k + rank_1based(d))` with configurable `k` (default 60) and per-leg candidate pool size; absent from a list → contributes nothing; dedup by stable `id`.
- The **existing** weighted/bonus hybrid engine stays available as a separate, clearly-labeled configuration (the prior session's "extras") so we can measure whether its type/intent/entity bonuses actually help, rather than deleting them on faith.

### C4. Cross-encoder reranking (wire the existing `PluggableReranker` mode)
- Pool bounded to top ~20 fused candidates; `_rerank_score` kept raw (documented as NOT a calibrated probability); latency measured separately from retrieval; model-load failure is **surfaced** (config marked unavailable) instead of silently degrading to heuristic.

### C5. `abstention.py` — relevance decision
- A distinct stage after ranking. Threshold is calibrated **per upstream score type** (cosine ∈ [-1,1], BM25 unbounded, RRF small positive, cross-encoder logits) on the **dev split only**.
- Optional gap signal (top1 − top2) included only if dev-set evidence supports it.
- Returns `[]` when nothing passes; otherwise returns everything above threshold (low-confidence-but-relevant memories preserved).
- Config flag lets the same pipeline run with/without abstention for a controlled comparison.

## 6. Phase D — Harness & Metrics

`evaluate.py` computes, per configuration, over the **same** dataset, fixture, order, and k=5:

- **MRR@5**, **Precision@5** (computed only over returned results — explicit convention), **Recall@5**, **HitRate@5** (≥1 relevant in top-5)
- For unanswerable queries: **no-answer rejection rate** (returned `[]` when expected), **false-positive rate** (returned memories when none expected), plus count of irrelevant memories returned
- For answerable queries: **false-negative/abstention rate** (returned `[]` when answers existed) and count of relevant memories surviving
- **Latency**: end-to-end p50/p95 + per-stage (retrieve / fuse / rerank / abstain) via `time.perf_counter`; peak RSS via `resource`/`psutil` if available, else omitted honestly
- Per-query JSON: retrieved ids/values, scores of every component, ranks, latency, expected, correctness flags

Eight named configurations (spec §6): `dense`, `bm25`, `hybrid_rrf`, `dense_ce`, `hybrid_ce`, `dense_abstain`, `hybrid_abstain`, `hybrid_ce_abstain` (+ `baseline_current` = today's exact path, kept as a labeled reference).

One result JSON per config in `experiments/retrieval/results/`; a summary table is generated, never hand-edited.

## 7. Phase E — Failure Analysis Protocol

For every config, persist traces (score components + threshold decision) for at minimum:
`"Where does ChunkdUp run?"`, `"What database does ChunkdUp use?"`, `"What technology does it use for vector search?"`, `"What is the weather forecast today?"`, plus one query per config where a relevant memory is retrieved but outranked by irrelevant ones (auto-detected from per-query output: RR < 1 while hit present).

Analysis rules: explain *why* from the traces; propose ranking changes only when a trace demonstrates the failure; rerun to confirm any change's effect.

## 8. Milestone Order & Verification Gates

| # | Milestone | Gate before proceeding |
|---|---|---|
| M1 | Environment fixed; existing tests + benchmark reproduce saved baseline | `pytest tests/` passes (SDK/entity tests); `retrieval_benchmarks.py` runs; numbers recorded in CONTEXT-CHECK |
| M2 | Dataset (60 rows) committed; every expected id verified against fixture | scripted check: every `relevant_memory_ids` exists in fixture; every category covered |
| M3 | Dense + BM25 retrievers + unit tests | unit tests pass; `dense` and `bm25` configs evaluated on dev split; strengths/weaknesses recorded |
| M4 | Standard RRF fusion + unit tests | `hybrid_rrf` evaluated; compared against dense-only, bm25-only, and legacy-bonus hybrid |
| M5 | Cross-encoder wired + bounded pool + latency | `dense_ce` / `hybrid_ce` evaluated; pre/post-rank comparison recorded |
| M6 | Abstention calibrated on dev only | `*_abstain` configs evaluated; FP/FN tradeoffs tabled; threshold values + curves recorded |
| M7 | Full comparison on dev **and** held-out; traces for required queries | results JSONs complete; docs/retrieval-experiments.md drafted |
| M8 | Integration tests; final report + decision | all tests pass; deliverables (spec §11) complete |

## 9. Decision Rule (final selection)

Simplest pipeline with a repeatable, meaningful improvement on the held-out set across quality
(MRR/P/R/HitRate), unanswerable-query rejection, latency, and dependency cost — judged with the small
sample size in mind (state uncertainty; no invented thresholds). If nothing beats
`baseline_current` convincingly, keep the baseline and say so with evidence.

## 10. Reproducible Commands (final form)

```bash
# Environment (one-time, current session)
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install sentence-transformers rank-bm25

# Baseline reproduction (frozen artifact)
python evaluation/retrieval_benchmarks.py

# One configuration
python experiments/retrieval/run_experiment.py --config hybrid_ce_abstain

# All configurations + summary table
python experiments/retrieval/run_experiment.py --all

# Threshold calibration (dev split only)
python experiments/retrieval/calibrate.py

# Tests
python -m pytest tests/ -v
```
