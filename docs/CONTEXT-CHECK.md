# Context Check — Retrieval Pipeline Task (ChunkdUp)

> Companion documents: [IMPLEMENTATION-PLAN.md](IMPLEMENTATION-PLAN.md) · [REMAINING-WORK.md](REMAINING-WORK.md)
> Task source: "Build and Evaluate 5 Semantic Retrieval Pipelines in ChunkdUp"
> Last updated: 2026-10-12

This document records, step by step, what has been done on the retrieval-pipeline task so far —
by prior sessions (reconstructed from git history and committed artifacts) and by the current
session — together with every result obtained and the evidence for each claim.

---

## 0. Environment Record

| Item | State at start of current session |
|---|---|
| Repo / branch | `D:\Development\Projects\core\ChunkdUp`, branch `main`, working tree clean |
| Windows Python | 3.11.9 — had `pytest 8.4.2`, `numpy 2.3.4`, `SQLAlchemy 2.0.44`, `python-dotenv`; **no** torch / sentence-transformers / rank-bm25 / psycopg2 |
| WSL venv (`.venv/`) | Python 3.12.3, **empty** (only `pip 24.0`). Prior sessions' environment is gone. |
| `wheels/` | Orphaned manylinux **cp311** wheels (`numpy 1.26.4`, `tokenizers 0.19.1`, `transformers 4.40.2`) — built for a cp311 Linux env that no longer matches the 3.12 venv (see commit `575fdc7 "fix: isolate dependency mismatch"`) |
| Local models | `models/all-MiniLM-L6-v2/` — full local copy of the embedding model (safetensors + ONNX variants). No cross-encoder model cached. |
| Network | Verified working (PyPI reachable) on 2026-10-12 |
| Current session action | Started install of `torch` (CPU wheel) + `sentence-transformers` + `rank-bm25` into Windows Python 3.11. Reason: evaluations cannot run otherwise. This does **not** modify `requirements.txt`. |

---

## Phase A — Inspect and Freeze the Baseline

### What was done (prior sessions, reconstructed from git)

| Commit | What it introduced |
|---|---|
| `5a035ec` | Semantic search feature, RRF, hybrid search mechanism |
| `96bde2d` | Contextual indexing, hybrid search + RRF fusion, adaptive-k selector, TTL query caching, query rewriter |
| `61a7128` | Type classification moved before RRF, adaptive-k relaxed |
| `5290ad9` | "Reranking issue" fix (heuristic reranker feature weighting) |
| `df5e6e3` | Trailing-space fixes, type-boost tuning for specific fields |
| `6f05a2f` | Entity extraction technique (`entity_extractor.py`, entity-overlap candidate generation) |

### What exists today (verified by reading the code)

**Execution path** — `Memory.retrieve()` (`ai-system/chunkdup/memory.py:625`) →
`MemoryRetriever.retrieve()` (`ai-system/chunkdup/retrieval_class.py:48`):

1. **TTL cache** lookup (`retrieval/cache.py`, ttl=300s, max 1000 entries).
2. **Exact-lookup fast path**: queries of ≤ 2 words go straight to `repository.get_by_key()` and return a single result, bypassing ranking entirely.
3. **Candidate pool size**: `AdaptiveKSelector.select_candidate_k()` = `max(10, 2k)`.
4. **Query rewriting** (`retrieval/query_rewriter.py`): filler removal (suppressed for preference/career queries), abbreviation expansion, plus **hardcoded expansions** for synthesis ("tech stack" → a fixed 10-token string) and preference queries.
5. **Candidate generation** (two legs):
   - *Keyword leg*: `repository.search()` — for `InMemoryRepository` this is **substring matching** (`word in combined_text`), not BM25. Plus **entity-overlap candidates** appended to the keyword list (`_entity_score` = Jaccard-style overlap of casefolded entity sets).
   - *Semantic leg*: `repository.search_semantic()` — exists **only** on `PostgresRepository` (pgvector cosine). `InMemoryRepository` has no such method → the semantic leg is **always empty** in the benchmark.
   - *Fallback*: if both legs are empty, a token-overlap scan over `get_all()`.
6. **Threshold filters**: keyword candidates filtered by `_score >= 0.15`; the semantic filter is `_similarity > -1.0` — **a no-op** (accepts everything).
7. **RRF fusion** (`retrieval/hybrid_search.py`): weighted RRF (`keyword_weight=0.4`, `semantic_weight=0.6`, k=60) **multiplied** by a static `type_boost` dict and a **dynamic intent boost** (hand-written keyword rules for career/preference/editor queries), **plus** an additive entity bonus (`entity_weight=0.5 × _entity_score`). This deviates from standard RRF.
8. **Reranking** (`retrieval/reranker.py`): three modes — `heuristic` (feature-weighted: 0.75·semantic + 0.15·lexical + 0.10·hybrid, × type weights), `llm` (**a simulation** — token overlap added to `_score`; no LLM), `cross_encoder` (real `cross-encoder/ms-marco-MiniLM-L-12-v2` via sentence-transformers, **falls back silently to heuristic** on load failure).
9. **Adaptive k**: expands k for "synthesis" keyword queries; if `max_score < 0.01` returns 1 result (not 0).
10. **Cache store**, return.

**Files responsible** (all verified):
- Retrieval: `ai-system/chunkdup/retrieval_class.py`, `retrieval/{hybrid_search,reranker,adaptive_k,cache,contextual,query_rewriter}.py`
- Storage: `in_memory_repository.py`, `postgres_repository.py` (pgvector + embeddings), `repository.py` (ABC)
- Embeddings: `embeddings.py` (SentenceTransformer, normalized vectors, model from `CHUNKDUP_EMBEDDING_MODEL` env or `all-MiniLM-L6-v2`)
- Entities: `entity_extractor.py` (deterministic domain vocab + optional spaCy NER gated by `CHUNKDUP_ENABLE_SPACY_NER=1`)
- Evaluation: `evaluation/retrieval_benchmarks.py` (retrieval benchmark), `evaluation/runner.py` + `evaluation/scenarios.json` (54 policy-decision scenarios), `evaluation/retrieval_results.json` (saved results)
- Tests: `tests/test_sdk.py`, `tests/test_entity_retrieval.py`, `tests/test_postgres.py` are real runnable tests; `tests/test_ranking.py`, `tests/test_semantic_scope.py`, `tests/test_semantic_similarity.py` are **data files only** (no assertions, never executed by pytest).

### Baseline results obtained (saved artifact)

`evaluation/retrieval_results.json` — 7-query benchmark, fixture of 10 memories:

| Reranker mode | MRR | Mean P@K | Mean R@K |
|---|---|---|---|
| `heuristic` | **0.7857** | **0.3821** | **0.8571** |
| `llm` (simulated) | **0.7857** | **0.3821** | **0.8571** |

Per-query facts from the saved details:
- `"What is the weather forecast today?"` → returns `["Python"]` (heuristic) / 5 memories (llm mode) instead of `[]` → **known false positive on out-of-domain queries**.
- `"What programming languages do I use?"` → precision 0.25 (8 retrieved, 2 relevant).
- `"What editor or IDE do I prefer?"` → precision 0.2; `"How should you respond to me?"` → precision 0.2.
- `"Where do I work and what is my role?"` → RR 0.5 (an irrelevant memory outranks one relevant memory).
- `"Python"` (exact lookup) → perfect via fast path.

Historical milestones recorded in `learnings/current_imp.md` (self-reported, not independently re-verified this session):
- Initial: MRR 0.4571 / P 0.3000 / R 0.4571 → after threshold+RRF+adaptive-k fixes: 0.5714 / 0.3929 / 0.4571 → after intent boosts + fast path + rewriter expansion: claimed 0.8333 / 0.5250 / 1.0000.
- Policy decision runner: **54/54 scenarios, 100% decision and value accuracy** (`evaluation/runner.py` + `evaluation/scenarios.json`).

### Known limitations identified during inspection

1. **Dense retrieval is not exercised at all** in the current benchmark (no `search_semantic` on the in-memory repo). "Baseline" is keyword + entity + reranking, not dense.
2. **No BM25** anywhere; keyword leg is naive substring matching (substring semantics can over/under-match, e.g. `"use"` matches `"Concise and technical"`... via contextual text).
3. **RRF deviates from the standard formula**: multiplied type/intent boosts and additive entity bonus are hardcoded without experimental support (the task spec explicitly forbids unscored bonuses).
4. **The `llm` reranker mode is not an LLM** — it is a token-overlap heuristic; both saved modes produce identical metrics because heuristic + llm collapse to nearly the same ordering.
5. **Cross-encoder mode exists but has never been evaluated** in any saved result; it also silently degrades to heuristic on load failure, which would make an evaluation of it meaningless unless verified.
6. **Abstention is partial**: the 0.15 threshold applies to keyword `_score` only (a substring-match count, not a calibrated relevance score); the semantic filter is a no-op; adaptive-k floors at 1 result; the weather query still leaks a result.
7. **Hardcoded query-domain logic** (rewriter expansions, intent boosts, adaptive-k "ambiguous keywords") is tuned to the 7 benchmark queries — overfitting risk that a larger dataset must expose.
8. **Fast-path exact lookup** returns a single memory with no competition — it masks precision problems on short queries.
9. **Cache** returns stale results if memories change within TTL (no invalidation on writes).
10. **Test coverage gaps**: the three "semantic" test files contain expectations but no executable tests; required queries like "Where does ChunkdUp run?" reference memories (`runtime_environment`/Docker, `embedding_model`/all-MiniLM-L6-v2) that are **not in the benchmark fixture**.

### Baseline freeze status

- The baseline code path and its saved metrics are frozen as-is: `evaluation/retrieval_benchmarks.py` and `evaluation/retrieval_results.json` stay untouched as the historical artifact.
- Current-session reproduction of the baseline (running the benchmark in the fixed environment) — **pending**, blocked on the package install; recorded in REMAINING-WORK.md.

---

## Phase B — Fixed Evaluation Dataset

**Not started** (by prior sessions or this one). The current benchmark has 7 hardcoded queries; the spec requires ~60 manually reviewed, version-controlled queries with stable IDs, categories, answerability flags, dev/held-out split, and a fixture extended to support the required known-failure queries.

## Phase C — Five Approaches

| Approach | Status |
|---|---|
| 1. Dense retrieval | **Missing** for the in-memory store (needs a real dense leg + cosine scoring). Postgres/pgvector path exists but is not part of the benchmark. |
| 2. BM25 | **Missing.** |
| 3. Hybrid + RRF | **Partially exists** (`HybridSearchEngine`) but non-standard (bonus terms); needs standard-RRF mode + configurability. |
| 4. Cross-encoder reranking | **Exists but unevaluated**; needs bounded candidate pool, verified model load, separate latency, failure handling. |
| 5. Relevance filtering / abstention | **Partial**: threshold on the wrong score; no per-pipeline calibration; weather false positive still present in saved results. |

## Phase D — Shared Evaluation Harness

**Missing.** Current benchmark covers 2 modes × 7 queries × 3 metrics. The spec's 8 configurations, Hit-Rate@K, abstention FP/FN accounting, per-stage latency, and per-query machine-readable output do not exist.

## Phase E — Failure / Score-Component Inspection

**Partial.** `MemoryRetriever.last_diagnostics` and `HybridSearchEngine.last_diagnostics` already carry per-candidate keyword/semantic/entity/rrf/final components. `learnings/current_imp.md` documents a manual root-cause pass (threshold, boost timing, adaptive-k, rewriter). No persisted traces for the spec's required queries; no re-rank-vs-pre-rank comparison.

---

## Session Log (current session, appended as work proceeds)

| # | Step | Result |
|---|---|---|
| 1 | Full repo inspection (code, git history, saved results, docs) | Architecture and gaps recorded above; baseline artifacts identified and frozen |
| 2 | Environment verification | WSL venv empty; Windows Python lacked ML stack; network OK; local MiniLM model present |
| 3 | Package install (torch-CPU 2.14.1, sentence-transformers 6.1.0, rank-bm25 0.2.2) | **Done, exit 0.** `requirements.txt` untouched; cross-encoder `ms-marco-MiniLM-L-12-v2` downloaded and smoke-tested (relevant pair +7.6, irrelevant −11.1) |
| 4 | Baseline reproduction + existing tests | **Baseline reproduced byte-identically** (MRR 0.7857 / P 0.3821 / R 0.8571, both modes); 10/10 existing tests pass (postgres excluded — needs live DB) |
| 5 | Dataset build (`experiments/retrieval/fixture.py` + `dataset.jsonl` + validator) | 70 queries (56 dev / 14 heldout; 51 answerable / 19 unanswerable), all 9 categories, all 6 required queries verbatim; validator passes |
| 6 | Components: `retrievers.py` (dense+BM25), `fusion.py` (standard RRF), `abstention.py`, `config.py`, `evaluate.py`, `run_experiment.py` | All implemented; 23 unit tests pass. Finding: BM25 stopwords leaked matches → minimal stopword set added, documented |
| 7 | Threshold calibration (`calibrate.py`, dev split only) | cosine ≥ 0.2625 / RRF ≥ 0.023969 / CE logit ≥ −10.947; objective (hit+reject)/2 = 0.845 / 0.799 / 0.913; saved to `results/calibration.json` |
| 8 | All 9 configs × dev + heldout | `hybrid_ce_abstain` best combined (MRR 0.8595, HitRate 0.8857, reject 0.895); `dense_abstain` best precision (0.696) at ~10 ms; `baseline_current` worst (0.5238 — overfit to old 7-query set). `dense_ce` ≡ `hybrid_ce` (pool saturation artifact). Results in `experiments/retrieval/results/*.json` |
| 9 | Failure/score-component analysis (Phase E) | Traces extracted for q45/q46/q47/q53 + remaining outranked cases; CE jargon failure on q46 verified against raw model (−8.06 on correct memory in every text format); baseline overfit pattern quantified |
| 10 | Integration tests | 11 end-to-end tests pass (answerable/unanswerable/dedup/inactive/empty-store/updated-memory/baseline-CE paths); **full suite 44 passed** |
| 11 | Reports | `docs/retrieval-baseline.md` + `docs/retrieval-experiments.md` written; decision: **hybrid_ce_abstain selected, dense_abstain documented fallback** |
| 12 | R2 | Round-2 checkpoint commits: C1 round-1 state → C2 scale-test code → C3 scale results → C4 held-out expansion → C5 CE jargon audit → C6 FP/false-abstention analysis → C7 doc decision update |
| 13 | R2.1 scale test (12/100/500/1000) | hybrid_ce_abstain leads at every scale, gap widens (+0.074 → +0.14 MRR over dense_abstain); raw retrieval collapses (dense 0.657→0.143); frozen CE threshold does NOT transfer (rej 0.895→0.737; recalibrated hits 1.0 at −2.13); latency isolated via bench: dense flat ~8-10ms, BM25 0.9ms@1000, CE pool-bounded ~200-450ms. Scale-12 run reproduces round-1 exactly |
| 14 | R2.2 held-out expansion (14→36) | Flip PERSISTS: dense_abstain 0.819 vs CE 0.780 MRR (+0.039 ≈ 1.5 queries), rejection 0.909 vs 0.818; disagreement map: dense wins 5 queries, CE wins 2; q16 shows CE gate filtering the correct memory |
| 15 | R2.3 CE jargon audit | 2/15 (13.3%) technical-term queries show CE rank inversion (q46 + q87, both on the pgvector memory — clusters on jargon); strict anomaly pattern 1/15; q87 passes the frozen gate by 0.10 logit |
| 16 | R2.4 false-abstention accounting | hybrid_ce_abstain 0.068–0.12 false-abstain WITH 0.82–0.92 rejection (best gate profile); dense_abstain false-abstains 0.16–0.18; bm25 0.24–0.27; no-gate configs 0/0 |
| 17 | R2.5 FP pattern analysis | CONCENTRATED: project_specific queries produce FPs in every config (12/69 queries, 35 FP memories in dense_abstain and bm25) — 'ChunkdUp' token pulls the project memory into attribute-seeking answers; 6th experiment named: type-aware entity scoring |
| 18 | R2.6 latency tradeoff | Written as a product decision in retrieval-experiments.md §12.6; decision left to the maintainer |
| 19 | R2 docs | retrieval-experiments.md §12 added (revised decision referencing all 5 risks); REMAINING-WORK follow-ups updated; 7 granular commits |
| 20 | R3 P1 decision memo | `docs/round3-decision.md`: hybrid_ce_abstain ships as default — deciding factor = risk-4 false abstention (0.068–0.12 vs 0.16–0.18), NOT the held-out flip; fast-path rule is latency/deployment-only (scale data shows no size where the CE gap is negligible: +0.074 MRR at 12 memories is the smallest); recalibration declared a hard shipping dependency. Commit `37d0f45` |
| 21 | R3 P2.1–P2.4 production hardening | New `retrieval_production.py` wired into Memory SDK: CE fallback chain w/ 1.0s timeout (2× p95 + headroom) → dense gate → frozen baseline, all logged + sticky (`6869c61`); JSONL query logging w/ full contract tests (`4e3c497`); CHUNKDUP_RETRIEVAL_CONFIG rollback + README incident docs (`2146662`); scale-aware recalibrate()/should_recalibrate() — the memo's hard dependency delivered (`9f9524f`). Suite 50→54→57→62 passing |
| 22 | R3 P3.1 entity-focus slice | `dataset_entity_focus.jsonl`: 15 hand-verified queries (8 project-name+attribute, 7 jargon incl. 2 must-abstain traps + e12 rule-safety probe); named splits in load_dataset; merged integrity check 107 rows. Commit `93a188a` |
| 23 | R3 P3.2 experiment 6 (type-aware entity demotion) | ADOPTED, zero regressions: sink-below semantics (first ×0.25 design was structurally ineffective on CE logits — observed, replaced); hybrid_ce_abstain_ed pooled-92 MRR 0.8397→0.8424, P 0.6366→0.6475, 2 better/0 worse; dense_abstain_ed pooled 0.7880→0.8152, 5 better/0 worse; slice MRR +0.072/+0.122, P +0.105 both |
| 24 | R3 P3.3 experiment 7 (CE entity-guard) | REJECTED with evidence: fixes q46 (0.33→0.50) + q87 (0.25→0.50) but breaks q45 (1.00→0.50) and costs precision broadly (dev P −0.042, slice P −0.039); exp 6 captures most of the benefit at no cost |
| 25 | R3 P3.4 docs | retrieval-experiments.md §13 + §14 updated; round3-decision.md recalibration status marked DELIVERED (commit `9f9524f`); REMAINING-WORK updated; suite at end of round: 74 passed |
| 26 | Closeout P1 | Exp-6 demotion wired into `retrieval_production.py` + pool aligned to validated 20-candidate config; production path reproduces round-3 numbers EXACTLY on 92 queries (0.8424/0.6475 with demotion, 0.8397/0.6366 without — both MATCH via verify_production.py). Caught on the way: pool-size transfer bug (10 vs 20) + fixture-status artifact in the harness. Commit `be31673`, 77 tests |
| 27 | Closeout P2 | CE timeout re-derived from the named round-2 ISOLATED benchmark (worst median 452.6ms, scale 500; 2× → 1.0s confirmed); scale-1000 steady-state test added (no fallback fires). Commit `bf00613`, 78 tests |
| 28 | Closeout P3 | `fast_mode=True` explicit opt-in → dense_abstain_ed (demotion included), documented as a planned choice distinct from failure degradation; precedence documented; README accuracy fixes. Commits `2b917e7` + `5feef2d` (de-flaked wall-clock bound), 82 tests stable ×2 |
| 29 | Closeout P4 | retrieval-experiments.md §15 final arc summary; tracker closed with "future work, not started" section; README verified accurate; final suite **82 passed**; closing commit |
