# Remaining Work — Retrieval Pipeline Task (ChunkdUp)

> Companion documents: [CONTEXT-CHECK.md](CONTEXT-CHECK.md) · [IMPLEMENTATION-PLAN.md](IMPLEMENTATION-PLAN.md)
> Status: round 1 (M1–M8) complete; round 2 (five-risk verification) complete; **round 3 complete** — decision memo, production hardening (fallback/logging/rollback/recalibration), entity experiments (exp 6 adopted with zero regressions, exp 7 rejected with evidence). Suite at end of round 3: **74 passed**. See retrieval-experiments.md §13 and round3-decision.md.

## M1 — Environment & Baseline Reproduction ✅

- [x] torch-CPU 2.14.1 / sentence-transformers 6.1.0 / rank-bm25 0.2.2 installed and import-verified
- [x] `pytest tests/` — 10/10 existing tests pass (`test_postgres.py` excluded: needs live DB)
- [x] `evaluation/retrieval_benchmarks.py` reproduces the saved baseline **byte-identically** (MRR 0.7857 / P 0.3821 / R 0.8571)
- [x] `docs/retrieval-baseline.md` written (architecture, execution flow, frozen metrics, known failures)

## M2 — Fixed Evaluation Dataset (Phase B) ✅

- [x] `experiments/retrieval/fixture.py`: 10 baseline memories + Docker runtime + embedding-model memories, grounded in repo facts, + 5 inactive historical memories
- [x] `experiments/retrieval/dataset.jsonl`: **70 rows** (56 dev / 14 heldout; 51 answerable / 19 unanswerable), all 9 categories, 6 required queries verbatim
- [x] `validate_dataset.py` passes (unique ids, expected ids exist and are active, category coverage, required queries)
- [x] Every relevance label hand-checked against the fixture

## M3 — Dense & BM25 Retrievers ✅

- [x] `DenseRetriever` (local MiniLM, normalized cosine, index-time embeddings, explicit failure on load error)
- [x] `BM25Retriever` (rank-bm25; tokenizer preserves `pgvector`, `all-MiniLM-L6-v2`, `C++`, `22.04`; stopword leakage found and fixed with a minimal documented stopword set)
- [x] Unit tests pass (23 in `tests/test_retrieval_components.py`)
- [x] Evaluated: BM25 = precision-oriented with natural rejection; dense = recall-oriented. Recorded in the experiments report

## M4 — Standard RRF Fusion ✅

- [x] `fusion.py`: standard formula, configurable k, id-based dedup, per-leg contribution in `_rrf_parts`
- [x] Legacy weighted/bonus engine kept as `baseline_current` reference — **measured, not deleted: it is the worst config on the new dataset (combined MRR 0.5238)**, evidence its boosts overfit the old 7-query benchmark
- [x] Unit tests for formula/dedup/absent-list behavior
- [x] `hybrid_rrf` evaluated: marginal gains over dense alone (+0.012 MRR combined)

## M5 — Cross-Encoder Reranking ✅

- [x] Model downloaded and load-failure made explicit (no silent heuristic fallback in experiment configs)
- [x] Bounded pool (20) + separate `rerank_ms` latency (~250 ms CPU for 20 pairs)
- [x] `dense_ce` / `hybrid_ce` evaluated: no gain without abstention; identical to each other (pool-saturation artifact, documented)

## M6 — Relevance Filtering / Abstention ✅

- [x] `abstention.py`: post-ranking gate, per-pipeline score semantics, optional gap veto (disabled by default, unit-tested anyway)
- [x] `calibrate.py`: dev-only thresholds — cosine 0.2625 / RRF 0.023969 / CE logit −10.947 (+ sweep curves in `results/calibration.json`)
- [x] `*_abstain` configs evaluated with FP/FN accounting; **finding: abstaining on raw RRF scores is unreliable (unstable scale, 21 false abstains — rejected)**

## M7 — Shared Harness Run & Failure Analysis ✅

- [x] Full metric set (MRR/P/R/HitRate, rejection, FP-memories, false-abstains, per-stage latency) in `evaluate.py`
- [x] All 9 configs on dev **and** held-out; per-config JSONs in `experiments/retrieval/results/`
- [x] Traces for q45/q46/q47/q53 + remaining outranked cases (q18/q38/q46) with all score components; unavailable components marked null
- [x] Combined table machine-generated in `docs/retrieval-experiments.md`

## M8 — Tests, Report, Decision ✅

- [x] 11 integration tests (`tests/test_retrieval_integration.py`): answerable/unanswerable/empty-store/inactive-memory/duplicate-candidates/updated-memory/baseline + CE end-to-end; **full suite 44 passed**
- [x] `docs/retrieval-experiments.md`: hypotheses, environment, configs, metric definitions, comparison tables, per-query analysis, latency, limitations, decision
- [x] Final question answered: **hybrid_ce_abstain selected (MRR 0.860 / HitRate 0.886 / rejection 0.895); dense_abstain documented fallback (~10 ms, best precision); rejected configs listed with reasons**
- [x] Spec §11 deliverables complete (architecture summary, files+reasons, repro commands, test outcomes, comparison table, per-query analysis, decision, limitations + next experiment)

## Follow-up Work (post-task, recommended — not started)

- [x] ~~Re-run with a 100+ memory fixture~~ **DONE in round 2 (risk 1)**: scales 100/500/1000 tested; hybrid_ce_abstain confirmed at scale; scale-aware gate recalibration identified as an operational requirement (`results/scale/`)
- [x] ~~Held-out flip at n=14~~ **RESOLVED in round 2 (risk 2)**: persists at n=36 with an explicit margin (+0.039 MRR to dense_abstain); combined-92 still favors hybrid_ce_abstain (`results/round2.*.json`)
- [x] ~~**Experiment 6 — type-aware entity scoring**~~ **DONE in round 3, ADOPTED for the experiment configs** (`6efd36c`): sink-below demotion of project_name memories matching attribute-seeking entity queries purely on the entity token; pooled-92 MRR/P improved on both leading configs, per-query 7 better / **0 worse**, abstention unchanged
- [x] ~~**Experiment 7 — CE entity-guard**~~ **TESTED in round 3, REJECTED with evidence**: fixes q46/q87 but breaks q45 (1.00→0.50) and costs precision broadly (dev P −0.042); a future variant must distinguish jargon miscalibration from legitimate CE correction
- [ ] **Product decision — latency vs quality** (round-2 §12.6, unchanged by round 3): hybrid_ce_abstain (~200–450 ms, fewer false abstentions) vs dense_abstain (~10 ms, 16–18% false abstention, and now stronger with exp 6) — maintainer's call
- [x] ~~**Operational — scale-aware gate recalibration**~~ **BUILT in round 3** (`26d96e3`): `ProductionRetriever.recalibrate(sample)` + `should_recalibrate()` (~50% store-growth trigger); README documents the schedule tied to round-2 drift data
- [ ] Wire experiment 6's demotion into `ProductionRetriever` (validated but experiment-scoped this round)
- [ ] Domain-tuned or alternative reranker (e.g. bge-reranker) vs ms-marco-MiniLM-L-12-v2 on this harness
- [ ] Query-type / answerability detection for historically-phrased questions ("Did I ever work at X?")
- [ ] Pin new dependencies in `requirements.txt` (deliberately deferred — decision belongs to the maintainer)
- [ ] Cache invalidation on memory writes (pre-existing production gap, out of task scope)

## Standing Constraints — compliance notes

- Public `Memory` SDK API untouched; `evaluation/` frozen artifacts untouched; no hosted APIs; no tuning on held-out; no expected-label edits; all failures reported rather than papered over.
