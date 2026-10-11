# Retrieval Experiments Report (Phases B–E)

> Companion: [retrieval-baseline.md](retrieval-baseline.md) · [CONTEXT-CHECK.md](CONTEXT-CHECK.md)
> All numbers below are machine-generated (`experiments/retrieval/results/*.json`), never hand-edited.
> Runs: 2026-10-12, Windows Python 3.11.9, torch 2.14.1+cpu, sentence-transformers 6.1.0, rank-bm25 0.2.2.

## 1. Hypotheses Under Test

1. **H1 (from the task brief):** candidate discovery is not the main problem; too many irrelevant memories are returned alongside relevant ones. Relevance filtering / abstention should add more value than more elaborate retrieval.
2. **H2:** BM25 improves exact technical lookups over dense retrieval.
3. **H3:** hybrid RRF improves recall over either single retriever.
4. **H4:** cross-encoder reranking improves the rank position of relevant memories enough to justify its latency.
5. **H5:** a relevance threshold must be calibrated per score type (cosine vs BM25 vs RRF vs cross-encoder logits are not comparable).

## 2. Dataset and Protocol

- **Fixture:** 12 active memories (the original 10 baseline memories plus `m11` Docker runtime and `m12` all-MiniLM-L6-v2 embedding model, grounded in repo facts) + 5 inactive historical memories for the updated-facts category. Identical fixture, order, and settings for every configuration.
- **Queries:** 70 manually reviewed rows in `experiments/retrieval/dataset.jsonl` (validated by `validate_dataset.py`): 51 answerable / 19 unanswerable; categories: exact_lookup (8), paraphrase (11), short_ambiguous (6), indirect_contextual (6), multi_memory (9), updated_facts (6), project_specific (8), out_of_domain (8), absent_answer (8). All six required known-failure queries included verbatim.
- **Splits:** dev = 56 (threshold calibration, debugging), held-out = 14 (touched once, for final comparison). Calibration used **only** the dev split.
- **k = 5** everywhere.

## 3. Metric Conventions (explicit)

- **Precision@5** is computed only over returned results: a correct `[]` on an unanswerable query scores 1.0; returning junk scores 0.0.
- **Recall@5**: |relevant ∩ returned| / |relevant|; unanswerable queries score 1.0 iff `[]` returned.
- **HitRate@5**: ≥1 relevant memory in top-5 (unanswerable: a correct rejection counts).
- **MRR**: 1/rank of the first relevant memory (correct rejection = 1.0).
- Abstention accounting reported separately: false-abstain count on answerable queries, rejection rate / false-positive memories on unanswerable queries — so correct rejections never hide inside aggregate precision.

## 4. Configurations

`baseline_current` (frozen production path: substring keyword + entity candidates + weighted RRF with type/intent/entity bonuses + heuristic reranker + adaptive k) plus the 8 spec configurations built from new components (`experiments/retrieval/`): `dense`, `bm25`, `hybrid_rrf` (standard RRF, k=60), `dense_ce`, `hybrid_ce` (cross-encoder `ms-marco-MiniLM-L-12-v2` over a bounded 20-candidate pool), `dense_abstain` (cosine gate), `hybrid_abstain` (RRF gate), `hybrid_ce_abstain` (cross-encoder-logit gate).

Calibrated dev-split thresholds (`results/calibration.json`): cosine ≥ **0.2625**; RRF ≥ **0.023969**; cross-encoder logit ≥ **−10.947**. No shared numeric threshold anywhere; each gate understands its own score semantics (H5).

## 5. Results

### 5.1 Dev split (56 queries)

| Config | MRR | P@5 | R@5 | HitRate | Reject (unans.) |
|---|---|---|---|---|---|
| baseline_current | 0.5655 | 0.4643 | 0.5932 | 0.6250 | 0.3333 |
| dense | 0.7054 | 0.1964 | 0.7503 | 0.7679 | 0.0000 |
| bm25 | 0.7143 | 0.6827 | 0.6792 | 0.7321 | **0.9167** |
| hybrid_rrf | 0.7292 | 0.2036 | 0.7622 | 0.7857 | 0.0000 |
| dense_ce | 0.7083 | 0.1893 | 0.7140 | 0.7500 | 0.0000 |
| hybrid_ce | 0.7083 | 0.1893 | 0.7140 | 0.7500 | 0.0000 |
| dense_abstain | 0.7679 | **0.7068** | 0.7798 | 0.8036 | **0.9167** |
| hybrid_abstain | 0.7054 | 0.6827 | 0.6792 | 0.7321 | **0.9167** |
| hybrid_ce_abstain | **0.8780** | 0.6351 | **0.8622** | **0.9107** | **0.9167** |

### 5.2 Held-out split (14 queries — small; differences ≈ 1 query are noise)

| Config | MRR | P@5 | R@5 | HitRate | Reject |
|---|---|---|---|---|---|
| baseline_current | 0.3571 | 0.3571 | 0.3214 | 0.3571 | 0.1429 |
| dense | 0.4643 | 0.1143 | 0.5000 | 0.5000 | 0.0000 |
| bm25 | 0.5714 | 0.5143 | 0.5357 | 0.5714 | 0.7143 |
| hybrid_rrf | 0.4286 | 0.1143 | 0.5000 | 0.5000 | 0.0000 |
| dense_ce | 0.3929 | 0.1000 | 0.4286 | 0.4286 | 0.0000 |
| hybrid_ce | 0.3929 | 0.1000 | 0.4286 | 0.4286 | 0.0000 |
| dense_abstain | **0.8571** | 0.6536 | **0.8571** | **0.8571** | 0.8571 |
| hybrid_abstain | 0.6071 | 0.5286 | 0.6071 | 0.6429 | 0.7143 |
| hybrid_ce_abstain | 0.7857 | **0.7286** | 0.7857 | 0.7857 | 0.8571 |

### 5.3 Combined (70 queries)

| Config | MRR | P@5 | R@5 | HitRate | Reject | FP memories (unans.) | False abstains (ans.) |
|---|---|---|---|---|---|---|---|
| baseline_current | 0.5238 | 0.4429 | 0.5388 | 0.5714 | 0.2632 | 18 | 1 |
| dense | 0.6572 | 0.1800 | 0.7002 | 0.7143 | 0.0000 | 95 | 0 |
| bm25 | 0.6857 | 0.6490 | 0.6505 | 0.7000 | 0.8421 | 3 | 21 |
| hybrid_rrf | 0.6691 | 0.1857 | 0.7098 | 0.7286 | 0.0000 | 95 | 0 |
| dense_ce | 0.6452 | 0.1714 | 0.6569 | 0.6857 | 0.0000 | 95 | 0 |
| hybrid_ce | 0.6452 | 0.1714 | 0.6569 | 0.6857 | 0.0000 | 95 | 0 |
| dense_abstain | 0.7857 | **0.6962** | 0.7953 | 0.8143 | **0.8947** | 2 | 12 |
| hybrid_abstain | 0.6857 | 0.6519 | 0.6648 | 0.7143 | 0.8421 | 3 | 21 |
| hybrid_ce_abstain | **0.8595** | 0.6538 | **0.8469** | **0.8857** | **0.8947** | 2 | **6** |

### 5.4 Latency (dev means; single CPU, warm)

| Config | Total ms | Breakdown |
|---|---|---|
| baseline_current | 0.77 | substring + heuristic (no model) |
| bm25 | 0.04 | BM25 only |
| dense / dense_abstain | ~10 | query embedding ~10 ms |
| hybrid_rrf / hybrid_abstain | ~10.5 | dense 10 + BM25 0.1 + fuse 0.1 |
| dense_ce / hybrid_ce / hybrid_ce_abstain | ~260–268 | **rerank ≈ 250 ms** for 20 pairs on CPU |

Memory/CPU: dense adds one 90 MB embedding model (already required by the SDK); cross-encoder adds a second ~90 MB model. Peak-RAM instrumentation was not feasible on this Windows environment and is reported as unavailable rather than estimated.

## 6. Hypothesis Verdicts (evidence-backed)

- **H1 — CONFIRMED, and it is the dominant effect.** Adding the calibrated gate to dense moves MRR 0.657 → 0.786, precision 0.180 → 0.696, rejection 0.000 → 0.895, false-positive memories on unanswerable queries 95 → 2. No retrieval-side change comes close.
- **H2 — PARTIALLY.** BM25 wins precision among retrievers (0.649) and rejects unanswerable queries naturally (0.842; post-stopword cleanup), but its recall ceiling (0.651) confirms it misses paraphrases; exact-lookup queries (`pgvector`, `all-MiniLM-L6-v2`, `C++`) work because the tokenizer preserves technical tokens.
- **H3 — MARGINAL, with a serious caveat.** hybrid_rrf over dense-only: MRR +0.012, recall +0.010 (combined) — small. Worse, **abstaining on raw RRF scores is unreliable** (hybrid_abstain: 21 false abstains, worst of all configs): when one leg returns nothing, fused scores halve, so a single calibrated threshold misclassifies. RRF's score scale is not stable across queries.
- **H4 — MIXED.** The cross-encoder alone does *not* beat dense (0.645 vs 0.657 MRR) and costs 25× latency. But as a *relevance signal for the gate* it is the best available: hybrid_ce_abstain has the fewest false abstentions (6 vs 12) and the best MRR/recall/hit-rate on the combined set. Note `dense_ce` ≡ `hybrid_ce` exactly: with 12 active memories and pool 20, both rerank the identical full pool — candidate generation cannot matter below pool saturation.
- **H5 — CONFIRMED.** The three calibrated thresholds differ by ~4 orders of magnitude (0.26 cosine / 0.024 RRF / −10.9 logit); sharing any single number would have been meaningless.

## 7. Per-Query Error Analysis (traces in `results/*.json`)

### Required queries

- **`"Where does ChunkdUp run?"` (q45, expected m11 Docker)** — dense 0.537 and BM25 2.06 both rank the project memory m3 "ChunkdUp" first (the query text literally contains the project name), Docker second → RR 0.5. The cross-encoder fixes it: Docker 1.89 vs ChunkdUp −0.31 → **rank 1**.
- **`"What database does ChunkdUp use?"` (q46, expected m9)** — BM25 ranks m9 first (2.86); dense ranks m3 first; **the cross-encoder scores m9 at −8.06** (below Docker 0.61 and ChunkdUp 0.58), demoting the correct answer to rank 3. Verified against the raw model: "PostgreSQL with pgvector" scores negative against this query in *every* text format (value-only −9.8, key+value −5.7) — database jargon is out-of-distribution for an MS-MARCO-trained reranker. This is a genuine model limitation, not an input-formatting bug.
- **`"What technology does it use for vector search?"` (q47, expected m9)** — BM25 returns only m12 (its contextual text contains "vector search"; m9's "pgvector" is a different token, and its contextual text says "backend storage infrastructure"). Dense gets m9 second (0.285 vs m12 0.367) → RR 0.5; hybrid_ce_abstain's gate filters m9 (CE < −10.95) → RR 0. Synergy failure: neither leg alone is sufficient.
- **`"What is the weather forecast today?"` (q53, must return `[]`)** — the frozen baseline returned `["Python"]`. Every gated config now returns `[]`. The fix is a calibrated threshold on score semantics, **not** a weather-specific rule: `"Tell me a joke."`, `"When is my birthday?"`, `"sourdough bread"` all reject the same way, and BM25 rejects them with no threshold at all.

### Relevant-but-outranked cases (remaining, winner config)

- q18 "What container technology do I run projects in?" → m3 outranks m11 (RR 0.5) — "projects" lexically attracts the project memory.
- q38 "What tools do I use to write and run code?" → m9/m12 outrank the languages (RR 0.33) — the CE overrates tool/jargon memories.
- q46 above (RR 0.33).

### Baseline failure pattern (why `baseline_current` scores worst)

The production path leaks on 8 of 12 dev unanswerable queries ("Tell me a joke.", "What is my favorite programming book?"...), has 6 rank-below-irrelevant cases on dev answerable queries, and misses across *every* category. Its type/intent boosts and rewriter expansions were tuned on the original 7-query benchmark (where it scores MRR 0.7857) and do not transfer to 70 queries (0.5238 combined) — a textbook overfit, and the reason the task demanded a larger fixed dataset.

## 8. Decision

**Selected pipeline for the personal-memory use case: `hybrid_ce_abstain` — dense (all-MiniLM-L6-v2) + BM25 → standard RRF (k=60) → cross-encoder rerank (bounded pool 20) → calibrated abstention (logit ≥ −10.95).**

Evidence: best combined MRR (0.860), recall (0.847), hit rate (0.886), tied-best rejection (0.895), fewest false abstentions (6), 2 stray memories across all 19 unanswerable queries. The reranker's role is twofold: fixing rank order (q45) and acting as the strongest relevance signal for the gate (H4/H5). ~268 ms/query on CPU is acceptable for an agent memory layer that answers before an LLM call, and both models are free, local, and one-time downloads.

**Documented fallback: `dense_abstain`** (dense → calibrated cosine gate ≥ 0.2625): MRR 0.786, precision 0.696 (best), ~10 ms, one model, no CE download. Choose it if 250 ms/query or the extra model is unacceptable. The held-out split (n=14) actually ranks it first — the two configs are within ~1 query of each other there, and the combined data favors `hybrid_ce_abstain` on every metric except precision; treat the gap between them as unresolved at this sample size, not as a tie or a win.

**Rejected configurations (with reasons):**
- `baseline_current` — worst on both splits (overfit to the old 7-query set); keep as reference only.
- Reranking without abstention — no quality gain, 25× latency.
- Abstention on raw RRF scores (`hybrid_abstain`) — unstable score scale causes the worst false-abstention rate (21).
- BM25-only — natural rejection is attractive, but the recall ceiling on paraphrases is structural.

## 9. Limitations

1. **Sample size.** 70 queries (56/14). Held-out differences of ±0.07 are single queries. Percent-level conclusions are not supportable; the large gaps (abstention vs none) are robust, the `dense_abstain` vs `hybrid_ce_abstain` ordering is not settled.
2. **Fixture size (12 active memories).** Pools (20) exceed the store, so candidate-generation differences are compressed; `dense_ce` ≡ `hybrid_ce` is an artifact of this. BM25/dense divergence will grow with store size.
3. **Cross-encoder domain mismatch.** MS-MARCO-trained, uncalibrated on technical jargon (q46: −8.06 on the correct memory). Its scores are used as *relative* relevance signals only, never probabilities.
4. **One unanswerable query is irrecoverable by thresholds** (11/12 dev ceiling in every config): "Did I ever work at OpenAI?" is semantically adjacent to the current-employer memory. Fixing it requires query-type/answerability detection, which is out of scope here.
5. **No peak-RAM measurement** (Windows environment); reported as unavailable.
6. **`requirements.txt` not pinned** to the new environment — intentional; production adoption should pin `torch`/`sentence-transformers`/`rank-bm25` deliberately.

## 10. Next Highest-Value Experiment

A **relevance-reranker trained or distilled on this domain** (or a lightweight alternative such as a bge-reranker) evaluated against ms-marco-MiniLM-L-12-v2 on the same harness — the CE is simultaneously the best relevance signal and the source of the worst remaining failures (q46/q47). Second: re-run this harness with a 100+ memory fixture to measure how BM25/dense/pool-size tradeoffs change once candidate generation stops being saturated.

## 11. Reproduction

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install sentence-transformers rank-bm25
python evaluation/retrieval_benchmarks.py                    # frozen baseline (7 queries)
python experiments/retrieval/validate_dataset.py             # dataset checks
python experiments/retrieval/calibrate.py                    # dev-split thresholds
python experiments/retrieval/run_experiment.py --all --split both
python -m pytest tests/ -v --ignore=tests/test_postgres.py   # 44 tests
```
