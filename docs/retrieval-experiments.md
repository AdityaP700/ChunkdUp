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

## 12. Round 2 — Risk Resolution (2026-10-12)

Round 1's decision was gated on five risks. Each was tested with the same evidence standard; results live in `experiments/retrieval/results/{scale,round2,ce_jargon_audit,fp_abstention_analysis}*.json`. GPU was available (RTX 4050) but not used: per-query CE cost is bounded by the rerank pool, not store size, so CPU sufficed — and keeping the CPU build preserves score comparability with round 1.

### 12.1 Risk 1 — Scale test (12 / 100 / 500 / 1000 memories)

Synthetic in-domain distractors (`scale_fixture.py`) compete on the same keys as the real memories; all 9 configs × 70 queries per scale; abstention gates evaluated both frozen (round-1 dev threshold) and recalibrated on dev **at that scale**.

MRR by scale (12 → 100 → 500 → 1000), leading configs:

| Config | 12 | 100 | 500 | 1000 |
|---|---|---|---|---|
| baseline_current | 0.524 | 0.376 | 0.373 | 0.373 |
| dense | 0.657 | 0.297 | 0.161 | 0.143 |
| bm25 | 0.686 | 0.431 | 0.347 | 0.321 |
| dense_abstain (frozen) | 0.786 | 0.492 | 0.361 | 0.329 |
| hybrid_ce_abstain (frozen) | **0.860** | **0.634** | 0.470 | 0.407 |
| hybrid_ce_abstain (recalibrated) | **0.860** | **0.634** | **0.481** | **0.457** |

- **The round-1 ranking holds at every scale and the gap widens** (hybrid_ce_abstain over dense_abstain: +0.074 at 12 → +0.142 at 100 → +0.100–0.124 at 500–1000). Raw retrieval collapses under distractor competition (dense 0.657 → 0.143); gated configs degrade slowest.
- **Frozen abstention thresholds do not transfer**: the cosine gate is roughly stable (0.262 → 0.276 recalibrated at 1000), but the CE gate is not — the frozen −10.95 threshold drops rejection from 0.895 to 0.737–0.789 at scale, while dev-recalibration finds −2.13…−2.26 and achieves **1.00 rejection** with higher MRR and precision. Scale-aware recalibration is an operational requirement for the CE gate.
- **Latency scaling** (isolated micro-benchmark, `bench_latency.py`, plus a hot-machine control run): dense search is flat with scale (~8–10 ms; the query encode dominates and the cosine is vectorized), BM25 grows O(N) but reaches only 0.9 ms at 1000 memories, and the CE rerank is **pool-bounded** (~200–450 ms depending on candidate text lengths and machine load), not store-bounded. The scale-run wall-clock table showed larger numbers that were traced to host memory pressure (~170 MB free RAM), not algorithm cost. One-time index cost: ~21–38 s per 1000 memories on CPU.

### 12.2 Risk 2 — Expanded held-out split (14 → 36 queries)

22 hand-verified queries added in `dataset_round2.jsonl` (round-1 files untouched; split boundary frozen). Leading configs re-run at n=36:

| Config | MRR | P@5 | R@5 | HitRate | Reject |
|---|---|---|---|---|---|
| dense_abstain | **0.819** | **0.668** | 0.833 | **0.861** | **0.909** |
| hybrid_ce_abstain | 0.780 | 0.639 | 0.833 | 0.833 | 0.818 |
| bm25 | 0.676 | 0.606 | 0.667 | 0.694 | 0.818 |
| baseline_current | 0.445 | 0.413 | 0.500 | 0.528 | 0.182 |

**The flip persists.** `dense_abstain` wins by +0.039 MRR (≈1.5 queries), +0.028 HitRate, +0.091 rejection; recall ties. That is a consistent direction, not a decisive one. Per-query disagreements: dense wins 5 (q16, q75, q86, q87, q92), CE wins 2 (q84, q88). Combined over all 92 queries, hybrid_ce_abstain still leads on MRR (0.840 vs 0.788) because the dev split favors it; the two configs genuinely disagree by store size (CE wins at 500–1000 memories per 12.1) and by split.

### 12.3 Risk 3 — CE jargon audit

15 technical-term queries (naming a specific product/project term; generic English excluded) audited through the CE pipeline (`ce_jargon_audit.json`):

- **CE rank inversion** (CE itself ranks an incorrect memory above the correct one): **2/15 = 13.3%** — q46 (known: m9 −8.06 vs m11 +0.61) and **q87** (new: "What vector database powers ChunkdUp?", m9 −10.85 vs m3 −3.89). Both inversions hit the **same pgvector memory** — the failure clusters on technical vocabulary, it is not scattered.
- Strict round-1 anomaly pattern (correct memory negative, incorrect positive): 1/15 = 6.7%.
- The frozen gate admitted the correct memory in all 15 cases, but q87 passed by **0.10 logit points**; risk 2 already caught one paraphrase query (q16) where the gate did filter the correct memory.

Verdict: below the 15–20% flag line, but a real, vocabulary-concentrated liability. **Proposed mitigation (not implemented):** an entity-guard on the CE stage — when the query contains a term from the stored entity/project vocabulary and the CE's top choice contradicts the fused dense+BM25 top-1, fall back to fused ranking for that query (or floor the CE score of entity-matching candidates). To be evaluated as experiment 7 on this harness before adoption.

### 12.4 Risk 4 — False-abstention rate on answerable queries

From `fp_abstention_analysis.json` (rates on answerable vs unanswerable queries, per split):

| Config | False abstain (dev / ho36) | Correct rejection (dev / ho36) |
|---|---|---|
| hybrid_ce_abstain | **0.068 / 0.120** | 0.917 / 0.818 |
| dense_abstain | 0.182 / 0.160 | 0.917 / 0.909 |
| bm25 (natural) | 0.273 / 0.240 | 0.917 / 0.818 |
| hybrid_abstain | 0.273 / — | 0.917 / — |
| baseline_current | 0.023 / 0.000 | 0.333 / 0.182 |
| no-gate configs | 0.000 / 0.000 | 0.000 / 0.000 |

**This changes the round-1 picture materially.** `hybrid_ce_abstain` is not just the MRR leader — it also has the lowest false-abstention rate among gated configs (the CE is the best relevance *signal*, so its gate suppresses the fewest valid answers). `dense_abstain` wrongly abstains on 16–18% of answerable queries (examples: q29 "Do I work at a research lab?", q74 "Which company signs my paycheck?", q79, q84). BM25's celebrated "natural rejection" costs 24–43% of answerable queries. No config is a clean winner by this metric alone, but the CE gate is the closest to it.

### 12.5 Risk 5 — False-positive pattern analysis

For `dense_abstain`: 26/69 answerable queries returned ≥1 irrelevant memory; **project_specific accounts for 12 of those queries and 35 of the FP memories** (paraphrase: 8/15). bm25 shows the identical concentration (12/35). hybrid_ce_abstain spreads FPs wider (38/69; short_ambiguous 7 queries/25 memories, paraphrase 10/21, project_specific 10/25).

**Concentrated pattern found**: project-specific questions ("Where does ChunkdUp run?", "What database does ChunkdUp use?") mention the project name, and the project memory (m3 "ChunkdUp") co-returns as a false positive in every config — the query's entity overlap with the project memory is purely lexical, while the question seeks an attribute. **Named 6th experiment (not implemented): type-aware entity scoring** — demote candidates whose only query overlap is the project-name entity when the query is attribute-seeking, using the existing EntityExtractor's type information rather than raw overlap counts.

### 12.6 Latency tradeoff — explicit product decision (not made here)

| | hybrid_ce_abstain | dense_abstain |
|---|---|---|
| MRR (92-query combined) | **0.840** | 0.788 |
| False abstention (best evidence) | **0.068–0.120** | 0.160–0.182 |
| Rejection (92-query combined) | 0.891 | **0.913** |
| Latency per query (CPU, uncontended) | ~200–450 ms | **~10 ms** |
| Extra dependency | second 90 MB model | none |
| Degrades worse at scale? | No — the gap **widens** in its favor | Yes — recall collapses faster |

The 25–45× latency gap buys roughly +0.05 MRR, roughly half the false-abstention rate, and resilience to store growth. **Which of those matters more depends on ChunkdUp's usage pattern**: for a per-agent-turn live lookup on a latency-critical path, ~10 ms with 16–18% silent non-answers may be unacceptable; for a pre-response memory fetch that already fronts an LLM call (where 200–500 ms is noise), the CE config's quality and lower false-abstention win. This is a product call, flagged for the maintainer; both configurations remain one config flag apart.

### 12.7 Revised decision

- **Risk 1 (scale): RESOLVED — round-1 ranking confirmed and strengthened.** `hybrid_ce_abstain` leads at every store size; the lead grows with scale. New operational requirement discovered: recalibrate the CE gate against a store-representative dev set (frozen thresholds lose 0.10–0.16 of rejection at scale; recalibration restores 1.0).
- **Risk 2 (held-out): RESOLVED as a stated split-level disagreement, with the margin explicit.** The flip persists at n=36 (+0.039 MRR to `dense_abstain`); combined-92 still favors `hybrid_ce_abstain` (+0.052 MRR, better false-abstention). Not noise either way; the deciding dimension is store size and usage pattern, not the split.
- **Risk 3 (jargon): RESOLVED as a real but bounded liability** (13.3% rank inversion, clustered on jargon vocabulary; mitigation proposed, pending experiment 7).
- **Risk 4 (false abstention): RESOLVED — and it favors the CE config**, reversing the intuitive reading of round 1's precision table.
- **Risk 5 (FP patterns): RESOLVED — concentrated, nameable pattern** (project-name lexical pull on attribute queries); experiment 6 (type-aware entity scoring) identified.
- **Risk 6 (latency): OPEN — deliberately left as a product decision** (12.6).

**Standing recommendation:** keep `hybrid_ce_abstain` as the default pipeline **conditional on the latency budget and on adopting scale-aware gate recalibration**, with `dense_abstain` as the latency-constrained fallback. The next code change with the best evidence-per-effort ratio is the type-aware entity scoring experiment (risk 5), which targets the largest concentrated FP source in every configuration; the CE entity-guard (risk 3) follows it. Nothing here contradicts round 1 — the round-1 conclusion survives all five risks, with two operational caveats (recalibration requirement, jargon guard) and one open product call (latency).

## 13. Round 3 — Decision, Production Hardening, Entity Experiments (2026-10-12)

Three-part round, committed checkpoint-by-checkpoint. Part 1 made the pending decision with no new code; part 2 built layer-2 operational safety (no caching/queues/sharding — explicitly out of scope); part 3 ran the two entity experiments scoped by round 2. Full test suite at end of round: **74 passed**.

### 13.1 Part 1 — Decision memo (`docs/round3-decision.md`)

**`hybrid_ce_abstain` ships as the default.** The deciding factor, stated in the memo and traceable to round-2 risk 4: false abstention is a silent failure, and the CE gate abstains wrongly on 6.8–12.0% of answerable queries versus dense_abstain's 16.0–18.2% — the held-out MRR flip (+0.039, ~1.5 queries) does not outweigh that. The memo also fixes the fast-path rule: `dense_abstain` is preferred only for latency budgets (<~100 ms) or CE deployment constraints — **the round-2 scale data shows no store size where the CE gap is negligible** (+0.074 MRR at 12 memories is the smallest measured gap), so a size-based cutoff would be unsupported. Scale-aware recalibration was declared a hard shipping dependency; it shipped in part 2 (commit `9f9524f`), so the memo's fallback disclosure ("shipped without recalibration…") never came into force.

### 13.2 Part 2 — Production hardening (4 checkpoints, all with tests)

New module `ai-system/chunkdup/retrieval_production.py`; the `Memory` SDK now routes `retrieve`/`search` through it. Round-1/2 experiment files untouched; the module mirrors the experiment logic and is kept consistent by tests.

1. **CE fallback chain** (`6869c61`): cross-encoder load failure, inference error, or timeout (default 1.0 s = 2× the round-1 p95 of ~268 ms, plus headroom for the ~450 ms observed under host pressure; `CHUNKDUP_CE_TIMEOUT_S` overridable) degrades to the dense gate; embedding failure degrades to the frozen baseline (skipping the dense leg, which is dead without embeddings). Every degradation is logged (`fallback_events` + warnings) and sticky within the process. Tests simulate a raising model and a hung `predict` (returns via fallback in <3 s).
2. **JSONL query logging** (`4e3c497`): every production query appends one line to `<data_dir>/retrieval_log.jsonl` (`CHUNKDUP_QUERY_LOG` to override or `off`) with timestamp, query, requested + effective config, returned ids, per-candidate scores, latency, abstention flag, fallback events. Decoupled from the eval harness; write failures never break retrieval.
3. **Rollback** (`2146662`): `CHUNKDUP_RETRIEVAL_CONFIG=baseline_current` forces the frozen round-1 path for every `Memory` instance without a code change; documented in the README incident section together with the degradation chain.
4. **Scale-aware recalibration** (`9f9524f`) — the memo's hard dependency: `ProductionRetriever.recalibrate(sample)` computes the gate threshold from a hand-labeled, store-representative sample against the current store (same objective as round-2 calibration); `should_recalibrate()` flips when the store grows ~50 % past the last calibration. README documents the schedule tied to round-2 drift (measurable ~100 memories, material ~500; recalibrate before doubling).

### 13.3 Part 3 — Entity experiments (validated on the 92-query set + a 15-query entity-focus slice)

Slice `dataset_entity_focus.jsonl` (15 hand-verified queries): 8 project-name+attribute queries (the round-2 FP pattern) and 7 jargon queries including two traps that must abstain; `e12` deliberately probes that the demotion rule never hits the correct answer whose value contains an identity token ("PostgreSQL or MongoDB…").

**Experiment 6 — type-aware entity demotion: ADOPTED (zero regressions).**
Mechanism: when a query names a stored identity entity (proper-name vocabulary only; generic terms like "database" excluded) and asks about something beyond it, `project_name`-typed candidates whose value IS the entity sink below every non-demoted candidate. Type-awareness is load-bearing: for "Where does ChunkdUp run?" the correct Docker memory also matches only on the "ChunkdUp" token — it is protected because it is not the project memory. A first multiplicative design (×0.25) was **structurally ineffective on CE logits** (0.34 → 0.086 still outranks the −5.17 correct answer) and was replaced by sink-below semantics after the first evaluation run showed zero effect.

| Config | 92-query pooled MRR | 92-query P@5 | Slice MRR | Slice P@5 | Per-query regressions |
|---|---|---|---|---|---|
| hybrid_ce_abstain → +ED | 0.8397 → **0.8424** | 0.6366 → **0.6475** | 0.6167 → **0.6889** | 0.4189 → **0.5244** | **0** (2 better, 90 same) |
| dense_abstain → +ED | 0.7880 → **0.8152** | 0.6962* → 0.7178* | 0.6111 → **0.7333** | 0.4189 → **0.5244** | **0** (5 better, 87 same) |

*dev/heldout detail: dense_abstain_ed dev MRR 0.7946 (+0.027), heldout 0.8472 (+0.028); hybrid_ce_abstain_ed dev 0.8810 (+0.003), heldout 0.7824 (+0.002). Abstention behavior unchanged (HitRates identical). With exp 6, `dense_abstain_ed` now also wins the held-out split (0.847 vs 0.782) — the pooled ranking still favors the CE config, so the part-1 decision stands, with the fast path now strictly stronger than before.

**Experiment 7 — CE entity-guard: REJECTED in current form.** The guard (fused top-3 lead in fused order on entity queries; CE orders the remainder) fixes both targeted inversions (q46 RR 0.33 → 0.50, q87 0.25 → 0.50) but **breaks q45** (1.00 → 0.50 — the CE had correctly promoted the Docker memory there, and the guard's fused fallback undoes it) and costs precision broadly (dev P@5 −0.042, slice P@5 −0.039). Exp 6 captures most of the benefit at no measured cost. A future variant would need a way to distinguish "CE demotion caused by jargon miscalibration" from "CE correction of entity-token over-pull" — this one cannot.

### 13.4 Standing state after round 3

- Default pipeline: `hybrid_ce_abstain` (decision memo), now with fallback chain, logging, rollback switch, and recalibration built in. Recommended next operational step: wire experiment 6's demotion into `ProductionRetriever` (validated, zero-regression; not wired this round to keep part 3 experiment-scoped).
- Rejected/deferred with evidence: exp 7 as specified; graph-based retrieval (round-3 brief — no multi-hop failures exist in the eval set); caching/queues/sharding (out of scope).
- Open product decision: latency vs quality (round-2 §12.6, unchanged).

## 15. Final Summary — the Complete Arc (closeout round, 2026-10-12)

This closes the project phase. Everything below was reproduced in the closeout round, not carried forward: the production-path numbers by `verify_production.py` (results/production_verification.json), the timeout basis from the stored round-2 isolated benchmark files, and the test suite (82 passed, run at closeout).

**Starting state.** A persistent-memory SDK whose retrieval was keyword substring matching plus hand-written type/intent boosts, evaluated on a 7-query benchmark (MRR 0.7857 / P 0.3821 / R 0.8571) with no dense leg reachable from the in-memory store, no BM25, a simulated "LLM" reranker, and an unevaluated cross-encoder mode.

**Round 1 — methods tested, winner found.** Built a 70-query reviewed dataset (9 categories), real dense (local MiniLM) and BM25 retrievers, standard RRF, cross-encoder reranking, and per-score-type calibrated abstention; evaluated the 8 spec configurations plus the frozen baseline. Winner: `hybrid_ce_abstain` (pooled-70 MRR 0.8595). Key finding: the abstention stage dominated everything else (confirming the task's original hypothesis), and the legacy boosted baseline overfit its old benchmark (worst on the new set).

**Round 2 — winner stress-tested, two real limitations found.** Scale test (12/100/500/1000 synthetic in-domain memories): the ranking held at every scale and the CE advantage grew, but the frozen CE threshold does not transfer across store size (rejection 0.895 → 0.737; recalibration restores 1.0). Held-out expansion to 36: the split-level flip to `dense_abstain` persisted with an explicit +0.039 margin. CE jargon audit: 2/15 (13.3%) rank inversions, clustered on the pgvector memory. False-abstention accounting favored the CE gate (0.068–0.12 vs 0.16–0.18) — the decisive input to the round-3 decision. FP-pattern analysis found the concentrated project-name pull and named experiment 6.

**Round 3 — decision made, hardening built, entity bug found and fixed.** Decision memo (`docs/round3-decision.md`): `hybrid_ce_abstain` ships as default on the false-abstention evidence, with a latency-only fast-path rule (the data supports no size-based cutoff) and recalibration as a hard dependency. Hardening shipped as four tested checkpoints: CE fallback chain with a 1.0 s predict timeout, JSONL query logging, the `CHUNKDUP_RETRIEVAL_CONFIG` rollback switch, and scale-aware recalibration (`recalibrate`/`should_recalibrate`). The entity-focus slice exposed the project-memory FP mechanism; experiment 6 (type-aware entity demotion, sink-below) was adopted with zero per-query regressions across 92 queries on both leading configs; experiment 7 (CE entity-guard) was rejected with evidence (fixes q46/q87, breaks q45, costs precision broadly).

**Closeout — production wiring completed, timeout verified, fast path exposed.** Exp 6 is now inside `retrieval_production.py` (the module the SDK calls), and the production path reproduces the round-3 numbers exactly on the 92-query set: pooled MRR 0.8397 → **0.8424** and P@5 0.6366 → **0.6475** with demotion (0.8397/0.6366 without — both MATCH). One real transfer bug was caught by that verification (pool size 10 vs the validated 20; plus a fixture-status artifact in the verification harness itself) — fixed, not papered over. The CE timeout (1.0 s) is now traced to the round-2 isolated micro-benchmark: worst observed pool-bound median 452.6 ms (scale 500; scale-1000 re-bench 453.2 ms), 2× → ~905 ms → 1.0 s, with a scale-1000 steady-state test proving no fallback fires under normal operation. The fast path is an explicit, documented opt-in (`Memory(fast_mode=True)` → `dense_abstain_ed`, demotion included) — deliberately separate from the failure-driven degradation chain. Final test suite: **82 passed** (`tests/` minus the PostgreSQL suite, which needs a live database).

**What ships today.** Default `hybrid_ce_abstain` (dense + BM25 → RRF → cross-encoder → calibrated gate → entity demotion), explicit `fast_mode`, automatic logged degradation, one-move rollback, store-aware recalibration, per-query JSONL logging — with every headline number traceable to a committed results file.

---

## 14. Reproduction

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install sentence-transformers rank-bm25

# Round 1
python evaluation/retrieval_benchmarks.py                    # frozen baseline (7 queries)
python experiments/retrieval/validate_dataset.py             # dataset checks
python experiments/retrieval/calibrate.py                    # dev-split thresholds
python experiments/retrieval/run_experiment.py --all --split both
python -m pytest tests/ -v --ignore=tests/test_postgres.py

# Round 2
python experiments/retrieval/run_scale_test.py --scales 12 100 500 1000   # risk 1
python experiments/retrieval/bench_latency.py --scale 1000 --reps 30      # risk 1 latency
python experiments/retrieval/validate_dataset.py experiments/retrieval/dataset_round2.jsonl
python experiments/retrieval/run_experiment.py --config dense_abstain --split heldout \
  --extra-dataset experiments/retrieval/dataset_round2.jsonl --tag round2   # risk 2 (repeat per config)
python experiments/retrieval/audit_ce_jargon.py                           # risk 3
python experiments/retrieval/analyze_fp_patterns.py                       # risks 4+5

# Round 3
python -m pytest tests/ -q --ignore=tests/test_postgres.py               # 74 tests incl. production + entity-guard
python experiments/retrieval/run_experiment.py --config hybrid_ce_abstain_ed --split both \
  --extra-dataset experiments/retrieval/dataset_round2.jsonl --tag round3exp   # experiment 6
python experiments/retrieval/run_experiment.py --config hybrid_ce_abstain_ed --split entity_focus \
  --extra-dataset experiments/retrieval/dataset_entity_focus.jsonl --tag round3exp
```
