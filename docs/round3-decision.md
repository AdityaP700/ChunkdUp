# Round-3 Decision Memo — Default Retrieval Configuration

> Status: decision made 2026-10-12, before any round-3 implementation work.
> Evidence base: round-1 (`docs/retrieval-experiments.md` §5) and round-2 (§12) results only — no new benchmarks were run for this memo.
> This memo gates round-3 parts 2 and 3.

## Decision

**`hybrid_ce_abstain` ships as the default retrieval configuration.** `dense_abstain` is the documented fast path (rule below).

## Deciding factor: the false-abstention finding, stated explicitly

The deciding evidence is round-2 risk 4 (`results/fp_abstention_analysis.json`), **not** the held-out MRR flip:

- `hybrid_ce_abstain` wrongly abstains on **6.8%–12.0%** of answerable queries (dev 0.068, held-out-36 0.120) while correctly rejecting 82–92% of unanswerable ones.
- `dense_abstain` wrongly abstains on **16.0%–18.2%** (dev 0.182, held-out-36 0.160) at comparable rejection rates.

Reasoning, traceable to the numbers: a false abstention is a **silent failure** — an agent memory layer that answers a valid question with `[]` gives its caller nothing and gives no signal that anything was missed. A lower precision score, by contrast, still delivers the relevant memory with some noise around it. The round-1 "confirm hypothesis" framing (abstention adds value) cut both ways once risk 4 quantified the cost side: the gate's quality is determined by *both* rates, and the CE gate dominates on the rate that fails silently. That is why the default goes to `hybrid_ce_abstain` even though `dense_abstain` wins the held-out split.

On the MRR evidence itself, for completeness: the held-out-36 flip (+0.039 to `dense_abstain`) is ~1.5 queries, contradicted by the dev split (+0.110 to `hybrid_ce_abstain`) and by the pooled 92-query set (+0.052 to `hybrid_ce_abstain`). It is acknowledged and **not ignored** — it is exactly what the fast-path rule below responds to — but a 1-of-3 evaluation-views win at the smallest store size does not outweigh the false-abstention evidence.

## Fast-path rule: when to prefer `dense_abstain`

`dense_abstain` is competitive at small/medium stores (it won the held-out-36 split) and costs ~10 ms/query versus ~200–450 ms for the CE pipeline, with no second model to deploy. Prefer it when **any** of the following hold:

1. **Latency budget**: the caller requires sub-~100 ms retrieval per query (per-agent-turn hot path), or sets `CHUNKDUP_RETRIEVAL_CONFIG=dense_abstain` (the round-3 rollback/selection flag, part 2).
2. **Deployment constraint**: the cross-encoder model (~90 MB, torch) cannot be shipped or loaded in the target environment. The pipeline degrades to the dense gate automatically on CE load failure (part 2).
3. **Store size — checked against round-2 scale data, and the data says: there is no size cutoff.** The brief allowed for a threshold "below which the CE gap is negligible." The measured gap (recalibrated gates, MRR) is **+0.074 at 12 memories** — the *smallest* tested — and +0.142 / +0.107 / +0.100 at 100 / 500 / 1000. The CE advantage is meaningful at every store size examined, so size alone never justifies `dense_abstain`; only latency or deployment constraints do. State this where the rule is documented so nobody later "discovers" a size cutoff that the data does not support.

## Hard dependency: scale-aware gate recalibration

The frozen CE abstention threshold (−10.947, calibrated on the 12-memory dev set) is **not shippable as a constant**: round-2 risk 1 showed rejection degrading from 0.895 (12 memories) to 0.842 (100), 0.737 (500), and 0.737 (1000), while dev-split recalibration at each scale restored **1.00** rejection at thresholds of −2.1…−2.3 with higher MRR and precision.

Therefore: **`hybrid_ce_abstain` may only ship with store-aware gate recalibration built in.** Delivery status: required in this round as round-3 part 2 item 4 (`retrieval_production.recalibrate` — pending at memo-writing time; see the git log for the recalibration commit that satisfies this memo). If it had not shipped, this memo would be required to state, explicitly rather than silently: *"shipped without recalibration — known limitation: the abstention gate's rejection accuracy degrades from ~0.90 to ~0.74 once the store passes roughly 500 memories, and the gate must be recalibrated before that point."* Re-calibration guidance (tied to the round-2 drift finding, not an arbitrary schedule): recalibrate on startup, and again whenever the active store has grown by ~50% past the size of the last calibration — drift is already measurable at 100 memories and material by ~500.

## Incident path

Alongside this decision, the frozen round-1 baseline remains reachable in one move: `CHUNKDUP_RETRIEVAL_CONFIG=baseline_current` (round-3 part 2 item 3, documented in the README). Any production incident with the new pipeline rolls back without a code change.

## What would revisit this decision

- Part-2 fallback telemetry showing the CE stage failing in practice more often than the round-2 evidence suggests (the memo would be reopened and the default reconsidered).
- Experiment 6/7 results (round-3 part 3) materially changing the quality picture — those target the known FP/jargon weaknesses of the CE pipeline and are expected to *strengthen*, not weaken, this decision.
- A future evaluation view (larger held-out set, bigger store) showing `dense_abstain` winning the false-abstention dimension, not just MRR.
