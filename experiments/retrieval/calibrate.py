# experiments/retrieval/calibrate.py
"""
Calibrates abstention thresholds for the *_abstain configurations using the
DEV split only. The held-out split is never touched here.

For each abstention config:
  1. Run the pipeline with the gate disabled; record every query's ranked
     candidates and the score the gate would look at.
  2. Sweep candidate thresholds over the observed score distribution.
  3. Objective per threshold:
         0.5 * correct_rejection_rate_on_unanswerable
       + 0.5 * hit_rate_on_answerable (>=1 relevant memory in filtered top-k)
     Ties break toward the LOWER threshold (favor recall preservation).
  4. Persist threshold + sweep curve to results/calibration.json.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

from config import CONFIGS, ExperimentConfig
from run_experiment import (
    DEFAULT_ABSTAIN_FIELD,
    RESULTS_DIR,
    ExperimentPipeline,
    load_dataset,
)

K = 5


def collect_scores(config: ExperimentConfig, rows: List[Dict]) -> List[Dict]:
    """Run the pipeline with the gate disabled; return per-query top scores."""
    disabled = ExperimentConfig(
        name=config.name, retriever=config.retriever, fusion=config.fusion,
        rerank=config.rerank, abstain=False, pool_size=config.pool_size,
        rrf_k=config.rrf_k, rerank_pool=config.rerank_pool, k=config.k,
    )
    pipeline = ExperimentPipeline(disabled)
    score_field = config.abstain_score_field or DEFAULT_ABSTAIN_FIELD[
        (config.retriever, config.rerank == "cross_encoder")
    ]

    observations = []
    for row in rows:
        _final, _timings, ranked = pipeline.retrieve(row["query"], k=K)
        top_score = float(ranked[0].get(score_field, 0.0)) if ranked else None
        scores = [float(c.get(score_field, 0.0)) for c in ranked]
        observations.append({
            "id": row["id"],
            "answerable": row["answerable"],
            "expected": row["relevant_memory_ids"],
            "candidates": [{"id": c["id"]} for c in ranked],
            "top_score": top_score,
            "scores": scores,
        })
    return observations


def simulate(observations: List[Dict], threshold: float, k: int = K) -> Dict:
    answerable = [o for o in observations if o["answerable"]]
    unanswerable = [o for o in observations if not o["answerable"]]

    answerable_hit = 0
    for o in answerable:
        kept = 0
        for pos, score in enumerate(o["scores"][:k]):
            if score >= threshold:
                if o["candidates"][pos]["id"] in o["expected"]:
                    kept += 1
        if kept:
            answerable_hit += 1
    answerable_fn = sum(1 for o in answerable if o["top_score"] is None or o["top_score"] < threshold)
    reject = sum(1 for o in unanswerable if o["top_score"] is None or o["top_score"] < threshold)
    fp = len(unanswerable) - reject

    n_ans, n_unans = len(answerable), len(unanswerable)
    hit_rate = answerable_hit / n_ans if n_ans else 0.0
    reject_rate = reject / n_unans if n_unans else 0.0
    return {
        "threshold": round(threshold, 6),
        "answerable_hit_rate": round(hit_rate, 4),
        "answerable_false_abstain": answerable_fn,
        "unanswerable_reject_rate": round(reject_rate, 4),
        "unanswerable_false_positives": fp,
        "objective": round(0.5 * hit_rate + 0.5 * reject_rate, 4),
    }


def calibrate(config_name: str) -> Dict:
    config = CONFIGS[config_name]
    if not config.abstain:
        raise ValueError(f"{config_name} is not an abstention config")
    rows = [r for r in load_dataset("dev")]
    observations = collect_scores(config, rows)

    observed = sorted({o["top_score"] for o in observations if o["top_score"] is not None})
    if not observed:
        raise RuntimeError(f"No scores observed for {config_name}; cannot calibrate.")

    # Sweep: observed scores plus midpoints below the minimum, to allow
    # thresholds that reject everything observed.
    candidates = []
    for i, s in enumerate(observed):
        candidates.append(s)
        if i + 1 < len(observed):
            candidates.append((s + observed[i + 1]) / 2)
    candidates.append(observed[-1] + 1e-6)  # reject everything
    candidates.append(0.0 if observed[0] > 0 else observed[0] - 1e-6)
    candidates = sorted(set(candidates))

    sweep = [simulate(observations, t) for t in candidates]
    best = max(sweep, key=lambda s: (s["objective"], -s["threshold"]))

    return {
        "config": config_name,
        "score_field": config.abstain_score_field or DEFAULT_ABSTAIN_FIELD[(config.retriever, config.rerank == "cross_encoder")],
        "threshold": best["threshold"],
        "objective": best["objective"],
        "dev_metrics_at_threshold": best,
        "sweep": sweep,
    }


def main() -> int:
    RESULTS_DIR.mkdir(exist_ok=True)
    output = {}
    for name in ("dense_abstain", "hybrid_abstain", "hybrid_ce_abstain"):
        print(f"Calibrating {name} on dev split ...")
        result = calibrate(name)
        output[name] = result
        b = result["dev_metrics_at_threshold"]
        print(
            f"  threshold={result['threshold']}  objective={result['objective']}  "
            f"hit_rate={b['answerable_hit_rate']}  reject_rate={b['unanswerable_reject_rate']}  "
            f"false_abstains={b['answerable_false_abstain']}  fps={b['unanswerable_false_positives']}"
        )
    out_path = RESULTS_DIR / "calibration.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved calibration to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
