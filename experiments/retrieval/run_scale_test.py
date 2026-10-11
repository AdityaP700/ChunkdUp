# experiments/retrieval/run_scale_test.py
"""
Round-2 risk 1: scale test. Runs all 9 configurations against synthetic
memory stores of 12 / 100 / 500 / 1000 memories (real fixture + in-domain
distractors from scale_fixture.py) over the fixed 70-query benchmark.

Method notes:
  - Abstention configs share candidate pipelines with their non-abstention
    counterparts (dense_abstain<-dense, hybrid_abstain<-hybrid_rrf,
    hybrid_ce_abstain<-hybrid_ce); the gate is simulated on the collected
    ranked lists, so each store is traversed once per stage-combination.
  - Two gate variants are reported per abstention config:
      frozen    = round-1 dev-calibrated threshold (as deployed)
      recali    = threshold re-calibrated on the dev split AT THIS SCALE
    Recalibration never touches held-out queries.
  - Latency is wall-clock per query on CPU; per-query cross-encoder cost is
    bounded by the rerank pool (20), not the store size.

Usage: python experiments/retrieval/run_scale_test.py [--scales 12 100 500 1000]
Output: experiments/retrieval/results/scale/scale_{n}.json + scale_summary.json
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

from config import BASELINE_CURRENT, CONFIGS, ExperimentConfig
from evaluate import aggregate, score_query
from run_experiment import (
    BaselinePipeline,
    ExperimentPipeline,
    load_calibration,
    load_dataset,
)
from scale_fixture import generate_scale_memories, validate_scale_fixture

RESULTS_DIR = Path(__file__).parent / "results" / "scale"

ABSTAIN_SOURCE = {  # abstain config -> (source config, score field)
    "dense_abstain": ("dense", "_similarity"),
    "hybrid_abstain": ("hybrid_rrf", "_rrf_score"),
    "hybrid_ce_abstain": ("hybrid_ce", "_rerank_score"),
}


def collect(pipeline, rows):
    """Run every query once; return per-query records with ranked candidates."""
    records = []
    for row in rows:
        t0 = time.perf_counter()
        results, timings, ranked = pipeline.retrieve(row["query"], k=5)
        elapsed = (time.perf_counter() - t0) * 1000
        records.append({
            "row": row,
            "returned_ids": [r["id"] for r in results],
            "ranked_ids": [c["id"] for c in ranked],
            "scores": [float(c.get(_pick_score_field(pipeline), 0.0)) for c in ranked],
            "timings": {**timings, "total_ms": elapsed},
        })
    return records


def _pick_score_field(pipeline):
    # The collection pipeline is the non-abstain source; expose the score the
    # abstain config would gate on.
    name = pipeline.config.name
    if name == "dense":
        return "_similarity"
    if name == "hybrid_rrf":
        return "_rrf_score"
    if name == "hybrid_ce":
        return "_rerank_score"
    return "_score"


def metrics_from_ids(rows, ids_per_query):
    per_query = []
    for row, ids in zip(rows, ids_per_query):
        sq = score_query(row, ids)
        expected = set(row["relevant_memory_ids"])
        sq.update({
            "n_relevant_returned": len([i for i in ids if i in expected]),
            "n_irrelevant_returned": len([i for i in ids if i not in expected]),
            "abstained": not ids,
        })
        per_query.append(sq)
    return aggregate(per_query, rows)


def gate_ids(record, threshold):
    """Simulate AbstentionGate on a collected record."""
    scores = record["scores"]
    if not scores or scores[0] < threshold:
        return []
    return [rid for rid, s in zip(record["ranked_ids"], scores) if s >= threshold][:5]


def recalibrate(dev_records):
    """Same objective as calibrate.py: 0.5*hit + 0.5*reject, ties -> lower threshold."""
    observed = sorted({r["scores"][0] for r in dev_records if r["scores"]})
    if not observed:
        return None
    candidates = []
    for i, s in enumerate(observed):
        candidates.append(s)
        if i + 1 < len(observed):
            candidates.append((s + observed[i + 1]) / 2)
    candidates.append(observed[-1] + 1e-6)
    candidates.append(min(0.0, observed[0] - 1e-6))
    candidates = sorted(set(candidates))

    best_threshold, best_objective = None, -1.0
    for t in candidates:
        ans_hits = ans_total = unans_reject = 0
        for r in dev_records:
            ids = gate_ids(r, t)
            expected = set(r["row"]["relevant_memory_ids"])
            if r["row"]["answerable"]:
                ans_total += 1
                if any(i in expected for i in ids):
                    ans_hits += 1
            else:
                unans_reject += 1 if not ids else 0
        hit_rate = ans_hits / ans_total if ans_total else 0.0
        reject_rate = unans_reject / max(1, sum(1 for r in dev_records if not r["row"]["answerable"]))
        objective = 0.5 * hit_rate + 0.5 * reject_rate
        if objective > best_objective or (objective == best_objective and best_threshold is not None and t < best_threshold):
            best_threshold, best_objective = t, objective
    return {"threshold": best_threshold, "objective": round(best_objective, 4)}


def run_scale(scale: int, rows: List[Dict], frozen_calibration: Dict) -> Dict:
    print(f"=== scale {scale} ===")
    import gc

    memories = generate_scale_memories(scale)
    validate_scale_fixture(memories)

    scale_result = {"scale": scale, "n_memories": len(memories), "configs": {}}
    collected: Dict[str, List] = {}

    # Stage 1: model-free pipelines (baseline, bm25).
    t0 = time.perf_counter()
    pipelines = {
        BASELINE_CURRENT: BaselinePipeline(ExperimentConfig(name=BASELINE_CURRENT, retriever="both"), memories=memories),
        "bm25": ExperimentPipeline(CONFIGS["bm25"], memories=memories),
    }
    for name in (BASELINE_CURRENT, "bm25"):
        records = collect(pipelines[name], rows)
        collected[name] = records
        ids_per_query = [r["returned_ids"] for r in records]
        metrics = metrics_from_ids(rows, ids_per_query)
        scale_result["configs"][name] = {"metrics": metrics, "latency_ms": _latency(records), "variant": "direct"}
        print(f"  {name:<18} MRR={metrics['MRR']:<7} P={metrics['Precision@K']:<7} "
              f"R={metrics['Recall@K']:<7} Hit={metrics['HitRate@K']:<7} "
              f"rej={metrics['unanswerable']['correct_rejection_rate']}")
    del pipelines[BASELINE_CURRENT], pipelines["bm25"]
    gc.collect()

    # Stage 2: one shared embedding model -> dense + hybrid_rrf (also feed the
    # dense_abstain / hybrid_abstain gate simulations).
    from sentence_transformers import SentenceTransformer
    from retrievers import embedding_model_source

    st = SentenceTransformer(embedding_model_source())
    index_seconds = time.perf_counter() - t0
    pipelines = {}
    for name in ("dense", "hybrid_rrf"):
        pipelines[name] = ExperimentPipeline(CONFIGS[name], memories=memories, dense_model=st)
    for name in ("dense", "hybrid_rrf"):
        records = collect(pipelines[name], rows)
        collected[name] = records
        ids_per_query = [r["returned_ids"] for r in records]
        metrics = metrics_from_ids(rows, ids_per_query)
        scale_result["configs"][name] = {"metrics": metrics, "latency_ms": _latency(records), "variant": "direct"}
        print(f"  {name:<18} MRR={metrics['MRR']:<7} P={metrics['Precision@K']:<7} "
              f"R={metrics['Recall@K']:<7} Hit={metrics['HitRate@K']:<7} "
              f"rej={metrics['unanswerable']['correct_rejection_rate']}")

    # Stage 3: one shared cross-encoder -> dense_ce + hybrid_ce (also feed the
    # hybrid_ce_abstain gate simulation). Freed before metric assembly.
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    pipelines = {}
    for name in ("dense_ce", "hybrid_ce"):
        pipelines[name] = ExperimentPipeline(CONFIGS[name], memories=memories, dense_model=st, cross_model=ce)
    for name in ("dense_ce", "hybrid_ce"):
        records = collect(pipelines[name], rows)
        collected[name] = records
        ids_per_query = [r["returned_ids"] for r in records]
        metrics = metrics_from_ids(rows, ids_per_query)
        scale_result["configs"][name] = {"metrics": metrics, "latency_ms": _latency(records), "variant": "direct"}
        print(f"  {name:<18} MRR={metrics['MRR']:<7} P={metrics['Precision@K']:<7} "
              f"R={metrics['Recall@K']:<7} Hit={metrics['HitRate@K']:<7} "
              f"rej={metrics['unanswerable']['correct_rejection_rate']}")
    del ce, pipelines
    gc.collect()

    scale_result["index_seconds"] = round(index_seconds, 2)

    # Stage 4: gate simulations from collected ranked lists (no models needed).
    for name, (source, score_field) in ABSTAIN_SOURCE.items():
        records = collected[source]
        frozen = frozen_calibration.get(name, {}).get("threshold")
        variants = {}
        if frozen is not None:
            ids_per_query = [gate_ids(r, frozen) for r in records]
            variants["frozen"] = {"threshold": frozen, "metrics": metrics_from_ids(rows, ids_per_query)}
        dev_records = [r for r in records if r["row"]["split"] == "dev"]
        recal = recalibrate(dev_records)
        if recal:
            ids_per_query = [gate_ids(r, recal["threshold"]) for r in records]
            variants["recalibrated"] = {
                "threshold": recal["threshold"],
                "objective": recal["objective"],
                "metrics": metrics_from_ids(rows, ids_per_query),
            }
        scale_result["configs"][name] = {"score_field": score_field, "latency_ms": _latency(records), "variants": variants}
        for vname, v in variants.items():
            m = v["metrics"]
            print(f"  {name:<18}[{vname:<12}] MRR={m['MRR']:<7} P={m['Precision@K']:<7} "
                  f"R={m['Recall@K']:<7} Hit={m['HitRate@K']:<7} "
                  f"rej={m['unanswerable']['correct_rejection_rate']}  thr={v['threshold']}")
    del st
    gc.collect()
    return scale_result


def _latency(records) -> Dict:
    out = {}
    for stage in ("total_ms", "retrieve_dense_ms", "retrieve_bm25_ms", "fuse_ms", "rerank_ms"):
        values = [r["timings"][stage] for r in records if stage in r["timings"]]
        if values:
            out[stage] = {
                "mean": round(statistics.mean(values), 3),
                "p50": round(statistics.median(values), 3),
                "p95": round(sorted(values)[int(0.95 * (len(values) - 1))], 3),
            }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", type=int, nargs="+", default=[12, 100, 500, 1000])
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_dataset("both")
    frozen = load_calibration()

    summary = []
    for scale in args.scales:
        result = run_scale(scale, rows, frozen)
        out = RESULTS_DIR / f"scale_{scale}.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  saved {out}")
        summary.append({"scale": scale, "file": out.name})

    (RESULTS_DIR / "scale_summary.json").write_text(
        json.dumps({"protocol": "70-query benchmark; distractors from scale_fixture.py; "
                                "gate variants: frozen (round-1 dev threshold) vs recalibrated at scale"},
                   indent=2) , encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
