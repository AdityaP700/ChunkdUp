# experiments/retrieval/bench_latency.py
"""
Isolated latency micro-benchmark for the scale test.

The scale-run wall-clock numbers are contaminated by machine-level memory
pressure (the host was at ~170 MB free RAM during collection, so everything
slows and degrades with resident data). This benchmark isolates the
ALGORITHMIC per-stage cost by timing the exact stage operations in tight
repetition loops in a fresh process per scale:

  - bm25.search      (expected ~O(N) corpus scan)
  - dense.search     (query encode ~const + O(N) cosine)
  - ce.predict 20    (constant pairs -> expected ~constant)

Usage: python experiments/retrieval/bench_latency.py --scale 1000 [--reps 30]
Writes experiments/retrieval/results/scale/latency_bench_{scale}.json
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

from config import CONFIGS
from run_experiment import ExperimentPipeline, load_dataset
from scale_fixture import generate_scale_memories


def time_stage(fn, reps: int) -> Dict:
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "median_ms": round(statistics.median(times), 3),
        "mean_ms": round(statistics.mean(times), 3),
        "min_ms": round(min(times), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, required=True)
    parser.add_argument("--reps", type=int, default=30)
    args = parser.parse_args()

    from sentence_transformers import CrossEncoder, SentenceTransformer
    from retrievers import embedding_model_source

    memories = generate_scale_memories(args.scale)
    rows = load_dataset("both")
    queries = [r["query"] for r in rows]

    out: Dict = {"scale": args.scale, "n_memories": len(memories), "reps": args.reps, "stages": {}}

    # BM25 stage
    bm25 = ExperimentPipeline(CONFIGS["bm25"], memories=memories)
    out["stages"]["bm25_search"] = time_stage(lambda: bm25.bm25.search(queries[0], 20), args.reps)
    out["stages"]["bm25_search_all_queries_avg"] = time_stage(
        lambda: [bm25.bm25.search(q, 20) for q in queries], 3
    )
    del bm25

    # Dense stage (shared ST)
    st = SentenceTransformer(embedding_model_source())
    dense = ExperimentPipeline(CONFIGS["dense"], memories=memories, dense_model=st)
    out["stages"]["dense_search"] = time_stage(lambda: dense.dense.search(queries[0], 20), args.reps)
    out["stages"]["dense_search_all_queries_avg"] = time_stage(
        lambda: [dense.dense.search(q, 20) for q in queries], 3
    )
    del dense

    # Cross-encoder stage (constant 20 pairs)
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    rerank_pipeline = ExperimentPipeline(CONFIGS["hybrid_ce"], memories=memories, dense_model=st, cross_model=ce)
    candidates = rerank_pipeline.dense.search(queries[0], 20)
    out["stages"]["ce_predict_20_pairs"] = time_stage(lambda: ce.predict([[queries[0], "sample memory text"]] * 20), args.reps)
    out["stages"]["ce_full_rerank"] = time_stage(lambda: rerank_pipeline.reranker.rerank(queries[0], candidates, 20), args.reps)

    out_path = Path(__file__).parent / "results" / "scale" / f"latency_bench_{args.scale}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"scale {args.scale}:")
    for stage, timings in out["stages"].items():
        print(f"  {stage:<28} median={timings['median_ms']}ms  min={timings['min_ms']}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
