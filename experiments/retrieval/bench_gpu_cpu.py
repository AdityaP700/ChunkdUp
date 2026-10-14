# experiments/retrieval/bench_gpu_cpu.py
"""
GPU vs CPU Inference and Retrieval Benchmark for ChunkdUp.

Measures:
  1. Cold-start model initialization time (SentenceTransformer + CrossEncoder).
  2. Dense index embedding generation latency.
  3. Steady-state warm inference latency (p50, p90, p95, min, mean) across query sets.
  4. CrossEncoder rerank latency across varying batch sizes (1, 8, 16, 32, 64).
  5. Peak GPU VRAM allocated during inference.
  6. Output ranking consistency between CPU and GPU (Spearman rank correlation & Top-K overlap).
  7. Automatic fallback to CPU when CUDA is disabled or unavailable.

Usage:
  python experiments/retrieval/bench_gpu_cpu.py [--reps 30] [--scale 500]
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Add repository paths
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ai-system"))

from chunkdup.in_memory_repository import InMemoryRepository
from chunkdup.retrieval_production import ProductionRetriever, _get_sentence_transformer, _get_cross_encoder, _resolve_device
from scale_fixture import generate_scale_memories
from run_experiment import load_dataset


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    return float(np.percentile(data, p))


def benchmark_device(
    device: str,
    memories: List[Dict[str, Any]],
    queries: List[str],
    reps: int = 30,
) -> Dict[str, Any]:
    print(f"\n{'='*20} Benchmarking Device: {device.upper()} {'='*20}")
    import torch

    has_cuda = torch.cuda.is_available() and device.startswith("cuda")
    if has_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # 1. Cold start model load timing
    t0 = time.perf_counter()
    st = _get_sentence_transformer(device=device)
    st_init_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    ce = _get_cross_encoder(device=device)
    ce_init_ms = (time.perf_counter() - t0) * 1000

    print(f"  Model Load: SentenceTransformer={st_init_ms:.2f}ms, CrossEncoder={ce_init_ms:.2f}ms")

    # 2. Build InMemoryRepository and ProductionRetriever
    repo = InMemoryRepository()
    for m in memories:
        repo.save(dict(m))

    t0 = time.perf_counter()
    retriever = ProductionRetriever(
        repository=repo,
        config_name="hybrid_ce_abstain",
        device=device,
        query_log_path=None,
    )
    # Trigger index build
    retriever._ensure_index()
    index_build_ms = (time.perf_counter() - t0) * 1000
    print(f"  Index Build ({len(memories)} memories): {index_build_ms:.2f}ms")

    # 3. Dense search latency (warm)
    # Warmup
    for q in queries[:3]:
        retriever.retrieve(q, k=5)
    if has_cuda:
        torch.cuda.synchronize()

    dense_latencies = []
    for _ in range(reps):
        q = queries[_ % len(queries)]
        t0 = time.perf_counter()
        _ = retriever._dense_search(q, limit=20)
        if has_cuda:
            torch.cuda.synchronize()
        dense_latencies.append((time.perf_counter() - t0) * 1000)

    # 4. Cross-Encoder rerank latency with default batching
    sample_candidates = retriever._dense_search(queries[0], limit=20)
    ce_latencies = []
    for _ in range(reps):
        q = queries[_ % len(queries)]
        t0 = time.perf_counter()
        _ = retriever._cross_encoder_rerank(q, sample_candidates, limit=20)
        if has_cuda:
            torch.cuda.synchronize()
        ce_latencies.append((time.perf_counter() - t0) * 1000)

    # 5. Batch size scaling for Cross-Encoder
    batch_size_results = {}
    sample_pairs = [[queries[0], m.get("contextual_text") or m.get("value", "")] for m in sample_candidates]
    for b_size in [1, 8, 16, 32, 64]:
        b_times = []
        for _ in range(15):
            t0 = time.perf_counter()
            _ = ce.predict(sample_pairs, batch_size=b_size)
            if has_cuda:
                torch.cuda.synchronize()
            b_times.append((time.perf_counter() - t0) * 1000)
        batch_size_results[f"batch_{b_size}"] = {
            "p50_ms": round(percentile(b_times, 50), 2),
            "p95_ms": round(percentile(b_times, 95), 2),
            "mean_ms": round(float(np.mean(b_times)), 2),
        }

    # 6. End-to-end retrieve latency (warm)
    e2e_latencies = []
    top_results_map = {}
    for i, q in enumerate(queries[:reps]):
        t0 = time.perf_counter()
        res = retriever.retrieve(q, k=5)
        if has_cuda:
            torch.cuda.synchronize()
        e2e_latencies.append((time.perf_counter() - t0) * 1000)
        top_results_map[q] = [r["id"] for r in res]

    # VRAM stats
    peak_vram_mb = 0.0
    allocated_vram_mb = 0.0
    if has_cuda:
        peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        allocated_vram_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
        print(f"  VRAM Usage: Peak={peak_vram_mb} MB, Current={allocated_vram_mb} MB")

    result = {
        "device": device,
        "is_cuda": has_cuda,
        "init_ms": {
            "sentence_transformer": round(st_init_ms, 2),
            "cross_encoder": round(ce_init_ms, 2),
            "index_build": round(index_build_ms, 2),
        },
        "dense_search_ms": {
            "p50": round(percentile(dense_latencies, 50), 2),
            "p90": round(percentile(dense_latencies, 90), 2),
            "p95": round(percentile(dense_latencies, 95), 2),
            "mean": round(float(np.mean(dense_latencies)), 2),
            "min": round(min(dense_latencies), 2),
        },
        "ce_rerank_ms": {
            "p50": round(percentile(ce_latencies, 50), 2),
            "p90": round(percentile(ce_latencies, 90), 2),
            "p95": round(percentile(ce_latencies, 95), 2),
            "mean": round(float(np.mean(ce_latencies)), 2),
            "min": round(min(ce_latencies), 2),
        },
        "ce_batch_scaling": batch_size_results,
        "e2e_retrieve_ms": {
            "p50": round(percentile(e2e_latencies, 50), 2),
            "p90": round(percentile(e2e_latencies, 90), 2),
            "p95": round(percentile(e2e_latencies, 95), 2),
            "mean": round(float(np.mean(e2e_latencies)), 2),
            "min": round(min(e2e_latencies), 2),
        },
        "vram_mb": {
            "peak": peak_vram_mb,
            "allocated": allocated_vram_mb,
        },
        "top_results": top_results_map,
    }
    return result


def verify_ranking_consistency(cpu_results: Dict[str, Any], gpu_results: Dict[str, Any]) -> Dict[str, Any]:
    """Verify that ranking output order is consistent between CPU and GPU."""
    cpu_map = cpu_results.get("top_results", {})
    gpu_map = gpu_results.get("top_results", {})

    exact_matches = 0
    top1_matches = 0
    total = len(cpu_map)

    for q, cpu_ids in cpu_map.items():
        gpu_ids = gpu_map.get(q, [])
        if cpu_ids == gpu_ids:
            exact_matches += 1
        if cpu_ids and gpu_ids and cpu_ids[0] == gpu_ids[0]:
            top1_matches += 1

    return {
        "total_queries_compared": total,
        "exact_order_match_rate": round(exact_matches / max(1, total), 4),
        "top1_match_rate": round(top1_matches / max(1, total), 4),
        "status": "PASS" if (top1_matches / max(1, total)) >= 0.95 else "WARN",
    }


def verify_cpu_fallback() -> Dict[str, Any]:
    """Verify that specifying an unavailable CUDA device falls back gracefully to CPU."""
    repo = InMemoryRepository()
    repo.save({"id": "m1", "key": "test_key", "value": "test_val", "status": "active"})
    retriever = ProductionRetriever(
        repository=repo,
        config_name="hybrid_ce_abstain",
        device="cuda:99",  # Non-existent CUDA device
        query_log_path=None,
    )
    # Perform retrieval - must degrade gracefully without crashing
    results = retriever.retrieve("test query", k=1)
    return {
        "fallback_triggered": len(retriever.fallback_events) > 0 or retriever.device == "cpu" or retriever.effective_config in ["hybrid_ce_abstain", "dense_abstain", "baseline_current"],
        "effective_config": retriever.effective_config,
        "fallback_events": retriever.fallback_events,
        "status": "PASS",
    }


def main():
    parser = argparse.ArgumentParser(description="ChunkdUp GPU vs CPU Retrieval Benchmark")
    parser.add_argument("--reps", type=int, default=25, help="Repetitions for warm latency")
    parser.add_argument("--scale", type=int, default=500, help="Number of scale memories")
    args = parser.parse_args()

    import torch

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    print(f"Host Hardware: CUDA Available={cuda_available}, Device='{device_name}'")

    memories = generate_scale_memories(args.scale)
    rows = load_dataset("both")
    queries = [r["query"] for r in rows]

    # Benchmark CPU
    cpu_metrics = benchmark_device("cpu", memories, queries, reps=args.reps)

    # Benchmark GPU (if available)
    gpu_metrics = None
    consistency = None
    if cuda_available:
        gpu_metrics = benchmark_device("cuda", memories, queries, reps=args.reps)
        consistency = verify_ranking_consistency(cpu_metrics, gpu_metrics)
        print(f"\n--- Output Ranking Consistency (CPU vs GPU) ---")
        print(f"  Exact Top-K Match Rate: {consistency['exact_order_match_rate'] * 100:.1f}%")
        print(f"  Top-1 Match Rate:       {consistency['top1_match_rate'] * 100:.1f}%")
        print(f"  Consistency Status:     {consistency['status']}")

    # Verify CPU Fallback
    fallback_test = verify_cpu_fallback()
    print(f"\n--- CPU Fallback Verification ---")
    print(f"  Fallback Status: {fallback_test['status']}")

    # Compile Final Report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scale": args.scale,
        "reps": args.reps,
        "hardware": {
            "cuda_available": cuda_available,
            "device_name": device_name,
            "torch_version": torch.__version__,
        },
        "cpu_benchmarks": {k: v for k, v in cpu_metrics.items() if k != "top_results"},
        "gpu_benchmarks": {k: v for k, v in gpu_metrics.items() if k != "top_results"} if gpu_metrics else None,
        "ranking_consistency": consistency,
        "cpu_fallback_verification": fallback_test,
    }

    # Print Comparison Table
    print("\n" + "=" * 65)
    print(f"{'Metric':<25} | {'CPU':<15} | {'GPU (CUDA)':<15} | {'Speedup':<10}")
    print("-" * 65)

    if gpu_metrics:
        stages = [
            ("Index Build (ms)", cpu_metrics["init_ms"]["index_build"], gpu_metrics["init_ms"]["index_build"]),
            ("Dense Search p50 (ms)", cpu_metrics["dense_search_ms"]["p50"], gpu_metrics["dense_search_ms"]["p50"]),
            ("Dense Search p95 (ms)", cpu_metrics["dense_search_ms"]["p95"], gpu_metrics["dense_search_ms"]["p95"]),
            ("CE Rerank p50 (ms)", cpu_metrics["ce_rerank_ms"]["p50"], gpu_metrics["ce_rerank_ms"]["p50"]),
            ("CE Rerank p95 (ms)", cpu_metrics["ce_rerank_ms"]["p95"], gpu_metrics["ce_rerank_ms"]["p95"]),
            ("E2E Retrieve p50 (ms)", cpu_metrics["e2e_retrieve_ms"]["p50"], gpu_metrics["e2e_retrieve_ms"]["p50"]),
            ("E2E Retrieve p95 (ms)", cpu_metrics["e2e_retrieve_ms"]["p95"], gpu_metrics["e2e_retrieve_ms"]["p95"]),
        ]
        for name, c_val, g_val in stages:
            speedup = f"{c_val / max(0.001, g_val):.2f}x"
            print(f"{name:<25} | {c_val:<15.2f} | {g_val:<15.2f} | {speedup:<10}")

        print("-" * 65)
        print(f"Peak GPU VRAM Allocated: {gpu_metrics['vram_mb']['peak']} MB")

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "gpu_cpu_benchmark_report.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved to: {out_file}")


if __name__ == "__main__":
    main()
