# experiments/retrieval/run_experiment.py
"""
Shared evaluation harness: run one (or all) experiment configurations against
the fixed dataset and fixture.

Usage:
    python experiments/retrieval/run_experiment.py --config dense --split dev
    python experiments/retrieval/run_experiment.py --all --split both

Outputs one JSON per config+split under experiments/retrieval/results/.
Abstention configs require a calibrated threshold in results/calibration.json
(produced by calibrate.py); missing calibration is reported as an error rather
than silently using an invented threshold.
"""
import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from ai_system_path import ai_system_path  # noqa: F401
from abstention import AbstentionGate
from config import BASELINE_CURRENT, CONFIGS, ExperimentConfig
from evaluate import K, aggregate, score_query
from fixture import BASE_MEMORIES, all_memories
from fusion import rrf_fuse
from retrievers import BM25Retriever, DenseRetriever, index_text

RESULTS_DIR = Path(__file__).parent / "results"
DATASET_PATH = Path(__file__).parent / "dataset.jsonl"

DEFAULT_ABSTAIN_FIELD = {
    ("dense", False): "_similarity",
    ("bm25", False): "_bm25_score",
    ("both", False): "_rrf_score",
    ("dense", True): "_rerank_score",
    ("both", True): "_rerank_score",
}


def load_dataset(split: str) -> List[Dict]:
    rows = [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if split in ("dev", "heldout"):
        return [r for r in rows if r["split"] == split]
    return rows


def load_calibration() -> Dict:
    path = RESULTS_DIR / "calibration.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


class CrossEncoderReranker:
    """Wraps the pretrained cross-encoder. Load failure is explicit.
    Accepts a preloaded CrossEncoder so callers can share one instance."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2", model=None):
        self.model_name = model_name
        if model is not None:
            self.model = model
        else:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(model_name)  # raises on failure; no silent fallback

    def rerank(self, query: str, candidates: List[Dict], limit: int) -> List[Dict]:
        if not candidates:
            return []
        pairs = [[query, index_text(c)] for c in candidates[:limit]]
        scores = self.model.predict(pairs)
        scored = []
        for candidate, score in zip(candidates[:limit], scores):
            candidate = dict(candidate)
            candidate["_rerank_score"] = float(score)
            scored.append(candidate)
        scored.sort(key=lambda c: c["_rerank_score"], reverse=True)
        return scored


class ExperimentPipeline:
    """One configuration over a memory store (defaults to the shared fixture).
    Accepts preloaded model instances for memory-constrained environments."""

    def __init__(self, config: ExperimentConfig, calibration: Optional[Dict] = None,
                 memories: Optional[List[Dict]] = None, dense_model=None, cross_model=None):
        self.config = config
        self.dense = DenseRetriever(memories, model=dense_model) if config.retriever in ("dense", "both") else None
        self.bm25 = BM25Retriever(memories) if config.retriever in ("bm25", "both") else None
        self.reranker = CrossEncoderReranker(model=cross_model) if config.rerank == "cross_encoder" else None

        self.abstain_score_field = config.abstain_score_field
        if config.abstain:
            if self.abstain_score_field is None:
                self.abstain_score_field = DEFAULT_ABSTAIN_FIELD[(config.retriever, config.rerank == "cross_encoder")]
            threshold = config.abstain_threshold
            if threshold is None:
                entry = (calibration or {}).get(config.name)
                if not entry or "threshold" not in entry:
                    raise RuntimeError(
                        f"Config '{config.name}' needs an abstention threshold; "
                        f"run calibrate.py first (no calibration entry found)."
                    )
                threshold = entry["threshold"]
            self.gate = AbstentionGate(self.abstain_score_field, threshold)
        else:
            self.gate = None

    def retrieve(self, query: str, k: int = 5) -> Tuple[List[Dict], Dict, List[Dict]]:
        """Returns (final_results, stage_timings_ms, ranked_candidates_for_trace)."""
        timings = {}
        cfg = self.config

        t0 = time.perf_counter()
        dense_results = self.dense.search(query, cfg.pool_size) if self.dense else []
        t1 = time.perf_counter()
        bm25_results = self.bm25.search(query, cfg.pool_size) if self.bm25 else []
        t2 = time.perf_counter()
        timings["retrieve_dense_ms"] = (t1 - t0) * 1000
        timings["retrieve_bm25_ms"] = (t2 - t1) * 1000

        if cfg.fusion == "rrf":
            candidates = rrf_fuse([dense_results, bm25_results], k=cfg.rrf_k, limit=cfg.pool_size)
        elif self.dense:
            candidates = dense_results
        else:
            candidates = bm25_results
        t3 = time.perf_counter()
        timings["fuse_ms"] = (t3 - t2) * 1000

        if self.reranker:
            candidates = self.reranker.rerank(query, candidates, limit=cfg.rerank_pool)
        t4 = time.perf_counter()
        timings["rerank_ms"] = (t4 - t3) * 1000

        abstained = False
        top_score = None
        gap = None
        if self.gate:
            decision = self.gate.decide(candidates)
            abstained = decision["abstained"]
            top_score = decision["top_score"]
            gap = decision["gap"]
            candidates = self.gate.filter(candidates)
        t5 = time.perf_counter()
        timings["abstain_ms"] = (t5 - t4) * 1000

        final = candidates[:k]
        timings["total_ms"] = (t5 - t0) * 1000
        return final, timings, candidates


class BaselinePipeline:
    """The current production path (MemoryRetriever over InMemoryRepository),
    kept as a labeled reference. Includes its exact-lookup fast path, entity
    candidates, weighted RRF with type/intent bonuses, and heuristic reranker."""

    def __init__(self, config: ExperimentConfig, memories: Optional[List[Dict]] = None):
        from chunkdup.in_memory_repository import InMemoryRepository
        from chunkdup.retrieval_class import MemoryRetriever

        self.config = config
        self.repo = InMemoryRepository()
        self.retriever = MemoryRetriever(repository=self.repo, reranker_mode="heuristic", enable_cache=False)
        for memory in (memories if memories is not None else BASE_MEMORIES):
            self.retriever.index_memory(dict(memory))
            self.repo.save(dict(memory))

    def retrieve(self, query: str, k: int = 5) -> Tuple[List[Dict], Dict, List[Dict]]:
        t0 = time.perf_counter()
        results = self.retriever.retrieve(query, k=k)
        elapsed = (time.perf_counter() - t0) * 1000
        timings = {"total_ms": elapsed}
        return results, timings, results


def run_config(config_name: str, split: str, k: int = 5) -> Dict:
    rows = load_dataset(split)
    calibration = load_calibration()

    if config_name == BASELINE_CURRENT:
        pipeline = BaselinePipeline(ExperimentConfig(name=BASELINE_CURRENT, retriever="both"))
    else:
        pipeline = ExperimentPipeline(CONFIGS[config_name], calibration)

    per_query = []
    records = []
    for row in rows:
        results, timings, ranked = pipeline.retrieve(row["query"], k=k)
        returned_ids = [r["id"] for r in results]
        expected = row["relevant_memory_ids"]
        sq = score_query(row, returned_ids)
        sq.update({
            "n_relevant_returned": len([i for i in returned_ids if i in set(expected)]),
            "n_irrelevant_returned": len([i for i in returned_ids if i not in set(expected)]),
            "abstained": not returned_ids,
        })
        per_query.append(sq)

        trace_candidates = []
        for rank, cand in enumerate(ranked[:10], start=1):
            trace_candidates.append({
                "rank": rank,
                "id": cand["id"],
                "value": cand.get("value"),
                "_similarity": round(cand["_similarity"], 4) if "_similarity" in cand else None,
                "_bm25_score": round(cand["_bm25_score"], 4) if "_bm25_score" in cand else None,
                "_rrf_score": round(cand["_rrf_score"], 6) if "_rrf_score" in cand else None,
                "_rrf_parts": cand.get("_rrf_parts"),
                "_rerank_score": round(cand["_rerank_score"], 4) if "_rerank_score" in cand else None,
            })
        records.append({
            "id": row["id"],
            "query": row["query"],
            "category": row["category"],
            "expected": expected,
            "returned": returned_ids,
            "returned_values": [r.get("value") for r in results],
            "abstained": not returned_ids,
            "top_score_field": pipeline.abstain_score_field if getattr(pipeline, "gate", None) else None,
            "metrics": {k2: sq[k2] for k2 in ("precision", "recall", "reciprocal_rank", "hit")},
            "timings_ms": {k2: round(v, 3) for k2, v in timings.items()},
            "candidates": trace_candidates,
        })

    summary = aggregate(per_query, rows)
    stage_means = {}
    for stage in ("retrieve_dense_ms", "retrieve_bm25_ms", "fuse_ms", "rerank_ms", "abstain_ms", "total_ms"):
        values = [r["timings_ms"][stage] for r in records if stage in r["timings_ms"]]
        if values:
            stage_means[stage] = {
                "mean": round(statistics.mean(values), 3),
                "p50": round(statistics.median(values), 3),
                "p95": round(sorted(values)[int(0.95 * (len(values) - 1))], 3),
            }

    return {
        "config": config_name,
        "split": split,
        "k": k,
        "summary": summary,
        "latency_ms": stage_means,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "per_query": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run retrieval experiment configurations.")
    parser.add_argument("--config", choices=[BASELINE_CURRENT] + list(CONFIGS.keys()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--split", choices=["dev", "heldout", "both"], default="dev")
    parser.add_argument("--k", type=int, default=K)
    args = parser.parse_args()

    if not args.all and not args.config:
        parser.error("Provide --config NAME or --all")

    names = [BASELINE_CURRENT] + list(CONFIGS.keys()) if args.all else [args.config]
    splits = ["dev", "heldout"] if args.split == "both" else [args.split]

    RESULTS_DIR.mkdir(exist_ok=True)
    failures = []
    for name in names:
        for split in splits:
            try:
                result = run_config(name, split, k=args.k)
            except Exception as exc:  # noqa: BLE001 - report failures honestly
                failures.append(f"{name}/{split}: {exc}")
                print(f"[FAIL] {name} [{split}]: {exc}")
                continue
            out_path = RESULTS_DIR / f"{name}.{split}.json"
            out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            s = result["summary"]
            print(
                f"[OK] {name} [{split}]  MRR={s['MRR']}  P@K={s['Precision@K']}  "
                f"R@K={s['Recall@K']}  HitRate={s['HitRate@K']}  "
                f"reject={s['unanswerable']['correct_rejection_rate']}  "
                f"-> {out_path.name}"
            )
    if failures:
        print("\nFailed runs (reported, not silently skipped):")
        for f in failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
