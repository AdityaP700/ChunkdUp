# experiments/retrieval/config.py
"""
Experiment configurations. Each named config is one evaluation pipeline over
the shared dataset + fixture. The 8 spec configurations are listed in
CONFIGS; baseline_current is the existing production path kept as a labeled
reference.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExperimentConfig:
    name: str
    retriever: str                 # "dense" | "bm25" | "both"
    fusion: str = "none"           # "none" | "rrf"
    rerank: str = "none"           # "none" | "cross_encoder"
    abstain: bool = False
    pool_size: int = 20            # per-leg candidate pool
    rrf_k: int = 60                # RRF smoothing constant
    rerank_pool: int = 20          # candidates passed to the cross-encoder
    k: int = 5                     # final result cap
    abstain_threshold: Optional[float] = None
    abstain_score_field: Optional[str] = None  # defaults derived from pipeline
    notes: str = ""


CONFIGS = {
    "dense": ExperimentConfig(
        name="dense", retriever="dense",
        notes="Approach 1: dense retrieval only (local all-MiniLM-L6-v2, cosine).",
    ),
    "bm25": ExperimentConfig(
        name="bm25", retriever="bm25",
        notes="Approach 2: BM25 keyword retrieval only.",
    ),
    "hybrid_rrf": ExperimentConfig(
        name="hybrid_rrf", retriever="both", fusion="rrf",
        notes="Approach 3: dense + BM25 fused with standard RRF (k=60).",
    ),
    "dense_ce": ExperimentConfig(
        name="dense_ce", retriever="dense", rerank="cross_encoder",
        notes="Approach 4a: dense candidates reranked by ms-marco-MiniLM-L-12-v2 cross-encoder.",
    ),
    "hybrid_ce": ExperimentConfig(
        name="hybrid_ce", retriever="both", fusion="rrf", rerank="cross_encoder",
        notes="Approach 4b: hybrid RRF candidates reranked by cross-encoder.",
    ),
    "dense_abstain": ExperimentConfig(
        name="dense_abstain", retriever="dense", abstain=True,
        abstain_score_field="_similarity",
        notes="Approach 5a: dense + relevance threshold (cosine semantics), calibrated on dev.",
    ),
    "hybrid_abstain": ExperimentConfig(
        name="hybrid_abstain", retriever="both", fusion="rrf", abstain=True,
        abstain_score_field="_rrf_score",
        notes="Approach 5b: hybrid RRF + relevance threshold (RRF semantics), calibrated on dev.",
    ),
    "hybrid_ce_abstain": ExperimentConfig(
        name="hybrid_ce_abstain", retriever="both", fusion="rrf", rerank="cross_encoder", abstain=True,
        abstain_score_field="_rerank_score",
        notes="Approach 5c: hybrid + cross-encoder + relevance threshold (cross-encoder logit semantics), calibrated on dev.",
    ),
}

# Legacy reference: the exact current production path (keyword + entity +
# weighted RRF with type/intent bonuses + heuristic reranker + adaptive k).
BASELINE_CURRENT = "baseline_current"

ALL_CONFIG_NAMES = [BASELINE_CURRENT] + list(CONFIGS.keys())
