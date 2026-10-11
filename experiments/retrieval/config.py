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
    entity_demote: bool = False    # round-3 experiment 6 (type-aware entity scoring)
    entity_guard: bool = False     # round-3 experiment 7 (CE entity-guard)
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
    # Round-3 experiments (entity-aware scoring). Thresholds are the SAME
    # round-1 dev-calibrated values as the parent configs (results/calibration.json):
    # the entity stages reorder candidates but do not change gate score semantics.
    "hybrid_ce_abstain_ed": ExperimentConfig(
        name="hybrid_ce_abstain_ed", retriever="both", fusion="rrf", rerank="cross_encoder", abstain=True,
        abstain_score_field="_rerank_score", abstain_threshold=-10.947404, entity_demote=True,
        notes="Experiment 6: hybrid_ce_abstain + type-aware entity demotion (round-2 risk-5 FP pattern).",
    ),
    "dense_abstain_ed": ExperimentConfig(
        name="dense_abstain_ed", retriever="dense", abstain=True,
        abstain_score_field="_similarity", abstain_threshold=0.262475, entity_demote=True,
        notes="Experiment 6 on the fast-path config: dense_abstain + type-aware entity demotion.",
    ),
    "hybrid_ce_abstain_eg": ExperimentConfig(
        name="hybrid_ce_abstain_eg", retriever="both", fusion="rrf", rerank="cross_encoder", abstain=True,
        abstain_score_field="_rerank_score", abstain_threshold=-10.947404, entity_guard=True,
        notes="Experiment 7: hybrid_ce_abstain + CE entity-guard (CE may not demote fused top-3 on entity queries).",
    ),
    "hybrid_ce_abstain_ed_eg": ExperimentConfig(
        name="hybrid_ce_abstain_ed_eg", retriever="both", fusion="rrf", rerank="cross_encoder", abstain=True,
        abstain_score_field="_rerank_score", abstain_threshold=-10.947404, entity_demote=True, entity_guard=True,
        notes="Experiments 6+7 combined on the default config.",
    ),
}

# Legacy reference: the exact current production path (keyword + entity +
# weighted RRF with type/intent bonuses + heuristic reranker + adaptive k).
BASELINE_CURRENT = "baseline_current"

ALL_CONFIG_NAMES = [BASELINE_CURRENT] + list(CONFIGS.keys())
