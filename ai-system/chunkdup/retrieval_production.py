# CLOSEOUT-EDIT-CHECK
# ai-system/chunkdup/retrieval_production.py
"""
Production retrieval pipeline — the hardened, selected configuration from
docs/round3-decision.md (hybrid_ce_abstain), with operational safety:

  - graceful degradation: cross-encoder failure or timeout falls back to the
    dense gate (dense_abstain); embedding-model failure falls back further to
    the frozen round-1 baseline path;
  - JSONL query/result logging for post-hoc drift inspection;
  - one-move rollback via CHUNKDUP_RETRIEVAL_CONFIG=baseline_current;
  - scale-aware gate recalibration (see recalibrate()).

The dense/BM25/RRF/gate implementations here intentionally mirror the
experiment modules in experiments/retrieval/ (round-1/2 evidence); tests in
tests/test_production_retrieval.py assert behavioral consistency between the
two. Scope note (round-3 brief): no caching layers, async queues, sharding,
load balancing, or multi-tenant isolation.
"""
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration selection (rollback switch)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = "hybrid_ce_abstain"
KNOWN_CONFIGS = ("hybrid_ce_abstain", "dense_abstain", "dense_abstain_ed", "baseline_current")

# Configs whose retrieval path includes the round-3 experiment-6 entity
# demotion (validated: pooled-92 MRR/P improved, zero per-query regressions).
# The FAILURE fallback (plain dense_abstain) keeps its round-3 semantics
# unchanged - demotion is a quality feature, not part of degradation.
ENTITY_DEMOTION_CONFIGS = ("hybrid_ce_abstain", "dense_abstain_ed")

# Round-1 dev-split calibrated thresholds (docs/retrieval-experiments.md §4).
# The CE threshold is a starting point ONLY — scale-aware recalibration
# (recalibrate()) is a hard shipping dependency per docs/round3-decision.md.
FROZEN_COSINE_THRESHOLD = 0.262475
FROZEN_CE_THRESHOLD = -10.947404

# CE rerank timeout, traced to a named benchmark (closeout round, part 2):
# the round-2 ISOLATED micro-benchmark (experiments/retrieval/bench_latency.py,
# results/scale/latency_bench_{12,100,500,1000}.json - the run that separated
# algorithmic cost from host noise via a hot-machine control), whose worst
# observed pool-bound median CE rerank latency is 452.6 ms (scale 500; scale
# 1000 re-bench: 453.2 ms). 2x that worst case = ~905 ms, rounded to 1.0 s.
# (The value was ORIGINALLY derived from the round-1 harness p95 of 264-268 ms
# in results/dense_ce.dev.json - a valid but less isolated basis; the
# recomputation from the isolated benchmark lands on the same 1.0 s.)
DEFAULT_CE_TIMEOUT_S = 1.0


def select_config() -> str:
    """Resolve the active configuration: explicit env var wins, else default."""
    configured = os.getenv("CHUNKDUP_RETRIEVAL_CONFIG", DEFAULT_CONFIG).strip()
    if configured not in KNOWN_CONFIGS:
        logger.warning(
            "Unknown CHUNKDUP_RETRIEVAL_CONFIG=%r; using default %r (known: %s)",
            configured, DEFAULT_CONFIG, KNOWN_CONFIGS,
        )
        return DEFAULT_CONFIG
    return configured


# ---------------------------------------------------------------------------
# Candidate-generation primitives (mirroring experiments/retrieval)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+.\-]*")
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "i", "my", "me", "you", "your", "it", "its", "of", "to", "in", "on",
    "for", "with", "and", "or", "do", "does", "did", "am",
    "what", "which", "who", "whose", "where", "when", "how", "why",
}


def _tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _index_text(memory: Dict[str, Any]) -> str:
    contextual = memory.get("contextual_text") or ""
    return (
        f"key: {memory.get('key', '')}\n"
        f"type: {memory.get('type', '')}\n"
        f"value: {memory.get('value', '')}\n"
        f"contextualized source: {contextual}"
    ).strip()


def _rrf_fuse(result_lists: List[List[Dict]], k: int = 60, limit: int = 20) -> List[Dict]:
    """Standard RRF: RRF(d) = sum over lists of 1/(k + rank_1based(d));
    dedup by id; absent from a list contributes nothing."""
    scores: Dict[str, float] = {}
    candidates: Dict[str, Dict] = {}
    for results in result_lists:
        for rank, candidate in enumerate(results, start=1):
            cid = candidate["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in candidates:
                candidates[cid] = dict(candidate)
    fused = []
    for cid, score in scores.items():
        candidate = candidates[cid]
        candidate["_rrf_score"] = score
        fused.append(candidate)
    fused.sort(key=lambda c: c["_rrf_score"], reverse=True)
    return fused[:limit]


# Identity-entity vocabulary (proper names only - "ChunkdUp", "Docker",
# "PostgreSQL", ...). Generic domain words ("database", "backend") are
# deliberately excluded: treating them as identities would demote correct
# attribute answers. Mirrors experiments/retrieval/entity_guard.py; kept in
# sync by tests/test_production_retrieval.py.
from .entity_extractor import EntityExtractor  # noqa: E402

_IDENTITY_VOCAB = set()
for _value in EntityExtractor.CUSTOM_ENTITIES.values():
    _IDENTITY_VOCAB.update(_tokenize(_value))


def _apply_entity_demotion(query: str, ranked: List[Dict], score_field: str) -> List[Dict]:
    """Round-3 experiment 6, production wiring (sink-below semantics).

    When the query names a stored identity entity AND asks about something
    beyond it, project_name-typed candidates whose value IS the entity sink
    below every non-demoted candidate (CE logits are unbounded, so a
    multiplicative penalty cannot cross the ~10-logit gap to irrelevant
    candidates - observed and replaced during round 3). Type-awareness is
    load-bearing: the correct Docker memory for "Where does ChunkdUp run?"
    also matches only on the entity token, but it is not the project memory.
    Mirrors experiments/retrieval/entity_guard.apply_entity_demotion.
    """
    query_tokens = set(_tokenize(query))
    identity = query_tokens & _IDENTITY_VOCAB
    non_entity = query_tokens - identity
    if not identity or not non_entity:
        return ranked

    demoted_flags = []
    for candidate in ranked:
        is_demoted = False
        if candidate.get("type") == "project_name":
            candidate_tokens = set(_tokenize(_index_text(candidate)))
            substantive = (query_tokens & candidate_tokens) - identity
            value_tokens = set(_tokenize(str(candidate.get("value", "")).lower()))
            if not substantive and value_tokens and value_tokens <= identity:
                is_demoted = True
        demoted_flags.append(is_demoted)

    if not any(demoted_flags):
        return ranked

    live_scores = [float(c.get(score_field, 0.0)) for c, d in zip(ranked, demoted_flags) if not d]
    sink_floor = (min(live_scores) - 1.0) if live_scores else 0.0

    adjusted = []
    sink_offset = 0
    for candidate, is_demoted in zip(ranked, demoted_flags):
        candidate = dict(candidate)
        if is_demoted:
            candidate[f"_pre_demotion_{score_field}"] = candidate.get(score_field)
            candidate[score_field] = sink_floor - 0.001 * sink_offset
            candidate["_entity_demoted"] = True
            sink_offset += 1
        adjusted.append(candidate)

    adjusted.sort(key=lambda c: c.get(score_field, 0.0), reverse=True)
    return adjusted


try:
    from .retrieval.grounding_gate import GroundingGate, GroundedRetrievalResult
except ImportError:
    from retrieval.grounding_gate import GroundingGate, GroundedRetrievalResult

try:
    from .entity.entity_graph import EntityGraph
except ImportError:
    try:
        from entity.entity_graph import EntityGraph
    except ImportError:
        EntityGraph = None

# Alias for backwards compatibility with legacy tests/code
_Gate = GroundingGate


# ---------------------------------------------------------------------------
# Model singletons (loaded once per process, lazily)
# ---------------------------------------------------------------------------

_MODEL_CACHE: Dict[str, Any] = {}


def _get_sentence_transformer():
    if "st" not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        env_source = os.getenv("CHUNKDUP_EMBEDDING_MODEL")
        if env_source:
            source = env_source
        else:
            models_dir = os.getenv("CHUNKDUP_MODELS_DIR")
            local_path = Path(__file__).resolve().parent.parent.parent / "models" / "all-MiniLM-L6-v2"
            cwd_path = Path.cwd() / "models" / "all-MiniLM-L6-v2"
            if models_dir and (Path(models_dir) / "all-MiniLM-L6-v2").exists():
                source = str(Path(models_dir) / "all-MiniLM-L6-v2")
            elif local_path.exists():
                source = str(local_path)
            elif cwd_path.exists():
                source = str(cwd_path)
            else:
                source = "all-MiniLM-L6-v2"
        logger.info("Loading embedding model from source: %s", source)
        orig_threads = None
        try:
            import torch
            orig_threads = torch.get_num_threads()
            torch.set_num_threads(1)
        except Exception:
            pass
        try:
            _MODEL_CACHE["st"] = SentenceTransformer(source)
        finally:
            if orig_threads is not None:
                try:
                    torch.set_num_threads(orig_threads)
                except Exception:
                    pass
        _MODEL_CACHE["st_source"] = source
    return _MODEL_CACHE["st"]


def _get_cross_encoder():
    if "ce" not in _MODEL_CACHE:
        from sentence_transformers import CrossEncoder

        model_name = os.getenv("CHUNKDUP_CE_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")
        models_dir = os.getenv("CHUNKDUP_MODELS_DIR")
        if models_dir and (Path(models_dir) / "ms-marco-MiniLM-L-12-v2").exists():
            model_name = str(Path(models_dir) / "ms-marco-MiniLM-L-12-v2")
        logger.info("Loading cross-encoder model: %s", model_name)
        orig_threads = None
        try:
            import torch
            orig_threads = torch.get_num_threads()
            torch.set_num_threads(1)
        except Exception:
            pass
        try:
            _MODEL_CACHE["ce"] = CrossEncoder(model_name)
        finally:
            if orig_threads is not None:
                try:
                    torch.set_num_threads(orig_threads)
                except Exception:
                    pass
    return _MODEL_CACHE["ce"]


# ---------------------------------------------------------------------------
# ProductionRetriever
# ---------------------------------------------------------------------------


class ProductionRetriever:
    """The hardened production retrieval path.

    Effective configuration resolution order:
      1. explicit `config_name` argument
      2. CHUNKDUP_RETRIEVAL_CONFIG environment variable
      3. DEFAULT_CONFIG (hybrid_ce_abstain) per docs/round3-decision.md

    Degradation chain: hybrid_ce_abstain -> dense_abstain (on CE load error,
    inference error, or timeout) -> baseline_current (on embedding-model
    failure). Every degradation is logged, never silent.
    """

    def __init__(
        self,
        repository,
        config_name: Optional[str] = None,
        ce_timeout_s: Optional[float] = None,
        query_log_path: Optional[str] = None,
        fast_mode: bool = False,
    ):
        self.repository = repository
        # Selection precedence: explicit config_name > fast_mode request >
        # CHUNKDUP_RETRIEVAL_CONFIG env var > default. fast_mode is a PLANNED
        # choice of the exp-6-enhanced fast path (dense_abstain_ed, ~10 ms,
        # no cross-encoder) and is independent of the failure-driven
        # degradation chain (CE failure -> dense_abstain -> baseline_current).
        if config_name:
            self.config_name = config_name
        elif fast_mode:
            self.config_name = "dense_abstain_ed"
        else:
            self.config_name = select_config()
        self.ce_timeout_s = float(
            ce_timeout_s if ce_timeout_s is not None
            else os.getenv("CHUNKDUP_CE_TIMEOUT_S", DEFAULT_CE_TIMEOUT_S)
        )
        self.query_log_path = query_log_path
        env_log = os.getenv("CHUNKDUP_QUERY_LOG")
        if env_log == "off":
            self.query_log_path = None
        elif env_log:
            self.query_log_path = env_log

        self.effective_config = self.config_name
        self.fallback_events: List[Dict[str, Any]] = []
        self._index_signature = None
        self._dense_memories: List[Dict] = []
        self._dense_matrix = None
        self._bm25 = None
        self._ce = None
        self._ce_failed = False  # sticky: stop retrying a model that keeps failing
        self.gate = _Gate("_rerank_score", FROZEN_CE_THRESHOLD)
        self._dense_gate = _Gate("_similarity", FROZEN_COSINE_THRESHOLD)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chunkdup-ce")
        self.last_calibration_size = None
        self.last_calibration = None
        self.demote_enabled = self.config_name in ENTITY_DEMOTION_CONFIGS
        self.entity_graph = EntityGraph() if EntityGraph is not None else None
        self.entity_expansion_enabled = os.getenv("CHUNKDUP_ENTITY_EXPANSION", "1") == "1"

        if self.config_name == "baseline_current":
            from .retrieval_class import MemoryRetriever

            self._legacy = MemoryRetriever(repository=self.repository, enable_cache=False)
            self.effective_config = "baseline_current"
        else:
            self._legacy = None

    # -- index maintenance ---------------------------------------------------

    def _repository_signature(self):
        memories = self.repository.get_all()
        if not memories:
            return (0, None)
        latest = max(m.get("updated_at") or "" for m in memories)
        return (len(memories), latest)

    def _ensure_index(self) -> None:
        if self.effective_config == "baseline_current":
            return
        signature = self._repository_signature()
        if signature == self._index_signature:
            return
        active = self.repository.get_all()
        for memory in active:
            memory.setdefault(
                "contextual_text",
                ContextualTextBuilder.build(memory),
            )
        self._dense_memories = active
        self._dense_matrix = None
        self._bm25 = None
        if active:
            if self.entity_graph and self.entity_expansion_enabled:
                self.entity_graph.index_repository(self.repository)
            try:
                import numpy as np
                from rank_bm25 import BM25Okapi

                corpus = [_tokenize(_index_text(m)) for m in active]
                self._bm25 = BM25Okapi(corpus) if any(corpus) else None
                texts = [_index_text(m) for m in active]
                st = _get_sentence_transformer()
                vectors = st.encode(texts, normalize_embeddings=True)
                self._dense_matrix = vectors
            except Exception as exc:  # embedding/BM25 unavailable -> degrade
                self._degrade("embedding_model_failure", str(exc))
        self._index_signature = signature

    def _degrade(self, reason: str, detail: str) -> None:
        if reason == "embedding_model_failure":
            # Without embeddings the dense leg is dead: skip straight to the
            # model-free baseline instead of stopping at dense_abstain.
            new_config = "baseline_current"
        else:
            order = ["hybrid_ce_abstain", "dense_abstain", "baseline_current"]
            idx = order.index(self.effective_config) if self.effective_config in order else 0
            new_config = order[min(idx + 1, len(order) - 1)]
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "detail": detail[:500],
            "from_config": self.effective_config,
            "to_config": new_config,
        }
        self.fallback_events.append(event)
        logger.warning("Retrieval degradation: %s (%s): %s -> %s",
                       reason, detail, self.effective_config, new_config)
        self.effective_config = new_config
        if reason == "embedding_model_failure":
            from .retrieval_class import MemoryRetriever

            self._legacy = MemoryRetriever(repository=self.repository, enable_cache=False)
            self._dense_matrix = None
            self._bm25 = None

    # -- legs ------------------------------------------------------------------

    def _dense_search(self, query: str, limit: int) -> List[Dict]:
        if self._dense_matrix is None:
            return []
        import numpy as np

        st = _get_sentence_transformer()
        q_vec = st.encode(query, normalize_embeddings=True)
        sims = self._dense_matrix @ q_vec
        order = np.argsort(-sims)[:limit]
        out = []
        for i in order:
            candidate = dict(self._dense_memories[int(i)])
            candidate["_similarity"] = float(sims[int(i)])
            out.append(candidate)
        return out

    def _bm25_search(self, query: str, limit: int) -> List[Dict]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        paired = sorted(
            zip(self._dense_memories, scores), key=lambda p: p[1], reverse=True
        )
        out = []
        for memory, score in paired[:limit]:
            if score <= 0.0:
                break
            candidate = dict(memory)
            candidate["_bm25_score"] = float(score)
            out.append(candidate)
        return out

    def _cross_encoder_rerank(self, query: str, candidates: List[Dict], limit: int) -> List[Dict]:
        """CE rerank with explicit timeout. Raises on failure/timeout so the
        caller can fall back; never falls back silently."""
        if self._ce_failed:
            raise RuntimeError("cross-encoder previously failed; sticky failure")
        if self._ce is None:
            self._ce = _get_cross_encoder()  # may raise -> handled by caller
        pairs = [[query, _index_text(c)] for c in candidates[:limit]]
        future = self._executor.submit(self._ce.predict, pairs)
        scores = future.result(timeout=self.ce_timeout_s)  # raises TimeoutError
        scored = []
        for candidate, score in zip(candidates[:limit], scores):
            candidate = dict(candidate)
            candidate["_rerank_score"] = float(score)
            scored.append(candidate)
        scored.sort(key=lambda c: c["_rerank_score"], reverse=True)
        return scored

    # -- public API --------------------------------------------------------------

    def _collect_ranked(self, query: str, k: int):
        """Candidate pipeline WITHOUT the relevance gate.
        Returns (ranked_candidates, score_field); (None, None) on the
        baseline path (no gate applies)."""
        self._ensure_index()

        if self.effective_config == "baseline_current":
            return None, None

        # Candidate pool mirrors the validated experiment configuration
        # (ExperimentConfig.pool_size=20, rerank_pool=20): every round-1/2/3
        # number was produced with a 20-candidate pool, so the production
        # path must see the same candidates to reproduce those numbers.
        pool = max(20, 2 * k)
        try:
            dense_results = self._dense_search(query, pool)
        except Exception as exc:
            self._degrade("embedding_model_failure", str(exc))
            dense_results = []
        if self.effective_config == "baseline_current":
            return None, None

        if self.effective_config == "hybrid_ce_abstain":
            try:
                bm25_results = self._bm25_search(query, pool)
            except Exception as exc:
                logger.warning("BM25 leg failed (%s); continuing dense-only", exc)
                bm25_results = []
            fused = _rrf_fuse([dense_results, bm25_results], k=60, limit=pool)

            # Conservative bounded 1-hop entity expansion for multi-hop evidence
            if self.entity_expansion_enabled and self.entity_graph:
                expanded = self.entity_graph.expand_candidates(
                    query=query,
                    initial_candidates=fused,
                    repository=self.repository,
                    max_expansion=5,
                )
                seen_ids = {c.get("id") for c in fused}
                for exp in expanded:
                    if exp.get("id") not in seen_ids:
                        fused.append(exp)
                        seen_ids.add(exp.get("id"))

            try:
                ce_pool_limit = min(25, len(fused))
                return self._cross_encoder_rerank(query, fused, limit=ce_pool_limit), "_rerank_score"
            except Exception as exc:
                self._ce_failed = True  # sticky: never retry a failed CE mid-process
                self._degrade(
                    "timeout" if isinstance(exc, TimeoutError) else "cross_encoder_failure",
                    str(exc),
                )
                return fused, "_similarity"
        return dense_results, "_similarity"

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        t0 = time.perf_counter()
        ranked, score_field = self._collect_ranked(query, k)

        if ranked is None:
            results = self._legacy.retrieve(query, k=k)
            self._log_query(query, results, [], (time.perf_counter() - t0) * 1000, not results)
            return results

        # Round-3 experiment 6, production wiring (see ENTITY_DEMOTION_CONFIGS)
        if self.demote_enabled and (
            (score_field == "_rerank_score" and self.effective_config == "hybrid_ce_abstain")
            or (score_field == "_similarity" and self.effective_config == "dense_abstain_ed")
        ):
            ranked = _apply_entity_demotion(query, ranked, score_field)

        gate = self.gate if score_field == "_rerank_score" else self._dense_gate
        store_size = len(self._dense_memories) if self._dense_memories else len(self.repository.get_all())
        grounded = gate.filter_and_ground(query, ranked, store_size=store_size, k=k)
        final = [g.to_dict() for g in grounded]
        latency_ms = (time.perf_counter() - t0) * 1000
        self._log_query(query, final, ranked, latency_ms, not final)
        return final

    def retrieve_grounded(self, query: str, k: int = 5) -> List[GroundedRetrievalResult]:
        """Retrieve memories matching a query as typed GroundedRetrievalResult objects."""
        t0 = time.perf_counter()
        ranked, score_field = self._collect_ranked(query, k)

        if ranked is None:
            legacy = self._legacy.retrieve(query, k=k)
            return [
                GroundedRetrievalResult(
                    id=str(r.get("id") or r.get("key") or ""),
                    memory=r,
                    score=1.0,
                    score_field="legacy",
                    source_turns=[],
                    confidence=1.0,
                    is_sufficient=True,
                )
                for r in legacy
            ]

        if self.demote_enabled and (
            (score_field == "_rerank_score" and self.effective_config == "hybrid_ce_abstain")
            or (score_field == "_similarity" and self.effective_config == "dense_abstain_ed")
        ):
            ranked = _apply_entity_demotion(query, ranked, score_field)

        gate = self.gate if score_field == "_rerank_score" else self._dense_gate
        store_size = len(self._dense_memories) if self._dense_memories else len(self.repository.get_all())
        return gate.filter_and_ground(query, ranked, store_size=store_size, k=k)

    # -- scale-aware gate recalibration (part 2.4) --------------------------------

    def recalibrate(self, sample: List[Dict[str, Any]], k: int = 5) -> Dict[str, Any]:
        """Recompute the abstention threshold from a labeled, dev-representative
        sample against the CURRENT store state.

        This is the hard shipping dependency from docs/round3-decision.md: the
        frozen CE threshold (calibrated on a 12-memory dev set) loses rejection
        accuracy as the store grows (round-2 risk 1: rejection 0.895 at 12
        memories -> 0.737 by ~500; recalibration restored 1.00 at -2.1..-2.3).

        Args:
            sample: [{"query": str, "relevant_memory_ids": [ids], "answerable": bool}]
                A small set of representative queries for THIS store, labeled
                by hand the same way as the evaluation dataset.
        Returns:
            {"changed": bool, "threshold": float|None, "score_field": str,
             "objective": float|None, "reason": str}
        Schedule guidance: call on startup, and re-call whenever
        should_recalibrate() turns True (store grew ~50% past the last
        calibration). Drift is measurable from ~100 memories and material by
        ~500 (round-2 scale data), so do not let the store more than double
        past the calibration point.
        """
        self._ensure_index()
        if self.effective_config == "baseline_current":
            return {"changed": False, "reason": "no gate on the baseline path", "threshold": None}
        if not sample:
            return {"changed": False, "reason": "empty sample", "threshold": None}

        config_before = self.effective_config
        score_field = "_rerank_score" if config_before == "hybrid_ce_abstain" else "_similarity"

        collected = []
        for row in sample:
            ranked, field = self._collect_ranked(row["query"], k)
            if ranked is None:
                return {"changed": False, "reason": "no gate on the baseline path", "threshold": None}
            if field != score_field or self.effective_config != config_before:
                return {"changed": False,
                        "reason": f"degraded during calibration ({config_before} -> {self.effective_config})",
                        "threshold": None}
            collected.append((row, ranked))

        observed = sorted({float(ranked[0].get(score_field, 0.0)) for _row, ranked in collected if ranked})
        if not observed:
            return {"changed": False, "reason": "no candidates observed for any sample query", "threshold": None}

        def evaluate(threshold: float):
            ans_hits = ans_total = unans_reject = 0
            for row, ranked in collected:
                scores = [float(c.get(score_field, 0.0)) for c in ranked]
                if not scores or scores[0] < threshold:
                    ids = []
                else:
                    ids = [c["id"] for c in ranked if float(c.get(score_field, 0.0)) >= threshold]
                expected = set(row["relevant_memory_ids"])
                if row["answerable"]:
                    ans_total += 1
                    if any(i in expected for i in ids):
                        ans_hits += 1
                else:
                    unans_reject += 1 if not ids else 0
            n_ans = max(1, ans_total)
            n_unans = max(1, sum(1 for row, _ in collected if not row["answerable"]))
            hit_rate = ans_hits / ans_total if ans_total else 0.0
            reject_rate = unans_reject / n_unans
            return 0.5 * hit_rate + 0.5 * reject_rate, hit_rate, reject_rate

        # Sweep observed top-scores + midpoints + reject-all, ties -> lower threshold.
        thresholds = set(observed)
        for a, b in zip(observed, observed[1:]):
            thresholds.add((a + b) / 2)
        thresholds.add(observed[-1] + 1e-6)
        thresholds.add(min(0.0, observed[0] - 1e-6))

        best_threshold, best_objective, best_hit, best_reject = None, -1.0, 0.0, 0.0
        for t in sorted(thresholds):
            objective, hit_rate, reject_rate = evaluate(t)
            if objective > best_objective or (
                objective == best_objective and best_threshold is not None and t < best_threshold
            ):
                best_threshold, best_objective, best_hit, best_reject = t, objective, hit_rate, reject_rate

        old_threshold = self.gate.threshold if score_field == "_rerank_score" else self._dense_gate.threshold
        if score_field == "_rerank_score":
            self.gate.threshold = best_threshold
            self.gate.base_threshold = best_threshold
        else:
            self._dense_gate.threshold = best_threshold
            self._dense_gate.base_threshold = best_threshold
        self.last_calibration_size = len(self.repository.get_all())
        self.last_calibration = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "score_field": score_field,
            "old_threshold": old_threshold,
            "threshold": best_threshold,
            "objective": round(best_objective, 4),
            "sample_size": len(sample),
        }
        logger.info("Gate recalibrated: %s %.6f -> %.6f (objective %.4f)",
                    score_field, old_threshold, best_threshold, best_objective)
        return {
            "changed": best_threshold != old_threshold,
            "threshold": best_threshold,
            "old_threshold": old_threshold,
            "score_field": score_field,
            "objective": round(best_objective, 4),
            "answerable_hit_rate": round(best_hit, 4),
            "unanswerable_reject_rate": round(best_reject, 4),
        }

    def should_recalibrate(self) -> bool:
        """True when the store has grown ~50% past the last calibration size
        (or nothing was ever calibrated). Round-2 drift data: measurable from
        ~100 memories, material by ~500 - so recalibrate before doubling."""
        if self.effective_config == "baseline_current":
            return False
        if self.last_calibration_size is None:
            return True
        return len(self.repository.get_all()) >= self.last_calibration_size * 1.5

    # -- logging (part 2.2) ------------------------------------------------------

    def _log_query(self, query, final, ranked, latency_ms, abstained) -> None:
        try:
            if not self.query_log_path:
                return
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query": query,
                "config": self.effective_config,
                "requested_config": self.config_name,
                "returned_ids": [r.get("id") for r in final],
                "top_scores": [
                    {"id": c.get("id"),
                     "similarity": round(c["_similarity"], 4) if "_similarity" in c else None,
                     "bm25": round(c["_bm25_score"], 4) if "_bm25_score" in c else None,
                     "rrf": round(c["_rrf_score"], 6) if "_rrf_score" in c else None,
                     "rerank": round(c["_rerank_score"], 4) if "_rerank_score" in c else None}
                    for c in ranked[:5]
                ],
                "latency_ms": round(latency_ms, 3),
                "abstained": bool(abstained),
                "fallback_events": list(self.fallback_events),
            }
            path = Path(self.query_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:  # logging must never break retrieval
            logger.warning("Query log write failed: %s", exc)


class ContextualTextBuilder:
    """Mirrors chunkdup.retrieval.contextual.ContextualRetriever (kept local to
    avoid a circular import with the legacy retrieval_class)."""

    @staticmethod
    def build(memory: Dict[str, Any]) -> str:
        memory_type = memory.get("type", "unknown")
        key = memory.get("key", "attribute")
        value = memory.get("value", "")
        topic = memory.get("conversation_topic", "general user context")
        source = memory.get("source", "user preference")
        return (
            f"This memory is a {memory_type} regarding '{key}' from a conversation about {topic}. "
            f"Source domain: {source}. The recorded user context state is: \"{value}\"."
        ).strip()
