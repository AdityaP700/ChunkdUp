# tests/test_production_retrieval.py
"""Tests for the hardened production retrieval path (round-3 part 2).

P2.1: graceful degradation - cross-encoder load failure, inference failure,
and timeout must fall back to the dense gate (dense_abstain) and embedding
failures to the frozen baseline, with every fallback logged, never silent.
"""
import json
import os
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "ai-system"))

import chunkdup.retrieval_production as rp  # noqa: E402
from chunkdup.in_memory_repository import InMemoryRepository  # noqa: E402
from chunkdup.retrieval_production import ProductionRetriever  # noqa: E402


def make_repo():
    repo = InMemoryRepository()
    memories = [
        {"id": "m1", "type": "programming_language", "key": "language", "value": "Python", "conversation_topic": "tech stack", "status": "active"},
        {"id": "m9", "type": "tool", "key": "database", "value": "PostgreSQL with pgvector", "conversation_topic": "backend storage infrastructure", "status": "active"},
        {"id": "m11", "type": "environment", "key": "runtime_environment", "value": "Docker", "conversation_topic": "ChunkdUp runtime environment", "status": "active"},
        {"id": "m12", "type": "tool", "key": "embedding_model", "value": "all-MiniLM-L6-v2", "conversation_topic": "vector search stack", "status": "active"},
    ]
    for memory in memories:
        repo.save(dict(memory))
    return repo


class TestProductionFallback(unittest.TestCase):
    def test_normal_path_returns_relevant_result(self):
        retriever = ProductionRetriever(make_repo(), config_name="hybrid_ce_abstain")
        results = retriever.retrieve("What database does ChunkdUp use?", k=5)
        self.assertEqual(retriever.effective_config, "hybrid_ce_abstain")
        self.assertTrue(results, "hybrid pipeline abstained on an answerable store")
        # The database memory must be returned; its RANK is the known CE
        # jargon behavior (round-2 q46) addressed by experiment 7, not here.
        self.assertIn("m9", [r["id"] for r in results])

    def test_ce_load_failure_falls_back_to_dense(self):
        retriever = ProductionRetriever(make_repo(), config_name="hybrid_ce_abstain")
        with mock.patch.object(rp, "_get_cross_encoder", side_effect=RuntimeError("model files missing")):
            results = retriever.retrieve("What database does ChunkdUp use?", k=5)
        self.assertEqual(retriever.effective_config, "dense_abstain")
        self.assertTrue(retriever.fallback_events)
        self.assertEqual(retriever.fallback_events[-1]["reason"], "cross_encoder_failure")
        self.assertTrue(results, "fallback path returned nothing for an answerable store")
        self.assertTrue(all("_similarity" in r for r in results), "fallback results must come from the dense gate")

    def test_ce_timeout_falls_back(self):
        class HangingCE:
            def predict(self, pairs):
                time.sleep(0.75)
                return [0.0] * len(pairs)

        retriever = ProductionRetriever(make_repo(), config_name="hybrid_ce_abstain", ce_timeout_s=0.15)
        with mock.patch.object(rp, "_get_cross_encoder", return_value=HangingCE()):
            t0 = time.perf_counter()
            results = retriever.retrieve("What database does ChunkdUp use?", k=5)
            elapsed = time.perf_counter() - t0
        self.assertEqual(retriever.effective_config, "dense_abstain")
        self.assertEqual(retriever.fallback_events[-1]["reason"], "timeout")
        self.assertLess(elapsed, 3.0, "timeout fallback should not block on the hung model")
        self.assertTrue(results, "fallback path returned nothing for an answerable store")

    def test_failure_is_sticky_no_retry(self):
        retriever = ProductionRetriever(make_repo(), config_name="hybrid_ce_abstain")
        with mock.patch.object(rp, "_get_cross_encoder", side_effect=RuntimeError("broken")):
            retriever.retrieve("What database does ChunkdUp use?", k=5)
        self.assertEqual(retriever.effective_config, "dense_abstain")
        with mock.patch.object(rp.ProductionRetriever, "_cross_encoder_rerank") as ce_mock:
            retriever.retrieve("What editor do I prefer?", k=5)
            ce_mock.assert_not_called()
        self.assertEqual(retriever.effective_config, "dense_abstain")

    def test_embedding_failure_degrades_to_baseline(self):
        retriever = ProductionRetriever(make_repo(), config_name="hybrid_ce_abstain")
        with mock.patch.object(rp, "_get_sentence_transformer", side_effect=RuntimeError("torch missing")):
            results = retriever.retrieve("What database does ChunkdUp use?", k=5)
        self.assertEqual(retriever.effective_config, "baseline_current")
        self.assertEqual(retriever.fallback_events[-1]["reason"], "embedding_model_failure")
        self.assertTrue(results, "baseline fallback returned nothing for an answerable store")

    def test_empty_store_returns_empty(self):
        retriever = ProductionRetriever(InMemoryRepository(), config_name="hybrid_ce_abstain")
        self.assertEqual(retriever.retrieve("anything at all", k=5), [])


class TestQueryLogging(unittest.TestCase):
    """P2.2: JSONL query/result logging for post-hoc drift inspection."""

    def setUp(self):
        import tempfile

        self.tmp = tempfile.mkdtemp()
        self.log_path = __import__("os").path.join(self.tmp, "retrieval_log.jsonl")

    def _read_log(self):
        with open(self.log_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_log_written_with_required_fields(self):
        retriever = ProductionRetriever(make_repo(), config_name="hybrid_ce_abstain",
                                        query_log_path=self.log_path)
        retriever.retrieve("What database does ChunkdUp use?", k=5)
        retriever.retrieve("sourdough bread baking", k=5)  # expect abstention

        records = self._read_log()
        self.assertEqual(len(records), 2)
        for record in records:
            for key in ("timestamp", "query", "config", "requested_config",
                        "returned_ids", "top_scores", "latency_ms", "abstained",
                        "fallback_events"):
                self.assertIn(key, record)
        self.assertEqual(records[0]["query"], "What database does ChunkdUp use?")
        self.assertIn("m9", records[0]["returned_ids"])
        self.assertFalse(records[0]["abstained"])
        self.assertTrue(records[0]["top_scores"], "scores must be recorded")
        self.assertTrue(records[0]["timestamp"] <= records[1]["timestamp"])
        self.assertTrue(records[1]["abstained"], "no-overlap query must be logged as abstained")

    def test_log_disabled_by_env(self):
        retriever = ProductionRetriever(make_repo(), config_name="hybrid_ce_abstain",
                                        query_log_path=self.log_path)
        with mock.patch.dict(__import__("os").environ, {"CHUNKDUP_QUERY_LOG": "off"}):
            retriever = ProductionRetriever(make_repo(), config_name="hybrid_ce_abstain",
                                            query_log_path=self.log_path)
            retriever.retrieve("What database does ChunkdUp use?", k=5)
        self.assertFalse(__import__("os").path.exists(self.log_path))

    def test_log_records_fallback_events(self):
        retriever = ProductionRetriever(make_repo(), config_name="hybrid_ce_abstain",
                                        query_log_path=self.log_path)
        with mock.patch.object(rp, "_get_cross_encoder", side_effect=RuntimeError("broken")):
            retriever.retrieve("What database does ChunkdUp use?", k=5)
        record = self._read_log()[0]
        self.assertEqual(record["config"], "dense_abstain")
        self.assertEqual(record["requested_config"], "hybrid_ce_abstain")
        self.assertTrue(record["fallback_events"])

    def test_memory_writes_log_to_data_dir(self):
        import tempfile

        from chunkdup import Memory

        data_dir = tempfile.mkdtemp()
        memory = Memory(store="memory", data_dir=data_dir)
        memory.remember("I use Python")
        memory.retrieve("What language do I use?")
        log_path = __import__("os").path.join(data_dir, "retrieval_log.jsonl")
        self.assertTrue(__import__("os").path.exists(log_path))
        record = json.loads(open(log_path, encoding="utf-8").readline())
        self.assertEqual(record["query"], "What language do I use?")


class TestRollback(unittest.TestCase):
    """P2.3: one-move rollback via CHUNKDUP_RETRIEVAL_CONFIG."""

    def test_env_rollback_forces_baseline(self):
        repo = make_repo()
        with mock.patch.dict(__import__("os").environ, {"CHUNKDUP_RETRIEVAL_CONFIG": "baseline_current"}):
            retriever = ProductionRetriever(repo)
        self.assertEqual(retriever.config_name, "baseline_current")
        self.assertEqual(retriever.effective_config, "baseline_current")
        results = retriever.retrieve("What database does ChunkdUp use?", k=5)
        self.assertTrue(results, "rollback path returned nothing for an answerable store")
        self.assertTrue(retriever._legacy is not None, "rollback must use the frozen legacy path")

    def test_unknown_config_falls_back_to_default(self):
        with mock.patch.dict(__import__("os").environ, {"CHUNKDUP_RETRIEVAL_CONFIG": "graph_rag_ultra"}):
            retriever = ProductionRetriever(make_repo())
        self.assertEqual(retriever.config_name, rp.DEFAULT_CONFIG)

    def test_fast_mode_selection(self):
        with mock.patch.dict(__import__("os").environ, {"CHUNKDUP_RETRIEVAL_CONFIG": "dense_abstain"}):
            retriever = ProductionRetriever(make_repo())
        self.assertEqual(retriever.config_name, "dense_abstain")
        results = retriever.retrieve("What database does ChunkdUp use?", k=5)
        self.assertEqual(retriever.effective_config, "dense_abstain")
        self.assertTrue(results)


class TestScaleAwareRecalibration(unittest.TestCase):
    """P2.4: gate thresholds computed from a store-representative labeled
    sample (the hard shipping dependency from docs/round3-decision.md)."""

    SAMPLE = [
        {"query": "What database does ChunkdUp use?", "relevant_memory_ids": ["m9"], "answerable": True},
        {"query": "Where does ChunkdUp run?", "relevant_memory_ids": ["m11"], "answerable": True},
        {"query": "What is my favorite programming book?", "relevant_memory_ids": [], "answerable": False},
        {"query": "sourdough bread baking", "relevant_memory_ids": [], "answerable": False},
    ]

    def test_recalibrate_updates_gate_and_result(self):
        retriever = ProductionRetriever(make_repo(), config_name="hybrid_ce_abstain")
        result = retriever.recalibrate(self.SAMPLE)
        self.assertTrue(result["changed"], f"expected recalibration to move the frozen threshold: {result}")
        self.assertIsNotNone(result["threshold"])
        self.assertIn("objective", result)
        self.assertEqual(retriever.gate.threshold, result["threshold"])
        self.assertEqual(retriever.last_calibration_size, 4)
        # retrieval still works with the recalibrated gate
        results = retriever.retrieve("What database does ChunkdUp use?", k=5)
        self.assertTrue(results)

    def test_recalibrate_dense_config(self):
        retriever = ProductionRetriever(make_repo(), config_name="dense_abstain")
        result = retriever.recalibrate(self.SAMPLE)
        self.assertEqual(result["score_field"], "_similarity")
        self.assertEqual(retriever._dense_gate.threshold, result["threshold"])

    def test_recalibrate_empty_sample_no_change(self):
        retriever = ProductionRetriever(make_repo(), config_name="hybrid_ce_abstain")
        frozen = retriever.gate.threshold
        result = retriever.recalibrate([])
        self.assertFalse(result["changed"])
        self.assertEqual(retriever.gate.threshold, frozen)

    def test_baseline_skips_recalibration(self):
        retriever = ProductionRetriever(make_repo(), config_name="baseline_current")
        result = retriever.recalibrate(self.SAMPLE)
        self.assertFalse(result["changed"])
        self.assertFalse(retriever.should_recalibrate())

    def test_should_recalibrate_tracks_store_growth(self):
        retriever = ProductionRetriever(make_repo(), config_name="hybrid_ce_abstain")
        self.assertTrue(retriever.should_recalibrate(), "never-calibrated store must recalibrate")
        retriever.recalibrate(self.SAMPLE)
        self.assertFalse(retriever.should_recalibrate())
        repo = retriever.repository
        for i in range(3):  # 4 -> 7 memories: >= 1.5x the calibration size of 4
            repo.save({"id": f"extra{i}", "type": "learning", "key": f"topic{i}",
                       "value": f"Kubernetes internals part {i}",
                       "conversation_topic": "study plan", "status": "active"})
        self.assertTrue(retriever.should_recalibrate(), "store grew past 1.5x calibration size")




def make_repo_with_project():
    repo = InMemoryRepository()
    memories = [
        {"id": "m3", "type": "project_name", "key": "project", "value": "ChunkdUp", "conversation_topic": "AI memory project", "status": "active"},
        {"id": "m9", "type": "tool", "key": "database", "value": "PostgreSQL with pgvector", "conversation_topic": "backend storage infrastructure", "status": "active"},
        {"id": "m11", "type": "environment", "key": "runtime_environment", "value": "Docker", "conversation_topic": "ChunkdUp runtime environment", "status": "active"},
    ]
    for memory in memories:
        repo.save(dict(memory))
    return repo


class TestEntityDemotionConsistency(unittest.TestCase):
    """Closeout Part 1: the production demotion must be behaviorally
    identical to the validated experiment implementation (same rule, same
    sink-below semantics), and the default production path must apply it."""

    def _ranked(self):
        return [
            {"id": "m3", "type": "project_name", "key": "project", "value": "ChunkdUp",
             "contextual_text": "This memory is a project_name regarding 'project' from a conversation about AI memory project. Source domain: user preference. The recorded user context state is: \"ChunkdUp\".",
             "_rerank_score": 5.0, "_similarity": 0.55},
            {"id": "m9", "type": "tool", "key": "database", "value": "PostgreSQL with pgvector",
             "contextual_text": "This memory is a tool regarding 'database' from a conversation about backend storage infrastructure. Source domain: user preference. The recorded user context state is: \"PostgreSQL with pgvector\".",
             "_rerank_score": 2.0, "_similarity": 0.40},
        ]

    def test_production_demotion_matches_experiment(self):
        exp_path = os.path.join(os.path.dirname(__file__), "..", "experiments", "retrieval")
        sys.path.insert(0, os.path.abspath(exp_path))
        import entity_guard as exp_guard
        import chunkdup.retrieval_production as prod_mod

        for query, field in (("What database does ChunkdUp use?", "_rerank_score"),
                             ("PostgreSQL or MongoDB for my vector storage?", "_similarity"),
                             ("What is ChunkdUp?", "_rerank_score")):
            exp_out = exp_guard.apply_entity_demotion(query, self._ranked(), field)
            prod_out = prod_mod._apply_entity_demotion(query, self._ranked(), field)
            self.assertEqual([c["id"] for c in exp_out], [c["id"] for c in prod_out], query)
            for e, p in zip(exp_out, prod_out):
                self.assertAlmostEqual(e[field], p[field], places=9, msg=query)
                self.assertEqual(e.get("_entity_demoted", False), p.get("_entity_demoted", False), query)
        self.assertEqual(exp_guard._IDENTITY_VOCAB, prod_mod._IDENTITY_VOCAB)

    def test_default_production_path_applies_demotion(self):
        retriever = ProductionRetriever(make_repo_with_project(), config_name="hybrid_ce_abstain")
        self.assertTrue(retriever.demote_enabled)
        results = retriever.retrieve("What database does ChunkdUp use?", k=5)
        ids = [r["id"] for r in results]
        self.assertIn("m9", ids, "attribute answer must be returned")
        if "m3" in ids:
            self.assertLess(ids.index("m9"), ids.index("m3"),
                            "entity-only project match must sink below the attribute answer")

    def test_failure_fallback_config_has_no_demotion(self):
        retriever = ProductionRetriever(make_repo(), config_name="dense_abstain")
        self.assertFalse(retriever.demote_enabled,
                         "the failure fallback (plain dense_abstain) keeps round-3 semantics")


class TestCEScaleTimeout(unittest.TestCase):
    """Closeout part 2: sanity check that the CE completes within the chosen
    timeout at the largest tested scale (1000 memories) under normal
    operation - i.e. the timeout is not tighter than legitimate worst-case
    latency. The dense leg uses deterministic fake vectors (the CE is the
    stage under test; its cost depends on the pair texts, which are real)."""

    def test_ce_completes_within_timeout_at_1000_memories(self):
        import time as _time

        sys.path.insert(0, os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "experiments", "retrieval")))
        from scale_fixture import generate_scale_memories, validate_scale_fixture

        memories = generate_scale_memories(1000)
        validate_scale_fixture(memories)
        repo = InMemoryRepository()
        for memory in memories:
            repo.save(dict(memory))

        retriever = ProductionRetriever(repo, config_name="hybrid_ce_abstain", query_log_path=None)
        self.assertEqual(retriever.effective_config, "hybrid_ce_abstain")

        import numpy as np

        rng = np.random.default_rng(42)

        class FakeST:
            def encode(self, texts, normalize_embeddings=False):
                single = isinstance(texts, str)
                vecs = rng.standard_normal((1 if single else len(texts), 384)).astype("float32")
                if normalize_embeddings:
                    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
                return vecs[0] if single else vecs

        with mock.patch.object(rp, "_get_sentence_transformer", return_value=FakeST()):
            # Warmup call: one-time CE model load + index build are correctly
            # NOT subject to the per-query predict timeout. The timed call
            # below measures steady-state retrieval at 1000 memories.
            retriever.retrieve("What database does ChunkdUp use?", k=5)

            t0 = _time.perf_counter()
            results = retriever.retrieve("What editor or IDE do I prefer?", k=5)
            elapsed = _time.perf_counter() - t0

        self.assertEqual(retriever.effective_config, "hybrid_ce_abstain",
                         "CE must complete without degradation at 1000 memories")
        self.assertEqual([e["reason"] for e in retriever.fallback_events], [],
                         "no fallback may fire under normal operation")
        self.assertLess(elapsed, retriever.ce_timeout_s,
                        f"steady-state retrieval took {elapsed:.3f}s, over the "
                        f"{retriever.ce_timeout_s}s timeout - the timeout is tighter "
                        f"than legitimate worst-case latency")
        self.assertTrue(results)


if __name__ == "__main__":
    unittest.main()
