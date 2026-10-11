# tests/test_production_retrieval.py
"""Tests for the hardened production retrieval path (round-3 part 2).

P2.1: graceful degradation - cross-encoder load failure, inference failure,
and timeout must fall back to the dense gate (dense_abstain) and embedding
failures to the frozen baseline, with every fallback logged, never silent.
"""
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


if __name__ == "__main__":
    unittest.main()
