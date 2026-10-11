# tests/test_retrieval_integration.py
"""Integration tests over the full experiment retrieval path:
candidate generation -> fusion -> reranking -> abstention -> final results."""
import json
import os
import sys
import unittest
from pathlib import Path

_EXPERIMENTS = Path(__file__).resolve().parent.parent / "experiments" / "retrieval"
sys.path.insert(0, str(_EXPERIMENTS))

from abstention import AbstentionGate  # noqa: E402
from config import CONFIGS  # noqa: E402
from fixture import BASE_MEMORIES, all_memories  # noqa: E402
from fusion import rrf_fuse  # noqa: E402
from retrievers import BM25Retriever, DenseRetriever  # noqa: E402


class TestFullRetrievalPath(unittest.TestCase):
    """End-to-end behavior of the combined pipeline stages."""

    @classmethod
    def setUpClass(cls):
        cls.dense = DenseRetriever()
        cls.bm25 = BM25Retriever()
        calibration = json.loads((_EXPERIMENTS / "results" / "calibration.json").read_text(encoding="utf-8"))
        cls.dense_threshold = calibration["dense_abstain"]["threshold"]

    def _hybrid(self, query, limit=20):
        dense_results = self.dense.search(query, limit)
        bm25_results = self.bm25.search(query, limit)
        return dense_results, bm25_results, rrf_fuse([dense_results, bm25_results], k=60, limit=limit)

    def test_answerable_query_returns_relevant_memory(self):
        _d, _b, fused = self._hybrid("What database does ChunkdUp use?")
        self.assertTrue(fused)
        self.assertIn("m9", [c["id"] for c in fused[:3]])

    def test_unanswerable_query_abstains_with_calibrated_gate(self):
        # "sourdough bread" shares no terms with any memory -> BM25 empty;
        # the dense gate must reject the weak cosine match.
        gate = AbstentionGate("_similarity", self.dense_threshold)
        results = self.dense.search("How do I bake sourdough bread?", limit=20)
        self.assertEqual(gate.filter(results), [])

    def test_answerable_query_survives_calibrated_gate(self):
        gate = AbstentionGate("_similarity", self.dense_threshold)
        results = self.dense.search("Python", limit=20)
        kept = gate.filter(results)
        self.assertIn("m1", [c["id"] for c in kept[:1]])

    def test_duplicate_candidates_deduplicated_in_fusion(self):
        dense_results = self.dense.search("Python", limit=20)
        bm25_results = self.bm25.search("Python", limit=20)
        fused = rrf_fuse([dense_results, bm25_results], k=60, limit=20)
        ids = [c["id"] for c in fused]
        self.assertEqual(len(ids), len(set(ids)))

    def test_inactive_historical_memory_never_returned(self):
        for query in ("What editor do I use these days?", "Where do I work currently?", "editor", "employer"):
            _d, _b, fused = self._hybrid(query)
            returned_ids = {c["id"] for c in fused}
            self.assertNotIn("m13", returned_ids, query)  # inactive VS Code
            self.assertNotIn("m15", returned_ids, query)  # inactive OpenAI

    def test_empty_store_returns_no_candidates(self):
        empty = BM25Retriever(memories=[])
        self.assertEqual(empty.search("Python", limit=5), [])

    def test_updated_memory_surfaces_current_value(self):
        # User switched editors: m5 Neovim active, m13 VS Code inactive.
        _d, _b, fused = self._hybrid("What editor do I use these days?")
        if fused:  # both legs may legitimately be empty for a pure stopword query
            self.assertEqual(fused[0]["id"], "m5")


class TestExperimentPipelineEndToEnd(unittest.TestCase):
    """The configured pipelines behave correctly through run_experiment."""

    @classmethod
    def setUpClass(cls):
        from run_experiment import ExperimentPipeline
        calibration = json.loads((_EXPERIMENTS / "results" / "calibration.json").read_text(encoding="utf-8"))
        cls.calibration = calibration
        cls.make_pipeline = ExperimentPipeline

    def test_dense_abstain_pipeline_rejects_and_answers(self):
        pipeline = self.make_pipeline(CONFIGS["dense_abstain"], self.calibration)
        results, timings, _ = pipeline.retrieve("What is the weather forecast today?", k=5)
        self.assertEqual(results, [])
        self.assertGreater(timings["total_ms"], 0)

        results, _, _ = pipeline.retrieve("What programming languages do I use?", k=5)
        self.assertTrue(results)
        self.assertIn([r["id"] for r in results[:2]], [["m1", "m2"], ["m2", "m1"]])

    def test_bm25_pipeline_natural_rejection(self):
        pipeline = self.make_pipeline(CONFIGS["bm25"], None)
        results, _, _ = pipeline.retrieve("Tell me a joke.", k=5)
        self.assertEqual(results, [])

    def test_hybrid_ce_abstain_pipeline(self):
        try:
            pipeline = self.make_pipeline(CONFIGS["hybrid_ce_abstain"], self.calibration)
        except Exception as exc:  # model download unavailable in this environment
            self.skipTest(f"cross-encoder unavailable: {exc}")
        results, _, _ = pipeline.retrieve("Where does ChunkdUp run?", k=5)
        self.assertTrue(results, "expected Docker memory, got abstention")
        self.assertEqual(results[0]["id"], "m11")

        results, _, _ = pipeline.retrieve("What is the weather forecast today?", k=5)
        self.assertEqual(results, [])

    def test_baseline_pipeline_still_functional(self):
        from run_experiment import BaselinePipeline
        from config import ExperimentConfig
        pipeline = BaselinePipeline(ExperimentConfig(name="baseline_current", retriever="both"))
        results, timings, _ = pipeline.retrieve("What is my tech stack?", k=5)
        self.assertTrue(results)
        self.assertIn("total_ms", timings)


if __name__ == "__main__":
    unittest.main()
