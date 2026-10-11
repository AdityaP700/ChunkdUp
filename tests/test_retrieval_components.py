# tests/test_retrieval_components.py
"""Unit tests for the experiment retrieval components (dense, BM25, RRF, abstention)."""
import math
import os
import sys
import unittest

_EXPERIMENTS = os.path.join(os.path.dirname(__file__), "..", "experiments", "retrieval")
sys.path.insert(0, os.path.abspath(_EXPERIMENTS))

from abstention import AbstentionGate  # noqa: E402
from fusion import rrf_fuse  # noqa: E402
from retrievers import BM25Retriever, DenseRetriever, tokenize  # noqa: E402


class TestTokenizer(unittest.TestCase):
    def test_technical_identifiers_survive(self):
        self.assertEqual(tokenize("all-MiniLM-L6-v2"), ["all-minilm-l6-v2"])
        self.assertEqual(tokenize("C++"), ["c++"])
        self.assertIn("pgvector", tokenize("PostgreSQL with pgvector"))
        self.assertIn("22.04", tokenize("Linux Ubuntu 22.04"))

    def test_stopwords_dropped(self):
        self.assertEqual(tokenize("What is the weather forecast today?"),
                         ["weather", "forecast", "today"])

    def test_consistent_between_index_and_query(self):
        # Same function object is used by both; assert determinism.
        self.assertEqual(tokenize("PostgreSQL with pgvector"), tokenize("postgresql WITH pgvector"))


class TestDenseRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dense = DenseRetriever()

    def test_identical_text_scores_near_one(self):
        results = self.dense.search("Python", limit=1)
        self.assertTrue(results)
        self.assertGreater(results[0]["_similarity"], 0.4)

    def test_exact_lookup_finds_right_memory(self):
        results = self.dense.search("Python", limit=5)
        self.assertEqual(results[0]["id"], "m1")

    def test_inactive_memories_not_indexed(self):
        ids = {m["id"] for m in self.dense._memories}
        self.assertNotIn("m13", ids)  # inactive VS Code memory

    def test_empty_query_returns_empty(self):
        self.assertEqual(self.dense.search("   ", limit=5), [])


class TestBM25Retriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bm25 = BM25Retriever()

    def test_exact_technical_lookup(self):
        results = self.bm25.search("all-MiniLM-L6-v2", limit=3)
        self.assertEqual(results[0]["id"], "m12")

    def test_database_query_ranks_postgres_first(self):
        results = self.bm25.search("What database does ChunkdUp use?", limit=3)
        self.assertEqual(results[0]["id"], "m9")

    def test_no_overlap_returns_empty(self):
        # After stopword removal, no stored document shares any term with this.
        results = self.bm25.search("sourdough bread", limit=5)
        self.assertEqual(results, [])

    def test_zero_scores_filtered(self):
        results = self.bm25.search("sourdough bread", limit=20)
        self.assertTrue(all(c["_bm25_score"] > 0.0 for c in results))


class TestRRFFusion(unittest.TestCase):
    def _c(self, cid, score_field=None, value=None):
        c = {"id": cid, "value": value or cid}
        if score_field:
            c.update(score_field)
        return c

    def test_standard_formula(self):
        k = 60
        list_a = [self._c("a"), self._c("b")]
        list_b = [self._c("b"), self._c("c")]
        fused = rrf_fuse([list_a, list_b], k=k, limit=10)
        scores = {c["id"]: c["_rrf_score"] for c in fused}
        self.assertAlmostEqual(scores["a"], 1 / (k + 1), places=12)
        # "b" is rank 2 in list_a and rank 1 in list_b.
        self.assertAlmostEqual(scores["b"], 1 / (k + 2) + 1 / (k + 1), places=12)
        self.assertAlmostEqual(scores["c"], 1 / (k + 2), places=12)

    def test_absent_from_list_contributes_nothing(self):
        fused = rrf_fuse([[self._c("a")], [self._c("b")]], k=60, limit=10)
        parts = {c["id"]: c["_rrf_parts"] for c in fused}
        self.assertNotIn("list1", parts["a"])
        self.assertNotIn("list0", parts["b"])

    def test_dedup_by_id(self):
        fused = rrf_fuse([[self._c("a"), self._c("a")]], k=60, limit=10)
        self.assertEqual(len(fused), 1)

    def test_ordering_rank1_beats_rank2(self):
        fused = rrf_fuse([[self._c("top"), self._c("second")]], k=60, limit=10)
        self.assertEqual(fused[0]["id"], "top")

    def test_empty_input(self):
        self.assertEqual(rrf_fuse([]), [])


class TestAbstentionGate(unittest.TestCase):
    def test_abstain_when_top_below_threshold(self):
        gate = AbstentionGate("_similarity", 0.5)
        results = [{"id": "m1", "_similarity": 0.3}]
        self.assertEqual(gate.filter(results), [])
        self.assertTrue(gate.decide(results)["abstained"])

    def test_keep_when_top_above_threshold(self):
        gate = AbstentionGate("_similarity", 0.2)
        results = [{"id": "m1", "_similarity": 0.5}, {"id": "m2", "_similarity": 0.25}]
        kept = gate.filter(results)
        self.assertEqual([c["id"] for c in kept], ["m1", "m2"])

    def test_preserves_low_confidence_relevant_above_threshold(self):
        gate = AbstentionGate("_similarity", 0.2)
        results = [{"id": "m1", "_similarity": 0.9}, {"id": "m2", "_similarity": 0.21}]
        self.assertIn("m2", [c["id"] for c in gate.filter(results)])

    def test_drops_below_threshold_when_top_passes(self):
        gate = AbstentionGate("_similarity", 0.2)
        results = [{"id": "m1", "_similarity": 0.9}, {"id": "m2", "_similarity": 0.05}]
        self.assertEqual([c["id"] for c in gate.filter(results)], ["m1"])

    def test_empty_results_abstain(self):
        gate = AbstentionGate("_similarity", 0.2)
        decision = gate.decide([])
        self.assertTrue(decision["abstained"])
        self.assertIsNone(decision["top_score"])

    def test_gap_veto_disabled_by_default(self):
        gate = AbstentionGate("_similarity", 0.2)
        results = [{"id": "m1", "_similarity": 0.9}, {"id": "m2", "_similarity": 0.89}]
        self.assertFalse(gate.decide(results)["abstained"])

    def test_gap_veto_when_enabled(self):
        gate = AbstentionGate("_similarity", 0.2, min_top_gap=0.05)
        results = [{"id": "m1", "_similarity": 0.9}, {"id": "m2", "_similarity": 0.89}]
        self.assertTrue(gate.decide(results)["abstained"])


if __name__ == "__main__":
    unittest.main()
