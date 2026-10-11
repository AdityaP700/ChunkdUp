# tests/test_entity_guard.py
"""Unit tests for round-3 experiments 6 (type-aware entity demotion) and
7 (CE entity-guard). Pure functions over pre-ranked candidate lists - no
models required."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", "retrieval")))

from entity_guard import (  # noqa: E402
    apply_entity_demotion,
    apply_entity_guard,
    identity_tokens_in_query,
)


def m3_project():
    return {"id": "m3", "type": "project_name", "key": "project", "value": "ChunkdUp",
            "contextual_text": "This memory is a project_name regarding 'project' from a conversation about AI memory project. Source domain: user preference. The recorded user context state is: \"ChunkdUp\"."}


def m9_database():
    return {"id": "m9", "type": "tool", "key": "database", "value": "PostgreSQL with pgvector",
            "contextual_text": "This memory is a tool regarding 'database' from a conversation about backend storage infrastructure. Source domain: user preference. The recorded user context state is: \"PostgreSQL with pgvector\"."}


def m11_docker():
    return {"id": "m11", "type": "environment", "key": "runtime_environment", "value": "Docker",
            "contextual_text": "This memory is an environment regarding 'runtime_environment' from a conversation about ChunkdUp runtime environment. Source domain: user preference. The recorded user context state is: \"Docker\"."}


class TestIdentityTokens(unittest.TestCase):
    def test_proper_names_are_identity(self):
        self.assertEqual(identity_tokens_in_query("What database does ChunkdUp use?"), {"chunkdup"})

    def test_generic_terms_are_not_identity(self):
        # "database" is a TECHNICAL_TERM, deliberately excluded from identity vocab
        self.assertNotIn("database", identity_tokens_in_query("database query"))
        self.assertEqual(identity_tokens_in_query("What is my favorite editor?"), set())


class TestEntityDemotion(unittest.TestCase):
    def test_project_memory_demoted_on_attribute_query(self):
        project, database = m3_project(), m9_database()
        project["_rerank_score"] = 5.0
        database["_rerank_score"] = 2.0
        ranked = [project, database]
        adjusted = apply_entity_demotion("What database does ChunkdUp store memories in?", ranked, "_rerank_score")
        by_id = {c["id"]: c for c in adjusted}
        self.assertTrue(by_id["m3"].get("_entity_demoted"))
        self.assertEqual(by_id["m3"]["_pre_demotion__rerank_score"], 5.0)
        self.assertFalse(by_id["m9"].get("_entity_demoted", False))
        self.assertLess(adjusted.index(by_id["m9"]), adjusted.index(by_id["m3"]),
                        "demoted project memory must fall below the attribute memory")

    def test_correct_runtime_memory_not_demoted(self):
        # q45: the Docker memory also matches only on 'ChunkdUp' but is NOT a
        # project_name memory - type-awareness must protect it.
        ranked = [m3_project(), m11_docker()]
        adjusted = apply_entity_demotion("Where does ChunkdUp run?", ranked, "_rerank_score")
        by_id = {c["id"]: c for c in adjusted}
        self.assertTrue(by_id["m3"].get("_entity_demoted"))
        self.assertFalse(by_id["m11"].get("_entity_demoted", False))

    def test_no_demotion_when_query_is_about_the_entity(self):
        # q49 "What is ChunkdUp?": no non-entity tokens -> m3 must survive.
        ranked = [m3_project()]
        adjusted = apply_entity_demotion("What is ChunkdUp?", ranked, "_rerank_score")
        self.assertFalse(adjusted[0].get("_entity_demoted", False))

    def test_no_demotion_without_identity_token(self):
        ranked = [m3_project()]
        adjusted = apply_entity_demotion("What project am I building?", ranked, "_rerank_score")
        self.assertFalse(adjusted[0].get("_entity_demoted", False))

    def test_postgres_value_not_demoted_safety_probe(self):
        # e12: correct answer contains 'PostgreSQL' as an identity token, but
        # it is a tool memory - the type-aware rule must never touch it.
        ranked = [m9_database()]
        adjusted = apply_entity_demotion("PostgreSQL or MongoDB for my vector storage?", ranked, "_rerank_score")
        self.assertFalse(adjusted[0].get("_entity_demoted", False))

    def test_substantive_overlap_protects_project_memory(self):
        # A project memory the query genuinely asks about stays untouched.
        project = m3_project()
        project["contextual_text"] += " It stores memories in a database."
        ranked = [project]
        adjusted = apply_entity_demotion("What database does the project store in?", ranked, "_rerank_score")
        self.assertFalse(adjusted[0].get("_entity_demoted", False))


class TestEntityGuard(unittest.TestCase):
    def test_restores_fused_top1_that_ce_demoted(self):
        # q46 pattern: fused top-1 m9, CE pushes it to rank 3. Guard: fused
        # top-3 lead in fused order, CE orders the remainder.
        fused = [m9_database(), m3_project(), m11_docker()]
        ce_ranked = [m11_docker(), m3_project(), m9_database()]
        final = apply_entity_guard("What database does ChunkdUp use?", ce_ranked, fused)
        self.assertEqual([c["id"] for c in final], ["m9", "m3", "m11"])
        self.assertTrue(next(c for c in final if c["id"] == "m9").get("_entity_guard_restored"))

    def test_q87_pattern_restores_fused_rank2(self):
        # fused: m12, m9, ...; CE: m12, m3, m11, m9 (m9 dropped to rank 4)
        fused = [{"id": "m12"}, {"id": "m9"}, {"id": "m3"}, {"id": "m11"}]
        ce_ranked = [{"id": "m12"}, {"id": "m3"}, {"id": "m11"}, {"id": "m9"}]
        final = apply_entity_guard("What vector database powers ChunkdUp?", ce_ranked, fused)
        ids = [c["id"] for c in final]
        self.assertLess(ids.index("m9"), 3, "m9 (fused rank 2) must sit inside the top-3 after the guard")

    def test_noop_without_identity_token(self):
        fused = [{"id": "a"}, {"id": "b"}]
        ce_ranked = [{"id": "b"}, {"id": "a"}]
        final = apply_entity_guard("What is my favorite editor?", ce_ranked, fused)
        self.assertEqual([c["id"] for c in final], ["b", "a"])

    def test_noop_when_ce_agrees_with_fusion(self):
        fused = [{"id": "m12"}, {"id": "m9"}]
        ce_ranked = [{"id": "m12"}, {"id": "m9"}, {"id": "m3"}]
        final = apply_entity_guard("What vector database powers ChunkdUp?", ce_ranked, fused)
        self.assertEqual([c["id"] for c in final], ["m12", "m9", "m3"])
        self.assertFalse(any(c.get("_entity_guard_restored") for c in final))


if __name__ == "__main__":
    unittest.main()
