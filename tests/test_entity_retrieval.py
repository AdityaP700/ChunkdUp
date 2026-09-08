import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ai-system"))

from chunkdup import Memory


class TestEntityRetrieval(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.memory = Memory(store="memory", data_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_entity_match_recovers_runtime_answer(self):
        self.memory.remember("ChunkdUp runs inside Docker.")

        results = self.memory.retrieve("Where does ChunkdUp run?")

        self.assertTrue(results)
        self.assertEqual(results[0]["key"], "runtime_environment")
        self.assertEqual(results[0]["value"], "Docker")
        self.assertIn("ChunkdUp", results[0]["entities"])


if __name__ == "__main__":
    unittest.main()