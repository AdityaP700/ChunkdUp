# tests/test_sdk.py
"""
Unit tests for ChunkdUp Phase 1 SDK Polish.
"""

import unittest
import asyncio
import os
import sys
import tempfile
import shutil

# Ensure ai-system is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ai-system"))

from chunkdup import Memory, AsyncMemory, MemoryDict, RememberResult, SearchResult


class TestChunkdUpSDK(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.memory = Memory(store="memory", data_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_store_configuration_validation(self):
        # Invalid store type
        with self.assertRaises(ValueError):
            Memory(store="invalid_store")

        # Postgres store without connection_url
        with self.assertRaises(ValueError):
            Memory(store="postgres")

    def test_remember_and_retrieve(self):
        res = self.memory.remember("I use Python")
        self.assertIn("count", res)
        self.assertGreaterEqual(res["count"], 1)

        memories = self.memory.retrieve("Python")
        self.assertTrue(len(memories) > 0)
        self.assertEqual(memories[0]["key"], "programming_language")
        self.assertEqual(memories[0]["value"], "Python")

    def test_search_and_get_all(self):
        self.memory.remember("My favorite editor is Neovim.")
        search_res = self.memory.search("editor")
        self.assertTrue(len(search_res) > 0)
        self.assertEqual(search_res[0]["key"], "editor")

        all_memories = self.memory.get_all()
        self.assertTrue(any(m["key"] == "editor" for m in all_memories))

    def test_update_memory(self):
        self.memory.remember("I use Python")
        update_res = self.memory.update("programming_language", "Go")
        self.assertEqual(update_res["decision"], "UPDATE")
        self.assertEqual(update_res["value"], "Go")

        retrieved = self.memory.retrieve("programming_language")
        self.assertEqual(retrieved[0]["value"], "Go")

    def test_update_non_existent_memory_raises(self):
        with self.assertRaises(ValueError):
            self.memory.update("non_existent_key", "Some Value")

    def test_delete_memory(self):
        self.memory.remember("I use Python")
        all_mem = self.memory.get_all()
        mem_id = all_mem[0]["id"]

        del_res = self.memory.delete(mem_id)
        self.assertTrue(del_res["success"])
        self.assertEqual(del_res["id"], mem_id)

        all_after_del = self.memory.get_all()
        self.assertEqual(len(all_after_del), 0)

    def test_get_stats_and_clear(self):
        self.memory.remember("I use Python")
        stats = self.memory.get_stats()
        self.assertEqual(stats["total_memories"], 1)
        self.assertEqual(stats["store_type"], "memory")
        self.assertIn("programming_language", stats["keys"])

        self.memory.clear()
        self.assertEqual(len(self.memory.get_all()), 0)
        self.assertEqual(len(self.memory.get_conversation_history()), 0)

    def test_legacy_aliases(self):
        self.memory.add("I use Python")
        memories = self.memory.get_all_memories()
        self.assertTrue(len(memories) > 0)
        stats = self.memory.get_statistics()
        self.assertIn("total_memories", stats)

    def test_async_memory(self):
        async def run_async_tests():
            async_mem = AsyncMemory(store="memory", data_dir=self.temp_dir)
            await async_mem.remember("I use Python")
            results = await async_mem.retrieve("Python")
            self.assertTrue(len(results) > 0)
            self.assertEqual(results[0]["value"], "Python")

            await async_mem.update("programming_language", "Rust")
            all_mem = await async_mem.get_all()
            self.assertEqual(all_mem[0]["value"], "Rust")

            stats = await async_mem.get_stats()
            self.assertEqual(stats["total_memories"], 1)

            del_res = await async_mem.delete(all_mem[0]["id"])
            self.assertTrue(del_res["success"])

            await async_mem.clear()
            cleared_all = await async_mem.get_all()
            self.assertEqual(len(cleared_all), 0)

        asyncio.run(run_async_tests())


if __name__ == "__main__":
    unittest.main()
