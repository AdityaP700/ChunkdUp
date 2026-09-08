#!/usr/bin/env python
"""
Semantic Retrieval Test Suite for ChunkdUp.
Run: python evaluation/test_semantic_retrieval.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ai-system'))

from chunkdup import Memory
from chunkdup.postgres_repository import PostgresRepository
import json

class SemanticRetrievalTest:
    def __init__(self):
        self.repo = PostgresRepository()
        self.memory = Memory(store="postgres", repository=self.repo)
        self.results = []

    def run_test(self, test_name, query, expected_key, expected_value):
        """Run a single semantic retrieval test."""
        results = self.memory.search(query, limit=5)

        # Find if expected memory is in results
        found = False
        rank = None
        for i, r in enumerate(results, 1):
            if r.get("key") == expected_key and r.get("value") == expected_value:
                found = True
                rank = i
                break

        result = {
            "test": test_name,
            "query": query,
            "expected": {"key": expected_key, "value": expected_value},
            "found": found,
            "rank": rank,
            "top_3": [(r.get("key"), r.get("value"), r.get("_similarity", 0))
                      for r in results[:3]]
        }
        self.results.append(result)
        return result

    def print_summary(self):
        """Print the test results."""
        print("\n" + "="*70)
        print("        CHUNKDUP — SEMANTIC RETRIEVAL TEST SUITE")
        print("="*70 + "\n")

        passed = 0
        failed = 0

        for r in self.results:
            status = "✅" if r["found"] else "❌"
            if r["found"]:
                passed += 1
            else:
                failed += 1

            print(f"{status} {r['test']}")
            print(f"   Query: \"{r['query']}\"")
            print(f"   Expected: {r['expected']['key']} = {r['expected']['value']}")
            print(f"   Found: {r['found']} (Rank: {r['rank'] if r['rank'] else 'N/A'})")
            print(f"   Top 3: {r['top_3']}")
            print()

        print("="*70)
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f" Accuracy: {passed/(passed+failed)*100:.1f}%")
        print("="*70)

def main():
    # Ensure memories exist
    repo = PostgresRepository()
    memory = Memory(store="postgres", repository=repo)

    # Clear and seed
    print("Seeding memories for testing...")
    memory.clear()
    memories = [
        "I use Python for backend development.",
        "ChunkdUp uses PostgreSQL with pgvector.",
        "ChunkdUp runs inside Docker on WSL2.",
        "The embedding model is all-MiniLM-L6-v2."
    ]
    for m in memories:
        memory.remember(m)
    print("✅ Seeded 4 memories\n")

    # Run tests
    tester = SemanticRetrievalTest()

    tests = [
        ("Exact: Where runs?", "Where does ChunkdUp run?", "runtime_environment", "Docker"),
        ("Exact: Database?", "What database does ChunkdUp use?", "database", "PostgreSQL"),
        ("Exact: Embedding model?", "What embedding model?", "embedding_model", "all-MiniLM-L6-v2"),
        ("Exact: Language?", "What language?", "programming_language", "Python"),
        ("Semantic: Hosted?", "Where is ChunkdUp hosted?", "runtime_environment", "Docker"),
        ("Semantic: Vector search?", "What technology for vector search?", "database", "PostgreSQL with pgvector"),
        ("Semantic: Model?", "What model for embeddings?", "embedding_model", "all-MiniLM-L6-v2"),
        ("Semantic: Code?", "What language for backend?", "programming_language", "Python"),
    ]

    for name, query, key, value in tests:
        tester.run_test(name, query, key, value)

    tester.print_summary()

if __name__ == "__main__":
    main()