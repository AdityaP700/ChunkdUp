# test_semantic.py
import os
import pytest

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from chunkdup.postgres_repository import PostgresRepository
    from chunkdup import Memory
except (ImportError, ModuleNotFoundError):
    PostgresRepository = None


def test_semantic_search():
    if PostgresRepository is None:
        pytest.skip("PostgreSQL dependencies (SQLAlchemy/pgvector) not installed")
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL not configured; skipping live PostgreSQL test")

    print("🧪 Testing Semantic Search...")

    # Connect to PostgreSQL
    DATABASE_URL = database_url
    repo = PostgresRepository(DATABASE_URL)
    memory = Memory(store="postgres", repository=repo)

    # Clear existing memories
    memory.clear()

    # Add diverse memories
    print("\n📝 Adding memories...")
    memory.remember("I use Python for backend development")
    memory.remember("I use Go for microservices")
    memory.remember("I'm building a web app called ChunkdUp")
    memory.remember("My favorite editor is Neovim")
    memory.remember("I work at Google as a software engineer")
    memory.remember("I prefer dark mode for my IDE")

    # Test 1: Keyword Search
    print("\n🔍 Keyword Search: 'Python'")
    keyword_results = repo.search("Python", limit=5)
    for r in keyword_results:
        print(f"  {r['key']}: {r['value']} (freq: {r['frequency']})")

    # Test 2: Semantic Search
    print("\n🧠 Semantic Search: 'What programming languages do I use?'")
    semantic_results = repo.search_semantic("What programming languages do I use?", limit=5)
    for r in semantic_results:
        print(f"  {r['key']}: {r['value']} (freq: {r['frequency']})")

    # Test 3: Semantic Search (different phrasing)
    print("\n🧠 Semantic Search: 'What is my tech stack?'")
    semantic_results = repo.search_semantic("What is my tech stack?", limit=5)
    for r in semantic_results:
        print(f"  {r['key']}: {r['value']} (freq: {r['frequency']})")

    # Test 4: Hybrid Search
    print("\n🔀 Hybrid Search: 'What am I working on?'")
    hybrid_results = repo.search_hybrid("What am I working on?", limit=5)
    for r in hybrid_results:
        print(f"  {r['key']}: {r['value']} (freq: {r['frequency']})")

    # Test 5: Edge Case - No Results
    print("\n🧠 Semantic Search: 'What is the weather?' (should return nothing or low relevance)")
    semantic_results = repo.search_semantic("What is the weather?", limit=5)
    if not semantic_results:
        print("  ✅ No results (correct - no weather memories)")
    else:
        for r in semantic_results:
            print(f"  {r['key']}: {r['value']}")

if __name__ == "__main__":
    test_semantic_search()