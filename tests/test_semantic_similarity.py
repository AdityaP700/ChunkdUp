# test_semantic_similarity.py

SEMANTIC_TESTS = [
    {
        "query": "Where does ChunkdUp run?",
        "expected": ["runtime_environment", "Docker"],
        "reason": "Directly stated in memory: 'ChunkdUp runs inside Docker'"
    },
    {
        "query": "What database does ChunkdUp use?",
        "expected": ["database", "PostgreSQL"],
        "reason": "Directly stated: 'ChunkdUp uses PostgreSQL'"
    },
    {
        "query": "What embedding model does ChunkdUp use?",
        "expected": ["embedding_model", "all-MiniLM-L6-v2"],
        "reason": "Directly stated: 'The embedding model is all-MiniLM-L6-v2'"
    },
    {
        "query": "What language is ChunkdUp written in?",
        "expected": ["programming_language", "Python"],
        "reason": "Directly stated: 'I use Python for backend development'"
    },
    {
        "query": "What technology is used for vector search?",
        "expected": ["database", "PostgreSQL with pgvector"],
        "reason": "Implied: 'PostgreSQL with pgvector'"
    }
]