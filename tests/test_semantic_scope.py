# test_semantic_scope.py

SEMANTIC_SCOPE_TESTS = [
    {
        "query": "What runs ChunkdUp?",
        "expected": ["runtime_environment", "Docker"],
        "reason": "Semantic: 'runs' → 'Docker'"
    },
    {
        "query": "Where is ChunkdUp hosted?",
        "expected": ["runtime_environment", "Docker"],
        "reason": "Semantic: 'hosted' → 'Docker'"
    },
    {
        "query": "What platform does ChunkdUp use?",
        "expected": ["runtime_environment", "Docker"],
        "reason": "Semantic: 'platform' → 'Docker'"
    }
]