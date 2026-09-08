

# test_ranking.py

RANKING_TESTS = [
    {
        "query": "Where does ChunkdUp run?",
        "expected_rank": 1,
        "expected_key": "runtime_environment",
        "acceptable_rank": 2  # If not rank 1, at least rank 2
    },
    {
        "query": "What database does ChunkdUp use?",
        "expected_rank": 1,
        "expected_key": "database",
        "acceptable_rank": 2
    },
    {
        "query": "What embedding model?",
        "expected_rank": 1,
        "expected_key": "embedding_model",
        "acceptable_rank": 2
    }
]