# evaluation/retrieval_benchmarks.py
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Adjust paths
sys.path.insert(0, str(Path(__file__).parent.parent / "ai-system"))
from chunkdup.in_memory_repository import InMemoryRepository
from chunkdup.retrieval_class import MemoryRetriever

BENCHMARK_MEMORIES = [
    {"id": "m1", "type": "programming_language", "key": "language", "value": "Python", "conversation_topic": "tech stack", "status": "active"},
    {"id": "m2", "type": "programming_language", "key": "language", "value": "Go", "conversation_topic": "microservices", "status": "active"},
    {"id": "m3", "type": "project_name", "key": "project", "value": "ChunkdUp", "conversation_topic": "AI memory project", "status": "active"},
    {"id": "m4", "type": "environment", "key": "os", "value": "Linux Ubuntu 22.04", "conversation_topic": "dev machine setup", "status": "active"},
    {"id": "m5", "type": "editor", "key": "ide", "value": "Neovim", "conversation_topic": "editor preferences", "status": "active"},
    {"id": "m6", "type": "preference", "key": "response_style", "value": "Concise and technical", "conversation_topic": "user communication style", "status": "active"},
    {"id": "m7", "type": "company", "key": "employer", "value": "Google DeepMind", "conversation_topic": "employment history", "status": "active"},
    {"id": "m8", "type": "role", "key": "job_title", "value": "Senior AI Architect", "conversation_topic": "professional career", "status": "active"},
    {"id": "m9", "type": "tool", "key": "database", "value": "PostgreSQL with pgvector", "conversation_topic": "backend storage infrastructure", "status": "active"},
    {"id": "m10", "type": "cloud_provider", "key": "cloud", "value": "AWS", "conversation_topic": "cloud infrastructure", "status": "active"}
]

RETRIEVAL_TEST_SCENARIOS = [
    {
        "query": "What programming languages do I use?",
        "expected_keys": ["language"],
        "expected_values": ["Python", "Go"],
        "category": "language"
    },
    {
        "query": "What is my tech stack?",
        "expected_keys": ["language", "project", "ide", "database"],
        "expected_values": ["Python", "Go", "ChunkdUp", "Neovim", "PostgreSQL with pgvector"],
        "category": "synthesis"
    },
    {
        "query": "Python",
        "expected_keys": ["language"],
        "expected_values": ["Python"],
        "category": "exact_lookup"
    },
    {
        "query": "What editor or IDE do I prefer?",
        "expected_keys": ["ide"],
        "expected_values": ["Neovim"],
        "category": "tool"
    },
    {
        "query": "What is the weather forecast today?",
        "expected_keys": [],
        "expected_values": [],
        "category": "out_of_domain"
    },
    {
        "query": "How should you respond to me?",
        "expected_keys": ["response_style"],
        "expected_values": ["Concise and technical"],
        "category": "preference"
    },
    {
        "query": "Where do I work and what is my role?",
        "expected_keys": ["employer", "job_title"],
        "expected_values": ["Google DeepMind", "Senior AI Architect"],
        "category": "career"
    }
]

def evaluate_retrieval_pipeline(reranker_mode: str = "heuristic") -> Dict[str, Any]:
    repo = InMemoryRepository()

    retriever = MemoryRetriever(repository=repo, reranker_mode=reranker_mode)

    # Index memories with Anthropic-style Contextual Prepending
    for mem in BENCHMARK_MEMORIES:
        enriched = retriever.index_memory(mem)
        repo.save(enriched)

    total_scenarios = len(RETRIEVAL_TEST_SCENARIOS)
    reciprocal_ranks = []
    precision_at_k = []
    recall_at_k = []

    scenario_details = []

    for test in RETRIEVAL_TEST_SCENARIOS:
        query = test["query"]
        expected_values = set(test["expected_values"])

        results = retriever.retrieve(query, k=5)
        retrieved_values = [r.get("value") for r in results]

        # Calculate metrics
        hits = [v for v in retrieved_values if v in expected_values]
        prec = len(hits) / len(retrieved_values) if retrieved_values else (1.0 if not expected_values else 0.0)
        rec = len(hits) / len(expected_values) if expected_values else (1.0 if not retrieved_values else 0.0)

        # Reciprocal rank calculation
        rr = 0.0
        for rank, val in enumerate(retrieved_values):
            if val in expected_values:
                rr = 1.0 / (rank + 1)
                break
        if not expected_values and not retrieved_values:
            rr = 1.0

        reciprocal_ranks.append(rr)
        precision_at_k.append(prec)
        recall_at_k.append(rec)

        scenario_details.append({
            "query": query,
            "category": test["category"],
            "retrieved_count": len(results),
            "retrieved": retrieved_values,
            "expected": list(expected_values),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "reciprocal_rank": round(rr, 4)
        })

    mrr = sum(reciprocal_ranks) / total_scenarios if total_scenarios > 0 else 0.0
    mean_precision = sum(precision_at_k) / total_scenarios if total_scenarios > 0 else 0.0
    mean_recall = sum(recall_at_k) / total_scenarios if total_scenarios > 0 else 0.0

    return {
        "reranker_mode": reranker_mode,
        "metrics": {
            "MRR": round(mrr, 4),
            "Mean_Precision@K": round(mean_precision, 4),
            "Mean_Recall@K": round(mean_recall, 4)
        },
        "details": scenario_details
    }

if __name__ == "__main__":
    modes = ["heuristic", "llm"]
    all_results = {}
    for mode in modes:
        res = evaluate_retrieval_pipeline(reranker_mode=mode)
        all_results[mode] = res
        print(f"=== Mode: {mode.upper()} ===")
        print(f"MRR: {res['metrics']['MRR']}")
        print(f"Mean Precision@K: {res['metrics']['Mean_Precision@K']}")
        print(f"Mean Recall@K: {res['metrics']['Mean_Recall@K']}\n")

    out_path = Path(__file__).parent / "retrieval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved evaluation benchmark results to {out_path}")
