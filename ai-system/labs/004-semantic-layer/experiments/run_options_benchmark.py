import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple

# Mock memory dataset representing realistic system memory states and edge cases / stress queries
MEMORIES_DATASET = [
    {"id": "m1", "type": "programming_language", "key": "language", "value": "Python", "embedding_text": "Python programming language for AI backend", "status": "active"},
    {"id": "m2", "type": "programming_language", "key": "language", "value": "Go", "embedding_text": "Go programming language for microservices", "status": "active"},
    {"id": "m3", "type": "project_name", "key": "project", "value": "ChunkdUp", "embedding_text": "ChunkdUp project for AI memory management", "status": "active"},
    {"id": "m4", "type": "environment", "key": "os", "value": "Linux Ubuntu 22.04", "embedding_text": "Linux operating system dev environment", "status": "active"},
    {"id": "m5", "type": "editor", "key": "ide", "value": "Neovim", "embedding_text": "Neovim code editor and IDE customization", "status": "active"},
    {"id": "m6", "type": "preference", "key": "response_style", "value": "Concise and technical", "embedding_text": "User prefers concise technical responses", "status": "active"},
    {"id": "m7", "type": "response_style", "key": "tone", "value": "Direct and analytical", "embedding_text": "Tone should be direct, clean, and analytical", "status": "active"},
    {"id": "m8", "type": "company", "key": "employer", "value": "Google DeepMind", "embedding_text": "Google DeepMind AI research team", "status": "active"},
    {"id": "m9", "type": "role", "key": "job_title", "value": "Senior AI Architect", "embedding_text": "Senior AI systems architect working on agentic workflows", "status": "active"},
    {"id": "m10", "type": "tool", "key": "database", "value": "PostgreSQL with pgvector", "embedding_text": "PostgreSQL database storage with pgvector extension", "status": "active"}
]

# Simple deterministic embedding mock using keyword features to simulate cosine similarity deterministically
# We hash terms into fixed vector dimensions for simulation without heavy neural dependencies
VOCAB = [
    "python", "go", "language", "code", "programming", "stack", "project", "chunkdup",
    "building", "working", "linux", "os", "environment", "neovim", "editor", "ide",
    "tool", "weather", "forecast", "sunny", "rain", "temperature", "preference",
    "concise", "style", "database", "postgres", "vector", "google", "role", "architect"
]

def text_to_embedding(text: str) -> np.ndarray:
    tokens = text.lower().replace("-", " ").split()
    vec = np.zeros(len(VOCAB), dtype=float)
    for token in tokens:
        if token in VOCAB:
            vec[VOCAB.index(token)] += 1.0
    # Add slight semantic crosstalk to simulate embedding similarity (e.g. weather having low similarity to environment/editor)
    if "weather" in tokens or "forecast" in tokens:
        # Cross-talk noise with environment/editor to simulate embedding bug (Failure 1)
        vec[VOCAB.index("environment")] += 0.2
        vec[VOCAB.index("editor")] += 0.15
        vec[VOCAB.index("os")] += 0.1
    
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

# Pre-compute embeddings for dataset
for mem in MEMORIES_DATASET:
    mem["embedding"] = text_to_embedding(f"{mem['key']} {mem['value']} {mem['type']} {mem['embedding_text']}")

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))

TYPE_BOOST = {
    "programming_language": 2.0,  # High priority
    "project_name": 1.5,          # Medium priority
    "environment": 1.3,           # Medium priority
    "editor": 1.0,                # Normal priority
    "preference": 0.8,            # Lower priority
    "response_style": 0.6,        # Lower priority
}

def detect_query_type(query: str) -> str:
    query_lower = query.lower()
    
    # Language detection
    if any(word in query_lower for word in ["language", "lang", "programming", "code", "stack"]):
        return "language"
    
    # Project detection
    if any(word in query_lower for word in ["project", "building", "working on"]):
        return "project"
    
    # Tool detection
    if any(word in query_lower for word in ["tool", "editor", "ide", "use"]):
        return "tool"
    
    return "general"

class SearchEvaluator:
    def __init__(self, memories: List[Dict[str, Any]]):
        self.memories = memories

    def base_keyword_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        query_words = set(query.lower().split())
        results = []
        for mem in self.memories:
            combined = f"{mem['key']} {mem['value']} {mem['type']} {mem['embedding_text']}".lower()
            mem_words = set(combined.split())
            overlap = len(query_words & mem_words)
            if overlap > 0:
                res = dict(mem)
                res["_relevance"] = float(overlap)
                results.append(res)
        results.sort(key=lambda x: x["_relevance"], reverse=True)
        return results[:limit]

    def base_semantic_search(self, query: str, limit: int = 10, threshold: float = 0.0) -> List[Dict[str, Any]]:
        q_emb = text_to_embedding(query)
        results = []
        for mem in self.memories:
            sim = cosine_similarity(q_emb, mem["embedding"])
            if sim >= threshold:
                res = dict(mem)
                res["_relevance"] = sim
                results.append(res)
        results.sort(key=lambda x: x["_relevance"], reverse=True)
        return results[:limit]

    # --- OPTION A: Similarity Threshold ---
    def option_a_search(self, query: str, limit: int = 5, threshold: float = 0.3) -> List[Dict[str, Any]]:
        return self.base_semantic_search(query, limit=limit, threshold=threshold)

    # --- OPTION B: Type-Based Filtering/Boost ---
    def option_b_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        results = self.base_semantic_search(query, limit=limit*2, threshold=0.0)
        for result in results:
            memory_type = result.get("type")
            boost = TYPE_BOOST.get(memory_type, 1.0)
            result["_relevance"] *= boost
        results.sort(key=lambda x: x["_relevance"], reverse=True)
        return results[:limit]

    # --- OPTION C: Query Type Detection ---
    def option_c_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        query_type = detect_query_type(query)
        allowed_types = {
            "language": ["programming_language", "learning"],
            "project": ["project_name"],
            "tool": ["editor", "environment", "tool"],
            "general": ["programming_language", "project_name", "editor", "environment", "company", "role", "preference"]
        }
        results = self.base_semantic_search(query, limit=limit*2, threshold=0.0)
        allowed = allowed_types.get(query_type, [])
        results = [r for r in results if r.get("type") in allowed]
        return results[:limit]

    # --- OPTION D: Hybrid (Keyword + Semantic with Reranking) ---
    def option_d_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        keyword_results = self.base_keyword_search(query, limit=limit*2)
        semantic_results = self.base_semantic_search(query, limit=limit*2, threshold=0.3)
        query_type = detect_query_type(query)

        # Merge pool and apply reranking boosts
        pool = {}
        for r in keyword_results + semantic_results:
            rid = r["id"]
            if rid not in pool:
                pool[rid] = dict(r)
                pool[rid]["_relevance"] = pool[rid].get("_relevance", 0.0)
            
            boost = TYPE_BOOST.get(pool[rid].get("type"), 1.0)
            pool[rid]["_relevance"] *= boost
            
            if query_type == "language" and pool[rid].get("type") in ["programming_language", "learning"]:
                pool[rid]["_relevance"] *= 1.5

        # RRF Fusion simulation
        combined_list = list(pool.values())
        combined_list.sort(key=lambda x: x["_relevance"], reverse=True)
        return combined_list[:limit]

# Benchmark queries to test standard behavior and stress-test failure points
TEST_SUITE = [
    {"query": "What is the weather today?", "scenario": "Stress Test: Completely Out-of-Domain Query (Weather Problem)"},
    {"query": "What programming languages do I use?", "scenario": "Language Precision: Should prefer Python/Go over tools/projects"},
    {"query": "What is my tech stack?", "scenario": "Tech Stack Synthesis: Balanced recall for languages + tools"},
    {"query": "Python", "scenario": "Exact Keyword Match: Direct lookup precision"},
    {"query": "How should you respond to me?", "scenario": "Preference Retrieval: Finding communication/response style preferences"},
    {"query": "Where do I work and what is my role?", "scenario": "General Entity Retrieval: Unseen type fallback"}
]

def run_experiments():
    evaluator = SearchEvaluator(MEMORIES_DATASET)
    options = ["Option A", "Option B", "Option C", "Option D"]
    
    results_by_option = {opt: [] for opt in options}

    for test in TEST_SUITE:
        q = test["query"]
        scen = test["scenario"]

        # Option A
        res_a = evaluator.option_a_search(q, limit=5, threshold=0.3)
        results_by_option["Option A"].append({
            "query": q,
            "scenario": scen,
            "returned_count": len(res_a),
            "top_memories": [{"type": r["type"], "value": r["value"], "score": round(r["_relevance"], 4)} for r in res_a]
        })

        # Option B
        res_b = evaluator.option_b_search(q, limit=5)
        results_by_option["Option B"].append({
            "query": q,
            "scenario": scen,
            "returned_count": len(res_b),
            "top_memories": [{"type": r["type"], "value": r["value"], "score": round(r["_relevance"], 4)} for r in res_b]
        })

        # Option C
        res_c = evaluator.option_c_search(q, limit=5)
        results_by_option["Option C"].append({
            "query": q,
            "scenario": scen,
            "detected_query_type": detect_query_type(q),
            "returned_count": len(res_c),
            "top_memories": [{"type": r["type"], "value": r["value"], "score": round(r["_relevance"], 4)} for r in res_c]
        })

        # Option D
        res_d = evaluator.option_d_search(q, limit=5)
        results_by_option["Option D"].append({
            "query": q,
            "scenario": scen,
            "detected_query_type": detect_query_type(q),
            "returned_count": len(res_d),
            "top_memories": [{"type": r["type"], "value": r["value"], "score": round(r["_relevance"], 4)} for r in res_d]
        })

    # Output JSON artifacts for each option
    out_dir = os.path.dirname(os.path.abspath(__file__))
    for opt, data in results_by_option.items():
        filename = f"results_{opt.lower().replace(' ', '_')}.json"
        filepath = os.path.join(out_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {filepath}")

if __name__ == "__main__":
    run_experiments()
