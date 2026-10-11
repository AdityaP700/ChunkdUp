# experiments/retrieval/fixture.py
"""
Single shared memory fixture for every retrieval experiment.

Extends the 10-memory baseline fixture from evaluation/retrieval_benchmarks.py
(ids m1..m10, keys and values identical) with project-fact memories that the
task's required queries reference, plus inactive historical memories used by
the updated/conflicting-facts category.

Grounding of the added memories (verified against the repository itself):
- m11 "Docker"          — tests/test_entity_retrieval.py: "ChunkdUp runs inside Docker."
- m12 "all-MiniLM-L6-v2" — ai-system/chunkdup/embeddings.py default model
- m13..m17 historical (inactive) facts used only to test that superseded
  memories are not returned for current-state queries.
"""
from typing import Dict, List, Any

BASE_MEMORIES: List[Dict[str, Any]] = [
    {"id": "m1", "type": "programming_language", "key": "language", "value": "Python", "conversation_topic": "tech stack", "status": "active"},
    {"id": "m2", "type": "programming_language", "key": "language", "value": "Go", "conversation_topic": "microservices", "status": "active"},
    {"id": "m3", "type": "project_name", "key": "project", "value": "ChunkdUp", "conversation_topic": "AI memory project", "status": "active"},
    {"id": "m4", "type": "environment", "key": "os", "value": "Linux Ubuntu 22.04", "conversation_topic": "dev machine setup", "status": "active"},
    {"id": "m5", "type": "editor", "key": "ide", "value": "Neovim", "conversation_topic": "editor preferences", "status": "active"},
    {"id": "m6", "type": "preference", "key": "response_style", "value": "Concise and technical", "conversation_topic": "user communication style", "status": "active"},
    {"id": "m7", "type": "company", "key": "employer", "value": "Google DeepMind", "conversation_topic": "employment history", "status": "active"},
    {"id": "m8", "type": "role", "key": "job_title", "value": "Senior AI Architect", "conversation_topic": "professional career", "status": "active"},
    {"id": "m9", "type": "tool", "key": "database", "value": "PostgreSQL with pgvector", "conversation_topic": "backend storage infrastructure", "status": "active"},
    {"id": "m10", "type": "cloud_provider", "key": "cloud", "value": "AWS", "conversation_topic": "cloud infrastructure", "status": "active"},
    {"id": "m11", "type": "environment", "key": "runtime_environment", "value": "Docker", "conversation_topic": "ChunkdUp runtime environment", "status": "active"},
    {"id": "m12", "type": "tool", "key": "embedding_model", "value": "all-MiniLM-L6-v2", "conversation_topic": "vector search stack", "status": "active"},
]

# Inactive historical memories: superseded facts. Retrieval must NOT return
# these for current-state queries, and must not treat them as relevant.
HISTORICAL_MEMORIES: List[Dict[str, Any]] = [
    {"id": "m13", "type": "editor", "key": "ide", "value": "VS Code", "conversation_topic": "editor preferences", "status": "inactive"},
    {"id": "m14", "type": "environment", "key": "os", "value": "Mac", "conversation_topic": "dev machine setup", "status": "inactive"},
    {"id": "m15", "type": "company", "key": "employer", "value": "OpenAI", "conversation_topic": "employment history", "status": "inactive"},
    {"id": "m16", "type": "cloud_provider", "key": "cloud", "value": "GCP", "conversation_topic": "cloud infrastructure", "status": "inactive"},
    {"id": "m17", "type": "role", "key": "job_title", "value": "Machine Learning Engineer", "conversation_topic": "professional career", "status": "inactive"},
]

ALL_MEMORIES: List[Dict[str, Any]] = BASE_MEMORIES + HISTORICAL_MEMORIES


def active_memories() -> List[Dict[str, Any]]:
    return [dict(m) for m in BASE_MEMORIES]


def all_memories() -> List[Dict[str, Any]]:
    """Memories in storage order, including inactive historical ones."""
    return [dict(m) for m in ALL_MEMORIES]


def memory_by_id(memory_id: str) -> Dict[str, Any]:
    for m in ALL_MEMORIES:
        if m["id"] == memory_id:
            return dict(m)
    raise KeyError(f"Unknown fixture memory id: {memory_id}")


def value_by_id(memory_id: str) -> str:
    return memory_by_id(memory_id)["value"]
