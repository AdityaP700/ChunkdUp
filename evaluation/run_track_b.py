# evaluation/run_track_b.py
"""
Track B: End-to-End Conversational Memory Lifecycle & Retention Evaluation.

Evaluates ChunkdUp's memory runtime across:
1. Dialogue Turn Ingestion (Level 0 Turn Log)
2. Eligibility Filtering (Level 1)
3. Fact Extraction (Level 2: Rules Baseline vs. Groq LLM Extractor)
4. Schema Validation & Provenance Linkage (Level 3)
5. Decision Engine & Temporal Supersession (Level 4: STORE/MERGE/UPDATE/IGNORE)
6. Grounded Retrieval & Evidence Retention

Compares:
  - Baseline Lifecycle: Rules / Regex pattern extractor
  - Neural Lifecycle: Groq LLM extractor (with schema validation & fallback)

Generates evaluation/results/track_b_lifecycle_report.json
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Path setup
REPO_ROOT = Path(__file__).resolve().parent.parent
AI_SYSTEM_DIR = REPO_ROOT / "ai-system"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(AI_SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(AI_SYSTEM_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(str(REPO_ROOT / ".env"))
except ImportError:
    pass

from chunkdup.memory import Memory
from evaluation.adapters.locomo_adapter import LoCoMoLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("chunkdup.track_b")

RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Synthetic Benchmark: Temporal Progression & Evolution
# ---------------------------------------------------------------------------

SYNTHETIC_CONVERSATION = [
    {
        "turn_id": "turn-1",
        "speaker": "User",
        "text": "I work at Stripe as a senior infrastructure engineer.",
        "timestamp": "2026-03-01T09:00:00Z",
    },
    {
        "turn_id": "turn-2",
        "speaker": "User",
        "text": "For our primary cloud deployment, we use AWS and Docker.",
        "timestamp": "2026-03-01T09:05:00Z",
    },
    {
        "turn_id": "turn-3",
        "speaker": "User",
        "text": "My primary programming language is Python.",
        "timestamp": "2026-03-01T09:10:00Z",
    },
    {
        "turn_id": "turn-4",
        "speaker": "User",
        "text": "I'm building a project called ChunkdUp for persistent AI memory.",
        "timestamp": "2026-03-01T09:15:00Z",
    },
    {
        "turn_id": "turn-5",
        "speaker": "User",
        "text": "We recently migrated our infrastructure: I now use GCP instead of AWS.",
        "timestamp": "2026-03-02T14:00:00Z",
    },
    {
        "turn_id": "turn-6",
        "speaker": "User",
        "text": "I now use Rust for backend performance instead of Python.",
        "timestamp": "2026-03-02T14:30:00Z",
    },
]

SYNTHETIC_EVAL_QUERIES = [
    {
        "query": "Where do I work and what is my role?",
        "expected_fact_keys": ["company", "role"],
        "expected_values": ["Stripe", "infrastructure engineer"],
        "expected_turn_ids": ["turn-1"],
    },
    {
        "query": "What cloud provider do I use currently?",
        "expected_fact_keys": ["cloud_provider"],
        "expected_values": ["GCP"],
        "expected_turn_ids": ["turn-5"],
    },
    {
        "query": "What programming language do I use now?",
        "expected_fact_keys": ["programming_language"],
        "expected_values": ["Rust"],
        "expected_turn_ids": ["turn-6"],
    },
    {
        "query": "What open source project am I building?",
        "expected_fact_keys": ["project_name"],
        "expected_values": ["ChunkdUp"],
        "expected_turn_ids": ["turn-4"],
    },
]


def run_synthetic_lifecycle_eval(extractor_mode: str) -> Dict[str, Any]:
    """
    Run the synthetic conversation through raw turn ingestion, fact extraction,
    persistence, temporal supersession, and retrieval.
    """
    mem = Memory(
        store="memory",
        extractor=extractor_mode,
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    )

    ingestion_decisions = []
    t_start = time.perf_counter()

    for turn in SYNTHETIC_CONVERSATION:
        use_groq = extractor_mode == "groq"
        res = mem.remember(
            text=turn["text"],
            speaker=turn["speaker"],
            timestamp=turn["timestamp"],
            use_groq=use_groq,
        )
        ingestion_decisions.append({
            "turn_id": turn["turn_id"],
            "text": turn["text"],
            "memories": res.get("memories", []),
        })

    ingest_duration = time.perf_counter() - t_start

    # Evaluate active memory state
    active_memories = mem.get_all()
    active_by_key = {m.get("key"): m for m in active_memories if m.get("key")}

    # Evaluate queries
    query_eval_results = []
    for q in SYNTHETIC_EVAL_QUERIES:
        results = mem.retrieve(q["query"], limit=5)
        retrieved_keys = [r.get("key") for r in results]
        retrieved_values = [str(r.get("value", "")).lower() for r in results]
        retrieved_turn_ids = [r.get("source_turn_id") for r in results if r.get("source_turn_id")]

        keys_found = any(k in retrieved_keys for k in q["expected_fact_keys"])
        values_found = any(any(exp.lower() in v for v in retrieved_values) for exp in q["expected_values"])
        provenance_valid = any(tid in retrieved_turn_ids for tid in q["expected_turn_ids"])

        success = keys_found and (values_found or provenance_valid)

        query_eval_results.append({
            "query": q["query"],
            "success": success,
            "keys_found": keys_found,
            "values_found": values_found,
            "provenance_valid": provenance_valid,
            "retrieved_count": len(results),
        })

    # Historical query as of Turn 3 (prior to migration)
    pit_results = mem.retrieve("What cloud provider do I use?", as_of="2026-03-01T12:00:00Z")
    pit_values = [str(r.get("value", "")).lower() for r in pit_results]
    pit_aws_found = any("aws" in v for v in pit_values)

    success_count = sum(1 for q in query_eval_results if q["success"])

    return {
        "extractor_mode": extractor_mode,
        "total_turns": len(SYNTHETIC_CONVERSATION),
        "active_memory_count": len(active_memories),
        "ingest_duration_seconds": round(ingest_duration, 2),
        "queries_evaluated": len(SYNTHETIC_EVAL_QUERIES),
        "queries_successful": success_count,
        "success_rate": round(success_count / len(SYNTHETIC_EVAL_QUERIES), 4),
        "point_in_time_reconstruction_correct": pit_aws_found,
        "query_results": query_eval_results,
        "active_keys": list(active_by_key.keys()),
    }


# ---------------------------------------------------------------------------
# LoCoMo Benchmark Track B: Conversational Evidence Retention
# ---------------------------------------------------------------------------

def run_locomo_track_b_eval(
    data_path: Path,
    extractor_mode: str,
    max_turns: int = 20,
    max_questions: int = 5,
) -> Dict[str, Any]:
    """
    Run a real LoCoMo conversational sample through Track B end-to-end memory lifecycle.
    """
    samples, _ = LoCoMoLoader.load(
        data_path,
        smoke=True,
        smoke_sample_ids=["conv-26"],
        smoke_max_questions=max_questions,
    )
    if not samples:
        return {"error": "No samples loaded"}

    sample = samples[0]
    turns_to_feed = sample.turns[:max_turns]

    mem = Memory(
        store="memory",
        extractor=extractor_mode,
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    )

    t_start = time.perf_counter()
    turn_decisions = {}
    turn_stored_keys = {}

    for turn in turns_to_feed:
        text = f"{turn.speaker}: {turn.text}"
        use_groq = extractor_mode == "groq"
        res = mem.remember(
            text=text,
            session_id=f"session_{turn.session_num}",
            speaker=turn.speaker,
            timestamp=turn.date_time,
            use_groq=use_groq,
        )

        decisions = []
        keys = []
        for m in res.get("memories", []):
            dec = m.get("decision", "IGNORE").upper()
            decisions.append(dec)
            if dec in ("STORE", "MERGE", "UPDATE") and m.get("key"):
                keys.append(m["key"])

        turn_decisions[turn.dia_id] = decisions
        turn_stored_keys[turn.dia_id] = keys

    ingest_time = time.perf_counter() - t_start
    active_memories = mem.get_all()
    active_keys = {m.get("key") for m in active_memories if m.get("key")}

    # Evaluate question evidence retention
    questions_evaluated = []
    for q in sample.questions[:max_questions]:
        if not q.is_answerable or not q.evidence_dia_ids:
            continue

        retrieved = mem.retrieve(q.question, limit=5)
        retrieved_keys = {r.get("key") for r in retrieved if r.get("key")}
        retrieved_turn_ids = {r.get("source_turn_id") for r in retrieved if r.get("source_turn_id")}

        # Evaluate lifecycle status for each evidence turn
        evidence_statuses = []
        for ev_id in q.evidence_dia_ids:
            # Check if this evidence turn was fed
            if not any(t.dia_id == ev_id for t in turns_to_feed):
                evidence_statuses.append("EVIDENCE_NOT_IN_SESSION_WINDOW")
                continue

            decisions = turn_decisions.get(ev_id, ["IGNORE"])
            keys = turn_stored_keys.get(ev_id, [])

            if "STORE" not in decisions and "MERGE" not in decisions and "UPDATE" not in decisions:
                status = "RETENTION_NEVER_EXTRACTED"
            elif any(k in retrieved_keys for k in keys):
                status = "SUCCESS"
            elif any(k in active_keys for k in keys):
                status = "RETRIEVAL_RANKED_TOO_LOW"
            else:
                status = "RETENTION_OVERWRITTEN"

            evidence_statuses.append(status)

        questions_evaluated.append({
            "question": q.question,
            "category": q.category_name,
            "evidence_dia_ids": q.evidence_dia_ids,
            "evidence_statuses": evidence_statuses,
            "retrieved_count": len(retrieved),
        })

    return {
        "extractor_mode": extractor_mode,
        "turns_ingested": len(turns_to_feed),
        "active_memory_count": len(active_memories),
        "ingest_duration_seconds": round(ingest_time, 2),
        "questions_evaluated": questions_evaluated,
    }


def main():
    logger.info("Executing ChunkdUp Track B Lifecycle & Retention Benchmark...")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "groq_api_available": bool(os.getenv("GROQ_API_KEY")),
        "groq_model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        "synthetic_benchmark": {},
        "locomo_benchmark": {},
    }

    # 1. Synthetic Benchmark Comparison
    logger.info("Running Synthetic Benchmark (Rules Baseline vs Groq LLM)...")
    results["synthetic_benchmark"]["rules_baseline"] = run_synthetic_lifecycle_eval("rules")
    if results["groq_api_available"]:
        results["synthetic_benchmark"]["groq_llm"] = run_synthetic_lifecycle_eval("groq")

    # 2. LoCoMo Track B Comparison
    locomo_path = REPO_ROOT / "evaluation" / "data" / "locomo10.json"
    if locomo_path.exists():
        logger.info("Running LoCoMo Track B Sample (Rules Baseline vs Groq LLM)...")
        results["locomo_benchmark"]["rules_baseline"] = run_locomo_track_b_eval(locomo_path, "rules", max_turns=18, max_questions=4)
        if results["groq_api_available"]:
            results["locomo_benchmark"]["groq_llm"] = run_locomo_track_b_eval(locomo_path, "groq", max_turns=18, max_questions=4)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / "track_b_lifecycle_report.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(f"Saved Track B report to {out_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("TRACK B END-TO-END MEMORY LIFECYCLE & RETENTION SCOREBOARD")
    print("=" * 80)
    synth = results.get("synthetic_benchmark", {})
    for mode, data in synth.items():
        print(f"[{mode.upper()}] Success Rate: {data['success_rate']*100:.1f}% ({data['queries_successful']}/{data['queries_evaluated']}) | "
              f"Active Memories: {data['active_memory_count']} | PIT Reconstructed: {data['point_in_time_reconstruction_correct']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
