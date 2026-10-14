# evaluation/run_track_b.py
"""
Track B: End-to-End Conversational Memory Lifecycle & Retention Evaluation.

Evaluates ChunkdUp's memory runtime across:
1. Dialogue Turn Ingestion (Level 0 Turn Log)
2. Eligibility Filtering (Level 1)
3. Fact Extraction (Level 2: Rules Baseline vs. Groq LLM Extractor)
4. Schema Validation & Provenance Linkage (Level 3)
5. Decision Engine & Temporal Supersession (Level 4: STORE/MERGE/UPDATE/IGNORE)
6. Grounded Retrieval & Evidence Retention across:
   - Baseline retrieval (unreranked / model-free)
   - Production retrieval (hybrid BM25 + dense + Cross-Encoder + calibrated gate)
   - Entity-augmented retrieval (with bounded 1-hop relation expansion)

Evaluates the complete development split:
  - conv-26 (56 turns, 199 questions)
  - conv-30 (40 turns, 105 questions)
  - Combined dev split (96 turns, 304 questions)

Generates evaluation/results/track_b_dev_report.json
"""

import argparse
from collections import Counter
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
from chunkdup.retrieval_production import ProductionRetriever
from evaluation.adapters.locomo_adapter import ConversationSample, LoCoMoLoader, QuestionItem

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("chunkdup.track_b")

RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Metric Utilities
# ---------------------------------------------------------------------------

def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth string."""
    pred_tokens = re.findall(r"\w+", str(prediction).lower())
    gt_tokens = re.findall(r"\w+", str(ground_truth).lower())
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


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
            turn_id=turn["turn_id"],
            use_groq=use_groq,
        )
        ingestion_decisions.append({
            "turn_id": turn["turn_id"],
            "text": turn["text"],
            "memories": res.get("memories", []),
        })

    ingest_duration = time.perf_counter() - t_start
    active_memories = mem.get_all()
    active_by_key = {m.get("key"): m for m in active_memories if m.get("key")}

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
# LoCoMo Track B Development Evaluation Runner
# ---------------------------------------------------------------------------

def evaluate_sample_track_b(
    sample: ConversationSample,
    extractor_mode: str,
    max_turns: Optional[int] = None,
    max_questions: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Ingest a LoCoMo conversation into Memory and evaluate across:
      1. Baseline retrieval (baseline_current)
      2. Production retrieval (hybrid_ce_abstain)
      3. Entity-augmented retrieval (hybrid_ce_abstain + 1-hop expansion)
    """
    turns_to_ingest = sample.turns[:max_turns] if max_turns else sample.turns
    questions_to_eval = sample.questions[:max_questions] if max_questions else sample.questions

    mem = Memory(
        store="memory",
        extractor=extractor_mode,
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    )

    t_ingest_start = time.perf_counter()
    turn_decisions: Dict[str, List[str]] = {}
    extracted_facts_by_turn: Dict[str, List[str]] = {}

    for turn in turns_to_ingest:
        text = f"{turn.speaker}: {turn.text}"
        use_groq = extractor_mode == "groq"
        res = mem.remember(
            text=text,
            session_id=f"session_{turn.session_num}",
            speaker=turn.speaker,
            timestamp=turn.date_time,
            turn_id=turn.dia_id,
            use_groq=use_groq,
        )
        decs = [m.get("decision", "IGNORE").upper() for m in res.get("memories", [])]
        keys = [m.get("key") for m in res.get("memories", []) if m.get("key")]
        turn_decisions[turn.dia_id] = decs
        extracted_facts_by_turn[turn.dia_id] = keys

    ingest_time_s = time.perf_counter() - t_ingest_start
    active_memories = mem.get_all()
    all_stored_turns = {
        tid for tid, decs in turn_decisions.items()
        if any(d in ("STORE", "MERGE", "UPDATE") for d in decs)
    }

    # Set up the three retrieval configurations
    retrievers = {
        "baseline": ProductionRetriever(
            repository=mem.repository,
            config_name="baseline_current",
            query_log_path=None,
        ),
        "production": ProductionRetriever(
            repository=mem.repository,
            config_name="hybrid_ce_abstain",
            entity_graph=mem.entity_graph,
            query_log_path=None,
        ),
        "entity_augmented": ProductionRetriever(
            repository=mem.repository,
            config_name="hybrid_ce_abstain",
            entity_graph=mem.entity_graph,
            query_log_path=None,
        ),
    }
    retrievers["production"].entity_expansion_enabled = False
    retrievers["entity_augmented"].entity_expansion_enabled = True

    # Warm indices
    for ret in retrievers.values():
        ret._ensure_index()

    pipeline_results: Dict[str, Any] = {}

    for pipe_name, retriever in retrievers.items():
        answerable_total = 0
        evidence_hits = 0
        total_f1 = 0.0
        adversarial_total = 0
        adversarial_abstained = 0

        error_breakdown = {
            "EXTRACTION_FAILURE": 0,
            "RETRIEVAL_FAILURE": 0,
            "STALE_MEMORY": 0,
            "FALSE_ABSTENTION": 0,
        }

        category_metrics: Dict[int, Dict[str, Any]] = {
            cat: {"total": 0, "hits": 0, "total_f1": 0.0, "abstained": 0}
            for cat in range(1, 6)
        }

        for q in questions_to_eval:
            cat = q.category
            ground_truth_turns = set(q.evidence_dia_ids)

            # Retrieve top-5 results
            results = retriever.retrieve(q.question, k=5)
            retrieved_turn_ids = {r.get("source_turn_id") for r in results if r.get("source_turn_id")}
            abstained = len(results) == 0

            # Compute best token F1 between retrieved memory values and question answer
            best_f1 = 0.0
            for r in results:
                cand_text = f"{r.get('value', '')} {r.get('contextual_text', '')}"
                score = compute_token_f1(cand_text, str(q.answer))
                if score > best_f1:
                    best_f1 = score

            is_evidence_hit = bool(ground_truth_turns & retrieved_turn_ids) or best_f1 >= 0.5

            if cat in category_metrics:
                category_metrics[cat]["total"] += 1
                if is_evidence_hit:
                    category_metrics[cat]["hits"] += 1
                category_metrics[cat]["total_f1"] += best_f1
                if abstained:
                    category_metrics[cat]["abstained"] += 1

            if cat == 5:
                # Category 5: Adversarial / Unanswerable
                adversarial_total += 1
                if abstained:
                    adversarial_abstained += 1
            else:
                # Categories 1-4: Answerable questions
                answerable_total += 1
                if is_evidence_hit:
                    evidence_hits += 1
                total_f1 += best_f1

                # Classify error if question missed
                if not is_evidence_hit:
                    # Check if evidence turn was ever stored
                    has_stored_evidence = bool(ground_truth_turns & all_stored_turns)
                    if not has_stored_evidence:
                        error_breakdown["EXTRACTION_FAILURE"] += 1
                    elif abstained:
                        error_breakdown["FALSE_ABSTENTION"] += 1
                    else:
                        error_breakdown["RETRIEVAL_FAILURE"] += 1

        acc = round(evidence_hits / max(1, answerable_total), 4)
        avg_f1 = round(total_f1 / max(1, answerable_total), 4)
        abstain_rate = round(adversarial_abstained / max(1, adversarial_total), 4)

        pipeline_results[pipe_name] = {
            "answerable_questions": answerable_total,
            "evidence_hits": evidence_hits,
            "accuracy_recall_at_5": acc,
            "mean_token_f1": avg_f1,
            "adversarial_questions": adversarial_total,
            "adversarial_abstained": adversarial_abstained,
            "adversarial_abstention_rate": abstain_rate,
            "error_breakdown": error_breakdown,
            "categories": {
                cat: {
                    "total": m["total"],
                    "accuracy": round(m["hits"] / max(1, m["total"]), 4),
                    "mean_f1": round(m["total_f1"] / max(1, m["total"]), 4),
                    "abstention_rate": round(m["abstained"] / max(1, m["total"]), 4),
                }
                for cat, m in category_metrics.items() if m["total"] > 0
            },
        }

    return {
        "sample_id": sample.sample_id,
        "extractor_mode": extractor_mode,
        "turns_ingested": len(turns_to_ingest),
        "active_memory_count": len(active_memories),
        "ingest_time_seconds": round(ingest_time_s, 2),
        "questions_evaluated": len(questions_to_eval),
        "pipelines": pipeline_results,
    }


def aggregate_sample_metrics(samples_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-sample results into combined split metrics."""
    if not samples_results:
        return {}

    combined_pipelines: Dict[str, Any] = {}
    pipe_names = list(samples_results[0]["pipelines"].keys())

    for pipe in pipe_names:
        tot_ans = sum(s["pipelines"][pipe]["answerable_questions"] for s in samples_results)
        tot_hits = sum(s["pipelines"][pipe]["evidence_hits"] for s in samples_results)
        tot_adv = sum(s["pipelines"][pipe]["adversarial_questions"] for s in samples_results)
        tot_adv_abs = sum(s["pipelines"][pipe]["adversarial_abstained"] for s in samples_results)

        err_agg = {
            "EXTRACTION_FAILURE": sum(s["pipelines"][pipe]["error_breakdown"]["EXTRACTION_FAILURE"] for s in samples_results),
            "RETRIEVAL_FAILURE": sum(s["pipelines"][pipe]["error_breakdown"]["RETRIEVAL_FAILURE"] for s in samples_results),
            "STALE_MEMORY": sum(s["pipelines"][pipe]["error_breakdown"]["STALE_MEMORY"] for s in samples_results),
            "FALSE_ABSTENTION": sum(s["pipelines"][pipe]["error_breakdown"]["FALSE_ABSTENTION"] for s in samples_results),
        }

        # Weighted F1
        w_f1 = sum(s["pipelines"][pipe]["mean_token_f1"] * s["pipelines"][pipe]["answerable_questions"] for s in samples_results) / max(1, tot_ans)

        combined_pipelines[pipe] = {
            "answerable_questions": tot_ans,
            "evidence_hits": tot_hits,
            "accuracy_recall_at_5": round(tot_hits / max(1, tot_ans), 4),
            "mean_token_f1": round(w_f1, 4),
            "adversarial_questions": tot_adv,
            "adversarial_abstained": tot_adv_abs,
            "adversarial_abstention_rate": round(tot_adv_abs / max(1, tot_adv), 4),
            "error_breakdown": err_agg,
        }

    return {
        "total_samples": len(samples_results),
        "total_turns": sum(s["turns_ingested"] for s in samples_results),
        "total_active_memories": sum(s["active_memory_count"] for s in samples_results),
        "total_questions": sum(s["questions_evaluated"] for s in samples_results),
        "pipelines": combined_pipelines,
    }


def main():
    parser = argparse.ArgumentParser(description="ChunkdUp Track B Development Split Benchmark")
    parser.add_argument("--max-turns", type=int, default=None, help="Cap turns per conversation")
    parser.add_argument("--max-questions", type=int, default=None, help="Cap questions per conversation")
    parser.add_argument("--skip-groq", action="store_true", help="Skip Groq extraction")
    args = parser.parse_args()

    locomo_path = REPO_ROOT / "evaluation" / "data" / "locomo10.json"
    if not locomo_path.exists():
        logger.error(f"Dataset not found: {locomo_path}")
        return 1

    samples, _ = LoCoMoLoader.load(locomo_path, split="dev")
    sample_ids = [s.sample_id for s in samples]
    total_turns_count = sum(len(s.turns) for s in samples)
    total_q_count = sum(len(s.questions) for s in samples)
    print(f"\nLoaded LoCoMo Development Split: {sample_ids} ({total_turns_count} turns, {total_q_count} questions)")

    has_groq = bool(os.getenv("GROQ_API_KEY")) and not args.skip_groq
    report: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dev_samples": sample_ids,
        "groq_api_available": has_groq,
        "synthetic_benchmark": {},
        "rules_extractor_eval": {},
        "groq_extractor_eval": {} if has_groq else None,
    }

    # 1. Synthetic Lifecycle Evaluation
    print("\n" + "=" * 60)
    print("1. Synthetic Lifecycle & Point-in-Time Evaluation")
    print("-" * 60)
    report["synthetic_benchmark"]["rules"] = run_synthetic_lifecycle_eval("rules")
    print(f"  Rules Baseline: Success Rate = {report['synthetic_benchmark']['rules']['success_rate']*100:.1f}%, PIT Reconstructed = {report['synthetic_benchmark']['rules']['point_in_time_reconstruction_correct']}")
    if has_groq:
        report["synthetic_benchmark"]["groq"] = run_synthetic_lifecycle_eval("groq")
        print(f"  Groq Neural:    Success Rate = {report['synthetic_benchmark']['groq']['success_rate']*100:.1f}%, PIT Reconstructed = {report['synthetic_benchmark']['groq']['point_in_time_reconstruction_correct']}")

    # 2. Rules Extractor on Dev Split (conv-26, conv-30)
    print("\n" + "=" * 60)
    print("2. Rules Extractor on Dev Split (conv-26 and conv-30)")
    print("-" * 60)
    rules_per_sample = []
    for sample in samples:
        print(f"  Evaluating {sample.sample_id} ({len(sample.turns)} turns, {len(sample.questions)} questions)...")
        res = evaluate_sample_track_b(
            sample=sample,
            extractor_mode="rules",
            max_turns=args.max_turns,
            max_questions=args.max_questions,
        )
        rules_per_sample.append(res)
        report["rules_extractor_eval"][sample.sample_id] = res

    rules_combined = aggregate_sample_metrics(rules_per_sample)
    report["rules_extractor_eval"]["combined_dev_split"] = rules_combined

    # 3. Groq Extractor on Dev Split (paired evaluation window)
    if has_groq:
        print("\n" + "=" * 60)
        print("3. Groq Neural Extractor on Dev Split")
        print("-" * 60)
        groq_per_sample = []
        # For live API, evaluate 25 turns and 25 questions per conversation to respect RPM
        groq_max_turns = args.max_turns or 25
        groq_max_q = args.max_questions or 25
        for sample in samples:
            print(f"  Evaluating {sample.sample_id} (turns={groq_max_turns}, Qs={groq_max_q})...")
            res = evaluate_sample_track_b(
                sample=sample,
                extractor_mode="groq",
                max_turns=groq_max_turns,
                max_questions=groq_max_q,
            )
            groq_per_sample.append(res)
            report["groq_extractor_eval"][sample.sample_id] = res

        groq_combined = aggregate_sample_metrics(groq_per_sample)
        report["groq_extractor_eval"]["combined_dev_split"] = groq_combined

    # Save final machine-readable report
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / "track_b_dev_report.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved Track B report to: {out_file}")

    # Print Final Scoreboard
    print("\n" + "=" * 80)
    print(f"{'Configuration':<28} | {'Recall@5':<10} | {'Token F1':<10} | {'Adv Abstain':<12} | {'Top Error':<16}")
    print("-" * 80)

    comb = report["rules_extractor_eval"]["combined_dev_split"]["pipelines"]
    for pipe, pdata in comb.items():
        top_err = max(pdata["error_breakdown"].items(), key=lambda x: x[1])[0]
        print(f"Rules + {pipe:<19} | {pdata['accuracy_recall_at_5']*100:<9.1f}% | {pdata['mean_token_f1']*100:<9.1f}% | {pdata['adversarial_abstention_rate']*100:<11.1f}% | {top_err}")

    if has_groq:
        comb_g = report["groq_extractor_eval"]["combined_dev_split"]["pipelines"]
        for pipe, pdata in comb_g.items():
            top_err = max(pdata["error_breakdown"].items(), key=lambda x: x[1])[0]
            print(f"Groq + {pipe:<20} | {pdata['accuracy_recall_at_5']*100:<9.1f}% | {pdata['mean_token_f1']*100:<9.1f}% | {pdata['adversarial_abstention_rate']*100:<11.1f}% | {top_err}")

    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
