"""
Phase 1 Diagnostic: Cross-Strategy Discrepancy Analysis.
Identifies questions where Strategy A fails but C, D, E, or F succeeds, and vice versa.
"""

import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
with open(RESULTS_DIR / "full_locomo_dev_evaluation.json", "r", encoding="utf-8") as f:
    records = json.load(f)

# Group records by (sample_id, question_idx) -> {strategy_id: record}
by_question = defaultdict(dict)
for r in records:
    key = (r["sample_id"], r["question_idx"])
    by_question[key][r["strategy_id"]] = r

# Analyze discrepancies
a_fail_others_succeed = defaultdict(list)
a_succeed_others_fail = defaultdict(list)

for key, strats in by_question.items():
    a_rec = strats.get("Strategy_A")
    if not a_rec:
        continue
    a_hit = bool(a_rec["hit_at_5"])

    for s_id in ["Strategy_C", "Strategy_D", "Strategy_E", "Strategy_F"]:
        s_rec = strats.get(s_id)
        if not s_rec:
            continue
        s_hit = bool(s_rec["hit_at_5"])

        if not a_hit and s_hit:
            a_fail_others_succeed[s_id].append({
                "sample_id": key[0],
                "question_idx": key[1],
                "category": a_rec["category"],
                "category_name": a_rec["category_name"],
                "question": a_rec["question"],
                "answer": a_rec["answer"],
                "expected": a_rec["expected_evidence_ids"],
                "a_retrieved": a_rec["retrieved_evidence_ids"],
                "s_retrieved": s_rec["retrieved_evidence_ids"],
            })
        elif a_hit and not s_hit:
            a_succeed_others_fail[s_id].append({
                "sample_id": key[0],
                "question_idx": key[1],
                "category": a_rec["category"],
                "question": a_rec["question"],
            })

print("=" * 80)
print("PHASE 1 DISCREPANCY ANALYSIS SUMMARY (DEV SPLIT: conv-26, conv-30)")
print("=" * 80)
print(f"Total Unique Questions: {len(by_question)}")
print("-" * 80)
for s_id in ["Strategy_C", "Strategy_D", "Strategy_E", "Strategy_F"]:
    print(f"Strategy A Fails (Hit@5=0) BUT {s_id} Succeeds (Hit@5=1): {len(a_fail_others_succeed[s_id])} questions")
    print(f"Strategy A Succeeds (Hit@5=1) BUT {s_id} Fails (Hit@5=0): {len(a_succeed_others_fail[s_id])} questions")
    print()

out_file = RESULTS_DIR / "strategy_discrepancies.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump({
        "a_fails_others_succeed": a_fail_others_succeed,
        "a_succeeds_others_fail_counts": {k: len(v) for k, v in a_succeed_others_fail.items()}
    }, f, indent=2)

print(f"Saved detailed discrepancy report to: {out_file}")

# Print sample questions where other strategies succeeded when A failed
print("\nSample Questions where C/D/E/F succeeded when A failed:")
seen_qs = set()
for s_id, q_list in a_fail_others_succeed.items():
    for q in q_list:
        if q["question"] not in seen_qs:
            print(f"[{s_id}] (Cat {q['category']}: {q['category_name']}) Q: {q['question']}")
            print(f"   Expected: {q['expected']} | A got: {q['a_retrieved'][:2]} | {s_id} got: {q['s_retrieved'][:2]}")
            seen_qs.add(q["question"])
            if len(seen_qs) >= 6:
                break
    if len(seen_qs) >= 6:
        break
