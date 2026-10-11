# experiments/retrieval/validate_dataset.py
"""Validates dataset.jsonl against fixture.py: unique ids, known categories,
every expected memory id exists, split assignment, category coverage."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fixture import ALL_MEMORIES, BASE_MEMORIES

REQUIRED_CATEGORIES = {
    "exact_lookup", "paraphrase", "short_ambiguous", "indirect_contextual",
    "multi_memory", "updated_facts", "project_specific", "out_of_domain",
    "absent_answer",
}
KNOWN_CATEGORIES = REQUIRED_CATEGORIES

REQUIRED_QUERIES = {
    "What is the weather forecast today?",
    "Where does ChunkdUp run?",
    "What database does ChunkdUp use?",
    "What technology does it use for vector search?",
    "What programming languages do I use?" ,
    "What editor or IDE do I prefer?",
}


def load_rows(extra_path=None):
    base = Path(__file__).parent / "dataset.jsonl"
    rows = [json.loads(line) for line in base.read_text(encoding="utf-8").splitlines() if line.strip()]
    if extra_path:
        extra = Path(extra_path)
        rows.extend(json.loads(line) for line in extra.read_text(encoding="utf-8").splitlines() if line.strip())
    return rows


def main() -> int:
    extra_path = sys.argv[1] if len(sys.argv) > 1 else None
    rows = load_rows(extra_path)
    errors = []
    allow_new_splits = extra_path is not None  # named slices may declare their own split

    ids = [r["id"] for r in rows]
    dupes = [i for i, c in Counter(ids).items() if c > 1]
    if dupes:
        errors.append(f"Duplicate query ids: {dupes}")

    fixture_ids = {m["id"] for m in ALL_MEMORIES}
    active_ids = {m["id"] for m in BASE_MEMORIES}
    for r in rows:
        for mid in r["relevant_memory_ids"]:
            if mid not in fixture_ids:
                errors.append(f"{r['id']}: unknown memory id {mid}")
            elif mid not in active_ids:
                errors.append(f"{r['id']}: expected memory {mid} is not active in fixture")
        if r["answerable"] != bool(r["relevant_memory_ids"]):
            errors.append(f"{r['id']}: answerable={r['answerable']} but relevant_memory_ids={r['relevant_memory_ids']}")
        if r["expected_behavior"] not in {"results", "empty_result"}:
            errors.append(f"{r['id']}: bad expected_behavior {r['expected_behavior']}")
        if r["category"] not in KNOWN_CATEGORIES:
            errors.append(f"{r['id']}: unknown category {r['category']}")
        if r["split"] not in {"dev", "heldout"} and not (allow_new_splits and r["split"] == "entity_focus"):
            errors.append(f"{r['id']}: unknown split {r['split']}")

    categories = Counter(r["category"] for r in rows)
    missing = REQUIRED_CATEGORIES - set(categories)
    if missing:
        errors.append(f"Missing categories: {missing}")

    queries = {r["query"] for r in rows}
    for q in REQUIRED_QUERIES:
        if q not in queries:
            errors.append(f"Required query missing from dataset: {q!r}")

    splits = Counter(r["split"] for r in rows)
    answerable = Counter(r["answerable"] for r in rows)

    print(f"rows={len(rows)}  dev={splits['dev']}  heldout={splits['heldout']}  "
          f"answerable={answerable[True]}  unanswerable={answerable[False]}")
    for cat, n in sorted(categories.items()):
        print(f"  {cat:<20} {n}")

    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nDATASET OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
