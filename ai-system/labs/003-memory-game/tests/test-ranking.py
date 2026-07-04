import os
import json
import sys
from datetime import datetime, timezone, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.experiment_002 import MemoryRepository, MemoryScorer, MemoryRanker

def main():
    labs_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    memory_path = os.path.join(labs_dir, "data", "tests", "output-ranking.json")
    
    # Ensure starting clean
    if os.path.exists(memory_path):
        os.remove(memory_path)
        
    repo = MemoryRepository(memory_path)
    scorer = MemoryScorer()
    ranker = MemoryRanker(scorer)

    now = datetime.now(timezone.utc)
    
    # Memory A: High semantic, High freq, Old (8 months)
    mem_a = {
        "id": "mem-a",
        "type": "project",
        "key": "project_name",
        "value": "ChunkdUp",
        "frequency": 200,
        "created_at": (now - timedelta(days=240)).isoformat().replace("+00:00", "Z"),
        "updated_at": (now - timedelta(days=240)).isoformat().replace("+00:00", "Z"),
        "source": "conversation",
        "status": "active"
    }
    
    # Memory B: Low semantic, Low freq, Very recent (2 minutes)
    mem_b = {
        "id": "mem-b",
        "type": "preference",
        "key": "drink",
        "value": "coffee",
        "frequency": 1,
        "created_at": (now - timedelta(minutes=2)).isoformat().replace("+00:00", "Z"),
        "updated_at": (now - timedelta(minutes=2)).isoformat().replace("+00:00", "Z"),
        "source": "conversation",
        "status": "active"
    }

    # Memory C: Mid semantic, Mid freq, Mid age (14 days)
    mem_c = {
        "id": "mem-c",
        "type": "tool",
        "key": "editor",
        "value": "Neovim",
        "frequency": 5,
        "created_at": (now - timedelta(days=14)).isoformat().replace("+00:00", "Z"),
        "updated_at": (now - timedelta(days=14)).isoformat().replace("+00:00", "Z"),
        "source": "conversation",
        "status": "active"
    }

    # Injecting directly to repository to bypass extraction/managers
    # We want to test just the ranking logic on the stored dataset.
    repo.save([mem_a, mem_b, mem_c])

    # Now load and rank
    memories = repo.get_all()
    print("--- Unranked Memories (Insertion Order) ---")
    for m in memories:
        print(f"Type: {m['type']}, Value: {m['value']}")

    ranked = ranker.rank(memories)
    
    print("\n--- Ranked Memories (Composite Order) ---")
    for m in ranked:
        print(f"Type: {m['type']} | Value: {m['value']} | Composite Score: {m['_ranking_debug']['composite_score']}")
        print(f"   -> Semantic: {m['_ranking_debug']['semantic']} | Freq Bonus: {m['_ranking_debug']['frequency_bonus']} | Recency Bonus: {m['_ranking_debug']['recency_bonus']}")

    # Save output to inspect JSON
    repo.save(ranked)

if __name__ == "__main__":
    main()
