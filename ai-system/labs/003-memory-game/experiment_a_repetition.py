import os
import json
import sys
#importing from the experiment_002
from experiments.experiment_002 import MemoryRepository, DecisionEngine, MemoryScorer, MemoryManager, MemoryExtractor

def main():
    labs_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # We use a dedicated memory file for this experiment
    memory_path = os.path.join(labs_dir, "data", "memory_repetition.json")

    # Ensure starting clean
    if os.path.exists(memory_path):
        os.remove(memory_path)

    repo = MemoryRepository(memory_path)
    engine = DecisionEngine()
    scorer = MemoryScorer()

    # A modified manager could go here, but we will use the one from experiment_002
    manager = MemoryManager(repo, engine, scorer)
    extractor = MemoryExtractor()

    # Simulate three separate conversations over time where the user states they use Linux.
    conversations = [
        "User: I use Linux.",
        "User: By the way, I use Linux at work.",
        "User: I use Linux on my laptop too."
    ]
#thats how the for loop is applied
    for i, conv in enumerate(conversations, 1):
        print(f"\n--- Conversation {i} ---")
        print(f"Input: {conv}")
        #memories is calling the extract method
        memories = extractor.extract(conv)

        for memory in memories:
            manager.process(memory)

    print("\n--- Final Repository State ---")
    print(json.dumps(repo.get_all(), indent=2))

if __name__ == "__main__":
    main()
