import os
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.experiment_002 import MemoryRepository, DecisionEngine, MemoryScorer, MemoryManager, MemoryExtractor

def main():
    labs_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # We use a dedicated memory file for this experiment
    memory_path = os.path.join(labs_dir, "data", "tests", "output-contradiction.json")
    
    # Ensure starting clean
    if os.path.exists(memory_path):
        os.remove(memory_path)
        
    repo = MemoryRepository(memory_path)
    engine = DecisionEngine()
    scorer = MemoryScorer()
    
    manager = MemoryManager(repo, engine, scorer)
    extractor = MemoryExtractor()

    # Simulate contradiction
    conversations = [
        "User: I use Windows.",
        "User: I use Linux."
    ]

    for i, conv in enumerate(conversations, 1):
        print(f"\n--- Conversation {i} ---")
        print(f"Input: {conv}")
        memories = extractor.extract(conv)
        
        for memory in memories:
            manager.process(memory)
            
    print("\n--- Final Repository State ---")
    print(json.dumps(repo.get_all(), indent=2))

if __name__ == "__main__":
    main()
