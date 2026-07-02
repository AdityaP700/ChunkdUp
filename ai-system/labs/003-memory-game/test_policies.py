import os
import sys

# Ensure we can import from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policies import Decision
# We need DecisionEngine which is currently inside experiment_002.py, 
# but testing it this way might be circular. Let's just define a mock engine here 
# or import it if possible.
# Actually, the user just wants the test cases moved out. I'll write a clean test script.

from experiments.experiment_002 import DecisionEngine

def run_tests():
    print("\n--- Running DecisionEngine Test Cases ---")
    engine = DecisionEngine()

    # Case 1: Brand new memory
    res1 = engine.decide(None, {"type": "environment", "key": "os", "value": "Windows"})
    print(f"Case 1: {res1.name} == Expected: STORE")

    # Case 2: Exact duplicate memory
    res2 = engine.decide(
        {"type": "environment", "key": "os", "value": "Windows"}, 
        {"type": "environment", "key": "os", "value": "Windows"}
    )
    print(f"Case 2: {res2.name} == Expected: IGNORE")

    # Case 3: Memory value changed
    res3 = engine.decide(
        {"type": "environment", "key": "os", "value": "Windows"}, 
        {"type": "environment", "key": "os", "value": "Linux"}
    )
    print(f"Case 3: {res3.name} == Expected: UPDATE")

    # Case 4: Different memory type/key entirely
    res4 = engine.decide(
        {"type": "environment", "key": "os", "value": "Windows"}, 
        {"type": "tool", "key": "editor", "value": "VS Code"}
    )
    print(f"Case 4: {res4.name} == Expected: STORE")
    print("-----------------------------------------\n")

if __name__ == "__main__":
    run_tests()
