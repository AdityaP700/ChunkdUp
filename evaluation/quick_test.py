#!/usr/bin/env python
"""Quick test to verify evaluation framework works."""

from runner import MemoryEvaluator
from pathlib import Path

def test_single_scenario():
    """Test a single scenario manually."""
    evaluator = MemoryEvaluator()

    # Test STORE
    result = evaluator.run_scenario({
        "id": "test_001",
        "conversation": ["I use Python"],
        "expected_decision": "STORE",
        "expected_key": "programming_language",
        "expected_value": "Python",
        "reason": "New memory"
    })

    print(f"Result: {result}")
    print(f"Correct: {result['correct']['decision']}")

if __name__ == "__main__":
    test_single_scenario()