#!/usr/bin/env python
"""Run memory policy evaluation and generate metrics."""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add parent directory and ai-system to path
sys.path.insert(0, str(Path(__file__).parent.parent / "ai-system"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from chunkdup import Memory
from chunkdup.in_memory_repository import InMemoryRepository


class MemoryEvaluator:
    """Evaluate memory policy decisions against expected outcomes."""

    def __init__(self):
        self.results = []
        self.memory = None

    def run_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single scenario and return the result."""
        # Reset memory for each scenario
        repo = InMemoryRepository()
        self.memory = Memory(store="memory", repository=repo)
        self.memory.clear()

        # Process each conversation line
        for line in scenario["conversation"]:
            self.memory.remember(line)

        # Get actual results
        all_memories = self.memory.get_all()

        # Determine what actually happened
        actual_decision = self._determine_decision(
            all_memories,
            scenario.get("expected_key"),
            scenario.get("expected_value")
        )

        # Check if expected memory exists
        memory_found = self._find_memory(
            all_memories,
            scenario.get("expected_key")
        )

        return {
            "scenario_id": scenario["id"],
            "input": scenario["conversation"],
            "expected": {
                "decision": scenario["expected_decision"],
                "key": scenario.get("expected_key"),
                "value": scenario.get("expected_value"),
                "frequency": scenario.get("expected_frequency", 1)
            },
            "actual": {
                "decision": actual_decision,
                "memory_found": memory_found is not None,
                "memory_value": memory_found.get("value") if memory_found else None,
                "frequency": memory_found.get("frequency") if memory_found else 0
            },
            "correct": {
                "decision": actual_decision == scenario["expected_decision"],
                "value": (
                    memory_found.get("value") == scenario.get("expected_value")
                    if memory_found and scenario.get("expected_value")
                    else not memory_found
                )
            },
            "reason": scenario.get("reason", "")
        }

    def _determine_decision(
        self,
        memories: List[Dict],
        expected_key: Optional[str],
        expected_value: Optional[str] = None
    ) -> str:
        """Infer what decision was made based on memory state."""
        if not memories:
            return "IGNORE"

        # Check if expected key exists
        if expected_key:
            for m in memories:
                if m.get("key") == expected_key:
                    if m.get("frequency", 1) > 1:
                        return "MERGE"
                    return "STORE"

        # Check if any memory exists
        if memories:
            return "STORE"

        return "IGNORE"

    def _find_memory(self, memories: List[Dict], key: Optional[str]) -> Optional[Dict]:
        """Find a memory by key."""
        if not key:
            return None
        for m in memories:
            if m.get("key") == key and m.get("status") == "active":
                return m
        return None

    def run_all(self, scenarios_path: str) -> Dict[str, Any]:
        """Run all scenarios from a JSON file."""
        with open(scenarios_path, "r") as f:
            data = json.load(f)

        scenarios = data["scenarios"]
        results = []

        for scenario in scenarios:
            result = self.run_scenario(scenario)
            results.append(result)

        # Calculate metrics
        metrics = self._calculate_metrics(results)

        return {
            "timestamp": datetime.now().isoformat(),
            "total_scenarios": len(results),
            "results": results,
            "metrics": metrics
        }

    def _calculate_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate metrics from results."""
        total = len(results)
        correct_decisions = sum(1 for r in results if r["correct"]["decision"])
        correct_values = sum(1 for r in results if r["correct"]["value"])

        # Breakdown by expected decision
        decisions = {}
        for r in results:
            expected = r["expected"]["decision"]
            if expected not in decisions:
                decisions[expected] = {"total": 0, "correct": 0}
            decisions[expected]["total"] += 1
            if r["correct"]["decision"]:
                decisions[expected]["correct"] += 1

        # Confusion matrix
        confusion = {}
        for r in results:
            expected = r["expected"]["decision"]
            actual = r["actual"]["decision"]
            key = f"{expected}→{actual}"
            confusion[key] = confusion.get(key, 0) + 1

        return {
            "accuracy": {
                "decision": round(correct_decisions / total * 100, 2) if total > 0 else 0,
                "value": round(correct_values / total * 100, 2) if total > 0 else 0
            },
            "by_decision": decisions,
            "confusion_matrix": confusion,
            "summary": {
                "total": total,
                "correct_decisions": correct_decisions,
                "incorrect_decisions": total - correct_decisions
            }
        }


def main():
    """Run the evaluation."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    evaluator = MemoryEvaluator()

    scenarios_path = Path(__file__).parent / "scenarios.json"
    results = evaluator.run_all(str(scenarios_path))

    # Save results
    output_path = Path(__file__).parent / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("📊 CHUNKDUP EVALUATION RESULTS")
    print("="*60)

    metrics = results["metrics"]
    print(f"\n📈 Overall Accuracy:")
    print(f"  Decision Accuracy: {metrics['accuracy']['decision']}%")
    print(f"  Value Accuracy: {metrics['accuracy']['value']}%")

    print(f"\n📋 By Decision Type:")
    for decision, stats in metrics["by_decision"].items():
        acc = round(stats["correct"] / stats["total"] * 100, 2)
        print(f"  {decision}: {stats['correct']}/{stats['total']} ({acc}%)")

    print(f"\n🔄 Confusion Matrix:")
    for key, count in metrics["confusion_matrix"].items():
        print(f"  {key}: {count}")

    print(f"\n📁 Results saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()