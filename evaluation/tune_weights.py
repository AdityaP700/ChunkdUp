# evaluation/tune_weights.py
"""
Tune composite MemoryScorer weights empirically against the benchmark evaluation suite.
"""

import sys
import os
import json
from typing import Dict, Any, List

# Ensure chunkdup package importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ai-system"))

from chunkdup.memory import Memory, MemoryScorer, MemoryManager, InMemoryRepository, DecisionEngine


def run_evaluation_with_weights(config: Dict[str, float], scenarios_path: str) -> Dict[str, Any]:
    with open(scenarios_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenarios = data.get("scenarios", [])
    correct_decisions = 0
    correct_values = 0
    total = len(scenarios)

    for scenario in scenarios:
        repo = InMemoryRepository()
        scorer = MemoryScorer(
            type_weight=config["type"],
            freq_weight=config["freq"],
            recency_weight=config["recency"]
        )
        memory = Memory(store="memory", repository=repo)
        memory.manager.scorer = scorer
        memory.clear()

        step_decisions = []
        for line in scenario["conversation"]:
            res = memory.remember(line)
            for m in res.get("memories", []):
                step_decisions.append(m)

        all_memories = memory.get_all()

        # Check decision for expected key
        expected_key = scenario.get("expected_key")
        expected_decision = scenario.get("expected_decision")
        expected_value = scenario.get("expected_value")

        actual_decision = "IGNORE"
        if step_decisions and expected_key:
            for step in reversed(step_decisions):
                if step.get("key") == expected_key:
                    actual_decision = step.get("decision", "STORE")
                    break

        if actual_decision == "IGNORE" and step_decisions:
            non_ignore = [s for s in step_decisions if s.get("decision") != "IGNORE"]
            if non_ignore:
                actual_decision = non_ignore[-1].get("decision", "STORE")

        if actual_decision == expected_decision:
            correct_decisions += 1

        # Check value
        memory_found = None
        if expected_key:
            for m in all_memories:
                if m.get("key") == expected_key and m.get("status") == "active":
                    memory_found = m
                    break

        val_correct = (
            memory_found.get("value") == expected_value
            if memory_found and expected_value
            else not memory_found
        )
        if val_correct:
            correct_values += 1

    decision_acc = (correct_decisions / total) * 100.0 if total > 0 else 0.0
    value_acc = (correct_values / total) * 100.0 if total > 0 else 0.0

    return {
        "config": config,
        "decision_accuracy": decision_acc,
        "value_accuracy": value_acc,
        "total": total,
        "correct_decisions": correct_decisions
    }


def tune_weights():
    """Run evaluation with different weight configurations to find optimal weights."""
    scenarios_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios.json")

    configs = [
        {"name": "Balanced", "type": 0.5, "freq": 0.3, "recency": 0.2},
        {"name": "Type-Heavy", "type": 0.6, "freq": 0.2, "recency": 0.2},
        {"name": "Frequency-Heavy", "type": 0.3, "freq": 0.4, "recency": 0.3},
        {"name": "Recency-Heavy", "type": 0.3, "freq": 0.3, "recency": 0.4},
        {"name": "Empirical Optimal", "type": 0.55, "freq": 0.25, "recency": 0.20},
    ]

    print("=" * 60)
    print("EMPIRICAL WEIGHT TUNING BENCHMARK")
    print("=" * 60)

    best_config = None
    best_score = -1.0

    for config in configs:
        res = run_evaluation_with_weights(config, scenarios_path)
        acc = res["decision_accuracy"]
        val_acc = res["value_accuracy"]
        print(f"Config: {config['name']:<18} | Type: {config['type']:.2f}, Freq: {config['freq']:.2f}, Recency: {config['recency']:.2f}")
        print(f"  -> Decision Accuracy: {acc:.2f}% ({res['correct_decisions']}/{res['total']}) | Value Accuracy: {val_acc:.2f}%\n")

        if acc > best_score:
            best_score = acc
            best_config = config

    print("=" * 60)
    print(f"Best Configuration: {best_config['name']} with {best_score:.2f}% Accuracy")
    print("=" * 60)
    return best_config


if __name__ == "__main__":
    tune_weights()
