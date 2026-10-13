# tests/test_ablations.py
import pytest
from pathlib import Path
from evaluation.run_ablations import (
    load_frozen_rows,
    evaluate_frozen_suite_config,
)


def test_frozen_suite_ablation_smoke():
    """Verify that all 3 ablation configurations run successfully on the regression suite."""
    all_rows = load_frozen_rows()
    assert len(all_rows) >= 12

    # Fast 5-query slice for unit test
    smoke_rows = all_rows[:5]

    for cfg in ["legacy_baseline", "dense_abstain_ed", "hybrid_ce_abstain"]:
        res = evaluate_frozen_suite_config(cfg, smoke_rows)
        assert res["config_key"] == cfg
        assert res["n_queries"] == 5
        assert 0.0 <= res["MRR"] <= 1.0
        assert 0.0 <= res["P@5"] <= 1.0
        assert res["avg_latency_ms"] >= 0.0
