import pytest

from scripts.analyze_hybrid_evaluation import (
    exact_mcnemar,
    paired_bootstrap_ci,
    analyze,
)


def test_mcnemar_equal_results():
    assert exact_mcnemar(0, 0) == 1.0


def test_mcnemar_balanced_disagreements():
    assert exact_mcnemar(1, 1) == 1.0


def test_mcnemar_strong_difference():
    assert exact_mcnemar(10, 0) < 0.05


def test_bootstrap_identical_differences():
    assert paired_bootstrap_ci([0] * 250) == (0.0, 0.0)


def test_analysis_rejects_unmatched_episode_ids():
    with pytest.raises(ValueError, match="Episode IDs"):
        analyze({"1": {}}, {"2": {}})
