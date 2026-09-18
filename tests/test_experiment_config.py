from src.evaluation.experiment_config import (
    DEFAULT_MAX_STEPS,
    EXPERIMENT_SEEDS,
    LATENCY_LEVELS,
    METHODS,
    PERFECT_EXECUTION_PROBABILITY,
    PRIMARY_AGENT_COUNTS,
    TRAINING_METHODS,
    UNCERTAINTY_CONDITIONS,
    UNCERTAIN_EXECUTION_PROBABILITY,
)


def test_experiment_configuration():
    assert LATENCY_LEVELS == [0, 1, 2, 3, 4]

    assert PRIMARY_AGENT_COUNTS == [8, 20]

    assert METHODS == [
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
        "M6",
    ]

    assert TRAINING_METHODS == [
        "M2",
        "M3",
        "M6",
    ]

    assert EXPERIMENT_SEEDS == [
        0,
        1,
        2,
        3,
        4,
    ]

    assert PERFECT_EXECUTION_PROBABILITY == 1.0
    assert UNCERTAIN_EXECUTION_PROBABILITY == 0.8

    assert DEFAULT_MAX_STEPS == 200


def test_uncertainty_conditions():
    assert UNCERTAINTY_CONDITIONS["perfect"] == {
        "latency": 0,
        "immediate_probability": 1.0,
    }

    assert UNCERTAINTY_CONDITIONS["latency_only"] == {
        "latency": None,
        "immediate_probability": 1.0,
    }

    assert UNCERTAINTY_CONDITIONS["execution_only"] == {
        "latency": 0,
        "immediate_probability": 0.8,
    }

    assert UNCERTAINTY_CONDITIONS["combined"] == {
        "latency": None,
        "immediate_probability": 0.8,
    }
