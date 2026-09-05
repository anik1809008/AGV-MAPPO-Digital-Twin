from src.evaluation.experiment_config import (
    LATENCY_LEVELS,
    M5_THRESHOLDS,
    PRIMARY_AGENT_COUNTS,
    METHODS,
)


def test_experiment_config_values():
    assert LATENCY_LEVELS == [0, 1, 2, 3, 4]
    assert M5_THRESHOLDS == [2, 3, 4, 5]
    assert PRIMARY_AGENT_COUNTS == [8, 20]
    assert METHODS == [
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
    ]
