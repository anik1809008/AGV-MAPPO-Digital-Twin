from src.evaluation.experiment_config import (
    DEFAULT_MAX_STEPS,
    EXPERIMENT_SEEDS,
    LATENCY_LEVELS,
    METHODS,
    PERFECT_EXECUTION_PROBABILITY,
    PRIMARY_AGENT_COUNTS,
    TRAINING_BUDGETS,
    TRAINING_METHODS,
    UNCERTAINTY_CONDITIONS,
    UNCERTAIN_EXECUTION_PROBABILITY,
    expand_uncertainty_conditions,
    build_experiment_matrix,
    expand_uncertainty_conditions,
    build_experiment_matrix,
    build_validation_matrix,
)
def test_experiment_configuration():
    assert LATENCY_LEVELS == [0, 1, 2, 3, 4]

    assert PRIMARY_AGENT_COUNTS == [8, 20]
    assert TRAINING_BUDGETS == [
        5,
        10,
        20,
        40,
    ]
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
def test_expand_uncertainty_conditions():
    expanded = expand_uncertainty_conditions()

    assert len(expanded) == 10

    assert {
        "name": "perfect",
        "latency": 0,
        "immediate_probability": 1.0,
    } in expanded

    assert {
        "name": "execution_only",
        "latency": 0,
        "immediate_probability": 0.8,
    } in expanded

    for latency in [1, 2, 3, 4]:
        assert {
            "name": "latency_only",
            "latency": latency,
            "immediate_probability": 1.0,
        } in expanded

        assert {
            "name": "combined",
            "latency": latency,
            "immediate_probability": 0.8,
        } in expanded
def test_build_experiment_matrix():
    matrix = build_experiment_matrix(
        methods=["M2", "M3"],
        agent_counts=[8],
        seeds=[0, 1],
        scenario_ids=[21, 22],
    )

    assert len(matrix) == 80

    first = matrix[0]

    assert first["method"] == "M2"
    assert first["agent_count"] == 8
    assert first["seed"] == 0
    assert first["scenario_id"] == 21

    assert first["uncertainty_condition"] in {
        "perfect",
        "latency_only",
        "execution_only",
        "combined",
    }

    assert first["latency"] in [0, 1, 2, 3, 4]

    assert first["immediate_probability"] in {
        1.0,
        0.8,
    }
def test_build_validation_matrix():
    matrix = build_validation_matrix(
        methods=["M3"],
        agent_counts=[8],
        seeds=[0],
        scenario_ids=[16, 17],
        training_budgets=[1, 5],
    )

    assert len(matrix) == 40

    first = matrix[0]

    assert first["method"] == "M3"
    assert first["agent_count"] == 8
    assert first["seed"] == 0
    assert first["scenario_id"] == 16
    assert first["episodes_per_scenario"] == 1

    assert first["uncertainty_condition"] in {
        "perfect",
        "latency_only",
        "execution_only",
        "combined",
    }

    assert first["latency"] in [0, 1, 2, 3, 4]

    assert first["immediate_probability"] in {
        1.0,
        0.8,
    }

    assert {
        row["episodes_per_scenario"]
        for row in matrix
    } == {1, 5}
