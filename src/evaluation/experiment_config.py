LATENCY_LEVELS = [0, 1, 2, 3, 4]

M5_THRESHOLDS = [2, 3, 4, 5]

PRIMARY_AGENT_COUNTS = [8, 20]

METHODS = [
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
]

TRAINING_METHODS = [
    "M2",
    "M3",
    "M6",
]

EXPERIMENT_SEEDS = [
    0,
    1,
    2,
    3,
    4,
]

PERFECT_EXECUTION_PROBABILITY = 1.0
UNCERTAIN_EXECUTION_PROBABILITY = 0.8

DEFAULT_MAX_STEPS = 200


UNCERTAINTY_CONDITIONS = {
    "perfect": {
        "latency": 0,
        "immediate_probability": (
            PERFECT_EXECUTION_PROBABILITY
        ),
    },
    "latency_only": {
        "latency": None,
        "immediate_probability": (
            PERFECT_EXECUTION_PROBABILITY
        ),
    },
    "execution_only": {
        "latency": 0,
        "immediate_probability": (
            UNCERTAIN_EXECUTION_PROBABILITY
        ),
    },
    "combined": {
        "latency": None,
        "immediate_probability": (
            UNCERTAIN_EXECUTION_PROBABILITY
        ),
    },
}
def expand_uncertainty_conditions():
    expanded = []

    for name, config in UNCERTAINTY_CONDITIONS.items():
        if config["latency"] is None:
            latencies = [
                latency
                for latency in LATENCY_LEVELS
                if latency > 0
            ]
        else:
            latencies = [config["latency"]]

        for latency in latencies:
            expanded.append(
                {
                    "name": name,
                    "latency": latency,
                    "immediate_probability": (
                        config["immediate_probability"]
                    ),
                }
            )

    return expanded
def build_experiment_matrix(
    methods,
    agent_counts,
    seeds,
    scenario_ids,
):
    matrix = []

    uncertainty_conditions = (
        expand_uncertainty_conditions()
    )

    for method in methods:
        for agent_count in agent_counts:
            for seed in seeds:
                for scenario_id in scenario_ids:
                    for condition in uncertainty_conditions:
                        matrix.append(
                            {
                                "method": method,
                                "agent_count": agent_count,
                                "seed": seed,
                                "scenario_id": scenario_id,
                                "uncertainty_condition": (
                                    condition["name"]
                                ),
                                "latency": condition[
                                    "latency"
                                ],
                                "immediate_probability": (
                                    condition[
                                        "immediate_probability"
                                    ]
                                ),
                            }
                        )

    return matrix
def build_validation_matrix(
    methods,
    agent_counts,
    seeds,
    scenario_ids,
    training_budgets,
):
    matrix = []

    uncertainty_conditions = (
        expand_uncertainty_conditions()
    )

    for method in methods:
        for agent_count in agent_counts:
            for seed in seeds:
                for scenario_id in scenario_ids:
                    for budget in training_budgets:
                        for condition in uncertainty_conditions:
                            matrix.append(
                                {
                                    "method": method,
                                    "agent_count": (
                                        agent_count
                                    ),
                                    "seed": seed,
                                    "scenario_id": (
                                        scenario_id
                                    ),
                                    "episodes_per_scenario": (
                                        budget
                                    ),
                                    "uncertainty_condition": (
                                        condition["name"]
                                    ),
                                    "latency": condition[
                                        "latency"
                                    ],
                                    "immediate_probability": (
                                        condition[
                                            "immediate_probability"
                                        ]
                                    ),
                                }
                            )

    return matrix
