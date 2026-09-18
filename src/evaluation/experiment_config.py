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
