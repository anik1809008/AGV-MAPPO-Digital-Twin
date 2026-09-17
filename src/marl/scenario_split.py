SCENARIO_ROOT = "benchmarks/movingai/scen-random"

TRAIN_SCENARIO_IDS = tuple(range(1, 16))
VALIDATION_SCENARIO_IDS = tuple(range(16, 21))
TEST_SCENARIO_IDS = tuple(range(21, 26))


def get_random_scenario_path(scenario_id):
    if scenario_id < 1 or scenario_id > 25:
        raise ValueError(
            "scenario_id must be between 1 and 25"
        )

    return (
        f"{SCENARIO_ROOT}/"
        "warehouse-10-20-10-2-1-"
        f"random-{scenario_id}.scen"
    )
