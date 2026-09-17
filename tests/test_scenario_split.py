from src.marl.scenario_split import (
    TRAIN_SCENARIO_IDS,
    VALIDATION_SCENARIO_IDS,
    TEST_SCENARIO_IDS,
    get_random_scenario_path,
)


def test_scenario_splits_do_not_overlap():
    train = set(TRAIN_SCENARIO_IDS)
    validation = set(VALIDATION_SCENARIO_IDS)
    test = set(TEST_SCENARIO_IDS)

    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)


def test_scenario_splits_cover_all_random_files():
    combined = (
        set(TRAIN_SCENARIO_IDS)
        | set(VALIDATION_SCENARIO_IDS)
        | set(TEST_SCENARIO_IDS)
    )

    assert combined == set(range(1, 26))


def test_random_scenario_path():
    path = get_random_scenario_path(1)

    assert path.endswith(
        "warehouse-10-20-10-2-1-random-1.scen"
    )
