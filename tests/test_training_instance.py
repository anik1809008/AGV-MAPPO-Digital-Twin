from src.marl.training_instance import (
    build_training_instance,
)


def test_build_training_instance_from_movingai():
    result = build_training_instance(
        map_path=(
            "benchmarks/movingai/"
            "warehouse-10-20-10-2-1.map"
        ),
        scenario_path=(
            "benchmarks/movingai/scen-random/"
            "warehouse-10-20-10-2-1-random-1.scen"
        ),
        agent_count=8,
    )

    assert len(result["starts"]) == 8
    assert len(result["goals"]) == 8

    assert len(result["grid"]) == 63
    assert len(result["grid"][0]) == 161

    assert result["starts"][0] == (143, 57)
    assert result["goals"][0] == (10, 16)


def test_build_training_instance_with_start_index():
    result = build_training_instance(
        map_path=(
            "benchmarks/movingai/"
            "warehouse-10-20-10-2-1.map"
        ),
        scenario_path=(
            "benchmarks/movingai/scen-random/"
            "warehouse-10-20-10-2-1-random-1.scen"
        ),
        agent_count=8,
        start_index=8,
    )

    assert len(result["starts"]) == 8
    assert len(result["goals"]) == 8

    assert result["starts"][0] == (21, 42)
    assert result["goals"][0] == (39, 37)


def test_invalid_training_instance_start_index():
    try:
        build_training_instance(
            map_path=(
                "benchmarks/movingai/"
                "warehouse-10-20-10-2-1.map"
            ),
            scenario_path=(
                "benchmarks/movingai/scen-random/"
                "warehouse-10-20-10-2-1-random-1.scen"
            ),
            agent_count=8,
            start_index=-1,
        )
    except ValueError as exc:
        assert str(exc) == (
            "start_index must be >= 0"
        )
    else:
        raise AssertionError(
            "Expected ValueError"
        )
def test_training_instance_rejects_out_of_range_request():
    map_path = (
        "benchmarks/movingai/"
        "warehouse-10-20-10-2-1.map"
    )

    scenario_path = (
        "benchmarks/movingai/scen-random/"
        "warehouse-10-20-10-2-1-random-1.scen"
    )

    try:
        build_training_instance(
            map_path=map_path,
            scenario_path=scenario_path,
            agent_count=20,
            start_index=990,
        )
    except ValueError as exc:
        assert (
            "scenario file does not contain enough agents"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Expected ValueError for out-of-range scenarios"
        )
