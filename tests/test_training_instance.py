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
