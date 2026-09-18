from src.marl.checkpoint_paths import (
    build_training_checkpoint_path,
)


def test_build_training_checkpoint_path():
    path = build_training_checkpoint_path(
        method="M6",
        agent_count=8,
        seed=2,
        scenario_id=4,
    )

    assert path == (
        "results/checkpoints/"
        "m6_agents8_seed2_scenario4.pt"
    )


def test_checkpoint_paths_change_with_configuration():
    first = build_training_checkpoint_path(
        method="M2",
        agent_count=8,
        seed=0,
        scenario_id=1,
    )

    second = build_training_checkpoint_path(
        method="M3",
        agent_count=20,
        seed=1,
        scenario_id=2,
    )

    assert first != second
