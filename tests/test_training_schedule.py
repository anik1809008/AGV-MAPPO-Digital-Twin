from src.marl.training_schedule import (
    build_training_schedule,
)


def test_training_schedule_uses_full_training_split():
    schedule = build_training_schedule()

    assert schedule == list(
        range(1, 16)
    )

    assert len(schedule) == 15
    assert schedule[0] == 1
    assert schedule[-1] == 15
