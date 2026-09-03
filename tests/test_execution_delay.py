from src.environment.execution_delay import ExecutionDelayModel


def test_delay_is_only_zero_or_one():
    model = ExecutionDelayModel(
        immediate_probability=0.8,
        seed=42,
    )

    samples = [model.sample_delay() for _ in range(100)]

    assert set(samples).issubset({0, 1})


def test_always_immediate():
    model = ExecutionDelayModel(
        immediate_probability=1.0,
        seed=42,
    )

    samples = [model.sample_delay() for _ in range(20)]

    assert samples == [0] * 20


def test_always_delayed():
    model = ExecutionDelayModel(
        immediate_probability=0.0,
        seed=42,
    )

    samples = [model.sample_delay() for _ in range(20)]

    assert samples == [1] * 20
