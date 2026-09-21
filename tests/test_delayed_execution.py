from src.environment.actions import Action
from src.environment.execution_delay import ExecutionDelayModel
from src.environment.delayed_execution import DelayedCommandExecutor


def test_delayed_command_executor():
    delay_model = ExecutionDelayModel(
        immediate_probability=0.0,
        seed=1,
    )

    executor = DelayedCommandExecutor(
        delay_model=delay_model,
    )

    executor.queue_commands(
        actions={
            0: Action.EAST,
            1: Action.WEST,
        },
        current_timestep=0,
    )

    assert executor.get_ready_actions(0) == {}

    ready = executor.get_ready_actions(1)

    assert ready == {
        0: Action.EAST,
        1: Action.WEST,
    }
def test_executor_uses_latest_ready_command():
    class FixedSequenceDelayModel:
        def __init__(self):
            self.delays = iter([1, 0])

        def sample_delay(self):
            return next(self.delays)

    executor = DelayedCommandExecutor(
        delay_model=FixedSequenceDelayModel(),
    )

    executor.queue_commands(
        actions={
            0: Action.NORTH,
        },
        current_timestep=0,
    )

    executor.queue_commands(
        actions={
            0: Action.EAST,
        },
        current_timestep=1,
    )

    ready = executor.get_ready_actions(
        current_timestep=1,
    )

    assert ready == {
        0: Action.EAST,
    }
