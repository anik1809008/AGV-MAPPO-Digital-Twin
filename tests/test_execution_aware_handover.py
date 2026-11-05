from src.environment.delayed_execution import DelayedCommandExecutor
from src.environment.execution_delay import ExecutionDelayModel
from src.environment.simulator import GroundTruthSimulator
from src.hybrid.execution_aware_handover import ExecutionAwareHandover


def make_system():
    simulator = GroundTruthSimulator(
        grid=[[0, 0, 0, 0]],
        agent_positions={0: (0, 0)},
        agent_goals={0: (3, 0)},
    )
    executor = DelayedCommandExecutor(
        ExecutionDelayModel(
            immediate_probability=0.0,
            seed=1,
        )
    )
    return simulator, executor


def test_delayed_command_drains_before_m7():
    simulator, executor = make_system()
    executor.queue_commands(
        actions={0: 3},
        current_timestep=0,
    )

    handover = ExecutionAwareHandover()
    handover.start()

    assert not handover.can_resume(0)

    result = handover.step(
        simulator, executor, timestep=1
    )

    assert result["ready"]
    assert not result["failed"]
    assert simulator.agent_positions[0] == (1, 0)
    assert not handover.can_resume(1)
    assert handover.can_resume(2)


def test_handover_without_pending_commands():
    simulator, executor = make_system()
    handover = ExecutionAwareHandover()
    handover.start()

    result = handover.step(
        simulator, executor, timestep=0
    )

    assert result["ready"]
    assert handover.can_resume(1)


def test_collision_prevents_m7_resumption():
    simulator = GroundTruthSimulator(
        grid=[[0, 0, 0]],
        agent_positions={0: (0, 0), 1: (2, 0)},
        agent_goals={0: (2, 0), 1: (0, 0)},
    )

    executor = DelayedCommandExecutor(
        ExecutionDelayModel(
            immediate_probability=0.0,
            seed=1,
        )
    )

    executor.queue_commands(
        actions={0: 3, 1: 4},
        current_timestep=0,
    )

    handover = ExecutionAwareHandover()
    handover.start()

    result = handover.step(
        simulator, executor, timestep=1
    )

    assert result["failed"]
    assert not result["ready"]
    assert not handover.can_resume(2)
