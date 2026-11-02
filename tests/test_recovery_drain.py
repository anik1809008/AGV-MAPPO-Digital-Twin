from src.environment.delayed_execution import DelayedCommandExecutor
from src.environment.execution_delay import ExecutionDelayModel
from src.environment.simulator import GroundTruthSimulator
from src.hybrid.recovery_drain import drain_pending_commands


def test_drain_pending_command_without_new_commands():
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

    executor.queue_commands(
        actions={0: 3},
        current_timestep=0,
    )

    result = drain_pending_commands(
        simulator,
        executor,
        timestep=1,
    )

    assert simulator.agent_positions[0] == (1, 0)
    assert result["path_length"] == 1
    assert result["collision"] is False
    assert result["pending_commands"] is False


def test_drain_reports_multi_agent_collision():
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

    result = drain_pending_commands(
        simulator,
        executor,
        timestep=1,
    )

    assert result["collision"] is True
    assert result["path_length"] == 0
    assert result["pending_commands"] is False
    assert simulator.agent_positions == {
        0: (0, 0),
        1: (2, 0),
    }
