from src.communication.telemetry import TelemetryMessage
from src.digital_twin.digital_twin import DigitalTwin
from src.digital_twin.state import AgentTwinState
from src.environment.delayed_execution import DelayedCommandExecutor
from src.environment.execution_delay import ExecutionDelayModel
from src.environment.simulator import GroundTruthSimulator
from src.hybrid.recovery_drain import drain_pending_commands
from src.marl.digital_twin_adapter import get_digital_twin_inputs


def test_post_drain_telemetry_confirms_position():
    grid = [[0] * 5]
    simulator = GroundTruthSimulator(
        grid=grid,
        agent_positions={0: (0, 0)},
        agent_goals={0: (4, 0)},
    )
    twin = DigitalTwin({
        0: AgentTwinState(
            agent_id=0,
            last_trusted_position=(0, 0),
            last_trusted_timestamp=0,
            goal=(4, 0),
        ),
    })
    executor = DelayedCommandExecutor(
        ExecutionDelayModel(immediate_probability=0.0, seed=1)
    )

    executor.queue_commands({0: 3}, current_timestep=0)
    twin.record_command(0, 3, timestep=0)

    # Command executes at t=1, after its telemetry snapshot.
    drain = drain_pending_commands(simulator, executor, timestep=1)
    assert not drain["pending_commands"]

    before_confirmation = twin.get_possible_current_positions(
        0, grid, allow_command_skip=True
    )
    assert len(before_confirmation) > 1

    # Fresh telemetry at t=2 confirms the post-execution position.
    twin.process_telemetry(
        TelemetryMessage(
            agent_id=0,
            position=simulator.agent_positions[0],
            source_timestamp=2,
        ),
        retain_previous_command=True,
    )

    dt = get_digital_twin_inputs(
        digital_twin=twin,
        grid=grid,
        current_timestep=2,
        allow_command_skip=True,
    )

    assert dt["aoi_values"][0] == 0
    assert dt["possible_current_positions"][0] == {(1, 0)}
    assert not any(executor.pending_commands.values())
