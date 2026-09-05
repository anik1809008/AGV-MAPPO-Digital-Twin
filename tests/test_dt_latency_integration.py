from src.communication.latency_channel import FixedLatencyChannel
from src.communication.telemetry import TelemetryMessage
from src.digital_twin.digital_twin import DigitalTwin
from src.digital_twin.state import AgentTwinState
from src.environment.actions import Action


def test_latency_updates_aoi_and_reachable_occupancy():
    grid = [
        [0, 0, 0, 0],
    ]

    digital_twin = DigitalTwin({
        0: AgentTwinState(
            agent_id=0,
            last_trusted_position=(0, 0),
            last_trusted_timestamp=0,
            goal=(3, 0),
        ),
    })

    channel = FixedLatencyChannel(
        latency_steps=2
    )

    channel.send(
        TelemetryMessage(
            agent_id=0,
            position=(0, 0),
            source_timestamp=0,
        ),
        current_timestep=0,
    )

    digital_twin.record_command(
        agent_id=0,
        action=Action.EAST,
        timestep=0,
    )

    channel.send(
        TelemetryMessage(
            agent_id=0,
            position=(1, 0),
            source_timestamp=1,
        ),
        current_timestep=1,
    )

    digital_twin.record_command(
        agent_id=0,
        action=Action.EAST,
        timestep=1,
    )

    assert digital_twin.get_aoi(0, 1) == 1

    assert digital_twin.get_reachable_occupancy(
        0,
        grid,
    ) == {
        (0, 0),
        (1, 0),
        (2, 0),
    }

    messages = channel.receive_ready(
        current_timestep=2
    )

    for message in messages:
        digital_twin.process_telemetry(message)

    assert digital_twin.get_aoi(0, 2) == 2

    assert digital_twin.get_reachable_occupancy(
        0,
        grid,
    ) == {
        (0, 0),
        (1, 0),
    }
