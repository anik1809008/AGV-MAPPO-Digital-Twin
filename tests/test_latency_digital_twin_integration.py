from src.communication.latency_channel import FixedLatencyChannel
from src.communication.telemetry import TelemetryMessage
from src.digital_twin.digital_twin import DigitalTwin
from src.digital_twin.state import AgentTwinState
from src.environment.actions import Action


def test_delayed_telemetry_keeps_newer_unconfirmed_commands():
    grid = [[0, 0, 0, 0, 0]]

    state = AgentTwinState(
        agent_id=1,
        last_trusted_position=(3, 0),
        last_trusted_timestamp=20,
        goal=(0, 0),
    )

    dt = DigitalTwin({1: state})
    channel = FixedLatencyChannel(latency_steps=2)

    message = TelemetryMessage(
        agent_id=1,
        position=(2, 0),
        source_timestamp=21,
    )

    channel.send(message, current_timestep=21)

    dt.record_command(1, Action.WEST, 21)
    dt.record_command(1, Action.WEST, 22)

    received = channel.receive_ready(23)

    for message in received:
        dt.process_telemetry(message)

    assert state.last_trusted_position == (2, 0)
    assert state.last_trusted_timestamp == 21
    assert state.command_history == [(22, 4)]

    assert dt.get_aoi(1, 23) == 2

    assert dt.get_reachable_occupancy(1, grid) == {
        (2, 0),
        (1, 0),
    }
