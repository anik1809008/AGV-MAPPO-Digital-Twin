from src.communication.telemetry import TelemetryMessage
from src.digital_twin.digital_twin import DigitalTwin
from src.digital_twin.state import AgentTwinState


def test_process_fresh_telemetry():
    state = AgentTwinState(
        agent_id=1,
        last_trusted_position=(8, 5),
        last_trusted_timestamp=20,
        goal=(1, 1),
    )

    state.command_history.extend([4, 4])

    dt = DigitalTwin({1: state})

    msg = TelemetryMessage(
        agent_id=1,
        position=(7, 5),
        source_timestamp=21,
    )

    updated = dt.process_telemetry(msg)

    assert updated is True
    assert state.last_trusted_position == (7, 5)
    assert state.last_trusted_timestamp == 21
    assert state.command_history == []
    assert dt.get_aoi(1, 23) == 2


def test_ignore_older_telemetry():
    state = AgentTwinState(
        agent_id=1,
        last_trusted_position=(7, 5),
        last_trusted_timestamp=21,
        goal=(1, 1),
    )

    dt = DigitalTwin({1: state})

    old_msg = TelemetryMessage(
        agent_id=1,
        position=(8, 5),
        source_timestamp=20,
    )

    updated = dt.process_telemetry(old_msg)

    assert updated is False
    assert state.last_trusted_position == (7, 5)
    assert state.last_trusted_timestamp == 21
from src.environment.actions import Action


def test_record_command():
    state = AgentTwinState(
        agent_id=1,
        last_trusted_position=(8, 5),
        last_trusted_timestamp=20,
        goal=(1, 1),
    )

    dt = DigitalTwin({1: state})

    dt.record_command(1, Action.WEST)
    dt.record_command(1, Action.NORTH)

    assert state.command_history == [4, 1]
def test_digital_twin_reachable_occupancy():
    grid = [[0, 0, 0, 0, 0]]

    state = AgentTwinState(
        agent_id=1,
        last_trusted_position=(3, 0),
        last_trusted_timestamp=20,
        goal=(0, 0),
    )

    dt = DigitalTwin({1: state})

    dt.record_command(1, Action.WEST)
    dt.record_command(1, Action.WEST)

    reachable = dt.get_reachable_occupancy(
        agent_id=1,
        grid=grid,
    )

    assert reachable == {
        (3, 0),
        (2, 0),
        (1, 0),
    }
def test_aoi_and_reachable_occupancy_grow_with_stale_telemetry():
    grid = [[0, 0, 0, 0, 0]]

    state = AgentTwinState(
        agent_id=1,
        last_trusted_position=(3, 0),
        last_trusted_timestamp=20,
        goal=(0, 0),
    )

    dt = DigitalTwin({1: state})

    assert dt.get_aoi(1, 20) == 0
    assert dt.get_reachable_occupancy(1, grid) == {
        (3, 0),
    }

    dt.record_command(1, Action.WEST)

    assert dt.get_aoi(1, 21) == 1
    assert dt.get_reachable_occupancy(1, grid) == {
        (3, 0),
        (2, 0),
    }

    dt.record_command(1, Action.WEST)

    assert dt.get_aoi(1, 22) == 2
    assert dt.get_reachable_occupancy(1, grid) == {
        (3, 0),
        (2, 0),
        (1, 0),
    }
def test_reconnection_resynchronizes_digital_twin():
    grid = [[0, 0, 0, 0, 0]]

    state = AgentTwinState(
        agent_id=1,
        last_trusted_position=(3, 0),
        last_trusted_timestamp=20,
        goal=(0, 0),
    )

    dt = DigitalTwin({1: state})

    dt.record_command(1, Action.WEST)
    dt.record_command(1, Action.WEST)

    assert dt.get_aoi(1, 22) == 2
    assert dt.get_reachable_occupancy(1, grid) == {
        (3, 0),
        (2, 0),
        (1, 0),
    }

    fresh_msg = TelemetryMessage(
        agent_id=1,
        position=(2, 0),
        source_timestamp=22,
    )

    dt.process_telemetry(fresh_msg)

    assert dt.get_aoi(1, 22) == 0
    assert dt.get_reachable_occupancy(1, grid) == {
        (2, 0),
    }
    assert state.command_history == []
