from src.digital_twin.digital_twin import DigitalTwin
from src.digital_twin.state import AgentTwinState
from src.hybrid.recovery_confirmation import recovery_state_confirmed


def make_twin(timestamp=6):
    return DigitalTwin({
        0: AgentTwinState(
            agent_id=0,
            last_trusted_position=(1, 0),
            last_trusted_timestamp=timestamp,
            goal=(4, 0),
        ),
    })


def check(timestamp=6, positions=None, pending=False, drain=5):
    return recovery_state_confirmed(
        digital_twin=make_twin(timestamp),
        aoi_values={0: 0},
        possible_current_positions=(
            {0: {(1, 0)}} if positions is None else positions
        ),
        pending_commands=pending,
        last_drain_timestep=drain,
    )


def test_fresh_post_drain_telemetry_passes():
    assert check()


def test_telemetry_from_drain_timestep_is_insufficient():
    assert not check(timestamp=5)


def test_pending_commands_block_confirmation():
    assert not check(pending=True)


def test_multiple_possible_positions_block_confirmation():
    assert not check(positions={0: {(1, 0), (2, 0)}})


def test_missing_drain_timestamp_blocks_confirmation():
    assert not check(drain=None)


def test_all_agents_must_have_post_drain_confirmation():
    twin = DigitalTwin({
        0: AgentTwinState(
            agent_id=0,
            last_trusted_position=(1, 0),
            last_trusted_timestamp=6,
            goal=(4, 0),
        ),
        1: AgentTwinState(
            agent_id=1,
            last_trusted_position=(3, 0),
            last_trusted_timestamp=5,
            goal=(0, 0),
        ),
    })

    kwargs = dict(
        digital_twin=twin,
        aoi_values={0: 0, 1: 0},
        possible_current_positions={
            0: {(1, 0)},
            1: {(3, 0)},
        },
        pending_commands=False,
        last_drain_timestep=5,
    )

    assert recovery_state_confirmed(**kwargs) is False

    twin.agent_states[1].last_trusted_timestamp = 6

    assert recovery_state_confirmed(**kwargs) is True
