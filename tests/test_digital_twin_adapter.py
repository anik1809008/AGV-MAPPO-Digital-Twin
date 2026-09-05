from src.digital_twin.state import AgentTwinState
from src.digital_twin.digital_twin import DigitalTwin
from src.marl.digital_twin_adapter import get_digital_twin_inputs


def test_get_digital_twin_inputs():
    grid = [
        [0, 0, 0],
    ]

    states = {
        0: AgentTwinState(
            agent_id=0,
            last_trusted_position=(0, 0),
            last_trusted_timestamp=5,
            goal=(2, 0),
        ),
        1: AgentTwinState(
            agent_id=1,
            last_trusted_position=(2, 0),
            last_trusted_timestamp=4,
            goal=(0, 0),
        ),
    }

    digital_twin = DigitalTwin(states)

    result = get_digital_twin_inputs(
        digital_twin=digital_twin,
        grid=grid,
        current_timestep=6,
    )

    assert result["trusted_positions"] == {
        0: (0, 0),
        1: (2, 0),
    }

    assert result["aoi_values"] == {
        0: 1,
        1: 2,
    }

    assert result["reachable_occupancies"] == {
        0: {(0, 0)},
        1: {(2, 0)},
    }
