from src.digital_twin.state import AgentTwinState


def test_age_of_information():
    state = AgentTwinState(
        agent_id=1,
        last_trusted_position=(8, 5),
        last_trusted_timestamp=20,
        goal=(1, 1),
    )

    assert state.age_of_information(23) == 3


def test_initial_command_history_is_empty():
    state = AgentTwinState(
        agent_id=1,
        last_trusted_position=(8, 5),
        last_trusted_timestamp=20,
        goal=(1, 1),
    )

    assert state.command_history == []
