from src.digital_twin.state import AgentTwinState


class DigitalTwin:
    def __init__(self, agent_states):
        self.agent_states = dict(agent_states)

    def process_telemetry(self, message):
        state = self.agent_states[message.agent_id]

        if message.source_timestamp < state.last_trusted_timestamp:
            return False

        state.last_trusted_position = message.position
        state.last_trusted_timestamp = message.source_timestamp
        state.command_history.clear()

        return True

    def get_aoi(self, agent_id, current_timestep):
        return self.agent_states[agent_id].age_of_information(
            current_timestep
        )
    def record_command(self, agent_id, action):
        self.agent_states[agent_id].command_history.append(int(action))
