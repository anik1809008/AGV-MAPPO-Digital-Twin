from src.digital_twin.reachability import compute_reachable_occupancy
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

        state.command_history = [
            (timestamp, action)
            for timestamp, action in state.command_history
            if timestamp > message.source_timestamp
        ]



        return True

    def get_aoi(self, agent_id, current_timestep):
        return self.agent_states[agent_id].age_of_information(
            current_timestep
        )
    def record_command(self, agent_id, action, timestep):
        self.agent_states[agent_id].command_history.append(
            (timestep, int(action))
        )


    def get_reachable_occupancy(self, agent_id, grid):
        state = self.agent_states[agent_id]

        return compute_reachable_occupancy(
            grid=grid,
            last_trusted_position=state.last_trusted_position,
            command_history=state.command_history,
        )
