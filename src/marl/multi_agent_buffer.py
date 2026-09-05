from src.marl.buffer import RolloutBuffer


class MultiAgentRolloutBuffer:
    def __init__(self, num_agents):
        self.num_agents = num_agents

        self.agent_buffers = [
            RolloutBuffer()
            for _ in range(num_agents)
        ]

        self.centralized_states = []

    def add_step(
        self,
        agent_observations,
        actions,
        log_probs,
        rewards,
        value,
        dones,
        centralized_state,
    ):
        self.centralized_states.append(
            centralized_state
        )

        for agent_id in range(self.num_agents):
            self.agent_buffers[agent_id].add(
                observation=agent_observations[agent_id],
                action=actions[agent_id],
                log_prob=log_probs[agent_id],
                reward=rewards[agent_id],
                value=value,
                done=dones[agent_id],
            )

    def clear(self):
        for buffer in self.agent_buffers:
            buffer.clear()

        self.centralized_states.clear()

    def __len__(self):
        return len(self.centralized_states)
