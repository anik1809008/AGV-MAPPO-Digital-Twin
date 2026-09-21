from collections import deque


class DelayedCommandExecutor:
    def __init__(self, delay_model):
        self.delay_model = delay_model
        self.pending_commands = {}

    def queue_commands(
        self,
        actions,
        current_timestep,
    ):
        for agent_id, action in actions.items():
            delay = self.delay_model.sample_delay()

            execution_timestep = (
                current_timestep + delay
            )

            if agent_id not in self.pending_commands:
                self.pending_commands[agent_id] = deque()

            self.pending_commands[agent_id].append(
                (
                    execution_timestep,
                    action,
                )
            )

    def get_ready_actions(
        self,
        current_timestep,
    ):
        ready_actions = {}

        for agent_id, queue in self.pending_commands.items():
            if not queue:
                continue

            latest_ready_action = None

            while queue:
                execution_timestep, action = queue[0]

                if execution_timestep > current_timestep:
                    break

                queue.popleft()
                latest_ready_action = action

            if latest_ready_action is not None:
                ready_actions[agent_id] = latest_ready_action
        return ready_actions
