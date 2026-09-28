from src.baselines.classical_planner import shortest_path
from src.environment.actions import Action


class LocalRecoveryPlanner:
    def __init__(self, grid):
        self.grid = grid

    def select_recovery_action(
        self,
        agent_id,
        current_position,
        goal_position,
        occupied_positions,
    ):
        path = shortest_path(
            grid=self.grid,
            start=current_position,
            goal=goal_position,
        )

        if not path:
            return Action.WAIT

        for action in path:
            candidate = self._next_position(
                current_position,
                action,
            )

            if candidate not in occupied_positions:
                return action

            break

        return Action.WAIT

    def _next_position(self, position, action):
        x, y = position

        if action == Action.NORTH:
            return (x, y - 1)

        if action == Action.SOUTH:
            return (x, y + 1)

        if action == Action.EAST:
            return (x + 1, y)

        if action == Action.WEST:
            return (x - 1, y)

        return position
