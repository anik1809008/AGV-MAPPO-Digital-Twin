from copy import deepcopy

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
        recovery_grid = deepcopy(self.grid)

        for position in occupied_positions:
            if position == current_position:
                continue

            x, y = position

            if (
                0 <= y < len(recovery_grid)
                and 0 <= x < len(recovery_grid[0])
            ):
                recovery_grid[y][x] = 1

        path = shortest_path(
            grid=recovery_grid,
            start=current_position,
            goal=goal_position,
        )

        if not path:
            return Action.WAIT

        return path[0]
