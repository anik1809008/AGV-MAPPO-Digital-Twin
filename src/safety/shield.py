from src.environment.actions import Action, ACTION_DELTAS


class SafetyShield:
    def __init__(self, grid):
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0])

    def is_inside(self, position):
        x, y = position

        return (
            0 <= x < self.width
            and 0 <= y < self.height
        )

    def is_free(self, position):
        x, y = position

        if not self.is_inside(position):
            return False

        return self.grid[y][x] == 0

    def next_position(self, position, action):
        dx, dy = ACTION_DELTAS[Action(action)]

        candidate = (
            position[0] + dx,
            position[1] + dy,
        )

        if not self.is_free(candidate):
            return position

        return candidate
    def has_vertex_conflict(
        self,
        agent_id,
        candidate_position,
        other_next_positions,
    ):
        for other_agent_id, other_position in other_next_positions.items():
            if other_agent_id == agent_id:
                continue

            if candidate_position == other_position:
                return True

        return False
