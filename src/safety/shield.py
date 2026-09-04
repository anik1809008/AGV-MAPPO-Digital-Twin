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
    def has_edge_swap_conflict(
        self,
        agent_id,
        current_position,
        candidate_position,
        other_current_positions,
        other_next_positions,
    ):
        for other_agent_id, other_current in other_current_positions.items():
            if other_agent_id == agent_id:
                continue

            other_next = other_next_positions.get(other_agent_id)

            if other_next is None:
                continue

            if (
                candidate_position == other_current
                and other_next == current_position
            ):
                return True

        return False
    def intersects_reachable_occupancy(
        self,
        agent_id,
        candidate_position,
        reachable_occupancies,
    ):
        for other_agent_id, occupancy in reachable_occupancies.items():
            if other_agent_id == agent_id:
                continue

            if candidate_position in occupancy:
                return True

        return False
    def possible_next_positions(
        self,
        possible_current_positions,
        action,
    ):
        return {
            self.next_position(position, action)
            for position in possible_current_positions
        }
    def is_action_safe(
        self,
        agent_id,
        possible_current_positions,
        action,
        other_current_positions,
        other_next_positions,
        reachable_occupancies,
    ):
        possible_next = self.possible_next_positions(
            possible_current_positions,
            action,
        )

        for current_position in possible_current_positions:
            candidate_position = self.next_position(
                current_position,
                action,
            )

            if self.has_vertex_conflict(
                agent_id=agent_id,
                candidate_position=candidate_position,
                other_next_positions=other_next_positions,
            ):
                return False

            if self.has_edge_swap_conflict(
                agent_id=agent_id,
                current_position=current_position,
                candidate_position=candidate_position,
                other_current_positions=other_current_positions,
                other_next_positions=other_next_positions,
            ):
                return False

        for candidate_position in possible_next:
            if self.intersects_reachable_occupancy(
                agent_id=agent_id,
                candidate_position=candidate_position,
                reachable_occupancies=reachable_occupancies,
            ):
                return False

        return True
