from src.environment.actions import Action, ACTION_DELTAS


class GroundTruthSimulator:
    def __init__(self, grid, agent_positions, agent_goals=None):
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0])

        self.agent_positions = dict(agent_positions)
        self.agent_goals = dict(agent_goals or {})
    def is_inside(self, position):
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, position):
        x, y = position

        if not self.is_inside(position):
            return False

        return self.grid[y][x] == 0

    def get_next_position(self, position, action):
        dx, dy = ACTION_DELTAS[Action(action)]

        next_position = (
            position[0] + dx,
            position[1] + dy,
        )

        if self.is_free(next_position):
            return next_position

        return position

    def move_agent(self, agent_id, action):
        current_position = self.agent_positions[agent_id]

        next_position = self.get_next_position(
            current_position,
            action,
        )

        self.agent_positions[agent_id] = next_position

        return next_position


    def is_at_goal(self, agent_id):
        if agent_id not in self.agent_goals:
            return False

        return self.agent_positions[agent_id] == self.agent_goals[agent_id]

    def all_goals_reached(self):
        if not self.agent_goals:
            return False

        return all(
            self.is_at_goal(agent_id)
            for agent_id in self.agent_goals
        )



    def step_joint(self, actions):
        current_positions = dict(self.agent_positions)

        proposed_positions = {}

        for agent_id, action in actions.items():
            proposed_positions[agent_id] = self.get_next_position(
                current_positions[agent_id],
                action,
            )

        targets = {}

        for agent_id, position in proposed_positions.items():
            targets.setdefault(position, []).append(agent_id)

        vertex_conflicts = {
            position: agent_ids
            for position, agent_ids in targets.items()
            if len(agent_ids) > 1
        }

        edge_swap_conflicts = []

        agent_ids = list(proposed_positions.keys())

        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                a = agent_ids[i]
                b = agent_ids[j]

                if (
                    proposed_positions[a] == current_positions[b]
                    and proposed_positions[b] == current_positions[a]
                ):
                    edge_swap_conflicts.append((a, b))

        if vertex_conflicts or edge_swap_conflicts:
            return {
                "success": False,
                "vertex_conflicts": vertex_conflicts,
                "edge_swap_conflicts": edge_swap_conflicts,
                "positions": current_positions,
            }

        self.agent_positions = proposed_positions

        return {
            "success": True,
            "vertex_conflicts": {},
            "edge_swap_conflicts": [],
            "positions": dict(self.agent_positions),
        }
