from src.environment.pathfinding import shortest_path_distance


class ProgressMonitor:
    def __init__(self, grid, stagnation_steps=10):
        self.grid = grid
        self.stagnation_steps = stagnation_steps
        self.best_distances = {}
        self.no_progress_counts = {}

    def update(self, agent_id, position, goal):
        distance = shortest_path_distance(
            grid=self.grid,
            start=position,
            goal=goal,
        )

        if distance is None:
            self.no_progress_counts[agent_id] = (
                self.no_progress_counts.get(agent_id, 0) + 1
            )
        else:
            best_distance = self.best_distances.get(agent_id)

            if (
                best_distance is None
                or distance < best_distance
            ):
                self.best_distances[agent_id] = distance
                self.no_progress_counts[agent_id] = 0
            else:
                self.no_progress_counts[agent_id] = (
                    self.no_progress_counts.get(agent_id, 0) + 1
                )

        return self.is_stagnating(agent_id)

    def is_stagnating(self, agent_id):
        return (
            self.no_progress_counts.get(agent_id, 0)
            >= self.stagnation_steps
        )
