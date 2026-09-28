from src.environment.actions import Action
from src.environment.actions import Action
from src.environment.pathfinding import shortest_path_distance
class M7ActionFilter:
    def __init__(self):
        self.intervention_count = 0

    def select_safe_action(
        self,
        shield,
        agent_id,
        ranked_actions,
        possible_current_positions,
        other_current_positions,
        other_next_positions,
        reachable_occupancies,
        other_possible_transitions=None,
        current_position=None,
        goal_position=None,
        prefer_progress=False,
    ):
        original_action = ranked_actions[0][0]

        def is_safe(action):
            return shield.is_action_safe(
                agent_id=agent_id,
                possible_current_positions=(
                    possible_current_positions
                ),
                action=action,
                other_current_positions=(
                    other_current_positions
                ),
                other_next_positions=(
                    other_next_positions
                ),
                reachable_occupancies=(
                    reachable_occupancies
                ),
                other_possible_transitions=(
                    other_possible_transitions
                ),
            )

        original_probability = ranked_actions[0][1]
        if (
            prefer_progress
            and current_position is not None
            and goal_position is not None
        ):
            progress_candidates = []

            for action, probability in ranked_actions:
                if not is_safe(action):
                    continue

                next_position = shield.next_position(
                    current_position,
                    action,
                )

                distance = shortest_path_distance(
                    grid=shield.grid,
                    start=next_position,
                    goal=goal_position,
                )

                if distance is None:
                    continue

                progress_candidates.append(
                    (
                        distance,
                        -probability,
                        action,
                        probability,
                    )
                )

            if progress_candidates:
                progress_candidates.sort()

                _, _, action, probability = (
                    progress_candidates[0]
                )

                if action != original_action:
                    self.intervention_count += 1

                return action, probability
        if is_safe(original_action):
            return (
                original_action,
                original_probability,
            )

        for action, probability in ranked_actions:
            if Action(action) == Action.WAIT:
                continue

            if is_safe(action):
                self.intervention_count += 1
                return action, probability

        for action, probability in ranked_actions:
            if Action(action) != Action.WAIT:
                continue

            if is_safe(action):
                self.intervention_count += 1
                return action, probability

        self.intervention_count += 1

        return Action.WAIT, 0.0
