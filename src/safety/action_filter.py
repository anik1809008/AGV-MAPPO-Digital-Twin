from src.environment.actions import Action


class ShieldActionFilter:
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
    ):
        original_action = ranked_actions[0][0]

        for action, probability in ranked_actions:
            if shield.is_action_safe(
                agent_id=agent_id,
                possible_current_positions=possible_current_positions,
                action=action,
                other_current_positions=other_current_positions,
                other_next_positions=other_next_positions,
                reachable_occupancies=reachable_occupancies,
            ):
                if action != original_action:
                    self.intervention_count += 1

                return action, probability

        if original_action != Action.WAIT:
            self.intervention_count += 1

        return Action.WAIT, 0.0
