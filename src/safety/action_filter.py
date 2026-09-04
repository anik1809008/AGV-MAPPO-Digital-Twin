from src.environment.actions import Action


def select_safe_action(
    shield,
    agent_id,
    ranked_actions,
    possible_current_positions,
    other_current_positions,
    other_next_positions,
    reachable_occupancies,
):
    for action, probability in ranked_actions:
        if shield.is_action_safe(
            agent_id=agent_id,
            possible_current_positions=possible_current_positions,
            action=action,
            other_current_positions=other_current_positions,
            other_next_positions=other_next_positions,
            reachable_occupancies=reachable_occupancies,
        ):
            return action, probability

    return Action.WAIT, 0.0
