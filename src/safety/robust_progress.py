"""Optional robust progress selection without simulator ground truth."""

from src.environment.actions import Action
from src.environment.pathfinding import shortest_path_distance


def select_robust_progress_action(
    *,
    shield,
    agent_id,
    ranked_actions,
    possible_current_positions,
    goal_position,
    other_current_positions,
    other_next_positions,
    reachable_occupancies,
    other_possible_transitions=None,
):
    positions = set(possible_current_positions)

    if not positions:
        raise ValueError("Missing possible current positions")

    if goal_position is None:
        raise ValueError("Missing goal position")

    # Preserve policy ranking among robust, safe improving actions.
    for action, probability in ranked_actions:
        if Action(action) == Action.WAIT:
            continue

        improves_all = True

        for position in positions:
            before = shortest_path_distance(
                grid=shield.grid,
                start=position,
                goal=goal_position,
            )

            next_position = shield.next_position(
                position, action
            )

            after = shortest_path_distance(
                grid=shield.grid,
                start=next_position,
                goal=goal_position,
            )

            if (
                before is None
                or after is None
                or after >= before
            ):
                improves_all = False
                break

        if not improves_all:
            continue

        if shield.is_action_safe(
            agent_id=agent_id,
            possible_current_positions=positions,
            action=action,
            other_current_positions=other_current_positions,
            other_next_positions=other_next_positions,
            reachable_occupancies=reachable_occupancies,
            other_possible_transitions=other_possible_transitions,
        ):
            return action, probability

    return None
