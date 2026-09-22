from src.marl.policy import rank_actions_by_probability
from src.safety.action_filter import ShieldActionFilter


class M4Controller:
    def __init__(self, actor, shield):
        self.actor = actor
        self.shield = shield
        self.action_filter = ShieldActionFilter()

    def select_action(
        self,
        observation_vector,
        agent_id,
        possible_current_positions,
        other_current_positions,
        other_next_positions,
        reachable_occupancies,
        other_possible_transitions=None,
        preferred_action=None,
    ):
        ranked_actions = rank_actions_by_probability(
            self.actor,
            observation_vector,
        )

        if preferred_action is not None:
            preferred_entry = next(
                (
                    (action, probability)
                    for action, probability in ranked_actions
                    if action == preferred_action
                ),
                None,
            )

            if preferred_entry is not None:
                ranked_actions = [
                    preferred_entry,
                    *[
                        (action, probability)
                        for action, probability in ranked_actions
                        if action != preferred_action
                    ],
                ]

        action, probability = (
            self.action_filter.select_safe_action(
                shield=self.shield,
                agent_id=agent_id,
                ranked_actions=ranked_actions,
                possible_current_positions=(
                    possible_current_positions
                ),
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
        )

        return action, probability
