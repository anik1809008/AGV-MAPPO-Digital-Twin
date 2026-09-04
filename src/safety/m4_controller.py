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
    ):
        ranked_actions = rank_actions_by_probability(
            self.actor,
            observation_vector,
        )

        action, probability = self.action_filter.select_safe_action(
            shield=self.shield,
            agent_id=agent_id,
            ranked_actions=ranked_actions,
            possible_current_positions=possible_current_positions,
            other_current_positions=other_current_positions,
            other_next_positions=other_next_positions,
            reachable_occupancies=reachable_occupancies,
            other_possible_transitions=other_possible_transitions,
        )

        return action, probability
