from src.safety.robust_progress import select_robust_progress_action
from src.marl.policy import rank_actions_by_probability
from src.safety.m7_action_filter import M7ActionFilter
class M7Controller:
    def __init__(self, actor, shield, enable_progress_forwarding=False, enable_robust_progress=False):
        self.enable_progress_forwarding = enable_progress_forwarding
        self.enable_robust_progress = enable_robust_progress
        self.actor = actor
        self.shield = shield
        self.action_filter = M7ActionFilter()

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
        current_position=None,
        goal_position=None,
        prefer_progress=False,
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

        if self.enable_robust_progress:
            robust_result = select_robust_progress_action(
                shield=self.shield,
                agent_id=agent_id,
                ranked_actions=ranked_actions,
                possible_current_positions=possible_current_positions,
                goal_position=goal_position,
                other_current_positions=other_current_positions,
                other_next_positions=other_next_positions,
                reachable_occupancies=reachable_occupancies,
                other_possible_transitions=other_possible_transitions,
            )
            if robust_result is not None:
                return robust_result

        progress_kwargs = {}
        if self.enable_progress_forwarding:
            progress_kwargs = {
                "current_position": current_position,
                "goal_position": goal_position,
                "prefer_progress": prefer_progress,
            }

        action, probability = (
            self.action_filter.select_safe_action(
                **progress_kwargs,
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
