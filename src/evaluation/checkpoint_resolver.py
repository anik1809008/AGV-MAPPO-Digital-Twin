from src.evaluation.policy_mapping import get_policy_method
from src.evaluation.policy_mapping import get_policy_method
from src.marl.checkpoint_paths import (
    build_training_checkpoint_path,
    build_validation_checkpoint_path,
)
def resolve_evaluation_checkpoint(
    method,
    agent_count,
    seed,
    scenario_id,
):
    policy_method = get_policy_method(method)

    if policy_method is None:
        return None

    return build_training_checkpoint_path(
        method=policy_method,
        agent_count=agent_count,
        seed=seed,
        scenario_id=scenario_id,
    )
def resolve_validation_checkpoint(
    method,
    agent_count,
    seed,
    episodes_per_scenario,
):
    policy_method = get_policy_method(method)

    if policy_method is None:
        return None

    return build_validation_checkpoint_path(
        method=policy_method,
        agent_count=agent_count,
        seed=seed,
        episodes_per_scenario=episodes_per_scenario,
    )
