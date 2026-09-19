import os

from src.evaluation.checkpoint_resolver import (
    resolve_evaluation_checkpoint,
    resolve_validation_checkpoint,
)
from src.marl.checkpoint import load_checkpoint
from src.marl.training_components import build_mappo_components


def load_evaluation_models(
    method,
    agent_count,
    seed,
    scenario_id,
):
    components = build_mappo_components(
        agent_count=agent_count,
    )

    checkpoint_path = resolve_evaluation_checkpoint(
        method=method,
        agent_count=agent_count,
        seed=seed,
        scenario_id=scenario_id,
    )

    if checkpoint_path is None:
        return {
            "actor": components["actor"],
            "critic": components["critic"],
            "checkpoint_path": None,
            "extra_state": None,
        }

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    extra_state = load_checkpoint(
        path=checkpoint_path,
        actor=components["actor"],
        critic=components["critic"],
    )

    return {
        "actor": components["actor"],
        "critic": components["critic"],
        "checkpoint_path": checkpoint_path,
        "extra_state": extra_state,
    }


def load_validation_models(
    method,
    agent_count,
    seed,
    episodes_per_scenario,
):
    components = build_mappo_components(
        agent_count=agent_count,
    )

    checkpoint_path = resolve_validation_checkpoint(
        method=method,
        agent_count=agent_count,
        seed=seed,
        episodes_per_scenario=episodes_per_scenario,
    )

    if checkpoint_path is None:
        return {
            "actor": components["actor"],
            "critic": components["critic"],
            "checkpoint_path": None,
            "extra_state": None,
        }

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    extra_state = load_checkpoint(
        path=checkpoint_path,
        actor=components["actor"],
        critic=components["critic"],
    )

    return {
        "actor": components["actor"],
        "critic": components["critic"],
        "checkpoint_path": checkpoint_path,
        "extra_state": extra_state,
    }
