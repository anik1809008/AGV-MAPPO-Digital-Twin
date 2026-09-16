import os

from src.marl.networks import ActorNetwork, CriticNetwork
from src.marl.mappo_trainer import MAPPOTrainer


from src.marl.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_compatibility,
)

def test_checkpoint_save_and_load():
    actor = ActorNetwork(
        input_dim=247,
        action_dim=5,
    )

    critic = CriticNetwork(
        input_dim=1976,
    )

    trainer = MAPPOTrainer(
        actor=actor,
        critic=critic,
    )

    path = "/tmp/mappo_test_checkpoint.pt"

    save_checkpoint(
        path=path,
        actor=actor,
        critic=critic,
        actor_optimizer=trainer.actor_optimizer,
        critic_optimizer=trainer.critic_optimizer,
        extra_state={
            "episode": 10,
        },
    )

    extra_state = load_checkpoint(
        path=path,
        actor=actor,
        critic=critic,
        actor_optimizer=trainer.actor_optimizer,
        critic_optimizer=trainer.critic_optimizer,
    )

    assert os.path.exists(path)
    assert extra_state["episode"] == 10

    os.remove(path)
def test_checkpoint_compatibility_accepts_match():
    validate_checkpoint_compatibility(
        extra_state={
            "method": "M6",
            "agents": 8,
        },
        method="M6",
        agents=8,
    )


def test_checkpoint_compatibility_rejects_method_mismatch():
    try:
        validate_checkpoint_compatibility(
            extra_state={
                "method": "M6",
                "agents": 8,
            },
            method="M3",
            agents=8,
        )
    except ValueError as exc:
        assert (
            "Checkpoint method does not match"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Expected method mismatch ValueError"
        )


def test_checkpoint_compatibility_rejects_agent_mismatch():
    try:
        validate_checkpoint_compatibility(
            extra_state={
                "method": "M6",
                "agents": 8,
            },
            method="M6",
            agents=20,
        )
    except ValueError as exc:
        assert (
            "Checkpoint agent count does not match"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Expected agent-count mismatch ValueError"
        )
def test_checkpoint_compatibility_rejects_latency_mismatch():
    try:
        validate_checkpoint_compatibility(
            extra_state={
                "method": "M6",
                "agents": 8,
                "latency": 2,
                "immediate_probability": 0.8,
            },
            method="M6",
            agents=8,
            latency=3,
            immediate_probability=0.8,
        )
    except ValueError as exc:
        assert (
            "Checkpoint latency does not match"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Expected latency mismatch ValueError"
        )


def test_checkpoint_compatibility_rejects_execution_delay_mismatch():
    try:
        validate_checkpoint_compatibility(
            extra_state={
                "method": "M6",
                "agents": 8,
                "latency": 2,
                "immediate_probability": 0.8,
            },
            method="M6",
            agents=8,
            latency=2,
            immediate_probability=0.6,
        )
    except ValueError as exc:
        assert (
            "Checkpoint immediate_probability does not match"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Expected execution-delay mismatch ValueError"
        )
