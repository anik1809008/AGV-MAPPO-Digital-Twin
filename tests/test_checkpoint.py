import os

from src.marl.networks import ActorNetwork, CriticNetwork
from src.marl.mappo_trainer import MAPPOTrainer
from src.marl.checkpoint import save_checkpoint, load_checkpoint


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
