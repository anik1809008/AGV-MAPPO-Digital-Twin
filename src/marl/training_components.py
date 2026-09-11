from src.marl.mappo_trainer import MAPPOTrainer
from src.marl.networks import ActorNetwork, CriticNetwork


OBSERVATION_DIM = 247
ACTION_DIM = 5


def build_mappo_components(agent_count):
    if agent_count < 1:
        raise ValueError("agent_count must be >= 1")

    actor = ActorNetwork(
        input_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
    )

    critic = CriticNetwork(
        input_dim=OBSERVATION_DIM * agent_count,
    )

    trainer = MAPPOTrainer(
        actor=actor,
        critic=critic,
    )

    return {
        "actor": actor,
        "critic": critic,
        "trainer": trainer,
    }
