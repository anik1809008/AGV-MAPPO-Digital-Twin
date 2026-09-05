from src.marl.episode_runner import run_episode
from src.marl.episode_training import train_from_episode


def run_training_cycle(
    actor,
    critic,
    trainer,
    simulator,
    method,
    trusted_positions,
    reachable_occupancies,
    aoi_values,
    multi_agent_buffer,
    max_steps=200,
    next_values=None,
    gamma=0.99,
    gae_lambda=0.95,
    epochs=4,
    minibatch_size=64,
):
    episode_result = run_episode(
        actor=actor,
        critic=critic,
        simulator=simulator,
        method=method,
        trusted_positions=trusted_positions,
        reachable_occupancies=reachable_occupancies,
        aoi_values=aoi_values,
        multi_agent_buffer=multi_agent_buffer,
        max_steps=max_steps,
    )

    training_history = train_from_episode(
        trainer=trainer,
        multi_agent_buffer=multi_agent_buffer,
        next_values=next_values,
        gamma=gamma,
        gae_lambda=gae_lambda,
        epochs=epochs,
        minibatch_size=minibatch_size,
        clear_buffer=True,
    )

    return {
        "episode": episode_result,
        "training_history": training_history,
    }
