from src.marl.multi_agent_trainer import train_multi_agent_buffer


def train_from_episode(
    trainer,
    multi_agent_buffer,
    next_values=None,
    gamma=0.99,
    gae_lambda=0.95,
    epochs=4,
    minibatch_size=64,
    clear_buffer=True,
):
    history = train_multi_agent_buffer(
        trainer=trainer,
        multi_agent_buffer=multi_agent_buffer,
        next_values=next_values,
        gamma=gamma,
        gae_lambda=gae_lambda,
        epochs=epochs,
        minibatch_size=minibatch_size,
    )

    if clear_buffer:
        multi_agent_buffer.clear()

    return history
