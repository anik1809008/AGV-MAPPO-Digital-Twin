from src.marl.multi_agent_batch import prepare_multi_agent_batch


def train_multi_agent_buffer(
    trainer,
    multi_agent_buffer,
    next_values=None,
    gamma=0.99,
    gae_lambda=0.95,
    epochs=4,
    minibatch_size=64,
):
    batch = prepare_multi_agent_batch(
        multi_agent_buffer=multi_agent_buffer,
        next_values=next_values,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    return trainer.update_epochs(
        **batch,
        epochs=epochs,
        minibatch_size=minibatch_size,
    )
