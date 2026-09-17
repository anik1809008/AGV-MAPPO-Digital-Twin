import torch


def save_checkpoint(
    path,
    actor,
    critic,
    actor_optimizer=None,
    critic_optimizer=None,
    extra_state=None,
):
    checkpoint = {
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
    }

    if actor_optimizer is not None:
        checkpoint["actor_optimizer_state_dict"] = (
            actor_optimizer.state_dict()
        )

    if critic_optimizer is not None:
        checkpoint["critic_optimizer_state_dict"] = (
            critic_optimizer.state_dict()
        )

    if extra_state is not None:
        checkpoint["extra_state"] = extra_state

    torch.save(checkpoint, path)


def load_checkpoint(
    path,
    actor,
    critic,
    actor_optimizer=None,
    critic_optimizer=None,
    map_location="cpu",
):
    checkpoint = torch.load(
        path,
        map_location=map_location,
    )

    actor.load_state_dict(
        checkpoint["actor_state_dict"]
    )

    critic.load_state_dict(
        checkpoint["critic_state_dict"]
    )

    if (
        actor_optimizer is not None
        and "actor_optimizer_state_dict" in checkpoint
    ):
        actor_optimizer.load_state_dict(
            checkpoint["actor_optimizer_state_dict"]
        )

    if (
        critic_optimizer is not None
        and "critic_optimizer_state_dict" in checkpoint
    ):
        critic_optimizer.load_state_dict(
            checkpoint["critic_optimizer_state_dict"]
        )

    return checkpoint.get("extra_state")




def validate_checkpoint_compatibility(
    extra_state,
    method,
    agents,
    latency=None,
    immediate_probability=None,
):


    checkpoint_method = extra_state.get("method")
    checkpoint_agents = extra_state.get("agents")

    if checkpoint_method != method:
        raise ValueError(
            "Checkpoint method does not match "
            f"requested method: checkpoint="
            f"{checkpoint_method}, requested={method}"
        )

    if checkpoint_agents != agents:
        raise ValueError(
            "Checkpoint agent count does not match "
            f"requested agents: checkpoint="
            f"{checkpoint_agents}, requested={agents}"
        )
    checkpoint_latency = extra_state.get("latency")
    checkpoint_immediate_probability = extra_state.get(
        "immediate_probability"
    )

    if (
        latency is not None
        and checkpoint_latency != latency
    ):
        raise ValueError(
            "Checkpoint latency does not match "
            f"requested latency: checkpoint="
            f"{checkpoint_latency}, requested={latency}"
        )

    if (
        immediate_probability is not None
        and checkpoint_immediate_probability
        != immediate_probability
    ):
        raise ValueError(
            "Checkpoint immediate_probability does not match "
            f"requested value: checkpoint="
            f"{checkpoint_immediate_probability}, "
            f"requested={immediate_probability}"
        )
def get_resume_start_index(
    extra_state,
    agents,
):
    checkpoint_start_index = extra_state.get(
        "start_index",
        0,
    )

    checkpoint_episodes = extra_state.get(
        "episodes",
        0,
    )

    return (
        checkpoint_start_index
        + checkpoint_episodes * agents
    )
