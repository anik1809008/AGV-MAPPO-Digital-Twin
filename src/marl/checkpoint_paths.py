def build_training_checkpoint_path(
    method,
    agent_count,
    seed,
    scenario_id,
):
    return (
        "results/checkpoints/"
        f"{method.lower()}_"
        f"agents{agent_count}_"
        f"seed{seed}_"
        f"scenario{scenario_id}.pt"
    )
def build_validation_checkpoint_path(
    method,
    agent_count,
    seed,
    episodes_per_scenario,
):
    return (
        "results/checkpoints/validation/"
        f"{method.lower()}_"
        f"agents{agent_count}_"
        f"seed{seed}_"
        f"eps{episodes_per_scenario}.pt"
    )
