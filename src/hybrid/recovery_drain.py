"""Helpers for safely draining outstanding simulated AGV commands."""

from src.evaluation.path_length import compute_step_path_length


def drain_pending_commands(simulator, executor, timestep):
    """Execute ready commands without issuing any new commands."""
    before = dict(simulator.agent_positions)

    ready = executor.get_ready_actions(
        current_timestep=timestep,
    )

    actions = {
        agent_id: ready.get(agent_id, 0)
        for agent_id in before
    }

    outcome = simulator.step_joint(actions)

    pending = any(
        bool(queue)
        for queue in executor.pending_commands.values()
    )

    return {
        "executed_commands": bool(ready),
        "collision": bool(outcome["collision"]),
        "path_length": compute_step_path_length(
            before,
            simulator.agent_positions,
        ),
        "pending_commands": pending,
    }
