from src.marl.environment_adapter import build_all_agent_inputs
from src.marl.environment_step import execute_training_step
from src.marl.rollout import collect_single_step



def run_real_training_step(
    actor,
    critic,
    simulator,
    method,
    trusted_positions,
    reachable_occupancies,
    aoi_values,
    multi_agent_buffer=None,
    delayed_executor=None,
    current_timestep=0,
    m4_controller=None,
):



    agent_positions = list(
        simulator.agent_positions.values()
    )

    agent_goals = list(
        simulator.agent_goals.values()
    )

    observations = build_all_agent_inputs(
        method=method,
        grid=simulator.grid,
        agent_positions=agent_positions,
        agent_goals=agent_goals,
        trusted_positions=trusted_positions,
        reachable_occupancies=reachable_occupancies,
        aoi_values=aoi_values,
    )

    rollout = collect_single_step(
        actor=actor,
        critic=critic,
        agent_observations=observations,
    )




    if method == "M4" and m4_controller is not None:
        actions = {}

        for agent_id, observation in enumerate(observations):
            action, _ = m4_controller.select_action(
                observation_vector=observation,
                agent_id=agent_id,
                possible_current_positions=reachable_occupancies.get(
                    agent_id,
                    {agent_positions[agent_id]},
                ),
                other_current_positions={
                    other_id: trusted_positions[other_id]
                    for other_id in trusted_positions
                    if other_id != agent_id
                },
                other_next_positions={},
                reachable_occupancies=reachable_occupancies,
            )

            actions[agent_id] = action

    else:
        actions = {
            agent_id: action
            for agent_id, action in enumerate(
                rollout["actions"]
            )
        }





    if delayed_executor is not None:
        delayed_executor.queue_commands(
            actions=actions,
            current_timestep=current_timestep,
        )

        ready_actions = delayed_executor.get_ready_actions(
            current_timestep=current_timestep,
        )

        execution_actions = {
            agent_id: ready_actions.get(
                agent_id,
                0,
            )
            for agent_id in actions
        }
    else:
        execution_actions = actions

    environment_result = execute_training_step(
        simulator=simulator,
        actions=execution_actions,
    )



    if multi_agent_buffer is not None:
        multi_agent_buffer.add_step(
            agent_observations=observations,
            actions=rollout["actions"],
            log_probs=rollout["log_probs"],
            rewards=environment_result["rewards"],
            value=rollout["value"],
            dones=environment_result["dones"],
            centralized_state=rollout[
                "centralized_state"
            ],
        )




    return {
        "observations": observations,
        "actions": rollout["actions"],
        "log_probs": rollout["log_probs"],
        "value": rollout["value"],
        "centralized_state": rollout[
            "centralized_state"
        ],
        "rewards": environment_result["rewards"],
        "dones": environment_result["dones"],
        "collision": environment_result["collision"],
        "current_positions": environment_result[
            "current_positions"
        ],
    }
