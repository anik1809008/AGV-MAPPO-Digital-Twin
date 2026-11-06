from src.baselines.stale_wait import ReachableSetWaitBaseline
from src.marl.environment_adapter import build_all_agent_inputs
from src.marl.dt_only_observation import build_dt_only_agent_inputs
from src.marl.environment_step import execute_training_step
from src.marl.rollout import collect_single_step
from src.marl.dt_decision_support import (
    get_dt_decision_positions,
    update_dt_progress,
    select_dt_recovery_action,
    reserve_dt_next_positions,
)
def run_real_training_step(
    actor,
    critic,
    simulator,
    method,
    trusted_positions,
    reachable_occupancies,
    aoi_values,
    possible_current_positions=None,
    possible_transitions=None,
    multi_agent_buffer=None,
    delayed_executor=None,
    current_timestep=0,
    m4_controller=None,
    m5_baseline=None,
    m7_controller=None,
    m7_progress_monitor=None,
    m7_recovery_planner=None,
    use_dt_observations=False,
):
    agent_positions = list(
        simulator.agent_positions.values()
    )

    agent_goals = list(
        simulator.agent_goals.values()
    )

    if use_dt_observations:
        if method != "M7":
            raise ValueError("DT-only observations require M7")

        observations = build_dt_only_agent_inputs(
            grid=simulator.grid,
            agent_goals=dict(simulator.agent_goals),
            trusted_positions=trusted_positions,
            reachable_occupancies=reachable_occupancies,
            aoi_values=aoi_values,
        )
    else:
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
        reserved_next_positions = {}

        if method == "M7" and m7_progress_monitor is not None:
            for agent_id, position in enumerate(
                agent_positions
            ):
                m7_progress_monitor.update(
                    agent_id=agent_id,
                    position=position,
                    goal=agent_goals[agent_id],
                )

        for agent_id, observation in enumerate(
            observations
        ):
            action, _ = m4_controller.select_action(
                observation_vector=observation,
                agent_id=agent_id,
                preferred_action=rollout["actions"][
                    agent_id
                ],
                possible_current_positions=(
                    possible_current_positions.get(
                        agent_id,
                        {agent_positions[agent_id]},
                    )
                    if possible_current_positions is not None
                    else reachable_occupancies.get(
                        agent_id,
                        {agent_positions[agent_id]},
                    )
                ),
                other_current_positions={
                    other_id: trusted_positions[other_id]
                    for other_id in trusted_positions
                    if other_id != agent_id
                },
                other_next_positions=(
                    reserved_next_positions
                ),
                reachable_occupancies=(
                    reachable_occupancies
                ),
                other_possible_transitions=(
                    {}
                    if possible_transitions is None
                    else possible_transitions
                ),
            )

            actions[agent_id] = action

            reserved_next_positions[agent_id] = (
                m4_controller.shield.next_position(
                    agent_positions[agent_id],
                    action,
                )
            )

    elif method == "M7" and m7_controller is not None:
        if use_dt_observations:
            if possible_current_positions is None:
                raise ValueError("DT mode requires possible current positions")

            agent_ids = range(len(agent_goals))
            dt_positions = get_dt_decision_positions(
                trusted_positions,
                possible_current_positions,
                agent_ids,
            )

            if set(reachable_occupancies) != set(agent_ids):
                raise ValueError("Incomplete reachable occupancies")

            actions = {}
            reserved_occupancies = {
                agent_id: set(occupancy)
                for agent_id, occupancy in reachable_occupancies.items()
            }

            if m7_progress_monitor is not None:
                for agent_id in agent_ids:
                    update_dt_progress(
                        monitor=m7_progress_monitor,
                        agent_id=agent_id,
                        position=dt_positions[agent_id],
                        goal=agent_goals[agent_id],
                        aoi=aoi_values[agent_id],
                    )

            for agent_id, observation in enumerate(observations):
                candidates = set(possible_current_positions[agent_id])
                use_recovery = (
                    m7_progress_monitor is not None
                    and m7_recovery_planner is not None
                    and m7_progress_monitor.is_stagnating(agent_id)
                )

                recovery_action = None
                if use_recovery:
                    recovery_action = select_dt_recovery_action(
                        planner=m7_recovery_planner,
                        agent_id=agent_id,
                        possible_positions=possible_current_positions,
                        reachable_occupancies=reserved_occupancies,
                        goal=agent_goals[agent_id],
                    )

                action, _ = m7_controller.select_action(
                    observation_vector=observation,
                    agent_id=agent_id,
                    preferred_action=(
                        recovery_action if recovery_action is not None
                        else rollout["actions"][agent_id]
                    ),
                    current_position=dt_positions[agent_id],
                    goal_position=agent_goals[agent_id],
                    prefer_progress=use_recovery,
                    possible_current_positions=candidates,
                    other_current_positions={
                        other_id: position
                        for other_id, position in dt_positions.items()
                        if other_id != agent_id
                    },
                    # Prior decisions are represented by sets in
                    # reserved_occupancies, not ground-truth singletons.
                    other_next_positions={},
                    reachable_occupancies=reserved_occupancies,
                    other_possible_transitions=(
                        {} if possible_transitions is None
                        else possible_transitions
                    ),
                )

                actions[agent_id] = action
                reserved_occupancies[agent_id].update(
                    reserve_dt_next_positions(
                        shield=m7_controller.shield,
                        possible_positions=candidates,
                        action=action,
                    )
                )
        else:
            actions = {}
            reserved_next_positions = {}

            if m7_progress_monitor is not None:
                for agent_id, position in enumerate(
                    agent_positions
                ):
                    m7_progress_monitor.update(
                        agent_id=agent_id,
                        position=position,
                        goal=agent_goals[agent_id],
                    )

            for agent_id, observation in enumerate(
                observations
            ):
                use_recovery = (
                    m7_progress_monitor is not None
                    and m7_recovery_planner is not None
                    and m7_progress_monitor.is_stagnating(
                        agent_id
                    )
                )
                recovery_action = None

                if use_recovery:
                    recovery_action = (
                        m7_recovery_planner.select_recovery_action(
                            agent_id=agent_id,
                            current_position=(
                                agent_positions[agent_id]
                            ),
                            goal_position=(
                                agent_goals[agent_id]
                            ),
                            occupied_positions={
                                 position
                                 for other_id, position
                                 in trusted_positions.items()
                                 if other_id != agent_id
                            },
                        )
                    )
                action, _ = m7_controller.select_action(
                    observation_vector=observation,
                    agent_id=agent_id,
                    preferred_action=(
                       recovery_action
                       if recovery_action is not None
                       else rollout["actions"][agent_id]
                    ),
                    current_position=agent_positions[agent_id],
                    goal_position=agent_goals[agent_id],
                    prefer_progress=(
                        m7_progress_monitor is not None
                        and m7_progress_monitor.is_stagnating(
                            agent_id
                        )
                    ),
                    possible_current_positions=(
                        possible_current_positions.get(
                            agent_id,
                            {agent_positions[agent_id]},
                        )
                        if possible_current_positions is not None
                        else reachable_occupancies.get(
                            agent_id,
                            {agent_positions[agent_id]},
                        )
                    ),
                    other_current_positions={
                        other_id: trusted_positions[other_id]
                        for other_id in trusted_positions
                        if other_id != agent_id
                    },
                    other_next_positions=(
                        reserved_next_positions
                    ),
                    reachable_occupancies=(
                        reachable_occupancies
                    ),
                    other_possible_transitions=(
                        {}
                        if possible_transitions is None
                        else possible_transitions
                    ),
                )

                actions[agent_id] = action

                reserved_next_positions[agent_id] = (
                    m7_controller.shield.next_position(
                        agent_positions[agent_id],
                        action,
                    )
                )
    elif method == "M5" and m5_baseline is not None:
        actions = {}

        for agent_id, proposed_action in enumerate(
            rollout["actions"]
        ):
            reachable_size = len(
                reachable_occupancies.get(
                    agent_id,
                    {agent_positions[agent_id]},
                )
            )

            actions[agent_id] = m5_baseline.select_action(
                proposed_action,
                reachable_size,
            )
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

        ready_actions = (
            delayed_executor.get_ready_actions(
                current_timestep=current_timestep,
            )
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
            actions=[
                actions[agent_id]
                for agent_id in range(
                    len(agent_positions)
                )
            ],
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
        "actions": [
            actions[agent_id]
            for agent_id in range(
                len(agent_positions)
            )
        ],
        "executed_actions": [
            execution_actions[agent_id]
            for agent_id in range(
                len(agent_positions)
            )
        ],
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
