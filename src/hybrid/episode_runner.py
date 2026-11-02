"""One-way LNS-SAT -> M7 hybrid episode, reusing existing DT/M7 components.

A first integration milestone; validate experimentally before making performance claims.
"""
import time

from src.communication.telemetry import TelemetryMessage
from src.evaluation.path_length import compute_step_path_length
from src.mapf.lns_sat import solve_lns_sat
from src.mapf.mdd_sat import position_paths_to_actions
from src.marl.digital_twin_adapter import get_digital_twin_inputs
from src.marl.real_training_step import run_real_training_step


def run_hybrid_episode(context, supervisor, max_iterations=100):
    simulator = context["simulator"]
    twin = context["digital_twin"]
    channel = context["telemetry_channel"]
    executor = context["delayed_executor"]
    if supervisor.mode != "LNS-SAT" or supervisor.events:
        raise ValueError("Use a fresh HybridSupervisor for each episode")
    if context.get("m7_controller") is None:
        raise ValueError("M7 controller is required")

    start = time.perf_counter()
    solution = solve_lns_sat(
        grid=simulator.grid,
        starts=dict(simulator.agent_positions),
        goals=dict(simulator.agent_goals),
        max_iterations=max_iterations,
    )
    planning_time = time.perf_counter() - start
    if solution is None:
        return {
            "steps": 0, "collision": False, "deadlock": False,
            "all_goals_reached": False, "planning_failed": True,
            "path_length": 0, "planning_time": planning_time,
            "switch_events": [], "lns_steps": 0, "m7_steps": 0,
        }

    paths = solution["paths"]
    action_paths = position_paths_to_actions(paths)
    horizon = solution["makespan"]
    total_path_length = 0
    lns_steps = m7_steps = steps = 0
    collision = deadlock = planning_failed = False

    for step in range(context["max_steps"]):
        if simulator.all_goals_reached():
            break
        # Match the existing run_episode telemetry/update ordering.
        for agent_id, position in simulator.agent_positions.items():
            channel.send(
                TelemetryMessage(agent_id=agent_id, position=position,
                                 source_timestamp=step),
                current_timestep=step,
            )
        uncertain_execution = (
            getattr(executor.delay_model, "immediate_probability", 1.0) < 1.0
        )
        for message in channel.receive_ready(current_timestep=step):
            twin.process_telemetry(
                message, retain_previous_command=uncertain_execution,
            )
        dt = get_digital_twin_inputs(
            digital_twin=twin, grid=simulator.grid,
            current_timestep=step, allow_command_skip=uncertain_execution,
        )

        planned_next = {
            agent_id: path[step + 1]
            for agent_id, path in paths.items()
            if step + 1 < len(path)
        } if step < horizon else {}
        supervisor.observe(
            timestep=step,
            aoi_values=dt["aoi_values"],
            planned_next_positions=planned_next,
            possible_current_positions=dt["possible_current_positions"],
        )
        before = dict(simulator.agent_positions)
        if supervisor.mode == "LNS-SAT":
            if step >= horizon:
                planning_failed = True
                break
            commanded = {
                agent_id: actions[step]
                for agent_id, actions in action_paths.items()
            }
            executor.queue_commands(actions=commanded, current_timestep=step)
            ready = executor.get_ready_actions(current_timestep=step)
            executed = {agent_id: ready.get(agent_id, 0)
                        for agent_id in commanded}
            result = simulator.step_joint(executed)
            collision = bool(result.get("collision", False))
            lns_steps += 1
        else:
            result = run_real_training_step(
                actor=context["actor"], critic=context["critic"],
                simulator=simulator, method="M7",
                trusted_positions=dt["trusted_positions"],
                reachable_occupancies=dt["reachable_occupancies"],
                possible_current_positions=dt["possible_current_positions"],
                aoi_values=dt["aoi_values"],
                possible_transitions=dt["possible_transitions"],
                multi_agent_buffer=context["buffer"],
                delayed_executor=executor, current_timestep=step,
                m7_controller=context["m7_controller"],
                m7_progress_monitor=context.get("m7_progress_monitor"),
                m7_recovery_planner=context.get("m7_recovery_planner"),
            )
            commanded = dict(enumerate(result["actions"]))
            collision = bool(result["collision"])
            m7_steps += 1

        for agent_id, action in commanded.items():
            twin.record_command(agent_id=agent_id, action=action, timestep=step)
        total_path_length += compute_step_path_length(
            before, dict(simulator.agent_positions)
        )
        steps = step + 1
        deadlock = simulator.is_deadlocked()
        if collision or deadlock or simulator.all_goals_reached():
            break

    return {
        "steps": steps,
        "collision": collision,
        "deadlock": deadlock,
        "all_goals_reached": simulator.all_goals_reached(),
        "planning_failed": planning_failed,
        "path_length": total_path_length,
        "planning_time": planning_time,
        "switch_events": list(supervisor.events),
        "lns_steps": lns_steps,
        "m7_steps": m7_steps,
    }
