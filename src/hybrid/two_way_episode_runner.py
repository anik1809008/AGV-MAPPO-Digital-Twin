"""Two-way LNS-SAT <-> M7 execution; keeps legacy one-way runner unchanged.

Return decisions use only DT reachable positions, never simulator truth.
"""
import time

from src.communication.telemetry import TelemetryMessage
from src.hybrid.recovery_confirmation import recovery_state_confirmed
from src.hybrid.recovery_drain import drain_pending_commands
from src.hybrid.execution_aware_handover import ExecutionAwareHandover
from src.hybrid.recovery_pause import RecoveryPause
from src.hybrid.recovery_trigger import should_start_recovery
from src.evaluation.path_length import compute_step_path_length
from src.mapf.lns_sat import solve_lns_sat
from src.mapf.mdd_sat import position_paths_to_actions
from src.marl.digital_twin_adapter import get_digital_twin_inputs
from src.marl.real_training_step import run_real_training_step


def _plan(grid, starts, goals, max_iterations):
    began = time.perf_counter()
    solution = solve_lns_sat(
        grid=grid, starts=starts, goals=goals,
        max_iterations=max_iterations,
    )
    elapsed = time.perf_counter() - began
    if solution is None:
        return None, elapsed
    if set(solution["paths"]) != set(starts):
        raise ValueError("LNS-SAT plan must cover every agent")
    if any(not path or tuple(path[0]) != tuple(starts[agent_id])
           for agent_id, path in solution["paths"].items()):
        raise ValueError("LNS-SAT plan must start at observed positions")
    return {
        "actions": position_paths_to_actions(solution["paths"]),
        "horizon": solution["makespan"],
        "offset": 0,
    }, elapsed


def _has_pending(executor):
    return any(bool(queue) for queue in executor.pending_commands.values())


def run_two_way_episode(
    context,
    supervisor,
    max_iterations=100,
    return_max_iterations=10,
    replan_cooldown_steps=20,
    enable_recovery_pause=False,
    enable_execution_handover=False,
    recovery_max_pause_steps=8,
):
    """Run an episode; start/replan from DT-trusted positions only on return.

    Requires a fresh TwoWayHybridSupervisor. Keeps full-timestep telemetry
    processing and the existing trained M7 control step unchanged.
    """
    simulator = context["simulator"]
    twin = context["digital_twin"]
    channel = context["telemetry_channel"]
    executor = context["delayed_executor"]
    if supervisor.mode != "LNS-SAT" or supervisor.events:
        raise ValueError("A fresh two-way supervisor is required")
    if context.get("m7_controller") is None:
        raise ValueError("M7 controller is required")

    plan, planning_time = _plan(
        simulator.grid, dict(simulator.agent_positions),
        dict(simulator.agent_goals), max_iterations,
    )
    if return_max_iterations < 1:
        raise ValueError("return_max_iterations must be >= 1")
    if replan_cooldown_steps < 1:
        raise ValueError("replan_cooldown_steps must be >= 1")

    next_return_attempt = 0
    planning_failures = 0
    if plan is None:
        planning_failures += 1
        supervisor.fallback_to_m7(timestep=0)

    recovery = RecoveryPause(
        max_pause_steps=recovery_max_pause_steps,
        cooldown_steps=replan_cooldown_steps,
    )
    handover = (ExecutionAwareHandover()
                if enable_execution_handover else None)
    recovery_pause_steps = 0
    recovery_attempts = 0
    last_drain_timestep = None
    recovery_confirmed = False

    total_path_length = 0
    lns_steps = m7_steps = steps = replans = 0
    collision = deadlock = False

    for step in range(context["max_steps"]):
        if simulator.all_goals_reached():
            break

        # Match existing episode runner: send ground-truth sensor messages
        # through the channel, but never use those positions to choose mode.
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

        # Position predictions are derived from the planned path, not truth.
        planned_next = {}
        if supervisor.mode == "LNS-SAT" and plan is not None:
            index = step - plan["offset"]
            if index < plan["horizon"]:
                for agent_id, actions in plan["actions"].items():
                    if index < len(actions):
                        predicted = dt["possible_current_positions"].get(agent_id, set())
                        if len(predicted) == 1:
                            # Predicted next cell from the trusted current cell.
                            position = next(iter(predicted))
                            planned_next[agent_id] = simulator.get_next_position(
                                position, actions[index],
                            )

        if (
            enable_recovery_pause
            and not recovery.active
            and supervisor.reliable_steps < supervisor.recovery_steps
            and step >= next_return_attempt
        ):
            if should_start_recovery(
                mode=supervisor.mode,
                timestep=step,
                last_switch_timestep=supervisor.last_switch_timestep,
                aoi_values=dt["aoi_values"],
                min_dwell_steps=supervisor.min_dwell_steps,
            ) and recovery.start(step):
                recovery_attempts += 1
                last_drain_timestep = None
                recovery_confirmed = False

        if recovery.active:
            recovery_confirmed = recovery_state_confirmed(
                digital_twin=twin,
                aoi_values=dt["aoi_values"],
                possible_current_positions=dt["possible_current_positions"],
                pending_commands=_has_pending(executor),
                last_drain_timestep=last_drain_timestep,
            )

        decision = supervisor.observe(
            timestep=step, aoi_values=dt["aoi_values"],
            planned_next_positions=planned_next,
            possible_current_positions=dt["possible_current_positions"],
            pending_commands=(
                _has_pending(executor)
                or (recovery.active and not recovery_confirmed)
            ),
        )
        if (
            handover is not None
            and getattr(decision, 'previous_mode', None) == 'LNS-SAT'
            and getattr(decision, 'new_mode', None) == 'M7'
        ):
            handover.start()

        if (
            supervisor.mode == "M7"
            and decision == "RETURN_READY"
            and (not recovery.active or recovery_confirmed)
            and step >= next_return_attempt
        ):
            candidates = dt["possible_current_positions"]
            # Observe() already verifies singleton sets and AoI=0.
            observed_starts = {
                agent_id: next(iter(candidates[agent_id]))
                for agent_id in sorted(candidates)
            }
            new_plan, elapsed = _plan(
                simulator.grid, observed_starts,
                dict(simulator.agent_goals), return_max_iterations,
            )
            planning_time += elapsed
            replans += 1
            next_return_attempt = step + replan_cooldown_steps
            if new_plan is not None:
                new_plan["offset"] = step
                plan = new_plan
                supervisor.confirm_return(timestep=step)
                if recovery.active:
                    recovery.finish(step)
                    recovery_confirmed = False
                    last_drain_timestep = None
            else:
                # Stay in M7; avoid attempting the same replan every timestep.
                planning_failures += 1
                supervisor.reliable_steps = 0
                if recovery.active:
                    recovery.finish(step)
                    recovery_confirmed = False
                    last_drain_timestep = None

        before = dict(simulator.agent_positions)
        if supervisor.mode == "LNS-SAT":
            index = step - plan["offset"]
            if index >= plan["horizon"]:
                # The plan may have ended without completion under delay.
                supervisor.fallback_to_m7(timestep=step)
                plan = None
                planning_failures += 1

        if handover is not None and handover.active:
            transition = handover.step(simulator, executor, step)
            collision = transition['collision']
            commands = {}
        elif recovery.active and supervisor.mode == "M7":
            drain = drain_pending_commands(simulator, executor, step)
            collision = drain["collision"]
            recovery_pause_steps += 1

            if drain["executed_commands"]:
                last_drain_timestep = step

            if last_drain_timestep is None and not drain["pending_commands"]:
                last_drain_timestep = step

            if collision or recovery.expired(step):
                recovery.finish(step)
                recovery_confirmed = False
                supervisor.reliable_steps = 0

            commands = {}
        elif supervisor.mode == "LNS-SAT":
            commands = {
                agent_id: actions[index]
                for agent_id, actions in plan["actions"].items()
            }
            executor.queue_commands(actions=commands, current_timestep=step)
            ready = executor.get_ready_actions(current_timestep=step)
            executed = {agent_id: ready.get(agent_id, 0) for agent_id in commands}
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
            commands = dict(enumerate(result["actions"]))
            collision = bool(result["collision"])
            m7_steps += 1

        for agent_id, action in commands.items():
            twin.record_command(agent_id=agent_id, action=action, timestep=step)
        total_path_length += compute_step_path_length(
            before, dict(simulator.agent_positions),
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
        "planning_failed": plan is None and m7_steps == 0,
        "planning_failures": planning_failures,
        "path_length": total_path_length,
        "planning_time": planning_time,
        "switch_events": list(supervisor.events),
        "lns_steps": lns_steps,
        "m7_steps": m7_steps,
        "return_replans": replans,
        "recovery_pause_steps": recovery_pause_steps,
        "recovery_attempts": recovery_attempts,
    }
