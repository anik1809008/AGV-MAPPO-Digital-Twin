"""Two-way LNS-SAT <-> M7 execution; keeps legacy one-way runner unchanged.

Return decisions use only DT reachable positions, never simulator truth.
"""
from contextlib import contextmanager, nullcontext
import cProfile
import io
import pstats
import json
import os
import time

from src.communication.telemetry import TelemetryMessage
from src.hybrid.recovery_confirmation import recovery_state_confirmed
from src.hybrid.recovery_drain import drain_pending_commands
from src.hybrid.execution_aware_handover import ExecutionAwareHandover
from src.hybrid.recovery_pause import RecoveryPause
from src.hybrid.recovery_trigger import should_start_recovery
from src.evaluation.path_length import compute_step_path_length
from src.mapf.lns_sat import solve_lns_sat
from src.hybrid.single_agv_completion import plan_single_agv_completion
from src.hybrid.two_agv_completion import plan_two_agv_completion
from src.hybrid.recovery_route_guard import (
    build_expected_positions,
    recovery_route_is_valid,
)
from src.hybrid.single_agv_progress import acknowledge_recovery_progress
from src.environment.actions import Action, ACTION_DELTAS
from src.mapf.mdd_sat import (
    position_paths_to_actions,
    return_planning_budget,
    ReturnPlanningBudgetExceeded,
)
from src.marl.digital_twin_adapter import get_digital_twin_inputs
from src.marl.real_training_step import run_real_training_step



@contextmanager
def _profile_return_planning(enabled):
    path = os.environ.get("AGV_RETURN_PROFILE_PATH")

    if not enabled or not path:
        yield
        return

    profiler = cProfile.Profile()
    profiler.enable()

    try:
        yield
    finally:
        profiler.disable()
        stream = io.StringIO()

        pstats.Stats(
            profiler,
            stream=stream,
        ).sort_stats("cumulative").print_stats(35)

        with open(path, "a") as handle:
            handle.write("\n=== RETURN PLANNING PROFILE ===\n")
            handle.write(stream.getvalue())


def _plan(
    grid, starts, goals, max_iterations,
    budget_return=False, diagnostics=None,
):
    began = time.perf_counter()
    context = (
        return_planning_budget(diagnostics=diagnostics)
        if budget_return else nullcontext()
    )
    try:
        with context, _profile_return_planning(budget_return):
            solution = solve_lns_sat(
                grid=grid, starts=starts, goals=goals,
                max_iterations=max_iterations,
            )
    except ReturnPlanningBudgetExceeded as exc:
        solution = None
        if diagnostics is not None:
            diagnostics["outcome"] = "BUDGET_EXHAUSTED"
            diagnostics["reason"] = str(exc)

    elapsed = time.perf_counter() - began

    if diagnostics is not None and "outcome" not in diagnostics:
        diagnostics["outcome"] = (
            "PLAN_FOUND" if solution is not None
            else "NO_PLAN_WITHIN_LNS_LIMITS"
        )
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



def _positions_changed_since_failure(dt, failed_starts):
    if failed_starts is None:
        return True

    candidates = dt["possible_current_positions"]
    if set(candidates) != set(failed_starts):
        return False

    for agent_id, old_position in failed_starts.items():
        positions = candidates[agent_id]
        if len(positions) != 1:
            return False
        if tuple(next(iter(positions))) != tuple(old_position):
            return True

    return False

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
    use_dt_observations=False,
    use_dt_inferred_centers=False,
    allow_return_to_lns=True,
    diagnostic_episode_id=None,
    enable_single_agv_completion=False,
    enable_two_agv_completion=False,
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
    last_failed_return_starts = None
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
            allow_return_to_lns
            and supervisor.mode == "M7"
            and decision == "RETURN_READY"
            and (not recovery.active or recovery_confirmed)
            and step >= next_return_attempt
            and _positions_changed_since_failure(
                dt, last_failed_return_starts
            )
        ):
            candidates = dt["possible_current_positions"]
            # Observe() already verifies singleton sets and AoI=0.
            observed_starts = {
                agent_id: next(iter(candidates[agent_id]))
                for agent_id in sorted(candidates)
            }
            return_diagnostics = {}
            new_plan = None
            elapsed = 0.0

            # Experimental return-only completion mode.
            # Never use simulator positions to select this route.
            use_single_agv = (
                enable_single_agv_completion
                and getattr(
                    executor.delay_model,
                    "immediate_probability", 0.0
                ) == 1.0
                and not _has_pending(executor)
                and (handover is None or not handover.active)
                and all(value == 0 for value in dt["aoi_values"].values())
                and all(
                    len(positions) == 1
                    for positions in dt["possible_current_positions"].values()
                )
            )

            if use_single_agv:
                began = time.perf_counter()
                candidate = plan_single_agv_completion(
                    grid=simulator.grid,
                    positions=observed_starts,
                    goals=dict(simulator.agent_goals),
                )
                elapsed = time.perf_counter() - began

                if candidate is not None:
                    agent_id, actions = candidate
                    if len(actions) <= context["max_steps"] - step:
                        new_plan = {
                            "actions": {
                                a: (
                                    actions
                                    if a == agent_id
                                    else [0] * len(actions)
                                )
                                for a in observed_starts
                            },
                            "horizon": len(actions),
                            "offset": 0,
                        }
                        # Expected positions come from the DT start
                        # and the independently planned action sequence.
                        route_positions = [observed_starts[agent_id]]
                        for action in actions:
                            dx, dy = ACTION_DELTAS[Action(action)]
                            x, y = route_positions[-1]
                            route_positions.append((x + dx, y + dy))

                        new_plan["single_agv_completion"] = True
                        new_plan["active_agent"] = agent_id
                        new_plan["route_positions"] = route_positions

                        return_diagnostics["outcome"] = (
                            "SINGLE_AGV_PLAN_FOUND"
                        )
                        return_diagnostics["active_agent"] = agent_id

            # Experimental sequential two-AGV recovery.
            # Uses verified DT positions, never simulator positions.
            use_two_agv = (
                enable_two_agv_completion
                and new_plan is None
                and getattr(
                    executor.delay_model,
                    "immediate_probability", 0.0
                ) == 1.0
                and not _has_pending(executor)
                and (handover is None or not handover.active)
                and all(
                    value == 0
                    for value in dt["aoi_values"].values()
                )
                and set(observed_starts) == set(simulator.agent_goals)
                and all(
                    len(positions) == 1
                    for positions in
                    dt["possible_current_positions"].values()
                )
            )

            if use_two_agv:
                began = time.perf_counter()
                candidate = plan_two_agv_completion(
                    grid=simulator.grid,
                    positions=observed_starts,
                    goals=dict(simulator.agent_goals),
                )
                elapsed += time.perf_counter() - began

                if (
                    candidate is not None
                    and candidate["horizon"]
                    <= context["max_steps"] - step
                ):
                    new_plan = candidate
                    return_diagnostics["outcome"] = (
                        "TWO_AGV_PLAN_FOUND"
                    )
                    return_diagnostics["order"] = list(
                        candidate["order"]
                    )
                    return_diagnostics["stages"] = (
                        candidate["stages"]
                    )

            if new_plan is None:
                new_plan, sat_elapsed = _plan(
                    simulator.grid, observed_starts,
                    dict(simulator.agent_goals), return_max_iterations,
                    budget_return=True,
                    diagnostics=return_diagnostics,
                )
                elapsed += sat_elapsed

            planning_time += elapsed
            replans += 1

            diagnostic_path = os.environ.get("AGV_RETURN_DIAGNOSTICS_PATH")
            if diagnostic_path:
                goals = simulator.agent_goals
                remaining_manhattan = sum(
                    abs(observed_starts[a][0] - goals[a][0])
                    + abs(observed_starts[a][1] - goals[a][1])
                    for a in observed_starts
                )
                record = {
                    "episode_id": diagnostic_episode_id,
                    "timestep": step,
                    "attempt_index": replans,
                    "observed_starts": observed_starts,
                    "remaining_manhattan": remaining_manhattan,
                    "planning_seconds": elapsed,
                    "plan_found": new_plan is not None,
                    "budget_diagnostics": return_diagnostics,
                }
                with open(diagnostic_path, "a") as handle:
                    handle.write(json.dumps(record) + "\n")

            if new_plan is not None and (
                new_plan.get("single_agv_completion")
                or new_plan.get("two_agv_completion")
            ):
                new_plan["recovery_expected_positions"] = (
                    build_expected_positions(
                        observed_starts,
                        new_plan["actions"],
                        new_plan["horizon"],
                    )
                )

            next_return_attempt = step + replan_cooldown_steps
            if new_plan is not None:
                last_failed_return_starts = None
                new_plan["offset"] = step
                plan = new_plan
                supervisor.confirm_return(timestep=step)
                if recovery.active:
                    recovery.finish(step)
                    recovery_confirmed = False
                    last_drain_timestep = None
            else:
                last_failed_return_starts = dict(observed_starts)
                # Stay in M7; avoid attempting the same replan every timestep.
                planning_failures += 1
                supervisor.reliable_steps = 0
                if recovery.active:
                    recovery.finish(step)
                    recovery_confirmed = False
                    last_drain_timestep = None

        # Fail closed if an experimental recovery route diverges
        # from fresh, unambiguous Digital Twin observations.
        if (
            supervisor.mode == "LNS-SAT"
            and plan is not None
            and plan.get("recovery_expected_positions") is not None
        ):
            recovery_index = step - plan["offset"]

            if not recovery_route_is_valid(
                plan=plan,
                dt=dt,
                index=recovery_index,
                pending_commands=_has_pending(executor),
            ):
                # Avoid repeated recovery attempts from an unchanged
                # rejected DT state.
                next_return_attempt = max(
                    next_return_attempt,
                    step + replan_cooldown_steps,
                )

                candidates = dt["possible_current_positions"]
                if (
                    set(candidates) == set(simulator.agent_goals)
                    and all(len(p) == 1 for p in candidates.values())
                ):
                    last_failed_return_starts = {
                        agent: next(iter(positions))
                        for agent, positions in candidates.items()
                    }

                supervisor.fallback_to_m7(timestep=step)
                plan = None
                planning_failures += 1

                if handover is not None and not handover.active:
                    handover.start()

        before = dict(simulator.agent_positions)
        progress_before = simulator.non_progress_steps
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

            if enable_single_agv_completion:
                acknowledge_recovery_progress(
                    simulator=simulator,
                    plan=plan,
                    index=index,
                    before=before,
                    executed=executed,
                    collision=collision,
                )

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
                use_dt_observations=use_dt_observations,
                use_dt_inferred_centers=use_dt_inferred_centers,
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

        # Diagnostic only: never influences controller decisions.
        trace_path = os.environ.get("AGV_SINGLE_AGV_TRACE_PATH")
        if (
            trace_path
            and str(diagnostic_episode_id) in {"2", "12"}
            and 110 <= step <= 140
        ):
            agent_id = 0
            current = simulator.agent_positions[agent_id]
            previous = before[agent_id]
            goal = simulator.agent_goals[agent_id]

            record = {
                "episode_id": diagnostic_episode_id,
                "step": step,
                "mode": supervisor.mode,
                "before": previous,
                "after": current,
                "goal": goal,
                "command": commands.get(agent_id),
                "moved": current != previous,
                "distance_before": simulator.navigation_distance(
                    previous, goal
                ),
                "distance_after": simulator.navigation_distance(
                    current, goal
                ),
                "non_progress_before": progress_before,
                "non_progress_after": simulator.non_progress_steps,
                "deadlock": deadlock,
                "collision": collision,
                "plan_offset": (
                    plan.get("offset") if plan is not None else None
                ),
            }

            with open(trace_path, "a") as handle:
                handle.write(json.dumps(record) + "\n")

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
