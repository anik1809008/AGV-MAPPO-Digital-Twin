"""Eligibility checks for experimental DT-inferred completion."""


def get_inferred_completion_positions(
    *,
    dt,
    goals,
    immediate_probability,
    pending_commands,
    handover_active=False,
    max_aoi=2,
):
    """Return eligible DT-inferred positions or None."""

    if max_aoi < 1:
        raise ValueError("max_aoi must be >= 1")

    if immediate_probability != 1.0:
        return None

    if pending_commands or handover_active:
        return None

    candidates = dt.get("possible_current_positions", {})
    ages = dt.get("aoi_values", {})
    agent_ids = set(goals)

    if not agent_ids:
        return None

    if set(candidates) != agent_ids:
        return None

    if set(ages) != agent_ids:
        return None

    if any(not 0 <= age <= max_aoi for age in ages.values()):
        return None

    # Preserve the existing fresh-state recovery path.
    if not any(age > 0 for age in ages.values()):
        return None

    positions = {}

    for agent_id in sorted(agent_ids):
        possible = candidates[agent_id]

        if len(possible) != 1:
            return None

        positions[agent_id] = tuple(next(iter(possible)))

    # Reject inconsistent inferred occupancy.
    if len(set(positions.values())) != len(positions):
        return None

    unfinished = [
        agent_id
        for agent_id in agent_ids
        if positions[agent_id] != tuple(goals[agent_id])
    ]

    if len(unfinished) not in (1, 2):
        return None

    return positions


def inferred_recovery_route_is_valid(
    *,
    plan,
    dt,
    index,
    immediate_probability,
    pending_commands,
    handover_active=False,
    max_aoi=2,
):
    """Validate an experimental DT-inferred recovery route.

    Uses DT observations and expected route positions only.
    Rejects uncertain or inconsistent execution states.
    """

    if max_aoi < 1:
        raise ValueError("max_aoi must be >= 1")

    if immediate_probability != 1.0:
        return False

    if pending_commands or handover_active:
        return False

    expected = plan.get("recovery_expected_positions")

    if expected is None:
        return False

    if not 0 <= index < len(expected):
        return False

    target = expected[index]

    if not target:
        return False

    candidates = dt.get("possible_current_positions", {})
    ages = dt.get("aoi_values", {})

    if set(candidates) != set(target):
        return False

    if set(ages) != set(target):
        return False

    verified_positions = []

    for agent_id, position in target.items():
        if not 0 <= ages[agent_id] <= max_aoi:
            return False

        possible = candidates[agent_id]

        if len(possible) != 1:
            return False

        inferred = tuple(next(iter(possible)))

        if inferred != tuple(position):
            return False

        verified_positions.append(inferred)

    if len(set(verified_positions)) != len(verified_positions):
        return False

    return True


def build_inferred_completion_plan(
    *,
    grid,
    positions,
    goals,
    remaining_steps,
    enable_single=True,
    enable_two=True,
):
    """Build a bounded completion plan from inferred DT positions."""

    from src.environment.actions import Action
    from src.hybrid.single_agv_completion import (
        plan_single_agv_completion,
    )
    from src.hybrid.two_agv_completion import (
        plan_two_agv_completion,
    )
    from src.hybrid.recovery_route_guard import (
        build_expected_positions,
    )

    if remaining_steps <= 0:
        return None

    if set(positions) != set(goals):
        return None

    unfinished = [
        agent_id
        for agent_id in goals
        if tuple(positions[agent_id]) != tuple(goals[agent_id])
    ]

    if len(unfinished) == 1 and enable_single:
        candidate = plan_single_agv_completion(
            grid, positions, goals
        )

        if candidate is None:
            return None

        agent_id, actions = candidate
        horizon = len(actions)

        if not 0 < horizon <= remaining_steps:
            return None

        plan = {
            "actions": {
                a: (
                    actions
                    if a == agent_id
                    else [int(Action.WAIT)] * horizon
                )
                for a in positions
            },
            "horizon": horizon,
            "offset": 0,
            "single_agv_completion": True,
            "active_agent": agent_id,
        }

    elif len(unfinished) == 2 and enable_two:
        plan = plan_two_agv_completion(
            grid, positions, goals
        )

        if plan is None:
            return None

        if not 0 < plan["horizon"] <= remaining_steps:
            return None

    else:
        return None

    expected = build_expected_positions(
        positions,
        plan["actions"],
        plan["horizon"],
    )

    # Validate that the generated route has no duplicate
    # occupancy and moves exactly one agent per step.
    for before, after in zip(expected, expected[1:]):
        if len(set(after.values())) != len(after):
            return None

        moving = [
            agent_id
            for agent_id in before
            if before[agent_id] != after[agent_id]
        ]

        if len(moving) != 1:
            return None

        for agent_id, position in after.items():
            x, y = position

            if not (
                0 <= y < len(grid)
                and 0 <= x < len(grid[y])
            ):
                return None

            if grid[y][x] != 0:
                return None

            if (
                before[agent_id] == tuple(goals[agent_id])
                and after[agent_id] != before[agent_id]
            ):
                return None

    if expected[-1] != {
        a: tuple(goal) for a, goal in goals.items()
    }:
        return None

    plan["recovery_expected_positions"] = expected
    plan["dt_inferred_completion"] = True

    if plan.get("single_agv_completion"):
        agent_id = plan["active_agent"]

        plan["route_positions"] = [
            state[agent_id] for state in expected
        ]

    return plan
