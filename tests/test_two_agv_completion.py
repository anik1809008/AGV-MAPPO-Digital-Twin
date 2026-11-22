from src.environment.actions import Action
from src.environment.simulator import GroundTruthSimulator
from src.hybrid.two_agv_completion import plan_two_agv_completion


def make_case():
    grid = [[0] * 6 for _ in range(3)]
    positions = {0: (0, 0), 1: (5, 2), 2: (3, 1)}
    goals = {0: (5, 0), 1: (0, 2), 2: (3, 1)}
    return grid, positions, goals


def test_two_agv_plan_reaches_all_goals():
    grid, positions, goals = make_case()
    plan = plan_two_agv_completion(grid, positions, goals)

    assert plan is not None
    assert plan["two_agv_completion"]
    assert set(plan["actions"]) == set(positions)

    simulator = GroundTruthSimulator(grid, positions, goals)

    for step in range(plan["horizon"]):
        commands = {
            agent: actions[step]
            for agent, actions in plan["actions"].items()
        }

        assert sum(
            action != int(Action.WAIT)
            for action in commands.values()
        ) == 1

        transition = simulator.step_joint(commands)
        assert not transition["collision"]
        assert not simulator.is_deadlocked()

    assert simulator.all_goals_reached()


def test_rejects_more_than_two_unfinished_agents():
    grid, positions, goals = make_case()
    goals[2] = (4, 1)

    assert plan_two_agv_completion(grid, positions, goals) is None


def test_rejects_only_one_unfinished_agent():
    grid, positions, goals = make_case()
    goals[1] = positions[1]

    assert plan_two_agv_completion(grid, positions, goals) is None


def test_rejects_blocked_routes():
    grid = [[0, 0, 0]]
    positions = {0: (0, 0), 1: (1, 0), 2: (2, 0)}
    goals = {0: (2, 0), 1: (0, 0), 2: (2, 0)}

    assert plan_two_agv_completion(grid, positions, goals) is None
