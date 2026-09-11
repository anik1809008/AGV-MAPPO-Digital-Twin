from src.environment.movingai_map import load_movingai_map
from src.environment.movingai_scenario import load_movingai_scenario


def build_training_instance(
    map_path,
    scenario_path,
    agent_count,
):
    if agent_count < 1:
        raise ValueError("agent_count must be >= 1")

    map_data = load_movingai_map(map_path)
    scenarios = load_movingai_scenario(scenario_path)

    if len(scenarios) < agent_count:
        raise ValueError(
            "scenario file does not contain enough agents"
        )

    selected = scenarios[:agent_count]

    starts = {
        agent_id: scenario["start"]
        for agent_id, scenario in enumerate(selected)
    }

    goals = {
        agent_id: scenario["goal"]
        for agent_id, scenario in enumerate(selected)
    }

    return {
        "grid": map_data["grid"],
        "starts": starts,
        "goals": goals,
    }
