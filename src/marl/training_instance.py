from src.environment.movingai_map import load_movingai_map
from src.environment.movingai_scenario import load_movingai_scenario


def build_training_instance(
    map_path,
    scenario_path,
    agent_count,
    start_index=0,
):
    if agent_count < 1:
        raise ValueError("agent_count must be >= 1")

    if start_index < 0:
        raise ValueError("start_index must be >= 0")

    map_data = load_movingai_map(map_path)
    scenarios = load_movingai_scenario(scenario_path)

    end_index = start_index + agent_count

    if end_index > len(scenarios):
        raise ValueError(
            "scenario file does not contain enough agents"
        )

    selected = scenarios[start_index:end_index]

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
