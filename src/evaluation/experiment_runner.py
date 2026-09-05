from src.evaluation.results_writer import append_result_csv


def run_experiment_sweep(
    methods,
    latency_levels,
    run_method_fn,
    results_path,
    scenario_id,
    agent_count,
    m5_threshold=None,
):
    results = []

    for latency_steps in latency_levels:
        for method in methods:
            metrics = run_method_fn(
                method=method,
                latency_steps=latency_steps,
            )

            metrics["scenario_id"] = scenario_id
            metrics["agent_count"] = agent_count
            metrics["latency_steps"] = latency_steps

            if method == "M5":
                metrics["m5_threshold"] = m5_threshold
            else:
                metrics["m5_threshold"] = ""

            append_result_csv(
                results_path,
                metrics,
            )

            results.append(metrics)

    return results
