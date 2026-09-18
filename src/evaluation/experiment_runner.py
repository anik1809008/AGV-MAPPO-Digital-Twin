from src.evaluation.results_writer import append_result_csv
def run_experiment_sweep(
    methods,
    latency_levels,
    run_method_fn,
    results_path,
    scenario_id,
    agent_count,
    m5_threshold=None,
    context_factory=None,
    seed=None,
    uncertainty_condition="latency_only",
    immediate_probability=1.0,
):
    results = []

    for latency_steps in latency_levels:
        for method in methods:
            context = None

            if context_factory is not None:
                context = context_factory(
                    method=method,
                    latency_steps=latency_steps,
                )

            if context is None:
                metrics = run_method_fn(
                    method=method,
                    latency_steps=latency_steps,
                )
            else:
                metrics = run_method_fn(
                    method=method,
                    latency_steps=latency_steps,
                    context=context,
                )
            metrics["scenario_id"] = scenario_id
            metrics["agent_count"] = agent_count
            metrics["seed"] = (
                "" if seed is None else seed
            )
            metrics["uncertainty_condition"] = (
                uncertainty_condition
            )
            metrics["latency_steps"] = latency_steps
            metrics["immediate_probability"] = (
                immediate_probability
            )
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
