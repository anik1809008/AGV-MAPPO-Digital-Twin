from src.baselines.m1_episode_runner import run_m1_episode
from src.evaluation.episode_metrics_adapter import (
    build_marl_episode_metrics,
)
from src.evaluation.metrics import measure_planning_time
from src.evaluation.marl_method_runner import run_marl_method


def run_real_method(
    method,
    latency_steps,
    context,
):
    if method == "M1":
        result, elapsed = measure_planning_time(
            run_m1_episode,
            simulator=context["simulator"],
            max_steps=context["max_steps"],
        )

        metrics = dict(result["metrics"])
        metrics["planning_time"] = elapsed
        return metrics

    episode_result, elapsed = measure_planning_time(
        run_marl_method,
        method=method,
        actor=context["actor"],
        critic=context["critic"],
        simulator=context["simulator"],
        multi_agent_buffer=context["buffer"],
        digital_twin=context.get("digital_twin"),
        telemetry_channel=context.get("telemetry_channel"),
        delayed_executor=context.get("delayed_executor"),
        m4_controller=context.get("m4_controller"),
        m5_baseline=context.get("m5_baseline"),
        trusted_positions=context.get("trusted_positions"),
        reachable_occupancies=context.get(
            "reachable_occupancies"
        ),
        aoi_values=context.get("aoi_values"),
        max_steps=context["max_steps"],
    )

    shield_interventions = 0

    if (
        method == "M4"
        and context.get("m4_controller") is not None
    ):
        shield_interventions = (
            context["m4_controller"]
            .action_filter
            .intervention_count
        )

    return build_marl_episode_metrics(
        method=method,
        episode_result=episode_result,
        planning_time=elapsed,
        shield_interventions=shield_interventions,
    )
