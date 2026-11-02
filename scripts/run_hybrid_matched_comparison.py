"""Small paired-instance pilot: LNS-SAT nominal vs M7/Hybrid under uncertainty.

WARNING: Execution conditions and planning-time definitions differ; this is a
presentation pilot, NOT a controlled estimate of hybrid improvement over LNS-SAT.
"""
import argparse
import csv
import os
import random
import time

import numpy as np
import torch

from src.environment.execution_delay import ExecutionDelayModel
from src.environment.simulator import GroundTruthSimulator
from src.evaluation.context_factory import build_experiment_context
from src.evaluation.controller_factory import build_method_controllers
from src.evaluation.model_loader import load_validation_models
from src.evaluation.real_method_callback import run_real_method
from src.hybrid.episode_runner import run_hybrid_episode
from src.hybrid.supervisor import HybridSupervisor
from src.mapf.lns_sat_runner import run_lns_sat_episode
from src.marl.scenario_split import get_random_scenario_path
from src.marl.training_instance import build_training_instance

MAP_PATH = 'benchmarks/movingai/warehouse-10-20-10-2-1.map'
METHODS = ('LNS-SAT', 'M7', 'HYBRID')
FIELDS = (
    'method', 'scenario_id', 'start_index', 'agent_count', 'seed',
    'checkpoint', 'uncertainty_condition', 'latency_steps',
    'immediate_probability', 'success', 'collision', 'deadlock',
    'planning_failed', 'makespan', 'path_length', 'planning_time',
    'time_definition', 'lns_steps', 'm7_steps', 'switch_count',
    'switch_timestep', 'switch_reason', 'shield_interventions',
)


def trial_schedule(count):
    if not 1 <= count <= 50:
        raise ValueError('trials must be between 1 and 50')
    # Same 50-instance index order as the expanded LNS-SAT benchmark.
    return [(16 + (i % 5), 8 * (i // 5)) for i in range(count)]


def normalize_lns(result):
    if result.get('planning_failed', False):
        return dict(success=False, collision=False, deadlock=False,
                    planning_failed=True, makespan='', path_length='',
                    planning_time='', shield_interventions=0)
    metrics = result['metrics']
    return dict(success=bool(metrics['success']),
                collision=bool(metrics['collision']),
                deadlock=bool(metrics['deadlock']), planning_failed=False,
                makespan=metrics['makespan'], path_length=metrics['path_length'],
                planning_time=metrics['planning_time'],
                shield_interventions=0)


def run_trial(method, instance, args, models):
    # Identical start/goal instance across methods; fresh state each time.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if method == 'LNS-SAT':
        simulator = GroundTruthSimulator(
            grid=instance['grid'], agent_positions=dict(instance['starts']),
            agent_goals=dict(instance['goals']))
        result = normalize_lns(run_lns_sat_episode(simulator))
        return result, {'time_definition': 'LNS-SAT planning only'}

    controllers = build_method_controllers(
        method='M7', actor=models['actor'], grid=instance['grid'])
    context = build_experiment_context(
        grid=instance['grid'], starts=instance['starts'], goals=instance['goals'],
        actor=models['actor'], critic=models['critic'],
        latency_steps=args.latency, max_steps=args.max_steps,
        m7_controller=controllers['m7_controller'],
        m7_progress_monitor=controllers['m7_progress_monitor'],
        m7_recovery_planner=controllers['m7_recovery_planner'],
        execution_delay_model=ExecutionDelayModel(
            immediate_probability=args.immediate_probability, seed=args.seed),
    )
    if method == 'M7':
        # Existing callback's planning_time measures full episode wall time.
        metrics = run_real_method(method='M7', latency_steps=args.latency,
                                  context=context)
        return dict(
            success=bool(metrics['success']), collision=bool(metrics['collision']),
            deadlock=bool(metrics['deadlock']), planning_failed=False,
            makespan=metrics['makespan'], path_length=metrics['path_length'],
            planning_time=metrics['planning_time'],
            shield_interventions=metrics.get('shield_interventions', 0),
        ), {'time_definition': 'full M7 episode wall time'}

    supervisor = HybridSupervisor(aoi_threshold=args.aoi_threshold)
    hybrid = run_hybrid_episode(context=context, supervisor=supervisor)
    events = hybrid['switch_events']
    return dict(
        success=bool(hybrid['all_goals_reached']),
        collision=bool(hybrid['collision']), deadlock=bool(hybrid['deadlock']),
        planning_failed=bool(hybrid['planning_failed']),
        makespan=hybrid['steps'], path_length=hybrid['path_length'],
        planning_time=hybrid['planning_time'],
        shield_interventions=(controllers['m7_controller'].action_filter.intervention_count),
    ), dict(
        time_definition='initial LNS-SAT planning only',
        lns_steps=hybrid['lns_steps'], m7_steps=hybrid['m7_steps'],
        switch_count=len(events),
        switch_timestep=events[0].timestep if events else '',
        switch_reason=events[0].reason if events else '',
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--episodes-per-scenario', type=int, default=40)
    parser.add_argument('--latency', type=int, default=2)
    parser.add_argument('--aoi-threshold', type=int, default=2)
    parser.add_argument('--immediate-probability', type=float, default=0.8)
    parser.add_argument('--max-steps', type=int, default=400)
    parser.add_argument('--results-path', default='results/hybrid_comparison_pilot_10.csv')
    args = parser.parse_args()
    schedule = trial_schedule(args.trials)
    if not 0.0 <= args.immediate_probability <= 1.0:
        parser.error('immediate-probability must be in [0, 1]')
    if os.path.exists(args.results_path):
        parser.error('Output exists; select a new --results-path to avoid overwriting data')
    os.makedirs(os.path.dirname(args.results_path) or '.', exist_ok=True)

    models = load_validation_models(method='M7', agent_count=8,
                                    seed=args.seed,
                                    episodes_per_scenario=args.episodes_per_scenario)
    print('CHECKPOINT:', models['checkpoint_path'], flush=True)
    with open(args.results_path, 'x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for i, (scenario_id, start_index) in enumerate(schedule, 1):
            instance = build_training_instance(
                map_path=MAP_PATH,
                scenario_path=get_random_scenario_path(scenario_id),
                agent_count=8, start_index=start_index,
            )
            for method in METHODS:
                metrics, extras = run_trial(method, instance, args, models)
                condition = ('nominal' if method == 'LNS-SAT'
                             else 'combined_telemetry_and_execution_uncertainty')
                row = dict(
                    method=method, scenario_id=scenario_id,
                    start_index=start_index, agent_count=8, seed=args.seed,
                    checkpoint=(models['checkpoint_path'] if method != 'LNS-SAT' else ''),
                    uncertainty_condition=condition,
                    latency_steps=(0 if method == 'LNS-SAT' else args.latency),
                    immediate_probability=(1.0 if method == 'LNS-SAT' else args.immediate_probability),
                )
                row.update(metrics)
                row.update(extras)
                writer.writerow({name: row.get(name, '') for name in FIELDS})
                f.flush()
                print(f'[{i}/{len(schedule)}] {method:7} scenario={scenario_id} '
                      f'start={start_index} success={row["success"]} '
                      f'collision={row["collision"]} '
                      f'planning_failed={row["planning_failed"]}', flush=True)
    print('CSV:', args.results_path)
    print('NOTE: LNS-SAT nominal vs M7/Hybrid uncertain; planning_time definitions differ.')


if __name__ == '__main__':
    main()
