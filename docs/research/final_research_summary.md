# Final Research Summary

## Project Title

Uncertainty-Aware Two-Way Hybrid Digital Twin for
Multi-AGV Navigation in Smart Manufacturing

## Research Objective

Develop and evaluate an uncertainty-aware multi-AGV
navigation architecture that combines classical path
planning, multi-agent reinforcement learning, Digital Twin
state estimation, safety mechanisms, and adaptive recovery.

## Final Architecture

The final proposed architecture incorporates:

- LNS-SAT classical multi-agent path planning.
- MAPPO-based M7 adaptive navigation.
- Age of Information (AoI).
- Command-aware Digital Twin state reconstruction.
- Reachability-based uncertainty representation.
- Safety shielding.
- Robust progress monitoring.
- Local recovery.
- Execution-aware bidirectional switching.
- Execution-aware command handover.
- Single-AGV completion recovery.
- Sequential Two-AGV completion recovery.
- DT-Inferred Completion Recovery (DICR).

## Final Experimental Configuration

- Benchmark: MovingAI warehouse.
- Number of AGVs: 8.
- Test scenarios: 21-25.
- Random seeds: 0-4.
- Episode horizon: 400 timesteps.
- Matched benchmark episodes: 1000 per configuration.
- Telemetry latency: 0, 1, 2 and 4 timesteps.
- Immediate command execution probability: 0.8 or 1.0.
- Existing frozen MAPPO checkpoints.
- No additional MAPPO retraining for completion recovery
  or DICR.

## Final Matched Recovery Ablation Results

| Configuration | Successes | Success Rate | Collisions | Deadlocks |
|---|---:|---:|---:|---:|
| Original Two-Way Hybrid | 727 | 72.7% | 0 | 7 |
| Single-AGV Recovery | 778 | 77.8% | 0 | 7 |
| Two-AGV Recovery | 788 | 78.8% | 0 | 7 |
| Combined Recovery | 808 | 80.8% | 0 | 7 |
| Combined Recovery + DICR | 847 | 84.7% | 0 | 7 |

The final architecture rescued 120 additional episodes
relative to the original Two-Way Hybrid, with no observed
success regressions.

DICR contributed 39 additional successful episodes relative
to Combined Recovery.

## Condition-Specific DICR Results

| Latency | Execution Probability | Episodes | Without DICR | With DICR |
|---:|---:|---:|---:|---:|
| 0 | 1.0 | 500 | 98.6% | 98.6% |
| 0 | 0.8 | 100 | 71% | 71% |
| 1 | 1.0 | 100 | 79% | 100% |
| 2 | 1.0 | 100 | 82% | 100% |
| 2 | 0.8 | 100 | 51% | 51% |
| 4 | 0.8 | 100 | 32% | 32% |

Aggregate results:

- Perfect conditions: 98.6% to 98.6%.
- Imperfect conditions: 63.0% to 70.8%.
- Deterministic delayed conditions: 80.5% to 100%.
- Stochastic execution: 51.33% to 51.33%.

## DICR Working Principle

The original completion recovery mechanism requires fresh
and reliable Digital Twin observations.

DICR extends completion recovery to bounded telemetry
latency when the Digital Twin reconstructs a unique
possible current position for every AGV.

Activation requires:

- Deterministic command execution.
- AoI bounded by two timesteps, with at least one
  observation having positive AoI.
- Unique inferred positions for all AGVs.
- No overlapping inferred positions.
- No pending execution commands.
- No active handover or recovery pause.
- One or two unfinished AGVs.
- Minimum supervisor dwell time.
- A feasible completion route within the remaining horizon.

The inferred state and expected recovery route are
validated before execution and at subsequent timesteps.

Invalid recovery routes trigger fallback to M7 through
execution-aware handover.

DICR reuses the existing completion planners rather than
introducing a new pathfinding algorithm.

## Statistical Validation

Paired comparisons account for repeated seeds using
route-cluster bootstrap confidence intervals and
cluster sign-flip tests.

| Comparison | Effect | 95% Route CI | Holm-adjusted Route p |
|---|---:|---:|---:|
| Original Hybrid vs Final DICR | +12.0 pp | [8.0, 16.3] | 0.0000399996 |
| Combined Recovery vs DICR | +3.9 pp | [1.6, 6.5] | 0.0014648438 |
| Imperfect conditions | +7.8 pp | [3.6, 12.8] | 0.0014648438 |
| Deterministic delayed conditions | +19.5 pp | [9.5, 31.0] | 0.0014648438 |

All four comparisons are significant at the route-cluster
level under the adopted statistical methodology.

Scenario-cluster sensitivity analysis did not establish
statistical significance. Only five scenario clusters
were available, and Holm-adjusted scenario p-values
were 0.25.

The confidence intervals are individual, not simultaneous.

## Efficiency Trade-Off

Among the 808 commonly successful episodes:

| Metric | Combined Recovery | DICR |
|---|---:|---:|
| Mean makespan | 170.8936 | 173.8106 |
| Mean path length | 668.7141 | 668.4295 |
| Mean planning time (seconds) | 13.8733 | 13.3989 |

Across all 1000 episodes:

- Mean planning time: 17.4926 to 16.4285 seconds.
- P95 planning time: 70.8954 to 68.5783 seconds.
- P99 planning time: 81.8040 to 78.2589 seconds.
- Maximum planning time: 103.8903 to 85.1578 seconds.

Among commonly successful episodes, makespan improved
in 3, remained unchanged in 718, and worsened in 87.

These efficiency comparisons are descriptive.

## Research Contributions

1. Systematic investigation of multi-AGV navigation
   through the M1-M7 methodology.

2. Command-aware Digital Twin representation of AGV
   state uncertainty.

3. Integration of classical planning and adaptive MARL
   through execution-aware two-way switching.

4. Integration of safety, progress monitoring and
   recovery mechanisms.

5. Completion-specific recovery for one or two
   unfinished AGVs.

6. Extension of completion recovery to bounded
   telemetry latency using Digital Twin-inferred states.

7. Matched experimental evaluation and cluster-aware
   statistical analysis of recovery contributions.

The contribution is the integrated architecture and
recovery decision mechanism. Individual algorithms
such as MAPPO and BFS are established methods.

## Research Limitations

- DICR does not improve stochastic execution conditions.
- DICR currently requires deterministic execution and
  bounded telemetry latency.
- Sequential completion recovery can increase makespan.
- Scenario-level statistical significance was not established.
- Independent generalization remains unverified because
  some benchmark episodes informed DICR development.
- Zero observed collisions is not a formal safety guarantee.
- The evaluation is simulation-based.
- Physical industrial deployment remains unverified.
- Worldwide novelty of the integrated DICR mechanism
  requires comprehensive literature verification.

## Experimental Evidence

- results/step8_final1000_matrix.csv
- results/step8_final1000_baseline.csv
- results/step11_single_only1000.csv
- results/step11_two_only1000.csv
- results/step8_final1000_improved.csv
- results/step12_ablation_statistics.csv
- results/step12_ablation_analysis.txt
- results/step13_failure_diagnostics.jsonl
- results/step13_failure_diagnostics.txt
- results/step13_dt_accuracy.jsonl
- results/step13_dt_accuracy.txt
- results/step13_dicr_pilot20_matrix.csv
- results/step13_dicr_pilot20.csv
- results/step13_dicr_pilot20_flag_off.csv
- results/step13_dicr_final1000.csv
- results/step14_dicr_statistics.csv
- results/step14_dicr_analysis.txt

## Implementation Validation

432 automated software tests passed.

DICR component and integration tests: 69 passed.

DICR matched pilot: 4 additional successes, zero
success regressions and zero observed collisions.

Flag-off reproducibility: 20/20 matched episodes
across 14 compared non-timing fields.

## Historical Benchmark Distinction

The earlier 1000-episode held-out benchmark reported:

- LNS-SAT: 29.0%.
- DT-Robust M7: 58.4%.
- One-Way Hybrid: 59.9%.
- Two-Way Hybrid: 60.5%.

These historical results belong to a different
experimental matrix and must not be directly treated
as controlled comparisons against the final 84.7%
recovery benchmark.

## Conclusion

The selected final architecture achieved 84.7% success
across 1000 matched benchmark episodes, with zero
observed collisions and seven deadlocks.

Compared with the original Hybrid on the same matrix,
the final architecture improved success by 12 percentage
points.

DICR specifically demonstrated that Digital Twin-inferred
state information can extend completion recovery to
bounded delayed communication under deterministic
execution, achieving 200/200 successful episodes in
the evaluated deterministic delayed conditions.

The findings support the proposed architecture within
the evaluated simulation conditions, subject to the
stated research limitations.
