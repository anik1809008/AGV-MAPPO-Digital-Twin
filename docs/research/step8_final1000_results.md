# Step 8: Frozen Two-Way Hybrid Evaluation

## Experiment

- Repository branch: feature/execution-aware-hybrid
- Frozen implementation commit: 4193f66
- Episodes: 1,000 per method
- Distinct start/goal configurations: 200
- AGVs: 8
- Test scenarios: 21–25
- Pretrained model seeds: 0–4
- Maximum episode steps: 400
- Perfect-condition episodes: 500
- Uncertain-condition episodes: 500

## Methods

1. Original Two-Way Hybrid
2. Improved Two-Way Hybrid
   - Single-AGV recovery
   - Sequential Two-AGV recovery
   - Digital Twin route validation
   - Route-aware progress verification
   - Safe fallback and execution-aware handover

## Final Results

| Metric | Original | Improved |
|---|---:|---:|
| Overall success | 727/1000 (72.7%) | 808/1000 (80.8%) |
| Perfect-condition success | 412/500 (82.4%) | 493/500 (98.6%) |
| Uncertain-condition success | 315/500 (63.0%) | 315/500 (63.0%) |
| Collisions | 0 | 0 |
| Deadlocks | 7 | 7 |
| Total planning time (s) | 20780.06 | 17492.63 |

## Paired Outcomes

- Baseline failures recovered: 81
- Baseline successes lost: 0
- Overall improvement: +8.1 percentage points
- Perfect-condition improvement: +16.2 percentage points
- Uncertain-condition improvement: 0 percentage points
- Unadjusted exact McNemar p-value: approximately 8.27e-25

## Research Interpretation

The improved controller substantially increases completion
success under reliable telemetry and immediate execution.

No overall success-rate improvement was observed in the
uncertain-condition subset.

No collisions were observed. The number of recorded
deadlocks remained unchanged.

## Limitations

- Single warehouse map.
- Five held-out scenario files.
- Five pretrained model seeds.
- Repeated seeds share start/goal configurations.
- Configuration-aware confidence intervals are pending.
- Ablation and published-method comparisons are pending.
- The new specialized recovery mechanisms require fresh,
  unambiguous Digital Twin state and immediate execution.

## Research Status

Promising performance contribution, but scientific novelty,
broad generalization, and robustness improvements are
not yet established.

## Reproduction Files

- results/step8_final1000_matrix.csv
- results/step8_final1000_baseline.csv
- results/step8_final1000_improved.csv
