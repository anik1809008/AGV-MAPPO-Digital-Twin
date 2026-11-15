# Uncertainty-Aware Hybrid Digital Twin for Multi-AGV Navigation

Research framework for multi-AGV navigation in smart manufacturing.

## Architecture

- LNS-SAT multi-agent path planning
- MAPPO M7 adaptive navigation
- AoI-aware Digital Twin monitoring
- Uncertainty-aware Robust Progress
- Bidirectional switching between LNS-SAT and MAPPO

## Held-Out Evaluation

- MovingAI warehouse benchmark
- Eight AGVs
- 1,000 matched episodes per method
- Scenarios 21-25
- Communication latency: 0-4 steps
- Execution probability: 0.8 or 1.0

| Method | Success Rate |
|---|---:|
| LNS-SAT | 29.0% |
| DT-Robust M7 | 58.4% |
| One-Way Hybrid | 59.9% |
| Two-Way Hybrid | 60.5% |

## Reproducibility

Evaluation results and checkpoint fingerprints:
reproducibility/heldout_1000/

Checkpoint binaries are not included in Git.

## Testing

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q

## Status

Evaluation and statistical analysis completed.
Journal manuscript preparation ongoing.
