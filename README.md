# Uncertainty-Aware Two-Way Hybrid Digital Twin for Multi-AGV Navigation

A simulation-based research framework for reliable
multi-AGV navigation in smart manufacturing under
communication latency and uncertain command execution.

## Research Objective

Integrate classical multi-agent path planning,
multi-agent reinforcement learning, uncertainty-aware
Digital Twin state inference, safety mechanisms and
adaptive completion recovery.

## Final Architecture

- MDD-SAT and LNS-SAT classical planning.
- MAPPO-based M7 adaptive navigation.
- Age of Information (AoI).
- Command-aware Digital Twin state inference.
- Reachability-based uncertainty representation.
- Uncertainty-aware safety shielding.
- Robust progress monitoring and local recovery.
- Execution-aware bidirectional switching.
- Pending-command management and safe handover.
- Single-AGV completion recovery.
- Sequential Two-AGV completion recovery.
- DT-Inferred Completion Recovery (DICR).

## Research Development: M1-M7

| Method | Description |
|---|---|
| M1 | Classical navigation baseline |
| M2 | Conventional MAPPO |
| M3 | Uncertainty-aware DT-MAPPO |
| M4 | DT-MAPPO with safety shielding |
| M5 | Reachable-set threshold waiting |
| M6 | AoI-only MAPPO ablation |
| M7 | Safety, progress monitoring and local recovery |

The M1-M7 investigation provided the foundation
for the final two-way Hybrid architecture.

## Final Experimental Configuration

- MovingAI warehouse-10-20-10-2-1 benchmark.
- Eight AGVs.
- Training scenarios: 1-15.
- Validation scenarios: 16-20.
- Test scenarios: 21-25.
- Five pretrained model seeds: 0-4.
- 200 distinct start/goal configurations.
- 1000 matched episodes per configuration.
- Maximum episode horizon: 400 timesteps.
- Telemetry latency: 0, 1, 2 and 4 timesteps.
- Immediate execution probability: 0.8 or 1.0.

## Final Matched Experimental Results

| Configuration | Success Rate | Collisions | Deadlocks |
|---|---:|---:|---:|
| Original Two-Way Hybrid | 72.7% | 0 | 7 |
| Single-AGV Recovery | 77.8% | 0 | 7 |
| Two-AGV Recovery | 78.8% | 0 | 7 |
| Combined Recovery | 80.8% | 0 | 7 |
| Combined Recovery + DICR | **84.7%** | 0 | 7 |

The final architecture recovered 120 additional
episodes relative to the original Hybrid on the
same experimental matrix, without success regressions.

DICR contributed 39 additional successful episodes
relative to Combined Recovery.

## DICR Performance

| Condition | Without DICR | With DICR |
|---|---:|---:|
| Perfect conditions | 98.6% | 98.6% |
| All imperfect conditions | 63.0% | 70.8% |
| Deterministic delayed conditions | 80.5% | 100% |
| Stochastic execution | 51.33% | 51.33% |

DICR enables guarded classical completion recovery
from unique Digital Twin-inferred positions under
bounded telemetry latency and deterministic execution.

No additional MAPPO retraining was required.

## Statistical Validation

The primary DICR comparison achieved:

- Success improvement: +3.9 percentage points.
- Route-cluster 95% CI: [1.6, 6.5] percentage points.
- Holm-adjusted route-cluster p-value: 0.0014648438.
- Zero success regressions.
- Zero new deadlocks.
- Zero observed collisions.

Scenario-level statistical significance was not
established.

## Historical Held-Out Benchmark

An earlier 1000-episode benchmark reported:

| Method | Success Rate |
|---|---:|
| LNS-SAT | 29.0% |
| DT-Only Hybrid | 55.5% |
| DT-Robust M7 | 58.4% |
| One-Way Robust Hybrid | 59.9% |
| Two-Way Robust Hybrid | 60.5% |

This historical benchmark uses a different
experimental matrix and must not be directly
compared with the final 84.7% recovery benchmark.

## Research Documentation

- [Complete Research Journey](docs/research/research_journey.md)
- [Final Research Summary](docs/research/final_research_summary.md)
- [Final Methodology](docs/research/final_methodology.md)
- [Frozen Step 8 Evaluation](docs/research/step8_final1000_results.md)
- [Historical Experimental Evidence](results/research_history_extraction.txt)
- [Recovery Ablation Analysis](results/step12_ablation_analysis.txt)
- [Final DICR Statistical Analysis](results/step14_dicr_analysis.txt)

## Reproducibility

- [Final DICR Reproduction Instructions](reproducibility/final_dicr_1000/README.md)
- [Final Reproducibility Manifest](reproducibility/final_dicr_1000/manifest.json)
- [Historical Reproducibility Package](reproducibility/heldout_1000/)

Recorded software environment:

- Python 3.12.3
- NumPy 2.5.2
- PyTorch 2.14.0+cpu
- Pytest 9.1.1
- Python-SAT 1.9.dev15

All five frozen M3 checkpoint fingerprints match
the historical checkpoint manifest.

Checkpoint binaries are not included in Git and
are required for full experimental reproduction.

## Software Testing

Run from the repository root:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q

Final recorded result:

    432 passed

## Research Limitations

- Simulation-based evaluation on one warehouse map.
- Five test scenario files and five pretrained seeds.
- DICR requires deterministic command execution.
- No DICR improvement under stochastic execution.
- Increased makespan in some successful episodes.
- Scenario-level statistical significance not established.
- Independent generalization remains unverified.
- Some evaluation episodes informed algorithm development.
- Simulator-based recovery progress verification.
- Zero observed collisions is not a formal safety proof.
- Physical industrial deployment remains unverified.
- Scientific novelty requires literature verification.

## Project Status

Algorithm development, experimental evaluation,
statistical analysis and research documentation
have been completed within the current research scope.

The final selected architecture is the execution-aware
Two-Way Hybrid with Combined Completion Recovery
and DICR.

Final recorded performance: 847/1000 successful
episodes, zero observed collisions and seven deadlocks.
