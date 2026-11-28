# Final DICR Hybrid Reproducibility

## Research Configuration

This directory documents the final matched 1000-episode
evaluation of the uncertainty-aware two-way Hybrid
Digital Twin with Single-AGV, Two-AGV and DT-Inferred
Completion Recovery (DICR).

The historical heldout_1000 benchmark is separate and
uses a different experimental matrix.

## Recorded Final Results

| Method | Successes | Success Rate |
|---|---:|---:|
| Original Hybrid | 727 | 72.7% |
| Single-AGV Recovery | 778 | 77.8% |
| Two-AGV Recovery | 788 | 78.8% |
| Combined Recovery | 808 | 80.8% |
| Combined Recovery + DICR | 847 | 84.7% |

All five configurations recorded zero collisions
and seven deadlocks.

## Implementation

Branch:

    feature/execution-aware-hybrid

Final DICR implementation commit:

    f975eeb

Combined recovery implementation commit:

    4193f66

Final benchmark results commit:

    f0d115b

Final statistical analysis commit:

    367880e

Full commit identifiers are recorded in manifest.json.

## Software Environment

- Python 3.12.3
- numpy==2.5.2
- torch==2.14.0+cpu
- pytest==9.1.1
- python-sat==1.9.dev15

Create and activate an appropriate Python virtual
environment and install the recorded dependencies.

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

Verify the installed PyTorch build separately because
requirements.txt does not explicitly select the CPU build.

## Required Pretrained Checkpoints

The following frozen M3 checkpoint files are required:

    results/checkpoints/validation/m3_agents8_seed0_eps20.pt
    results/checkpoints/validation/m3_agents8_seed1_eps20.pt
    results/checkpoints/validation/m3_agents8_seed2_eps20.pt
    results/checkpoints/validation/m3_agents8_seed3_eps20.pt
    results/checkpoints/validation/m3_agents8_seed4_eps20.pt

The checkpoint binaries are not included in Git.

Their SHA-256 fingerprints are recorded in manifest.json.

All five checkpoint fingerprints match the historical
held-out benchmark manifest.

Obtaining the exact checkpoint binaries is required
for full experimental reproduction.

## Software Verification

Run from the repository root:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q

Recorded result:

    432 passed

## Experimental Matrix

    results/step8_final1000_matrix.csv

- Eight AGVs.
- Five test scenarios.
- Five pretrained seeds.
- 200 distinct start/goal configurations.
- 1000 matched episodes.
- Maximum episode horizon: 400 timesteps.

Do not regenerate the experimental matrix when
reproducing the recorded comparison.

## Reproduction Commands

Run all commands from the repository root.

Use a separate output directory to avoid overwriting
the original research evidence.

    mkdir -p results/reproduction

### Original Hybrid

    PYTHONPATH=. python -u scripts/run_four_method_evaluation.py \
        --method DT-ROBUST-HYBRID \
        --matrix results/step8_final1000_matrix.csv \
        --limit 1000 --max-steps 400 \
        --output results/reproduction/baseline.csv

### Single-AGV Recovery

    PYTHONPATH=. python -u scripts/run_four_method_evaluation.py \
        --method DT-ROBUST-HYBRID \
        --matrix results/step8_final1000_matrix.csv \
        --limit 1000 --max-steps 400 \
        --single-agv-completion \
        --output results/reproduction/single.csv

### Two-AGV Recovery

    PYTHONPATH=. python -u scripts/run_four_method_evaluation.py \
        --method DT-ROBUST-HYBRID \
        --matrix results/step8_final1000_matrix.csv \
        --limit 1000 --max-steps 400 \
        --two-agv-completion \
        --output results/reproduction/two.csv

### Combined Recovery

    PYTHONPATH=. python -u scripts/run_four_method_evaluation.py \
        --method DT-ROBUST-HYBRID \
        --matrix results/step8_final1000_matrix.csv \
        --limit 1000 --max-steps 400 \
        --single-agv-completion \
        --two-agv-completion \
        --output results/reproduction/combined.csv

### Final DICR Hybrid

    PYTHONPATH=. python -u scripts/run_four_method_evaluation.py \
        --method DT-ROBUST-HYBRID \
        --matrix results/step8_final1000_matrix.csv \
        --limit 1000 --max-steps 400 \
        --single-agv-completion \
        --two-agv-completion \
        --dt-inferred-completion \
        --output results/reproduction/dicr.csv

These commands document the intended reproduction
configuration. A complete independent rerun using
this final repository snapshot has not been performed.

## Statistical Analysis

The existing statistical analysis scripts use the
original archived result filenames.

To reproduce the analysis of archived results without
overwriting the preserved files:

    PYTHONPATH=. python - <<'PY'
    from pathlib import Path
    import sys

    sys.path.insert(0, "scripts")

    import analyze_step14_dicr as analysis

    analysis.OUTPUT = Path(
        "results/reproduction/statistics.csv"
    )

    analysis.main()
    PY

Create results/reproduction first if necessary.

To analyze newly reproduced evaluations instead,
replace analysis.FILES with the corresponding
reproduction CSV paths before calling analysis.main().

Statistical settings:

- Route-cluster bootstrap: 10000 repetitions.
- Monte Carlo sign-flip: 100000 repetitions.
- Analysis random seed: 2026.
- Confidence level: 95%.
- Holm multiple comparison correction.

## Reproducibility Limitations

SHA-256 fingerprints verify archived artifact identity.

They do not replace missing checkpoint binaries or
establish independent experimental reproducibility.

Planning times may vary with processor performance,
system load, library versions and solver behavior.

Exact outcome reproduction can also be affected by
time-dependent planning budgets.

The final benchmark includes episodes examined during
algorithm development.

Consequently, independent generalization has not
been established.

Zero observed collisions is not a formal safety proof.
