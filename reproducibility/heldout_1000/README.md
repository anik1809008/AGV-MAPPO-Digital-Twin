# Held-out Multi-AGV Evaluation

Four primary methods evaluated on 1,000 matched episodes:

- LNS-SAT (open-loop baseline)
- DT-Robust MAPPO M7
- One-Way Robust Hybrid
- Two-Way Robust Hybrid

An additional DT-Only Hybrid ablation is included.

Scenarios: 21–25; eight AGVs; 400-step maximum.
Communication latency: 0–4 steps.
Immediate command execution probability: 0.8 or 1.0.

The checkpoint SHA-256 hashes are recorded in manifest.json.
Checkpoint binaries are not included in this package.

LNS-SAT is open-loop and does not receive telemetry updates.
Results are from one MovingAI warehouse layout, not multiple maps.
