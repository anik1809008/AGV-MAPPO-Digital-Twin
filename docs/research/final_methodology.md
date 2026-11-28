# Final Research Methodology

## 1. Research Objective

Develop an uncertainty-aware two-way Hybrid Digital Twin
architecture for reliable multi-AGV navigation under
communication latency and uncertain command execution.

The architecture combines classical multi-agent planning,
multi-agent reinforcement learning, Digital Twin state
inference, safety mechanisms and adaptive recovery.

## 2. Problem Formulation

The warehouse is represented as a discrete occupancy grid.

Each AGV i has:

- Current physical position x_i(t).
- Initial position s_i.
- Assigned goal position g_i.
- Discrete action a_i(t).

The action space consists of:

- WAIT
- NORTH
- SOUTH
- EAST
- WEST

The navigation objective is to move every AGV to its
assigned goal while avoiding static obstacles, vertex
conflicts, edge-swap conflicts and persistent stagnation.

An episode succeeds when all AGVs reach their goals
within the maximum episode horizon.

The final evaluation uses a horizon of 400 timesteps.

## 3. Ground-Truth Simulator

Implementation:

- src/environment/simulator.py
- src/environment/actions.py
- src/environment/pathfinding.py

The simulator maintains the actual AGV positions and
evaluates executed joint actions.

It detects:

- Static obstacle and boundary violations.
- Vertex conflicts.
- Edge-swap conflicts.
- Goal completion.
- Deadlock conditions.

Completed AGVs remain at their goals.

Obstacle-aware navigation distance is used in progress
evaluation rather than relying exclusively on geometric
distance to the destination.

The simulator represents the physical system in this
simulation-based research. It is not a physical AGV fleet.

## 4. Communication Uncertainty

Implementation:

- src/communication/telemetry.py
- src/communication/latency_channel.py
- src/communication/scheduled_latency_channel.py

Telemetry messages contain AGV position information and
associated source timestamps.

For AGV i, Age of Information is:

AoI_i(t) = t - tau_i

where tau_i is the timestamp of the latest trusted
telemetry observation available to the Digital Twin.

A positive AoI indicates that the most recent trusted
telemetry observation is older than the current timestep.

The final experimental matrix includes communication
latencies of 0, 1, 2 and 4 timesteps.

Earlier experimental matrices also included latency 3.

The communication model must not be described as
validated physical wireless networking or packet-loss
robustness without additional supporting experiments.

## 5. Execution Uncertainty

Implementation:

- src/environment/execution_delay.py
- src/environment/delayed_execution.py

The immediate command execution probability is p.

The evaluated probabilities are:

- p = 1.0: deterministic immediate execution.
- p = 0.8: uncertain command execution.

The bounded execution-delay model allows commands to
be delayed rather than immediately executed.

Pending commands are important because an AGV may
execute a previously issued action after its controller
has changed.

The Hybrid supervisor and handover mechanism therefore
consider unresolved commands during switching.

This is a modeled execution-delay process, not a
general-purpose industrial actuator fault model.

## 6. Digital Twin State Representation

Implementation:

- src/digital_twin/state.py
- src/digital_twin/digital_twin.py
- src/digital_twin/reachability.py
- src/marl/digital_twin_adapter.py

The Digital Twin maintains:

- Latest trusted AGV position.
- Trusted telemetry timestamp.
- Timestamped command history.
- Possible current positions.
- Reachable occupancy.
- Possible movement transitions.
- Age of Information.

Let X_i(t) represent the set of possible current
positions for AGV i.

The Digital Twin reconstructs X_i(t) by propagating the
latest trusted position through the recorded command
history under the modeled execution uncertainty.

Under deterministic execution, valid command replay
can produce a singleton position set.

Under stochastic execution, multiple candidate
positions can remain possible.

A singleton candidate set indicates a unique position
under the implemented model. It is not independently
guaranteed to equal the physical position under
unmodeled disturbances or incorrect command history.

## 7. MAPPO Navigation

Implementation:

- src/marl/networks.py
- src/marl/observations.py
- src/marl/observation_builder.py
- src/marl/centralized_state.py
- src/marl/mappo_update.py
- src/marl/mappo_trainer.py
- src/marl/multi_agent_training.py

Multi-Agent Proximal Policy Optimization (MAPPO) uses
centralized training and decentralized execution.

The actor selects actions using local observations.

The centralized critic estimates state values during
training.

The Digital Twin-aware observations incorporate
navigation information and reliability information.

The training framework includes:

- Multi-agent rollout collection.
- Generalized Advantage Estimation.
- PPO policy optimization.
- Value-function learning.
- Multi-epoch minibatch updates.
- Checkpoint saving and loading.

The conceptual PPO clipped objective is:

L_clip = E[min(r*A, clip(r, 1-epsilon, 1+epsilon)*A)]

where r is the policy probability ratio and A is
the estimated advantage.

This equation describes the standard PPO formulation.
Exact network dimensions and optimizer hyperparameters
should be obtained from the implementation before
publication.

The final Hybrid uses the existing frozen M3 checkpoints
through the M7 navigation controller.

No additional MAPPO training was required for the
completion recovery mechanisms or DICR.

## 8. M1-M7 Research Framework

M1:
Classical navigation baseline.

M2:
Conventional MAPPO without the full DT uncertainty
representation.

M3:
Digital Twin uncertainty-aware MAPPO.

M4:
M3 with uncertainty-aware safety shielding.

M5:
Reachable-set threshold-based waiting baseline.

M6:
AoI-only observation ablation.

M7:
Safety-aware navigation with progress monitoring,
stagnation detection and local recovery.

The complete developmental results are documented
separately in research_journey.md.

The historical M1 baseline requires qualification
because its execution pathway does not fully represent
all modeled uncertainty conditions.

## 9. Uncertainty-Aware Safety Shield

Implementation:

- src/safety/shield.py
- src/safety/action_filter.py
- src/safety/m4_controller.py

The safety shield checks candidate navigation actions
against known static obstacles and modeled AGV occupancy
and transition conflicts.

It considers uncertainty in the positions and movements
of other AGVs.

Unsafe policy actions can be replaced by alternative
safe actions or WAIT.

The shield aims to reduce collisions under the modeled
uncertainty assumptions.

Zero observed collisions in an experiment must not be
interpreted as a formal universal safety guarantee.

## 10. M7 Progress and Local Recovery

Implementation:

- src/safety/m7_controller.py
- src/safety/m7_action_filter.py
- src/safety/progress_monitor.py
- src/safety/local_recovery.py
- src/safety/robust_progress.py

M7 addresses the limitation that collision avoidance
alone does not guarantee successful task completion.

It combines:

- MAPPO action preferences.
- Safety-aware action selection.
- Obstacle-aware progress monitoring.
- Stagnation detection.
- Local path recovery.
- Detour fallback.

Robust progress evaluates candidate actions against
possible AGV positions represented by the Digital Twin.

A robust progress action should reduce obstacle-aware
goal distance across the relevant possible positions,
subject to the implemented safety checks.

This conservative condition can restrict useful detours
when immediate progress is unavailable.

The DT-derived observation and robust progress variants
were investigated to reduce dependence on direct
simulator-state navigation information.

## 11. Classical Planning

Implementation:

- src/mapf/mdd_sat.py
- src/mapf/lns_sat.py

MDD-SAT provides the classical multi-agent planning
foundation using a satisfiability-based formulation.

LNS-SAT applies local repair to multi-agent paths,
combining large-neighborhood search with SAT-based
repair.

The classical controller is used when reliable state
information supports coordinated route execution.

Planning can fail or exceed the adopted computational
budget. The Hybrid architecture therefore requires
adaptive navigation as a fallback.

The bounded LNS-SAT implementation is not claimed to be
a complete or universally optimal multi-agent solver.

## 12. Two-Way Hybrid Supervisor

Implementation:

- src/hybrid/supervisor.py
- src/hybrid/two_way_supervisor.py
- src/hybrid/execution_aware_supervisor.py
- src/hybrid/two_way_episode_runner.py

The two main controller modes are:

- LNS-SAT classical mode.
- M7 adaptive navigation mode.

The episode begins with classical planning.

Classical-to-adaptive switching can occur because of:

- Classical planning failure.
- Excessive Age of Information.
- Stale occupancy conflicts.
- Pending execution uncertainty.

The normal return to classical mode requires reliable
Digital Twin information, appropriate supervisor state,
resolved execution uncertainty and feasible replanning.

Default supervisor parameters:

- AoI threshold: 2.
- Recovery reliability steps: 3.
- Minimum dwell steps: 3.

The supervisor uses hysteresis and cooldown mechanisms
to limit repeated or premature controller switching.

DICR extends this behavior through an additional guarded
completion-specific transition.

## 13. Execution-Aware Handover

Implementation:

- src/hybrid/execution_aware_handover.py
- src/hybrid/recovery_drain.py

Previously issued commands can remain pending when
the active navigation controller changes.

The handover mechanism manages pending execution before
normal command generation resumes.

This reduces the risk that commands from different
controller modes interfere with each other.

Recovery fallback preserves this execution-aware
handover behavior.

## 14. Single-AGV Completion Recovery

Implementation:

- src/hybrid/single_agv_completion.py
- src/hybrid/single_agv_progress.py

Activation is restricted to situations with exactly
one unfinished AGV and suitable reliable state information.

The planner uses obstacle-aware breadth-first search.

Other AGVs are treated as stationary occupied cells.

The unfinished AGV follows the feasible completion
route while the remaining AGVs wait.

Recovery progress is verified against the expected route.

The ordinary recovery path requires fresh DT state
and deterministic immediate execution.

## 15. Sequential Two-AGV Completion Recovery

Implementation:

- src/hybrid/two_agv_completion.py
- src/hybrid/two_agv_progress.py

When exactly two AGVs remain unfinished, the planner
evaluates both possible sequential completion orders.

The selected feasible order permits one AGV to move
while the remaining AGVs wait.

Once the first AGV reaches its destination, the
second AGV completes its route.

The planner prefers the shorter feasible sequential
completion plan.

The method does not perform unrestricted joint
two-agent optimal path planning.

Progress verification accounts for valid planned
movements that may not immediately reduce distance
to the final goal.

The simulation progress-acknowledgment mechanism uses
executed simulator outcomes to prevent false deadlock
classification during verified recovery.

This simulator-based verification must be distinguished
from DT-only navigation-state inference.

## 16. DT-Inferred Completion Recovery (DICR)

Implementation:

- src/hybrid/dt_inferred_completion.py
- src/hybrid/dt_inferred_completion_supervisor.py
- src/hybrid/two_way_episode_runner.py

DICR extends completion recovery to bounded delayed
telemetry by using unique inferred Digital Twin positions.

It does not replace the existing completion planners.

### 16.1 Eligibility

Let N denote the number of AGVs.

Let X_i(t) denote the possible current position set
for agent i.

Let U denote the set of unfinished AGVs.

DICR requires:

1. Immediate execution probability equals 1.0.
2. Every agent has a corresponding DT position set.
3. Every agent has a corresponding AoI value.
4. Every AoI is between 0 and 2 inclusive.
5. At least one AoI is positive at activation.
6. Every possible position set has cardinality one.
7. All inferred positions are distinct.
8. Exactly one or two AGVs remain unfinished.
9. No pending commands are present.
10. No active execution handover or recovery pause.
11. The supervisor is in M7 mode.
12. Minimum supervisor dwell time is satisfied.
13. A feasible completion route exists within the
    remaining episode horizon.

The eligibility function returns inferred positions
when its specific state checks pass, or None otherwise.

Additional supervisor and runner conditions are checked
outside that eligibility function.

### 16.2 Mathematical Eligibility Representation

For each agent i:

0 <= AoI_i(t) <= 2

|X_i(t)| = 1

At least one agent must satisfy:

AoI_i(t) > 0

For any distinct agents i and j:

x_hat_i(t) != x_hat_j(t)

The unfinished agent set is:

U = {i : x_hat_i(t) != g_i}

with:

|U| in {1, 2}

The recovery horizon H must satisfy:

0 < H <= T_max - t

where T_max = 400 in the final experiment.

These conditions are necessary parts of the implemented
activation process, together with supervisor state,
execution status and route feasibility.

### 16.3 Completion Plan Construction

If exactly one AGV remains unfinished, DICR calls
the existing Single-AGV completion planner.

If exactly two AGVs remain unfinished, DICR calls
the existing sequential Two-AGV completion planner.

The planner constructs expected positions for each
AGV throughout the planned recovery horizon.

The generated plan is rejected if:

- Duplicate occupancy occurs.
- Other than exactly one AGV moves per planned step.
- A position lies outside the grid.
- A position intersects a static obstacle.
- A completed AGV leaves its goal.
- The final expected positions do not match all goals.
- The route exceeds the remaining episode horizon.

### 16.4 Active Route Guard

At every active recovery timestep:

- Execution probability must remain 1.0.
- Pending commands must be absent.
- Handover must not be active.
- Expected route positions must be available.
- The route index must be valid.
- Every AGV must have a singleton inferred position.
- Every AoI must remain within the range 0-2.
- Inferred positions must match expected positions.
- Inferred positions must remain distinct.

Unlike initial activation, active route validation
allows all telemetry observations to become fresh.

### 16.5 Supervisor Integration

The DICR supervisor inherits the existing execution-aware
two-way supervisor.

A valid DICR activation changes the controller mode
from M7 to LNS-SAT and sets:

inferred_completion_active = True

The recorded switch reason is:

dt_inferred_completion

The LNS-SAT mode designation here represents execution
of the specialized classical completion route, rather
than necessarily invoking a new general LNS-SAT search.

If active DICR validation fails, the switch reason is:

dt_inferred_route_invalid

The supervisor returns to M7 and clears the inferred
completion activation flag.

The existing fresh-state return mechanism is preserved.

DICR does not require three consecutive eligible
observations. It applies separate eligibility checks
and the minimum supervisor dwell requirement.

### 16.6 Scope

DICR is a bounded completion-specific recovery mechanism.

It does not claim to:

- Solve arbitrary uncertain multi-agent planning.
- Recover every stochastic execution failure.
- Guarantee correctness under unmodeled disturbances.
- Provide a new general-purpose pathfinding algorithm.
- Establish formal collision-free operation.

## 17. Conceptual Final Algorithm

The following pseudocode summarizes the architecture.
It is not a line-by-line replacement for the runner.

Initialize simulator and Digital Twin.
Load frozen MAPPO checkpoint.
Initialize execution-aware supervisor.
Attempt initial classical planning.

For each timestep:

    Deliver available telemetry.
    Update Digital Twin and command history.
    Compute AoI and possible current positions.

    If DICR is active:
        Validate the expected recovery route.
        If invalid:
            Fall back to M7 using safe handover.

    Evaluate supervisor switching conditions.

    If M7 mode and normal return conditions hold:
        Attempt guarded classical completion or replanning.

    If M7 mode and DICR eligibility conditions hold:
        Construct a bounded inferred completion plan.
        Validate the plan.
        Activate DICR if all conditions pass.

    If execution handover is active:
        Manage pending commands.
    Else if classical mode is active:
        Generate commands from the active plan.
    Else:
        Generate M7 navigation commands.

    Execute commands through the modeled executor.
    Update simulator state.
    Record issued commands in the Digital Twin.
    Update evaluation metrics.

    Stop on goal completion, collision, deadlock,
    or maximum episode horizon.

Return episode results.

The actual control ordering and additional cooldown
conditions are defined in two_way_episode_runner.py.

## 18. Experimental Protocol

Benchmark:
MovingAI warehouse-10-20-10-2-1.

Scenario split:

- Training: scenarios 1-15.
- Validation: scenarios 16-20.
- Testing: scenarios 21-25.

Final recovery evaluation:

- Eight AGVs.
- Five pretrained model seeds: 0-4.
- 200 distinct start/goal configurations.
- 1000 matched episodes per method.
- Maximum episode horizon: 400.

Final uncertainty distribution:

| Latency | Immediate probability | Episodes |
|---:|---:|---:|
| 0 | 1.0 | 500 |
| 0 | 0.8 | 100 |
| 1 | 1.0 | 100 |
| 2 | 1.0 | 100 |
| 2 | 0.8 | 100 |
| 4 | 0.8 | 100 |

The historical 1000-episode benchmark uses a different
experimental matrix and must be reported separately.

## 19. Evaluation Metrics

Success rate:

Successful episodes / Total episodes.

Collision rate:

Episodes with recorded collisions / Total episodes.

Deadlock rate:

Episodes with recorded deadlocks / Total episodes.

Makespan:

Number of executed timesteps until episode termination,
reported with the appropriate success or failure context.

Path length:

Total AGV movement distance accumulated in an episode.

Planning time:

Measured planning computation time according to the
corresponding evaluation runner.

Historical timing definitions should be checked before
comparing measurements across different runner versions.

Additional diagnostic metrics include:

- Supervisor switch count.
- Return replanning attempts.
- Successful classical returns.
- Planning failures.
- Recovery attempts.
- Recovery pause duration.

## 20. Statistical Methodology

The final analysis compares matched episode outcomes.

Statistical methods:

- Exact McNemar test.
- Route-cluster bootstrap confidence intervals.
- Route-cluster sign-flip tests.
- Scenario-cluster sensitivity analysis.
- Holm multiple comparison correction.

Statistical settings:

- Bootstrap repetitions: 10000.
- Monte Carlo permutations: 100000.
- Analysis random seed: 2026.
- Confidence level: 95%.
- Significance threshold: 0.05.

Route clusters are defined using scenario ID and
start index.

Scenario clustering evaluates sensitivity to dependence
within each scenario file.

The scenario-level test has limited resolution because
only five scenario clusters are available.

The adopted route-cluster methodology assumes adequate
independence between route clusters.

Confidence intervals are individual, not simultaneous.

Efficiency comparisons are descriptive.

## 21. Final Results

| Configuration | Success | Collisions | Deadlocks |
|---|---:|---:|---:|
| Original Hybrid | 72.7% | 0 | 7 |
| Single-AGV recovery | 77.8% | 0 | 7 |
| Two-AGV recovery | 78.8% | 0 | 7 |
| Combined recovery | 80.8% | 0 | 7 |
| Combined recovery + DICR | 84.7% | 0 | 7 |

DICR rescued 39 additional episodes relative to the
combined recovery architecture, with zero success
regressions.

Deterministic delayed communication:

161/200 to 200/200 successful episodes.

The primary DICR improvement was 3.9 percentage points.

Route-cluster 95% confidence interval:

[1.6, 6.5] percentage points.

Holm-adjusted route-cluster p-value:

0.0014648438.

Scenario-level statistical significance was not established.

## 22. Software Environment

Recorded dependencies:

- numpy==2.5.2
- pytest==9.1.1
- torch==2.14.0
- python-sat==1.9.dev15

The historical reproducibility manifest records
Python 3.12.3.

The final runtime environment and checkpoint fingerprints
must be verified separately during research freeze.

Historical checkpoint binaries are not included in Git.

## 23. Assumptions and Limitations

- Simulation-based warehouse navigation.
- One warehouse map and eight AGVs in the main benchmark.
- Five test scenario files.
- Modeled bounded telemetry and execution delays.
- DICR requires deterministic immediate execution.
- No DICR success improvement under stochastic execution.
- Some successful episodes experience longer makespan.
- Simulator-based recovery progress verification.
- No formal universal safety proof.
- No established independent generalization.
- Some benchmark episodes informed algorithm development.
- No physical industrial deployment validation.
- Scientific novelty requires comprehensive literature
  verification.

## 24. Supporting Research Documents

- docs/research/research_journey.md
- docs/research/final_research_summary.md
- docs/research/step8_final1000_results.md
- results/research_history_extraction.txt
- results/step12_ablation_analysis.txt
- results/step13_dt_accuracy.txt
- results/step13_failure_diagnostics.txt
- results/step14_dicr_analysis.txt
- results/step14_dicr_statistics.csv

The final architecture is supported by the measured
experimental findings within the stated assumptions
and evaluation conditions.
