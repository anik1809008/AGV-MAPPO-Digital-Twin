# Research development journey

Uncertainty-Aware Two-Way Hybrid Digital Twin for Multi-AGV Navigation in Smart Manufacturing

## Evidence and scope

This reconstruction uses the supplied Git history, the 223-file historical CSV extraction, and reported matched analyses in Steps 11–14. The extraction contains aggregate summaries, not raw paired rows or a complete decision log. File names identify candidate experiments but do not establish exact chronological order or controlled equivalence. The appendix preserves the supplied aggregate evidence verbatim; it is not 223 independent experiments. Some files are matrices, repeated runs, partial results or aggregate tables.

## 1. Foundation

MovingAI warehouse maps and scenarios, discrete WAIT/N/S/E/W actions, simulator vertex and edge-swap conflict detection, goal persistence and deadlock monitoring were developed first. The DT stores trusted observations, timestamps, AoI, recorded commands and possible current positions. Telemetry latency and bounded execution delay are modeled separately. MAPPO uses decentralized actors and a centralized critic, with PPO/GAE training. The historical protocol separates training scenarios 1–15, validation 16–20 and test 21–25; the main experiments use eight AGVs and seeds 0–4. The final recovery experiments reuse existing frozen M3 validation checkpoints (20 episodes per scenario); no retraining was needed for DICR. Exact checkpoint fingerprints and training configuration should be checked against the reproducibility manifest.

## 2. M1–M6 historical test results

Success percentages; denominators are 25 perfect, 100 latency-only, 25 execution-only and 100 combined per method.

| Method | Perfect | Latency | Execution | Combined | Combined collisions | Combined deadlocks |
|---|---:|---:|---:|---:|---:|---:|
| M1 classical | 40 | 40 | 40 | 40 | 60 | 0 |
| M2 conventional MAPPO | 48 | 41 | 48 | 41 | 58 | 1 |
| M3 DT-aware MAPPO | 40 | 39 | 44 | 42 | 53 | 5 |
| M4 safety-shielded M3 | 48 | 47 | 44 | 51 | 0 | 15 |
| M5 reachable-set waiting | 32 | 27 | 24 | 17 | 56 | 15 |
| M6 AoI-only MAPPO | 40 | 40 | 32 | 33 | 64 | 2 |

M1's historical execution pathway did not fully represent uncertainty; do not treat it as a controlled robust baseline. M3 does not consistently beat M2; M4 eliminates observed collisions but leaves incomplete episodes. M5 waiting is not an adequate substitute for safety checks. M6 tests whether age alone is sufficient. These observations motivate M7 progress and recovery.

## 3. Parameter and implementation investigations

Early M3 reports show zero successes in the original, corrected, fixed, reward-fixed and path-fixed series, with varying collision counts. Git history records deadlock terminal handling, goal reward only on first arrival, normalized goal direction, obstacle-aware distance (19245c6), route guidance, and command queue corrections. Individual causal effects cannot be isolated from the aggregate extraction.

| M3 horizon | Success / 50 | Collisions | Deadlocks | Mean termination steps | Mean planning seconds |
|---|---:|---:|---:|---:|---:|
| 200 | 21 | 16 | 0 | 134.78 | 5.581882 |
| 300 (route-guided) | 32 | 16 | 0 | 143.72 | 5.629284 |
| 400 | 34 | 16 | 0 | 144.76 | 5.727063 |
| 500 | 34 | 16 | 0 | 144.76 | 5.766161 |

400 was selected (8510c4f): no additional successes at 500. The 300 result is not a clean horizon-only comparison. Candidate training budgets were 5, 10, 20 and 40 episodes per scenario; pooled sweeps M2 631/1000, M4 821/1000, M6 558/1000 do not establish the best budget individually. The exact per-budget selection table is missing from the aggregate evidence.

M5 threshold pilots at 2, 3, 4 and 5 all collided on the same scenario-21 seed-0 case; the validation sweep pooled 10/20 successes and the expanded validation 27/50. Earlier project records identify threshold 4 as the chosen threshold, but per-threshold expanded results require raw CSV grouping. Disclose that the pilot touched a test scenario.

| M4 developmental file variant | Success / 50 | Collisions | Deadlocks |
|---|---:|---:|---:|
| fixed | 4 | 9 | 16 |
| goal lock | 26 | 1 | 1 |
| sampled fixed | 21 | 12 | 1 |
| transition fixed | 21 | 12 | 1 |
| timestamp fixed | 26 | 0 | 1 |
| execution uncertainty fixed | 23 | 0 | 0 |
| relaxed blocked | 47 | 0 | 3 |
| safe movement | 50 | 0 | 0 |

These are developmental variants, not independent test results. The final M4 historical test still had 51/100 success under combined uncertainty, with no collisions and 15 deadlocks.

## 4. M7 and DT-only control

M7 adds stagnation monitoring, BFS action ranking, recovery and detour fallback. Initial validation 247/250 (one deadlock), V2 216/250 (34 deadlocks), final 249/250 (zero collisions and deadlocks). For scenario 19 seed 4 at latency 2 and 4, frontier fixes V1/V2 timed out at 400; V3 completed in 149 and 151 steps. Scenario 20 seed 2 latency 1 repeatedly deadlocked at 163 steps, self-stale fix completed at 154 and final recheck at 144. The final validation is tagged m7-validation-frozen at d4ff300. Later work imposed more restrictive DT-derived observations rather than relying on simulator positions; do not equate the earlier 99.6% with the final DT-only benchmark.

| DT validation variant | Success / 250 |
|---|---:|
| DT-observed M7 | 88 |
| DT-inferred M7 | 170 |
| DT-robust progress M7 | 190 |
| DT-only Hybrid | 168 |
| DT-robust Hybrid | 199 |

All had zero observed collisions and deadlocks. Robust progress requires modeled obstacle-aware distance improvement across possible positions and safety checks. M7 horizon sensitivity on 100 selected episodes: 400/500/700 steps gave 57/58/60 successes and 43/42/40 timeouts; mean wall seconds 33.054/34.841/37.683. Retain 400 for the main protocol.

## 5. Classical and Hybrid development

MDD-SAT foundation, LNS-SAT local repair, goal persistence, spatiotemporal repair zones, reliability-aware supervisor, two-way hysteresis and cooldown were introduced in that order in Git. LNS-SAT expanded pilot: 36/50 successes, 14 planning failures. Validation results (250 each): LNS-SAT 125 successes/15 collisions/110 deadlocks; initial two-way 216/5/28; recovery hybrid 214/5/30; execution-aware and early-switch-only each 248/0/1. These configurations have differing information constraints. Pending-command drain, recovery pause, telemetry confirmation and handover were investigated; early-switch ablation and DT-only observations then led to robust progress.

Historical 1000-episode matrix: 100 perfect, 900 imperfect, 500 deterministic, 500 stochastic; scenarios 21–25.

| Method | Success | Collision | Deadlock | Mean planning seconds |
|---|---:|---:|---:|---:|
| LNS-SAT | 290 | 100 | 190 | 25.261882 |
| DT-only Hybrid | 555 | 0 | 1 | 30.260149 |
| DT-robust M7 | 584 | 0 | 10 | 34.011131 |
| One-way robust Hybrid | 599 | 0 | 10 | 23.659887 |
| Two-way robust Hybrid | 605 | 0 | 10 | 44.505128 |

Two-way return planning was expensive (maximum 1465.849442 s, 503 return replans, 20 successful returns). The historical two-way gain over one-way was six successes. The subsequent final matrix has different routes and condition weights, so 60.5% and 84.7% are not a controlled before/after comparison.

## 6. Return optimization and completion recovery (Steps 3–7)

Return budget experiments investigated expensive SAT replanning; historical project records describe bounded solver calls and conflicts and a soft deadline, not a guaranteed time cap. Step 3 matched episode 22 succeeded at 107 steps with total planning 96.679 s; episode 172 remained incomplete at a 45-step diagnostic cutoff with planning 38.542 s. The short diagnostics are not full-horizon failures. Step 4 six rescues: 6/6, mean planning 93.545 s. Step 5 fast-grid six: 6/6, 66.272 s; final six: 6/6, 63.786 s. Caching and profiling short runs are not evidence of final success. Step 5 failure pilot: 0/6 at 400 steps. Single-AGV episode 172 completed at 227 steps; five-episode pilot 3/5 with two deadlocks, two-case trace rerun reproduced both deadlocks, corrected two-case run 2/2. Four-recovery final 4/4. Step 6 baseline 8/10, recovery-only 8/10, combined and guarded 10/10; guarded two-case run 2/2. Step 7 unseen ten controls 10/10 both; remaining 24: 18 to 23 successes; fresh seed-0 100: 79 to 96 (four deadlocks) to 100 (zero deadlocks) after progress correction. These pilot sets may overlap and must not be added together as independent episodes. Exact selection rules and profiling decisions require original logs.

Single recovery uses BFS with other AGVs treated as stationary obstacles. Two recovery tries sequential orders of the two unfinished agents and selects a feasible shorter plan; it is not a general joint optimal solver. Progress acknowledgement prevents false deadlocks on validated recovery movement. Relevant commits: 034e76a, 85fe674, 4193f66.

## 7. Final matched matrix and ablation (Steps 8–12)

1000 episodes per method; 8 AGVs; scenarios 21–25; five seeds; horizon 400. Conditions: (latency, immediate probability, count) = (0,1,500), (0,.8,100), (1,1,100), (2,1,100), (2,.8,100), (4,.8,100). This is a different matrix from the earlier balanced evaluation.

| Method | Success | Collision | Deadlock | Mean planning seconds |
|---|---:|---:|---:|---:|
| Original | 727 | 0 | 7 | 20.780063 |
| Single | 778 | 0 | 7 | 18.821849 |
| Two | 788 | 0 | 7 | 18.498779 |
| Combined | 808 | 0 | 7 | 17.492627 |
| Combined + DICR | 847 | 0 | 7 | 16.428493 |

Single rescued 51, two 61, overlap 31, union 81; combined preserved all individual rescues. All 81 gains occurred in perfect conditions (412/500 to 493/500). Imperfect conditions remained 315/500. Step 12 route-cluster analysis supports original-to-combined improvement, with scenario-level sensitivity not significant. Commit 444adda preserves matrix and original/combined; e0f381f and 0c3b838 preserve ablation and statistics.

## 8. DICR (Step 13)

Of 192 remaining failures, 185 were timeouts and seven deadlocks. Twelve targeted failure diagnoses matched frozen outcomes. Four selected deterministic delayed episodes provided 12,800 same-timestep position comparisons with no inference errors; this is conditional diagnostic evidence, not a formal guarantee. DICR permits completion planning from inferred singleton positions despite positive AoI. Activation requires AoI 0–2 with at least one positive age, deterministic execution, no pending commands, unique nonoverlapping positions for every agent, one or two unfinished agents, no handover or recovery pause, minimum dwell (3 steps), and a feasible route within the remaining horizon. The route is checked every timestep before movement; rejection triggers M7 fallback and handover. Retry cooldown is 20 steps by default. Ordinary fresh-state return remains unchanged. The simulator progress bookkeeping uses actual movement; do not claim the whole simulator is ground-truth-free.

DICR reuses existing planners, not a new pathfinding algorithm. Component and integration tests 69/69; full suite 432/432. Pilot 12/20 to 16/20, four rescues, no regressions or collisions/deadlocks; two successful cases slower. Flag-off replay matched 20/20 across 14 non-timing fields. Full benchmark: 847/1000, 39 rescues, zero regressions, zero observed collisions, seven deadlocks. All 39 gains occurred in deterministic delayed conditions.

| Condition | Combined | DICR |
|---|---:|---:|
| Perfect | 493/500 | 493/500 |
| Latency 1, deterministic | 79/100 | 100/100 |
| Latency 2, deterministic | 82/100 | 100/100 |
| Latency 0, stochastic | 71/100 | 71/100 |
| Latency 2, stochastic | 51/100 | 51/100 |
| Latency 4, stochastic | 32/100 | 32/100 |

Commits: 0767626 diagnostics; 82e57d0 eligibility; 19d9439 guard; 23ec1ac supervisor; 56e8e3b planner; f975eeb integration; 9f7818c pilot; c42ca3f flag-off; f0d115b final evaluation.

## 9. Statistical validation (Step 14)

| Comparison | Effect (pp) | Individual 95% route CI | Holm-adjusted route p |
|---|---:|---:|---:|
| Original to DICR | 12 | [8,16.3] | 0.0000399996 |
| Combined to DICR | 3.9 | [1.6,6.5] | 0.0014648438 |
| Imperfect | 7.8 | [3.6,12.8] | 0.0014648438 |
| Deterministic delayed | 19.5 | [9.5,31] | 0.0014648438 |

Route clusters defined by scenario and start index, 10,000 bootstrap replicates, exact sign flips up to 16 nonzero clusters otherwise 100,000 Monte Carlo permutations, seed 2026. Scenario sensitivity: raw p=.0625, adjusted=.25 for each comparison; only five scenario clusters. Route inference assumes appropriate cluster independence and sign symmetry. Comparisons overlap; intervals are not simultaneous. Development reused some benchmark cases, so analysis is exploratory. Commit 367880e.

Among 808 common successes, makespan 170.8936 to 173.8106 (+1.71%; 3 faster, 718 unchanged, 87 slower), path length 668.7141 to 668.4295, planning seconds 13.8733 to 13.3989. All-episode mean planning seconds 17.4926 to 16.4285, median .2863 to .2934, P95 70.8954 to 68.5783, P99 81.8040 to 78.2589, maximum 103.8903 to 85.1578. Timing comparisons descriptive, not statistically validated.

## 10. Conclusion and open evidence gaps

The selected architecture integrates LNS-SAT, DT-robust M7, execution-aware two-way switching and handover, guarded one-/two-agent completion and DICR. It achieved 84.7% on the final matched benchmark, +12 pp vs original and +3.9 pp vs combined recovery. It extends completion recovery under bounded telemetry delay with deterministic execution. Stochastic execution remains 154/300 (51.33%). No formal safety guarantee, independent generalization, physical deployment, worldwide novelty or improvement under arbitrary uncertainty is established. The final benchmark is not the earlier balanced matrix. The report does not reconstruct unavailable per-budget training statistics, exact chronological order of every uncommitted pilot, or every historical configuration value. Preserve original logs and raw CSVs for those questions. No additional navigation experiment is required for this documentation step.

## Appendix

The accompanying `research_history_extraction.txt` contains the supplied summaries of all 223 CSV files, including configuration counts, observed metrics and small aggregate tables. Consult the original CSVs for row-level comparisons, grouping and reproducibility. Do not treat aggregate mixed-method pilot counts as per-method results.
