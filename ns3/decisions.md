# ns-3 transfer experiment: decisions

All decisions below were taken before any dev, eval, placebo or knockout outcome was seen. The
pilot (seed 999) and fake-data runs were used only to check that the pipeline runs.

## Scope

| # | decision | why |
|---|---|---|
| D1 | Interpreted "this experiment" as the system-simulator discovery study (randomised arbiter logs -> PMRT -> declarations vs knockout ground truth), transferred to ns-3. | It is the paper's central empirical claim, and the discussion lists "one simulated scenario, no real RAN" as a limitation. |
| D2 | Transfer only the validity and discovery parts (K0 placebo, declarations, ground-truth scoring), not the mitigation/referee study or the baseline suite. | Mitigation needs the full E6-P plant; validity is the property the paper says should transfer. |
| D3 | Use the repo's PMRT code unchanged (`fit_pmrt`, `run_pmrt`, `edges_pmrt`) through a hand-built `UnitTable`/`PmrtData`. | Tests the method as published, not a re-implementation. |

## Plant (ns-3 scenario)

| # | decision | why |
|---|---|---|
| D4 | ns-3.48 + 5G-LENA, 3x3 grid, ISD 300 m, UMa at 3.5 GHz, numerology 0, isotropic 1x1 antennas, PF scheduler. | A 3x3 grid gives every cell a non-empty `far` set (negative control); the 1x1 antennas and numerology 0 keep cost down. |
| D5 | One knob family, `ptx` (per-cell Tx-power offset, -9 to +3 dB, 3 dB step, base 43 dBm = 20 W). | Same knob and range as E6-P; it can be changed at runtime via `NrGnbPhy::SetTxPower`. Carrier/pico sleep is not used because 5G-LENA has no handover (UEs of a sleeping cell would just drop). |
| D6 | No handover; UEs attach to the strongest cell at nominal power and stay there. | 5G-LENA limitation. A power change acts only through SINR and interference, not border moves. Declared deviation from E6-P. |
| D7 | Two xApps: Coverage (raise +3 dB if the share of violated UE-seconds > 0.05) takes priority over PowerES (lower 3 dB if PRB util < 0.30). | Mirrors E6-P's Coverage vs PowerES same-knob conflict and its thresholds. A fixed priority keeps one request per cell per period. |
| D8 | Arbiter pi0: accept 0.5 / half 0.2 / reject 0.3, with propensities logged per unit; design RNG separate from the ns-3 RNG and drawn for every opened unit, also in warm-up, placebo and knockout runs. | Same pi0 as `PI0_HIGH_NO_RB`. The separate stream keeps mode draws aligned across arms of one seed. |
| D9 | Decision period 20 s per cell, staggered phases; warm-up 30 s with all accepted and no units logged. | Departs from the in-house T = 60 s / 120 s warm-up to get enough units from short episodes. |
| D10 | Episodes 400 s (in-house 720 s). | Runtime: ~2.3 h per 400 s episode under full load. 400 s still allows units with t0 in [90, 310] and full 150 s post bins for training. |
| D11 | Traffic: UDP downlink, mean 1.5 Mb/s per UE (per-UE scale 0.5 + 0.5 Exp(1)), multiplied by a per-cell log-OU load process (tau 30 s, sd 0.5). | Gives load variation that drives xApp requests (confounding the logging design must handle). Measured load ~0.5 PRB util. |
| D12 | KPIs per cell per second: `load` = PRB utilisation, `thr` (Mb/s), `viol` (UEs with delivered < 0.9 x offered), `e` (EARTH macro: 130 + 4.7 x util x 20 W x 10^(o/10)), `delay` (ms). 5 KPIs x {own, nbr, far} = 15 hypotheses. | PMRT requires a `load` KPI and uses it for receiver ranking. UE count is constant without handover, so PRB util stands in. Energy uses the SPEC EARTH formula. |
| D13 | Exposure set = cells within 1.01 x ISD (4-neighbourhood); far = everything else. | A fixed geometric rule chosen before outcomes. Diagonal cells (424 m) count as far, so far is "weaker coupling", not zero coupling. The ground truth decides whether far edges exist. |

## Speed

| # | decision | why |
|---|---|---|
| D14 | Separate optimized, NR-only build in `~/projects/ns-3-opt`; existing `~/projects/ns-3-dev` build untouched. | The default profile was >100x slower than real time. A separate tree avoids breaking the other project's binaries. |
| D15 | TwoRay (FTR) fading instead of ThreeGpp; shadowing off; SRS disabled; 5 MHz carrier; 3 UEs per cell. | The profile showed time spent in MIMO interference and SRS. These settings cut cost about 10x. 3 UEs per cell makes `viol` coarse (steps of 1/3). |
| D16 | 11 parallel jobs on 6 cores / 12 threads. | Throughput over per-run latency; leaves one thread for the machine. |

## Stages and analysis

| # | decision | why |
|---|---|---|
| D17 | Seeds: dev 1-12, eval 101-124, placebo 201-212; knockouts branch from placebo seeds. | Disjoint train/eval as PMRT requires. Fewer episodes than in-house (20/60/20) because of runtime. |
| D18 | Placebo runs draw and log modes but apply accept; K0 = binomial tail of the p <= 0.05 count over m tested hypotheses at level 0.01, plus no BY declarations. | Same K0 rule as `DISCOVERY.md` section 11.3. |
| D19 | Ground truth: CRN knockout, the reject arm re-simulated to t0+91 with all other units accepted; the accept arm is the placebo run of the same seed. y = sgn x (accept - reject) summed over the relation's cells over 90 s. TRUE/NULL/INDET by the episode-cluster bootstrap 95 % CI and delta_k = 5 % of max over relations of abs(mean), >= 10 units and >= 3 seeds. | The placebo run is already the all-accept trajectory on the same random numbers, so one simulation per knockout is saved. Labelling rule follows `DISCOVERY.md` section 6 (single draw, no k reseeds). Determinism is to be checked once. |
| D20 | Primary statistic: PMRT arm `best` with BY at q = 0.05 over the 15 hypotheses; all arms' p <= 0.05 counts are also reported. | `best` is the default arm of `edges_pmrt`; fixed before outcomes. |
| D21 | PMRT config default `PmrtConfig(B=9999)`; context features: cur, 10 s pre-mean util and violations, is-Coverage flag; `hist` built as in the in-house code. | Default B reaches the BY threshold. Context columns are obs-only and pre-assignment. |
| D22 | Naive pooled correlation (applied level x sgn vs y) reported as a design-blind descriptive baseline only. | The full baseline suite needs the in-house records format. |

## Environment

| # | decision | why |
|---|---|---|
| D23 | Installed `uv` with the official installer and synced the workspace (`--extra cpu`). | AGENTS.md mandates uv. It was not installed and there was no venv. |
| D24 | Experiment code lives in `ns3/` at the repo root on branch `ns3-transfer`, not in a workspace package; outputs in `~/oran-ns3-runs/`, not in git. | Keeps the frozen packages and the layering untouched; records are never in git per the README. |
| D25 | Ground truth uses 3 knockout units per placebo seed (36 in total), chosen uniformly among units with t0 in [90, 310] by RNG `[seed, 6613]`; one determinism check (accept-forced run vs placebo series). | Each knockout costs ~1-1.5 h of a core. 36 units over 12 seeds meets the >= 10 units, >= 3 seeds labelling rule. Fixed before any outcome was seen. |
| D26 | At most 8 concurrent ns-3 processes (user instruction, replaces D16's 11). Surplus runs are paused, not killed, by `ns3/cap.sh` (`oran-ns3-cap.service`). | Killing would discard 1-2 h of progress per run; SIGSTOP/SIGCONT keeps it. |
| D27 | Cap raised to 10 concurrent runs; lid-close suspend blocked for 16 h by `ns3-lid-inhibit.service` (user instruction). | User wants the runs to continue with the lid closed. |
| D28 | Governor runs the CAP oldest processes (FIFO) instead of pausing only the youngest surplus. | The old rule starved early-paused runs once xargs kept the count at the cap. The order of execution does not change any run's output. |
| D29 | Keep the running design despite RLF re-establishment (UE KPIs stay credited to the cell of initial attachment). Measure its frequency with 3 diagnostic reruns and report it as a caveat; if it is common, rerun with serving-cell attribution. | PMRT, the placebo check and the knockout ground truth all read the same per-cell series, so validity and the scoring stay internally consistent; only the meaning of "own cell" for thr/viol/delay is blurred. A full rerun costs ~30 h. Taken after seeing placebo 202's util/thr series but before any PMRT result. |
