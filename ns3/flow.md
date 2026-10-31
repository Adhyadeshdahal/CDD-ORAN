# ns-3 transfer experiment: work log

Goal: run the paper's randomised-arbiter logging and PMRT discovery test on an external simulator
(ns-3.48 + 5G-LENA) instead of the in-house system simulator. Branch `ns3-transfer`; nothing
committed yet.

## 2026-10-30

| time | step |
|---|---|
| ~19:45 | Surveyed the repo (README, `protocols/system/SPEC.md`, `DISCOVERY.md`, paper discussion and limitations). Found an existing ns-3.48 + 5G-LENA (nr) checkout at `~/projects/ns-3-dev` with a prior scenario (`scratch/oran-kpm-scenario.cc`, symlinked from `oran-llm-ran-control`). |
| ~19:50 | Delegated a read-only survey of the PMRT unit-table API (`core/data.py` `UnitTable`, `core/pmrt/units.py`, `methods/system/pmrt.py`) to an Explore agent. Result: PMRT needs a `PmrtData` with 4 post/pre bins at 30/60/90/150 s, `kpis` containing `load`, relations `own`/`nbr`/`far`, 4-column `probs` in `MODES` order; entry points `fit_pmrt` (train) then `run_pmrt` + `edges_pmrt` (BY). |
| ~20:00 | Created branch `ns3-transfer`, wrote `ns3/oran-pmrt-scenario.cc`, symlinked it into `~/projects/ns-3-dev/scratch/`, built it (default profile). |
| ~20:05 | Benchmark 1: 30 s episode on the default (debug-like) build did not finish in 10 min. Stopped it. |
| ~20:10 | Copied ns-3 to `~/projects/ns-3-opt` (no build dir, no `.git`), configured `--build-profile=optimized --enable-modules=nr --disable-examples --disable-tests`, built the scenario there. `~/projects/ns-3-dev` left untouched except for the scratch symlink. |
| ~20:20 | Benchmark 2 (optimized, ThreeGpp fading, 20 MHz, 6 UE/cell): 3 s simulated took ~4 min (~80x slower than real time). |
| ~20:25 | perf and gdb attach blocked (`perf_event_paranoid=4`, `ptrace_scope=1`). Added `--channelModel` (default `TwoRay`): 3 s took 137 s. |
| ~20:30 | Stack sampling by launching under gdb: time is in 5G-LENA's MIMO interference path (`NrInterference::AddSignalMimo`, `NrCovMat`, `CreateSpectrumChannelMatrix`) and UL SRS. |
| ~20:35 | Added `--bandwidth`; benchmarked 5 MHz / 3-4 UE per cell, 10 MHz / 4 UE. Disabled SRS (`EnableSrsInUlSlots/FSlots=false`). Best: 5 MHz, 3 UE/cell, ~7 s wall per simulated second when running alone. |
| ~20:40 | Wrote `ns3/run_stage.sh` (parallel runner, skips finished episodes) and `ns3/analyze.py` (ns-3 logs -> `UnitTable`/`PmrtData` -> `fit_pmrt` on dev -> `run_pmrt` on eval and placebo -> BY via `edges_pmrt(.., "best")`; knockout ground truth; scoring; naive correlation). |
| ~20:42 | Launched the main queue: `run_stage.sh ~/oran-ns3-runs 11` with dev 1-12, placebo 201-212, eval 101-124 (400 s episodes). Launched a 120 s pilot (seed 999) alongside. |
| ~20:45 | `uv` was not installed and the workspace had no venv. Installed uv with the official installer (`~/.local/bin`), ran `uv sync --all-packages --extra cpu`. |
| ~20:50 | Tested `analyze.py` on fake logs in the ns-3 output format (planted own-cell ptx effects): own load/thr/e declared with the right sign, nothing declared on placebo. Pipeline works end to end (~22 s at B=499). |
| ~20:52 | Patched `analyze.py`: ground truth uses the placebo run as the accept arm; added the K0 binomial tail per arm. Ran `ruff format`; `ruff check` and `ty` still report issues (docstrings, `itertuples` typing) to fix later. |
| ~21:30 | Pilot finished (~45 min for 120 s under full load, ~20 s wall per simulated second). Output sane: 14 logged units in 90 s, all three modes drawn and applied; energy, delay and violations respond to ptx. |
| 22:24 | Check: 11 dev episodes (1-11) still running, none finished (expected ~2.3 h each, so ~22:50-23:10). |

## Pending

1. Wait for dev, placebo and eval episodes (~10-13 h total at 11 jobs on 6 cores / 12 threads).
2. Once placebo seeds finish: pick units from each placebo run and launch knockout reject runs
   (`gt:<seed>:<cell>:<t0>:reject`), simulated to t0+91 s.
3. Verify common-random-number determinism once: an accept-forced knockout run must reproduce the
   placebo series up to t0+90.
4. Run `uv run python ns3/analyze.py ~/oran-ns3-runs OUT.json 9999` and report K0, declarations,
   ground truth and scores.
5. Clean up lint (`ruff check`, `ty`) before any commit.

## Files

| path | what |
|---|---|
| `ns3/oran-pmrt-scenario.cc` | ns-3 scenario (symlinked into both ns-3 trees' `scratch/`) |
| `ns3/run_stage.sh` | parallel episode runner |
| `ns3/analyze.py` | logs -> PMRT -> verdict JSON |
| `~/projects/ns-3-opt` | optimized NR-only ns-3 build |
| `~/oran-ns3-runs/` | episode outputs (`<stage>_<seed>/series.csv`, `units.csv`, `topology.csv`), `main.log` |

## Progress log

| time | step |
|---|---|
| 23:11 | 7 dev episodes done (2.0-2.5 h each). ~35 logged units per episode, ~18 inside the t0 window [90, 310]. Dev modes so far: accept 135, half 42, reject 78; Coverage 108, PowerES 147. Wrote `ns3/gt_specs.py` (3 knockout units per placebo seed, fixed RNG `[seed, 6613]`). |
| 00:11 | 11 dev episodes done (2.0-2.6 h). Found a launch bug: zsh does not word-split the unquoted `$specs`, so the queue was one argument and `dev:12 placebo:201` and `placebo:212 eval:101` became single lines; only the first spec of each ran. Started a side runner (`side.log`, 1 job) for placebo:201 then eval:101. 12 processes now on 12 threads. |
| 00:13 | Cleaned `ns3/analyze.py` and `ns3/gt_specs.py`: ruff and ty pass. Regression on the fake logs: eval declarations and placebo counts are identical to the first version. |
| 00:15 | The runs were plain `nohup` processes inside the GNOME Terminal tab's cgroup scope, so closing that tab would have killed them. Moved all 28 processes (both runners, xargs, job shells, 12 ns-3 binaries) into a transient systemd user scope `oran-ns3-runs.scope` via `StartTransientUnit` with `PIDs`; no run was interrupted. Checked: on AC power, GNOME auto-suspend on AC is `nothing`. Linger is off, so a logout or reboot would still stop them. |
| 00:18 | User asked for at most 8 threads. Sent SIGUSR2 x4 to the main xargs (GNU xargs lowers its max-procs by one per signal: 11 -> 7, plus the 1-job side runner = 8). Wrote `ns3/cap.sh`, a governor that keeps at most 8 ns-3 processes running and pauses the surplus with SIGSTOP (youngest first, so progress is kept), resuming the oldest paused ones as slots free. Started it as `oran-ns3-cap.service` (systemd user unit). Now 8 running, 4 paused (placebo 201, 209, 210, 211). |
| 00:22 | User asked for 10 threads with the lid closed. Restarted `oran-ns3-cap.service` with cap 10; SIGUSR1 x2 to the main xargs (7 -> 9 slots, plus the side runner = 10). Started `ns3-lid-inhibit.service` (`systemd-inhibit --what=handle-lid-switch`, 16 h) so closing the lid no longer suspends. Verified: 10 running, 2 paused, inhibitor listed as `block`. |
| 01:12 | Done: all 12 dev; placebo 204, 205, 206, 207, 209. Running: placebo 201-203, 208, 210-212, eval 102-105. CPU temperature reads 93 C (thermal zone). |
| 01:15 | Generated knockouts for the 5 finished placebo seeds (15 reject units) plus one determinism check (`gt:204:0:140:accept`, which must reproduce placebo_204 to t0+90). Launched as systemd unit `oran-ns3-gt1.service` (runner with 3 slots, `gt1.sh`, log `gt1.log`); the cap governor keeps the total at 10 by pausing the youngest runs. |
| 02:15 | Done: placebo 202, 203, 208, 210 (all 12 dev, 9 of 12 placebo). Running: placebo 201, 211, 212; eval 102-108; gt1 knockouts (3 slots, none finished yet). Started `oran-ns3-gt2.service` (`gt2.sh`, 2 slots, log `gt2.log`) with 12 reject knockouts for seeds 202, 203, 208, 210. RAM: 13 ns-3 processes use 3.5 GB, 7 GB available. CPU 95 C. Remaining work is about 100 slot-hours, so about 10 h at the 10-run cap (finish around 12:00); the lid inhibitor runs until about 16:20. |
| 09:32 | The laptop was suspended 03:12-09:32. Cause: running on battery from about 02:56 (upower shows discharging, ~2 %/min under load), then GNOME's battery idle-suspend (900 s). The lid inhibitor worked; idle suspend on battery is a separate path. All processes resumed on wake, nothing lost. Done since 02:15: placebo 201, 211, 212 (all 12 placebo), eval 104. |
| 09:35 | Found a governor starvation bug: xargs starts a new run as soon as one ends, so the count never fell below the cap and the paused knockouts (paused at 01:15) were never resumed while newer eval runs ran. Rewrote `cap.sh` to keep the CAP oldest processes running (FIFO) and restarted it; knockouts resumed, evals 101, 109-111 paused. |
| 09:37 | Started `oran-ns3-gt3.service` (`gt3.sh`, 2 slots, log `gt3.log`): 9 reject knockouts for placebo seeds 201, 211, 212. All 36 knockouts plus the determinism check are now queued. Started a second lid inhibitor (`ns3-lid-inhibit2`, 24 h) because the remaining ~100 slot-hours (~10 h) run past the first one's expiry (~16:20). |
| 10:33 | Determinism check passed: `gt_204_0_140_accept` matches `placebo_204` exactly (all 2079 cell-seconds to t = 231, every KPI column, max abs diff 0; logged units identical). The rerun reproduces the original episode bit for bit, so knockout differences come only from the flipped decision. Done since 09:32: eval 102, 106, 107, 108 (5 of 24 total). Charger back on (95 %). |
| 11:33 | Done: eval 101-110 except 101 (9 of 24), knockouts gt 202:5:91, 202:5:111, 204:0:140, 204:8:158 (4 of 36). Knockout sanity: own-cell differences appear from t0 on, with the expected signs (e.g. a rejected power-down keeps energy higher). |
| 11:40 | Found a measurement issue. In placebo 202 cell 5 had PRB util ~0 from t = 5 s while its one UE kept receiving ~3.4 Mb/s. Checked all runs: thr/util ratio is 4-19 everywhere except that cell. Built a separate diagnostic binary (`scratch/oran-diag.cc` in ns-3-opt, same scenario plus a per-second log of each UE's RRC serving cell; KPI output byte-identical) and ran seed 202 for 10 s: UE 16, attached to cell 5, re-established on cell 0 at t = 4 s (radio link failure then RRC re-establishment, which 5G-LENA does even without handover) and stayed there. The scenario credits UE KPIs (thr, viol, delay) to the cell of initial attachment, so after such an event they go to the wrong cell, while util and energy follow the real serving cell. |
| 11:45 | Kept the main runs (see D29). Queued full-length diagnostic reruns of placebo 202, 204, 207 (`oran-ns3-diag.service`, `diag.sh`, `diag_<seed>/ue_cells.csv`) to measure how often re-establishment happens. Extended `cap.sh` to govern the diagnostic binary too; FIFO puts these runs last. |
| 12:39 | Done: eval 12 of 24, reject knockouts 6 of 36, diagnostics 0 of 3. Fully charged, 91 C, no suspend since 09:32. |
| 13:38 | Done: eval 17 of 24, reject knockouts 10 of 36, diagnostics 0 of 3. About 56 slot-hours left, so done around 19:30. Fully charged, 95 C. |
| 14:37 | Done: eval 17 of 24, reject knockouts 11 of 36, diagnostics 0 of 3. 10 running, 5 paused; fully charged, 93 C. |
| 15:36 | Done: eval 23 of 24, reject knockouts 13 of 36, diag 202. Diag 202 over the full 400 s: only one re-establishment, the same UE 16 (cell 5 to cell 0 at t = 4 s); all other 26 UEs stay on their logged cell; 3.7 % of UE-seconds misattributed; KPI series identical to placebo_202. |
| 15:40 | Spare capacity: the main queue is down to eval 124, and the 3 knockout runners have only 7 slots for 23 knockouts (~8 h). Started `oran-ns3-gt4.service` (`gt4.sh`, 3 slots) with 6 specs from the tails of the gt1-gt3 queues (209:8:118, 209:0:300, 209:1:302, 210:7:236, 210:7:256, 212:2:304). Each target dir got a placeholder `units.csv` first, so the old runners' non-empty check skips them. A failed gt4 run would leave the placeholder; check `gt4.log` rc before analysis. |
| 16:37 | All 24 eval episodes done; knockouts 18 of 36 (gt4 runner working, rc=0). |
| 16:45 | Preliminary analysis on a snapshot (all dev/eval/placebo, the 18 finished knockouts), cap lowered to 9 while it ran (15 s at B = 9999), then back to 10. Saved `~/oran-ns3-runs/prelim1.json`. Units: dev 231, eval 532 (accept 295, half 91, reject 146), placebo 160. **K0 placebo:** 0 of 15 hypotheses at p <= 0.05 in every PMRT arm, no BY declarations (binomial tail 1.0): clean. **Eval:** BY declares 1 edge, own energy (+, p = 1e-4); `best` has 1 of 15 at p <= 0.05. **Preliminary ground truth** (18 units, 8 seeds): 9 TRUE edges (own load -, thr +, viol -, energy +, delay -; nbr thr +, viol -, delay -; far delay -), others INDET. Score: precision 1.0, recall 0.11, sign accuracy 1.0. Not final: labels will change with 36 knockouts. |
| 17:37 | Knockouts 23 of 36. Diag 204: no re-establishment at all (0 % of UE-seconds misattributed), KPI series identical to placebo_204. Diag 207 running. |
| 18:36 | Knockouts 28 of 36; the last 8 and diag 207 are running (none queued). |
| 19:01 | Diag 207: one re-establishment, UE 1 (cell 0 to cell 7 at t = 3 s), 3.7 % of UE-seconds; series identical to placebo_207. Across the 3 diagnostic episodes: 2 of 81 UEs re-established, both within the first 4 s (before the 30 s warm-up ends, so before any logged decision), none later. The misattribution is fixed per episode and the same in every arm, not caused by the knob. |
| 19:37 | All runs finished, every rc = 0, no placeholder left. Stopped both lid inhibitors (normal lid suspend restored); all oran-ns3 units have exited. Final analysis: `~/oran-ns3-runs/final.json` (4 s at B = 9999). |

## Results (final)

Data: dev 12 episodes (231 units), eval 24 (532 units: accept 295, half 91, reject 146), placebo 12
(160 units), 36 reject knockouts over 12 seeds, 1 determinism check (passed, bit-identical).

| check | result |
|---|---|
| K0 placebo | 0 of 15 hypotheses at p <= 0.05 in every PMRT arm; no BY declarations; binomial tail 1.0. Pass. |
| Eval declarations (BY q = 0.05, arm `best`) | 1: own energy, + (p = 1e-4). |
| Ground truth | 11 TRUE, 4 INDET, 0 NULL. |
| Score | TP 1, FP 0, FN 10: precision 1.0, recall 0.09, sign accuracy 1.0. |

| edge | ground truth (mean, 95 % CI) | PMRT p (sign) | placebo p |
|---|---|---|---|
| own load | TRUE- (-4.37 [-6.47, -2.46]) | 0.19 (-) | 0.15 |
| own thr | TRUE+ (1.5 [0.10, 3.19]) | 0.76 (+) | 0.42 |
| own viol | TRUE- (-2.56 [-4.81, -0.67]) | 0.47 (-) | 0.14 |
| own e | TRUE+ (668 [430, 917]) | **0.0001 (+), declared** | 0.81 |
| own delay | TRUE- (-212 [-409, -37]) | 0.65 (-) | 0.20 |
| nbr load | INDET | 0.98 | 0.76 |
| nbr thr | TRUE+ (6.12 [2.32, 11.0]) | 0.55 (+) | 0.73 |
| nbr viol | TRUE- (-7.53 [-12.3, -3.42]) | 0.30 (+) | 0.77 |
| nbr e | INDET | 0.71 | 0.38 |
| nbr delay | TRUE- (-542 [-1000, -174]) | 0.23 (+) | 0.95 |
| far load | INDET | 0.42 | 0.84 |
| far thr | TRUE+ (2.99 [1.0, 5.24]) | 0.56 (+) | 0.90 |
| far viol | TRUE- (-3.44 [-7.17, -0.14]) | 0.55 (-) | 0.78 |
| far e | INDET | 0.13 | 0.66 |
| far delay | TRUE- (-977 [-2480, -89]) | 0.89 (-) | 0.38 |

Reading: PMRT keeps its validity on an external simulator (clean placebo, no false declaration) but
has very low power here: only the near-mechanical energy edge is found. 8 of the 10 missed TRUE
edges have the right sign in the PMRT statistic, which points to too little data rather than bias.
The ground truth is physically coherent: more power means higher MCS, so fewer PRBs (own load
down), better own service, and less interference airtime, so better neighbour service. The `far`
set (diagonal cells at 424 m, D13) is not a null control in this plant: far thr/viol/delay are TRUE.
| 20:05 | Copied all outputs from `~/oran-ns3-runs/` into `ns3/runs/` (16 MB, 374 files, byte-identical) and the diagnostic scenario into `ns3/oran-diag.cc`, so code and data can be committed together (user request; the README keeps the main study records out of git). Launcher scripts in `runs/` still name the old `~/oran-ns3-runs` path. |
