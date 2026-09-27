# E6-scn-v1 xApp stack: post-Gate-A diagnostics (EXPLORATORY)

**Status: EXPLORATORY, 2026-09-28.** This is a diagnosis of the frozen `E6-scn-v1` scenarios and the v1 xApps
(`cdd_oran/envs/e6/xapps.py`) after the predeclared Gate A KILL (`scratchpad/e6_dev/GATE_A_RESULT.txt`,
`scratchpad/e6_dev/decision/SOL_GATE_A.md`). It does **not** reverse or reinterpret Gate A. Nothing here is
confirmatory: there are 10 paired episodes over 11 fresh seeds, with no significance testing. The numbers are
development evidence that a v2 spec can cite as the motivating defect. A v2 spec must not use them to tune thresholds.
No cdd_oran code, scenario, xApp or frozen document was changed.

- Script: `scratchpad/e6_dev/v1_diag.py`. Its wrappers are read-only and patch instances only. The `selfcheck`
  command confirms that an instrumented episode scores bit-identically to a plain one.
- Raw outputs: `scratchpad/e6_dev/v1_diag/` (per-episode `.npz` and `.json`, plus `analysis.txt`).
- Seeds: a new registered DEV diagnostic range, **140000–140059**, in `SEED_REGISTRY.json` → E6 →
  `dev_reserved.v1_diagnostic_episodes`. The mapping is seed = 140000 + 10·stratum + j, using the same stratum order
  as the collection seeds. Seeds 11–15 were not used.
- Configuration: the Gate A configuration (mix M4, all four xApps deployed, mobility `mixed`, KPM nominal, 120 s
  warm-up + 600 s scored).
- Arms:
  - `freeze`: rejects every request.
  - `TS`: `subset(("TS",))`, i.e. TS alone.
  - `MRO`: `subset(("MRO",))`, i.e. MRO alone.
- Paired arms use the same plant tape.

**Episodes run (all reported).** 22 episodes of about 22 s each, one process at a time:

| stratum | seeds | arms |
|---|---|---|
| base-medium | 140000 | freeze, TS |
| base-high | 140010 | freeze, TS |
| surge-medium | 140020, 140021 | freeze, TS |
| surge-high | 140030, 140031 | freeze, TS |
| mistune-medium | 140040, 140041 | freeze, MRO |
| mistune-high | 140050, 140051 | freeze, MRO |
| (selfcheck) | 140029 surge-medium | TS instrumented + TS plain; SVR 198.44 both; not analysed further |

The X4 geometry table also builds plants for the surge seeds without running any ticks.

| episode | SVR freeze | SVR xApp | ratio | eMBB viol freeze → xApp |
|---|---|---|---|---|
| base-medium 140000 (TS) | 515.1 | 500.8 | 0.972 | 24500 → 23907 |
| base-high 140010 (TS) | 655.8 | 670.6 | 1.023 | 30430 → 31179 |
| surge-medium 140020 (TS) | 118.2 | 118.4 | 1.002 | 4731 → 4705 |
| surge-medium 140021 (TS) | 57.2 | 55.5 | 0.970 | 2264 → 2178 |
| surge-high 140030 (TS) | 237.3 | 234.9 | 0.990 | 10484 → 10328 |
| surge-high 140031 (TS) | 604.2 | 606.5 | 1.004 | 27194 → 27217 |
| mistune-medium 140040 (MRO) | 393.6 | 383.5 | 0.974 | 15749 → 15476 |
| mistune-medium 140041 (MRO) | 218.9 | 207.8 | 0.949 | 8299 → 7848 |
| mistune-high 140050 (MRO) | 513.1 | 515.2 | 1.004 | 21634 → 21764 |
| mistune-high 140051 (MRO) | 463.1 | 457.4 | 0.988 | 18569 → 18454 |

The fresh seeds reproduce the Gate A picture. TS/freeze ranges 0.97–1.02 and MRO/freeze 0.95–1.00. Nothing comes
near 0.85×.

---

## Q1. TS mechanism audit: where the chain breaks

Each TS request is traced through: proposal → `feasible`/`_quantise` → ACK/NACK → realised CIO → handovers on the
pushed pair → SINR and violations of the moved UEs.

| stage | observation (6 TS episodes, range) | verdict |
|---|---|---|
| Trigger funnel over (10 s cycle × neighbour pair), 13 392 per episode | `u_s>u_hi` 3240–8080 → `& suffer` 884–6030 → `& u_n<u_tgt` 290–978 → `& gap` 287–975 | fires often: 425–959 requests / 720 s |
| Quantisation / ACK | 100 % ACK, 0 NACK. Every request is an integer ±1 dB CIO step. The 5 s CIO dwell is shorter than the 10 s TS cadence, so dwell never binds. | **no break** |
| Realised moves | 425–959 moves per episode. **Down-moves outnumber up-moves (+199…438 / −226…535).** 72–231 direction reversals over 148–196 knobs. | **break: oscillation** |
| CIO move → UEs move | After a realised +1 dB CIO(s→n), the mean number of HOs s→n in the next 10 s is **0.16–0.22**. **82–87 % of up-moves move nobody.** Only 23–198 of 3670–4988 HOs per episode (0.6–4 %) are CIO-induced, i.e. would not fire at CIO 0. | **break: almost no border UEs move** |
| Where UEs go | **Only 30–58 % of all HOs go to a cell on the TS/MRO neighbour list** (X1 below). | **break: neighbour list** |
| Moved UEs' radio | CIO-induced HOs land at a median target SINR of −0.7…−2.4 dB. That is **1.5–2.2 dB worse than at the source (61–75 % worse)**. Ping-pong 0–1 %. | moved UEs pay SINR |
| Moved eMBB UEs' violations (10 s before → after) | 200→170, 341→265, 218→155, 193→163, 41→46, 16→11 violated UE-s | small local gain |
| Paired eMBB violations, TS − freeze | −593, +749, −26, −86, −156, +23 | ≈ 0 net |
| By class (TS/freeze counts, Q2 taxonomy) | `cap_relievable_reach4` falls where TS acts (base-medium 855 → 134; surge-high 146 → 91, 170 → 125). `cap_unrelievable` rises by a similar or larger amount (base-medium 21669 → 21964). | TS does relieve its addressable class, but the class is ≤ 5 % of violations and the relief is offset elsewhere |

**Where it breaks.** Proposal, quantisation and ACK work. The chain fails in five places:

1. **Neighbour relation (structural).** `Layout._neighbours` builds the list from the same-site sectors plus the 6
   nearest cells *by site-position distance*. Sectors of one site share a position, so the list ends up as argsort
   ties: 2 other sites × 3 sectors, or nearby picos. It does not follow radio adjacency or boresight.
   - 42–70 % of actual HOs, and 46–84 % of too-late RLF (cur, best) pairs, go to cells that **no xApp can tune**.
   - The knob registry (`env.knobs`) is built from the same list.
   - For static UEs in the surge disk, the second-best cell is on the list in only 61–77 % of cases.
2. **Few border UEs.** A +1 dB CIO step moves 0.2 UEs in 10 s. With Hys 2 dB and TS `cio_max` 4 dB, A3 can only
   fire for UEs within 2 dB of the neighbour. X4 shows only 0–22 % of static in-disk UEs within 2 dB of their
   second-best cell (18–31 % within 4 dB).
3. **Co-channel range-expansion penalty.** The UEs that do move lose 1.5–2.2 dB of SINR and land near 0 dB. With the
   serving cell fully loaded, the second-best cell's SINR is −9…−11 dB for the median disk UE (X4).
4. **Oscillation, from the reset rule and the reverse write.**
   - Every push also writes CIO(n,s) − 1.
   - The `|d| < 0.05 → CIO toward 0` reset is evaluated every 10 s on a noisy 1 s utilisation sample (Q4).
   - Result: more down-moves than up-moves and 72–231 reversals per episode. The CIO random-walks instead of holding
     an offload.
5. **Load estimate.** The target check `u_n < u_tgt` uses PRB utilisation. That is roughly the share of ticks with
   ≥ 1 backlogged UE, because one 4 Mbit file demands more than the whole carrier in a tick (X2). It saturates at 1:
   surge cells with offered/capacity 1.07 and 1.88 both report util 1.0. It carries no information about the
   target's per-UE rate for an incoming *edge* UE.

The other hypotheses do not explain the failure:

- **Reserved PRBs in the utilisation.** Not active here: the TS-alone arm rejects SLICE, so `ll_ratio` stays 0. It
  remains a latent coupling in the M4 arms.
- **Too late / too small.** Not the primary cause: TS issues 425–959 accepted ±1 dB steps per episode, yet a step moves 0.2 UEs.
- **Target also loaded.** Partly true: only 4/8–10/11 of the surge cells' neighbours have util < 0.5 (Q2).

## Q2. Surge eMBB violation decomposition: can load balancing relieve the surge?

Each violated eMBB UE-second, taken from the end-of-second snapshot in the freeze arm, is put in one class:

- `outage`: RLF, or SINR < −10 dB.
- `sinr_lim`: the UE could not get 2 Mb/s even alone on the full band.
- `cap_relievable_reach4`: some listed neighbour would give it ≥ 2 Mb/s at an equal share with one more UE, *and*
  A3 would fire at CIO 4 dB.
- `cap_relievable_unreach`: such a neighbour exists but is out of CIO reach.
- `cap_unrelievable`: no neighbour would give it ≥ 2 Mb/s.

This is a *move-the-violator* bound. It counts every UE backlogged at any time in the second as concurrent, which
over-states concurrency, and it ignores the gain to the UEs left behind. It is a diagnostic, not a capacity-region
computation.

| freeze episode | violated eMBB UE-s | outage | sinr_lim | relievable reach4 | relievable unreach | unrelievable |
|---|---|---|---|---|---|---|
| base-medium 140000 | 24500 | 6.6 % | 0.0 | 3.5 | 1.5 | 88.4 |
| base-high 140010 | 30430 | 6.4 | 0.2 | 0.5 | 0.3 | 92.6 |
| surge-medium 140020 | 4731 | 22.7 | 0.2 | 1.4 | 1.8 | 73.8 |
| surge-medium 140021 | 2264 | 48.2 | 0.1 | 1.6 | 0.5 | 49.5 |
| surge-high 140030 | 10484 | 11.0 | 0.1 | 1.6 | 0.7 | 86.6 |
| surge-high 140031 | 27194 | 5.0 | 0.2 | 0.5 | 0.9 | 93.4 |

Surge event detail, from the plateau (181 s) and the whole event:

| seed | UEs in disk | in-disk eMBB viol over the event (share of all violated UE-s) | top surge cells: offered/capacity (util) | their neighbours: offered/capacity, cells with util < 0.5 |
|---|---|---|---|---|
| 140020 medium | 49 | 2428 (**41.1 %**) | 1.51, 0.88, 0.91 (1.0, 1.0, 1.0) | 0.34, 4/8 |
| 140021 medium | 24 | 108 (**3.8 %**) | 0.24, 0.45, 0.61 (0.30, 0.54, 0.67) | 0.23, 10/11 |
| 140030 high | 28 | 1387 (**11.7 %**) | 1.55, 0.46, 0.62 (1.0, 0.55, 0.77) | 0.38, 6/14 |
| 140031 high | 34 | 2357 (**7.8 %**) | 1.07, 1.88 (1.0, 1.0) | 0.49, 4/13 |

**Answer.** In aggregate the surge is relievable: 3 of 4 seeds overload 1–2 cells (offered/capacity 1.07–1.88) while
neighbours average 0.23–0.49. Per UE and per geometry, CIO-based offload can barely touch it:

- **Violations are mostly background.** 74–93 % of violated eMBB UE-s (49.5 % in the outage-heavy seed 140021) sit in shared cells
  where no listed neighbour would give the violator 2 Mb/s. Only 0.5–3.5 % are relievable within CIO 4 dB.
- **Little border population.** The disk (radius 250 m = ISD/2) spreads over 5–8 serving cells, and 0–22 % of static
  in-disk UEs are within 2 dB of their second-best cell.
- **The surge is a minority of SVR.** In-disk violations during the event are 3.8–41 % of all violated UE-s (median
  about 10 %). Even total relief of the surge UEs could not reach 0.85× freeze in 3 of 4 seeds.
- **Other drivers.**
  - The background eMBB violation in every scenario is the load-driven equal-share rate of a fully co-channel,
    interference-coupled network. In base, SVR is 515–656 on these seeds, versus 118–604 in surge.
  - Outage (RLF, or SINR < −10 dB) is 5–48 % of violations. CIO cannot remove it.

## Q3. MRO in mistune: why RLF improves but SVR, HO and ping-pong do not

| seed | RLF/UE·h freeze → MRO | HO/UE·h | ping-pong | realised MRO changes by type | final cluster Hys / TTT |
|---|---|---|---|---|---|
| 140040 medium | 15.10 → 9.60 | 215 → 222 | 394 → 395 | CIO+ 39, CIO− 8, **Hys+ 8**, TTT− 1 | 3.0…4.5 dB / 480 (one cell 320) |
| 140041 medium | 4.22 → 2.52 | 182 → 183 | 545 → 541 | CIO+ 4, CIO− 3, **Hys+ 3** | 3.0 / 480 |
| 140050 high | 4.22 → 2.94 | 194 → 161 | 1525 → 642 | CIO+ 5, CIO− 7, **Hys+ 10** | 3.0…**5.0** / 480 |
| 140051 high | 4.08 → 3.32 | 219 → 208 | 534 → 501 | CIO+ 3, CIO− 2, **Hys+ 2** | 3.0…4.0 / 480 |

**MRO never lowers Hys, and it raises it.** There is no Hys− path in `MRO.propose`. The too-early/ping-pong branch
(`te ≥ 3 and tl == 0` on a pair) raises Hys by 0.5 dB, and in all 4 seeds it did so on **cluster cells that were
already mis-set to 3 dB**, up to 5 dB. TTT is lowered only after a pair's CIO reaches +6 dB, which happened once
(1 cell 480 → 320). The mis-set is therefore never repaired. It is partly compensated with pair CIO, which applies
to every UE of the pair and not only corridor UEs.

Why RLF improves while SVR does not:

1. **RLF is the tip, and outage is the mass.**
   - Corridor too-late RLF drops strongly (505 → 249, 129 → 41, 74 → 6, 51 → 17).
   - Corridor eMBB *outage* (per corridor-eMBB UE-s, post-push) falls only 0.322 → 0.286, 0.203 → 0.191,
     0.346 → 0.315, 0.329 → 0.295.
   - Most corridor outage is SINR < −10 dB dwell before a late HO: 3.8–10.9 % of corridor UE-s post-push, versus
     1.3–5.6 % pre-push. That dwell never becomes an RLF, so the MRO counters (RLF classification only) do not see it.
2. **The mistune is a small slice of SVR.**
   - The push-induced increase in corridor eMBB outage is about **1.4–7.4 % of all violated UE-s** (the pre-push
     reference is only 60 s).
   - All post-push corridor eMBB violations are 15–32 %.
   - The other 68–85 % are the Q2 background, which MRO does not address. A perfect mistune repair would move SVR
     by less than 10 %.
3. **Blind pairs.** Only 16–54 % of too-late RLF (cur, best) pairs are on the neighbour list (X1). MRO also needs
   `den ≥ 3` and `tl ≥ 2` per pair in a 60 s window.
4. **HO and ping-pong.** In the paired fresh seeds they are mixed rather than uniformly worse: 140050 improves (Hys↑
   suppresses corridor ping-pong, 667 → 25), 140040 is worse on HO. Gate A's systematic worsening comes from CIO
   compensation adding HOs for non-corridor UEs, plus the stale-window repeats (Q4). Raising Hys trades ping-pong
   against more too-late behaviour on the corridor.

## Q4. Metric timing

| gran | window | delivery lag | age when handed to xApps | consumer / cadence |
|---|---|---|---|---|
| fast | 1 s | 0.10–0.50 s (mean 0.30) | +1.0 s after window end (next integer second) | TS/ES 10 s, SLICE 1 s |
| thp | 5 s | 0.10–0.50 | +1.0 s; TS decision age 2–5 s | TS |
| mob | 30 s | 0.10–0.50 | +1.0 s | MRO 30 s (60 s window) |
| energy | 60 s | 0.10–0.49 | +1.0 s | – |

KPM delay itself is not an artefact: 1 s against cadences of 10–30 s. There are two **sampling** artefacts:

- **TS reads one 1 s `fast` report per 10 s cycle.** `observe` keeps only the latest report, so 9 of 10 are
  discarded before the 0.7/0.3 EWMA. Compared with the true 10 s mean, the sample has mean |error| 0.12–0.16 and
  correlation 0.74–0.86. That error is larger than the reset band (|d| < 0.05) and comparable to the gap
  (0.15·scale), so triggers and resets are noise-driven (Q1-4).
- **MRO uses a 60 s window (2 × 30 s mob reports) at a 30 s cadence.** Every decision after a change is taken on a
  window that is at least half pre-change. 43–73 % of realised MRO changes (16/22, 3/7, 36/56, 5/10) repeat the same
  direction on the same knob within 60 s. This is a double-count / overshoot pattern.

## Mechanism-level defects a standards-based v2 must fix

These are for Worker N's v2 spec. Each should get a unit or mechanical test before any SLA outcome.

1. **Neighbour relations from radio adjacency (ANR-like).** Build the NRT and knob set from observed HO/measurement
   adjacency (TS 36.300 ANR; TS 38.300 cl. 15.3.3), not from 6-nearest site distance. Acceptance criteria:
   - ≥ 95 % of HOs and of too-late (cur, best) pairs are on a tunable pair.
   - Test: fraction of HOs to listed neighbours.

   `xapps_v2.py` currently reuses `lay.neighbours`, so it inherits this defect.
2. **MLB needs border-UE feasibility and target rate, not busy-time utilisation.**
   - Use a PRB-demand or composite-available-capacity measure (TS 36.423 Resource Status; composite available
     capacity) plus the count of UEs within reach of the offset: per-neighbour RSRP margin < offset + Hys, from
     MR/A3/A4-type reports.
   - Estimate the moved UEs' target SINR/rate and act only when the moved set gains. Budget the offset by expected
     moved load.
   - Test: a +1 dB step must move a predicted number of UEs.
3. **No reset/oscillation on noisy input.**
   - Use a window-average utilisation (all fast reports), apply hysteresis and a hold time to the release rule, and
     make push and release paired and symmetric (one decision per pair, not independent s→n and n→s writes).
   - Test: reversals per knob-hour on a stationary tape.
4. **MRO must act on the mis-set parameter.**
   - Add Hys− and TTT− paths driven by the too-late ratio (TR 36.902 cl. 4.5.5.2 lists Hys/TTT/CIO).
   - Forbid Hys+ on a cell whose too-late ratio is above threshold.
   - Add a too-late precursor KPI (time below Qout / SINR < −10 dB before HO), because RLF counts miss most of the
     cost.
   - Test: from the mistune rollout state with no other xApp, MRO returns cluster Hys/TTT toward the pre-push values.
5. **Evaluation windows aligned to changes.** After a change, discard pre-change evidence (reset the window) or wait
   one full window before re-acting on the same knob.
6. **Scenario/metric headroom caveat, not a defect to fix inside v2.**
   - In these seeds the stress events are 4–41 % (surge) and 15–32 % (mistune, all corridor violations) of violated
     UE-s. Background capacity-limited eMBB violation dominates SVR.
   - A 0.85× *episode-SVR* bar is therefore structurally hard for any MLB/MRO mechanism in 3 of 4 surge seeds and all
     mistune seeds.
   - A v2 contract must declare *before any v2 outcome* whether its gate is scored on the event cells/window or the
     whole episode. That choice must not be made from these numbers after the fact.
