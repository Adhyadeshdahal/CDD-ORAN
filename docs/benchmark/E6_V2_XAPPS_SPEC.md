# E6 xApp stack v2 (`E6-xapps-v2`): specification — DRAFT, not frozen

**Status.**
- Drafted 2026-09-28, after the predeclared Gate A kill of the v1 stack (`scratchpad/e6_dev/GATE_A_RESULT.txt`;
  adjudication in `scratchpad/e6_dev/decision/SOL_GATE_A.md`).
- Revised the same day to include the mechanism-level findings of Worker M's exploratory v1 audit,
  `docs/benchmark/E6_V1_DIAGNOSTICS.md`. That audit is development evidence, not confirmation.
- No v2 SLA, SVR or own-KPI outcome was run or inspected. Nothing below has been evaluated for efficacy.
- This spec becomes binding when `docs/benchmark/E6_V2_CONTRACT.json` is frozen: hashes filled, independent review
  done, and the user's go given. Until then any constant may change, but only on external sources or mechanism tests,
  never on a v2 outcome.

**What this is.**
- A separate experiment. It does not repair v1, and a v2 pass would not validate v1 retroactively.
- v1 Gate A stays published as a negative result: TS and TS+MRO were 1.00–1.03 × freeze, and the controllable fraction
  was 0.017–0.099. The v2 gate reruns v1 next to v2 on fresh tapes (§8).
- A v2 pass supports a narrower claim: *with standards-shaped MLB/MRO xApps, the E6 stress scenarios have enough
  removable SLA violation to test conflict mitigation*. It says nothing yet about Gate B or any arbiter.
- **Read §7 before the gate.** On M's decomposition, Gate A v2 may fail because of the plant's physics, however good the
  xApps are.

**Code.**
- `cdd_oran/envs/e6/xapps_v2.py`:
  - `NRT`: run-time, ANR-like neighbour relations.
  - `MLB`: role "TS".
  - `MROv2`: role "MRO".
  - `KPMV2`: the new `mr` and `mobq` reports, built by `border_snapshot` and `lowq_snapshot`.
  - `MIXES_V2`, `register_v2`, `all_pair_cio_knobs`, `make_env_v2`.
- `tests/test_e6_xapps_v2.py`: 35 mechanical tests (§10).
- v1 `xapps.py`, `ric.py`, `env.py`, `sim.py`, `config.py` and `baselines.py` are **not edited**.

**Provenance tags** (as in `config.py`).
- **[S]**: sourced, and the quoted text has been checked. The checked quotes are those in `E6_STRESS_SCENARIOS.md` §1
  and §4.
- **[V]**: cited, but **not** re-checked against the document for this draft. Every [V] must be checked before the
  freeze (§12).
- **[A]**: a modelling assumption, with its rationale.
- **inherited**: an unchanged v1 constant. It was fixed before any v1 outcome, so reusing it adds no degree of freedom.
- **[D]**: development evidence from `E6_V1_DIAGNOSTICS.md`. Exploratory: 10 paired episodes on seeds 140000–140051,
  no significance test. It motivates mechanisms and is never used to set a threshold.

---

## 1. Why a v2 stack, and what may change

The v1 rule: TS or TS+MRO ≤ 0.85 × freeze **and** controllable fraction ≥ 0.35, at both loads of at least one stress
scenario. v1 failed it in every cell. `SOL_GATE_A.md` allows a new version only under five conditions:

1. the redesign is motivated by external 3GPP / O-RAN MLB and MRO behaviour;
2. the mechanism is specified independently;
3. everything is frozen before any v2 SLA outcome;
4. v2 uses fresh registered seeds;
5. nothing is tuned on the inspected seeds 11–15, the surge placement, the mis-tune cluster or the observed failure
   ratios.

**What v2 changes.**
- The two mobility xApps.
- An additive KPM extension (`mr`, `mobq`).
- The xApps' neighbour relations (run-time NRT), and with them the CIO knob registry.

**What v2 does not change.**
- The scenarios (`E6-scn-v1`).
- The plant, the SLA definition, the RIC actuator limits and the WG3 semantics.
- ES and SLICE (§5).
- The Gate A rule (§8).

### 1.1 v1 mechanism defects and the v2 response

Findings from `E6_V1_DIAGNOSTICS.md` are marked [D]. Findings from reading the code are marked "code".

| # | v1 defect | Evidence | v2 response |
|---|---|---|---|
| N1 | **Static neighbour list, not radio adjacency.** `Layout._neighbours` = same-site sectors + 6 nearest cells by *site position*. The ties make it arbitrary. `env.knobs` is built from it, so pairs outside it are untunable. | [D] 42–70 % of HOs and 46–84 % of too-late (cur, best) pairs fall on pairs **no xApp can tune**. | ANR-like NRT built at run time from delivered KPM (§2.2). Knob registry = CIO on every ordered pair (§2.3). Mechanical coverage ≥ 0.93 (§10). |
| T1 | **No border-UE feasibility.** CIO moves whether or not anyone is within reach. | [D] 82–87 % of +1 dB steps move nobody. A step moves 0.16–0.22 UEs in 10 s. | The `mr` report counts active border UEs per 1 dB bin. Step only if ≥ 1 active UE would move, by the smallest such step. After the change, verify that UEs moved; if none did, release (§3.4). |
| T2 | **Load sampled 1 s in 10 s.** | [D] Sample vs true 10 s mean: \|error\| 0.12–0.16, corr 0.74–0.86. That is larger than the 0.05 reset band. | Window mean over all 10 `fast` reports and both 5 s `thp` reports of the cadence (§3.2). |
| T3 | **Busy-time utilisation as the load input.** One 4 Mbit file exceeds the carrier in one tick, so utilisation ≈ the share of ticks with ≥ 1 backlogged UE. It saturates at 1. | [D] Surge cells at offered/capacity 1.07 and 1.88 both report util 1.0. | Load = window-mean active UEs per shared capacity κ. Decision on the moved UEs' **rate** at target vs source (§3.3). |
| T4 | **Co-channel range-expansion penalty ignored.** | [D] Moved UEs land 1.5–2.2 dB worse, at a median SINR of −0.7…−2.4 dB. | `mr` carries RSRQ-like SE estimates at source and target for the border set. Push only if the moved set's expected rate at target ≥ (1 + γ) × at source (§3.3). |
| T5 | **Oscillation.** Independent s→n and n→s writes, and a reset evaluated on noisy input. | [D] Down-moves exceed up-moves; 72–231 reversals per episode. | One antisymmetric decision per pair. Push only after a 2-cycle hold. Release only after 60 s of recovery (hysteresis). Never push against the xApp's own reverse push (§3.3–3.4). |
| M1 | **MRO never lowers Hys, and it raises Hys on too-late cells.** TTT is lowered only after CIO reaches +6 dB. | [D] Hys+ on already mis-set cluster cells, up to 5 dB. The mis-set is never repaired. | Too-late steps Hys− or TTT−. **Hys+ is forbidden on a cell whose too-late ratio exceeds the threshold** (§4.3). |
| M2 | **RLF counts miss the cost of late HOs.** | [D] Corridor low-SINR dwell before a late HO is 3.8–10.9 % of corridor UE-s post-push. It never becomes an RLF. | New `mobq` KPI: low-SINR dwell while another cell is stronger. It counts as too-late evidence (§2.1, §4.2). |
| M3 | **Stale-window double counting.** 60 s window at a 30 s cadence. | [D] 43–73 % of MRO changes repeat the same direction on the same knob within 60 s. | After an own change on a cell, MROv2 waits a **full fresh window** (2 mob reports starting after the change) before acting on that cell again (§4.2). |
| — | **Stress is a minority of SVR.** | [D] See §7. | Not fixable inside an xApp. Declared as the expected physics ceiling (§7). |

---

## 2. Observability: KPM extension, NRT, knob registry

### 2.1 New KPM reports (`KPMV2`)

**`mr`, every 5 s: periodic UE measurement-report aggregate.** A snapshot of every connected UE (not in an HO
interruption or an RLF outage).

- **A3 margin.** For a UE served by s and each other cell n:

  m = L3-RSRP(n) + CIO[s, n] − Hys[s] − L3-RSRP(s)

  A3 enters when m > 0.
- **Fields.**
  - `conn[s]`: the number of connected UEs.
  - `border_all[s, n, k−1]`: cumulative count of UEs with m ∈ (−k, 0], for k = 1..6. These are the UEs a +k dB
    CIO[s, n] would bring into A3.
  - `border_act`: the same count, restricted to backlogged UEs.
  - `se_src[s, n, k−1]` and `se_tgt[s, n, k−1]`: the mean estimated spectral efficiency of the active border UEs in bin
    ≤ k, at the serving cell and at the target.
    - Estimate: SINR_c = P_c / (Σ_{j≠c} P_j·occ_j + N), from the UE's measured L3 RSRPs and the cells' current PRB
      occupancy, mapped through the E6 link curve (`Plant.se`, TR 36.942 A.2 [S]).
    - This is RSRQ-like [V: RSRQ, TS 36.214]. It needs cell loads, which gNBs exchange through Resource Status
      Reporting [V: TS 36.423].
- **Grounding.**
  - Measurement source: MeasurementReport [V: TS 36.331/38.331], MDT [V: TS 37.320].
  - The O-RAN Traffic Steering use case uses UE-level serving and neighbour measurements [V: O-RAN.WG1 Use Cases], and
    E2SM-KPM has UE-level report styles [V].
  - The per-pair margin histogram is a compact aggregate [A]. It adds nothing a gNB lacks.
  - Bins: K = 6 of 1 dB. 6 dB is `CIO_RANGE`'s half-width [S range]. 1 dB matches the Q-OffsetRange granularity within
    ±6 dB [V: TS 36.331].
  - Period: 5 s [A], the `thp` period.

**`mobq`, every 30 s, aligned with `mob`: pre-RLF low-quality dwell.**
- Once per second, every connected UE whose serving SINR (the plant's RLF-monitor EWMA) is below Qout = −8 dB [V:
  TR 36.839 sim assumption] **while another cell is measured stronger** adds 1 s to `lowq_dwell[s, n]`. Here n is the
  strongest other cell.
- Grounding [V]: T310 / RLF reports (TS 36.331 rlf-Report), MDT radio-link measurements (TS 37.320) and the
  "prolonged connection to non-optimal cell" wording of TR 36.902 cl. 4.5.1 [S].
- The 1 Hz snapshot sampling is an [A] approximation of a counter.
- Why this KPI is needed [D]: most late-HO cost is dwell that never becomes an RLF. Worker M's measure used
  SINR < −10 dB; v2 uses Qout, the standard RLF-detection threshold, and not a value from M's numbers.

**Tape safety.**
- `KPMV2.second()` runs the unmodified `KPM.second()` first.
- `mr` and `mobq` draw drop and delay from `default_rng([seed, 6600, 7101, sec, j])`, with j = 0 for mr and 1 for mobq,
  under the same delay/drop law.
- Result: every v1 report, delay and drop is unchanged (`test_kpmv2_leaves_v1_stack_bit_identical`).
- Tag 7101 is disjoint from `sim.STREAMS` and from the registry tags. It must be registered.

### 2.2 Run-time neighbour relation table (`NRT`): ANR-like

- **Standard.** ANR [V: TS 36.300 cl. 22.3.2a; TS 38.300 cl. 15.3.3; also cited by `E6_V1_DIAGNOSTICS.md`]: a UE
  reports a cell it detects, and the gNB adds the neighbour relation.
- **E6 rule.** Each xApp builds its own NRT from **delivered KPM only**. Relation (s, n) is added in both directions [A]
  when either:
  1. a `mob` report shows an HO attempt, a too-late / too-early / wrong-cell RLF or a ping-pong between s and n; or
  2. an `mr` report has `border_all[s, n, 5] > 0`, meaning a UE of s measures n within the 6 dB margin range.
- **Starting state.** Empty [A]. The 120 s warm-up populates it.
- **No removal.** Relations are never removed within an episode [A: NR removal is an OAM or ageing decision on a far
  longer time scale].
- The xApps iterate `nrt.neighbours(s)`. They never read `Layout.neighbours`.
- **Acceptance criterion** ([D] defect 1): ≥ 95 % of HO attempts and too-late pairs fall on a relation already in the
  NRT. The mechanical check is in §10.

### 2.3 Knob registry and integration hooks

- **The RIC can actuate CIO on any ordered pair; the NRT decides which pairs an xApp uses.**
  - `make_env_v2` sets `env.knobs` = CIO on all nc·(nc−1) ordered pairs (552 for 24 cells), followed by the unchanged
    v1 non-CIO knobs, plus the LL-ratio knobs when SLICE is deployed.
  - It then registers the V2 mixes (in-place `xapps.MIXES.update`, v1 keys untouched, clash-checked), swaps in
    `KPMV2`, and attaches the trace recorder last, so its knob index covers the full registry.
- **`ric.py`: no change is needed.** `feasible` / `LIMITS` / `knob_get` / `knob_set` are keyed by knob *type*, not by a
  neighbour list, so any pair is already actuatable. The per-knob dwell (`last_change`) works for any key.
- **`env.py`: the equivalent permanent hooks** (described, **not applied**), so that `E6Env(cfg)` accepts V2 mixes
  directly. None changes v1 behaviour:
  1. `self.xapps = [cls(self, i) for i, cls in enumerate({**MIXES, **MIXES_V2}[cfg.mix])]`
  2. `self.kpm = (KPMV2 if cfg.mix in MIXES_V2 else KPM)(cfg, self.plant)`
  3. **Knob registry** for V2 mixes: replace both CIO list comprehensions (the `lay.neighbours` forward list and the
     `fwd` reverse-pair list) with `all_pair_cio_knobs(lay.n_cells)`.
  4. `if cfg.mix in ("M4", "V2_M4"):` for the LL-ratio knobs.
  - Import: `from .xapps_v2 import KPMV2, MIXES_V2, all_pair_cio_knobs`. This is safe because `xapps_v2` imports `env`
    lazily.
  - `static()["neighbours"]` keeps returning `Layout.neighbours` for arbiters. An arbiter that wants radio adjacency
    must build its own NRT from `obs["new_reports"]`, which carries `mr` and `mobq`.
- **Comparators left as they are.**
  - `baselines.apply_static` (`cio_to_pico`) and `tuned_static` still use `Layout.neighbours` for the macro–pico pairs.
  - They are stack-independent comparators, and the Gate A rule fixes them as they are. The limitation is disclosed
    (§11) and not repaired, so the controllable-fraction definition stays unchanged.

### 2.4 Mixes

- `MIXES_V2`: `V2_none`, `V2_MRO`, `V2_TS`, `V2_M_TS_MRO`, `V2_M2`, and `V2_M4` = (MROv2, MLB, ES, SliceSLA).
- The positions match v1 M4, so the reused ES and SLICE get identical per-seed variant draws (tested).
- Role names stay `"MRO"` and `"TS"`. That keeps every arbiter, subset, priority rule, published baseline and gate
  script keyed by xApp name runnable unchanged.

---

## 3. MLB xApp (role "TS"), `xapps_v2.MLB`

### 3.1 Standards grounding

- **TR 36.902 cl. 4.6.1** [S]: "Optimisation of cell reselection/handover parameters in order to cope with the unequal
  traffic load and to minimize the number of handovers and redirections needed to achieve the load balancing."
- **cl. 4.6.3** [S]: "some of the UEs at the cell border hand over to a less loaded cell". This is the border-UE
  feasibility and spare-capacity test.
- **TS 28.313 cl. 6.1.1.5** [S] names LBO. TS 28.627 / 28.628 SON policy NRM [V] give operator targets and ranges.
  v2's target is the E6 eMBB SLA.
- **TS 36.423 Mobility Settings Change** [V] is a *cell-pair* HO-trigger change agreed by both ends. v2 makes one
  antisymmetric decision per pair: CIO[s, n] += k and CIO[n, s] −= k.
- **Load measure.** TS 36.423 Resource Status Reporting / Composite Available Capacity [V] and TS 28.552 "number of
  active UEs" [V] measure load as **active UEs per available capacity**.
  - The E6 scheduler is equal-share over the shared pool [A: code, `sim.Plant.tick` step 4].
  - There, a moved UE's rate ≈ SE_target · κ_n / (A_n + m).

### 3.2 Measurements (every 10 s cycle; all window averages)

- `A[c]`: the mean of backlogged eMBB + BE UEs over all 10 `fast` reports of the cycle. LL is excluded [A]: it is
  served first and its demand is small.
- `κ[c] = active carriers / carriers × (1 − ll_ratio[c])`, and 0 when the cell is asleep.
  - This is the shared-pool capacity share. It keeps the designed Slice→TS and ES→TS couplings.
- `suffer[c]`: the mean eMBB p5 rate over the 2 `thp` reports of the cycle is below 2 Mb/s. That is the E6 eMBB SLA
  target itself, with no margin.
  - NaN (no backlogged eMBB) counts as not suffering, and a single NaN report is ignored.
- `border_act`, `se_src` and `se_tgt` come from the latest `mr` report.
- **HO guard on pair (s, n)**, from the latest `mob` report: trips when all of
  - too-early[s, n] + wrong-cell[s, n] + ping-pong[s, n] + ping-pong[n, s] ≥ 3, and
  - that sum / HO attempts[s, n] > 0.10 × scale.

### 3.3 Push rule (one decision per pair; per source s, at most one pair per cycle)

**Source s may push** if all of these hold:
- s is suffering and κ[s] > 0;
- the oldest evidence in use (fast, thp, mr) post-dates s's last own change;
- s is not already involved in a rollback or release this cycle.

**Candidate n ∈ NRT(s)** must satisfy all of:
- n is not suffering, κ[n] > 0, the pair is not blocked, and the HO guard is not tripped;
- there is **no own push n → s** in force;
- **reachable border UEs:** the smallest k ∈ {1, 2} with m = `border_act[s, n, k−1]` ≥ 1 and CIO[s, n] + k ≤ 4 dB;
- **moved-UE rate gain:** `se_tgt·κ[n] / (A[n] + m)` ≥ (1 + γ) · `se_src·κ[s] / max(A[s], 1)`;
- **no overshoot:** κ[n] / (A[n] + m) ≥ κ[s] / max(A[s] − m, 1). The per-UE share of the target's UEs must not fall
  below that of the UEs left at the source.

Among the candidates, the largest gain ratio wins.

**Hold time.** Push only when the same (s, n) qualified on `HOLD_N` = 2 consecutive cycles (20 s).

**Action.** One pair decision: CIO[s, n] + k and CIO[n, s] − k, clipped to `CIO_RANGE`.

### 3.4 Verification, rollback, release

**Rollback** of the last own push on (s, n), with the pair then blocked for 120 s, happens in any of three cases:
1. **No UE moved.** The first `mob` window starting after the push has zero HO attempts s → n.
2. **HO guard.** It trips on a `mob` window starting after the push.
3. **Neighbour degraded.** A `thp` window after the push shows n suffering, although n was fine at push time.

**Release (hysteresis).**
- Once s has not suffered for 60 s, the xApp steps its own offset back 1 dB per cycle, one pair per cycle, as a pair
  decision.
- The enter condition (p5 < target, gain ≥ 1 + γ, held 20 s) differs from the leave condition (p5 ≥ target for
  60 s).
- MLB only removes offset it added itself. Its own pushes are tracked from ACKed applied deltas.

NACK behaviour (retry / escalate / hold) is inherited from v1.

### 3.5 Constants

| Constant | Value | Tag | Rationale |
|---|---|---|---|
| cadence | 10 s | inherited | v1 TS |
| `N_FAST`, `N_THP` | 10, 2 | [A] | exactly one cadence of reports (fixes T2) |
| target | 2 Mb/s eMBB p5 | [A] (E6 SLA) | MLB's objective is the SLA; no margin, so no free parameter |
| `GAMMA` γ | 0.20 × scale | [A] | Required moved-UE rate gain. It covers SE-estimate error and the one-UE granularity (1/(A+1)). A priori, not fitted. |
| `CIO_MAX` | 4 dB | inherited | v1 TS `cio_max` |
| `MAX_STEP` | 2 dB | [S range] | `ric.LIMITS["cio"]` |
| `HOLD_N` | 2 cycles | [A] | the push must survive one full independent cycle of window averages |
| `HO_GUARD`, `HO_MIN_EV` | 0.10 × scale, 3 | inherited | v1 MRO too-early threshold and minimum evidence |
| `BLOCK_S` | 120 s | [A] | = RIC min-dwell of the carrier/sleep knobs; = two MRO windows |
| `RESTORE_S` | 60 s | [A] | = the MRO window, and ES `low_since` |
| scale | U(0.75, 1.25) per seed | inherited | E6 implementation-variant mechanism |
| hidden update | γ → 0.5·γ | [A] | mirrors v1 (update = more eager); gates use `update=False` |

---

## 4. MRO xApp v2 (role "MRO"), `xapps_v2.MROv2`

### 4.1 Standards grounding

- **TR 36.902 cl. 4.5.1** [S]: incorrect hysteresis "may be the reason for either ping-pong effect or prolonged
  connection to non-optimal cell". So Hys must be adjustable in **both** directions.
- **cl. 4.5.2.1** [S] too-late HO; **cl. 4.5.2.2 / 4.5.2.3** [V] too-early and wrong cell. E6 counts all three per
  pair, attributed to the responsible source cell (`sim._rlf`).
- **cl. 4.5.5.2** [S]: MRO may optimise Hys, TTT and CIO.
  - [A] Cell-scoped Hys/TTT are used for issues over several neighbours. Pair-scoped CIO is used for an issue confined
    to one neighbour.
- **TS 28.313 cl. 6.4.1.2 step 3** [S]: "detects handover issues (e.g. too late HO, too early HO and HO to a wrong
  cell) ... and acts to mitigate the HO issues by adjusting HO related parameters."
- **TS 36.331 speed-dependent TTT scaling** (`timeToTrigger-SF`) [V]. It motivates TTT as the tie-break (§4.3).
- **Parameter envelope: TR 36.839 Tab 5.3.2.1 sets** [S numeric]. TTT spans 40–480 ms; the A3 offset spans −1…3 dB,
  carried by Hys ≥ 0, so Hys 0–3 dB [A mapping].
  - MROv2 **raises** a parameter only up to the envelope top.
  - It may **lower** a parameter from anywhere, including from outside the envelope.

### 4.2 Evidence (per cell s, every 30 s)

- **Window.** The last 2 delivered `mob` reports whose window **starts at or after s's last own applied change**,
  i.e. a full fresh window after any own change on s (fixes M3). With fewer than 2 such reports, MROv2 does nothing on
  s.
- **Per neighbour n, summed over the window.**
  - `late_eff[n]` = too_late[s, n] + lowq_dwell[s, n] / T310. A missing or dropped `mobq` report counts as 0.
  - E[n] = too_early[s, n] + ping-pong[s, n].
  - WC[n] = wrong_cell[s, n].
  - HO[n].
- **Dwell weight.** T310 = 1 s [V: TR 36.839 sim assumption, `config.T310_S`]. [A] One second below Qout while a
  stronger cell exists counts as one near-RLF, because a full T310 below Qout is an RLF.
- `den = ΣHO + Σtoo_late`. Nothing is done if den < 3.
- **late** = Σlate_eff ≥ 2 and Σlate_eff / den > 0.02 × scale.
- **early** = ΣE ≥ 3 and ΣE / den > 0.10 × scale.
- **too-late cell** = Σlate_eff / den > 0.02 × scale, with **no minimum count**. It forbids Hys+.

### 4.3 Actions (d = −1: HO earlier, for late; d = +1: HO later, for early)

If exactly one of late and early holds:

1. **Single-pair issue.** The detected events all come from one NRT neighbour n, and none come from cells outside the
   NRT. Then CIO[s, n] −= d·1 dB.
   - MROv2 never pushes CIO outward beyond ±3 dB; it may move CIO inward from anywhere.
   - If the band is exhausted, go to step 2.
2. **Cell-wide.** One step of Hys (0.5 dB) or TTT (one `TTT_SET_MS` index) in direction d, within the envelope.
   - **Too-late → Hys− or TTT−.** Lower the one with the higher normalised envelope position; ties go to TTT.
   - **Early → Hys+ or TTT+.** Raise the one with the lower position, **except that Hys+ is forbidden on a too-late
     cell**, in which case only TTT+ is possible.
   - **Overshoot reversal.** If the direction is opposite to s's last cell-level step, reverse that same knob, and lock
     the original direction on that knob for 120 s.

If both late and early hold, no trigger-wide change is made.

3. **Wrong cell.** For each NRT neighbour n not handled above, with WC[n] ≥ 2 and WC[n] / HO[n] > 0.02 × scale:
   CIO[s, n] −1 dB, within ±3 dB (TR 36.902 cl. 4.5.2.3).

Any applied own change on s resets the fresh-window cut-off for s.

### 4.4 Constants

| Constant | Value | Tag | Rationale |
|---|---|---|---|
| cadence, window | 30 s, 2 reports | inherited | v1 MRO; = mob granularity |
| `TL_THR` | 0.02 × scale | inherited | v1 MRO nominal; the frozen SMO alarm |
| `TE_THR` | 0.10 × scale | inherited | v1 MRO nominal (too-early + ping-pong) |
| `MIN_TL`, `MIN_EV` | 2, 3 | inherited | v1 MRO / SMO minimum evidence |
| `DWELL_PER_EVENT_S` | T310 = 1 s | [V] value / [A] use | 1 s below Qout = one near-RLF |
| Qout | −8 dB | [V] | `config.QOUT_DB` (TR 36.839) |
| `HYS_ENV` | 0–3 dB | [S numeric, A mapping] | TR 36.839 Tab 5.3.2.1 |
| `TTT_ENV` | 40–480 ms | [S] | TR 36.839 Tab 5.3.2.1 |
| `HYS_STEP` | 0.5 dB | [S range] | `ric.LIMITS["hys"]` |
| `CIO_ENV` | ±3 dB | [A] | pair CIO as fine correction (cl. 4.5.5.2); half the actuator range |
| `LOCK_S` | 120 s | [A] | two windows |
| wrong-cell ratio | 0.02 × scale | [A] | reuses the too-late threshold: a wrong-cell HO also ends in RLF |
| hidden update | Hys step 1.0 dB | [A] | mirrors v1's step doubling; gates use `update=False` |

---

## 5. ES and SLICE: unchanged v1 classes

- **Own goals are met.** Gate A's own-KPI lines show both xApps reaching their own goal against freeze in every cell:
  ES energy is 4–10 % lower, and SLICE LL violations are 9–56 % lower.
- **No ES or SLICE defect is reported.** `E6_V1_DIAGNOSTICS.md` audits TS and MRO only.
- **Noted, not changed.**
  - ES samples load the same way as v1 TS: one 1 s report per 10 s, plus an EWMA. It still reaches its own KPI, and
    the 60 s low-load hold smooths it.
  - [D] "Reserved PRBs in the utilisation" is a latent coupling in the M4 arms.
- **How a later fix would be made.** Any future ES/SLICE fix is a new class and a new mix name, never a silent change.

## 6. What v2 deliberately does NOT do

- **No scenario knowledge.** It has no defaults (2 dB / 320 ms), no mis-set cluster, no push time and no surge disk.
- **No plant reads beyond the applied configuration**, which it learns from E2 control ACKs, as v1 does.
- **Every KPI arrives through delivered KPM reports.** `mr` and `mobq` are declared new fields (§2.1).
- **No constant is set from a v1 or v2 outcome** (§11).

---

## 7. Expected physics ceiling (read before the gate)

Worker M's decomposition ([D], freeze arm, exploratory, 10 episodes) shows that most E6 SLA violation is not
addressable by MLB or MRO:

- **Capacity-limited background dominates.** 74–93 % of violated eMBB UE-s are in shared cells where **no** listed
  neighbour would give the violator 2 Mb/s (`cap_unrelievable`). The exception is 49.5 % in the outage-heavy seed
  140021.
- **Relievable by CIO load balancing: ≤ 3.5 %.** Violations relievable within CIO reach are 0.5–3.5 % of violated eMBB
  UE-s. A further 0.3–1.8 % are relievable but out of CIO reach.
- **The surge is a minority.** In-disk violations during the event are 3.8–41 % of all violated UE-s, with a median of
  about 10 %.
- **Mistune repair: ≤ 7.4 %.** The push-induced corridor eMBB outage is about 1.4–7.4 % of all violated UE-s. All
  post-push corridor eMBB violations are 15–32 %. A perfect repair would move SVR by less than 10 %.
- **Outage is out of reach.** RLF or SINR < −10 dB is 5–48 % of violations, and CIO cannot remove it.

**Plain statement.** The v2 Gate A primary is episode SVR, the v1 rule unchanged (§8). It needs TS or TS+MRO
≤ 0.85 × freeze, a 15 % cut in *whole-episode* SVR, at both loads of one stress scenario.
- Against the decomposition above, that needs removing several times the violation that M found addressable by
  CIO/MLB and by the mistune repair.
- **Gate A v2 may therefore fail on the plant's physics regardless of xApp quality.** A v2 FAIL would then mean that
  the E6 plant, not only v1's xApps, lacks enough MLB/MRO-controllable SVR at this bar. It must not be reported as
  "the v2 xApps are bad".
- The decomposition is a move-the-violator bound. It is diagnostic, not a capacity region: it over-states concurrency
  and ignores the gain to UEs left behind. It sets expectations only; it does not change the rule.
- The mechanical property the v2 xApps must show is in §10 (NRT coverage, border feasibility, MRO repair direction).
  The gate decides efficacy.

---

## 8. Gate A rerun (primary rule unchanged)

### 8.1 Primary (decides PASS/FAIL): the v1 rule, scored on the whole scored window

**Scoring window.** The whole scored window (120 s warm-up + 600 s scored, `E6Env.score()["svr"]`), exactly as v1.
This is **declared now, before any v2 outcome.**

**Rule** (verbatim, `DECISION.md` / `gate_a.py:GATE_RULE`). PASS iff, in surge or mistune, at **both** loads of that
scenario:
- min(TS alone, TS+MRO) mean SVR ratio ≤ 0.85 × freeze, **and**
- controllable fraction = (SVR_freeze − SVR_best_static_or_subset) / SVR_freeze ≥ 0.35.

**Computation.** The same as `gate_a.summarize()`:
- the mean over seeds of the per-seed arm/freeze ratio;
- the best subset chosen on the mean;
- the controllable fraction using the per-seed min(best subset, tuned static).

The verdict is computed **per stack**.

### 8.2 Secondary (labelled SECONDARY; descriptive; never gating, never used to rescue a FAIL)

These use the same arms, ratios and controllable fraction, computed on the event window only. The window is defined
now:
- **surge:** [t0, t_end] of `SurgeScenario`, the whole trapezoid.
- **mistune:** [push_t, end of scoring].

**Source.** Per-second trace labels (`decision.trace`: `lab_viol`, `lab_ue`), summed over all cells. Requires
`trace=True` on every arm. The recorder is untouched, so arms stay bit-identical.

Reported in a separate table titled "SECONDARY — event window". The primary verdict does not depend on it.

### 8.3 Matrix, arms, reporting

**Matrix.**
- Seeds 151000–151009.
- Scenarios: base (the control), surge and mistune. Loads: medium and high.
- mobility mixed, KPM nominal, `update=False`, 120 + 600 s.
- Stacks: v1 `M4` and v2 `V2_M4` on **identical tapes**; `KPMV2` is tape-neutral (§2.1).

**Arms, per stack.**
- noarb;
- the 14 non-trivial subsets (including TS, MRO and TS+MRO);
- mistune: SMO restore alone (accept-all) and wrapped around TS+MRO.

**Arms run once per tape.**
- freeze and tuned static (hindsight, unchanged grid). They are stack-independent.
- Check: freeze under M4 = freeze under V2_M4, bit for bit, on one seed, and the result is reported.

**Reporting.**
- v1 and v2 side by side per cell: ratios, controllable fraction, and own KPIs (MRO RLF / HO / ping-pong; TS eMBB
  violations; ES energy; SLICE LL violations).
- Also per cell: changes, requests, ACK rate, MLB pushes / rollbacks (by cause) / releases, MRO Hys / TTT / CIO actions
  by direction, and NRT size and coverage.
- Every seed is reported: no filtering, no reruns, crashes count as noarb, and a failed gate stays failed.

**Outcomes.**
- **v2 PASS:** Gate B may be considered under its own frozen protocol, and the claim is the narrower one in the status
  block.
- **v2 FAIL:** stop the SLA route for both stacks. Report the FAIL with §7.

**Cost.** Needs measuring before launch.
- One 720 s episode took about 25–30 s single-process on a 20 s smoke. v2's `mr` / `mobq` snapshots add a little.
- Per tape: 15 noarb/subset episodes (+2 SMO in mistune) × 2 stacks, + 1 freeze + about 13 tuned evaluations ≈ 45–48
  episodes.
- Over 60 tapes that is about 2 800 episodes, or 20–25 CPU-hours, parallel by tape.
- **The launch needs an explicit user go.**

---

## 9. Seeds (to be registered in `SEED_REGISTRY.json#E6` at freeze)

| Set | Range | Use | Allowed inspection |
|---|---|---|---|
| v2 DEV | 150000–150019 | mechanics and integration only (requests, ACKs, applied changes, NRT coverage, border histograms, MRO classifications) | **no SLA / SVR / own-KPI field** (runner drops them) |
| v2 Gate A confirmation | 151000–151009 (10 seeds × each scenario × load) | §8, v1 and v2 side by side | full, once, after freeze |
| v2 reserved | 152000–152029 | untouched until a v2 Gate B protocol is frozen | none |

- Existing E6 blocks: 0–30, 100000–100179, 110000–110179, 120000–120179, 130000–130179, TEST ≥ 960000, and Worker M's
  v1 diagnostics 140000–140059 (registered in the working tree). v2 avoids the whole 140000 block.
- The mechanical tests and checks in §10 used seed 7 (inside the existing DEV 0–30) and read no SLA field.

---

## 10. Mechanical verification (no efficacy)

`uv run pytest tests/test_e6_xapps_v2.py`: **35 passed**. All E6-importing tests together: 175 passed (§10.3). `ruff`: clean. The
unit tests use a fake plant whose `Layout.neighbours` is **empty**, so every v2 decision must come from the run-time
NRT.

### 10.1 Unit tests

**NRT.**
- Symmetric relations from HO/RLF statistics and measurement reports only.

**MLB.**
- Antisymmetric pair push toward the spare NRT neighbour.
- Hold time (2 cycles; resets when the condition breaks).
- No push without border UEs.
- **A moved-UE rate gain is required** (a target SE penalty blocks the push).
- Minimal step ≤ 2 dB.
- Suffering, not-spare or asleep neighbours are rejected.
- The capacity share counts carriers and the LL reservation.
- Window-mean rate, not the last sample (NaN handling).
- No overshoot.
- The 4 dB band holds, and there is no push against its own reverse push.
- The HO guard blocks.
- It waits for post-change evidence.
- Degradation rollback, plus the 120 s block.
- **Moved-UE verification:** a no-move rollback; stale HO evidence is ignored.
- Release after 60 s of recovery.
- 200 random inputs: every proposal is bounded and `ric.feasible` without clipping, and every outward push is paired.

**MROv2.**
- Late → TTT then Hys, **only after a full fresh window** (half a window is not enough).
- **A mistune-like 3 dB / 480 ms state is repaired toward the pre-push 2 / 320 and then stops.**
- **Low-SINR dwell alone triggers the late path.**
- **No Hys+ on a too-late cell** (TTT+ instead; Hys+ when not too-late).
- Early raises TTT from the floor.
- Overshoot reversal plus lock.
- Single-pair CIO within ±3 dB, falling back to cell-level.
- Minimum evidence.
- Late and early together leave the trigger alone.
- The envelope caps raises but allows cuts from 5 dB / 640 ms.
- Wrong cell → CIO −1.
- 200 random inputs: every proposal is bounded and feasible.

**KPM / integration.**
- `border_snapshot` bins and SE fields.
- `lowq_snapshot` attribution.
- **`KPMV2` leaves v1 M4 bit-identical** (40 s).
- `make_env_v2` (V2_M4, mistune, trace on):
  - the v1 registry is unchanged;
  - ES and SLICE draws equal v1;
  - the full 552-pair CIO registry is present;
  - every request is on a registered knob.
- **NRT coverage in a running env** (V2_M4, mistune, 240 s): ≥ 95 % of HO + too-late mass lands on relations already
  in the NRT, and more than the static list covers.

### 10.2 Mechanical integration observations (DEV seed 7, request / event counts only, no SLA field read)

**NRT coverage over 240 s** (mob reports from t ≥ 60 s; each report checked against the NRT built from earlier
reports).

| scenario | HO on NRT | HO on `Layout.neighbours` | too-late on NRT | too-late on list | NRT relations / cell |
|---|---|---|---|---|---|
| mistune | 0.987 (n = 2211) | 0.354 | 0.985 (n = 133) | 0.060 | 17.8 of 23 |
| surge | 0.970 (n = 811) | 0.409 | 0.947 (n = 19) | 0.526 | 17.5 |
| base | 0.972 (n = 813) | 0.411 | 0.929 (n = 14) | 0.429 | 17.3 |

- The too-late criterion (≥ 95 %) is missed on the small samples (14–19 events). The first occurrence of a new pair
  is uncovered by construction.
- The NRT is dense (about 17 of 23 other cells), because the 6 dB `mr` margin detects many cells. It is not pruned
  [A]: a wide NRT only widens what the xApps *may* use; every action still needs its own evidence.

**Request counts over 240 s, V2_M4, accept-all.**
- mistune: TS/cio 6, MRO ttt 15 / hys 5 / cio 9.
- surge: TS 8, MRO 13 / 4 / 4.
- base: TS 4, MRO 11 / 5 / 4.

**MROv2 trajectory over 600 s, `V2_MRO`, seed 7 (a mechanical property; disclosed for the review).**
- **base:** the low-SINR dwell term (147 s of dwell vs 26 too-late RLFs) moves TTT down on most cells. Final: TTT 256
  ms on 15 cells and 160 ms on 5; Hys 1.5–2 dB on 22 of 24 cells.
- **mistune:** the cluster moves from 3 dB / 480 ms toward the default. Final: 8 of 10 cluster cells at TTT ≤ 320 ms
  and 6 at Hys ≤ 2 dB.
- **Without the dwell term** (a mechanical variant check, not adopted): base stays near the default (TTT 320 on 19
  cells), and mistune repair is partial (5 of 10 cluster cells still at 480 ms).
- So the dwell KPI does what M2 asks for, and it also drives MROv2 to make TTT more aggressive in the calm base
  network. Whether that helps or hurts SLA is **unknown and was not looked at**.
- The design was not changed after this check. The dwell weight (1 s = 1 near-RLF) is an [A] item for review (§12).

### 10.3 Full E6 suite

Last run 2026-09-28, with `uv run pytest`:

- **111 passed** in the E6 files: `test_e6_xapps_v2.py`, `test_e6_scenarios.py`, `test_e6_env_wg3.py`,
  `test_e6_baselines.py` and `test_e6_published.py`.
- **64 passed** in the decision tests that import E6: `test_decision_arbiter`, `_discovery`, `_effect_model`,
  `_probe_crt`, `_rank_eval` and `_trace`.
- In total, **175 passed and 0 failed**.

---

## 11. Disclosure

**Seen before or while drafting.**
- The v1 Gate A table and own-KPI lines, `SOL_GATE_A.md`, `DECISION.md` and `E6_STRESS_SCENARIOS.md` (including its §7
  sanity ratios), and the v1 code.
- **`E6_V1_DIAGNOSTICS.md` and `v1_diag/analysis.txt` in full.** That covers per-episode SVR ratios on seeds
  140000–140051, the defect evidence, and the violation decomposition.

**Not seen.** Any v2 SLA / SVR / own-KPI outcome, any HOLDOUT variant, any TEST or v2 seed.

**Changes from M's findings.**
- **Mechanisms.** The findings motivated the mechanisms: NRT, rate-gain test, move verification, hold, pair decision,
  Hys− / no-Hys+, dwell KPI, and the full fresh window.
- **Thresholds.** No threshold was taken from M's numbers.
  - Qout comes from `config`, not M's −10 dB.
  - γ, the hold and the blocks are a-priori [A].
  - Inherited v1 constants are unchanged.

**Mechanical check after the design (§10.2).** The dwell / no-dwell trajectory comparison on DEV seed 7 was run after
the design was written. It read no SLA field and did not change the design. It is reported so that reviewers can judge
the dwell weight.

**New [A] constants.** γ 0.20, hold 2 cycles, MLB band 4 dB, MRO pair band ±3 dB, 120 s block and lock, 60 s release,
the 5 s `mr` period, the dwell weight = T310, the TTT tie-break, NRT without removal, and symmetric relations. None is
fitted.
- **TTT tie-break.** It favours the knob that the S2 mis-set moved furthest from the calm default. The rationale is TS
  36.331 speed-dependent TTT scaling, but the coincidence is disclosed.
- **Envelope.** The TR 36.839 envelope (TTT ≤ 480 ms, Hys ≤ 3 dB) comes from the same source as the S2 mis-set pair.

**Comparators.** The tuned-static comparator's `cio_to_pico` still uses `Layout.neighbours` (§2.3). It is unchanged
so that the v1 rule stays fixed.

**Author.** One worker wrote this spec. sol reviews it next against `SOL_GATE_A.md`: external motivation, independent
specification and no scenario leakage.

## 12. Open before freezing (user / reviewer decisions)

1. **Confirmation n.** 10 seeds are proposed, against 5 in v1.
2. **Hooks.** Apply the four `env.py` hooks (§2.3), or keep `make_env_v2` as the only entry point.
3. **Seed registration.** Register the §9 seeds and rng tag 7101.
4. **Dwell weight.** Currently 1 s = 1 near-RLF. It is the dominant too-late evidence in calm base (§10.2); the
   reviewer should decide. Any change must be made on standards or mechanism grounds, not SLA.
5. **NRT density.** No pruning (§10.2). Whether to add a detection-strength floor.
6. **Secondary event-window table (§8.2).** Keep it or drop it. It never gates.
7. **[V] citations** to check against the documents: TS 36.300 cl. 22.3.2a, TS 38.300 cl. 15.3.3, TS 36.423 (Mobility
   Settings Change, Resource Status / Composite Available Capacity), TS 28.552, TS 28.627/628, TS 36.331
   (Q-OffsetRange, `timeToTrigger-SF`, rlf-Report), TS 36.214 RSRQ, TS 37.320, O-RAN WG1 Use Cases / E2SM-KPM, TR
   36.902 cl. 4.5.2.2 / 4.5.2.3, and TR 36.839 Qout / T310.
