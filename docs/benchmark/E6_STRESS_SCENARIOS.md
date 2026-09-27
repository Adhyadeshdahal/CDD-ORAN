# E6 stress scenarios: S1 surge (MLB) and S2 mistune (MRO), version `E6-scn-v1`

**Status.**
- Specified 2026-09-28, before any arbitration or xApp SLA outcome in these scenarios was run or inspected.
- Revised the same day after an adversarial review (`scratchpad/e6_dev/decision/SOL_SCENARIO_REVIEW.md`). The
  revision added disclosure, validation, OAM/RIC semantics, an SMO reference and a frozen contract. Every numeric
  value is unchanged.
- Freeze record: `docs/benchmark/E6_SCENARIO_CONTRACT.json` (seeds, configuration matrix, reporting rule, file
  hashes). Any later edit to the mechanics or parameters gets a new version and a new holdout contract.

**Code and tests.**
- `cdd_oran/envs/e6/sim.py`: `SurgeScenario`, `MistuneScenario`.
- `cdd_oran/envs/e6/config.py`: `E6Config.scenario` and the `surge_*`, `corr_*`, `mis_*` fields;
  `validate_scenario()`; `SCENARIO_HOLDOUT`.
- `cdd_oran/envs/e6/baselines.py`: `SMORestore`.
- `cdd_oran/envs/e6/env.py`: the `_on_oam` hook.
- `tests/test_e6_scenarios.py`: mechanics tests only.

**What the scenarios are.** The use cases are standard 3GPP SON cases. The numerical scenarios are E6 modelling
assumptions informed by standards and measurements; they are not standardised traces. `mix="none"` checks establish
the mechanics only, not controllability.

## 1. Purpose

Calm E6 (`scenario="base"`) stays the pre-declared control. It is a tuned regular grid under a gentle load, and there
the only headroom is the harm the xApps do themselves. The two scenarios add the conditions under which SON
functions are supposed to earn their keep:

| | 3GPP use case | Stress |
|---|---|---|
| **S1 `surge`** | Mobility Load Balancing (TR 36.902 cl. 4.6; LBO in TS 28.313 cl. 6.1.1.5) | static event: a localised traffic surge (simplified offered-load experiment) |
| **S2 `mistune`** | Mobility Robustness Optimisation (TR 36.902 cl. 4.5; TS 28.313 cl. 6.4.1.2) | too-late-HO hypothesis: an OAM rollout of a slow HO parameter set on the cells a recurrent high-speed road crosses |

Standard text the scenarios draw on (quotes checked against the documents):

- **TR 36.902 V9.3.1, cl. 4.6.1** (MLB objective): "Optimisation of cell reselection/handover parameters in order to
  cope with the unequal traffic load and to minimize the number of handovers and redirections needed to achieve the
  load balancing."
- **cl. 4.6.3** (MLB expected result): "some of the UEs at the cell border hand over to a less loaded cell".
- **cl. 4.5.1** (MRO): "Incorrect HO parameter settings can negatively affect user experience and wasted network
  resources by causing HO ping-pongs, HO failures and radio link failures (RLF)." An incorrect hysteresis "may be the
  reason for either ping-pong effect or prolonged connection to non-optimal cell."
- **cl. 4.5.2.1** (too-late HO): "If the UE mobility is more aggressive than what the HO parameter settings allow for,
  handover can be triggered when the signal strength of the source cell is already too low - leading to a RLF".
- **cl. 4.5.5.2**: MRO may optimise Hysteresis, Time to Trigger, Cell Individual Offset and cell-reselection
  parameters.
- **TS 28.313 V19.2.0, cl. 6.4.1.2, step 3**: "The MRO function detects handover issues (e.g. too late HO, too early
  HO and HO to a wrong cell) ... and acts to mitigate the HO issues by adjusting HO related parameters."

## 2. Common rules

- **Same tape for every arm.** Random elements are drawn once, at plant construction, from the new stream
  `(seed, "scenario", k)`, with k = 1 for S1 and k = 2 for S2. No other stream changes. After that the scenario is a
  deterministic function of time and never reads the RIC-controlled configuration.
  - A test runs no-arbitration against freeze, mix M4, **through the S2 rollout**, and checks that positions,
    corridor state, cluster and push time are identical.
- **Timing is relative to scoring.** Event times are fractions of the scored window:
  `t = warmup_s + frac × scored_s`. Validation requires every event to end by the end of scoring.
- **Validation.** `E6Config.validate_scenario()` runs in every scenario builder. It rejects:
  - an unknown scenario name;
  - a surge multiplier below 1, a radius ≤ 0 or a negative speed;
  - fractions outside [0, 1], or a surge that ends after scoring;
  - an invalid corridor share (0, above 1, or 0 UEs), length or speed;
  - an unknown location mode;
  - a mis-set Hys off the 0.5 dB grid or outside `HYS_RANGE`, or a TTT not in `TTT_SET_MS`.
- **Base is untouched.** `scenario="base"` builds no scenario object and executes the pre-scenario code path; the
  only change is an unused `oam_hook` attribute.
  - Pinned full-score references are recorded in the contract JSON (seed 11, M4, mixed, 120 + 600 s): medium noarb
    SVR 199.1 / freeze 190.48; high noarb 611.6 / freeze 576.62.
  - The medium values equal the values quoted before the scenarios existed.
  - Bit identity is shown for these configurations only; it is not inferred for all configurations.
- Provenance tags follow `config.py`: **[S]** sourced, **[A]** modelling assumption, **[V]** cited but not verified.
  "[A] informed by [S]" means the number comes from a source, but applying it to this mechanism is an assumption.

## 3. S1 `surge`: static event surge (MLB), a simplified offered-load experiment

**Mechanism.** An anchor point is drawn from the scenario stream. While the event is on, every UE whose wrap-around
distance to the event centre is ≤ `surge_radius_m` has its eMBB and BE **file-arrival rate** multiplied by
`1 + (surge_mult − 1) · e(t)`. `e(t)` is a trapezoid: 0, a linear ramp up, 1, a linear ramp down, 0.

**Simplifications, disclosed.**
- The UE count is fixed.
- LL arrivals are unchanged.
- No UEs arrive, and the slice mix does not change.

Only the data-file rate of UEs already inside the disk rises. This is **not** the observed composition of event
traffic (more users, a different mix, uplink-heavy). It is a simplified offered-load experiment on the MLB use case.

**The DEV scenario is a static venue.** A moving crowd is a HOLDOUT variant only.

| Parameter | DEV value | Tag | Source / rationale |
|---|---|---|---|
| `surge_mult` | 3.0 | **[A] informed by [S]** | Shafiq et al., ACM SIGMETRICS 2013, abstract: "both downlink traffic volume and the number of users increase by 3 times, during the crowded events as compared to their average on routine days". Their ×3 is an **aggregate** over cell sectors within about 1 mile of the venue, for users and volume together. Carrying it over as ×3 **per existing UE's file rate inside a 250 m disk** is an assumption: the population and the spatial scale do not match. It is not claimed to be conservative. |
| `surge_radius_m` | 250 m | [A] | Venue-scale assumption (= ISD/2). Not sourced. |
| `surge_loc` | `band` | [A] | Centre 150–250 m from a random macro site at a random bearing. This is the same rule that places E6's own hotspots. |
| `surge_onset_frac` | 0.20 | [A] | Leaves a clean pre-event scored period. |
| `surge_ramp_frac` | 0.10 | [A] | Compressed arrival and egress, as with E6's compressed diurnal ramp. |
| `surge_hold_frac` | 0.30 | [A] | Plateau. The event lasts 20–70 % of scoring. |
| `surge_speed_mps` | 0 | [A] | Static venue. |

**Exposure varies strongly across seeds, and it is frozen unfiltered.** On seeds 0–5 the disk contains 13–83 UEs. It
is largest when the event overlaps one of E6's hotspot clusters or picos, and that overlap dominates the realised
stress. Seed draws, locations, radius and duration are frozen. No seed is selected or dropped, and the radius is not
re-tuned after gate A.

## 4. S2 `mistune`: too-late-HO hypothesis (MRO) on a recurrent high-speed road

**Mechanism.**

1. **Corridor, present from t = 0.** A straight road section `corr_len_m` long, with its centre anchor and direction
   drawn from the scenario stream.
   - `round(corr_frac_ue · n_ue)` UEs, also drawn from the scenario stream, are placed on the road. They are outdoor,
     drive at `corr_speed_kmh` and **make a U-turn at each end** [A].
   - At 120 km/h a UE turns at least every 30 s. It crosses the same borders again and again and generates far more
     HO attempts than a one-way passing car would. This is a **recurrent-road stress**, not a representative
     drive-through (the no-rollout control in §7 shows the corridor-only event counts).
2. **Mis-set cluster** [A]. Every cell that is the strongest server (outdoor gain incl. shadowing, all cells on) at
   any point of the road, sampled every 10 m. This is a function of the seed only.
   - 8–10 of 24 cells on seeds 0–5, and 6 on seed 11, including 0–1 picos.
   - Applying the mis-set to *all* best-server road cells is an assumption about the rollout scope.
3. **OAM rollout at `mis_onset_frac`.** The Hys and TTT of every cluster cell are set once to
   `mis_hys_db` / `mis_ttt_ms`.

**Values.** The numeric pair is TR 36.839's slowest calibration set. That TR does **not** document an erroneous
rollout on such a road: the scenario is a hypothesis that this pair, applied here, produces too-late HOs.

| Parameter | DEV value | Tag | Source / rationale |
|---|---|---|---|
| `corr_speed_kmh` | 120 | [S] | TR 36.839 V11.1.0 Tab 5.2.4.1, UE speed "3 km/h, 120km/h, 30km/h, 60km/h" (the highest listed). ITU-R M.2410-0 cl. 4.11: "Vehicular: 10 km/h to 120 km/h". Caveat: M.2410 Table 3 supports vehicular only "up to 30 km/h" in *Dense Urban* eMBB, so this road is an urban expressway, not generic urban traffic. |
| `corr_frac_ue` | 0.10 | [A] | About the vehicular share of `mobility="mixed"` (12 %). |
| `corr_len_m` | 1000 m | [A] | 2 × ISD, crossing several borders. |
| `corr_loc` | `edge` | [A] | Road centre midway between two adjacent macro sites. |
| `mis_onset_frac` | 0.10 | [A] | Not at t = 0: an unscored warm-up would let MRO repair the mis-set before scoring starts. |
| `mis_ttt_ms` | 480 ms | [S] (numeric) | TR 36.839 Tab 5.3.2.1, **Set 1** (TTT 480 ms, A3 offset 3 dB). |
| `mis_hys_db` | 3.0 dB | **[A] mapping** | Set 1's A3 *offset* of 3 dB is carried by E6's *Hys* (E6 has no separate offset knob). Both enter the A3 entering condition the same way, but this mapping changes **repairability**: the frozen MRO xApp never lowers Hys (too-late → CIO += 1 up to +6 dB; only then TTT one step down). The mis-set Hys can therefore only be compensated through CIO. The calm default is Hys 2 dB / TTT 320 ms. |

### 4.1 OAM push vs RIC history, and the SMO reference

The rollout is an **exogenous OAM change**. Freeze cannot reject it, and it bypasses churn accounting (8–10 of 24
cells). S2 therefore tests **recovery from an exogenous misconfiguration**, not only conflict mitigation. Two rules
keep that honest.

**1. RIC semantics** (implemented; `MistuneScenario.oam_set` → `plant.oam_hook` → `E6Env._on_oam`):
- An OAM write (the push, or an SMO restore) becomes the knob's **last-known-good** (`env.prev_val`). A RIC ROLLBACK
  therefore **cannot undo** an OAM write; it is a no-op until a RIC change happens.
- The next RIC change on that knob records the OAM value as its prior. Rolling back that RIC change restores the OAM
  value.
- An OAM write is **not** a RIC change:
  - it does not set actuator dwell (`last_change`), so the first post-push RIC write on that knob is not blocked;
  - it does not count toward `changes` (churn);
  - it produces no request and no NACK.
- The push happens once. A later RIC or xApp write to the knob is never re-overwritten.
- Tests cover each of these rules.

**2. SMO restore reference** (`baselines.SMORestore`), outside the near-RT RIC. It models post-rollout verification
with fallback:
- **Wrapping.** It wraps any arbiter (default: accept-all) and passes that arbiter's decisions through unchanged.
- **Alarm.** It watches the delivered mobility KPM reports whose window starts at or after the push. When the cluster
  too-late ratio `tl / (ho_att + tl)` exceeds **2 %** on **2 consecutive 30 s reports**, with at least 2 too-late
  and 3 events, it acts.
- **Action.** It restores the pre-push Hys/TTT on the whole cluster, **once**, as an OAM write. The restore is
  likewise not RIC-rollbackable.
- **Where the thresholds come from.** They are declared a priori as the frozen MRO xApp's own nominal too-late target
  (`xapps.MRO`, 0.02), its 60 s window, and its minimum-evidence rule. They were not tuned.
- **Disclosure.** The mechanical sanity run in §7 (corridor-only cluster ratio ≈ 0.65–0.8 %, mis-set ≈ 4–7 %) was
  seen before this reference was written. The threshold is nonetheless the pre-existing MRO constant, not a value
  picked from those numbers.
- **Privilege.** The SMO knows only what an SMO knows about its own rollout: which cells it pushed and their previous
  values.
- **Mechanics** (seed 11, mix none, 120 + 600 s): the alarm fires at 361 s (medium) and 241 s (high), after a push at
  180 s.

S2 claims must therefore beat **freeze, the tuned-static configuration and SMO restore** (alone and wrapped around the
method). Otherwise they must be stated explicitly as holding only for a WG3-only RIC without SMO fallback.

## 5. DEV vs reserved HOLDOUT variants

DEV runs use the defaults, redrawn per seed from the scenario stream. HOLDOUT variants live in
`config.SCENARIO_HOLDOUT`. Each changes one factor and passes validation. None has been run, not even for mechanics.

| Scenario | Variant | Override | Factor | Tag |
|---|---|---|---|---|
| surge | H-time | onset 0.45, ramp 0.05 | later, sharper event | [A] |
| surge | H-loc | `surge_loc="vertex"` | centre on a 3-site corner | [A] |
| surge | H-int | `surge_mult=4.0` | intensity | [A]: above the aggregate DL ×3; inside the aggregate UL range 4–8 |
| surge | H-move | `surge_speed_mps=0.833` | moving crowd at 3 km/h | speed [S] (TR 36.839 Tab 5.2.4.1; M.2410 cl. 4.11); mechanism [A] |
| mistune | H-time | `mis_onset_frac=0.35` | timing | [A] |
| mistune | H-loc | `corr_loc="vertex"` | road centred on a 3-site corner | [A] |
| mistune | H-int | Hys 5 dB, TTT 640 ms | harsher late mis-set | [A]: E6 actuator maxima |
| mistune | H-dir | Hys 0 dB, TTT 40 ms | early direction (too-early / ping-pong) | **[A] approximation**. TTT 40 ms is Set 5's; Set 5's A3 offset of −1 dB cannot be expressed as Hys ≥ 0 and is clipped to 0 dB. **This is not TR 36.839 Set 5.** |

## 6. Intentionally not tuned

- No parameter was chosen or changed after looking at any xApp, arbiter, oracle or baseline outcome in S1/S2. There
  have been no noarb/freeze/subset comparisons in these scenarios.
- The xApps, KPM, scheduler, radio, SLA definition and plant constants are the calm-E6 ones.
- HOLDOUT values are declared, not explored.
- If gate A fails, the negative result is kept. Changing intensity, road placement, repairability or the SMO rule
  starts a new version with a new holdout contract.

## 7. Mechanical sanity (plant only, `mix="none"`; not an arbitration outcome)

Warm-up 120 s, scored 600 s, mobility `mixed`. These are plant statistics under a single fixed arm.

**S1 surge vs base** (same seed):

| seed / load | Realised in-region eMBB/BE offered load (plateau) | Out-of-region | Pre-onset in-region | Utilisation of cells serving the region | Network mean utilisation | Violation fraction |
|---|---|---|---|---|---|---|
| 0 / medium | ×3.01 | ×0.995 | ×1.000 | 0.39 → 0.79 | 0.37 → 0.60 | 2.8 % → 6.7 % |
| 1 / high | ×3.09 | ×1.001 | ×1.000 | 0.70 → 0.89 | 0.64 → 0.73 | 11.9 % → 15.3 % |

**S2 mistune vs the corridor-only, no-rollout control** (same seed and corridor; cluster counts over the 540 s after
onset):

| seed / load | Cluster | Too-late HO | Too-early | Ping-pong | HO attempts | RLF (all cells) | Violation fraction |
|---|---|---|---|---|---|---|---|
| 0 / medium | 9 | 52 → 259 | 0 → 0 | 151 → 57 | 7930 → 6210 | 116 → 289 | 5.9 % → 6.6 % |
| 1 / high | 10 | 64 → 432 | 0 → 0 | 691 → 419 | 7981 → 6105 | 102 → 467 | 16.2 % → 16.7 % |

- The corridor alone already produces 52–64 too-late HOs and about 8000 HO attempts in the cluster per 540 s. This is
  the recurrent-road stress from the U-turns.
- The mis-set adds the TR 36.902 cl. 4.5.2.1 too-late signature. Neither scenario collapses the network.

## 8. Open items

- The arbiter contract must state the S2 claim scope (§4.1): WG3-only RIC vs RIC plus SMO fallback.
- The TEST episode length and TEST seeds are to be fixed in the contract before any TEST run.
- TS 32.500 / TS 32.522 are cited in the E6 spec as context and were not re-checked [V].

## 9. Sources (checked for this document)

- 3GPP TR 36.902 V9.3.1 (ETSI TR 136 902, 2011-05), cl. 4.5 and 4.6.
- 3GPP TS 28.313 V19.2.0 (2025-03), cl. 6.1.1.5 and 6.4.1.2.
- 3GPP TR 36.839 V11.1.0 (2012-12), Tab 5.2.4.1, Tab 5.3.2.1, §5.5.
- ITU-R M.2410-0 (11/2017), cl. 4.11 and Table 3.
- M. Z. Shafiq et al., "A first look at cellular network performance during crowded events", ACM SIGMETRICS 2013
  (abstract).
