# xTRUCE-plant conflict screen protocol (Gate A v2, XTS-v1)

Status: **DRAFT, FREEZE-READY, NOT FROZEN (2026-09-29).** This screen runs only if the E6-P screen
(`E6P_SCREEN_PROTOCOL.md`) ends without a PASS. To freeze it: commit this file together with the plant
(`cdd_oran/envs/xtruce/`, `XTRUCE_SIM_SPEC.md`) at one commit (`xts_commit`). Then set `FROZEN_SHA256` (this file) and
`FROZEN_PLANT_SHA256` (the plant files) in `scratchpad/e6_dev/xtruce_screen.py` and register the seed ranges of §3 in
`SEED_REGISTRY.json`. Before that, the driver runs only with `--smoke`. **Nothing here is tuned after any
outcome.** A code defect found by a mechanism test may be fixed so that the mechanics match this text. A *value* may
not change. If a frozen value makes a precondition fail, the affected (pair, stratum) is reported "not screenable
under XTS-v1". Any value change makes a new version (XTS-v2) with fresh seeds, and the v1 outcome is published as is.
This is a DEV screen. It cannot confirm an edge. It can only kill a pair or pass it to a separately preregistered
confirmation on 190200-190399. Each stage needs the user's explicit go.

Purpose: the same three questions as E6-P, asked on an independent, published plant (the xTRUCE re-implementation,
arXiv:2608.28532v2). Per deployed xApp set: (1) Is there a material protection loss at matched energy when every
request is accepted? (2) Can a budgeted O-RAN WG3 arbiter with perfect knowledge recover most of it? (3) Does that
recovery need state-dependent decisions, beyond the best static rule? (4) Does the oracle's advantage survive a
re-drawn future?

## 1. Common configuration
- Plant: `XConfig()` defaults (`XTRUCE_SIM_SPEC.md` §1) in every arm, with only these changes: `traffic_bps` and
  `hallucination` per stratum (§3), `xapps` per pair (§2), and `log=False` (logging only). `phys="clip"`,
  `all_hard=False`, `t_cfg=10`, `es_cap_w=53`, `qos_margin=0.30` are the defaults and are asserted by the driver.
- Episode: warm-up W = 30 epochs (t = 0..29, three configuration periods), then S = 600 scored epochs
  (t = 30..629, the paper's Fig. 3 length per level). Every KPI below is summed over scored epochs only. Every arm,
  the oracle included, acts from t = 0 (see §5, oracle).
- Deployment order of the xApps is the plant's `XAPP_NAMES` order restricted to the pair (QoS, ES, IC, LB). Under
  accept-all the last writer wins in that order, so on `pcap` ES overrides QoS and IC overrides both.
- WG3 action set: accept / reject / modify (half) / lock. The plant has no rollback primitive, so rollback is **not**
  in the action set (a declared difference from E6-P). Arbiters make no writes of their own. The paper's Direct,
  Clipping and xTRUCE arbiters are exceptions: they are target-to-action maps by definition and are reported as such.

## 2. Pairs (deployed xApp sets). The energy side is always ES, because the metric is taken at matched energy.
| pair | `xapps` | conflict class (spec §4 map) | role |
|---|---|---|---|
| X1 | (QoS, ES) | direct on `pcap`; indirect cap→PSD; indirect sleep→association | screened (see below) |
| X2 | (QoS, ES, IC) | + direct on `pcap` (IC lowers a single cell) | secondary |
| X3 | (QoS, ES, LB) | + indirect: LB `assoc` vs QoS, LB vs ES sleep | secondary |
| **X4 (primary)** | (QoS, ES, IC, LB) | all of the above | **primary**: the paper's own four-agent deployment (Sec. IV-D) |

The primary is X4 because it is the deployment the paper evaluates. **Disclosure:** X4 is also the only deployment
of the seeds 0-2 sanity run that showed an accept-all loss (§10).

X1 pairs the energy saver with the protection xApp; ES alone does nothing for
protection. It is the E6-P-P1 analogue (energy saver × protection guarantee) and is screened like every other pair.

## 3. Strata and seeds
Strata (4): load ∈ {`L6`, `L1`} × hallucination ∈ {`h0`, `h50`}, with `stratum = 2·load_idx + h_idx`:
0 L6-h0, 1 L6-h50, 2 L1-h0, 3 L1-h50.
- `L6`: `traffic_bps = 6e6`, the paper's Table I load. Every cell's demand load is above 1, so ES acts only through
  the 53-W cap.
- `L1`: `traffic_bps = 1e6`, a light load at which ES's sleep rule (demand load < 0.30, TR 38.864 Annex A) is
  active. The spec §4 sleep→association and LB↔ES conflicts exist only here. M6 must show that sleeping happens.
  1 Mb/s is the value of the disclosed sanity run (§10). It was not chosen from the outcomes of this screen.
- `h0`: `hallucination = 0`. `h50`: `hallucination = 0.5`, the plant's `fig3` preset (paper Fig. 3 mid-sweep). Each
  target is corrupted with probability 0.5, by a factor of up to 10^±1.

Fresh DEV range **190000-190399**. It is disjoint from every range in `SEED_REGISTRY.json` and from the seeds 0-2 of the
sanity run. The spec requires seeds ≥ 1000. **The range must be registered (new world `XTRUCE`) before any
non-smoke run.** Mapping: `seed = 190000 + 50·stratum + j`.

| use | j | seeds |
|---|---|---|
| screen candidates | 0-43 | the stratum's **screen seeds** are the first **N = 40** candidates (in j order) that pass `XEnv(cfg, seed).floors_reachable()`, the paper's seed screen [P, p. 8; operationalisation A]. This is a construction-only property: it reads no outcome. If fewer than 40 of the 44 candidates pass, the stratum runs with those that pass, and the number is reported. It is not refilled. |
| mechanism tests M4, M6, M7 | 44-48 | 5 seeds per stratum (no seed screen) |
| unit mechanism tests M1-M3, M5, M8 | - | 190044 (stratum 0, j = 44) |
| oracle timing pilot | 49 of stratum 3 | 190199, X4. Wall time only; scores are not recorded. |
| reserved: confirmation of a surviving pair | - | **190200-190399**, untouched by this protocol |

Paired arms reuse the episode seed. The fade, traffic and xApp streams do not depend on the action, so every arm on
a seed sees the same channel, arrival and corruption tape (spec §3; checked by M5). The freeze arm is identical for
all pairs (M5c), so it is run once per (seed, stratum) and shared.

RNG tags (new, disjoint from the registry's `rng_stream_tags`; to be registered): bootstrap
`default_rng([7019, pair_idx, stratum])` with pair_idx = 1-4 for X1-X4; oracle search draws
`default_rng([seed, t, 7020])`; re-drawn futures use `env.copy(reseed=1)`, which the plant keys by
(seed, stream, 0x7C5E, 1, t).

## 4. Quantities (per episode, scored epochs only)
- `pv`: protected violated UE-s, Σ_t Σ_{u protected} 1[R_u(t) < R_min]. This is the paper's e1 on capacity (eq. 14),
  the plant's `prot_below`. `pu` = protected UE-s = 3 · 600 = 1800 per episode.
- `V` = 3600 · Σpv / Σpu, in violated UE-s per protected UE-hour (PSVR units, as in E6-P), pooled over the
  stratum's seeds. 36 ≡ 1 % of protected UE-time.
- `E`: scored energy (J), Σ_t Σ_b E_b(t) · τ (eq. 16).
- `thr_np`: mean delivered throughput of the non-protected UEs (bit/s per UE).
- `oper`: epochs in which the executed action broke an operator change limit e2-e5 (eq. 20: per-RB share change
  0.25, per-RB power change 0.25 W, ≤ 1 cell (de)activated, ≤ 3 UEs re-associated), relative to the previous
  executed action. This is the plant's `oper_viol`.
- Reported only: `pv_thr` (delivered-throughput variant of the floor), `phys` (configured action outside c1-c4, clipped
  by the plant), sleep cell-seconds, applied knob changes, conflict epochs.

## 5. Arms (per pair, per screen seed)
| # | arm | definition |
|---|---|---|
| 1 | freeze | `RejectAll`: the initial equal-share, full-power action, all cells on, strongest-cell association |
| 2 | each xApp alone | `Subset((x,))` for every x in the pair (all stay deployed and proposing) |
| 3 | accept-all (AA) | every request, in deployment order, last writer wins |
| 4 | static subsets | every other non-empty proper `Subset(S)` (X4: 10, X2/X3: 3, X1: 0) |
| 5 | per-knob priority | `StaticPriority(order)`, every order of the pair's xApps (X4: 24, X2/X3: 6, X1: 2) |
| 6 | cell-priority lock | `CellPriorityLock(order, lease=60)`, every order. 60 epochs = the E6-P lease (60 s). |
| 7 | Direct | paper benchmark 1 (`arbiters.Direct`), run on the `phys="clip"` plant like every arm |
| 8 | Clipping | paper benchmark 2 (`arbiters.Clipping`) |
| 9 | per-cell hindsight static | For each of the 4 cells, a fixed xApp subset held for the whole episode. It is chosen per seed by one pass of coordinate descent from accept-all, over cells in index order, over all 2^n keep-sets per cell, minimising the episode's `V` among candidates that are energy-matched and pass the guardrails against that seed's AA. A request belongs to the cell of its knob (`knob_cells`; for `assoc`, the UE's current cell). Privileged, not deployable. **Stage 3 only.** |
| 10 | budgeted WG3 oracle | below. **Stage 2 only.** |
| 11 | xTRUCE (reported comparator) | `XTruce()` (SLSQP, maxiter 100). It is **not** a reference, **not** in R_static, **not** in any rollout. Run on the first 10 screen seeds of every (pair, stratum) (cost: 70-300 ms per epoch). R_xtruce is reported on those 10 seeds against the same seeds' AA and V_ref. |

Oracle details (arm 10), mirroring E6-P arm 10 (`wg3_oracle.py` mechanics):
- Decision points: every D = 10 epochs, at the configuration epochs t = 0, 10, …, 620 (63 per episode), starting at
  t = 0. Reason (declared before any outcome): ES writes its `pcap` cut in the first epochs and does not re-request it
  once it holds. An oracle that started after warm-up, like the E6-P one, could not arbitrate the main ES↔QoS
  conflict, while the static arms act from t = 0.
- A plan is a mode per (cell × xApp) in {accept, reject, half, lock}. Each request is decided by the mode of its
  (cell, xApp). `half` on a fast knob sets cur + (prop − cur)/2 (`arbiters.half`). On a configuration knob
  (sleep / assoc, not divisible) it alternates accept / reject per knob, with phase (knob index mod 2), which is the
  E6-P slew rule for single-quantum steps. `lock` rejects the request and calls `env.lock(knob, D)`, which blocks
  that knob for **every** xApp for D epochs.
- Search per decision point, budget **29 rollouts** (= E6-P's 3 + 6 + 10·2):
  - stage 1: the best of accept-all, reject-all, the incumbent plan and n_glob = 6 random network-wide plans;
  - stage 2: one coordinate-descent pass over the 4 cells in random order, with n_loc = 5 random local alternatives
    per cell.
  Each rollout holds the plan for H = 40 epochs (4 configuration periods; E6-P H/D = 4.5) on the **true** tape
  (`env.copy()`), truncated at the episode end.
- Objective over the horizon (every epoch in it counts): protected violated UE-s; lexicographic tie-break by larger
  non-protected delivered bits.
- Energy constraint: a plan is admissible only if the rollout's cumulative scored energy at the end of the horizon
  ≤ `E_f(t) − 0.90·(E_f(t) − E_ES(t))`. `E_f` and `E_ES` are the per-epoch cumulative scored energies of the freeze
  and ES-alone arms on the same tape (privileged). A trial replaces the incumbent only if it is admissible. If no
  plan is admissible, accept-all is used.
- Guardrail constraint (in addition to the energy constraint and the churn cap; amendment of 2026-09-29, see §10): a
  plan is admissible only if, over its H-epoch rollout, both hold against the accept-all-plan rollout from the same
  state on the same tape (the stage-1 accept-all candidate, which every decision already rolls out; no extra rollout):
  (a) non-protected delivered bits ≥ 0.90 × the accept-all rollout's; (b) epochs breaking an operator change limit
  e2-e5 (`oper`, §4) ≤ the accept-all rollout's + ⌈0.01 · H⌉ (= 1 for H = 40). Every epoch of the rollout counts, as
  in the objective. The accept-all plan is always admissible on these two conditions, so the fallback is unchanged.
  The two conditions are the per-rollout forms of G1 and G2 (§7). They do not change G1, G2 or eligibility (§7).
- Churn parity: the oracle's applied knob changes over the episode are capped at the AA arm's count on the same tape.
  Once the cap is reached, every further request is rejected. Rollouts see the same cap.
- "Same rights": same action class and actuator semantics as every request-level arm. Its only privilege is
  lookahead on the true tape, the same kind that arm 9 has.

## 6. Mechanism preconditions (Stage 0). All must pass before any screen seed is run.
| id | test | pass rule |
|---|---|---|
| T0 | Plant tests | `tests/test_xtruce_env.py` passes at `xts_commit` (local). |
| M1 | **Power cap acts as eq. (16)-(17) predict.** Seed 190044, state after W epochs of AA. For each cell b, apply `pcap_b` = 3 W (the ES target) and then P_max, with fast actions and channel frozen. | (a) executed Σ_{u∈b} p_u ≤ pcap_b + 1e-9. (b) E_b = P_cir + P_b within 1e-9 W. (c) For every UE not served by b, the realised interference change on every RB equals ΔP_b/K · G_{b,u,k} within 1e-9 relative. (d) Lowering pcap_b lowers no rate of another cell's UE. |
| M2 | **Rate eq. (14).** Same snapshot. | The realised R_u equals x_u W Σ_k log2(1 + G p_u /(K x_u N)), recomputed independently, within 1e-9 relative, for every UE. Doubling one UE's configured share at fixed cell power changes no rate in any other cell. |
| M3 | **Sleep / wake.** Same snapshot. | A slept cell has E_b = P_sleep, no UE served, and every moved UE on its strongest active cell. Every receiving cell's configured Σx is unchanged (work-conserving move). Waking restores the original association. The last active cell refuses sleep. A sleep write at a non-configuration epoch is dropped. |
| M4 | **Each xApp alone improves its own KPI vs freeze.** Mechanism seeds (5 per stratum), deployed as X4. Own KPIs: ES → E; QoS → V; IC → mean Σ_b I_out,b; LB → mean over epochs of max_b demand load. | Evaluated per xApp only in strata where it acts (accepted requests ≥ 1 per seed on average) and its freeze KPI is material. The materiality rule is E6-P 8a built in: QoS freeze V ≥ 36; LB freeze mean max demand load > `lb_cap` 0.80; energy and IC always material. There, the pooled own KPI must improve and at least 4 of 5 seeds must improve. An xApp with fewer than 2 acting, material strata is *inert*. A pair containing a failing or inert xApp is not screenable (reported, not fixed). The screen proceeds for the other pairs. |
| M5 | **Determinism and common random numbers.** Seed 190044. | (a) The same (cfg, seed) gives identical summaries twice. (b) `copy()` replays 50 epochs of AA exactly. (c) The freeze episode is bit-identical under all four deployments and both h levels (energy and pv per epoch), which justifies the shared freeze arm. (d) `copy(reseed=1)` keeps layout, shadowing and the protected set, and changes the arrivals. |
| M6 | **Low load is a sleep regime.** Mechanism seeds, ES alone. | In each L1 stratum, sleep cell-seconds > 0 in ≥ 4 of 5 seeds. If this fails, that stratum is not screenable. The L6 values are reported. |
| M7 | **Conflict incidence (diagnostic).** Mechanism seeds, AA, per (pair, stratum). | The fraction of scored epochs with ≥ 1 `conflict_groups` group is reported. **No exclusion.** Reason: the main ES↔QoS conflict is implicit. ES writes `pcap` in the first epochs and does not re-request it, so there may be no same-epoch co-request in the scored window even though the conflict is live. Criterion 1 is the test for whether a conflict matters. |
| M8 | **Oracle plumbing.** Seed 190044. | (a) An accept-all-plan rollout over H from a mid-episode state gives the same pv and energy as the true AA continuation. (b) An all-reject plan gives the same continuation as `RejectAll`. (c) A `lock` decision blocks a later request of another xApp on the same knob. In (a) and (b) the rollout's non-protected bits and e2-e5 count (the §5 guardrail quantities) must also match. |

Outcome visibility: M4, M6 and M7 reveal single-xApp, freeze and AA outcomes on mechanism seeds only. No co-deployment
criterion is computed, and screen seeds are untouched until Stage 0 passes.

## 7. Gate A v2 (per pair, per screenable stratum; pooled over the stratum's N screen seeds)
- `AA` = accept-all.
- Energy-matched: `E_f − E_a ≥ 0.90·(E_f − E_ES)`, where ES = the ES-alone arm (X = 0.90, as E6-P).
- Screenable: ES alone saves ≥ 1 % of freeze energy, and M4 or M6 did not exclude the stratum.
- Guardrails vs AA: (G1) `thr_np ≥ 0.90 · thr_np(AA)`; (G2, change-limit / power-change guardrail)
  `oper ≤ 1.10 · oper(AA) + 0.01 · (N · S)`. The absolute slack of 1 % of scored epochs keeps G2 meaningful when
  AA's count is near 0. `phys` is reported but is **not** a guardrail: under AA it is set in about 97 % of epochs,
  only because QoS's 0.02 dead-band leaves configured shares summing slightly above 1, which the plant then
  rescales (§11).
- Eligible arm: energy-matched and passing G1 and G2.
- **V_ref** = min V over the single-xApp arms (group 2), energy and guardrails ignored: the best single xApp,
  exactly as E6-P §7. (A draft used the best *eligible* proper subset; that makes V_ref itself a static arm, so
  R_static ≥ 1 and criterion 3 would demand R_or ≥ 1.10. Replaced before freeze and before any screen outcome.)
- Λ = (V_AA − V_ref)/V_ref. Recovery of an arm is R_a = (V_AA − V_a)/(V_AA − V_ref).
- **R_static** = max R over eligible arms of groups 2, 4, 5, 6, 7, 8, 9 (every subset, every static priority order,
  every cell-lock order, Direct, Clipping, per-cell hindsight). A single xApp that is not energy-matched is not
  eligible, so V_ref usually cannot enter R_static.
- **R_or** = R of the oracle if the oracle is eligible, else 0.
- Diagnostic only: Λ_subset = (V_AA − V_sub)/V_sub, where V_sub is the best eligible (energy-matched,
  guardrail-passing) proper subset.

Criteria, all in the **same** stratum:
1. **Loss:** screenable, AA energy-matched, Λ ≥ 0.15, `V_AA − V_ref ≥ 36`, and the paired-seed bootstrap (10 000
   resamples) one-sided 90 % lower bound of `V_AA − V_ref` > 0. In the bootstrap, V_ref is re-minimised per resample
   over the reference arms that are eligible at the point estimate.
2. **Recoverable:** R_or ≥ 0.50.
3. **State-dependent headroom:** R_or − R_static ≥ 0.10.
4. **Oracle sign reliability:** at every decision point where the oracle's plan is not accept-all, re-score the plan
   and accept-all on `env.copy(reseed=1)` (same state, re-drawn fading, arrivals and corruption) over the same
   horizon. ρ_sign = fraction of deviations whose lexicographic advantage sign over AA
   (protected violated UE-s, then non-protected bits) is the same on the re-drawn copy, pooled over the stratum.
   Ties on both copies count as kept. Criterion: ρ_sign ≥ 0.70, with ≥ 1 deviation.

**Absolute floor, scaled from E6-P.** E6-P's floor `V_AA − V_ref ≥ 36` is in PSVR units (violated UE-s per protected
UE-hour). It is 1 % of protected UE-time, not a count per 720-s episode. The scale-free quantity is kept: **36 per
protected UE-hour = 1 % of protected UE-time**. With 3 protected UEs × 600 scored epochs this is **18 violated UE-s
per seed-episode**, or 720 UE-s pooled over 40 seeds. Reasons: (i) it is the same materiality as E6-P, so
verdicts are comparable across plants. (ii) 1 % is one fifth of the 5 % unavailability that the E6-P 95 % availability
rule allows, and the same value that `qos_margin` [A\*] was checked against (QoS alone ≤ 5 %). (iii) With only 3
protected UEs per seed, a smaller floor would be met by single short outages of one UE: 18 UE-s is at least 6 s of
all three protected UEs, or 18 s of one. **Disclosure:** in the seeds 0-2 sanity run, X4's accept-all vs
ES + QoS gap at L6-h0 was 33 of 2700 protected UE-s = 1.2 %, just above this floor. The floor is E6-P's frozen value
in the same units. It was not set with that number in view, and it is not moved because of it.

Verdict per pair (as E6-P):
- **PASS** if at least one stratum satisfies 1-4.
- **NOISE STOP** if 1-3 hold somewhere but 4 fails in every such stratum.
- **DEAD (no loss)** if no screenable stratum satisfies 1.
- **DEAD (not recoverable)** if every stratum satisfying 1 fails 2.
- **NO-EDGE STOP** if 1 and 2 hold somewhere but 3 fails in every such stratum.
- **NOT SCREENABLE** if M4 excluded the pair, or M6 excluded every stratum.

If AA is not energy-matched in a stratum, criterion 1 fails there, and AA's energy shortfall (1 − retention) is
reported as an energy-side co-deployment loss. Criteria 2 and 3 are decided on point estimates (DEV screen).
Paired-bootstrap 5-95 % intervals are reported for Λ, R_or, R_or − R_static and R_xtruce. All four strata and all arms
are reported for every pair, whatever the verdict. The implication is a *screen* outcome only. A PASS licenses a
separately preregistered confirmation on 190200-190399. It is not a claim.

## 8. Stages and cost (each needs an explicit go; Stage 0c replaces the oracle estimate with a measurement)
Measured with `xtruce_screen.py estimate` (laptop, 1 thread, disclosed DEV seed 0, X4, L1-h50, full length, no scores
printed): 0.25 s per 630-epoch episode, xTRUCE 85 ms per epoch (the spec quotes 70-300), and one full-budget oracle
decision 0.46 s (31 rollouts) → ≈ 32 s per oracle episode (63 decisions + 3 reference episodes).
| stage | content | episodes | CPU-h (laptop) |
|---|---|---|---|
| 0 | T0, M1-M3, M5, M8 (unit-level) + M4/M6/M7: 4 strata × 5 seeds × (freeze + 4 singles + AA × 4 pairs) | 180 | < 0.05 |
| 0c | oracle timing pilot, 190199, X4 (scores not recorded) | 1 oracle episode | < 0.02 |
| 1 | arms 1-8, all pairs, 4 × 40 seed-strata; freeze shared (X1 9, X2 21, X3 21, X4 65 arms + freeze = 117 per seed-stratum) | 18 720 | 1.3 |
| 1x | arm 11 (xTRUCE), 4 pairs × 4 strata × 10 seeds | 160 | 2.4 (8.4 at 300 ms per epoch) |
| 2 | oracle, only in (pair, stratum) that pass criterion 1 | ≤ 16 × 40 = 640 | 0.35 per (pair, stratum); ≤ 5.6 if all 16 pass |
| 3 | arm 9, only where 1 and 2 pass | per seed 3 + 4 × (2^n − 1) (X4 63, X2/X3 31, X1 15) | 0.17 per X4 (pair, stratum); ≤ 1.5 if all 16 |
Total: ≈ 4-6 CPU-h in a typical case (1-4 cells reach Stage 2) and ≤ 17 CPU-h if every cell reaches Stage 3 with slow
xTRUCE epochs. Kaggle CPUs are ~1.5-2× slower per thread: plan for ≤ 35 CPU-h, i.e. ≤ 3 h on 3 kernels × 4 shards.

The stages are gated in sequence because the criteria are conjunctive. Skipping the oracle where criterion 1 failed
cannot change a verdict. Drivers use timeouts, and background processes are cleaned up after every stage.

## 9. Frozen at doc commit
This document's plant configuration, episode length, pairs, primary, strata, seed mapping, N, seed screen, arm
definitions, oracle constants (D 10, H 40, n_glob 6, n_loc 5, modes, t = 0 start, energy constraint, guardrail
constraint 0.90 / ⌈0.01 · H⌉, churn cap, objective), guardrails (0.90 throughput, 1.10 × + 1 % change-limit), X = 0.90, 1 % screenability, V_ref rule,
Gate A v2 thresholds (Λ 0.15, 36 per protected UE-hour, R_or 0.50, headroom 0.10, ρ_sign 0.70), bootstrap settings,
precondition pass rules and verdict rules. `XTRUCE_SIM_SPEC.md` §1 values (as `XConfig()` defaults at `xts_commit`).
Not frozen and not outcome-relevant: code structure, logging, plotting, sharding.

## 10. Disclosure (what was seen before this protocol was written)
- **Sanity run, seeds 0-2 × 300 epochs** (`XTRUCE_SIM_SPEC.md` §5). Freeze, each xApp alone, ES + QoS and accept-all
  of the four-xApp deployment were seen at 6 and 1 Mb/s, h = 0. At 6 Mb/s: AA 126 protected violated UE-s vs
  ES + QoS 93 (both energy-matched). At 1 Mb/s: AA 90 vs ES + QoS 125. The paper benchmarks Direct, Clipping and
  xTRUCE were run on 3 seeds × 200 epochs with `phys="none"`, and xTRUCE on 20 epochs of seed 0.
- **`qos_margin = 0.30` [A\*]** was chosen among 0.1 / 0.3 / 0.5 on those seeds (criterion: QoS alone ≤ 5 %
  protected violations), with freeze, QoS-alone and AA outputs visible.
- **Two xApp-rule changes after looking at seeds 0-2:** (a) association moves became work-conserving (the old rule
  left the source cell's share unused, so LB looked like an energy saver); (b) QoS moved from boosting protected UEs
  to P_rb per RB to a uniform PSD (the boost let protected UEs take the whole capped budget).
- While this protocol was being written (2026-09-29), the author also saw the following. Seed 0, 630 epochs, AA of
  X4 in all four strata: conflict-epoch counts, request counts, `oper` / `phys` counts and sleep cell-seconds, but not
  pv or energy. The `phys` driver (Σx > 1 from the QoS dead-band). Per-epoch speed of Direct, Clipping and xTRUCE.
  Also the `floors_reachable` pass rate on construction-only seeds 1000-1199 (194/200). These informed the
  decisions to start the oracle at t = 0, to drop `phys` as a guardrail, to use N = 40 from 44 candidates, and to run
  xTRUCE on 10 seeds. No criterion quantity (V, Λ, R) was computed on any seed for this protocol.
- The xTRUCE plant sanity numbers above are the only co-deployment outcomes seen. Seeds 190000-190399 are untouched.
- **Oracle guardrail amendment (2026-09-29, before freeze and before any run of this screen beyond the plumbing smoke
  on seeds 0-2).** The §5 guardrail constraint was added after the E6-P screen (`E6P_SCREEN_PROTOCOL.md`) ended DEAD.
  There, in every cell that passed criterion 1, the budgeted oracle recovered much raw loss (R 0.38-0.88) but was
  ineligible: it broke a guardrail (RLF 1.22-1.39 × accept-all, LL violated UE-s 1.107 × accept-all, margin 1.10),
  because its objective ignored the guardrails. It also over-saved energy (retention 1.14-1.36), and its ρ_sign was 0.55-0.66.
  An oracle that can break the guardrails that decide its eligibility does not measure what criterion 2 asks.
  The amendment makes the oracle's admissibility test check G1/G2-type guardrails against its own accept-all rollout.
  No xTRUCE-plant outcome informed it. The seeds 0-2 smoke runs are plumbing only (W 10 + S 40 epochs, 2 seeds per
  stratum, oracle H 10 with 1 + 1 random plans). Their summary prints smoke-length numbers, but none were used for
  this or any other decision. The change adds only the constraint of §5, its reuse of the anchor
  rollout, and the matching M8 plumbing check (§6). Nothing else changes: the energy constraint, churn cap,
  objective, search budget, D, H, eligibility, G1/G2, criteria and verdict rules are as before. The energy constraint
  stays one-sided. Over-saving energy is still admissible, and retention above 1 is reported, not penalised.

## 11. Known plant issues carried into the screen (reported, not fixed here)
- `phys_viol` is dominated by QoS's dead-band (`xapps.py` QoS request loop): configured Σx per cell reaches ≈ 1.02,
  flagged by `env.actuate` at tolerance 1e-6 and rescaled by `phys="clip"`. Physics is unaffected, but the flag is
  not a usable guardrail.
- The paper's Direct and Clipping have the same outcomes on a `phys="clip"` plant. Both are run (their `phys`
  counts differ), and both enter R_static.
