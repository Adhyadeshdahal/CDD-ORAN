# E6 DEV collection contract: `e6-collect-v3`

Status: DRAFT for the pilot, 2026-09-28. It answers SOL_BUILD_REVIEW.md items 1, 3 (collection side) and 4, plus
the MRO/ES support rule. Every number below is fixed before any E6 policy-data outcome is inspected. Any change
bumps the version (`e6-collect-v4`, ...) and re-runs the pilot on the same pilot seeds.

- Code: `cdd_oran/decision/collect.py` (`COLLECT_VERSION`, `collect_episode_v3`, `audit_episode`,
  `continuation_check`, `dev_seed`).
- Support rule: `cdd_oran/decision/effect_model.py` (`support_counts`, `SupportRule`, `EffectWM(support=...)`).
- Gate: `cdd_oran/decision/gate.py` (`ConformalGate`, default `method="episode_max"`).
- Grid runner: `scratchpad/decision_stack/collect_grid.py`.
- Seeds: `docs/benchmark/SEED_REGISTRY.json`, E6 `dev_reserved`.

The operator probe campaign (`probe.py` / `crt.py`, `e6-probe/1`) is a separate contract owned by Worker K. Only
its seeds and RNG tags are listed here.

## 1. Estimand and design

**Estimand.** The target is the arbiter's declared estimand: a joint WG3 plan is held for D = 20 s, then the
arbiter accepts every request (accept-all).
- The target quantity is the difference Δ_r(π | h) = E[C_H(π then accept-all) − C_H(accept-all) | h], for each
  region r.
- The horizon is H = 90 s.
- The same estimand is computed by `TrueSimWM(continuation="accept_all")` and `EffectWM(continuation="accept_all")`.

These are policy intention-to-treat effects. They include the xApps' reactions. They are not knob `do()` effects:
whether and what an xApp requests still depends on load.

**Episode.** Each episode uses:
- `E6Config(seed, load ∈ {medium, high}, scenario ∈ {base, surge, mistune}, mobility="mixed", mix="M4",
  warmup_s=120, scored_s=600)`;
- `E6Env(wg3=True)`, with no churn cap and no per-epoch churn allowance.

**Spaced slots (v3).**
- Slot k starts at t_k = 120 + 90k (slot_s = 90).
- At t_k every region draws its region plan (the mixture is in section 2) and holds it for the treatment window
  [t_k, t_k + 20).
- The whole network is then accept-all for the washout [t_k + 20, t_k + 90). The washout is 70 s, the minimum
  `MIN_WASHOUT_S`. Because it covers every region, it covers every region's CIO interference neighbourhood.
- Rollbacks are issued only at t_k.
- A WG3 lock set during the treatment lasts D s from its request, as in `plans.decide`. It can therefore extend at
  most 20 s into the washout. This is part of the plan's effect and is identical in the oracle rollout (section 5,
  F4).
- Same-slot assignments of the other regions are concurrent and randomized, and logged (`pol_nbr_*`). They are
  covariates of the joint-plan contrast.

**Labels.**
- Training rows are the slot starts whose window (t_k, t_k + 90] ends by t = 720. That gives k = 0..5, so 6 slots
  × 10 regions = **60 labelled rows per episode**.
- Slot 6 (t = 660) is played but unlabelled.
- H ≤ slot_s is enforced (`episode_from_trace` raises otherwise), so no later assignment of any region falls inside
  a label window.
- Labels are the privileged per-second counters in `trace.py`. They are never used as features.

**Plan semantics** (`plans.decide`, `half_rule="slew"`):
- accept, reject, and lock (reject + lock the knob for D s);
- **half = halve the slew rate**:
  - a requested step of 2 or more actuator quanta becomes a modify by ceil(n/2) quanta toward the proposal (TTT by
    index);
  - a single-quantum step is accepted on alternate requests per knob. The toggle phase is the parity of the knob's
    cell-index sum. The state is the collector's `half_state`.
- rb: at t_k, roll back the region's knobs changed in the last 60 s, except our own rollbacks.
- Measured realised fraction under half (300 s scored, pre-pilot DEV check): CIO 0.46 / 0.47, LL ratio 0.53 / 0.47
  (medium / high).

**Quiet epochs.** Not used in v3: `quiet_epochs` must be 0, because the washout replaces them. v2 keeps the option,
default off.

**v1 / v2.**
- Both are kept only to reproduce earlier data.
- v2 re-randomizes every 20 s. Its labels therefore identify "plan, then logging policy", not this estimand.
- v1 grid jobs use `half_rule="legacy"`.

## 2. Assignment mechanism (`StepMixture`, frozen)

For each (slot k, region r), `default_rng([cfg.seed, 7709, k, r])` draws u ~ U[0, 1). Draws continue in the order
listed below.

| branch | probability | plan |
|---|---|---|
| ACCEPT | p_accept = 0.50 | accept-all (code 0) |
| FRAC | p_frac = 0.40 | per xApp (MRO, TS, ES, SLICE order), independently `integers(3)` → reject / half / accept |
| LOCK | p_lock = 0.05 | one xApp chosen uniformly (`integers(4)`) → lock; the others accept |
| RB | p_rb = 0.05 | rollback flag; all xApps accept |

The assignment depends only on the seed and the keys. It is therefore fixed before any outcome and independent of
the plant tape, scenario and load (tested).

Exact propensities:
- Code propensity: P(code) = 0.5·[code=0] + 0.4/81·[rb=0, modes ∈ {acc, rej, half}] + 0.05/4·[rb=0, one lock,
  rest accept] + 0.05·[rb=1, all accept]. This is `StepMixture.code_probs()`, logged per row in `pol_prop`.
- Per-(region, xApp) mode marginals: accept 0.7208, reject 0.1333, half 0.1333, lock 0.0125 (logged in
  `pol_prop_x`).
- The propensities do not depend on context. The effect model therefore fits **unweighted** (`weighting="auto"`
  resolves to none for v2/v3); IPW stays selectable.

## 3. RNG stream tags and seeds

| tag | stream |
|---|---|
| 7707 | v1 collector `[seed, 7707, epoch, region]` |
| 7708 | v2 collector `[seed, 7708, epoch, region]` |
| 7709 | **v3 collector** `[seed, 7709, slot, region]` |
| 8808 | probe schedule / arms (`probe.py`) |
| 5151 | CRT resampling (`crt.py`) |
| 6262 | CRT injected-effect calibration (`crt.py`) |
| 4242, 99 | effect-model episode bootstrap / AIPW folds (fit-time only) |
| 13 | arbiter candidate search `[seed, t, 13]` |

Seed allocation (E6 world; all DEV, disjoint from DEV 0–30 and from TEST ≥ 960000):
- Rule: seed = base + 30·stratum + j, where stratum = 2·scenario_idx + load_idx.
- Stratum order: base-medium, base-high, surge-medium, surge-high, mistune-medium, mistune-high.
- The rule is implemented in `collect.dev_seed` / `dev_stratum`.

| kind | range | use |
|---|---|---|
| policy | 100000–100179 | v3 policy episodes. j < 20 fit, j ≥ 20 untouched diagnostics. **Pilot = j ∈ {0, 1}** (fit seeds only) |
| probe | 110000–110179 | operator probe (Worker K) |
| calib | 120000–120179 | gate calibration episodes. j < 5 = pooled design; j = 5..29 reserved for the per-stratum alternative |
| eval | 130000–130179 | gate final-evaluation episodes, same split rule as calib |

Paired accept-all references reuse the episode's own seed, which gives the same plant tape. They are used only for
collection cost, never for the inferential null. Probe episodes never enter TEST or WG3 policy runs.

## 4. Reports

**First stage and support.** Report per scenario × load × xApp × mode, computed on labelled rows by
`support_counts`:
- `treated`: eligible rows assigned that mode. Eligible means the xApp offered at least one request on the
  region's knobs during the treatment window [t_k, t_k + 20).
- `control`: eligible rows whose region plan is accept-all.
- `treated_start` / `control_start`: the same counts with eligibility at the slot-start second.
- `ep_treated` / `ep_control`: the number of independent episodes contributing.
- `ack`: ACKs on treated rows.
- `moved`: treated rows with a realised change.
- `real_frac`: mean realised/requested delta.
- For rollback ("RB"): treated = rb with at least one rollback requested; control = accept-all rows.

Also report:
- branch frequencies versus declared;
- realised fraction per knob family and assigned fraction;
- no-op assignments, i.e. treated region-slots with no eligible request;
- NACK counts by cause.

**Minimum-support rule** (`SupportRule(20, 20, 10)`):
- An (xApp, mode) policy effect is **identified** in a stratum iff there are at least 20 eligible treated rows and
  at least 20 eligible accept-all rows, each from at least 10 independent episodes.
- Otherwise it is **unidentified**:
  - `EffectWM(support=SupportRule().identified(counts, stratum))` excludes every plan that uses it from learned
    scoring (score = +inf, `ood`, `unidentified` listed);
  - no controller win may be claimed for it;
  - the probe cannot fill this gap.
- The rule is applied only to counts.
- Forbidden: retuning xApps, selecting seeds, or expanding the collection without a new contract version. Any
  expansion must be triggered by the blinded support counts alone.

**Collection cost** (paired, v3 minus the same-seed reference, per stratum). Report the mean, median, p90 and max
over episodes of the difference in:
- SVR;
- violated UE-s per slice (LL, eMBB, BE);
- outage;
- RLF/UE-h;
- severe incidents;
- energy kWh;
- applied changes (churn);
- changed-knob-seconds;
- HO/UE-h;
- ping-pong.

Report these together with the no-op/NACK blocks above. Aborts do not exist in the policy collector (report 0).

## 5. Pilot: predeclared failure criteria (checked BEFORE any efficacy is inspected)

The pilot is `collect_grid.py run --split pilot`: 12 v3 episodes plus 12 same-seed references, 2 fit seeds × 6
strata. The pilot fits no effect model and computes no treatment contrast.

If any criterion below fails, the pilot fails. The response is to fix the problem, bump the version and re-run on
the same pilot seeds. Diagnostic seeds stay untouched.

| # | check | criterion |
|---|---|---|
| F1 | trace reconstruction (`audit_episode`) | `labels_match_score`, `changes_match`, `requests_match`, `policy_requests_match` True in 24/24; `config_rebuild_share` ≥ 0.99 |
| F2 | propensity match | `propensity_exact` in 12/12. Pooled branch frequencies (840 draws) and per-xApp mode marginals within 4 binomial SD of section 2 |
| F3 | first stage (pooled pilot) | f = 0 (reject/lock): no realised change by an assigned request. f = 1: mean realised fraction ≥ 0.8 for CIO and LL ratio. half: realised fraction in [0.3, 0.7] for CIO and LL ratio when ≥ 30 such requests, else reported "insufficient". ACK + NACK + reject = requests, exactly |
| F4 | continuation semantics | `continuation_ok` in 12/12: at the first two slots, `TrueSimWM(continuation="accept_all")` objective on `env.copy()` == the realised objective over the same 90 s window (exact float equality) |
| F5 | washout, cap and abort accounting | `washout_accept_only`, `rollback_only_at_slot_start`, `cap_respected`, `churn_blocked_accounted` in 12/12; no job exception; 0 aborts |
| F6 | reference identity | every reference has zero non-default assignments and propensity 1; one reference per stratum reproduces the no-arbiter `E6Env.run()` score exactly |

Review trigger (not a failure; cost only): pause for review if, in any stratum, the median paired excess SVR is
greater than 25 % of the reference SVR, or severe incidents exceed 2× the reference. The design does not change
automatically.

## 6. Gate calibration and evaluation split

- The gate is `ConformalGate(method="episode_max")`, the default. Its unit is the episode, because epochs within an
  episode are dependent.
- Calibration target: the actual deployed search and plan class. The arbiter's full candidate search with the
  learned WM is run at each slot start, and the selected plan is scored by the oracle (`gate.oracle_record`).
- Calibration and final evaluation use **disjoint episodes** from the `calib` and `eval` seed ranges. Neither range
  is ever used for training.
- Guardrails (per-component effects) are bounded separately. The episode-max bound does not cover them.

**Pooled design (default).**
- 5 calibration + 5 evaluation episodes per stratum, i.e. 30 + 30 episodes. 30 is `min_units` in episode units.
- The coverage claim is **pooled over the six strata**. No per-stratum (conditional) claim is made.

**Per-stratum alternative.** A conditional claim per stratum needs 30 calibration + 30 evaluation episodes per
stratum: 360 episodes, 6× the pooled cost. It uses j < 30 in both ranges and requires a new contract version
before use. Rough cost per calibration episode: one episode, plus 2 oracle rollouts of 90 s at each of 6 slots,
plus the learned search, i.e. about 2.5 plain episodes (about 45 s locally).

## 7. Implied budget (pre-pilot planning numbers, not results)

Per episode there are 60 labelled rows. The expected eligible-treated count per mode is 60 × P(mode) ×
eligibility. The eligibility shares per region-epoch below come from the v2 DEV check (seed 100000, base-medium);
the pilot re-measures them per stratum.

| xApp (eligibility) | reject or half / episode | lock / episode | accept-all control / episode | episodes to meet 20/20/10 |
|---|---|---|---|---|
| TS (0.78) | 6.2 | 0.59 | 23.6 | reject/half: 10 (episode floor); lock: 34 |
| SLICE (0.10) | 0.80 | 0.075 | 3.0 | reject/half: 25; lock: 267 |
| ES (0.06) | 0.48 | 0.045 | 1.8 | reject/half: 42; lock: 444 |
| MRO (0.006) | 0.048 | 0.005 | 0.18 | reject/half: ~420; lock: ~4400 |

With the reserved 20 fit episodes per stratum, only TS reject/half and RB are expected to be identified (RB
observed at about 3.5 treated rows per episode). The following are expected to stay **unidentified**:
- SLICE reject/half (needs 25 episodes);
- TS lock (needs 34);
- ES in all modes;
- MRO in all modes;
- SLICE lock.

The rows per episode (60) are 4.3× fewer than v2's roughly 260, but they are uncontaminated. REPORT_G's "about 6
v2 episodes to detect a 5 UE-s mean eMBB effect" scales to about 26 v3 episodes per stratum, before any gain from
the lower dependence between rows. Any expansion follows section 4.
