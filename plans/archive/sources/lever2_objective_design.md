# Lever 2 — scope the QACM objective fix (DESIGN ONLY)

**Date:** 2026-08-25 | **Mode:** READ-ONLY design/scoping. No GPU, no code edits, no commit.
**Branch:** feat/v2 | **Inputs:** `negative_utility_diagnosis.md`, source read at `file:line`.

---

## 1. How the QACM objective works today (with the decisive lines)

QACM is a single-step grid search: for every `(bin, index)` candidate on the conflict param it
predicts the next state and scores it, keeping the **argmin cost** (`qacm.py:62-107`,
`qacm.py:103-105` strict `<`). The cost is the same in both code paths:

- **Ensemble path (m>1, the configured path — `predict_members: 8`, `env_i_cdl.yaml:48`):**
  `qacm.py:70-79` → `robust_returns_from_dist` (`ensemble.py:108-130`) → `score_batch`
  (`cost.py:62-100`).
- **m=1 manual path:** `qacm.py:82-101`.

The scalar cost per candidate (`cost.py:99`, mirrored at `qacm.py:101`) is:

```
cost = Σ_i  w_i · distance_i · τ   −  ( Σ_i satisfied_i )²
```

with, per xApp (direction 0 = maximiser, `cost.py:91-98` / `weighted_distance` `cost.py:29-39`):

```
norm_threshold = (threshold − mean)/std
unsat_i    = u_i < norm_threshold
distance_i = (norm_threshold − u_i) if unsat_i else 0.0     # HINGE, clamped at 0
satisfied_i= 0 if unsat_i else 1
```

**Where it goes flat.** Two regimes:

- **Unsatisfied** (`u_i < norm_threshold`): `distance_i = norm_threshold − u_i` is decreasing in
  `u_i`, so minimising cost **does** push utility up — but only *toward the threshold*.
- **Satisfied** (`u_i ≥ norm_threshold`): `distance_i` is clamped to **0** (`np.where(unsat, …, 0.0)`,
  `cost.py:93`) and the only remaining term is the constant satisfaction bonus `−(Σ sat)²`. **The
  objective is now completely indifferent to how far above threshold `u_i` sits.**

Because the argmin is a strict `<` scan (`qacm.py:103`), among a set of equally-satisfied candidates
(identical cost `= 0 − N²`) it keeps the **first candidate scanned** (lowest bin/index) — not the
highest-utility one. This is the smoking gun.

**Why this leaves the +0.94 frontier unrealised.** Env I thresholds are LOW in z-space
(diagnosis §1: KPI1 `z=−0.37`, KPI3 `z=+0.19`, KPI4 `z=+0.03`), while ceilings are `≈+1.3`. So a
candidate is declared "satisfied" at mediocre utility (e.g. KPI1 at `−0.37`) and the objective stops
climbing toward `+1.29` — pure on-the-table utility the planner is structurally blind to. **This is
model-independent**: it bites even with a perfect world model, which is why it is a genuine lever
distinct from Cause 1 (OOD model exploitation).

**Interaction with the risk aggregator.** Utilities are already risk-adjusted (LCB via
`risk_adjust`, `cost.py:9-26`) *inside* each member's `score_batch`, then per-member **total costs**
are aggregated by the quantile aggregator (`ensemble.py:70-80`, `combine`). Any fix must stay inside
the per-member cost so the pessimistic quantile is preserved — see the constraint in §3.

---

## 2. Fix options (minimal + principled)

Note: utilities are **already standardized z-scores** (`base.py:160-167`, `(value−mean)/std` over the
train range), so a *dimensionless* utility coefficient is comparable across envs by construction — the
key honesty lever (see §3).

### Option A (RECOMMENDED) — additive standardized utility term, default-off knob
Add a signed-utility reward to the per-member cost, exactly mirroring the existing additive-knob
pattern (`disagreement_penalty`, `risk_kappa` — both default-0 no-ops):

```
sign_i = +1 (dir 0)  |  −1 (dir 1)          # reward the satisfaction direction
cost  = Σ_i w_i·distance_i·τ  −  (Σ_i sat_i)²  −  λ · Σ_i sign_i · u_i
```

- **Change:** accumulate `sign_i·u_i` in `score_batch` (`cost.py:82-99`); subtract `λ·Σ` from `costs`.
  Mirror in `_score_batch_reference` (`cost.py:129`) and the m=1 path (`qacm.py:101`). Add
  `utility_weight: float = 0.0` to `PlannerConfig` (`config.py:83-96`) + `from_dict`
  (`config.py:203-208`) + a `0.0` key in each `configs/env_*_cdl.yaml`.
- **Effect on the bottleneck:** restores a utility gradient **everywhere, including past the
  threshold** → captures the KPI1/3/4 slack (the low-threshold cells leaving `~+1.3` on the table).
- **Trade-offs:** `λ` too large drowns the satisfaction/distance terms (could sacrifice a satisfied
  xApp to chase another's utility). `u_i` is on the standardized z-scale but the distance term carries
  `τ·w`; keep `λ` small so satisfaction still dominates the ordering (validate — §4). Adaptive variant:
  scale `λ` by `τ` per decision to match units if a fixed value proves fragile.
- **Risk-aggregator interaction:** the term lives **inside** each member's cost, so the quantile is
  taken over utility-inclusive per-member totals → **risk aversion preserved, does NOT become
  expected-utility.** `λ=0` reproduces current behaviour bit-for-bit (regression-safe).

### Option B — true lexicographic tie-break (zero hyperparameters)
Rank candidates by `(primary = current cost, secondary = risk-aggregated signed utility)`; argmin on
the composite key. Since all equally-satisfied candidates have identical primary cost (`0 − N²`), the
secondary key breaks exactly the flat region and nothing else.

- **Effect:** identical fix for the diagnosed flat region; **no new hyperparameter** → the strongest
  honesty guarantee (nothing to tune, nothing to leak).
- **Trade-offs:** needs an *extra* risk-aggregated utility tensor threaded through
  `score_batch → robust_returns_from_dist → qacm` (larger, more invasive diff than A). Only engages
  on satisfaction-count ties; when counts differ, higher count always wins regardless of utility gap
  (the intended lexicographic priority, but blunter than A's continuous trade). Preserve risk by
  aggregating the tie-break utility with the same quantile (use LCB `u`, not the mean).

### Option C — soft satisfaction (deferred)
Replace the hard hinge + `−(Σs)²` with a smooth saturating reward, e.g. per-xApp
`r_i = σ((u_i − thr_i)/β)` plus a residual push. Removes the threshold cliff entirely.
- **Trade-offs:** introduces a scale hyperparameter `β`, changes the reported-satisfaction semantics
  (`evaluate.py:616-633` records the hard rule), and loses the discrete satisfaction guarantee.
  Biggest departure; **not** the first move — revisit only if A/B under-deliver on the tail.

---

## 3. Honesty / generalization constraint (IMPORTANT)

**Env-agnostic setting of `λ` (Option A).** Because utilities are standardized z-scores, `λ` is
dimensionless and **not** tuned on test utility. Protocol: fix a small dimensionless `λ` (sweep
`{0.1, 0.25, 0.5}` for the ablation only), **select on Env III (positive, no-regression)** and on the
**in-distribution** eval, then apply the *chosen fixed* `λ` unchanged to the OOD evals of Env I/IV.
Selection never touches Env I/IV test utility → leakage-free. Option B removes the question entirely
(no hyperparameter).

**Folding in `disagreement_penalty` as a principled, env-agnostic uncertainty term.** The knob is
already self-measurable: `disagreement = std over members of the predicted-mean, averaged over KPIs`
(`ensemble.py:95-105`), added as `w·disagreement` (`ensemble.py:128-129`). To make its coefficient
env-agnostic, **calibrate on the in-distribution (train-range) disagreement scale** — e.g. z-score the
disagreement by its train-range mean/std, or set the coefficient so the penalty at the train-range 90th
percentile equals a fixed fraction of the utility scale — then apply that fixed coefficient to OOD.
That is the same leakage-free "calibrate on train, apply fixed to OOD" protocol as `λ`.

**Does the disagreement penalty even help in-distribution?** The diagnosis sweep only tested OOD
(diagnosis §3 lever 1). Cheap check (no retrain): re-run `evaluate` **in-distribution**
(`evaluation_param_ranges: train`, per diagnosis §4) with `disagreement_penalty` on vs off on the
existing checkpoint. If it does not hurt in-distribution and helps OOD, the calibrate-on-train
protocol is clean. **The two levers are complementary and should be validated together but reported
separately:** Option A/B fixes "leaves utility on the table once satisfied" (model-independent);
`disagreement_penalty` fixes "confidently-wrong OOD pick" (Cause 1). Lever 2 (the objective) is the
code change scoped here; the disagreement folding is config + a small train-range calibration helper.

---

## 4. Validation plan

- **Envs:** Env I (negative) + Env IV (negative) as targets; **Env III (positive) as the
  no-regression guard**; Env II if a checkpoint exists.
- **Regimes:** run **both in-distribution and OOD**. The objective fix must help in *both* — most
  cleanly in-distribution, where model error is small so the *only* remaining gap is the objective
  (predict: in-distribution `mean_util` jumps toward the ceiling with `λ>0`).
- **Gate:** reuse the matched-decision-set 2×2 gate (`scripts/gate_2x2.py`) with the existing **CRN
  seeding**, comparing baseline (`λ=0`) vs fix (`λ>0`) on the **same** decisions/checkpoint (no
  retrain — the fix is planner-only).
- **Ablation:** `λ ∈ {0, 0.1, 0.25, 0.5}`; expect near-monotone improvement then saturation. Select
  `λ` on Env III + in-distribution, apply fixed to OOD.
- **Success criteria:** Env I/IV `mean_util` moves **materially** from `−0.11` toward `+0.94` — at
  minimum beating the `+0.24` random/no-op reference — **while**: (a) `tail_p10` does not collapse,
  (b) `sat_rate` does not fall below baseline, (c) Env III positive utility does not regress. Report
  the per-cell regret drop, especially the low-threshold KPI1/KPI3/KPI4 cells (diagnosis §2).

---

## 5. Diff size + risk

**Files/functions touched (Option A):**
- `cdd_oran/planners/cost.py` — `score_batch` (`cost.py:82-100`), `_score_batch_reference`
  (`cost.py:129`), `_self_check` (`cost.py:133-185`, add a `λ=0` no-op parity assertion + a `λ>0`
  monotonicity assertion). *Shared by CEM/MPPI/MCTS too* — the term flows to them automatically; keep
  default 0 so it is opt-in and their goldens are untouched.
- `cdd_oran/planners/qacm.py` — m=1 manual path (`qacm.py:101`) mirrored (kept for parity even though
  the configured path is m=8).
- `cdd_oran/config.py` — `PlannerConfig` field (`config.py:83-96`) + `from_dict` (`config.py:203-208`).
- `configs/env_*_cdl.yaml` — one `utility_weight: 0.0` key each.
- (If threaded via the aggregator instead of `score_batch`: also `planners/__init__.py:11-22`,
  `ensemble.py`.)

**Estimated LOC:** ~30-50 (Option A). Option B is larger (~60-90: extra tensor threaded through 3
files + composite argmin). Option C larger still.

**What could regress:**
- **CRN parity self-check** (`cost.py:133-185`) and `_score_batch_reference` — *must* keep `λ=0`
  byte-identical; add the assertion so default behaviour is provably unchanged.
- **2×2 gate** (`scripts/gate_2x2.py`) and **CRN seeding** — unaffected at `λ=0`; the fix is compared
  under the same seeds.
- **Tests:** `tests/test_robust_planner.py` — the m=1 byte-identity test
  (`test_m1_scoring_is_byte_identical_to_pre_phase3_path`, lines 185-211) and
  `test_m1_planner_action_matches_baseline_scoring` (296-315) hold **because `λ` defaults to 0**; add
  a new test for `λ>0` picking the higher-utility satisfied candidate. `tests/test_counterfactual_metric.py`
  — unaffected (metric/utility_fns untouched; the fix is planner-side only).
- **Risk-aversion invariant:** verify the quantile aggregator still diverges from `mean` with `λ>0`
  (extend `test_risk_averse_aggregator_picks_safer_action_than_mean`) so the fix cannot silently
  become expected-utility.

**Overall risk:** LOW for Option A — it is a default-off additive knob following the established
`disagreement_penalty`/`risk_kappa` convention, with `λ=0` guaranteeing bit-identical current
behaviour; all regression exposure is confined to `λ>0` runs, which are new evaluations, not existing
goldens.

---

## Recommendation

**Adopt Option A** (additive standardized utility term, `utility_weight` default `0.0`), select `λ` on
Env III + in-distribution and apply it fixed to OOD, and **fold `disagreement_penalty` in as a
train-range-calibrated term** validated alongside. Keep **Option B (lexicographic)** as the
zero-hyperparameter fallback if reviewers reject any tunable, and **defer Option C**. This targets the
model-independent "flat once satisfied" lever, preserves the risk-averse quantile aggregator, and is
regression-safe by default.
