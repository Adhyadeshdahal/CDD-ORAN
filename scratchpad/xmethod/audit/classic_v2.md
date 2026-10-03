# Re-audit of the changed classic methods (audit-classic2, 2026-10-02, feat/v2 dea83cf)

Scope: `methods/{two_tower,shap_dag,pc,granger,_classic_common}.py` and `tests/test_xmethod_classic.py`. Evidence: scripts and outputs
in `scratchpad/xmethod/audit/classic2/`. All runs were local, on DEV seeds, at kappa .25 unless stated. No method code was changed.

## Verdicts

| method | verdict | one line |
|---|---|---|
| two_tower | OK | Row share is correct; adaptation label on every Result. CONTRACT sec 3 still lists it as "published O-RAN" (fix 5). |
| shap_dag | OK | XGBoost at package defaults; SHAP is additive; no early stopping at any n. |
| pc | OK (LOW, disclose) | eq arm matches Z_eq except in R4 (finding C). |
| granger | OK (LOW, optional) | eq arm is exactly the VARX F-test. At kappa 0 a collinear source gets an unlisted NaN (finding B). |
| _classic_common | **FIX NEEDED** | `tune()` ignores the arm and always tunes the default `eq` arm (finding A). |

Tests: `uv run --group baselines pytest tests/test_xmethod_classic.py` gives 42/42 passed. A plain `uv run pytest` fails 5 tests on
`import shap`, because shap and xgboost are in the non-default `baselines` group (finding D).

## 1. Earlier F7 findings

| audit / finding | status | evidence |
|---|---|---|
| two_tower 1 (HIGH): not the paper's method | PARTLY | Label at `two_tower.py:56,91` and in F6. CONTRACT sec 3 l.53 still files it under "published O-RAN". |
| two_tower 2 (HIGH): one absolute tau across rows | FIXED | a / row sum (`two_tower.py:83-88`). Max error vs `notes['gate']` 6e-8; rows sum to 1. |
| two_tower 3, 4: F3 wording, settings | FIXED (doc) | FIDELITY_CLASSIC F3 / F6. |
| shap_dag 1, 5: qualitative rule, F3 wording | FIXED (doc) | `shap_dag.py:14-18`; F3 row. |
| shap_dag 2: XGBoost | FIXED | `shap_dag.py:45-47`. 100 rounds, reg:squarederror, other params None (package defaults). SHAP additivity error 2.3e-5 (prediction sd 5.9). Adapter == direct computation. |
| shap_dag 3: early stopping changes with n | FIXED | XGBoost has none; hgbdt runs with `early_stopping=False` (`:50`, test). |
| pc 1, 2: pMax doc, pinv rate | FIXED | `pc.py:27-36,206-207`. Pinv rate is 0 in all 20 smoke runs at kappa .25. |
| pc 4: KCI cost is a projection | UNCHANGED | Out of scope. Needs an R-13 measurement if KCI is used. |
| granger 1: pairwise label | FIXED | `granger.py:121-123`; eq arm adds the conditional (VARX) form. |
| corr, notears | not re-audited | Not touched by fix-classic. |

## 2. Arms (R-17 / R-18 / R-25) (`arms_check.json`)

- **Action sources.** `cond_set` == `design_covariates(d, focal=a)` (eq) and == `design_covariates(d, False, False, focal=a,
  concurrent='all')` (native): same names, matrix and row_mask. 0 mismatches over every action of E1 R1, E2 R2, E3 R2, E4 R3, E4 R4.
- **Lagged-KPI sources.** The arm's base set without the source's own lag, plus all actions at t, as documented
  (`_classic_common.py:95-100`).
- **granger eq.** Restricted = intercept + the whole Z_eq; full adds the source; F(1, n - rank) is the authors' OLS F null.
  - p == statsmodels OLS on [Z_eq, x] to within relative 1.1e-10 (E1, E2, E3 R2; 176 candidates); 0 sign mismatches; 2 rows masked.
- **pc eq tiering is sound.** sp and @t-L are tier-0 X nodes, never candidates; nothing may enter X from A, L or Y (`pc.py:176-181`).
  eq_dropped is empty at kappa .25.
- **C (LOW): in R4 the pc eq node set is wider than Z_eq.**
  - PC keeps every action at t as a node. In R4, Z_eq(P0) has no concurrent column (designs none / iid / none), but PC can still
    condition on `P_placebo_conf`, a proxy of the latent Z.
  - In R1-R3 the node set equals Z_eq.
  - Fix: disclose in FIDELITY_CLASSIC Arms.

## 3. Not testable (R-22 / R-23) (`not_testable.json`)

Setup: n 1000, seed 3_000_000, all 12 cells (E4 at lam 1.0), pc and granger in both arms.
- **kappa .25: 0 not-testable and 0 NaN candidates everywhere, as expected.** The smallest residual share RSS/TSS of any target given
  Z_eq is .052 (E3 R1), against the 1e-10 threshold, so the rule cannot over-trigger.
- **kappa 0 (sanity only).** granger eq flags E1 32/36 (2 true), E3 46/50 (3 true), E4 R3 3/4 and E5 8/36, all at a residual
  share of about 1e-31.
- **B (LOW): silent NaN at kappa 0.** In E5, K2->K0, K2->K1 (true) and K2->K3 get NaN without being listed: lag K2 == P0@t-1, so the
  status is "degenerate" (`granger.py:75-76`).
  - Effect: scored as not declared, but `n_not_testable_true` undercounts by 1 (`degenerate.json`). pc avoids it (drops the X copy).

## 4. Signs, declarations, tau

- **Signs (R-4).** pc (eq, native), shap_dag and two_tower == sign of the OLS coefficient given the arm's `cond_set`, on E2 R2,
  E1 R2 and E4 R3: 0 mismatches over 293 candidates. granger == coefficient sign.
  - Smoke: every declared truth-signed edge has the right sign (E1 20/20 per method, E3 granger 20/20), except shap_dag E2 P6->K5 at 2/3.
- **BY (R-2).** granger declared flags == statsmodels `fdr_by` at q .05, per family (action / kpi / diagnostic), in both arms:
  0 mismatches.
- **tau.** Score > tau strictly; NaN never declared (`_classic_common.py:247`); tau = 2nd-largest placebo score, ties below.
- **A (MEDIUM): `ClassicBase.tune()` uses `self.default_config()` only** (`_classic_common.py:322`).
  - So pc and granger always tune `eq`, and no arm can be passed (`tune_arm.txt`: pc E2 R2 n 300 returns arm 'eq', tau .566;
    native runs on the same data give .790).
  - `scripts/xm_classic_integration.py:63` relies on `tune()`, so a native cell built with it would get the eq tau.

## 5. Dataset-only inputs

No Truth / truth_for / generate import; fit_two_tower sees only X_action and Y. With `meta` cut to the three candidate lists
(dropping block, dither, env_seed, episode, generator, obs_noise, placebo, placebo_conf, warmup), pc eq, granger eq, shap_dag and
two_tower give identical scores and signs.

## 6. Validity smoke (descriptive; kappa .25, n 1000, seeds 3_000_000-004; `smoke*.json`)

Each cell is summed over 5 seeds. Null = truth-null primary edges of real actions; Pl = P_placebo.
- Score-only methods: "decl" uses the in-sample tau (Pl <= 1 by construction); "LOO" uses a tau from the other 4 seeds.
- granger: "decl" = BY; raw = p <= .05. pc "@a" = PC adjacency at alpha .05.

| cell | method / arm | Null decl | Null LOO | Pl LOO | other | recall |
|---|---|---|---|---|---|---|
| E2 R2 | pc eq | 10/160 | 10/160 | 2/30 | @a: null 0, Pl 0 | 23/80 |
| E2 R2 | pc native | 6/160 | 6/160 | 2/30 | @a: null 28/160, Pl 5/30 | 21/80 |
| E2 R2 | shap_dag | 13/160 | 15/160 | 2/30 | | 74/80 |
| E2 R2 | two_tower | 8/160 | 8/160 | 3/30 | | 38/80 |
| E2 R2 | granger eq / native | 0 / 57 of 160 | - | - | raw null 10 / 83; Pl 3 / 9 of 30 | 29 / 50 of 80 |
| E1 R2 | pc eq | 12/60 | 14/60 | 2/20 | @a: 0, 0; tau = pMax < .39; P3->K2 4/5 | 20/20 |
| E1 R2 | pc native | 3/60 | 6/60 | 3/20 | @a: null 3/60, Pl 2/20 | 20/20 |
| E1 R2 | shap_dag | 17/60 | 17/60 | 2/20 | P0->K2 5/5, P1->K3 5/5 | 20/20 |
| E1 R2 | two_tower | 11/60 | 11/60 | 2/20 | P0->K2 5/5, P1->K3 5/5 | 20/20 |
| E1 R2 | granger eq / native | 0 / 23 of 60 | - | - | raw null 3 / 35; Pl 1 / 9 of 20 | 20/20 |
| E3 R2 | granger eq / native | 0 / 28 of 80 | - | - | raw null 3 / 32; Pl 2 / 16 of 25 | 20/20 |

Reading:
1. **granger eq holds its level.** Pooled raw rate .053 on nulls (16/300) and .08 on the placebo (6/75); BY declares no null.
   granger native is invalid under R2, as disclosed.
2. **PC at alpha is clean in eq and inflated in native** (17.5 %).
3. **shap_dag and two_tower declare the E1 indirect lagged nulls** (P0@t-1 -> K0 -> K2) because the setpoint persists. This is the
   design-blind property of a method without a conditioning interface (disclose), not a bug.
4. **pc eq E1 R2: null LOO .23 [.14,.35] vs placebo .10 [.03,.30].** Not conclusive at 5 seeds; watch it in the R-21 DEV check.

Cost per dataset at n 1000 (CPU-s, 1 thread): pc eq 7.5-23.6, native 0.7-1.9; granger 0.1-1.6; shap_dag 6.6-11.7; two_tower 12.8-24.1.

## Fixes (exact)

1. **A (MEDIUM).** `ClassicBase.tune(self, dev, truth_free=True, config=None)` with `base = self.default_config();
   base.update(config or {})`. Pass `{"arm": arm}` from `xm_classic_integration.py:63`. Alternatively, the protocol states that tau is
   computed post hoc per (method, arm, cell) with `score.placebo_tau` and never calls `tune()`.
2. **B (LOW).** `granger.py:114`: also list `st == "degenerate"` candidates as not testable ("source collinear with Z").
3. **C (LOW, doc).** FIDELITY_CLASSIC Arms: pc eq in R4 also conditions on the non-designed concurrent actions.
4. **D (LOW, dev).** Put `baselines` in `[tool.uv] default-groups`, or document `uv sync --group baselines` for the tests.
5. **Orchestrator (doc).** CONTRACT sec 3: list `two_tower` as "neural relevance gate (ours, inspired by arXiv:2601.13213)", not as
   "published O-RAN".
