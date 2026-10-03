# fix-classic status (2026-10-02): READY-TO-MERGE (F7 fixes, R-17/R-18 arms on the real helper, R-22)

Branch xm/fix-classic from feat/v2 b7456c2; feat/v2 merged in at 419635a (helper 1191678). Never pushed.

## Running
- Nothing (R-26 noted; no more cloud work here). Kaggle fixc-f7-a done; outputs in results/fix_classic/.

## QUESTIONS
- none open. Q-F1 → R-22/R-23 (not testable); Q-F2 (eq is the default arm) accepted.

## HAND-BACK
Commits: 9214c0b..HEAD (F7 fixes, F6 rows, arms, Kaggle results, Q-F1, helper switch).

Code: methods/{two_tower,shap_dag,pc,granger,_classic_common}.py; scripts/xm_classic_fidelity.py (`--config` for
f3_e2); tests/test_xmethod_classic.py. The stub `_covariates_stub.py` is deleted.

Docs: docs/xmethod/FIDELITY_CLASSIC.md: Arms section (R-25 sets, R-22 counts); F2, F3, F4, F6, integration and cost.

Fixes:
1. two_tower: score = ROW SHARE a[k,p] / sum_p a[k,p].
   - Not a/row-max: that sets the top of every row to 1, so the placebo tau saturates at 1.
   - Not sparsemax: it depends on the row's scale.
   - notes['label'] = 'adaptation (self-supervised E2 port), not the published supervised model'; raw gate in
     notes['gate'].
2. shap_dag: XGBRegressor at package defaults (paper; R-13); option 'hgbdt' with early_stopping=False.
3. pc: notes['fisherz_pinv_fallback_rate']; pcalg pMax cited.
4. granger: native arm labelled 'pairwise Granger'.
5. Arms (R-17/R-18/R-25): `config['arm']` 'eq' (default) | 'native'. Every set comes from `design_covariates` via
   `_classic_common.cond_set`.
   - Action source: eq = focal=<action> (concurrent designed); native = R-3 set (concurrent='all').
   - Lagged-KPI source (no helper focal): the arm's base set minus its own lag_kpi, plus all actions at t.
   - granger eq = conditional Granger / VARX on that set.
   - pc = actions + the arm's base helper set (eq: sp / @t-L as exogenous tier-0 nodes) + KPIs.
   - corr / notears / shap_dag / two_tower: native only, notes['arm_note'] = 'no conditioning interface'. Their
     signs use the helper's R-3 set.
6. R-22 not testable (eq arm, exact fit). Counts (R2, n 1000, seed 3_000_000):

   | world | method | not testable |
   |---|---|---|
   | E3 | granger and pc | 46 of 50 |
   | E1 | pc | 32 of 36 |
   | E2 | pc | 0 of 90 |
   | E5 | pc | 0 of 36 |

   The true edges among them are all KPI->KPI edges whose source lag is an exact copy of a lagged action (E3
   lag_kpi:K0 == P0@t-1). Primary true edges stay testable; R-24 noise removes these fits.

Dependency: xgboost 3.4.1 (`uv add --group baselines`); on Kaggle pip-installed at that pin (R-16, logged).

Tests: all xmethod tests pass, 205/205 (classic 42, covariates, harness, pmrt).

Kaggle fixc-f7-a (code 9214c0b; arm-independent):
- shap_dag XGBoost: F3 10/10, 16 edges, ratio .795 (ref .817); F4 R1 n 500 power 1.00/1.00, null-edge FPR .068
  [.046,.100]; cost at n 24000 51 s (hgbdt 159 s).
- two_tower row share, F4: power .95/1.00 (was .75/.90), null-edge FPR .047 [.029,.075].
- Integration: 24 jobs, 0 errors.

Not run here: the eq-arm F4 and integration for pc and granger (the R-21 DEV validity check covers them). The
integration table's pc / granger columns are the pre-arm native runs.
