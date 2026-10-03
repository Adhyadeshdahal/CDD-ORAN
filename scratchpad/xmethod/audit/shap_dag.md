# F7 audit: shap_dag (auditor aud1, 2026-10-02, feat/v2 b7456c2)

**Verdict: PASS-WITH-NOTES.**

Sources used:
- the paper: Sharma et al., "Towards xApp Conflict Evaluation with Explainable ML and Causal Inference in O-RAN" (NFV-SDN 2025), `archive/reference-papers/Towards_xApp_Conflict_Evaluation_...pdf` (converted with markitdown; sec IV-B/C, V);
- `cdd_oran/xmethod/methods/shap_dag.py`, `scripts/e2_baseline_shap_dag.py`, `docs/xmethod/FIDELITY_CLASSIC.md`.

## Findings (most severe first)

1. **MEDIUM: the paper's edge rule is qualitative.**
   - Sec IV-C1 builds the DAG with "causal edges from the RCPs which are the most influential towards their associated KPIs". There is no numeric threshold; the RCPs are read off SHAP plots (sec V, Fig. 4).
   - So the E2 relative rule (tau_rel .10, `native`) and the absolute score with the placebo tau (`shap_dag.py:71-72`) are both ours.
   - The absolute score, mean|SHAP| / sd(y), is the fair choice: it is scale-free and comparable across KPIs, unlike two_tower's gate.
   - **Paper text:** "SHAP-DAG discovery stage, with the threshold set by our placebo rule (the paper specifies none)".

2. **MEDIUM: XGBoost is replaced by sklearn HistGradientBoostingRegressor (300 iterations, lr .1).**
   - The paper picks XGBoost because it had the best R^2 (Table II) and states no hyperparameters (`e2_baseline_shap_dag.py:68`).
   - This is the paper's choice of regressor, not a tuning detail, so R-13 ("authors' defaults") favours XGBoost at its package defaults. It is cheap: `uv add xgboost`, and TreeExplainer supports it natively.
   - **Recommendation:** switch, or keep HistGBDT with the deviation stated and a one-cell sensitivity check (E2 R1 n 1000) showing the scores agree.

3. **LOW-MEDIUM: the estimator changes with n.**
   - sklearn HistGBR has `early_stopping='auto'`, which turns on at n > 10 000. So at n 24 000 the model has fewer trees and a 10 % validation hold-out, and the scores at n 24 000 come from a different estimator than at n <= 8000.
   - The F6 row records this as a package default. For a scaling curve it is a confound: fix `early_stopping=False`. XGBoost would also remove the issue, since it has no early stopping by default.

4. **LOW: no DoWhy ATE stage.**
   - The paper's ATE (linear backdoor on the DAG's other RCPs) is replaced by the sign of pcorr given all other actions, lags and context. That matches R-4, and the sign of a linear backdoor coefficient equals the partial-correlation sign on the same set.
   - The refutation and CATE stages are effect estimation, not discovery, so dropping them is correct for this study.

5. **LOW: the F3 "PASS" is a replay of our own earlier E2 port (ratio .8165 vs .8173).** It shows the wrapper is unchanged, not that the published result is reproduced. The paper's MATLAB data are not available. State it as a regression check.

6. **OK:**
   - Features are the actions only (the paper regresses KPIs on RCPs only), so KPI->KPI is NaN by design.
   - No Truth access.
   - Seed from tag 7803.
   - F4 R1 n 500: power .95/1.00, null FPR .059.
   - Placebo exchangeability: the score is invariant to feature scale (trees), and the R2 placebo carries its own setpoints and dither like the real actions (`generate.py:259-267`).

## Recommended fix

1. `uv add xgboost`; use XGBRegressor at package defaults with a seed (or document a sensitivity check).
2. `early_stopping=False` if HistGBDT stays.
3. Doc: state the threshold as ours (finding 1) and describe F3 as a regression check.
