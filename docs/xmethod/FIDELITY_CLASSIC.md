# xm-classic fidelity table (CONTRACT sec 4, F1-F6)

Status: DRAFT 2026-10-02, worker xm-classic. Code: `cdd_oran/xmethod/methods/{pc,notears,shap_dag,two_tower,corr,granger,pcorr_hac}.py`,
shared rules `_classic_common.py`, registry `classic.py`. Tests: `tests/test_xmethod_classic.py` (needs the
`baselines` group: `uv sync --group baselines`, then `uv run --group baselines pytest tests/test_xmethod_classic.py`;
a plain `uv run pytest` fails the shap_dag tests on `import shap`). Gate runner:
`scripts/xm_classic_fidelity.py` (synthetic data and the read-only earlier E2 replicate rows only).
pcorr_hac (R-32, xm/hac 2026-10-03): `tests/test_xmethod_pcorr_hac.py`, F4 script
`scratchpad/xmethod/pcorr_hac_fidelity.py` (results `scratchpad/xmethod/results/pcorr_hac_f4.json`).

Common to all six (orchestrator rulings, recorded in `scratchpad/xmethod/status/xm-classic.md`):
- Variables: action columns at t (incl. `P_placebo`), non-NaN lagged KPIs at t, context (R3), KPIs at t+1.
- Families (R-6): PRIMARY = action -> KPI (incl. placebo); SECONDARY = lagged KPI -> KPI. Each is its own BY family.
- Declaration: p-value methods (corr, granger, pcorr_hac) -> BY per family, q = .05. Score-only methods (pc, notears,
  shap_dag, two_tower) -> tau = the R-29 conformal cutoff at level .05 of the `P_placebo` scores pooled over the cell's
  tune datasets (`score.placebo_tau`: the ceil((M+1)(1-.05))-th smallest of the M scores, the largest if that index
  exceeds M; M = 0 -> +inf), declare iff score > tau (`tau_rule = 'placebo_conformal_0.05'`; replaced R-2's 2nd-largest
  score, R-38). The same tau is used for the KPI -> KPI family (no KPI placebo exists; Q-C2).
- Sign: NOTEARS = sign(W); corr = sign(r); granger = sign of the coefficient; pc / shap_dag / two_tower = sign of the
  partial correlation given all other actions + lagged KPIs (+ context) (`notes['sign_rule'] = 'pcorr_given_Z'`).
- `native` declarations (the method's own rule: PC at alpha, NOTEARS at w_threshold, the E2 relative tau_rel rule)
  are reported in `notes['native_declared']` and never used as the primary declaration.
- Every run is pinned to 1 BLAS / OpenMP / torch thread (`threadpoolctl`, `torch.set_num_threads`) so CPU-s is
  comparable; method RNG = `default_rng([7803, dataset seed, method index])`.
- R-9 (B = 9999 sequential): N/A, no xm-classic adapter resamples.

## Arms (rulings R-17 / R-18 / R-22 / R-25; `config['arm']`)

- Every conditioning set comes WHOLE from the shared helper `cdd_oran.xmethod.covariates.design_covariates` (R-25;
  `_classic_common.cond_set`), names `lag_kpi:<K>`, `ctx:<c>`, `sp:<P>`, `<P>@t-1`, `<P>@t-2`, `gap@t-<L>`,
  `concurrent:<P>`, `<name>:missing`.
  - `eq` (PRIMARY, default for pc and granger): action source = `design_covariates(data, focal=<action>)` (lagged
    KPIs, context, setpoints of every dither design incl. the focal one, all actions at t-1 / t-2, concurrent
    DESIGNED actions as pmrt_core); rows with `row_mask` False dropped.
  - `native` (SECONDARY): action source = the R-3 set `design_covariates(data, False, False, focal=<action>,
    concurrent='all')`.
  - Lagged-KPI source (secondary family; the helper has no non-action focal): the arm's base helper set without the
    source's own `lag_kpi:<K>` (and indicator) + every action at t (`concurrent:<P>`).
- granger eq: conditional Granger / VARX: restricted = intercept + the source's eq set (incl. the target's own lag),
  full adds the source, F(1, n - rank). One lag, so this is the partial-correlation F-test given that set. granger
  native: pairwise (restricted = the target's own lag).
- pc: nodes = actions at t + the arm's BASE helper set (no focal; the concurrent actions are the action nodes) + KPIs
  at t+1. In eq the setpoints and lagged actions are exogenous tier-0 nodes (kind X; background knowledge: nothing into
  them from an action, lagged KPI or KPI; X -> action / lag / KPI and X -- context / X -- X allowed), never
  candidates; an X column identical to an earlier node is dropped (`notes['eq_dropped']`). Sign rule (R-4) uses the
  candidate's set of the same arm. PC keeps every other action at t as a node, so in R4 the eq node set is wider than
  Z_eq: Z_eq(P0) has no concurrent column there (designs none / iid / none), but PC can still condition on the
  non-designed `P_placebo` (R1-R3: node set = Z_eq). `P_placebo_conf` is no longer a node of the primary graph (R-37,
  next bullet; this supersedes the audit-classic2 C disclosure that PC could condition on it as a proxy of the
  latent Z).
- R-37 (`P_placebo_conf`, the E4 R3 / R4 diagnostic of R-10, is never a conditioner of another source):
  `cond_set` removes every helper column carrying it (`concurrent:P_placebo_conf`, `P_placebo_conf@t-<L>`, their
  `:missing`; `_classic_common.is_diagnostic_column` / `drop_diagnostic`) from the set of every other action and
  lagged-KPI source, in every arm (so granger eq, pcorr_hac and every pcorr_given_Z sign); PC's primary graph has no
  such node, and the `P_placebo_conf -> K` candidates are read from a second PC run with every node
  (`notes['diagnostic_graph']`). `P_placebo_conf`'s own conditioning set is the full helper set. Test: scrambling the
  column leaves every other candidate's score / p / sign / declaration unchanged (pc eq / native, granger eq,
  pcorr_hac native / eq_min; E4 R3 and R4), and fails without the rule. R-38 extends this to notears, shap_dag and
  two_tower: the primary fit has no `P_placebo_conf` variable / feature / input, and its candidates are read from a
  second fit with every one (`notes['diagnostic_fit']`; two_tower: its own gate); same scramble test.
- `eq_min` (R-33; pcorr_hac only): action source = the R-3 set + `sp:<focal>` (`covariates.eq_min_covariates` =
  `design_covariates(..., include_setpoints=True, include_lagged_actions=False, concurrent='all',
  setpoints='focal')`); a focal column without a dither design (R1 / R3 / R4) has no setpoint, so eq_min = native
  there; a lagged-KPI source has no setpoint either (eq_min = native). |Z| of P0 native / eq_min / eq: E1 8 / 9 / 23,
  E2 14 / 15 / 41, E3 9 / 10 / 24, E4 2 / 3 / 8, E5 8 / 9 / 23 (R2); R1 eq_min = native (8 / 18 eq in E1).
- corr, notears, shap_dag, two_tower: no conditioning interface, native only (`notes['arm'] = 'native'`,
  `notes['arm_note'] = 'no conditioning interface'`; a requested `eq` is recorded in `notes['arm_requested']`); their
  sign rule uses the native (R-3) set from the helper.
- Not testable (R-22 / R-23): in the eq arm a candidate whose target is an exact linear fit (residual share <= 1e-10)
  of its eq set gets score NaN, p None, never declared, `notes['not_testable_reason'] = 'exact fit: deterministic
  world under Z'` with `n_not_testable` / `not_testable_edges` (the edges only there, the `score.py` format; a reason
  string under `notes['not_testable']` made `score()` raise, fix-classic2 Q1); scored as not declared. Without the rule the F ratio
  compares two rounding errors (the E3 placebo got p = 0 on all 5 KPIs). Smoke counts (R2, n 1000, DEV seed
  3_000_000, real helper): E3 granger and pc 46 of 50, E1 pc 32 of 36, E2 0 of 90, E5 0 of 36. The not-testable TRUE
  edges are all secondary KPI -> KPI edges whose source lag is an exact copy of a lagged action in Z_eq (E3: lag_kpi:K0
  == P0@t-1, K1 == P0@t-2, K4 == P3@t-1; E1: K0 == P0@t-1, K1 == P1@t-1): E3 K0->K1, K1->K2, K1->K3; E1 K0->K2,
  K1->K3. Every primary (action -> KPI) true edge stays testable. R-24 observation noise removes these exact fits.
  Native on the same data: granger placebo p .017 / .016 / .012 / .57 / .47 (E3); PC pinv-fallback rate eq .058 /
  native .210 (E3), .021 / .143 (E1), 0 (E2, E5). PC eq CPU 5.5-23 s vs native 0.6-1.8 s at n 1000.
  granger, both arms: a source that adds no rank to the restricted model (collinear with it, e.g. E5 at kappa 0:
  lag K2 == P0@t-1) is also not testable, reason 'source collinear with Z', listed in `not_testable_edges` and
  `notes['not_testable_collinear']` (audit-classic2 B; before, an unlisted NaN).

## F6 rows

| method | F1 source / version | deviations from the reference, and why |
|---|---|---|
| pc | causal-learn 0.1.4.8; arms eq (default) / native (section Arms) (`uv add causal-learn`), PC pieces called directly: skeleton_discovery (stable), orient_by_background_knowledge, uc_sepset (uc_rule 0, priority 2), Meek; Fisher-z, alpha .05 | (1) background knowledge (brief): tiers {actions, lags, context} -> {KPIs t+1}; nothing but context into an action; no action -> lag / context. (2) Score = -log10 of the largest CI p computed for the pair during the skeleton search = pcalg's `pMax` (Kalisch et al. 2012, JSS 47(11): the maximal p-value over all CI tests of the pair); in causal-learn's stable skeleton every conditioning set of the removal depth is tested, so a removed pair has pMax > alpha and a kept one pMax <= alpha (native == score > -log10 alpha). Tau on this score acts like a different alpha on the recorded tests, not a re-run of the search. (3) Singular Fisher-z sub-matrices (deterministic worlds) are recomputed with the pseudo-inverse, same formula (causal-learn raises); for an exactly singular set the partial correlation is undefined, so the rate is reported per dataset (`notes['fisherz_pinv_fallback_rate']`, F7 fix) and per cell in results. (4) causal-learn's p = 2(1 - Phi) underflows to 0 beyond ~8 sd, so scores saturate at 300 (ties among very strong edges, irrelevant for declaration). (5) Unsigned -> pcorr_given_Z. KCI: available (`indep_test='kci'`), cost below. |
| notears | authors' code, xunzheng/notears@4a9ab19 `linear.py`, vendored byte-identical (Apache-2.0) | (1) columns z-scored (authors only centre; arbitrary units would make lambda1 / w_threshold scale-dependent, and NOTEARS exploits var-sortability, Reisach et al. 2021). (2) Background knowledge as L-BFGS-B (0, 0) bounds on forbidden entries (the authors' own diagonal mechanism; `notears_linear_bk` = vendored function with only the bounds line changed; equal to vendored when nothing is forbidden, test). (3) Score = unthresholded |W|; w_threshold .3 only for `native`. lambda1 .1, l2, max_iter 100, h_tol 1e-8, rho_max 1e16: authors' defaults. |
| shap_dag | XGBoost (the paper's regressor, sec V / Table II) `xgboost.XGBRegressor` 3.4.1 at the package defaults (100 trees, depth 6, lr .3; the paper states no hyperparameters, R-13), seeded, no early stopping; shap 0.52.0 `TreeExplainer`. Sensitivity option `regressor='hgbdt'` = the earlier E2 port's sklearn HistGradientBoostingRegressor(300 it, lr .1), `early_stopping=False` (F7 fix) | (1) Discovery stage only: the paper's DAG rule is qualitative ("the most influential" RCPs, sec IV-C1), no numeric threshold, so every threshold here is ours. (2) KPIs regressed on the actions only (the paper's RCPs), so KPI -> KPI candidates are NaN (not scorable). (3) Score = mean|SHAP| / sd(target) (absolute, comparable across KPIs; a relative "most influential" rule declares >= 1 parent for every KPI, null KPIs included; same adaptation as the E6 baseline). The E2 relative rule tau_rel .10 is `native`. (4) No DoWhy ATE / refutation / CATE (effect estimation, not discovery): sign by pcorr_given_Z (a linear backdoor coefficient has the partial-correlation sign). (5) Until the F7 fix: HistGBDT with sklearn's 'auto' early stopping (on at n > 10 000), i.e. a different estimator at n 24 000; now no early stopping at any n (test). |
| two_tower | `scripts/e2_baseline_gnn.fit_two_tower` (unchanged), `TwoTower` re-sized by the same rebinding as `baselines_disc._two_tower_sized`; torch 2.10.0. **Label: adaptation (self-supervised E2 port), not the published supervised model** (`notes['label']`) | (1) The E2 port is not the paper's model (self-supervised MSE, scalar encoders, relative threshold instead of supervised BCE + scaled cosine + sparsemax; E6_PUBLISHED_BASELINES.md sec 6): the paper trains on a ground-truth label matrix Y (sec VI-A, eq. 14) that the study does not give to methods, so the faithful model is not a truth-free discovery method. (2) Actions only (the port's params-only reconstruction) -> KPI -> KPI NaN. (3) Score (F7 fix) = row share a[k, p] / sum_p' a[k, p'] over the action columns of KPI k (raw gate in `notes['gate']`): each per-KPI head can absorb a rescaling of its gate row, so raw gates are not comparable across KPIs while the placebo tau is one cut over all KPIs (F7 audit: E2 R1 K4 null gates .8-7.6 vs true gates .36-.9 in other rows). Row max (a / max a) was rejected: it sets the top entry of every row to 1, also in rows without an action parent (the F4 generator's Y2), so the 2nd-largest DEV placebo score saturates at 1; sparsemax (the paper's normalisation) was rejected: not invariant to the row scale. `native` = relative rule .10 on the gate (unchanged). (4) l1 = 1e-3 (script default): a synthetic planted-edge check (3 seeds, n 1000, b .5 / 1.0) ranked planted gates equally at 1e-3 and .5, so the E6 value .5 (needed there for a family-masked input) is not used. Sizes d 16, r 8, hidden 32, lr .01, epochs 500 = the script's E2 run (function default 800). |
| corr | scipy 1.18.1 `stats.pearsonr` | none. p is exact for i.i.d. rows; approximate under R2 serial dependence. |
| pcorr_hac | own OLS (FWL) t-test with Newey-West (1987) HAC SE, Bartlett kernel, Andrews (1991) AR(1) plug-in bandwidth; == statsmodels 0.15.0 `OLS.fit(cov_type='HAC', cov_kwds={'maxlags': L, 'use_correction': True}, use_t=True)` (test); arms native (default, R-3 set) / eq_min (R-33); rows in time order; no RNG | (1) statsmodels' HAC needs an integer `maxlags` (no automatic bandwidth), so the Andrews S_T = 1.1447 (alpha(1) T)^(1/3) is computed from the AR(1) coefficient of the TESTED coefficient's score series v_t = x~_t u_t (by FWL the HAC variance of that coefficient depends on v only; adaptation: one series instead of Andrews' weighted vector of all score elements x_t u_t) and L = floor(S_T) (Bartlett weights 1 - j / (L + 1), slightly above Andrews' 1 - j / S_T); rho clipped to [-.99, .99], L capped at n - 2; no prewhitening (Andrews-Monahan 1992 not used). (2) Small-sample factor n / (n - k) and t(n - k) reference (Stata `newey` convention) instead of the asymptotic normal. (3) Exactly collinear / constant Z columns dropped before the fit (k = rank). (4) Not testable (R-22): exact fit of intercept + Z + source, or source collinear with [1, Z]. (5) Variant `inference='fixed_b'` (R-38): same statistic, Kiefer-Vogelsang (2005, Econometric Theory 21, 1130-1164) fixed-b reference t*(b) = W(1) / sqrt(Q(b)) for the Bartlett kernel at b = (L + 1) / n; p = E[2 Phi(-|t| sqrt(Q(b)))] over 20 000 simulated bridges (T 1000 as in KV, fixed seed, `_fixedb.py`); b < .01 interpolates to the normal p at b = 0. |
| granger | own OLS F-test; arms eq (default, conditional / VARX) / native (pairwise); == statsmodels 0.15.0 `grangercausalitytests` ssr_ftest at maxlag 1 (test) | **pairwise Granger** (bivariate; `notes['label']`): restricted model = the target's own lag only, not the other actions / lags (that would be conditional Granger); one lag (rows are one-step transitions); the source being the target's own lag is tested against the intercept. E3 only in the study. |

## F2 (own port == reference on identical inputs; `tests/test_xmethod_classic.py`)

- pc: adapter pipeline graph == `causallearn...PC.pc(...)` with the same background knowledge (array-equal).
- notears: `notears_linear_bk(forbid=None)` == vendored `notears_linear` (array-equal); forbidden entry stays 0.
- granger: p == statsmodels ssr_ftest (rel 1e-8) for lagged-KPI sources; == statsmodels OLS f_test for actions.
- corr: |r| == numpy corrcoef.
- shap_dag / two_tower: adapter scores == the script functions called directly (seed 0): shap_dag option `hgbdt` ==
  `fit_shap_importances` (n <= 10 000); default == XGBRegressor(package defaults) + TreeExplainer computed directly;
  two_tower raw gate (`notes['gate']`) == the script's S and score == S / row sum.
- pcorr_hac: coefficient, SE, p == statsmodels OLS HAC (same design, same L; rel 1e-10 / 1e-8) on synthetic data (rho 0 / .6, L auto / 0 / 4, with a collinear and a constant Z column) and on E2 R2 / E4 R3 / E5 R1 / E3 R2 worlds (every 3rd candidate, both arms); sign == sign of pcorr given the same Z (R-4); row-shuffled data with time_index give the same p.
- granger eq arm == statsmodels OLS F-test of the source given intercept + its helper set (R-25; rel 1e-6); pc eq arm:
  design nodes present, never candidates; native at alpha == score > -log10 alpha in both arms.

## F3 (known results)

| method | known result | ours | verdict |
|---|---|---|---|
| pc | textbook linear-Gaussian X1 -> X3 <- X2, X3 -> X4 -> X5 (n 10 000): CPDAG fully oriented | causal-learn `pc` returns exactly the 4 directed edges, none undirected | PASS |
| notears | authors' README example (seed 1, n 100, d 20, ER s0 20, lambda1 .1): {fdr 0, tpr 1, fpr 0, shd 0, nnz 20} | {fdr 0, tpr .95, fpr 0, shd 1, nnz 19}: the missed true edge has W_est = -.296 (threshold .3; true weight -.598). Same result on scipy 1.7.3 / 1.10.1 / 1.14.1 / 1.15.3 / 1.18.1 and python-igraph 0.8.3 / 0.9.11 / 0.10.8 / 1.0.0 (vendored code byte-identical to upstream) | NOT REPRODUCED (one borderline edge); reported, not worked around |
| shap_dag | earlier E2 study, 10 replicates (`scratchpad/shap_dag/shap_dag_lean.json`, main checkout; rows `runs/e2slice-recovery`, R-11): P0 -> K5 recovered 10 / 10, 16 param edges each, mean P0 / max mean\|SHAP\| for K5 = .8173 (HistGBDT) | default XGBoost (F7 fix, Kaggle `fixc-f7-a`): 10 / 10, 16 edges each, mean ratio .795; option `hgbdt` (pre-fix replay, `xmc-f3e2-b`): 10 / 10, 16 edges, .8165 (per replicate within .01) | PASS (regression check of our E2 port, not a reproduction of the paper, whose data are not available) |
| two_tower | earlier E2 study (`scratchpad/gnn_baseline/gnn_results.json`): 10 / 10, 12 param edges each, mean gate ratio .8474 | 10 / 10, 12 edges each, mean ratio .8563 (the F7 row-share score leaves this check unchanged: the ratio and the native rule are relative to the row max) | PASS (regression check of our E2 port; the paper's supervised model is not reproducible truth-free) |
| pcorr_hac fixed-b | KV (2005) Table I, Bartlett: cv(b) = a0 + a1 b + a2 b^2 + a3 b^3 (90 / 95 / 97.5 / 99 %; e.g. 97.5 %: 1.9600 + 2.9694 b + .4160 b^2 - .5324 b^3) | two-sided fixed-b p at cv(b): b .02 .1999 / .0991 / .0495 / .0197; b .1 .1992 / .0969 / .0484 / .0194; b .5 .2011 / .0998 / .0513 / .0210; b 1 .1999 / .0994 / .0500 / .0205 (nominal .20 / .10 / .05 / .02; KV's Bartlett cubic fits have R2 .9957-.9995) | PASS |

The E2 replay ran on Kaggle (job `xmc-f3e2-b`, `--pin off` because shap's numba rejects numpy 2.4 under the pinned
stack; numpy 2.0.2, scipy 1.16.3, sklearn 1.6.1, shap 0.51.0, torch 2.10.0+cpu). The small two-tower ratio difference
is the torch / BLAS stack (same seed 0, 500 epochs). CPU per replicate (n 4000, 8 params, 6 KPIs): shap_dag 50.7 s
(HistGBDT) / 9.0 s (XGBoost, `fixc-f7-a`: same image + xgboost 3.4.1 pip-installed at the uv.lock pin, R-16, logged),
two_tower 16.2 s.

## F4 (synthetic: power on planted edges, null FPR)

Generator `_classic_synth` (planted A0 -> Y0 +, A1 -> Y1 -, K0 -> Y0 +, K1 -> Y2 +; nulls A2, P_placebo, K2 and all
other pairs; global null b = 0). Primary declaration rule of each method. Score-only methods: tau tuned on 10
SEPARATE synthetic DEV datasets of the same cell (placebo rule), then applied to `reps` fresh datasets. 95% Wilson CIs.

| method | regime | n | reps | b | power A0->Y0 / A1->Y1 | power K0->Y0 / K1->Y2 | null-edge FPR (alt data) | FPR global null | raw p <= .05 under H0 |
|---|---|---|---|---|---|---|---|---|---|
| corr | R1 | 1000 | 200 | .5 | .92 / .93 | 1.00 / 1.00 | .004 [.003, .007] | .001 [.001, .003] | .050 [.044, .057] |
| granger | R1 | 1000 | 200 | .5 | .93 / .93 | 1.00 / 1.00 | .004 [.002, .007] | .001 [.001, .003] | .049 [.043, .056] |
| corr | R2 | 1000 | 200 | .5 | .64 / .55 | 1.00 / 1.00 | .007 [.005, .011] | .001 [.000, .002] | .050 [.044, .057] |
| granger | R2 | 1000 | 200 | .5 | .47 / .53 | 1.00 / 1.00 | .006 [.004, .009] | .001 [.000, .002] | .050 [.043, .057] |
| pc | R1 | 1000 | 100 | .5 | 1.00 / 1.00 | 1.00 / 1.00 | .121 [.107, .138] | .027 [.021, .035] | - |
| pc | R2 | 1000 | 100 | .5 | .81 / .89 | 1.00 / 1.00 | .034 [.027, .044] | .136 [.122, .152] | - |
| notears | R1 | 1000 | 100 | .5 | .89 / .88 | 1.00 / 1.00 | .001 [.000, .003] | .002 [.001, .006] | - |
| notears | R2 | 1000 | 100 | .5 | .40 / .52 | 1.00 / 1.00 | .001 [.000, .004] | .001 [.000, .003] | - |
| shap_dag (XGBoost, F7 fix) | R1 | 500 | 20 | 1.0 | 1.00 / 1.00 | not scorable | .068 [.046, .100] | .062 [.043, .089] | - |
| two_tower (row share, F7 fix) | R1 | 500 | 20 | 1.0 | .95 / 1.00 | not scorable | .047 [.029, .075] | .064 [.045, .092] | - |
| shap_dag (pre-fix: HistGBDT) | R1 | 500 | 20 | 1.0 | .95 / 1.00 | not scorable | .059 [.038, .089] | .021 [.011, .040] | - |
| two_tower (pre-fix: raw gate) | R1 | 500 | 20 | 1.0 | .75 / .90 | not scorable | .053 [.034, .082] | .076 [.055, .106] | - |

Reading: corr / granger hold the nominal level (the raw-p column) and BY keeps the declared FPR far below it. For
score-only methods the realised FPR is set by the placebo tau, which with only 30 DEV placebo scores is noisy: the PC
R1 tau (.87 = p < .13 on every recorded test) and the PC R2 global-null tau (.85) fell low, and the other null edges
were then declared at the SAME rate as the placebo on test data (PC R1, 30 datasets: placebo 12 / 90 vs other nulls
~ 11%) - i.e. the placebo is exchangeable with the nulls, which is what the rule needs; the study's DEV sets are larger.
shap_dag and two_tower F4 used n 500 / 20 reps (cost); the F7-fix rows ran on Kaggle `fixc-f7-a` (code 9214c0b), the
pre-fix two_tower row on `xmc-f3e2-b`, the rest locally. With 20 reps the FPR CIs overlap between pre- and post-fix.
The synthetic R2 here has i.i.d. KPI noise; the harness R2 (20 setpoint blocks) is harsher, see Integration.


### pcorr_hac F4 (null level under serial dependence; `scratchpad/xmethod/pcorr_hac_fidelity.py`, 2000 reps per cell)

Synthetic null through the adapter (arm native): X = .5 W1 - .3 W2 + e, Y = .4 W1 + .2 W2 + u, W AR(1)(phi), e, u
independent AR(1)(rho), Z = {W1, W2, P_placebo}, X _||_ Y | Z exactly. Rejection of X -> Y at .05 [95 % Wilson CI];
plain = the R-5 partial-correlation t-test on the same rows and Z. Median Andrews L in brackets.

| scenario | n | pcorr_hac (t) | pcorr_hac fixed-b (R-38) | plain pcorr |
|---|---|---|---|---|
| iid | 500 / 1000 / 4000 | .049 [.040, .059] / .054 [.045, .064] / .051 [.042, .062] (L 1) | .047 [.039, .057] / .053 [.044, .063] / .051 [.042, .062] | .046 / .052 / .055 |
| AR(1) rho = phi = .5 | 500 / 1000 / 4000 | .070 [.060, .082] / .062 [.052, .073] / .055 [.045, .065] (L 5 / 7 / 11) | .063 [.053, .075] / .059 [.050, .070] / .053 [.044, .064] | .131 / .124 / .121 |
| AR(1) rho = phi = .8 | 500 / 1000 / 4000 | .104 [.091, .118] / .084 [.073, .097] / .060 [.050, .071] (L 14 / 18 / 30) | .091 [.079, .104] / .080 [.068, .092] / .056 [.047, .067] | .358 / .345 / .348 |

At .01: pcorr_hac .009-.017 (iid, rho .5), .034 / .028 / .012 (rho .8); plain .041-.044 (rho .5), .22-.24 (rho .8).
The i.i.d. `P_placebo` -> Y: all tests .043-.062 in every cell. Verdicts (R-30: INVALID if the CI lower bound > .05,
VALID if the upper bound <= .075; corrected 2026-10-03, the R-32 text misread rho .5): X -> Y, plain pcorr INVALID in
all 6 serially dependent cells; pcorr_hac (t) VALID in the 3 iid cells and at n 4000 (rho .5, .8), INVALID at n 500 /
1000 for rho .5 and .8; fixed-b VALID in the 3 iid cells, rho .5 n 1000 / 4000 and rho .8 n 4000, INVALID at rho .5 n
500 and rho .8 n 500 / 1000. The i.i.d. placebo is marginally INVALID (lower bound .0504-.0518) for both HAC variants at
iid n 4000 and rho .8 n 1000 (plain .059 / .0585, VALID): 2 of 9 placebo cells per HAC variant, read as chance at 2000
reps. Cause of the HAC excess: the known finite-sample downward bias of Bartlett-kernel HAC variance estimates under
strong persistence; reported, not tuned (no prewhitening). R-38 fixed-b variant (Kaggle `hac2-f4b-1`, same datasets: the
t column reproduces the R-32 local run; `scratchpad/xmethod/results/pcorr_hac_f4b.json`): max |rate - .05| over the 9
cells (X -> Y, .05) = .0405 (fixed-b) vs .0535 (t), so the PRE-FREEZE set-D member (R-38, decided on this synthetic grid
only) is **pcorr_hac with `inference='fixed_b'`**; the t variant is reported as secondary. At .01: t .009-.017 (iid, rho
.5), .034 / .028 / .012 (rho .8); fixed-b .010-.017, .027 / .024 / .010. F4 power: planted SYN edges found with the
right sign (n 3000), SYN global null (n 1000) nothing declared (tests).

## Integration on the harness worlds (merged feat/v2; `scripts/xm_classic_integration.py`)

n 1000, test seed 3_000_000, score-only tau from DEV seeds 3_000_001-3 (3 datasets: noisy, integration only), E4 at
lam 1.5, `cdd_oran.xmethod.runner run` + `cdd_oran.xmethod.score`. 0 errors in 62 jobs (corr / pc / notears 12 cells
each, granger 2: before the arms of R-17, i.e. pc / granger = NATIVE arm; shap_dag and two_tower 12 each rerun after
the F7 fixes on Kaggle `fixc-f7-a`, code 9214c0b; pre-fix values in git history). Primary set (placebos excluded):
precision / recall.

| cell | corr | granger | pc | notears | shap_dag | two_tower |
|---|---|---|---|---|---|---|
| E1 R1 | .80 / 1.00 | - | .50 / 1.00 | 1.00 / 1.00 | .80 / 1.00 | .50 / 1.00 |
| E1 R2 | .27 / 1.00 | - | 1.00 / 1.00 | 1.00 / 1.00 | .57 / 1.00 | .50 / 1.00 |
| E2 R1 | 1.00 / .38 | - | .67 / .50 | 1.00 / .25 | .94 / 1.00 | .77 / .62 |
| E2 R2 | .48 / .75 | - | .86 / .38 | 1.00 / .25 | .94 / .94 | .89 / .50 |
| E3 R1 | 1.00 / 1.00 | 1.00 / 1.00 | .80 / 1.00 | 1.00 / 1.00 | .57 / 1.00 | .67 / 1.00 |
| E3 R2 | .31 / 1.00 | .44 / 1.00 | .80 / 1.00 | 1.00 / 1.00 | .50 / 1.00 | .50 / 1.00 |
| E4 R1 | 1.00 / 1.00 | - | 1.00 / 1.00 | nan / 0 | 1.00 / 1.00 | 1.00 / 1.00 |
| E4 R2 | 1.00 / 1.00 | - | 1.00 / 1.00 | nan / 0 | 1.00 / 1.00 | 1.00 / 1.00 |
| E4 R3 | 1.00 / 1.00 (sign wrong) | - | 1.00 / 1.00 | nan / 0 | 1.00 / 1.00 | 1.00 / 1.00 |
| E4 R4 | 1.00 / 1.00 (sign wrong) | - | 1.00 / 1.00 (sign wrong) | 1.00 / 1.00 (sign wrong) | 1.00 / 1.00 (sign wrong) | 1.00 / 1.00 (sign wrong) |
| E5 R1 | 1.00 / .33 | - | .80 / .67 | 1.00 / .33 | 1.00 / 1.00 | .80 / .67 |
| E5 R2 | .29 / .67 | - | .67 / .33 | .67 / .33 | 1.00 / .67 | .80 / .67 |

Observations (one seed; not results): (a) corr / granger under the harness R2 declare the tuning placebo 2-4 times
and reach FDP .5-.7: the 20 setpoint blocks make rows serially dependent, so the i.i.d. p-values are invalid there
(a property of the method, kept). (b) PC hits singular Fisher-z sub-matrices in the deterministic E1 / E3 (pinv
fallback: up to 494 of 2348 tests). (c) With the observed confounder (E4 R3) PC, shap_dag and two_tower get the
sign right (pcorr given Z) while corr does not; in R4 every method has the wrong sign, as R-1 states. (d) shap_dag
and two_tower cannot score KPI -> KPI (secondary recall 0 by design); NOTEARS shrinks E4 P0 -> K0 to 0.

## Cost (CPU-s, peak RSS; 1 thread)

Measured on Kaggle job `xmc-cost-a` (4 vCPU, each run pinned to 1 thread, a fresh process per (method, n), timeout
7200 s = the R-13 budget of 2 CPU-h per (method, dataset)). Code at cc57165 (adapters identical to HEAD). The world is
E2 R2 at DEV seed 3_000_000 (9 actions, 6 KPIs, 90 candidates); granger uses E3 R2 (5 actions, 5 KPIs, 50 candidates).
The image ran with `--pin off`: numpy 2.0.2, scipy 1.16.3, sklearn 1.6.1, shap 0.51.0, torch 2.10.0+cpu, and
causal-learn 0.1.4.8 installed at the uv.lock pin (`uv pip`, logged in install.log). CPU-s covers one `run()` on one
dataset; data generation (<= .45 s) is excluded. `tune()` adds no fit: it is the same `run()` on each DEV dataset of
the cell. Raw rows are in `scratchpad/e6_dev/runs/xmc-cost-a/out/cost.jsonl`.

CPU-s per dataset (peak RSS MB in brackets):

| method | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|
| corr | .04 (99) | .04 (98) | .04 (99) | .05 (100) | .08 (106) |
| granger (E3) | .03 (98) | .03 (99) | .04 (99) | .06 (101) | .14 (106) |
| notears | .42 (80) | .53 (81) | 2.36 (84) | 4.62 (90) | 14.4 (111) |
| pc (fisherz, default) | .27 (221) | .40 (220) | .79 (224) | 1.03 (228) | 2.51 (247) |
| pc + KCI | 253 (244) | 4113 (307) | not run | not run | not run |
| shap_dag (XGBoost, F7 fix; `fixc-f7-a`) | 1.53 (358) | 2.53 (359) | 8.36 (360) | 16.7 (363) | 51.0 (376) |
| shap_dag option hgbdt, no early stopping (`fixc-f7-a`) | - | - | - | - | 159 (357) |
| shap_dag (pre-fix: HistGBDT, early stopping 'auto' at n > 10 000) | 5.14 (337) | 12.9 (339) | 33.7 (340) | 61.4 (343) | 116 (359) |
| two_tower | 6.72 (313) | 5.30 (300) | 13.5 (331) | 26.6 (381) | 74.8 (572) |

- All six methods in their configured (author-default) form finish every n up to 24000 far inside the budget. The
  largest cost is now two_tower at 74.8 CPU-s (shap_dag with XGBoost 51.0 s; the HistGBDT option without early
  stopping 159 s) and the largest RSS is two_tower at 572 MB, all at n 24000. They scale to n 8000 and 24000. The
  fixc-f7-a rows ran on the same image class (`--pin off`, 4 vCPU, 1 thread per run) plus xgboost 3.4.1.
- PC with KCI (causal-learn's kernel CI test, an optional variant and not the adapter default) costs 253 s at n 500
  and 4113 s (57 % of the budget) at n 1000. From 500 to 1000 the cost grows 16.2x (about n^4). At n >= 4000 it was
  not run. The same growth projects to about 1e6 s at n 4000 (n^3 would still give about 2.6e5 s), so it is expected
  to be "infeasible at n >= 4000". This is a projection, not a measurement; an R-13 measurement would be one 7200 s
  timeout run per n.
- These per-dataset figures supersede the rough ones in earlier status notes, which were taken on other worlds and
  machines.
