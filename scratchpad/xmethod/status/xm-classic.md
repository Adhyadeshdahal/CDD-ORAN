# xm-classic status (updated 2026-10-02)  -- HAND-BACK below (complete)

## Rulings applied
R-2/R-6 BY per family (primary action->KPI incl. P_placebo | secondary KPI->KPI | diagnostic P_placebo_conf, families
read from Dataset.meta); score-only tau = harness `score.placebo_tau` (2nd-largest DEV placebo score). R-3/R-4 sign
`pcorr_given_Z` for pc/shap_dag/two_tower. R-9 N/A (no resampling). R-10 diagnostic family. R-11 E2 rows read-only.
R-13: authors' own tests/defaults kept; cost per dataset measured at every n (pending).

## Running
- nothing. Kaggle `xmc-cost-a` COMPLETE (exit 0, 4830 s wall); cost table filled in the doc.

## QUESTIONS
- Q-C3 answered: pip inside ephemeral cloud kernels is fine, pinned to the uv.lock version (causal-learn==0.1.4.8),
  logged in job output; PC cost measured on the cloud only. Not needed: the image had uv (causal-learn 0.1.4.8 logged).

## HAND-BACK
Deliverables (branch xm/xm-classic, feat/v2 merged in at 8e3cbdb; never pushed):
- cdd_oran/xmethod/methods/{pc,notears,shap_dag,two_tower,corr,granger}.py (api.Method; tune() = placebo tau,
  run() = Result with score/p/sign/declared + notes family/native_declared/sign_rule); `_classic_common.py`;
  lazy registry `classic.py` (METHODS); `_classic_synth.py` (test generator); `_vendor/notears` (xunzheng/notears
  @4a9ab19, Apache-2.0, byte-identical). Runner refs: `cdd_oran.xmethod.methods.<module>:<Class>` (PC, Notears,
  ShapDag, TwoTowerM, Corr, Granger). Scripts: scripts/xm_classic_fidelity.py, scripts/xm_classic_integration.py.
- docs/xmethod/FIDELITY_CLASSIC.md: F6 rows, F2, F3, F4, integration table, Cost section.
- Dependency added: causal-learn 0.1.4.8 (`uv add`). Uses baselines group (shap, sklearn, scipy) + threadpoolctl.
Tests: tests/test_xmethod_classic.py 29/29 pass (contract, BY families, tau rule, sign rule, F2 equivalences).
Fidelity gates:
- F1: causal-learn 0.1.4.8; NOTEARS authors' code; repo E2 scripts (shap 0.52, sklearn 1.9.1; torch 2.10); scipy.
- F2 PASS all six (pc == causal-learn pc() with same BK; notears_bk == vendored; granger == statsmodels; wrappers).
- F3: pc PASS (textbook CPDAG); shap_dag PASS (E2: 10/10, 16 edges, ratio .8165 vs .8173); two_tower PASS (10/10,
  12 edges, .856 vs .847); notears FAIL-borderline: README example 19/20 (missed edge W -.296 vs threshold .3) on
  scipy 1.7-1.18 / igraph 0.8-1.0 - not worked around.
- F4: corr/granger level .050 [.044,.057], power .92 (R1); score-only FPR tracks the placebo tau (noisy at 30 DEV
  placebo scores; placebo exchangeable with nulls). Table in the doc.
- F5: placebo rule only; two_tower l1 kept at the script default 1e-3 (synthetic check: 1e-3 == .5).
Integration: 62 runner jobs on harness worlds (n 1000, seed 3_000_000), 0 errors, all 6 methods.
Cost (Kaggle xmc-cost-a, E2 R2 / granger E3 R2, CPU-s per dataset at n 500/1000/4000/8000/24000, 1 thread):
corr .04-.08; granger .03-.14; notears .42/.53/2.4/4.6/14; pc .27/.40/.79/1.0/2.5; shap_dag 5.1/13/34/61/116;
two_tower 6.7/5.3/14/27/75 (peak RSS <= 572 MB). All six scale to n 24000 inside the 2 CPU-h budget. pc+KCI
(optional variant): 253 s n 500, 4113 s n 1000; n >= 4000 not run, projected ~1e6 s = infeasible (projection only).
Deviations (all in F6): NOTEARS z-scored + BK as (0,0) bounds; PC score = -log10 max CI p (pinv fallback in
deterministic worlds; p underflow caps score at 300); shap_dag absolute mean|SHAP|/sd, actions only (KPI->KPI NaN);
two_tower = E2 port (not the supervised sparsemax paper model), actions only; Kaggle runs used --pin off (numba vs
numpy 2.4). Findings: corr/granger p invalid under harness R2 setpoint blocks (placebo declared 2-4x).
Proposals: api.Result could carry `family` per EdgeResult (now in notes['family']).
  If pc+KCI enters the protocol: one 7200 s timeout run per n >= 4000 to measure "infeasible" (R-13).
