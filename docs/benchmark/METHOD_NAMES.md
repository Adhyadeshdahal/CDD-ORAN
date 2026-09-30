# Method names (decided 2026-09-30)

## PMRT: Predictable Matched-Filter Randomization Test
PMRT is the discovery method used from protocol v4 on. It is a design-based conditional randomization test. Its
statistic correlates the propensity-centred assignment with a predictably residualised, matched-filtered outcome.
Declarations use prior-weighted, directional Benjamini-Yekutieli control across action x relation x KPI hypotheses.

| component | what it is | code |
|---|---|---|
| Assignment score | v = sgn x (level - E_pi0[level]), from the unit's logged propensity row | `crt_units_v2.design_regressor` |
| Predictable adjustment | ridge prediction from pre-window KPI bins, ctx and past-only modes, plus running (past-only) stratum means; valid when unit existence depends on earlier assignments | `crt_units_plus` |
| Matched filter | spatio-temporal weights K = Sigma^-1 mu over receiving-cell slots x time bins, sign-constrained ("loadsp"), fitted on old data and frozen | `crt_units_plus` (arm `loadsp_c`) |
| Robustness | predictable Huber clip (`_c`) | `crt_units_plus` |
| Inference | p-values by redrawing the logged per-unit propensities (CRT); asymptotically valid (martingale CLT), NOT exact; the exact companion is the e-process | `crt_units_plus`, `eprocess_units` |
| Declarations | weighted BY with frozen prior weights from independent data, one-sided where the prior \|z\| >= 3 (`wby1s`) | `mscr_multi` |

## Label mapping
- **"MSCR+"**, "MSCR+ loadsp_c + wby1s" and the artifact `E6P_MSCRPLUS_V4_FROZEN.json` are all **PMRT**. The frozen
  protocol v4 and the artifact keep the old label: frozen files are never edited.
- The module file names (`crt_units_plus.py`, `mscr_multi.py`, `mscr_*`) are unchanged. The artifact pins their
  sha256, so they must not be edited while v4 / option-(a) analyses depend on it.

## Earlier methods (historical names kept)
- **MSCR** (`cdd_oran/discovery/mscr.py`): Max-Stratified Correlation-Ratio test, a model-free CI test. It is the
  method of the E2-E5 studies, and its name stays with those results.
- **MSCR-CRT v1** (step 1 v1): the MSCR statistic inside a CRT.
- **"MSCR-CRT v2"** (step 1 v2/v3): despite the name, its statistic is a design-centred linear score with episode x
  sgn fixed effects; MSCR survives only as a descriptive side output. It is PMRT's direct predecessor, and its FE
  demeaning was later found biased under mode-dependent unit sets (fixed in PMRT).
