READY-TO-MERGE (worker calib, branch xm/kappa-calib from feat/v2 1191678; no push, no merge)

# kappa-calib: R-24 observation-noise calibration from E6-P DEV-v4

Data: ONLY the 20 DEV-v4 episodes (seeds 188000-188019, kernel bishalpanta/e6p-disc-v4dev-1-a, read only); enough.
Script scratchpad/xmethod/kappa_calib.py (method in its docstring). Run: Kaggle bishalpanta/xmk-kappa-1 (1 session,
395 s wall), in-cloud build from the source kernel; data sha256 a3925aa3... identical to the local build.
Rows: 28 800 (20 ep x 60 scored 10 s windows x 24 cells). Targets: lab KPI summed over the panel window.
Predictors ("base", 83): own 5 KPIs lags 1-6; all panel knob_own / knob_nbr states lags 0-3 (0 = in force at window
start); is_macro. Model: HistGradientBoosting (defaults, rs 0), one per KPI, pooled over cells. Train even j, test odd j.
u = 1 - R^2 (OOS, SST about test mean = marginal spread, as E5's sigma_k); kappa = sqrt(u / (1 - u)).
CI: 200 episode-bootstrap refits (resampling within each half), percentile 95 %, rng default_rng([7824, b]).

| KPI | frac 0 | u | kappa | 95 % CI |
|---|---|---|---|---|
| pv | .82 | .288 | .636 | [.532, 1.111] |
| v | .41 | .061 | .254 | [.229, .337] |
| e | .00 | .015 | .125 | [.119, .139] |
| rlf | .98 | 1.004 | inf (R^2 < 0 in 200/200 reps) | [inf, inf] |
| load | .04 | .023 | .154 | [.126, .206] |

PROPOSED kappa_E6 = .254 (round .25) = median over the 5 KPIs, 95 % CI [.229, .337] (the median KPI is v in 200/200
reps). R-24 sweep then = {.13, .51}.
Alternatives: mean over KPIs = inf (rlf); mean over the 4 finite KPIs .292 [.263, .431]; kappa of median u = .254 (same);
per-KPI above. Robustness of the median (point): swapped split .276, cross-fit .265; predictor variants (+ knob state at
window end, + neighbour KPI lags 1-3, + episode time, all) .253-.257. Descriptive "within" R^2 (SST about per-(episode,
cell) means, i.e. time variation only, between-cell level removed): median kappa .476 [.426, .697]
(pv .90, v .48, e .31, load .41, rlf inf).

CAVEAT (for the paper): u counts all variation the predictor does not explain, incl. unmodelled signal (other cells'
traffic, UE mobility, rare-event Poisson noise in pv / rlf), not only measurement noise. kappa_E6 is therefore an UPPER
bound on pure measurement noise (conservative for discovery). rlf (98 % zero windows) is unpredictable at 10 s: inf.
The bootstrap shifts u up (duplicated train episodes = less distinct training data), so CIs lean high (conservative).

## HAND-BACK
- Deliverables: kappa_calib.py; results/kappa_calib/{kappa_calib.json (all points, variants, CIs, features, versions),
  boot_u.npz (200 x 5 u, pooled / within), kaggle_job.log, kaggle_job.json}. Local cache (cache/, 2 MB) not committed.
- Env: Kaggle sklearn 1.9.1 (pip-pinned to uv.lock, R-16), numpy 2.4.2 / scipy 1.18.1 (pin match). No new uv deps.
- Seeds: no new data; DEV-v4 corpus read only; no EVAL / placebo / gt episodes read. Bootstrap tag 7824 (scratch).
- Deviations: none from the brief. My choices (not in the brief): 10 s windows only when fully scored (t >= 120 s);
  KPI lags 1-6, knob lags 0-3; is_macro included; "current" knob = state at window start (end-of-window = variant);
  B 200; median over KPIs taken on kappa (equal to kappa of median u).

## QUESTIONS
- Q1 Primary spread: pooled R^2 (.25, matches E5's marginal sigma_k) vs within-cell R^2 (.48). I used pooled per R-24
  / E5; confirm. The sweep upper end (.51) roughly covers the within-cell value.
- Q2 Rounding: use .254 as is or .25?
