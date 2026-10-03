# R-42 step 1: why pmrt_eq misses E2 / E5 edges (DEV, n 1000, kappa .25)
DEV records (xm/dev-runs a7b6e36), measurement seeds 3_000_100-119 (R1) / -159 (R2), read-only:
`pmrt_nl_diag.py` -> `diag_table.json`; shapes: `pmrt_nl_shapes.py` -> `shapes.json` (Monte Carlo on the frozen SCM
functions, R1 design, no study seeds). Score arms and "pmrt tau" use the R-29 placebo cutoff (like-for-like). Shares
of Var(Y_k) + noise: lin = linear in a; main = Var E[Y|a]; even = symmetric part of main; total = Sobol total effect
(main + interactions); resid = share left after a LINEAR adjustment on the other actions.

## Cell means (per-edge recall)
| cell | pmrt raw .05 | pmrt BY | pmrt tau | shap_dag tau | two_tower tau | pc_eq tau |
|---|---|---|---|---|---|---|
| E2 R1 (16 edges) | .375 | .319 | .419 | .953 | .684 | .347 |
| E2 R2 | .444 | .279 | .432 | .904 | .420 | .297 |
| E5 R1 (6 edges) | .467 | .367 | .425 | .967 | .750 | .442 |
| E5 R2 | .542 | .464 | .531 | .670 | .703 | .503 |

pmrt_eq truth-null raw-p rate .055 / .058 / .036 / .051: validity is not the problem. The like-for-like gap (pmrt tau
vs shap tau) is .53-.54 in R1: the loss is power, not the BY cutoff.

## Per edge: pmrt raw-p recall R1 / R2, shap tau R1 / R2, R1 shape
| edge | pmrt | shap | lin | main | even | total | shape (SCM) -> what a statistic needs |
|---|---|---|---|---|---|---|---|
| E2 P0->K0 | 0 / .52 | 1 / 1 | 0 | .51 | .51 | .67 | bump centred in range (width P1) -> EVEN in v |
| E2 P0->K1 | 0 / .52 | 1 / 1 | 0 | .40 | .40 | .67 | bump in P0+P2 centred -> even in v |
| E2 P3->K2 | 0 / .50 | 1 / 1 | 0 | .22 | .22 | .35 | width param, even in P3 -> even in v |
| E2 P4->K3, K4 | .3,.15 / .2,.27 | 1 / .93,.98 | 0 | .07 | .07 | .25 | width, even + interaction with P1+P5 |
| E2 P1->K3, K4 | .1 / .2,.1 | 1 / .85,.93 | 0 | .002 | 0 | .51 | bump in P5+P1: PURE interaction with concurrent P5 |
| E2 P2->K1 | 0 / .08 | .9 / .45 | 0 | .003 | 0 | .16 | bump in P0+P2: pure interaction with concurrent P0 |
| E2 P0->K5, P7->K5 | 0 / .17,.27 | 1,.75 / .98,1 | 0 | .002-.01 | ~0 | .88-.92 | narrow bump (width P6 <= 4) in P0+P7: pure interaction, sharp |
| E2 P6->K5 | .35 / .1 | .6 / .33 | .003 | .006 | 0 | .05 | monotone only inside the K5 bump (rare) |
| E2 P0->K2, P1->K0/K1, P5->K3/K4; E5 P0->K0/K2 | 1 / .52-1 | 1 / 1 | .02-.94 | .27-.94 | | | off-centre / monotone: linear works |
| E5 P2->K1 (G2) | 0 / .60 | 1 / 1 | 0 | .91 | .91 | .93 | dominant base bump centred at G2 = 25 = range mid -> even |
| E5 P3->K3 | .1 / .52 | 1 / 1 | 0 | .94 | .94 | .95 | bump centred at 0 = range mid -> even |
| E5 P1->K1 (G1) | .55 / .07 | 1 / 0 | .003 | .005 | 0 | .015 | small monotone 2 G1 (+ gate switch); resid 1.0: the G2 bump (.91 of Var) is not removed by the LINEAR adjustment |
| E5 P0->K1 | .15 / .07 | .8 / .02 | 0 | .003 | 0 | .011 | gated (5 % occupancy in G1, G2) subdominant 0.2 P0: interaction with concurrent G1, G2 |

## Diagnosis

1. Symmetric bumps (E2 P0->K0, P0->K1, P3->K2, P4->K3/K4; E5 P2->K1, P3->K3): the effect is even about the R1
   range midpoint, so Cov(v, Y) = 0 and the linear z is pure noise (median |z| .5-.8 in R1). In R2 the local slope at
   each setpoint has the setpoint's sign, so sum v w cancels across the 20 blocks (median z ~ 0, median |z| ~ 2:
   a hit only when blocks on one side dominate). Needed: even functions of v (v^2-type) and v x setpoint terms.
2. Pure interactions with a CONCURRENT action (E2 P1->K3/K4 with P5, P2->K1 with P0, P0->K5 / P7->K5 with each
   other; E5 P0->K1 gated by G1, G2): main share ~0, total share .16-.92. No function of v alone sees them; the
   statistic needs v x (concurrent actions / state), admissible since those are fixed under the redraw (R-8).
   P0 / P7 -> K5 is a sharp bump (width <= 4 on ranges 200-250): only an adaptive (learned) profile can resolve it.
3. Adjustment noise (E5 P1->K1, also every E2 edge: resid .67-1.0): the linear ridge leaves the dominant
   nonlinear effects of other actions in w, so even a monotone effect drowns. A nonlinear PREDICTABLE adjustment
   (past-only, no focal draw at t) shrinks Var(w) and keeps the martingale argument. (E5 P1->K1 in R2: G1's dither
   is +-.25, so 2 G1 moves K1 by <= .5 vs noise sd 6: no method finds it, shap .0.)

## Step 2 candidates (each a function of the redrawn v, with W, X fixed and predictable)
- N1 basis score: U = sum_t phi(v_t) (x) psi(s_t) w_t, phi = design-centred Legendre polynomials of v (deg 1-3),
  psi = [1, focal setpoint basis]; T = U' Sigma^-1 U, Sigma = the redraw covariance (known). Targets class 1.
- N2 conditional kernel (HSIC-type): Gaussian kernel on v (design-centred) x Gaussian kernel on the fixed state
  (concurrent actions, setpoints), via random Fourier features; T = ||U||^2. Targets classes 1-2 (coarse).
- N3 learned predictable matched filter: a past-only gradient-boosted model g(v, x) (refit at the geometric
  blocks); h_t(v) = g(v, x_t) - E_design g(., x_t) (the matched filter), w_t = y_t - E_design g(., x_t) (a nonlinear
  predictable adjustment); T = sum h_t(v_t) w_t / sqrt(sum Var_design h_t w_t^2). Targets 1-3. The brief's cross-fit
  "held-out gain" needs a refit per redraw (9999x) and uses future rows whose KPIs can depend on v_t (Q1).
