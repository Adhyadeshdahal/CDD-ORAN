# R-42 step 2: candidates and why each is a valid PMRT statistic

Code: `cdd_oran/xmethod/methods/pmrt_nl.py` (version `pmrt-nl-v1`; docstring = the full argument). Machinery shared
with pmrt_core: `assignment` (the known random part v and its law), the RNG stream `default_rng([seed, 7801, action,
split])`, B <= 9999 with Besag-Clifford h = 20 (R-9), BY over the primary action -> KPI family (R-2, R-6), no clip
(R-14), not-applicable rule (no design -> no test). All statistics are one-sided (large = evidence); the sign is the
linear PMRT sign sign(sum_t v_t w_t) on the same w. Settings fixed before any result, never tuned on DEV.

| id | statistic T(v) (W, psi, h fixed under the redraw) | adjustment w | targets (DIAG.md) |
|---|---|---|---|
| N1 `poly` | U = sum_t phi~(u_t) (x) psi_t w_t, phi = (u, u^2, u^3) centred by design moments, psi = [1, s, s^2 - mean] (s = focal setpoint, R2 only); T = U' Sigma^+ U, Sigma = sum_t w_t^2 Cov_design(phi_t) (x) psi_t psi_t' | pmrt_core ridge (`poly`) / gbm (`poly_g`) | symmetric bumps, setpoint-dependent slopes |
| N2 `rff` | same U with 10 random Fourier features of u (Gaussian kernel, length 1) and psi = [1, 20 RFF of the standardised state: concurrent designed actions, focal setpoint, context; length sqrt(dim)]; T = sum_j U_j^2 / Sigma_jj (HSIC-type) | ridge (`rff`) / gbm (`rff_g`) | + interactions with concurrent actions |
| N3 `gbm` | learned predictable matched filter: g_k past-only XGBoost (200 trees, depth 4, eta .1, 1 thread, seed 0) of y_k on [Z_eq, every designed action at t], refitted at pmrt_core's geometric blocks; h_t(v) = g_k(A_t with the focal random part = v, x_t) - m_t, linear interpolation on 25 grid points (exact on a categorical support); T = sum_t h_t(v_t) w_t / sqrt(sum_t Var_design(h_t) w_t^2) | gbm: w_t = y_t - m_t, m_t = E_design g_k(., x_t) | + gates, sharp interactions, nonlinear nuisance |

Candidates for the R-42 selection rule: N1 `poly`, N2 `rff`, N3 `gbm`. `poly_g` / `rff_g` are reported (recall)
so the statistic and adjustment effects are separable (orchestrator, Q2), not selection candidates.

## Exactness argument (per candidate; the same structure for all three)

Let F_{t-1} = everything in rows < t, x_t the row's fixed covariates (lagged KPIs, context, setpoints, lagged actions,
concurrent designed actions). By the design, v_t is independent of (F_{t-1}, x_t) with a known law (R-8 for the
concurrent actions); under H0 (v_t has no effect on y_t) also of y_t.

1. Exact case. If no quantity held fixed (W; psi; Sigma; the gbm model, hence h and m) depends on the focal action's
   random parts, then T(v_obs) and every T(v_b) are i.i.d. draws of the same function under H0 and the CRT p-value
   is exact in finite samples, for ANY statistic. This holds e.g. in R1 with covariates "r3" and no KPI dynamics
   driven by the action (unit test `test_null_level_exact_case`).
2. General case (the PMRT argument, unchanged). The fixed quantities are predictable: they use rows < t (the ridge /
   gbm fits are past-only, refitted at block boundaries) and x_t, but may depend on the focal action's PAST draws
   (lagged-action covariates, KPIs it drives, refits). Each statistic is built from d_t = phi~(v_t) (x) psi_t w_t
   (N1, N2) or h_t(v_t) w_t (N3), where the v_t factor has design mean 0 given (F_{t-1}, x_t) (centred by the exact
   design expectation: polynomial moments, categorical sums, the exact piecewise-linear integral for uniform designs;
   Gauss quadrature for normal / clipped normal) and w_t is (F_{t-1}, x_t, y_t)-measurable. So sum_t d_t is a
   (vector) martingale whose predictable covariance is exactly the covariance of the redraw distribution (Sigma for
   N1 / N2, sum Var_design(h_t) w_t^2 for N3). By the martingale CLT the sampling law and the redraw law of the
   statistic (a continuous function of the normalised sum: a quadratic form for N1 / N2, the scalar for N3) agree
   asymptotically: the CRT p-value is asymptotically valid, not exact (as pmrt_core). Finite-sample accuracy is what
   the F4 and DEV gates measure.
3. Why the gbm model is fitted past-only (Q1, orchestrator ruling 2026-10-03): a cross-fitted model uses future rows, whose
   KPIs can depend on v_t (e.g. through KPI dynamics); holding such a model fixed while redrawing v_t breaks step 2,
   and refitting it for every redraw costs up to 9999 fits per action.
