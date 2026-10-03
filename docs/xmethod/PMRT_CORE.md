# PMRT core (cross-method study, E1-E5)

Code: `cdd_oran/xmethod/methods/pmrt_core.py` (adapter `pmrt_core`, version `pmrt-core-v1`). Full PMRT on E6-P:
`cdd_oran/decision/pmrt.py` + `fdr_layer.py` (`docs/benchmark/METHOD_NAMES.md`).

## What it tests

For an action column a with a known assignment design and a KPI k, the null hypothesis is that the random part of
a's assignment at row t has no effect on k at t + 1. The test redraws only that random part from its known
distribution. Every other column, and every non-random part of a, stays fixed. Candidates whose source has no known
design (a lagged KPI, or an action in regime R4) are reported as not applicable. The method never falls back to a row
permutation for them.

## Statistic

1. **Assignment score.** v_t = A_t - E[A_t] under the design. In R1 (i.i.d. actions) this is the centred action.
   In R2 (setpoint + dither) it is the centred dither; the setpoint is not randomised and is never redrawn. In R3
   (logged policy) it is the action centred by its logged propensity. Var(v_t) is known from the design.
2. **Predictable adjustment.** Rows are taken in time order. For each row the covariates are the lagged KPI vector,
   the observed context, the setpoints of the dither actions, the actions of the previous two rows and the
   concurrent values of the other designed actions (ruling R-8: their designs are independent of a's given the
   observed context). None of these is affected by a's draw at t. For each KPI, a ridge regression of the outcome on these covariates is fitted on
   the past rows only (an expanding window, refitted at geometric block boundaries, with lambda chosen by GCV on
   those past rows). The weight w_t is the out-of-sample residual. The first max(30, d + 1) rows (d covariates) get
   weight 0. The covariates come from the shared builder `cdd_oran/xmethod/covariates.py` (`design_covariates`),
   which every equal-information method uses (ruling R-17); the refactor onto it is bit-identical
   (`scratchpad/xmethod/results/cov_equiv.jsonl`). Option `covariates="r3"` (ruling R-19 ablation) drops the
   setpoints and the lagged actions and keeps the R-3 set (lagged KPIs, context, concurrent actions).
3. **No clip (ruling R-14).** The weight is the residual itself, as in PMRT's `plain` arm. PMRT's predictable Huber
   clip is available as the option `huber_c` (clip at c times the MAD of past residuals) but is off; see "Why no
   clip" below.
4. **Test.** S = sum_t v_t w_t, sd = sqrt(sum_t Var(v_t) w_t^2), z = S / sd. The p-value comes from at most
   B = 9999 redraws of a's random part (two-sided, |z|) with Besag-Clifford sequential stopping (ruling R-9: stop
   at 20 exceedances, p = 20 / k). All KPIs of one action share the same draws.
5. **Declarations.** Benjamini-Yekutieli at q = .05 over the action -> KPI candidates including the placebo
   (ruling R-6; `fdr_layer.declare`, procedure `by`). The declared sign is sign(S).

Validity follows the PMRT argument. The weights are predictable and v_t has mean zero given them, so S is a
martingale whose predictable variance equals the variance of the redraw distribution. The CRT p-value is therefore
asymptotically valid (martingale central limit theorem), but not exact.

## PMRT components: kept, reduced, dropped

| PMRT component (E6-P) | PMRT core (E1-E5) | Why |
|---|---|---|
| Assignment score, propensity-centred (`design_regressor`) | **Kept**, generalised to i.i.d., dither and logged designs | same design-based principle |
| Predictable adjustment: ridge on pre-window KPI bins, ctx and past mode history | **Reduced**: lagged KPI vector, context, setpoints and the previous two rows' actions. The ridge is fitted on past rows of the same corpus, not on independent training episodes and then frozen. **Extended** by the concurrent values of the other actions (R-8); E6 excludes other families' later units because their existence can depend on earlier units, which cannot happen with independent row designs | one-step rows; no training corpus |
| Running (past-only) stratum centres, episode x sgn | **Reduced** to the intercept of the past-only ridge | no episodes or request directions in a row corpus |
| Matched filter K = Sigma^-1 mu over receiving-cell slots x time bins, sign-constrained `loadsp` profile | **Dropped**. With one target value per row the kernel is the scalar 1, which is PMRT's `plain` arm | no cells, no time bins |
| Log-variance weights (`_h`) | **Dropped** | not part of the PMRT primary arm |
| Predictable Huber clip (`_c`) | **Dropped** (option `huber_c`, scale = MAD of past residuals) | finite-sample over-rejection with the clip (R-14, "Why no clip") |
| CV arm selection, GBDT g-hat, arm combinations | **Dropped** | need training episodes |
| Receiver table rho, unit support rule | **Dropped** (E6 unit-table specific). A constant or non-finite target, or zero design variance, is "undetermined" | - |
| CRT redraw from the known design, B = 9999, asymptotic validity | **Kept** | - |
| Weighted BY, one-sided where the prior \|z\| >= 3 (`wby1s`) | **Reduced** to plain BY | no independent prior data |

## Why no clip (ruling R-14)

With the clip (c = 2.5), the null rejection rate on the truth-null edges of E2 R2 was .070 [.055, .087] at .05 and
.020 [.014, .027] at .01 (n 1000, 40 DEV seeds, seed-cluster bootstrap 95% CI; Kaggle `xm-pmrt-q4-1`,
`scratchpad/xmethod/results/q4/`). Without it the rate was .052 [.037, .067] and .009. The CRT holds the weights W
fixed under the redraws, clip threshold included. That is exact only if W is invariant to the focal action's
random draws. Here W is predictable but depends on the focal action's past dithers through the lagged-action
covariates, the lagged KPIs the action drives, and the ridge coefficients and clip scale refitted on them. Validity
then rests on the martingale central limit theorem, and at n 1000 the clipped statistic's null z was over-dispersed
(sd 1.07). Each ingredient alone was nominal: no clip .052; clip with the lagged actions dropped .057; clip with
weights invariant to the focal draws (an exact CRT in E2) .061 / .008. The excess vanished at n 4000 (.045 /
.005). It did not depend on the concurrent covariates (.072 without them), on the dither width (.059 / .070 / .061
at .05 / .10 / .20) or on E2's gated KPI. It was not a mislabelled true edge: no null edge had a consistent sign
across seeds, and the excess shrank with n. The same ingredients exist in the E6 arm `loadsp_c`; that is audited
separately (ruling R-15, Experiment D).

## Equivalence with pmrt.py

Script: `scratchpad/xmethod/pmrt_equiv.py`. Results: `scratchpad/xmethod/results/pmrt_equiv_B9999.json`. Data: the
small E6-P caches placebo1 and plxc2 (split 9) and dev1 (split 0). Parameters: the artifact `E6P_PMRT_V4.json`.

- **Engine.** The core's CRT was given pmrt's `plain_c` weights (kernel disabled, i.e. 1 on bins [0, 90)), the
  core's own logged-categorical assignment and pmrt's RNG stream. Its p2, p_plus and p_minus are bit-identical to
  `pmrt_bench.integrated_family` on all 180 tested hypotheses (B = 9999). z agrees to a relative 3e-15. The same
  holds for pmrt's unclipped `plain` arm (the arm the core now corresponds to): bit-identical p, z identical.
- **Own adjustment** (measured with the clip on, against `plain_c`). With the core's own past-only adjustment on
  the same unit tables, z correlates 0.91 with
  `plain_c`. Agreement at p <= .05 is 0.92 (17 vs 24 hypotheses), and the sign agrees on 0.96 of pmrt's p <= .05
  hypotheses. The two are not expected to be identical. pmrt's adjustment uses post-window slots and bins, is
  fitted on 480 independent ev2 episodes and is centred within episode x sgn strata. The core fits on the past
  units of the same 20 episodes. Families with fewer units than the burn-in (sleep: 51-96 units vs 63) get weight 0
  and z = 0.
