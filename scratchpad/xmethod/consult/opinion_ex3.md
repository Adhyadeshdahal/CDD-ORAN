# Opinion ex3: equal information across discovery methods

Reviewer stance: skeptical referee, causal inference / experimental design. Read: CONTEXT_equal_info.md, CONTRACT.md
(sec 8), PROTOCOL_NOTES.md, audit/pmrt_core.md, docs/xmethod/PMRT_CORE.md, `worlds/generate.py` (R2 dither),
xm-citests status (branch xm/xm-citests). No code run, so the validity claims below are reasoned predictions. Each one
names the DEV check that would settle it.

## Facts that drive the answer

- R2 is exact and simple. action = fixed_part + random_part. The setpoint is constant over n/20 rows. **All actions
  and the placebo share the same block clock** (`_draw_dither` `block`), and the dither is i.i.d. U(+-.1 range).
- Hence 20 block dummies span every setpoint. Conditioning on the focal action's own setpoint (linear, exact)
  leaves **exactly its dither**.
- The R2 placebo is itself setpoint + dither (lag-1 autocorr .74-.86; xm-citests Q10). R-10 says the placebo is
  "always i.i.d.", so the contract and the harness disagree. The observed "placebo declared 2-4x" is mostly
  spurious regression: a 20-level slow placebo against slow KPI residuals, about 20 effective observations
  (Granger-Newbold). This is the failure that conditioning on setpoints removes.

## 1. Which option

**A, but with the headline claim re-scoped.** The contrast "native competitors versus design-aware PMRT" confounds
two things: the *information* (which part of the action is random) and the *inference engine* (a CRT redraw versus
a row permutation or asymptotic test). The claim names the engine ("row-shuffling tests break; a design-based test
does not"), so only an equal-information comparison can test it. Under B the primary contrast is the confounded
one. Under C a referee writes one sentence ("the competitors were denied the block structure that the authors'
method uses; this is a strawman") and the paper has no data to answer it. `api.Design` already exposes
`fixed_part` to every adapter, and setpoints are configured values in a real RIC. So denying them is a protocol
choice, not a property of the methods.

## 2. Is it fair, does it change the methods, does it fix validity?

- **Fair for conditioning-set methods (pcorr, pdCor, RCoT, CMI-kNN, MSCR, PC's CI tests).** Adding covariates to Z is
  their native interface and does not change what they are. Define "same information" as **exactly PMRT's covariate
  set**: R-3 Z + all setpoints + two lagged action rows. Do not invent a richer or a poorer set.
- **It probably does fix their validity in R2.** Given the focal setpoint, the residual X is the i.i.d. dither,
  independent of everything under the null. sum_t x_t e_t then has mean 0 and variance sigma_x^2 sum e_t^2, *even
  when the KPI residual e_t is serially dependent*. Serial dependence in Y does not break a test whose regressor is
  i.i.d. and exogenous. Better still, a **row permutation of an i.i.d. dither is itself a valid randomization test**
  (exchangeable, same law as a redraw). So "row-shuffling" is not the defect. Shuffling the *wrong component* is.
  Expect pcorr + setpoints to be nominal, which is close to an OLS version of PMRT core.
  Caveats: (i) kNN / kernel tests in high-dim Z (E2: tens of columns) may still miscalibrate, because local
  permutation neighbours need not share a block. That is a genuine property of those methods; report it. (ii)
  In-sample residualisation on lagged KPIs (which depend on past focal dithers) is only asymptotically harmless.
  Check: DEV placebo + truth-null rates per method in the equal-information arm (a validity check, not tuning).
- **It does change the methods without a conditioning set** (corr, NOTEARS, SHAP-DAG, two-tower). corr given setpoints
  *is* pcorr. Granger with exogenous setpoints is VARX, which is legitimate. For PC and NOTEARS the setpoints would be
  exogenous tier-0 nodes. Keep these native and give at most a secondary "design-augmented" variant for Granger and PC.
- **What equal information cannot give the CI tests:** R3. Logged per-row propensities are not exchangeable, so a
  permutation of residuals is not exact. The CRT redraw from the logged law is (and aud2 showed .051 under live
  confounding). It is also robust to misspecifying E[Y|Z], because only the assignment law must be right. These are
  PMRT's real, defensible advantages.

## 3. Risks

- **A, to PMRT:** the R2 validity gap may vanish. That is not unfair. It is the finding, and it is better learned on
  DEV than from a referee. Real unfairness would come from per-method encoding choices made after seeing results, or
  from tuning on truth. Fix the encoding (PMRT's set, one encoding per method) before EVAL. Use the same
  placebo-only tau rule. Remember that PMRT's no-clip choice already used DEV truth (aud2 finding 3), so do not
  hand competitors a weaker rule.
- **B:** looks like the primary contrast was chosen because it flatters PMRT.
- **C:** leaves the strawman attack open, plus a second one. The headline "row-shuffling breaks" would be stated
  causally about an engine when it is really about information.
- **Q10 must be closed first.** If the placebo becomes i.i.d. (R-10 literally), the truth-free control no longer
  sees the R2 failure at all. Section 2 says "drawn with the same design as the real actions". Keep the dithered
  placebo (with its own setpoints) as the tuning and negative control, and amend R-10.

## 4. Recommendation (<400 words)

1. **Primary (all regimes):** the equal-information arm for every conditioning-set method. Z_eq = R-3 Z + all
   setpoints (block-constant values) + actions at t-1, t-2, the same set PMRT core uses. Use one pre-specified
   encoding per method and the same BY / tau rules. Freeze it before any EVAL seed. The contract is still DRAFT, so
   this is a legitimate pre-registration change; log it with date and reason.
2. **Pre-specified secondary:** native R-3 arm ("as typically applied, design-blind"). Also corr and pairwise
   Granger native, and Granger / PC with setpoints as exogenous nodes. NOTEARS, SHAP-DAG and two-tower stay native,
   with a sentence that they have no conditioning interface.
3. **Close Q10** (dithered placebo) and re-state R-10 before freezing.
4. **DEV check now:** placebo and truth-null rates of the equal-information arm, E2 / E5 R2 at n 500 and 1000.
5. **Re-scope the paper's claim.** For example: *"Under RIC-style setpoint + dither operation, design-blind tests are
   invalid (secondary arm). Validity requires the design decomposition. Given it, linear CI tests on the dithered
   component recover validity in R2 because the dither is i.i.d. A design-based CRT needs only the assignment law.
   It stays valid where residual permutation is not (logged policies, R3) and without a correct outcome model."*
   Report PMRT's power relative to the equal-information arm honestly. Drop "row-shuffling tests break" as an
   unqualified headline.
6. **Paper text:** say that all methods received the same observable information (configured setpoints, lagged
   actions). Say that native results are shown to quantify the cost of ignoring the design. Disclose the encoding
   and the date it was fixed.
