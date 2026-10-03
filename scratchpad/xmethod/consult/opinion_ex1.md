# Opinion ex1: equal information across discovery methods

Reviewer stance: causal-inference statistician (randomization inference, CRTs / model-X). I read
CONTEXT_equal_info.md, CONTRACT.md (incl. sec 8), PROTOCOL_NOTES.md, audit/pmrt_core.md, audit/corr.md,
audit/granger.md and the xm-harness hand-back (J3: R2 = 20 blocks of n/20 rows shared by all actions, setpoint
U(mid +/- .25 range), dither U(+/- .1 range) i.i.d., column == setpoint + dither exactly). I did not read other
opinion files. I ran nothing, so the predictions below are reasoning, not results.

## 0. The key fact

In R2 the focal action minus its setpoint **is** the dither, exactly. The dither is i.i.d. and its law is known.
So if a CI test conditions on the focal setpoint (or on block indicators), then the focal variation left after
adjustment is i.i.d. and independent of everything. A test is valid when **one** side of the pair is exchangeable
and independent, however serially dependent the other side (the KPI) is. Two examples:

- the OLS / partial-correlation t-test with an i.i.d. regressor that is independent of the errors;
- within-block permutation of i.i.d. dithers.

So in the equal-information arm the competitors stop being "row-shuffling tests on dependent data". They become
**model-X / design-based tests in disguise**. The permutation (or the asymptotic null) then stands in for PMRT's
redraw from the known law. The setpoint is the design information.

This settles most of the decision. Arm A does not test the paper's claim. It is an ablation that shows **why** the
claim holds.

## 1. Which option fits the stated claim

The claim is "row-shuffling independence tests break on RIC-like dependent data; a design-based test does not".
That claim is about the methods **as practitioners specify them**: condition on the other variables and their lags,
then shuffle rows or use an i.i.d. asymptotic null. R-3 encodes exactly that. So **native is the correct primary
arm for the validity claim**.

Option C (disclose only) is not enough. A reviewer will say, correctly: "you withheld the one confounder that
explains the failure, and that confounder is observable in a RIC". Only an experiment answers that objection.

So the answer is **B, with the equal-information arm pre-registered, reported in the main text, and tied to its own
hypothesis**. It must not sit in a supplement. Split primacy by question (estimand):

- **Validity (null / placebo rejection rate):**
  - native arm: primary;
  - equal-information arm: pre-registered mechanism ablation.
- **Power / recall at valid level:** use the equal-information arm wherever it is valid. A recall comparison
  against an invalid method means nothing. A recall comparison across unequal information sets mixes up "method"
  with "covariates".

## 2. Is "same information" fair, and does it fix validity?

**It is fair as an information grant.** The setpoints are configured values, and `api.Design` already exposes them.
But it changes what the methods are, in two ways:

- **(a) Methods with a conditioning-set interface** (pcorr, MSCR, pdCor, RCoT, CMI-kNN, a conditional Granger
  variant) can use the grant natively. Expected effect: R2 validity is largely restored for linear and
  residual-based tests, for the reason in sec 0. For kNN and kernel tests, validity depends on how well they
  condition on a discrete 20-level covariate. Local permutation matched on the setpoint should be close to
  nominal. A global residual shuffle after a misfit regression may not be.
- **(b) Methods with no conditioning set** (corr, pairwise Granger, PC, NOTEARS, SHAP-DAG, two-tower) cannot
  receive it without being redefined. The options are to add the setpoints or block indicators as extra nodes, or
  to residualise on them first. Either one is our invention, not the authors' method. Keep these methods native
  only. For score-only methods the placebo tau already absorbs miscalibration, so for them the information gap
  affects power only. Disclose that in one sentence.

**Conditioning does not remove the serial dependence in the KPIs.** It makes the focal side exchangeable, and that
is enough. Two caveats remain:

- **(i) Feedback.** Past dithers move other KPIs, which feed the target. Exact exchangeability of the whole dither
  vector can then fail for the truth-null action -> KPI edges, though not for P_placebo. PMRT has the same issue and
  handles it with predictable weights (cf. R-14). Expect the CI tests to be approximately, not exactly, valid there.
  Measure it; do not assume it.
- **(ii)** The fix needs the analyst to know which part is random. In this benchmark the setpoint is exogenous. In
  a real RIC the slow part may be closed-loop: an operator or xApp reacting to KPIs. Conditioning on a
  KPI-reactive setpoint is still fine for exchangeability of the dither, **but only if** the dither is known to be
  the random part. That is design knowledge, which is exactly the paper's thesis.

**Lagged actions** are an efficiency grant in R1 and R2: they are independent of the current dither. In E3 they
mainly add power. Give them to the competitors in the equal-information arm. Withholding them biases the recall
comparison toward PMRT.

## 3. Fairness risks

**To PMRT:**

- Equal-information pcorr is probably at least as powerful as PMRT core's plain-kernel CRT. The two are
  asymptotically the same linear statistic, and the t-test has no Monte Carlo error.
- If A is primary, the headline turns into "PMRT ~ pcorr given the design". That is not unfair. It is the true
  result and should be reported. But it makes PMRT's contribution look smaller than it is. That contribution is:
  - validity by construction, with no conditioning set to specify correctly;
  - any statistic, including nonlinear and matched-filter statistics;
  - a uniform treatment of logged policies (R3).

**Against the paper:**

- Option C leaves the "strawman" attack open.
- B without the ablation in the main text leaves the same attack open.
- There is a further risk of post-hoc framing: running A only after seeing native results, then choosing the
  story. Avoid it by freezing both arms and the hypotheses below before EVAL. Running a DEV pilot of the arm's
  placebo rate before freezing is legitimate, because it uses no truth.

## 4. Recommendation (under 400 words)

**Arms (pre-registered, both on EVAL seeds, same n / B / BY / budget):**

1. **Native (primary for validity).** R-3 conditioning sets as now.
2. **Design-informed, "DI" (pre-registered ablation, main text).** Every method with a conditioning-set interface
   gets exactly PMRT core's adjustment covariates: the setpoints of all actions, two lagged action rows, concurrent
   other actions and pre-state. For linear tests, use block fixed effects as a sensitivity check. Add a
   conditional (multivariate) Granger in DI for E3. Methods without a conditioning set stay native only, with a
   disclosure line.

**Hypotheses, stated before EVAL:**

- **H1 (validity, native):** in R2, the null and placebo rejection rates of row-shuffling and i.i.d.-null tests
  exceed .05. PMRT core is nominal.
- **H2 (mechanism, DI):** conditioning on the non-random design component restores near-nominal rates for the CI
  tests. Report the rates with CIs, whatever they turn out to be.
- **H3 (power):** compare recall only among (method, arm) cells whose null rate is not significantly above .05.
  DI is the primary power comparison.

**Paper wording (core paragraph):** "Off-the-shelf CI tests fail in R2 because the slow setpoint makes the action
series non-exchangeable. When we give them the design information PMRT uses (setpoints, lagged actions), their
validity is [restored / partly restored] (Table X). The failure is therefore a failure to use the randomization
design, not a defect of the statistics. PMRT core makes this information explicit and gives validity by
construction. It needs no correctly specified conditioning set, it admits any test statistic, and it extends to
logged policies (R3), where a conditioning set fixes validity only if the policy model is right."

**Retitle the claim** from "row-shuffling tests break" to "tests that ignore the randomization design break; using
the design (PMRT) fixes this by construction". This version survives whatever H2 shows.

**Also:** remove `meta["roles"]` from the Dataset (audit finding 7) before DI arms read more of the metadata.
