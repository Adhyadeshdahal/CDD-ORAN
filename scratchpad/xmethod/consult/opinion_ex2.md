# Opinion ex2: equal information across discovery methods

Reviewer: ex2 (experimental design / benchmarking methodology). Read: CONTEXT_equal_info.md, CONTRACT.md (incl. sec 8),
PROTOCOL_NOTES.md, audit/pmrt_core.md, status/xm-{classic,harness,pmrt}.md, R2 generator docstring
(`cdd_oran/xmethod/worlds/generate.py` on feat/v2: 20 setpoints per column held n/20 rows, i.i.d. U(+-.1 range) dither;
the placebo has its OWN independent setpoint blocks). I ran nothing. Statements marked "expect" are predictions to be
measured on DEV, not results.

## 1. Which option

**A, but framed as a 2x2 ablation, not as "competitors with a handicap removed".** The paper's claim is causal in
form: "row-shuffling tests fail on RIC-like data *because* they ignore the dependence; a design-based test does
not". Under native conditioning, the comparison confounds two factors: (i) information (PMRT sees the setpoints
and lagged actions; the CI tests do not) and (ii) inference basis (redraw the known random part vs permute rows /
use an i.i.d. asymptotic null). The claim is about (ii). Only a design that varies (ii) with (i) held equal
identifies it. Native-only (C), or native-primary (B), leaves the headline effect attributable to an omitted
covariate, which is the first thing a referee will check. Native results stay in as secondary ("default use").

The protocol is still DRAFT and no EVAL seed has been touched, so adding and ranking the arm now is a legitimate
pre-registration amendment. Record it as triggered by F7 finding 1, dated, before EVAL. Do not choose the primary
after seeing EVAL.

## 2. Is it fair "same information"? Does it fix validity?

- Same input, same method. Adding S (setpoints of all actions, incl. the focal one) and two lagged action rows to
  Z keeps each CI test what it is: a test of X indep. Y | Z'. Under the causal null (X has no effect on Y),
  X indep. Y | Z, S_X still holds, so the hypothesis stays a valid null implication. What competitors still cannot
  use is the *dither law*, i.e. randomization inference. That gap is exactly the paper's contribution, and A
  isolates it. Setpoints are configured values in a real RIC, so a competent analyst would condition on them;
  withholding them is the strawman.
- The focal setpoint is the key term. The placebo excess under R2 is nonsense correlation between a 20-level step
  series and slow KPI components: about 20 effective observations, not n. Conditioning on *other* actions'
  setpoints will not remove it. Conditioning on the focal column's own setpoint/blocks will: the residual of X is
  then about the i.i.d. dither, independent of everything under the null. For a linear t-test this makes the
  cross terms E[x_s x_t e_s e_t] vanish even with autocorrelated KPI errors, so **expect pcorr (and Granger with
  setpoints as exogenous regressors) to become near-nominal.** Expect permutation and kNN/kernel tests
  (MSCR, pdCor, CMI-kNN, RCoT) to improve but possibly stay off-nominal at small n: they estimate the
  residualization on a larger Z (3p + lagged KPIs; 20 block levels), and imperfect residualization leaves
  within-block dependence that row permutation treats as exchangeable. This must be measured, not argued.
- If pcorr+S turns out valid, that is a finding, not a threat. The honest story then is: "the failure is from
  ignoring the design. Once the design is known, a covariate-adjusted test is approximately valid in this simple
  exogenous-block R2. PMRT is exactly valid in finite samples, needs no model for the adjustment, and extends to
  logged (R3) designs, where covariate adjustment alone has no validity guarantee."

## 3. Risks

- A against PMRT: in R2, PMRT's power is low (recall .17 at n 1000) because only the dither is random. A valid
  pcorr+S uses the same dither variation and may match or beat it. That is a true result, and the paper must be
  able to report it. A is also *easy* for competitors: harness setpoints are exogenous, exactly known and
  block-constant. In a closed-loop RIC the non-random part responds to past KPIs, and covariate adjustment then
  needs a correct model. State this as a limitation. Optionally pre-register one feedback-setpoint R2 variant.
- Symmetry gap: PMRT got a DEV null calibration (R-14 no clip). If A competitors get covariates but no calibration
  check, that is fine. Do not tune their Z on DEV truth either: fix Z' mechanically (below).
- B: the secondary arm will be read as the real result. If it shows competitors are valid with S, "they break"
  in the abstract is contradicted by the paper's own table. C: the audit already named this as the attack
  surface, and a referee who reads the code (or Design.fixed_part in the API) will find it.

## 4. Recommendation (<400 words)

**Arms (pre-registered before EVAL; amendment dated, citing F7 finding 1):**
1. **Primary, equal-information:** every method that takes a conditioning set or covariates gets
   Z' = R-3 set + setpoints of all designed actions (incl. focal; `fixed_part`) + the same two lagged action rows
   PMRT uses (+ context in R3). That is exactly PMRT's covariate vector, no more and no less, fixed
   mechanically with no DEV tuning. Granger: S and the lagged actions as exogenous regressors. PC/NOTEARS:
   setpoints as exogenous nodes, using background knowledge (no incoming edges), not as candidates. SHAP-DAG /
   two-tower: extra input features. `corr` stays unconditioned by definition (the naive reference).
2. **Secondary, native:** R-3 as now ("default use").
3. **PMRT ablation, symmetric:** PMRT core with R-3 covariates only (no setpoints/lags). It should remain valid,
   because the covariates only buy power under a CRT. This is the cleanest evidence that its validity comes
   from the design and not from the information. Together the four cells form a 2x2: information (R-3 vs R-3+S) x
   inference (design redraw vs row-based).
4. Before freezing, on DEV only: truth-free placebo and truth-null rates for every A-arm test in R2 (n 500,
   1000, 4000). Report them; do not adjust.

**Paper text:** state the claim as "validity requires using the design. Covariate adjustment with the design
covariates recovers [measured] validity for linear tests in exogenous block designs, [measured] for
permutation/kernel tests. Only design-based redraw is exactly valid in finite samples and under logged
designs." Report the native arm as what default usage yields. Disclose that the equal-information arm was added
after the F7 audit and before EVAL, and that it is the primary arm.
