# Question for independent methodological review: equal information across discovery methods

Study: a pre-registered comparison of causal-discovery methods on synthetic worlds E1-E5 (repo CDD-ORAN, branch
feat/v2; rules scratchpad/xmethod/CONTRACT.md incl. rulings sec 8; notes scratchpad/xmethod/PROTOCOL_NOTES.md).
Target: edges action -> KPI (one-step), plus a placebo action with no effect. Methods: PMRT core (design-based
conditional randomization test: it redraws only the KNOWN random part of each action from its known distribution),
independence tests (MSCR, partial correlation, pdCor, RCoT, CMI-kNN), PC, NOTEARS, SHAP-DAG, two-tower, correlation,
pairwise Granger. Regimes: R1 actions i.i.d. random; R2 action = bounded setpoint (slow, NOT random, constant within
20 blocks, observed) + small random dither (known distribution); R3/R4 confounding (E4).

Finding (independent audit, scratchpad/xmethod/audit/pmrt_core.md finding 1): pmrt_core uses as adjustment covariates
the R2 setpoints of all actions and two lagged rows of actions (plus pre-state). The CI tests' conditioning sets
(ruling R-3) are: all other action columns + lagged KPIs (+ context in R3); they do NOT include the setpoints / block
structure or lagged actions. The R2 setpoint blocks are exactly what makes corr / Granger p-values invalid
(placebo declared 2-4x). Setpoints are observable in a real RIC (they are configured values).

The decision: (A) give every method that accepts a conditioning set the same information (setpoints / block
indicators + lagged actions) as the PRIMARY comparison, native as secondary; (B) native primary, equal-information
arm secondary; (C) disclose only. Or something else.

Please answer as an expert in causal inference / experimental design / methods benchmarking:
1. Which option is methodologically right for a journal paper whose claim is "row-shuffling independence tests
   break on RIC-like dependent data; a design-based test does not", and why?
2. Is giving competitors the setpoints a fair "same information" comparison, or does it change what those methods
   are (e.g. methods that cannot use a randomization design, only covariates)? Does conditioning on setpoints even
   fix their validity problem (serial dependence within blocks, row-permutation nulls)?
3. Any risk that option A or B makes the study unfair to PMRT, or that C leaves an attack surface for reviewers?
4. Concrete recommendation (which arms, which is primary, what to write in the paper), in under 400 words.
Write your answer to scratchpad/xmethod/consult/opinion_<your-name>.md. Read only; do not edit code or other files.
Work independently: do not read the other opinion files.
