# R-42 selection (applied mechanically, 2026-10-03)

Rule (CONTRACT R-42, fixed before any result): among the candidates that pass exactness (synthetic F4 + DEV
truth-null and placebo rates not INVALID in any cell, R-30) and the R-13 budget at n 4000, pick the highest mean
per-edge recall at raw p <= .05 over DEV E2 / E5 R1 / R2 n 1000 + the synthetic nonlinear scenarios (SYN-NL R1 / R2);
ties go to the cheaper one. Candidates: N1 `poly`, N2 `rff`, N3 `gbm` (CANDIDATES.md). DEV seeds 3_000_100-159 only.

| candidate | F4 INVALID cells (12 cells x 2 rates) | DEV INVALID cells (54 cells x 2 rates) | cost n 4000, max CPU-s (budget 7200) | recall (58 edges) | passes |
|---|---|---|---|---|---|
| gbm  | 0 | 0 | 57 | **.749** | yes |
| rff  | 0 | 0 | 87 | .666 | yes |
| poly | 0 | 0 | 26 | .638 | yes |
| (linear pmrt_eq, reference) | | | | .420 | |

**Winner: `gbm`** (learned predictable matched filter, past-only gbm adjustment). Implemented as
`pmrt_core.PmrtCoreConfig(statistic="gbm")`, version `pmrt-core-v2`; arm pmrt_nl_eq = config
`{"covariates": "eq", "statistic": "gbm"}`. The linear statistic stays the default (`pmrt-core-v1`, secondary arm).

Rates (R-30 class from seed-cluster bootstrap 95 % CIs, 2000 reps; agg_*.json):
- gbm DEV: truth-null .025-.066 (mean .042; VALID 23, INCONCLUSIVE 1, no null edge 30 (E4)); placebo .000-.150
  (mean .047; VALID 20, INCONCLUSIVE 34: 20-60 seeds x 1 placebo edge give wide CIs). F4: truth-null .026-.047, all
  VALID; placebo .033-.058.
- rff DEV: truth-null .025-.070 (mean .049), placebo .000-.100; F4 truth-null .037-.063, placebo .021-.083.
- poly DEV: truth-null .037-.070 (mean .049), placebo .000-.150; F4 truth-null .035-.058, placebo .025-.067.
- gbm recall per cell: E2 R1 .90, E2 R2 .78, E5 R1 .90, E5 R2 .72, SYN-NL R1 .65, SYN-NL R2 .34
  (linear: .38 / .44 / .47 / .54 / .45 / .30).
- Reported only (Q2, not candidates): rff_g .695, poly_g .653: the gbm adjustment adds .015 (poly) / .029 (rff); the
  learned gbm statistic adds .054 / .096 over those.

Runs: recall Kaggle pmrt-nl-recall-1; F4 Kaggle pmrt-nl-f4-k1; DEV validity gbm Kaggle pmrt-nl-valid-gbm-k1 (+ Lightning
pmrt-nl-rev-gbm, reverse order: p bit-identical to Kaggle on the 1081 overlapping units), rff Lightning pmrt-nl-vrff +
pmrt-nl-rev-rff, poly + cost Lightning pmrt-nl-cpoly. 0 errors in every plan.
