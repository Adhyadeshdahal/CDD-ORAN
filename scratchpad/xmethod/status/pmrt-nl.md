READY-TO-MERGE
# pmrt-nl status (R-42 nonlinear PMRT statistic; branch xm/pmrt-nl from feat/v2 93a4c9e; worktree xm-citests)

## Result: R-42 winner = `gbm` (results/pmrt_nl/SELECTION.md)
- Step 1 diagnosis (DIAG.md): PMRT lost power, not validity: centred bumps (Cov(v, Y) = 0; R2 slopes cancel across
  setpoints), pure interactions with concurrent actions / gates, nonlinear nuisance left in the linear residual.
- Step 2 candidates (CANDIDATES.md, exactness argument per candidate): N1 poly, N2 rff, N3 gbm (learned predictable
  matched filter + past-only gbm adjustment); module cdd_oran/xmethod/methods/pmrt_nl.py (pmrt-nl-v1).
- Step 3 gates (0 errors in every plan; agg_*.json):
  | cand | F4 INVALID | DEV INVALID | cost n 4000 max CPU-s | recall (58 edges) |
  |---|---|---|---|---|
  | gbm | 0 / 24 | 0 / 108 | 57 | **.749** |
  | rff | 0 / 24 | 0 / 108 | 87 | .666 |
  | poly | 0 / 24 | 0 / 108 | 26 | .638 |
  linear pmrt_eq .420; reported only: rff_g .695, poly_g .653. gbm DEV truth-null .025-.066 (mean .042), F4 .026-.047.
- Step 4: rule applied mechanically -> gbm. `PmrtCoreConfig(statistic="gbm")` -> pmrt_core version pmrt-core-v2
  (delegates to pmrt_nl); linear stays the default (pmrt-core-v1, code path unchanged). Arm pmrt_nl_eq =
  `{"covariates": "eq", "statistic": "gbm"}`.

## Questions
- Q1 Predictable (past-only) gbm instead of cross-fit. ANSWERED (orchestrator 2026-10-03): approved + argument.
- Q2 Nonlinear gbm adjustment in scope. ANSWERED (orchestrator 2026-10-03): yes; report poly_g / rff_g too.
- Q3 2nd Kaggle session for F4. ANSWERED (orchestrator 2026-10-03): yes (5 = cap).
- Q4 Lightning 401. ANSWERED (orchestrator 2026-10-03): credentials refreshed; launched.

## Compute (R-44 / R-46)
- Kaggle: pmrt-nl-recall-1 (3.7 h), pmrt-nl-f4-k1 (3.4 h), pmrt-nl-valid-gbm-k1 (5.6 h): all COMPLETE, pulled.
- Colab CPU pmrt-nl-cv-c1: LOST after 11 min (runtime reclaimed); its work moved to Lightning.
- Lightning (cpu-4; ALL STOPPED by 12:46 UTC): pmrt-nl-cpoly (cost + poly valid, 08:05-09:08), pmrt-nl-vrff (rff
  valid, 07:40-11:04), pmrt-nl-rev-rff (rff valid reversed, 10:19-11:04), pmrt-nl-rev-gbm (gbm valid reversed,
  10:19-12:46; p bit-identical to Kaggle on 1081 overlapping units). vrff / rev-rff were stopped externally at
  10:51 (cause unknown); restarted only to pull, union already complete.
- Lightning credits: account balance 3.7685 (07:40) -> 0.9745 (12:46) = 2.79 drop (under the 3.0 ask point), but
  the API's total_spent rose only 0.84 and my 4 studios ran ~7.4 studio-h; other workers may share the balance.
  Orchestrator: please reconcile before the next Lightning launch on this account.

## HAND-BACK
- Code: cdd_oran/xmethod/methods/pmrt_nl.py (new); pmrt_core.py (`statistic` / `adjust` options,
  PMRT_CORE_V2_VERSION, STATISTICS; linear path untouched; pmrt-diag made no pmrt_core change, R-45).
- Tests: tests/test_xmethod_pmrt_nl.py (13, incl. the pmrt_core option). Full xmethod suite: 343 passed, 1 skipped
  (tests/test_xmethod_citests.py not collected locally: tigramite not installed; pre-existing).
- Docs: docs/xmethod/PMRT_CORE.md section "Nonlinear statistic (ruling R-42)".
- Results: scratchpad/xmethod/results/pmrt_nl/ DIAG.md, CANDIDATES.md, SELECTION.md, agg_recall / f4 / valid_{gbm,
  rff, poly} / cost .json; scripts scratchpad/xmethod/pmrt_nl_{diag,shapes,runs}.py (SYN-NL pre-specified in runs).
- Seeds: DEV 3_000_100-159 only; 3_000_160-189 and EVAL never touched.
- Dependency: xgboost 3.4.1 (already in pyproject's baselines group / uv.lock); cloud kernels pip-pinned (R-16).
- Deviations: none from R-42. Runs script gained `--reverse` (order only; agg dedups keys). The test file and this
  status file had been committed with CRLF; normalized to LF (repo convention), so their diff is whole-file.
- Follow-up (orchestrator): E4 placebo cells up to .15 (gbm R1 n 1000, poly R4 n 500; 20 seeds x 1 placebo edge,
  CI to .35, INCONCLUSIVE); pmrt_nl_eq on the DEV T1 cells for the R-43 power pairs is not run here.
- Commits local only; never pushed or merged.
