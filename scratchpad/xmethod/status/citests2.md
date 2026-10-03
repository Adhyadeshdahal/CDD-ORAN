IN PROGRESS (branch xm/citests2; merged feat/v2 3ef0ac1 incl. xm/hac2 + audit-citests; waiting on Kaggle F4-heavy mscr / pdcor only)
# citests2 status  (updated 2026-10-02 ~22:30Z)
Worktree D:/academia/major-project/CDD-ORAN-wt/xm-citests. Fidelity: docs/xmethod/FIDELITY_CITESTS.md.

## Done
- R-33 arm "eq_min" for all five CI tests: per source = classic `cond_set(data, family, src, "eq_min")` (shared
  covariates.eq_min_covariates: R-3 set + sp:<focal>; lagged-KPI sources native; R-37 applied there), mapped onto
  the S columns like eq; constant sp left out (notes eq_dropped); no rows dropped. Tests: citests eq AND eq_min
  sets == classic cond_set exactly (E2 R2, E4 R3/R4, E5 R1), layout vs native + sp:<focal>, all five adapters run
  with eq_min, pcorr eq_min == OLS t-test.
- R-29: tune() tau = score.placebo_tau(results, alpha=.05) (conformal, from xm/hac2; test checks it is used and
  equals the local rule tau_from_scores: ceil((M+1) .95)-th smallest, max beyond M, +inf if M = 0). Only the
  secondary notes["declared_tau"]; primary declaration = BY on p-values.
- audit-citests 'Fixes (exact)' 1-6 + R-40:
  1 FIDELITY mscr row: frozen scope (randomized designs only, E1 FDR .81), single-conditioner max; R-40 label,
    mscr_eq excluded from C2b, mscr expected INVALID in R2 (R-21). 2 pdcor docstring (signed pdCor under proj_perm,
    |pdCor| x_perm), F6 Z standardization, R-40 dependence test; f2_local docstring (proj_perm == dcor's null).
  3 tau_from_scores / tune() -> score.placebo_tau (one implementation; it also fixes the (M+1)(1-alpha) rounding).
  4 pdcor / rcot2 guards: p NaN, listed not_testable "guard: ..." (outside BY; native rule unchanged).
  5 a STOP lists every candidate given to _test as not_testable "STOP: ...".
  6 mscr bank stream [seed, j, n_perm, g] for column group g >= 1 (group 0 unchanged = frozen discover_mscr
    stream, native bit-equality test still passes; only the R3 / R4 eq P_placebo_conf group changes).
  Tests: shared tau, guard / STOP stubs, mscr stream keys (37 passed).
- Kaggle xm-citests-gpu-k1 DONE (T4, torch 2.11): F2-GPU PASS exact (counts identical n 500-4000, 18/18 tests
  bit-identical, 6 worlds) -> cmi_knn production backend = "torch" (device cuda if available else torch-cpu;
  native_config = tigramite "cpu"); cmi_knn F4 PASS (null .044 n 500, .041 n 1000); GPU cost E5 R1 eq: n 1000
  172 s / dataset, n 4000 ~2.4k-6.1k s, n 8000 ~9k s (bounds in FIDELITY_CITESTS.md). Results/citests/gpu/.
- R-35: Kaggle scripts pin tigramite / dcor / momentchi2 / gymnasium / pyyaml / numba / llvmlite (and torch
  2.10.0+cu128 for the GPU job) to uv.lock. The two jobs already running (f4h-k1, gpu-k1) predate the numba /
  llvmlite / torch pins (Kaggle's preinstalled versions; recorded in their logs) -> disclosed.
- R-37 already satisfied (tests). R-30/31/32/34/36: campaign / dev-runs / hac / wording; nothing in my adapters.
- tests/test_xmethod_citests.py: 38 passed (single process, after the merge); ruff clean (package + tests).

## Running / next
- Kaggle xm-citests-f4h-k1 (F4 heavy mscr / pdcor, arm native): pull when done -> FIDELITY_CITESTS.md F4 rows ->
  line 1 READY-TO-MERGE. xm-citests-cost-1: superseded, ignored.

## QUESTIONS
- none.
