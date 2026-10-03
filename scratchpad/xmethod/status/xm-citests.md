READY-TO-MERGE (branch xm/xm-citests; F4-heavy + GPU results PENDING on Kaggle, listed below)
# xm-citests status  (updated 2026-10-02 ~20:10Z)
Decision log: scratchpad/xmethod/results/citests/NOTES_log.md. Fidelity table: docs/xmethod/FIDELITY_CITESTS.md.

## HAND-BACK
Deliverables
- Adapters cdd_oran/xmethod/methods/{mscr,pcorr,pdcor,rcot2,cmi_knn}.py, shared core _citests_common.py,
  F4 data _citests_synth.py; config arm "eq" (default, R-17/R-25/R-28 via _classic_common.cond_set) | "native"
  (R-18); R-22/R-23 not_testable in score.py format; cmi_knn backend "cpu" (tigramite) | "torch" (GPU).
- Tests tests/test_xmethod_citests.py: 28 passed; all xmethod tests pass after merging feat/v2 b7666d3; ruff clean.
- Scripts scratchpad/xmethod/citests/ (F2/F3/F4 gates, integrate.py --kappa required, gpu_cmi.py, Kaggle jobs).
- Shared runner change scratchpad/e6_dev/colab_run.py: env COLAB_ACCEL (tpu:v5e1 default | gpu:T4 | cpu);
  a failed launch aborts only on assignments no known session explains (concurrent known sessions allowed).
Gates (details + numbers in FIDELITY_CITESTS.md; all arm native)
- F1/F6: sources, versions, deviations per adapter in the table.
- F2: pcorr == OLS (1e-13); pdcor == dcor (3e-16); rcot2: momentchi2 py == R exact, null levels == RCIT 8/8,
  power 1-8 pp below RCIT in 3/4 alt (FAIL of the pre-stated tolerance; frozen statistic numerics); mscr / cmi_knn
  native == frozen / tigramite (tests); cmi_knn torch == tigramite bit for bit locally (Kaggle 6-world F2 PENDING).
- F3: mscr 48/48, rcot2 84/84, pdcor 84/84 EXACT on stored E2 replicate-00; analytic pcorr / cmi_knn (knn 10) PASS.
- F4: pcorr PASS (.052 / .049); rcot2 FAIL null level (.084 / .065) = FINDING: RCIT itself liberal (.075-.11);
  mscr, pdcor, cmi_knn PENDING (Kaggle).
- F5: tau rule only (truth-free tune(), DEV only).
Results PENDING (pull when done; fill FIDELITY_CITESTS.md F4 / F2-GPU / cost)
- Kaggle xm-citests-f4h-k1: F4 heavy mscr (n 500 / 1000 x100), pdcor null x100, planted n 500 x100, n 1000 x50.
- Kaggle xm-citests-gpu-k1: F2-GPU (exact vs CPU tigramite, 6 worlds), R-9 cost per dataset at n 1000 / 4000 /
  8000 (E5 R1 eq kappa .25), cmi_knn F4 on the GPU backend. If F2-GPU passes: torch = cmi_knn production backend.
- Kaggle xm-citests-cost-1 (native-arm cost, pre-arms code): superseded by dev-runs; ignored (no cancel command).
Cost / scaling (R-13, 2 CPU-h budget; n 1000 native kappa 0 measured, single-threaded)
- pcorr ~0 s, rcot2 <= 3 s, mscr <= 23 s: all n feasible. pdcor 3.5-1751 s at n 1000, O(n^2) memory/time per
  permutation: n 4000 likely over budget for the larger worlds, n >= 8000 infeasible (memory ~5 n^2 doubles).
  cmi_knn CPU: 3437 s at native B 500 (one E1 R2 dataset) -> over budget at R-9 for n >= 1000; GPU numbers PENDING.
  Final per-cell cost / validity / power: dev-runs.
Added deps: group citests tigramite 5.2.10.1, dcor 0.7, momentchi2 0.1.8 (MIT; numba 0.67 transitively).
Deviations: momentchi2 R side ran in a local Docker r-base container (1000 CDF evals); f3root data copies were
  in history once (1b761fd, removed d01b2d3); my Colab launcher double-launched f4h-a (NOTES_log 18:45Z).
api.py proposals: none.

## QUESTIONS (open)
- classic's cond_set keeps P_placebo_conf columns as conditioners of OTHER sources; citests removes them (R-10 as
  applied: the diagnostic never changes the primary analysis). Otherwise identical (tested). Align either way?
- Validity observation for dev-runs: cmi_knn E4 R1 eq placebo 0 exceedances / 2464 surrogates (FIDELITY F4 notes).
- Answered: Q10-Q15 (R-10 amended, R-11, R-24/R-27, helper names, R-23, R-28).
