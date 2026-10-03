# xm-citests fidelity table (CONTRACT sec 4, F1-F6)

Status: 2026-10-03, worker xm-citests (branches xm/xm-citests, xm/citests2; audit fixes 1-6 of
scratchpad/xmethod/audit/citests.md applied on xm/citests2). Code: `cdd_oran/xmethod/methods/{mscr,pcorr,pdcor,
rcot2,cmi_knn}.py`, shared core `_citests_common.py`, synthetic F4 data `_citests_synth.py`. Tests:
`tests/test_xmethod_citests.py` (38; `uv run pytest -q tests/test_xmethod_citests.py`, group `citests`: tigramite
5.2.10.1, dcor 0.7, momentchi2 0.1.8 MIT). Gate scripts and raw outputs: `scratchpad/xmethod/citests/`,
`scratchpad/xmethod/results/citests/` (decision log `NOTES_log.md`). All gate results are filled in (F4 heavy:
Kaggle `xm-citests-f4h-k1`, results/citests/f4/f4_heavy.jsonl + F4_heavy_summary.json; GPU: `xm-citests-gpu-k1`).

## Common to all five (orchestrator rulings)
- Test: `source _||_ target | Z` per candidate (R-3). Families (R-2/R-6): pooled BY q .05 per family, PRIMARY =
  action -> KPI incl. `P_placebo`; SECONDARY = lagged KPI -> KPI; `P_placebo_conf` (R3/R4) its own diagnostic family,
  never tuned on, and its columns are removed from every OTHER source's Z (R-10). Native rule in
  `notes["declared_native"]`; secondary score rule `notes["declared_tau"]` with tau = `score.placebo_tau(..., alpha
  .05)`, the R-29 conformal cutoff over the (arm, cell) tune seeds' P_placebo scores (one shared implementation).
- Arms (R-17/R-18/R-25/R-28), config `arm`: "eq" (PRIMARY, default) = each source's WHOLE Z from
  `_classic_common.cond_set` (action: `covariates.design_covariates(data, focal=a)`, concurrent "designed";
  lagged KPI: base set minus its own `lag_kpi` + every action at t), rows with `row_mask` False dropped, columns
  constant on kept rows left out (`notes["eq_dropped"]`); "native" (SECONDARY) = R-3 set (all other actions, lagged
  KPIs, context; tested equal to the helper's concurrent="all" set). "eq_min" (R-33, SECONDARY) = R-3 set +
  `sp:<focal>` only (`covariates.eq_min_covariates`; lagged-KPI sources native). eq and eq_min sets are taken from
  `cond_set` and equal classic's exactly, incl. R-37 (P_placebo_conf never conditions another source;
  `test_eq_arm_equals_classic_cond_set`). `native_config()` = arm native.
- Signs (R-4): pcorr native; others the sign of the partial correlation given the SAME Z (`sign_rule`).
- Resampling (R-9): Besag-Clifford, B max 9999, h 20 (`SequentialP`); native B only in `native_config()`.
- RNG: tag 7802 + dataset seed + method key (`method_seed`); per-edge streams as in each frozen source.
- R-22/R-23: a candidate whose target is an exact OLS fit of [1, source, Z] (resid var / var < 1e-8) is not tested:
  score NaN, p None, sign 0, not declared, `notes["not_testable"] = {"src->tgt": reason}` (score.py format) and
  `notes["not_testable_edges"]`. Noiseless E1 / E3 / E4 R3 (kappa 0, sanity only): all candidates (~1e-29);
  kappa .25 (R-27 primary): 0 in every cell, both arms (pcorr integration, n 1000). The same not-testable listing
  covers a numerical guard (pdcor denominator, rcot2 residual collapse: reason "guard: ...", p NaN, outside the BY
  family; the frozen native rule still sees p = 1) and a STOP of the wrapped numerics (every candidate given to the
  test, reason "STOP: ..."); audit L2 / L3.

## F1 source / version, null, F6 deviations

| adapter | F1 source / version | primary null | F6 deviations and why |
|---|---|---|---|
| mscr | own method `cdd_oran/discovery/mscr.py` (mscr-v2; frozen NC 6, NB 8, MIN 40) | frozen within-stratum random-partition bank | B 2999 -> sequential 9999 (R-9); every candidate source tested (v2: params only); conditioner pool = the arm's Z; bank seed from tag 7802 (per target [seed, j, n_perm]; a second column group g >= 1, the R3 / R4 P_placebo_conf group, [seed, j, n_perm, g], audit L4); pooled BY (v2: per-target BY). Frozen scope: VALIDATED ON RANDOMIZED DESIGNS ONLY (E1 FDR .81); S* = max over SINGLE conditioners, so Z_eq enters only as single conditioners. R-40: mscr_eq is reported as "single-conditioner max statistic; cannot condition on the joint design set" and is EXCLUDED from C2b; the audit expects mscr INVALID in R2 in both arms (R-21). |
| pcorr | `e1slice/discovery_v2.partial_correlation_scores` (frozen E1 v2) | partial-correlation t-test, df = n - 2 - \|Z\| (R-5) | E1 v2 had no null (largest-gap cut kept as native rule); Z per column group. |
| pdcor | frozen `e2slice/discovery` U-centred pdCor helpers; authors' test R energy 1.7-12 `pdcov.test` = python dcor 0.7 | authors' projection permutation (permute P_z(x)), one-sided n<Pxz,Pyz>, score signed pdCor (R-13) | frozen E2 null (permute x, re-project, two-sided \|pdCor\|, B 999) = cross-check `frozen_config()`; ties counted >= (energy >); B 9999 BC. Szekely & Rizzo 2014 (Ann. Statist. 42(6)) give no non-permutation test (sec 4.3, 5.1) and state pdCor = 0 is NOT conditional independence (sec 4.2): R-40, pdcor is reported as a DEPENDENCE test, not a CI test. Source, target and every Z column are z-standardized before the distances (frozen E2); energy / dcor use raw Z, so the Euclidean metric on Z differs when \|Z\| > 1. Score: signed pdCor (proj_perm), \|pdCor\| (x_perm cross-check). Fast permutation loop == frozen loop (test). |
| rcot2 | frozen `e2slice/discovery_rcot(_v2)` numerics; authors' null from RCIT (Strobl et al. 2019; github ericstrobl/RCIT 7a7fb2b) via momentchi2 0.1.8 (PyPI, MIT) | RCIT "lpd4": 1 - LPB4(eigenvalues, T), HBE fallback, clip 0 (p may be exactly 0, as RCIT) | num_f 100 (RCIT default; frozen 25 = cross-check), num_f2 5; frozen statistic numerics (first 200 rows for the median, centred covariance, eigenvalues > 1e-12) differ in detail from RCIT's R code -> F2 distributional; block-permutation null (dz 25, B 9999 BC) = cross-check `block_perm_config()`; numpy-2 shim for momentchi2's `np.math`. |
| cmi_knn | tigramite 5.2.10.1 `CMIknn` (Runge 2018) | tigramite nearest-neighbour restricted shuffle (permute Y, shuffle_neighbors 5) | sig_samples 500 -> sequential 9999 (R-9) by a subclass whose loop is tigramite's verbatim (== tigramite at native B, test); seed from tag 7802; defaults kept (knn .2n, ranks). Backend "torch" (GPU): same estimator and null, only the neighbour search replaced (noise + rank transform verbatim on CPU; integer max-norm counts on the device: tigramite's query_ball_point(0.999999999 eps) == count of d < eps for integer ranks; exact per-column distance cache). F2-GPU passed -> "torch" is the PRODUCTION backend (device cuda when available, else torch-cpu, both exact); `native_config()` keeps tigramite's own search ("cpu"). |

## Gates F2-F5 (all arm native: fidelity of the port itself)

| adapter | F2 port == reference | F3 known result | F4 synthetic null level (raw p <= .05, primary family, Wilson 95%) and recall | F5 |
|---|---|---|---|---|
| mscr | native mode bit-equal to `discover_mscr` (test) | EXACT: 48/48 stage-B p, E2 seed 0 (F3_FROZEN_mscr) | PASS: null .054 [.045, .064] n 500, .049 [.040, .059] n 1000 (100 reps each); planted-data null edges .048 / .048; planted P0 96/100, 100/100, P1 91/100, 100/100, gated P2 21/100, 74/100, P3->K2 9/100, 43/100 (n 500 / 1000); placebo declared (BY): null 0 / 0, planted 0 / 3 | tau rule only (DEV) |
| pcorr | p == OLS coefficient t-test, 720 tests, max rel diff 1.2e-13 (F2_LOCAL); eq arm too (test) | analytic Gaussian partial correlation within tol, null size .048 (F3_KNOWN) | PASS: null .052 [.046, .060] n 500, .049 [.042, .056] n 1000 (200 reps); planted P0, P1 200/200, gated P2 86 / 177 of 200, P3->K2 0/200 (gated, linear test misses) | tau rule only |
| pdcor | statistic == dcor 0.7, 60 cases, max diff 2.8e-16; null rates == dcor's test (F2_LOCAL) | EXACT: 84/84 stored E2 pdCor (max diff 6e-16, p diff 0); dcor doctest values (F3_KNOWN) | PASS on the synthetic null: .048 [.040, .058] n 500 and n 1000 (100 reps each); planted-data null edges .060 [.049, .073] n 500 (100 reps), .060 [.046, .079] n 1000 (50 reps): borderline, CI includes .05; planted P0, P1 all found, gated P2 45/100, 41/50, P3->K2 0 (n 500 / 1000); placebo declared (BY): null 1 / 0, planted 1 / 0. F2 finding: rejects a linear-Gaussian CI null 82% when x correlates with Z (pdCor = 0 != CI, authors' caveat; R-40 dependence test) | tau rule only |
| rcot2 | momentchi2 py 0.1.8 vs R 0.1.5 EXACT on 1000 CDF cases (lpb4 2.7e-11, hbe 1e-14, sw 1e-15; tol 1e-6). Primary vs R RCIT defaults, 200 reps x 12 scenarios: null rejection rates agree 8/8; power 1-8 pp below RCIT in 3/4 alt scenarios (fails the pre-stated 2-SE tolerance there; statistic numerics differ) | EXACT: 84/84 stored E2 RCoT-v2 statistics (rel 3e-15), p diff 0 | FAIL (finding, not fixed): null .084 [.076, .093] n 500, .065 [.058, .073] n 1000; RCIT itself is liberal on the same F2 nulls at \|Z\| 13 (.075-.11): the authors' approximate null, not the port. Planted P0 177 / 191, P1 169 / 194, P2 33 / 92 of 200 (n 500 / 1000) | tau rule only |
| cmi_knn | == tigramite at native B (test). torch backend == tigramite bit for bit (statistic and p) locally: E1 R2, E2 R1/R2, E4 R3 (n 400-1500), tests on torch-cpu. F2-GPU PASS (Kaggle Tesla T4, torch 2.11.0+cu128; results/citests/gpu/F2_GPU.json; tolerance EXACT): neighbour counts identical at n 500 / 1000 / 4000 / 4000 (\|Z\| 41 / 18 / 32 / 7); run_test_raw statistic and p bit-identical 18/18 (E1 R1, E2 R1, E2 R2, E3 R2, E4 R3, E5 R1; n 500 and 1000) | analytic Gaussian CMI at knn 10 within tol; default knn .2n biased (mean err -.054, descriptive) | PASS (GPU backend, = tigramite): null .044 [.036, .054] n 500 (100 reps), .041 [.031, .054] n 1000 (60 reps); planted P0 19/20, 8/8, P1 20/20, 8/8, gated P2 2/20, 3/8, P3->K2 1/20, 0/8 (n 500 / 1000; 20 / 8 reps); placebo 0 declared | tau rule only |

F4 data: `_citests_synth` (R1-like, i.i.d. uniform actions, AR(1) KPIs; planted P0->K0 +, P1->K1 -, P2,P3->K2 gated,
K3->K3), seeds 3_000_000 + rep (tag 7802/99). F4 heavy (mscr, pdcor, cmi_knn): primary candidates only (the
KPI -> KPI self edges would cost full B each). F4 heavy ran on Kaggle (4 CPUs, 12 382 s wall); mscr null n 500 reps 0-83
come from the earlier Colab run f4h-a (same code path for arm native: group-0 bank stream unchanged by audit L4),
the rest from f4h-k1; numba / llvmlite were Kaggle's preinstalled versions (the job predates the R-35 pins).
Validity observation for R-21 / dev-runs (not tuned on): harness E4 R1, n 500, kappa .25, arm eq, DEV smoke:
cmi_knn (knn .2n) gave P_placebo -> K0 zero exceedances in 2464 restricted-shuffle surrogates while the true
P0 -> K0 had p .22: the local-permutation null looks liberal with this Z (audit-citests: E4 R1 placebo
inconclusive; cmi_knn holds its level in E1 / E2 R2: eq nulls 1/60, 8/160, placebo 0/50).

## Cost (R-13; budget 2 CPU-h per (method, dataset))
Measured at n 1000, kappa 0, arm native, one DEV dataset per cell (Colab TPU host CPU, single-threaded; run
xm-citests-int-3): pcorr ~0 s; rcot2 <= 3 s; mscr <= 23 s; pdcor 3.5-1751 s (E2 R2 largest); cmi_knn CPU at
native B 500: 3437 s for E1 R2 (R-9 B 9999 ~ 20x more per true edge).
cmi_knn torch backend under R-9 (B 9999 BC), Kaggle Tesla T4, one DEV dataset E5 R1 seed 3_000_011, kappa .25, arm eq
(\|Z\| 18, 36 candidates), GPU wall time (R-35: reported separately from the CPU budget; results/citests/gpu/cost_*):
- n 1000: 0.0075 s per surrogate; whole dataset 172 s (23 071 surrogates; 2 candidates at the full 9999, 75 s each).
- n 4000: 0.074 s per surrogate; 31 of 36 candidates in the 40-min cap (2 399 s; 3 at 9999, ~740 s each); whole
  dataset ~2.4k s if the 5 remaining stop like the nulls, <= 6.1k s if all 5 reach 9999.
- n 8000: 0.27 s per surrogate (one true edge ~2.7k s); the cap ended inside the first true edge. Extrapolated with
  the n 4000 draw profile ~9k s; upper bound (all 36 at 9999) 98k s.
- Per-surrogate time grows ~n^1.7-1.9 (O(n^2) pair distances). For comparison, CPU tigramite on the Kaggle CPU:
  0.2 s (n 1000, \|Z\| 18) and 5.3 s (n 4000, \|Z\| 32) per neighbour search (F2_GPU counts), i.e. ~20-40x slower.
Cost / validity / power for every method and arm on the final grid (and the R-13 feasibility labels): dev-runs.

## Findings outside the gates
- Frozen `e2slice` `_hbe_pvalue` is Satterthwaite-Welch, not Hall-Buckley-Eagleson (frozen code untouched).
- In R2 the native arm declares nulls and the placebo (pcorr n 1000 kappa .25: null FPR E1 .17, E2 .38, E3 .19,
  E5 .30; placebo 4 in E2, 2 in E5); the eq arm declares none of them in the same cells.

## Deviations from the contract text (disclosed)
- The exact momentchi2 R side (1000 CDF evaluations) ran once in a local throw-away Docker r-base 4.6.0
  container (CRAN momentchi2 0.1.5), not on Kaggle/Colab ("R runs on Kaggle/Colab"): the Kaggle job's R script had
  skipped it; the step is now in `f2_rcit_job.sh`.
- F3 mscr regenerated the stage-B E2 seed-0 input in memory (R-11 spirit, confirmed).
- Data copies `scratchpad/xmethod/citests/f3root/` (read-only replicate-00 of runs/e2slice-recovery) were
  committed once by mistake (1b761fd) and removed in the next commit (d01b2d3); still in this branch's history.
