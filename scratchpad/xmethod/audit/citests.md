# Independent audit of the five CI-test adapters (audit-citests, 2026-10-03, feat/v2 e1d12f1)

Scope: `methods/{mscr,pcorr,pdcor,rcot2,cmi_knn,_citests_common}.py` (+ cmi_knn torch / GPU) vs CONTRACT R-1..R-37,
FIDELITY_CITESTS.md and the author's status. Evidence: `scratchpad/xmethod/audit/citests/`; DEV seeds; kappa .25
unless stated; no method code changed. Tests (`PYTHONPATH=. uv run --group baselines --group citests pytest -q
tests/test_xmethod_citests.py`): 28/28. A plain `uv run` re-syncs to default groups and uninstalls tigramite.

## Verdicts

| test | verdict | one line |
|---|---|---|
| pcorr | OK | p == OLS t-test (rel 1.7e-12). Eq arm holds its level in R2 (E2 R2 null .06, placebo BY 0). |
| pdcor | OK (port), DOC FIX | Projection-permutation null == dcor 0.7 bit for bit. Not a CI test: E2 R2 eq null .22; P_placebo_conf 5/5 (P1). |
| rcot2 | OK (port) | Authors' LPB4 null, num_f 100, num_f2 5. Eq arm liberal at \|Z\| 41: E2 R2 null .16 (R1). |
| cmi_knn | OK | Native == tigramite. GPU == CPU exactly (24/24). Holds its level in E1 / E2 R2. E4 R1 placebo inconclusive (C1). |
| mscr | **FIX NEEDED (protocol / doc)** | Faithful port. Invalid in R2 in BOTH arms (null 60/60 and 152/160). Its "eq" arm cannot use the design (M1). |
| _citests_common | OK (LOW L1-L5) | Arms, R-37, R-22, R-4, R-2, R-9, RNG, no leakage: all verified. |

## 1. Nulls are the authors' (R-13): `null_check.py`, `gpu_check.py`

- **pdcor** (`proj_perm`): T = n <P_z(x), P_z(y)>, permute P_z(x), one-sided, as energy `pdcov.test` / dcor. Same
  inputs and Generator as dcor (E2 R2 n 300 eq, B 499): T rel diff 1.2e-15, p identical 13/13. Ties >= (dcor >).
- **rcot2**: lpd4 = 1 - LPB4(eig, T), HBE fallback, clip at 0; dz 100, dxy 5. No resampling, so R-9 does not apply.
  LPB4 vs a 200k Monte Carlo of sum w_i chi2_1 on real eigenvalues (9 points): max |diff| .0036. F2 / F3 JSONs = doc.
- **cmi_knn**: tigramite 5.2.10.1 defaults (knn .2, shuffle_neighbors 5, ranks, permute Y) confirmed in source; the
  sequential loop is tigramite's line for line ((1+c)/(B+1)) + stopping; native vs fresh tigramite: equal 3/3.
- **mscr, pcorr**: as documented (native == `discover_mscr`, F3 48/48; t-test df n - 2 - |Z|, F2 1.2e-13). Stale doc:
  `f2_local.py` says the pdcor nulls "DIFFER by construction"; the proj_perm primary IS dcor's null.

## 2. Arms (R-17 / R-25 / R-28 / R-37): `arms_check.py`

- Every source of 14 cells (all worlds / regimes; E4 lambda 1 and 1.5), both arms, vs an independent expectation from
  `design_covariates` (action: helper(focal) / R-3 set; lagged KPI: base set minus own `lag_kpi`(+`:missing`) plus
  all actions at t; minus P_placebo_conf columns; mask rows; minus constants): 0 name and 0 value mismatches.
- **R-37 holds**: 0 P_placebo_conf columns (concurrent, @t-1, @t-2, sp:, :missing) in any other source's Z, in
  either arm. Classic's `cond_set` on this branch still has them; R-37 aligns classic on xm/hac.
- Rows: eq 998/1000 (t-1 / t-2), native all. eq_dropped empty (no constant S column) in 12 cells x 5 seeds, n 500 / 1000.

## 3. R-22 / R-4 / R-2 / R-9 / RNG / leakage: `arms_check.py`

- **R-22 / R-23** (pcorr, both arms): kappa .25: 0 listed, 0 unlisted NaN in all 12 cells (min residual share .052).
  kappa 0: E1 36/36, E3 50/50, E4 R3 4/4 listed; E5 see L3. Format = score.py's (dict + list); 0 overridden.
- **R-4**: sign == sign of the OLS coefficient given the arm's Z. 0 mismatches over 664 checks (166 candidates x pcorr
  native / rcot2 `pcorr_given_Z` x 2 arms; E2 R2, E1 R2, E4 R3, E5 R1).
- **R-2 / R-6 / R-10**: declared == statsmodels `fdr_by` per family (action_kpi, kpi_kpi, placebo_conf): 0 mismatches.
- **R-9**: B 9999 and h 20 in the mscr, pdcor and cmi_knn defaults. p = h/k at the stop, else (1+c)/(B+1).
  The fixed native B appears only in `native_config()`.
- **RNG**: SeedSequence([7802, data seed, key]); pdcor/rcot2 per (tgt, src); cmi_knn one stream; mscr per tgt (L4).
- **Leakage**: no truth import. With `meta` stripped to `context_names`, scores, p and signs are identical.

## 4. GPU cmi_knn (R-26): `gpu_check.py` (local RTX 2050, torch 2.10 cu128)

- torch == CPU tigramite (same seed, B 10-40): statistic and p diff 0 on 24 tests (E4 R1 n 1000, E4 R3 n 600, E1 R2
  n 500, E2 R2 n 400; both arms; |Z| 2-41); 0.02-0.04 vs 0.24-1.4 s per surrogate. R-26 passed on 3 worlds (the
  author's 6-world Kaggle F2-GPU still pending).
- CPU cost: E2 R2 eq n 1000 (|Z| 41) takes 5.0 s per surrogate single-threaded, ~14 CPU-h per true edge at B 9999,
  so the CPU backend is infeasible under R-13. GPU: 216-270 s per dataset (nulls + placebos, B cap 999).

## 5. Validity smoke (descriptive; n 1000, seeds 3_000_000-004, lambda 1): `smoke.py`, `smoke_summary.py`

Raw = p <= .05 (Wilson 95 %); BY = declared in action_kpi. pcorr / rcot2 local, all candidates; mscr / pdcor Kaggle
`aud-citests-smoke-2`; cmi_knn local GPU. pdcor / cmi_knn: null + placebo candidates only (cost), B cap 999 BC h 20.

| cell | test | eq null raw (CI) / BY | eq Pl raw / BY | native null raw (CI) / BY | native Pl raw / BY |
|---|---|---|---|---|---|
| E2 R2 | pcorr | 10/160 (.03,.11) / 0 | 3/30 / 0 | 79/160 (.42,.57) / 49 | 15/30 / 10 |
| E2 R2 | rcot2 | 26/160 (.11,.23) / 2 | 1/30 / 0 | 29/160 (.13,.25) / 12 | 7/30 / 0 |
| E2 R2 | pdcor | 35/160 (.16,.29) / 20 | 5/30 / 3 | 66/160 (.34,.49) / 53 | 11/30 / 6 |
| E2 R2 | cmi_knn | 8/160 (.03,.10) / 0 | 0/30 / 0 | 9/160 (.03,.10) / 0 | 3/30 / 0 |
| E2 R2 | mscr | 152/160 (.90,.97) / 145 | 27/30 / 27 | 149/160 (.88,.96) / 141 | 27/30 / 26 |
| E1 R2 | pcorr | 3/60 (.02,.14) / 0 | 1/20 / 0 | 11/60 (.11,.30) / 10 | 1/20 / 0 |
| E1 R2 | rcot2 | 5/60 (.04,.18) / 0 | 0/20 / 0 | 9/60 (.08,.26) / 4 | 0/20 / 0 |
| E1 R2 | pdcor | 2/60 (.01,.11) / 0 | 0/20 / 0 | 5/60 (.04,.18) / 5 | 10/20 / 8 |
| E1 R2 | cmi_knn | 1/60 (.00,.09) / 0 | 0/20 / 0 | 0/60 (.00,.06) / 0 | 1/20 / 0 |
| E1 R2 | mscr | 60/60 (.94,1) / 60 | 20/20 / 20 | 60/60 (.94,1) / 60 | 20/20 / 20 |

E4 (5 seeds; raw counts, eq / native):

| E4 row | pcorr | rcot2 | pdcor | cmi_knn | mscr |
|---|---|---|---|---|---|
| R1 placebo | 0 / 0 | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| R3 placebo | 1 / 1 | 0 / 1 | 2 / 2 | 1 / 0 | 0 / 0 |
| R3 P_placebo_conf | 0 / 0 | 4 / 2 | 5 / 5 | 0 / 1 | 5 / 5 |

Recall (BY-declared true primary edges, eq / native): pcorr E2 29 / 46 of 80, E1 20 / 30 of 30; rcot2 E2 45 / 52,
E1 20 / 27.

mscr control on iid R1 (`mscr_r1.py`, 2 seeds, both arms): E1 null 2/48, placebo 0/16; E2 null 3/128, placebo 0/24.

## Findings

- **M1 (HIGH, protocol / doc): mscr is invalid in R2 in both arms, and its eq arm is not equal information.**
  - Rates: E1 R2 null 60/60, placebo 20/20; E2 R2 152/160 and 27/30. On iid R1 it holds its level, so the port is
    not the cause.
  - Cause 1: its frozen scope (`discovery/mscr.py` docstring) says "VALIDATED ON RANDOMIZED DESIGNS ONLY" (E1 FDR .81).
    The within-stratum partition null assumes exchangeable rows; R2 rows are serially dependent (slow setpoints).
  - Cause 2: S* = max over SINGLE conditioners g, so the Z_eq columns only add more g's to that max. Nothing is
    conditioned jointly, so the design cannot help.
- **P1 (MEDIUM, disclose + doc): pdCor = 0 is not CI** (Szekely-Rizzo 2014 sec 4.2; already in F6).
  - Rates: E2 R2 eq null .22, native .41; P_placebo_conf 5/5 in both arms; native E1 R2 placebo 10/20.
  - The port is exact; R-21 / R-30 will mark it INVALID.
  - Doc errors: `pdcor.py:11` says the score is |pdCor| (proj_perm uses the SIGNED pdCor). Z is z-standardized before
    the Euclidean distance (frozen), while energy / dcor use raw Z: the metric differs for |Z| > 1 and is not in F6.
- **R1 (MEDIUM, disclose): rcot2 eq is liberal at large |Z|.**
  - E2 R2 null .16 [.11,.23] at |Z| 41; P_placebo_conf eq 4/5.
  - Consistent with F4 (.084 / .065) and RCIT's own .075-.11 at |Z| 13: 100 RFFs cannot residualize a 41-dim Z.
  - The authors' null; report, do not tune.
- **C1 (observation, inconclusive): cmi_knn E4 R1 placebo** (`smoke.py e4more`, 45 seeds, B 9999 BC).
  - Raw rates: n 500 eq 5/45, native 3/45; n 1000 eq 4/45, native 2/45.
  - Pooled 14/180 = .078 [.043,.127] (R-30: INCONCLUSIVE); eq 9/90 = .10.
  - The author's "0 exceedances in 2464 surrogates" was not reproduced (max 2335 draws, p .0086). 2464 draws with no
    h-stop under BC h 20 means a deadline-capped cost-probe run, whose p is invalid by design.
  - cmi_knn holds its level in E1 / E2 R2 (eq nulls 1/60 and 8/160; placebo 0/50).
- **L1 (LOW): the tau rule is still R-2.** `tau_from_scores` / `tune()` take the 2nd-largest placebo score, not R-29's
  conformal cutoff. So does `score.placebo_tau`, which EVAL uses (`eval_analysis.py:309`).
- **L2 (LOW): guarded candidates are scored p = 1.0 inside the BY family** (pdcor `pdcor.py:130,151`; rcot2
  `rcot2.py:157,170`), not NaN plus not-testable as R-22 asks. 0 guards in every smoke run.
- **L3 (LOW, kappa 0 only): STOP NaNs are not listed.**
  - pcorr eq on E5 R1 / R2 STOPs "rank-deficient" (lag K2 == P0@t-1, as classic finding B). 27 NaN candidates are not
    listed as not testable, so n_not_testable undercounts.
  - Latent: `_zstd` in pcorr / pdcor STOPs on any zero-variance column of S, even one no Z uses (0 cases seen).
- **L4 (LOW): mscr reuses the bank stream `[seed, j, n_perm]` across column groups of a target.** Eq has 1 group in
  E1 / E2 / E5, but 2 in R3 / R4 (main + P_placebo_conf), so the diagnostic shares the main bank's draws.
- **L5 (LOW, notes only):** `declared_native` of pcorr / pdcor / rcot2 includes P_placebo_conf edges (mscr excludes
  them).

## Fixes (exact)

1. M1:
   - FIDELITY_CITESTS F6 mscr row: add "frozen scope: randomized designs only (E1 FDR .81); max over single
     conditioners, so Z_eq enters only as single conditioners".
   - Protocol (orchestrator, Q1): exclude mscr_eq from C2b (R-31) or label it; expect INVALID for mscr in R2 at R-21.
2. P1: `pdcor.py:11`: "score: signed pdCor (proj_perm); |pdCor| for the x_perm cross-check". F6: add the Z
   standardization. `f2_local.py` docstring: proj_perm == dcor's null.
3. L1: implement R-29 once, in `score.placebo_tau`; `_citests_common.tau_from_scores` calls it.
4. L2: in pdcor / rcot2 `_test`, set guarded p to NaN and list the edge under `notes['not_testable']` (reason "guard").
5. L3: in `CITestBase.run`, when `_test` raises, list the kept candidates under `not_testable` with the STOP reason.
6. L4: `mscr.py:80`: `default_rng([seed, j, n_perm, group_index])` (changes R3 / R4 eq streams only), or disclose.
7. C1, R1, P1 need no code change. R-21 / R-30 on the full DEV grid decides validity.
