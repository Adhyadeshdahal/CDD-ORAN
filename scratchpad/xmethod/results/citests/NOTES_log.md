# xm-citests status
Updated: 2026-10-02 (start)

## Plan
1. Read existing impls (mscr.py, e1slice/discovery_v2 pcorr, e2slice/discovery pdCor, discovery_rcot_v2) + api.py.
2. Adapters in cdd_oran/xmethod/methods/: mscr, pcorr, pdcor, rcot2, cmi_knn (wrap, do not edit originals).
   Common base: candidate loop, conditioning set, BY at q=.05, placebo-tau tuning (CONTRACT sec 5).
3. Own synthetic api.Dataset generator (planted + null) for tests until xm-harness lands.
4. Fidelity gates F1-F6 per adapter: pdcor vs `dcor`; rcot2 vs R RCIT on Kaggle/Colab; cmi_knn via tigramite.
5. CPU-s / peak RAM at n in {500,1000,4000,8000,24000}.
6. tests/test_xmethod_citests.py; hand-back.

## Done
- c6463e9: adapters mscr, pcorr, pdcor, rcot2, cmi_knn in cdd_oran/xmethod/methods/ (+ _citests_common, _citests_synth);
  R-6 families, R-9 Besag-Clifford (B 9999, h 20), pcorr_given_Z sign rule, tau tune(); deps group `citests`:
  tigramite 5.2.10.1, dcor 0.7 (numba 0.67 transitively). tests/test_xmethod_citests.py 17 passed, incl. native
  mode == frozen code (mscr bit-equal to discover_mscr; rcot2 == rcot_pvalue_block_perm; cmi_knn == tigramite).

## Done (cont.)
- R-13 implemented: rcot2 primary = lpd4 (momentchi2 lpb4, HBE fallback), num_f 100; block_perm_config() (dz 25,
  BC 9999) + native_config() cross-checks. pdcor primary = authors' P_xz permutation, one-sided, signed pdCor;
  frozen x-permutation = frozen_config(). 20 tests pass.
- FINDING: frozen e2slice `_hbe_pvalue` is NOT HBE but Satterthwaite-Welch (== momentchi2.sw to 1e-15). Only its
  analytic sensitivity variant used it; the frozen selector was block_perm. Label error in frozen code (not edited).
- F3 mscr: adapter (native, seed 0) == stored MSCR-v2 stage-B seed-0 p-values, 48/48 EXACT (stored replicate-00
  rows predate E2 redesign 9d87a60, so stage-B input regenerated in memory by the frozen generator, code sha256 ==
  stage-B provenance; flag if this stretches R-11). F3 analytic (F3_KNOWN.json): pdcor dcor doctest 142.6664417
  exact; pcorr Gaussian partial corr err .019 (tol .063), t-test null size .0475 [.039,.058]; cmi_knn knn=10
  Gaussian CMI mean err -.007 (tol .02) PASS; at the TEST default knn=.2n mean err -.054 (descriptive, FAILS as an
  estimator check: large-k bias, documented by Runge 2018).
- INTEGRATION (n 1000, seed 3_000_000, all 14 cells) pcorr: runs, scores OK. R2 cells: pcorr declares P_placebo
  (E2-R2 4, E5-R2 2): harness R2 placebo has design "dither" (lag-1 autocorr .74-.86): row-iid tests invalid
  under serial dependence (expected R2 failure, not an adapter bug). Harness-R-10 wording "always i.i.d." vs the
  harness R2 placebo = setpoint + dither: orchestrator please check which is intended.

## Running / next  (updated 2026-10-02 ~08:30Z)
- Kaggle `xm-citests-f2rcit-1`: F2 rcot2 vs R RCIT v1 (pre-R-13 code, dz 25). Queued after it on Kaggle: v2
  (momentchi2 py vs R exact; rcot2 lpd4 num_f 100 vs RCIT defaults), then the R-13 COST campaign (cost.py: 5 methods
  x 5 worlds x n 500..24000, single-thread, hard 2 CPU-h cap, 20-min probes after the first over-budget n).
- Colab `xm-citests-int-3`: integration n 1000, 14 cells, pcorr/mscr/rcot2/pdcor primary configs, then cmi_knn at
  native B 500 (labelled; cmi_knn primary is too slow at n 1000). Then F4 heavy (f4_heavy_job.sh) on Colab.
- F4 local done (results/citests/f4/F4_local_summary.json, 200 reps): pcorr null level .052/.049 (n 500/1000)
  PASS; planted recall P0,P1 200/200, P2->K2 86/177 of 200, P3->K2 (pure interaction, zero mean effect) 0 (expected
  for a linear test). rcot2 primary (lpd4, num_f 100): null level .084 [.076,.093] n 500, .065 [.058,.073] n 1000:
  F4 null-level gate FAILS (liberal, shrinking with n); planted recall P0 177-191/200, P2 33-92, P3 4-0.
- F2 local PASS (F2_LOCAL.json): pdCor stat vs dcor 3e-16; frozen x-perm vs dcor proj-perm rejection agree; pdCor
  rejects a LINEAR-Gaussian CI null 82% when the source correlates with Z (paper sec 4.2 caveat, confirmed);
  pcorr p vs OLS t 1e-13.

## DECISIONS (orchestrator, 2026-10-02)
- Q1 pooled BY over all candidates of the dataset = primary; native rule in notes["declared_native"].
- Q2 conditioning set = all other actions (incl. P_placebo) + non-NaN lagged KPIs (+ context in R3).
- Q3 tau = 2nd-largest P_placebo score pooled over the cell's DEV datasets; declare iff score > tau;
  notes["declared_tau"]; score = raw statistic.
- Q4 pcorr p = partial-correlation t-test (df = n-2-|Z|).
- Q5 every method reports a sign; unsigned stats use sign of pcorr(source, target | same Z);
  notes["sign_rule"]="pcorr_given_Z".

- R-6 primary set + pooled-BY family = action->KPI candidates (incl. P_placebo); KPI->KPI = secondary, own BY
  family. Implemented: by_families(); notes["family_of_edge"].

- R-9 (answers Q6): every resampling method uses max B = 9999 with Besag-Clifford sequential stopping (h = 20:
  p = h/k at stop k, else (1+count)/(B+1)); null unchanged. F6 deviation "B raised from native X to sequential
  9999 for pooled-BY resolution". Measure CPU at n 4000. Native-B runs only as a fidelity-gate check.

- R-11 F3 may read stored datasets of completed studies read-only; outputs in scratchpad/xmethod/results/.
- R-12/R-13 (answers Q8): no n caps; same n grid / seeds / B 9999 / declaration rule for all. Authors' null where
  one exists: rcot2 PRIMARY = Strobl et al. RCoT approximate null (Lindsay-Pilla-Basak, RCIT "lpd4"); block perm
  = fidelity cross-check only. pdcor: check Szekely & Rizzo (2014) for a non-permutation test, use it if given.
  cmi_knn keeps its permutation null. Per-(method, dataset) compute budget identical for all (proposal 2 CPU-h):
  measure cost per dataset at n 500/1000/4000/8000/24000 on DEV; over budget -> "infeasible" with measured cost.
  Integration run: labelled config fine.

## QUESTIONS
- Q9 ANSWERED (confirmed; rcot2 num_f = 100 authors' default, dz 25 cross-check; no GPL port committed ->
  momentchi2 0.1.8 PyPI, MIT, D. Bodenham; numpy-2 shim local to it). Was: (R-13 pdcor finding). Szekely & Rizzo (2014, Ann. Statist. 42(6) 2382-2412,
  sec 4.3 + 5.1) give NO non-permutation test for pdCor/pdCov ("Both tests are implemented as permutation tests").
  Their implementation (R energy 1.7-12 pdcov.test, = python dcor 0.7) permutes the PROJECTED matrix P_z(x)
  (Pxz[i, i]) with statistic n*<Pxz, Pyz>, ONE-sided (replicates > stat). Our frozen E2 pdCor instead permutes the
  raw source then re-projects, two-sided |pdCor|. Per R-13 ("authors' own null"), DEFAULT now: pdcor primary =
  authors' projection permutation, one-sided, score = signed pdCor; frozen x-permutation kept as a cross-check
  (config null="x_perm"). Also: the paper notes pdCor = 0 is NOT conditional independence in general (sec 4.2).
- rcot2 note: RCIT's RCoT default num_f is 100 (current GitHub master 7a7fb2b), our frozen dz = 25; I keep 25
  (frozen calibration) and only swap the null to LPB4 (momentchi2 0.1.5 lpb4, ported; GPL-2|3 code -> licence
  note for the repo). Confirm 25 vs 100.
- Q8 ANSWERED by R-12/R-13. Was: (cost; affects the protocol) cmi_knn at R-9 resolution. One CMIknn evaluation (tigramite default knn = 0.2 n)
  costs ~0.6 s wall at n 1000 and ~7.7 s at n 4000 (16-core laptop, workers=-1); a true edge needs all 9999
  surrogates: ~1.7 h at n 1000, ~21 h at n 4000 PER TRUE EDGE (null edges stop after ~20-200). Even native B = 500
  costs ~4 h per E2 dataset at n 1000. pdcor: ~100 s per true edge at n 500, ~400 s at n 1000, ~6400 s at n 4000
  (O(n^2)). rcot2: ~95 s per true edge at n 500, ~760 s at n 4000. Proposal: cmi_knn only at n 500 (B 9999) and
  pdcor at n <= 1000; rcot2 at n <= 4000; mscr / pcorr full grid. For the integration run I use FULL config for
  pcorr/mscr/rcot2/pdcor and NATIVE B (500, no BC) for cmi_knn (plumbing only), labelled; please confirm.
- Q7 ANSWERED (R-11: allowed, outputs to scratchpad/xmethod/results/). Was: (F3 scope vs seed rule 3). F3 "reproduces a known result": I want to rerun my adapters (native config + the
  frozen seed constants) READ-ONLY on the stored frozen datasets runs/e2slice-recovery/replicate-00..(rows.npz;
  E2 seeds 0-9, an old completed study) and runs/e1slice-v2-envelope/replicate-00.., and match the stored
  discovery.json (pdCor), discovery_rcot_v2.json (RCoT-v2), E1 discovery_v2 (pcorr). No new data from those seeds;
  nothing written there. mscr F3: same on replicate-00 vs a fresh discover_mscr (MSCR v2 left no per-seed p file).
  Is reading those stored datasets allowed under rule 3? Waiting; F3 scripts prepared meanwhile.

## 2026-10-02 afternoon
- Kaggle xm-citests-f2v2-f3 COMPLETE (exit 1 only from a final `tail -5`; fixed to `tail -n 5`). F3 rcot2/pdcor EXACT.
- F2 v2 compare crashed on p_perm = NA (SKIP_PERM=1): compare now skips all-NA columns. f2_rcit.R never had an
  m2 (exact momentchi2) step: added f2_m2.R (wired into f2_rcit_job.sh); ran it once in local Docker r-base 4.6.0
  (CRAN momentchi2 0.1.5): PASS 2.7e-11. Results F2_RCIT.json.
- Network outage: Colab int-3 job-lost at 10:13Z (cmi_knn E1 R2 only; 3437 CPU-s, F1 1.0). Relaunch named int-5
  (int-4 name burnt by a failed TPU assignment, no session created). Kaggle cost-1 launched 15:16Z.

## 2026-10-02 evening: R-17..R-22
- R-17/R-18 arms implemented in the shared core (prepare(data, arm)); helper = local stub until xm/covariates
  lands (signature design_covariates(data) -> (names, matrix, row_mask)). Default arm eq; native_config() native.
- R-22: exact-fit detection = OLS resid var / var of the target on [1, source, Z] < 1e-8. Measured on DEV n 1000
  (seed 3_000_001): E1 / E3 all candidates ~1e-29 in BOTH arms (the native arm is equally degenerate: E1 / E3 KPI
  updates are exactly linear in actions + lagged KPIs); E4 R3 all 4 candidates (K_out = alpha A + theta Z + c,
  Z observed as context); every other cell >= .08. Earlier int-3 E1 / E3 / E4 R3 numbers were computed on
  floating-point residuals (~1e-15 scale) and are superseded.
- pcorr integration both arms: eq removes R2 placebo declarations (E2 R2 4 -> 0, E5 R2 2 -> 0).
- Colab relaunch re-planned as int-5: cmi_knn native (int-3 config) + mscr/rcot2/pdcor arm eq; pending TPU.

## 2026-10-02 ~18:45Z incident (mine): Colab double launch
- My retry launcher looked for "JOBLAUNCHED" (printed remotely only) instead of the local "job-launch" line, so it
  relaunched xm-citests-f4h-a with --force at 18:30 over the healthy 18:08 run; the second job exited 1 and the tick
  then stopped the runtime. Lost: ~30 min of part A (mscr planted n 500 reps); kept: 84 mscr null n 500 reps
  (results/citests/f4/f4_heavy.jsonl, also the seed of the relaunch). 0 orphan assignments afterwards (checked).
- Fixes: success = "job-launch"; retry ONLY on "could not create" (capacity), otherwise stop. F4 heavy split into
  <3 h parts (2-vCPU CPU runtimes when the TPU host is unavailable): mscr, pdcor_null, pdcor_p500, pdcor_p1000 (x30).
