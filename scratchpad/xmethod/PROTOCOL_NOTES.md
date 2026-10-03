# Items for the user's protocol review (collected during build)
- Harness judgement calls J1-J10 and deviations D1-D2: see xm-harness hand-back (branch xm/xm-harness,
  scratchpad/xmethod/status/xm-harness.md). Notable: R2 = Stage 0 F-dither (delta .10, 20 blocks); E5 = fully
  composed, observation noise kappa .3; E4 R3 = frozen behaviour policy rounded to the 101-point do-grid with the
  confounder observed; signs only where monotone (E2 has 1 signed edge).
- Primary candidate counts (true/null): E1 20 (4/16), E2 54 (16/38), E3 25 (4/21), E4 2 (1/1), E5 20 (6/14).
- Rulings R-1..R-10 in CONTRACT.md section 8 (declaration rule, conditioning sets, signs, comparison set, R3 policy,
  PMRT adjustment, common sequential B = 9999, placebo columns).
- Cross-platform: datasets byte-identical per platform only; EVAL data generated where methods run.
- PMRT core (xm-pmrt hand-back): engine bit-exact vs E6 pmrt plain_c; F4 null level .043-.058 (R1, R2, heavy-tailed);
  power: R1 good from n 1000; R2 needs n 8000-24000 because only the small dither is random (reductions table in
  docs/xmethod/PMRT_CORE.md, on branch xm/xm-pmrt). Plain BY (no independent prior), concurrent actions as covariates.
- Score-only tau for the secondary KPI->KPI family reuses the action-placebo tau (no KPI placebo exists) (xm-classic Q-C2).
- pdCor: authors' projection permutation (one-sided); RCoT: authors' num_f 100 + LPB4 approximate null (dependency, no GPL port); see xm-citests Q9.
- PMRT core without clip (R-14): null rates nominal everywhere (E2 R2 .052); power cost vs clip in E2 R2: recall .17 vs .31 (n 1000), .34 vs .48 (n 4000). Heavy tails (t3) n 500: .42-.46 vs .66-.68.
- Experiment D (E6 audit, xm-pmrt): GT-CI-contains-0 hypotheses: s60 .075 [.033,.117], s120 .083 [.033,.133] at .05
  (CIs cover .05); the apparent excess on all GT-NULL (.15) = 2 real sub-delta effects (sleep|own|e,
  carrier|own|load) detected with the right sign; clip barely matters on GT-NULL; placebo logs loadsp_c 6/60 (.100
  [.047,.201]) vs loadsp 4/60 (.067), one cluster, inconclusive. Frozen K0 (0 declarations) holds.
- DECISION FOR USER: NOTEARS F3 borderline fail (19/20 README edges; one edge at -.296 vs threshold .3; code
  byte-identical to the authors'). Include with disclosure, or exclude per the gate rule?
- DECISION FOR USER: "two_tower" is our E2 port, not the paper's supervised sparsemax model. Label it as an
  adaptation, or implement the faithful model?
- corr / granger p-values invalid under R2 (placebo declared 2-4x), as expected. pc+KCI infeasible beyond n 1000.
- Kaggle runs of xm-classic used --pin off (numba vs numpy 2.4); recheck pin handling before EVAL.
- F7 audit aud1 (classic): two_tower FAIL (not the paper's model; the faithful model trains on truth labels, cannot
  run truth-free; row-scale issue) -> fix row score + label as adaptation [USER: label]; notears PASS-WITH-NOTES
  (recommend include) [USER]; pc, shap_dag, granger PASS-WITH-NOTES; corr PASS. Fixes in progress (xm/fix-classic):
  two_tower row-normalised score, shap_dag -> xgboost (paper), pc fallback rate, "pairwise Granger" label.
- DECISION FOR USER: E1 and E3 are noiseless, so under the equal-information conditioning set every KPI is an exact linear fit and conditional tests are degenerate (R-22: reported as not testable). Option: add an observation-noise variant of E1/E3 (as E5 has, kappa .3) so the eq arm is informative there.
- R-28 (orchestrator, consistency): eq arm for KPI->KPI sources = helper base set minus own lag + all actions at t (classic and citests identical). For user review.
- DEV campaign (dev-runs, user-approved 2026-10-03): tune seeds 3_000_000-019, measure 3_000_100-119 (+120-159 in
  R2 all worlds + E4 R3, n <= 4000). E4 R1/R2 at lam 1 only (lam inert there). Score-only tau computed post hoc per
  (arm, cell) from tune-seed records. FOR USER REVIEW: validity flag (R-20) = lower bound of seed-cluster bootstrap
  95 % CI > .05 on raw p <= .05 rate (p arms; truth-null primary edges and P_placebo) or declaration rate (tau arms);
  not-testable left out of denominators, counted separately.
- 2026-10-03 night (user asleep, said "go through it all, don't await me, I review in the morning"; decisions
  logged for morning review):
  * NOTEARS: INCLUDED with disclosure of F3 19/20 (auditor recommendation; low cost; reversible at reporting).
  * two_tower: labelled adaptation (audit-classic2 fix 5).
  * Budget 2 CPU-h/dataset kept (pilot max 144 s, so no cell is near it).
- 2026-10-03 night: 3 independent protocol reviews (consult/protocol_review/). Adopted as R-29..R-36 (CONTRACT):
  conformal tau, three-way validity with symmetric burden, C2a/C2b split, pcorr_hac baseline, eq_min arm, E4 200
  seeds, integrity/pinning, wording + estimand check. FOR USER REVIEW (all reversible before EVAL is read).
