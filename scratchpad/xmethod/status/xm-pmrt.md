# xm-pmrt status (2026-10-02): READY-TO-MERGE (STEP 1, R-14; EXPERIMENT D done)

## HAND-BACK
**Deliverables** (branch xm/xm-pmrt, merged feat/v2 138576e, no conflicts; no push / merge by me)
- `cdd_oran/xmethod/methods/pmrt_core.py` PmrtCore (`pmrt_core`, `pmrt-core-v1`): design-based CRT on the known
  random part (iid / dither / logged incl. harness "categorical_rows", "clipped_normal"; kind none or unknown law ->
  not applicable, never a permutation); past-only ridge adjustment (lag KPIs, ctx, setpoints, 2 lagged action rows,
  concurrent other designed actions, R-8); NO clip by default (R-14; huber_c option); max B 9999 Besag-Clifford
  h 20 (R-9); RNG [seed, 7801, action_idx, split]; plain BY over meta["primary_candidates"] (R-6), P_placebo_conf
  its own BY; tune() = defaults + placebo tau. Runner path "cdd_oran.xmethod.methods.pmrt_core:PmrtCore".
- docs/xmethod/PMRT_CORE.md (paper text: statistic, kept / reduced / dropped table, "Why no clip" with numbers).
- tests/test_xmethod_pmrt.py: 17 pass. Scripts scratchpad/xmethod/{pmrt_equiv, pmrt_fidelity, pmrt_integration,
  pmrt_null_diag, pmrt_q4_diag, e6_audit}.py; results/ (f4 = with clip, f4_noclip = R-14 production, q4, nulldiag).
- No dependency added (scipy via `uv sync --group baselines`).

**Fidelity (R-14 production config)**
- F1 reference pmrt.py pmrt-v1 + E6P_PMRT_V4.json. F2/F3: core engine = pmrt bit-exact p for plain_c AND plain
  (180 hyp, 3 E6 caches, B 9999); own adjustment corr z .91 vs plain_c.
- F4 synthetic (Kaggle xm-pmrt-f4-noclip-1, 200 reps, 2800 null p / cell, B 9999 seq): rate at .05 R1 .055 / .045
  / .056, R2 (confounded slow setpoint) .051 / .053 / .053, R1t (t3) .044 / .044 / .050 at n 500 / 1000 / 4000, every
  CI covers .05. Power (BY, 50 reps, sign always right, mean FDP <= .02): R1 beta .1: 1.0 at every n; beta .05
  .30 / .84-.88 / 1.0; R2 beta .2 .02-.08 / .30-.58 / 1.0, beta .1 at 4000 .54-.72; R1t beta .05 at n 500
  .42-.46 (with clip .66-.68), n 1000 .92-.94 (.98).
- Harness worlds, truth-null primary edges (40 DEV seeds, n 1000, seed-cluster CI): E2 R2 .052 [.037,.067] / .009
  (with clip .070 / .020); E1 R1 .062, E1 R2 .048, E2 R1 .043, E3 R2 .062, E5 R2 .057, E4 R3 lam 1 / 1.5 .050 / .075
  (40 p each); all CIs cover .05; .01 rates .000-.014.
- F5 no tuning (tau via tune()); burn rule fixed before E-series data. F6 deviations: PMRT_CORE.md table + R-8
  concurrent covariates, R-9 sequential B (native pmrt fixed 9999), R-14 no clip, plain BY, past-only ridge.
- POWER COST of R-14 (descriptive, 10 DEV seeds, E2 R2): mean recall .17 no clip vs .31 clip at n 1000, .34 vs
  .48 at n 4000 (false positives 0 vs 1 in total). E2 is noiseless with Gaussian-bump KPIs; the clip helped there.

**Integration** (n 1000, seed 3_000_000, runner.load_method + score.score, R-14 config): E1 R1 / R2 4/4 fp 0;
E2 R1 5/16; E2 R2 0/16 (see power cost); E3 R1 / R2 4/4; E5 R1 / R2 2/6; E4 R1 1/1; E4 R2 0/1; E4 R3 lam 1 /
1.5 1/1 with the correct negative sign; E4 R4 P0 not applicable. 0 false positives, 0 placebo / placebo_conf
declarations in all 14 cells; sign accuracy 1.0 wherever defined.
**Scaling**: n 24000 42-48 CPU-s / 135 MB; n 4000 seq 11-14 s vs native 22 s (measured with the clip; the clip
is O(n) and does not change the cost).

## Q4: resolved by R-14 (mechanism and numbers: PMRT_CORE.md "Why no clip", results/q4/).

## EXPERIMENT D (R-15; E6 validity audit; descriptive, post-hoc, disclosed; frozen v4 verdict unchanged)
Kaggle xm-pmrt-e6audit-1: v4 loaders, artifact E6P_PMRT_V4.json read only, B 9999, same RNG streams; recomputed
loadsp_c p2 / p_plus / p_minus / used p = stored v4 tables on 1020 hyp-tables (0 mismatches). Two-sided raw p on v4
GT-NULL hypotheses, cluster = episode slice (10 x s60, 5 x s120; pooled / placebo: 1 cluster, Wilson over hyp).
- A, all 21 GT-NULL (R-15 definition): s60 .152 [.110,.195] at .05, .100 at .01; s120 .152 [.124,.181], .086;
  pooled 2/21. loadsp (no clip) the same: .157 / .143 at .05. The clip does not drive this.
- But v4 GT "NULL" is an EQUIVALENCE label (gt_p.classify: GT CI inside +-delta, delta = 5 % of the KPI's max
  |mean|), not a sharp null. The excess comes from 2 hypotheses whose GT CI is far from 0 but below delta:
  sleep|own|e (GT -187 [-192,-183], delta 359; PMRT rejects 10/10 s60, sign -1 = GT) and carrier|own|load (GT
  -26 [-32,-20], delta 45; 8/10, sign -1 = GT). That is detection of real sub-delta effects, not false positives.
- C, the other 19: s60 .074 [.032,.121] / .021; s120 .063 [.032,.095] / .000; pooled 0/19 (loadsp .079 / .053).
  B, GT CI contains 0 (12 hyp, the closest proxy for a true null): s60 .075 [.033,.117] / .017; s120 .083
  [.033,.133] / .000. CIs cover .05; point estimates .06-.08, upper limits ~.12; 10 / 5 clusters only.
- Placebo logs (exact sharp null: accept applied whatever the mode; 60 hyp, 40 eps): loadsp_c 6/60 = .100
  [.047,.201] at .05, 2/60 at .01 (used p 7/60); loadsp 4/60 = .067, 0/60 (used p 6/60). The clipped arm is a bit
  higher, in the E-series direction, but one cluster cannot separate them. The frozen K0 (0 declarations) still holds.
- Reading: no clear evidence of invalidity of PMRT in E6 on these data. Sets B / C sit at .06-.08 (CIs include .05);
  the clip matters little on GT-NULL; a small clip-related placebo excess cannot be ruled out. Files:
  results/e6_audit/{e6_audit.json, e6_audit_breakdown.json, job.log}; scripts e6_audit.py, e6_audit_breakdown.py.
