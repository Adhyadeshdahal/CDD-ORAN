READY-TO-MERGE (branch xm/audit-citests from feat/v2 e1d12f1; report only, no method code changed)
# audit-citests status (2026-10-03)

## HAND-BACK
Report: scratchpad/xmethod/audit/citests.md (139 lines). Evidence: scratchpad/xmethod/audit/citests/ (scripts,
JSON / JSONL, logs). Tests: tests/test_xmethod_citests.py 28/28 passed.

Verdicts: pcorr OK; pdcor OK (port) + doc fix; rcot2 OK (port); cmi_knn OK; _citests_common OK (LOW L1-L5);
mscr FIX NEEDED (protocol / doc, finding M1).
- Nulls == authors': pdcor proj_perm == dcor 0.7 on identical permutations (p identical 13/13); rcot2 LPB4 num_f 100
  (tail vs Monte Carlo .0036); cmi_knn native == tigramite; GPU torch == CPU tigramite, 24/24 exact (local RTX 2050).
- Arms R-25 / R-28 / R-37: 0 mismatches over 14 cells x 2 arms; no P_placebo_conf conditioner of any other source.
- R-22 at kappa .25: 0 not-testable in all cells. R-4 signs: 0 / 664 mismatches. R-2 BY per family: 0 mismatches.
  R-9 B 9999 h 20. Tag 7802. No truth leakage.
- Validity smoke (n 1000, 5 DEV seeds): pcorr eq clean, native invalid in R2; rcot2 eq liberal at |Z| 41 (E2 R2 .16);
  pdcor not a CI test (E2 R2 eq .22; P_placebo_conf 5/5); cmi_knn clean in E1 / E2 R2; mscr invalid in R2, both arms
  (null 60/60, 152/160), clean on iid R1.
- cmi_knn E4 R1 placebo (author's flag): 45 seeds, 14/180 = .078 [.043,.127], INCONCLUSIVE; the "0 / 2464" case
  did not reproduce (it looks like a deadline-capped probe run).

Compute: local 1 process at a time after the memory note (one early 12-process pool was stopped within minutes);
local GPU for cmi_knn; Kaggle aud-citests-smoke-1 (failed at startup: configs/ missing from the bundle, 31 s) and
aud-citests-smoke-2 (mscr + pdcor smoke, 21 min, complete). Smoke deviations: pdcor / cmi_knn B capped at 999 and
true edges skipped (descriptive raw-.05 rates only).

## QUESTIONS
- Q1 (M1) mscr's eq arm only adds Z_eq columns as extra single conditioners in its max statistic, so it is not
  equal information and is invalid in R2 in both arms. Exclude mscr_eq from C2b (R-31), or keep it with a label?
- Q2 (L1) R-29 conformal tau is not implemented anywhere yet (score.placebo_tau and citests tau_from_scores are
  still R-2's 2nd-largest). Who owns it?
