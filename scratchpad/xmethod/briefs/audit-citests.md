# Brief: audit-citests (independent audit of the five CI-test adapters)

You did not write these. Read CONTRACT.md (R-1..R-28), docs/xmethod/FIDELITY*.md for the CI tests, the author's
status scratchpad/xmethod/status/xm-citests.md, and scratchpad/xmethod/audit/classic_v2.md (format to follow).
Scope: cdd_oran/xmethod/methods/{mscr,pcorr,pdcor,rcot2,cmi_knn,_citests_common}.py (+ any GPU CMI-kNN code).
Check with evidence (file:line, scripts in scratchpad/xmethod/audit/citests/, numbers):
1. Each test's null is the authors' (R-13): pdCor projection permutation, RCoT LPB4 num_f 100, CMI-kNN tigramite
   shuffle test settings, MSCR and pcorr as in their fidelity docs; F2/F3 claims reproduce (spot-check).
2. Arms: eq = design_covariates(focal=<action>) exactly (R-25), lagged-KPI sources per R-28 (identical to
   _classic_common.cond_set), native = R-3 set; masked rows; eq_dropped logic.
3. R-22/R-23 not-testable format (score.py reads notes['not_testable_edges'] or a dict under not_testable), counts
   at kappa .25 (expect ~0); R-4 signs; R-2 BY per family; R-9 Besag-Clifford B 9999 h 20; RNG tag 7802 and
   no reuse across edges; no truth leakage.
4. GPU CMI-kNN reproduces the CPU statistic (tolerance stated) if present (R-26).
5. Validity smoke at kappa .25, n 1000, DEV seeds 3_000_000-004, E2 R2, E1 R2, E4 R1 and R3: truth-null and
   P_placebo rates per test/arm (descriptive). Check the cmi_knn E4 R1 placebo observation in the author's status.
Report only (no method code changes): scratchpad/xmethod/audit/citests.md (< 140 lines; verdict per test OK / FIX
NEEDED with exact fix). Branch xm/audit-citests from feat/v2 (after the citests merge). Status
scratchpad/xmethod/status/audit-citests.md (line 1 READY-TO-MERGE). Commit locally, never push.
