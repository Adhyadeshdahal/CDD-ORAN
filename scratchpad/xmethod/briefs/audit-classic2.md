# Brief: audit-classic2 (independent re-audit of the changed classic methods)

You are an independent auditor. You did not write these methods. Read first: scratchpad/xmethod/CONTRACT.md
(sections 1-5, rulings R-1..R-27), scratchpad/xmethod/briefs/audit-F7.md and the earlier audits in
scratchpad/xmethod/audit/{two_tower,shap_dag,pc,granger,corr,notears}.md, then docs/xmethod/FIDELITY_CLASSIC.md
and scratchpad/xmethod/status/fix-classic.md (the author's hand-back, merged in feat/v2 at the fix-classic merge).

Scope: cdd_oran/xmethod/methods/{two_tower,shap_dag,pc,granger,_classic_common}.py and their tests.
Check, with evidence (file:line, small scripts under scratchpad/xmethod/audit/classic2/, numbers):
1. Every earlier audit finding: fixed / partly / not fixed.
2. Arms (R-17/R-18/R-25): eq arm uses design_covariates(data, focal=<action>) exactly (concurrent='designed');
   native uses concurrent='all'; lagged-KPI sources handled as documented; pc tiering of exogenous nodes is sound;
   conditional Granger eq arm is a correct VARX test with the authors' null.
3. R-22/R-23: not-testable detection is correct and not over-triggering at kappa .25 (generate_dataset kappa=.25):
   count not-testable cells per world at n 1000, seed 3_000_000, both arms. Expect ~0.
4. Signs (R-4), declared flags vs BY q .05 (R-2), tau rule for score-only methods.
5. two_tower row-share score and adaptation label; shap_dag XGBoost defaults; no truth leaks into any Dataset path.
6. Quick validity smoke at kappa .25: E2 R2 and E1 R2, n 1000, 5 DEV seeds (3_000_000-004), truth-null primary edge
   and P_placebo declaration counts per method/arm (descriptive, not a gate; the full DEV check is another job).

Work on branch xm/audit-classic2 in your worktree (create from feat/v2). Do NOT change method code; report only.
Output: scratchpad/xmethod/audit/classic_v2.md (< 120 lines; verdict per method: OK / FIX NEEDED with the exact
fix) and status scratchpad/xmethod/status/audit-classic2.md (line 1 "READY-TO-MERGE" when done; questions as
"- Q<n> ..."). Local CPU is enough; if you need cloud, follow R-26. Commit locally, never push.
