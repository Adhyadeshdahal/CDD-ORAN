READY-TO-MERGE

# pmrt-diag (branch xm/pmrt-diag from feat/v2 79c96ee, worktree CDD-ORAN-wt/xm-classic, 2026-10-03)

Brief: scratchpad/xmethod/briefs/pmrt-diag.md (why pmrt_eq is liberal on E2 R2 truth-null edges at n 1000).
Deliverable: scratchpad/xmethod/results/pmrt_diag/REPORT.md (98 lines). pmrt_core is untouched on both branches.

## Result
- **Verdict:** no flaw in pmrt_core. On redrawn dithers, prod is nominal in every cell: E2 R2 n500 / 1000 / 4000,
  k .125 / .25 / .5, E2 R1, and E1 / E3 / E5 R2 (.049-.053; 600 datasets per cell).
- **Cause of the DEV excess:** chance in those datasets (H6), plus selection and kappa cells sharing seeds 100-119.
  - The DEV 20-seed cell .084 vs replicates .055 +- .009 (P .0005).
  - The exact `inv` is equally high on the same data.
- **H1** (W not invariant to the focal dither) is real in theory but at most +.003 (prod - inv, paired).
- **H2, H3, H4, H5 are false.** H5: fixed B vs Besag-Clifford gives +.0001 [-.0017, .0020].
- **Reserve seeds 160-189:** prod .051 VALID / .060 INCONCLUSIVE / .060 INCONCLUSIVE (k .5 / .25 / .125). None is
  INVALID.
- **Fix (from theory):**
  - Reword pmrt_core's validity as asymptotic (martingale CLT), exact only for a focal-invariant W.
  - Optionally add `inv` as an exact arm where the null implies H0g (no-memory worlds). Its power cost is BY -.017
    to -.025 in E2 and 0 to -.013 elsewhere.
  - Do not use `inv` in dynamic worlds.

## Compute
- Kaggle (all done): pmrt-diag-regen-1, conf-1 (pcorr lanes failed on missing configs/), conf-2, grid-1 (5807 s).
- No Colab or Lightning AI used; 0 Lightning credits (R-44 arrived after the last launch).
- Seeds: DEV 3_000_100-189 only (reserve 160-189 used once, rep -1). EVAL untouched.

## Questions
- Q1 Are E2 R1 + regen replicates acceptable as the "small synthetic analogue"? ANSWERED (orchestrator): yes; no
  separate toy model, and REPORT.md states this.

## HAND-BACK
- **Branch xm/pmrt-diag** (local commits only, never pushed) adds only scratchpad/xmethod/ files:
  - scripts: pmrt_diag.py, pmrt_diag_summary.py, pmrt_diag_power.py, pmrt_diag_kaggle{,2,3,4}.sh;
  - results/pmrt_diag/: REPORT, JSON summaries, raw jsonl;
  - this status file.
- **No src/ or docs/ change.** The PMRT_CORE.md wording fix and any `inv` arm are proposals for the orchestrator; I
  have not implemented them.
- **Follow-up for the orchestrator:** decide whether DEV pmrt cells on seeds 100-119 need a CRN / selection caveat
  in the protocol (affects every arm, pcorr_eq included).
