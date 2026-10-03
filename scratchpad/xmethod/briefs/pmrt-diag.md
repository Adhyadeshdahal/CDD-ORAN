# Brief: pmrt-diag (why is pmrt_eq liberal in E2 R2 at n 1000?)

Finding (dev-runs DEV full run, results on branch xm/dev-runs, worktree D:/academia/major-project/CDD-ORAN-wt/xm-pmrt,
scratchpad/xmethod/results/dev/): pmrt_eq truth-null raw-p rate (p <= .05) in E2 R2 n 1000: kappa .125 .070
[.053,.087], .25 .061 [.050,.074], .5 .084 [.062,.105]; n 500 .042, n 4000 .049; pmrt_r3 similar. P_placebo fine.
Diffuse over 32 null edges; sources P6 / P7 highest (.078 / .090). The estimand check (R-36) says these edges have
exactly zero lag-1 total effect, so a correct design-based CRT should be exact (up to Monte Carlo B). Something in
pmrt_core breaks exactness.

Read: CONTRACT.md (R-8, R-9, R-14, R-17, R-25, R-29..R-41), docs/xmethod/PMRT_CORE.md (esp. "Why no clip"),
cdd_oran/xmethod/methods/pmrt_core.py, covariates.py, worlds/ (E2, R2 design: dither law, setpoints, clipping).

Hypotheses to test (each with a decisive experiment; report all, not just the first that fits):
H1 covariates that are functions of the redrawn dither (own action at t-1 / t-2, or KPI lags affected by own past
   dither) are held fixed while the current dither is redrawn: the redraw distribution is then not the conditional
   law given the covariates (non-invariance; same mechanism as the R-14 clip finding).
H2 the redraw law differs from the realised dither law (clipping at action bounds, discretisation, the setpoint +
   dither sum being clipped, rounding) -> randomization distribution misspecified.
H3 overlapping/serially dependent rows (row t's target is row t+1's lag) with redraws that are independent per row
   while the realised dither enters later rows' covariates.
H4 concurrent designed actions (R-8) or setpoint columns create dependence between the statistic and the redraw.
H5 Monte Carlo / sequential B (Besag-Clifford h 20) p-value bias.
H6 chance (multiplicity over cells): check with fresh DEV reserve seeds 3_000_160-189 only.

Method: controlled experiments that vary one factor (drop own-lag covariates; recompute covariates consistently from
the redrawn column; unclipped dither; fixed B 9999 vs sequential; iid R1 analogue with same covariates). Use E2 R2 n
1000 kappa .5 and a small synthetic analogue. DEV seeds only (3_000_160-189 reserve for confirmation); never EVAL.
Deliverable: scratchpad/xmethod/results/pmrt_diag/REPORT.md (< 100 lines): which hypotheses hold, with numbers and
CIs; a proposed fix justified by THEORY (exactness / invariance argument), not by matching rates; its power cost on
the same seeds; and whether the same flaw can affect other eq arms (they use the same helper covariates).
Do NOT change pmrt_core on feat/v2; prototype on branch xm/pmrt-diag. Local CPU one process, or Kaggle (max 5
sessions shared; dev-runs holds 1-2). Status scratchpad/xmethod/status/pmrt-diag.md (READY-TO-MERGE line 1 when done,
questions "- Q<n> ..."). Commit locally, never push.
