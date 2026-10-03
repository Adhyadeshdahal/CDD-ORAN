# Brief: pmrt-nl (nonlinear PMRT statistic, ruling R-42)

Read CONTRACT.md (all, esp. R-8, R-9, R-14, R-17, R-25, R-29..R-42), docs/xmethod/PMRT_CORE.md,
cdd_oran/xmethod/methods/pmrt_core.py, covariates.py, worlds/ (E2 Gaussian-bump KPIs, E5 gated / harmonic), the DEV
results (branch xm/dev-runs; worktree D:/academia/major-project/CDD-ORAN-wt/xm-pmrt,
scratchpad/xmethod/results/dev/full/merged.jsonl.gz: per-edge p / scores of every arm).

Step 1 diagnosis: per true edge in E2 / E5 (R1, R2, n 1000, kappa .25): pmrt_eq raw-p recall vs shap_dag / two_tower /
pc_eq recall; effect shape from the SCM (monotone? bump? state-gated? interaction with which state); which edges
PMRT misses. Write results/pmrt_nl/DIAG.md (< 60 lines).
Step 2 candidates (2-3), each computed only from the redrawn random part R (dither / iid draw / logged-policy
redraw) and quantities fixed under the redraw (past-only covariates, residual of the past-only ridge adjustment):
e.g. (a) kernel / HSIC or distance covariance between R and the residual, optionally given state; (b) a basis-
expansion statistic (splines of R x state interactions, max or sum of squared scores); (c) held-out boosted-model
gain of adding R (cross-fit, fixed folds, fixed seed). Keep the PMRT redraw machinery, B 9999 Besag-Clifford (R-9),
BY (R-2); no Huber clip (R-14). Exactness argument written down per candidate (why the statistic's null law is
the redraw law).
Step 3 gates: synthetic F4 (null level n 500/1000/4000, iid + R2-like design + nonlinear null) ; DEV truth-null and
placebo rates (R-30 not INVALID); cost at n 4000 vs 2 CPU-h; recall per R-42's rule. DEV seeds 3_000_100-159 only;
NOT 3_000_160-189 (reserved). Never EVAL.
Step 4 apply the R-42 selection rule mechanically; implement the winner as a pmrt_core statistic option (new
version string), linear stays the default for the secondary arm; tests; docs/xmethod/PMRT_CORE.md section.
Coordinate: worker pmrt-diag (calib, branch xm/pmrt-diag) is fixing a small E2 R2 inflation in pmrt_core; do not
edit the same functions without reading its status (scratchpad/xmethod/status/pmrt-diag.md in worktree
D:/academia/major-project/CDD-ORAN-wt/xm-classic); build your statistic as a separate module plugged into pmrt_core.
Branch xm/pmrt-nl from feat/v2. Status scratchpad/xmethod/status/pmrt-nl.md (READY-TO-MERGE line 1; questions
"- Q<n> ..."). Heavy runs on Kaggle (max 5 sessions shared; check first) or Colab; local one process. Commit locally.
