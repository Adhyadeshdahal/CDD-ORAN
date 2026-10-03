# Brief: dev-runs (DEV validity, power and cost runs for Study A; feeds the protocol draft)

Read first: scratchpad/xmethod/CONTRACT.md (all, esp. sections 5-7 and R-2, R-9, R-12, R-13, R-17..R-27),
cdd_oran/xmethod/{api,runner,score,covariates}.py, worlds/generate.py, PROTOCOL_NOTES.md.

Goal: one shardable DEV campaign that produces, per (method, arm, world, regime, n, kappa):
(a) validity: truth-null primary edge rejection rate and P_placebo (and P_placebo_conf in R3/R4) declaration rate,
    with seed-cluster CIs (R-20, R-21);
(b) power: recall of true primary edges, FDP, sign accuracy, per seed (needed for the R-12 power calculation);
(c) cost: CPU-s and peak RAM per run (R-13 budget, Experiment C).

Methods: everything the runner can load from feat/v2: pmrt_core (covariates 'eq' and 'r3' = the R-19 2x2),
classic (eq + native where the method has an arm), and the five CI tests (mscr, pcorr, pdcor, rcot2, cmi_knn) once
xm-citests is merged (the orchestrator will tell you; build so they plug in via runner specs only).
Grid: worlds E1-E5 x their regimes (E4: R1-R4, lam per CONTRACT), n {500, 1000, 4000} (+8000 / 24000 for methods
that scale), kappa .25 primary (R-27), sweep {.125, .5} only for R2 cells and n 1000 (validity + power).
Seeds: DEV block 3_000_000-3_000_199 only. Keep tuning seeds (tune() / tau) disjoint from measurement seeds;
propose the split. NEVER touch 3_100_000+ (EVAL).
Budget (R-13): measure cost first; any (method, dataset) over 2 CPU-h is marked "infeasible at this n (measured
cost X)", never silently dropped.

Steps
1. Driver + spec files (one shard = (world, regime, n, seed, method, arm, kappa); resumable; merge tolerant of
   missing / duplicate shards; records carry code commit and versions). Tests for the merge/aggregate code.
2. Pilot: 2 measurement seeds per cell, all methods -> cost table + projected total CPU-h per platform.
   STOP and put the projection + platform plan (R-26: Kaggle long/unattended, Colab medium + GPU, Lightning
   reserve) in your status as a question before launching the full campaign.
3. Full DEV run (20 measurement seeds per cell unless the pilot says otherwise; state why).
4. Power calculation (R-12): from per-seed paired recall differences between methods (same seeds), the number of
   EVAL seeds per cell to detect a recall gap of .15 at alpha .05 two-sided, power .8, paired; report per cell and
   the max over cells used for all. Also seeds needed so a null-rate CI half-width is <= .02 at .05.
5. Report scratchpad/xmethod/results/dev/REPORT.md (< 150 lines): validity table (flag rates significantly above
   .05), recall table, cost table with infeasible cells, the seed-count proposal. Descriptive only: nothing is tuned
   on these numbers.

Branch xm/dev-runs in your worktree from feat/v2. Status scratchpad/xmethod/status/dev-runs.md (line 1
"READY-TO-MERGE" only at the end; questions as "- Q<n> ..."). Commit locally, never push.
