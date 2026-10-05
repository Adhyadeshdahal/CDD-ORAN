WORKING (X5 step 1 done; X6 relaunch running on Colab; X5 x5_fail running on the VPS, x5_adj next)

# diag status (2026-10-05 ~14:05Z): post hoc diagnostics X5 (advisor B) / X6 (advisor C), exploratory, R-60

## Branch xm/diag (from feat/v2 0fe4eb3; not pushed)
- 57cef93 declaration + seeds; d897cbe hooks / specs / tests; bc29a66 x5_adj amendment; 4348ab4 diag.py + step 1.
  xm/exp-b stays at 134529a (X2 / X3, READY-TO-MERGE). feat/v2 not merged into either.

## Orchestrator rulings (re-brief after compaction)
- Q1: the post-freeze `scratchpad/e6_dev/xm_dispatch.py` change is amendment A-1 (docs/xmethod/PROTOCOL_A_AMENDMENTS.md).
  `test_freeze_manifest_still_valid` fails only for that A-1 file; everything else passes.
- Q2: x5_fail after X7 (X7 was already running), then x5_adj. x5_adj was trimmed to 1000 seeds per cell
  (3300000-3300999) and declared before any X5 run (EXTRAS_PROTOCOL.md, X5 "Pre-run amendment"). X6 runs on Colab.
  No Kaggle.

## Declared (EXTRAS_PROTOCOL.md Amendments)
- X5:
  - Step 1 reads EVAL read-only. Step 2: x5_fail (2000 seeds per cell) and x5_adj (1000 seeds per cell).
  - Primary: one-sided exact binomial with Holm over 3 -> EXCESS / CHANCE / UNRESOLVED.
  - Step 3 = pmrt_r3 (McNemar) if EXCESS.
- X6: E1 R2 n1000, 200 seeds, 15 arms; 2x2 centring x redraw at x2; width grid .5-2; bias log; P1-P4.

## Done
- X5 step 1 (`diag.py x5-step1`, results/extras/diag/diag_x5_step1.json): reproduces .080 / .083 / .083.
  - pmrt_eq: 3 INVALID of 40 rates.
    - The declared approximation, which treats rates as independent, gives P(>= 3) = .007. It is liberal because
      lambdas share seeds.
    - The frozen dependence-aware F_max null (added after the declaration) gives F_max 2 and P(>= 3) = .037.
  - pmrt_r3: 1 INVALID. Its rejecting seeds look like pmrt_eq's.
  - The same seeds reject across lambdas. At n 24000, P_placebo's extreme tail is heavy at every lambda (rej .01
    o/e 2-3.3). Step 1 decides nothing.

## X6 session loss (2026-10-05)
- xm-diag-x6 (TPU host, 4 parts, code 3628f57) was job-lost at 14:44:47Z: "session gone before the exit code was
  pulled". All 4 parts were affected. The last pull was at 14:35Z.
- Kept (campaign merge rule): 1470 ok records, 0 broken lines, 96 complete datasets plus 4 partial ones, in
  results/extras/x6_gbm/session_a.jsonl.gz. The file is local only; it was untracked in b052a4f (disk / repo rule:
  only reports and tables in git). The raw shards were deleted.
- Relaunch of the missing units only, via `--skip-complete-from session_a` (R-35: every arm of the 4 partial
  datasets is re-run).
  - TPU v5e-1 assignment returned "Service Unavailable" twice (xm-diag-x6b, x6c; nothing was assigned).
  - So the relaunch runs on a plain Colab CPU runtime (`COLAB_ACCEL=cpu`, 2 vCPUs, 2 parts): xm-diag-x6d,
    launched 14:57:58Z, code 4348ab4. That code differs from 3628f57 only by the analysis module diag.py; the run
    code is unchanged.

## Running
- X6 relaunch xm-diag-x6d, Colab CPU runtime, 104 datasets left: GBM about 38 s per unit, ETA about 20:05Z; pulls to
  scratchpad/e6_dev/runs/xm-diag-x6d/. 15:58:22Z exec-error (pull "Unavailable"): transient, one of 3 allowed;
  the job was alive at 16:05Z.
- VPS: x5_fail, job xm-diag-x5f (scope cdd-xm-xm-diag-x5f, 7 procs, commit 70468c6, py 3.12.14, pins clean).
  Started 16:35:40Z after xm-citests freed the VPS (X7 exited 16:29Z). About 4.8 s per pmrt_eq unit; ETA 20:40Z.

## Next steps
1. When x5_fail exits: pull it, then `extras vps --spec .../x5_adj.json --name xm-diag-x5a --parts 7 --procs 7
   --cost-table .../diag_cost_agg.json` (about 4.3 h). A preflight refusal stops the lane. Hash provenance on Linux.
2. X6 done: merge session_a + x6d shards (local, untracked), then `diag x6`; then delete the shards.
3. X5 done: merge, then `diag x5 --fail ... --adj ...`, then `diag tables`. Merged records stay local, not in git.
4. Commit only results/extras/DIAG_REPORT.md + diag tables (JSON / md); then set line 1 to READY-TO-MERGE.
## Questions: none open
