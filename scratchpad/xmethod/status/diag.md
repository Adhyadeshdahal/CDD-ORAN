WORKING (X5 step 1 done; X6 session lost, missing units relaunched on Colab; X5 step 2 queued after X7)

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
- Kept (campaign merge rule): 1470 ok records, 0 broken lines, 96 complete datasets plus 4 partial ones. They are in
  results/extras/x6_gbm/session_a.jsonl.gz (+ .summary.json).
- Relaunch of the missing units only, via `--skip-complete-from session_a` (R-35: every arm of the 4 partial
  datasets is re-run).
  - TPU v5e-1 assignment returned "Service Unavailable" twice (xm-diag-x6b, x6c; nothing was assigned).
  - So the relaunch runs on a plain Colab CPU runtime (`COLAB_ACCEL=cpu`, 2 vCPUs, 2 parts): xm-diag-x6d,
    launched 14:57:58Z, code 4348ab4. That code differs from 3628f57 only by the analysis module diag.py; the run
    code is unchanged.

## Running
- X6 relaunch xm-diag-x6d, Colab CPU runtime, 104 datasets left: GBM about 38 s per unit, ETA about 20:05Z; pulls to
  scratchpad/e6_dev/runs/xm-diag-x6d/.
- VPS: xm-citests' X7 (xm-x7-v1, about 14:1xZ, ETA about 2.5-3 h). xm-citests will message when it has exited and
  been pulled.

## Next steps
1. When X7 is out: `uv run python -m cdd_oran.xmethod.extras vps --spec scratchpad/xmethod/specs/extras/x5_fail.json
   --parts 7 --procs 7 --cost-table scratchpad/xmethod/specs/extras/diag_cost_agg.json` (about 4 h), then the same
   for x5_adj (about 4.3 h). A preflight refusal stops the lane.
2. X6 done: merge session_a + x6d shards into results/extras/x6_gbm/merged.jsonl.gz, then `diag x6 --merged ... --out
   scratchpad/xmethod/results/extras/diag`.
3. X5 done: merge, then `diag x5 --fail ... --adj ...`, then `diag tables`.
4. Write results/extras/DIAG_REPORT.md (+ diag_tables.json), commit, and set line 1 to READY-TO-MERGE.

## Questions: none open
