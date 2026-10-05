WORKING (X5 step 1 done; X6 running on Colab; X5 step 2 queued on the VPS after xm-citests' X7)

# diag status (2026-10-05 ~14:05Z): post hoc diagnostics X5 (advisor B) / X6 (advisor C), exploratory, R-60

## Branch xm/diag (from feat/v2 0fe4eb3; not pushed)
- 57cef93 declaration + XMETHOD_DIAG seeds; d897cbe hooks / specs / tests; bc29a66 x5_adj pre-run amendment.
- xm/exp-b stays at 134529a (X2 / X3, READY-TO-MERGE). feat/v2 not merged into either.

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
- X5 step 1 (`cdd_oran/xmethod/diag.py x5-step1`): results/extras/diag/diag_x5_step1.json.
  - Reproduces the frozen rates (.080 / .083 / .083).
  - pmrt_eq has 3 INVALID among 40 rates. The declared independent approximation gives P(>= 3) = .007, but lambdas
    share seeds, so that approximation is liberal.
  - The frozen dependence-aware F_max null (added after the declaration, labelled as such) gives F_max 2 and
    P(>= 3) = .037. pmrt_r3: 1 INVALID, P(>= 1) = .26.
  - Concentration:
    - The same seeds reject across lambdas at a given n (lambdas share streams).
    - P_placebo and P_placebo_conf overlap is at chance.
    - At n 24000, P_placebo has a heavier extreme tail across all lambdas (rej .01 o/e 2-3.3).
    - pmrt_eq and pmrt_r3 look alike.
  - Step 1 decides nothing; it is read with step 2.

## Running
- X6 on Colab, job xm-diag-x6 (session launched 13:54Z, 4 parts, py 3.12.14, pin_check clean, code 3628f57).
  - About 11 s per GBM unit, so ETA about 15:25Z.
  - Task Scheduler ticks (CDD-ORAN-colab-xm-diag-x6, every 10 min) pull to scratchpad/e6_dev/runs/xm-diag-x6/.
  - The final tick stops the runtime.
- VPS: xm-citests' X7 (xm-x7-v1, about 14:1xZ, ETA about 2.5-3 h). xm-citests will message when it has exited and
  been pulled.

## Next steps
1. When X7 is out: `uv run python -m cdd_oran.xmethod.extras vps --spec scratchpad/xmethod/specs/extras/x5_fail.json
   --parts 7 --procs 7 --cost-table scratchpad/xmethod/specs/extras/diag_cost_agg.json` (about 4 h), then the same
   for x5_adj (about 4.3 h). A preflight refusal stops the lane.
2. X6 done: merge into results/extras/x6_gbm/merged.jsonl.gz, then `diag x6 --merged ... --out
   scratchpad/xmethod/results/extras/diag`.
3. X5 done: merge, then `diag x5 --fail ... --adj ...`, then `diag tables`.
4. Write results/extras/DIAG_REPORT.md (+ diag_tables.json), commit, and set line 1 to READY-TO-MERGE.

## Questions
- none open
