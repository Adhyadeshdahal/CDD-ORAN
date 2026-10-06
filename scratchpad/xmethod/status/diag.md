READY-TO-MERGE

# diag status (2026-10-06 ~01:30Z): post hoc diagnostics X5 (advisor B) / X6 (advisor C), exploratory, R-60

## Branch xm/diag (from feat/v2 0fe4eb3; not pushed)
- 57cef93 declaration + seeds; d897cbe hooks / specs / tests; bc29a66 x5_adj amendment; 4348ab4 diag.py + step 1;
  7ba3de3 X6 report; last commit = X5 final + READY-TO-MERGE. xm/exp-b stays at 134529a. feat/v2 not merged.

## Deliverable: scratchpad/xmethod/results/extras/DIAG_REPORT.md
- Tables: results/extras/diag/{diag_x5_step1,diag_x5,diag_x6,diag_tables}.json + DIAG_TABLES.md.
- For each of X5 / X6: what was run, numbers with CIs, the declared outcome, one Discussion sentence + confidence.
- Raw merged records (x5_fail, x5_adj, x6_gbm merged.jsonl.gz) are local only, not in git; all shards are deleted.

## Orchestrator rulings
- Q1: `scratchpad/e6_dev/xm_dispatch.py` = amendment A-1. `test_freeze_manifest_still_valid` fails only for that
  file (noted in the report). The frozen file and the manifest were not edited.
- Q2: x5_fail after X7, then x5_adj (trimmed to 1000 seeds, declared pre-run). X6 on Colab. No Kaggle.

## X6: declared outcome PARTIAL (failed P4, sub-prediction P4c)
- TT .079 [.068, .090] INVALID; true centring (tT) .038 [.031, .047] VALID; Tt .046, tt .045. P1-P3, P4a, P4b hold.
- P4c fails: |tT z_bias| .448 = 5x TT's .090. Centring is the cause; the first-order bias term does not measure its
  size. Confidence: high (centring), low (bias mechanism).
- Session loss: xm-diag-x6 (TPU) job-lost 14:44:47Z; kept 1470 records. TPU "Service Unavailable" x2, so the missing
  units were relaunched on Colab CPU (xm-diag-x6d, exit 0 at 20:28:30Z). Merge 3000/3000 ok = spec; 200/200 hashes
  equal.

## X5: answer to B = "chance" (all 3 primary pairs CHANCE)
- Primary (2000 fresh datasets each): .052 [.043, .063], .054 [.045, .065], .051 [.042, .062]; Holm p .657.
- Step 3 not triggered (McNemar p .34 / .26 / 1.0, reported). Adjacent cells: 20/20 VALID (.045-.058).
- Step 1: 3/40 INVALID. Declared independent P(>= 3) .007; dependence-aware .037 (not declared, labelled so).
- Provenance on Linux (VPS job xm-diag-x5q): 80/80 hashes equal; the Windows check differs (platform numerics).
- The EVAL extreme tail at n 24000 did not recur on fresh data (not declared, descriptive).
- Confidence: high (no excess > .015), moderate-high ("chance").

## Routine decisions (taken overnight, logged)
- Campaign `merge` does not read .gz inputs: session_a was decompressed to a temp file for the X6 merge.
- diag.py fix (7ba3de3): the primary rows' per-kind count overwrote the cell `n`, so step 3 found no pairs. The count
  was renamed to n_; the rates were unaffected and the tests pass.
- Linux provenance as a 1-process VPS job after x5_adj (one cdd-xm job at a time). The first try, xm-diag-x5p,
  failed: eval_analysis.py was not bundled, and my trailing `exit` meant no exit_code was written. It reran as
  xm-diag-x5q (exit 0). The merged copies on the VPS were deleted; the raw shards stay in /opt/cdd-xm/jobs.
- The VPS is free (no active cdd-xm job).

## Questions: none open
