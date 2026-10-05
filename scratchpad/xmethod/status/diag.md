WORKING (X6 done + reported; X5 primary done = "chance"; x5_adj running on the VPS, ETA ~01:00Z)

# diag status (2026-10-05 ~20:55Z): post hoc diagnostics X5 (advisor B) / X6 (advisor C), exploratory, R-60

## Branch xm/diag (from feat/v2 0fe4eb3; not pushed)
- 57cef93 declaration + seeds; d897cbe hooks / specs / tests; bc29a66 x5_adj amendment; 4348ab4 diag.py + step 1.
  xm/exp-b stays at 134529a (X2 / X3, READY-TO-MERGE). feat/v2 not merged into either.

## Orchestrator rulings
- Q1: the post-freeze `scratchpad/e6_dev/xm_dispatch.py` change = amendment A-1. `test_freeze_manifest_still_valid`
  fails only for that file (noted in DIAG_REPORT.md).
- Q2: x5_fail after X7, then x5_adj (trimmed to 1000 seeds, 3300000-3300999, declared pre-run). X6 on Colab. No Kaggle.
- Raw records stay local, outside git; delete intermediate shards after a verified merge.

## Report: scratchpad/xmethod/results/extras/DIAG_REPORT.md (X6 final; X5 primary final, x5_adj pending)
- Tables: results/extras/diag/{diag_x5_step1,diag_x6,diag_tables}.json + DIAG_TABLES.md.

## X6: DONE. Declared outcome PARTIAL (failed P4, sub-prediction P4c)
- Session loss: xm-diag-x6 (TPU, 4 parts) job-lost 14:44:47Z. Kept 1470 ok records (96 complete + 4 partial
  datasets). TPU relaunch "Service Unavailable" x2, so missing units only on Colab CPU: xm-diag-x6d (2 parts, 4348ab4;
  run code = 3628f57). It exited 0 at 20:28:30Z with 1560 records and clean pins.
- Merge: 3000/3000 ok (= spec: 200 seeds x 15 arms), 0 missing / errors / conflicts, 30 duplicates (the re-run
  partial datasets; later wins). 200/200 dataset hashes = frozen. Shards + session_a deleted after the merge;
  merged.jsonl.gz is local only.
- Routine fix (logged): campaign `merge` does not read .gz inputs (6406 bad lines on the first try). I decompressed
  session_a to a temp file and re-merged; the bad merge was deleted.
- TT .079 [.068, .090] INVALID; tT .038 [.031, .047] VALID; Tt .046, tt .045. P1, P2, P3, P4a, P4b hold.
- P4c fails: |tT z_bias| .448 = 5x TT's .090. Centring is the cause; the bias term does not measure its size.

## X5
- Step 1 done (diag_x5_step1.json): 3 of 40 INVALID. Declared independent P(>= 3) .007; dependence-aware .037
  (not declared, labelled so).
- x5_fail DONE: xm-diag-x5f exited 0 at ~20:40Z; 12000/12000 ok, 0 conflicts; merged locally, shards deleted.
  - Primary: .052 [.043, .063], .054 [.045, .065], .051 [.042, .062]; Holm p .657. All CHANCE, so the answer to B is
    "chance". Step 3 not triggered (McNemar p .34 / .26 / 1.0, reported).
- Routine fix (logged): in diag.py, the primary rows' per-kind count overwrote the cell `n`, so step 3 found no pairs.
  The count was renamed to n_; the tests pass and the rates were unaffected.
- Provenance: the hash check run on Windows differs from the VPS records (platform numerics; X6 E1 matched on
  Windows). It must run on Linux, i.e. on the VPS after x5_adj (one cdd-xm job at a time). Pending.
- x5_adj RUNNING: xm-diag-x5a, VPS, 7 procs, commit 264d351, started 20:42:23Z, pins ok, ETA ~01:00Z.

## Next steps
1. x5_adj exits: pull; merge locally; delete shards.
2. On the VPS (/opt/cdd-xm, systemd scope): `diag x5 --fail ... --adj ...` for Linux provenance; else run it locally
   and do provenance only on the VPS.
3. `diag tables`; finish the X5 section of DIAG_REPORT.md; commit the report + tables; line 1 -> READY-TO-MERGE.

## Questions: none open
