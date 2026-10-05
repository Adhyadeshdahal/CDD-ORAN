READY-TO-MERGE (report): all 7 specs run + merged; analyse done; results/extras/EXTRAS_REPORT.md (s.6 wording)

# extras status (2026-10-05): R-60 X2 + X3 declared feat/v2 86a9767; run, merged, analysed

## Result
- `scratchpad/xmethod/results/extras/`:
  - `EXTRAS_REPORT.md`: the EXTRAS_PROTOCOL s.6 sentences, filled from `extras_tables.json`, plus checks, notes and
    cost;
  - `EXTRAS_TABLES.md` / `extras_tables.json`: every cell, the pooled rates, recall and R-30 labels;
  - `<spec>/merged.jsonl.gz`;
  - `provenance_platform.json`, `analyse_vps.log`.
- Units: 18,240 / 18,240 ok (0 missing, error or infeasible). Cells: 364, all ok. Pins match; py 3.12.14.
- Provenance (run on Linux): x2_d10 and x3_told 360 + 180 datasets equal the frozen generator; the other 5 X2
  designs 360 / 360 differ.
- Cost: 63.9 Kaggle-ref CPU-h (VPS CPU-s converted with results/exp_c/calib/factors.json). DEV projection was 77.2.
- NA entries come from the frozen pipeline; reasons are in the report notes:
  - pcorr_eq with 5 blocks: the frozen pcorr arm STOPs (rank-deficient source design);
  - E4 R3 truth-null: empty denominator.
- Windows caveat: the frozen generator is not bit-identical to Linux on Windows for E2, E4 and E1 seed 3200028.
  This affects re-checks only; all records were generated on Linux.
- No EVAL output was opened. No frozen file was edited.

## Build (fee5171 = feat/v2 86a9767)
- Declaration: `scratchpad/xmethod/EXTRAS_PROTOCOL.md` (s.1-7; Amendments: none).
- `docs/benchmark/SEED_REGISTRY.json`: XMETHOD_EXTRAS 3200000-3200199.
  - X2 tune 3200000-019; X2 measure -059.
  - X3 measure 3200060-119.
- `cdd_oran/xmethod/extras.py`: in-process hooks around the frozen campaign, restored on exit.
  - X2: DITHER_DELTA / DITHER_BLOCKS at generation.
  - X3: told design (width / switch shift / E4 R3 lambda).
  - Also: seed guard, `integrity.extras` stamp, same-spec merge rule, and the specs / projection / analyse commands.
- Specs: `scratchpad/xmethod/specs/extras/` (x2_d02 / d05 / d10 = the frozen design / d20 / b05 / b80, x3_told).
- Tests: `tests/test_xmethod_extras.py`, incl. the bit-for-bit reproduction cell. Windows 15 + 1 skip; Linux 16.
- Freeze manifest valid (301 files, LF rule).

## Launch log (UTC, 10-05)
- 02:20 EVAL all-launched; 03:35 final EVAL v-job pulled; VPS idle.
- 03:41 x3_told launched on VPS (7 / 7, c41df44); 05:13 done exit 0; merged 4560 ok.
- 04:16 x2_d10 launched on Kaggle (4 parts, aeb98d4; X1 listed; total 4); complete (wall 9094 s); merged 2280 ok.
- 05:13 x2_d02 launched on Kaggle (total 4); 07:29 complete (wall 7505 s); merged 2280 ok.
- 05:33 x2_d05 launched on VPS (orchestrator: remaining X2 on VPS too; 34a7165); 06:04 done; merged 2280 ok.
- 06:06 x2_d20 launched on VPS (99fdb05); 06:36 done; merged 2280 ok.
- 06:11 x2_b05 launched on Kaggle (280d5e6; total 4); 08:46 complete (wall 8504 s); merged 2280 ok.
- 06:37 x2_b80 VPS launch REFUSED by preflight (others' load 1.45 > 1; nothing started). VPS lane stopped and
  reported.
- 06:57 x2_b80 launched on Kaggle (ff593b1; total 3); 09:17 complete (wall 8299 s); merged 2280 ok. All 7 merged.
- 07:15 d10 / X3 provenance pre-check: 540 / 540 equal on Linux, but they differ on Windows.
  - Result: `provenance_platform.json`.
  - So analyse runs on Linux.
- 09:20 analyse on VPS (scope, /opt/cdd-xm): 2 attempts stopped at start because the reused bundle lacked
  eval_analysis.py, then the specs (no output). The 3rd attempt, with both uploaded from HEAD: rc 0, 09:22.

## Questions: all ANSWERED
- (none open)
