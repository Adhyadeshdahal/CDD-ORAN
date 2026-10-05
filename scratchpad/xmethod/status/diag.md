WORKING (X5 / X6 declared; building)

# diag status (2026-10-05): post hoc diagnostics X5 (advisor B) and X6 (advisor C), exploratory under R-60

## Plan
- Declaration: `scratchpad/xmethod/EXTRAS_PROTOCOL.md` Amendments (X5, X6), committed before any analysis or run.
- Seeds: XMETHOD_DIAG in `docs/benchmark/SEED_REGISTRY.json`.
  - E4 3300000-3301999 (X5).
  - E1 3302000-3302199 (X6).
  - Not used: 3200120-199.
- X5 (VPS):
  - step 1: EVAL records, read-only;
  - step 2: x5_fail (3 failing cells) then x5_adj (5 adjacent);
  - step 3: = the pmrt_r3 arm on the same datasets (E4 R3 has no setpoints).
- X6 (Colab): E1 R2 n 1000, 200 seeds; 2x2 centring x redraw at told x2, plus the told-width grid; bias log.

## Cost projection
- X5: 87.5 VPS CPU-h (about 172 Kaggle-ref); x5_fail about 4 h wall, x5_adj about 8.5 h (7 procs).
- X6: 6.5 VPS CPU-h-equivalent (about 15 Kaggle-ref); Colab 4 procs about 2 h.

## Log (UTC)
- 13:30 Declared; VPS busy with xm-citests X4 (xm-x4-v1, nearly done). X5 queues after X4 unless X7 is already
  waiting.

## Questions
- (none)
