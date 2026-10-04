READY-TO-MERGE
# vps-factors status (2026-10-04): R-55 host factors in the EVAL projection

Branch xm/vps-factors from feat/v2 32d8f0e. Local only, not pushed. No EVAL seed, no compute. Input: xm-citests
message (sent at the orchestrator's request), factors.json on xm/exp-c aee2fab (LF sha256 f86cb61c...18ce),
Python 3.12.14, cal-v2 (51 paired units, Kaggle ref xm-expc-cal-k1 vs VPS xm-expc-cal-v2).

## HAND-BACK
- `freeze/eval_projection.py --factors`: costs stay Kaggle-reference CPU-s. A VPS / Colab process counts as
  f Kaggle processes for the workload mix (per-arm factor, else the pooled '*'; work-weighted harmonic). Wall columns
  use f_lo (conservative); `*_f` columns use the central f. Without `--factors`, the old numbers are reproduced exactly.
- Effective factors at S 40: VPS 2.09 (f_lo) / 2.35 (f); Colab 0.92 / 1.01.

  | S | CPU-h | Kaggle + Colab + VPS | Kaggle + VPS | Kaggle only | (old: all / K + VPS) |
  |---|---|---|---|---|---|
  | 40 | 509 | 14.1 h | 16.6 h | 31.8 h | 17.6 / 22.1 |
  | 50 | 564 | 15.6 h | 18.4 h | 35.2 h | 19.4 / 24.5 |
  | 60 | 618 | 17.1 h | 20.2 h | 38.6 h | 21.3 / 26.9 |

  With the central f: S 40 13.2 / 15.7 h, S 60 16.0 / 19.0 h.
- FREEZE_CHECKLIST A:
  - New table, the VPS calibration (3.12.14 only; 3.12.13 factors apply to Exp B parts 8-15 only).
  - T3 on VPS records = cost x f_hi vs 7200 (cdl 2 870 VPS CPU-s). The VPS does ~48 % of the Kaggle + VPS work.
  - B5: Colab and the VPS calibrated; factors.json goes into the freeze commit.
- PROTOCOL_A T6: projection numbers updated (~14 / 17 / 32 h at S 40; ~17 / 20 / 39 h at S 60).
- No test covers the projection script (none before either); ruff clean.

## Questions
- Q1 Should factors.json (xm-citests, xm/exp-c) be listed in T7 / FREEZE_NOTE as a freeze input? T3 reads it. I
  have noted it in checklist B5 but have not edited T7.
