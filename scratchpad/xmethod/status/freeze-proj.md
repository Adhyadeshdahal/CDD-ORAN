READY-TO-MERGE
# freeze-proj status (2026-10-04): EVAL projection with the measured pmrt_nl_eq and cdl costs

Branch xm/freeze-proj2 from feat/v2 1d33831; one local commit, not pushed. No EVAL seed, no compute, no dependency.

## HAND-BACK
- `freeze/eval_projection.py`, new options:
  - `--extra-agg` (more DEV aggregates, e.g. pmrt_nl/agg.json);
  - `--cdl-kaggle` (per-n CPU-s table);
  - `--proxy ARM=PROXY`, which shapes ARM's unmeasured cells by PROXY's per-cell / per-n cost ratio.
  Also new: a per-n fill (mean of measured cells, power law for unmeasured n) and a `wall_h_kaggle_only` column.
- Inputs:
  - pmrt_nl_eq from results/dev/pmrt_nl: mean / max 35 / 64, 54 / 98, 145 / 290 at n 500 / 1000 / 4000; n 8000 /
    24000 extrapolated n^0.68; E4 R3 / R4 cells scaled like pmrt_eq's.
  - cdl: Kaggle 17 / 34 / 121 / 232 / 487 CPU-s at n 500..24000.
- Result (full slots, 1 process per vCPU):

  | S | CPU-h | wall, Kaggle 4 + Colab 3 | wall, Kaggle 4 only |
  |---|---|---|---|
  | 40 | ~1 030 | ~47 h | ~64 h |
  | 50 | ~1 130 | ~51 h | ~70 h |
  | 60 | ~1 220 | ~56 h | ~77 h |

  - Setup and re-runs add ~10-15 %: Kaggle only, S 40 is ~3 days and S 60 ~3.5 days.
  - Unscaled upper bound (E4 at the per-n mean): S 40 1 356 CPU-h, 62 h / 85 h; S 60 1 559 CPU-h, 71 h / 97 h.
  - pmrt_nl_eq is now the largest cost (513 CPU-h, 50 % at S 40); cdl 271 (26 %).
- FREEZE_CHECKLIST section A rewritten:
  - the new table with a Kaggle-only column;
  - Colab may be unavailable;
  - cost basis and the command;
  - parts ~410 (S 40) / ~490 (S 60).
  B4: pmrt_nl_eq cost pilot estimate ~1 000 CPU-s at n 24000 (DEV max 290 at n 4000 x 6^0.68).
- PROTOCOL_A:
  - T3 maxima: pmrt_nl_eq 290 at n 4000 (DEV run), cdl Kaggle means up to 487.
  - T6 projection numbers updated; VALUE 60 unchanged (R-56).
  - Line 12 rewrapped (whitespace only).
- Tests (eval_analysis + eval_report): 46 pass + 1 skip; ruff clean.

- R-57 (second commit):
  - PROTOCOL_A: EVAL interpreter exactly Python 3.12.14 everywhere (s.7, s.11 stamp), and the VPS as an EVAL
    platform (7 processes, host type 'vps' + CPU model, in the R-55 calibration block, T3). T6 walls include the
    VPS; the T3 paragraph is reflowed.
  - Launch plan: 29 processes with Kaggle + Colab + VPS; S 40 35 h, Kaggle + VPS 45 h, Kaggle only 64 h; S 60 42 /
    53 / 77 h. Gating B5 / B6 add the VPS calibration, EVAL_PYTHON and a VPS dispatcher target.
  - Projection `--vps-procs`.

## Findings
- F1 The E4 scaling is a modelling choice. pmrt_eq costs about the same in E4 R3 as in E1 / E3, but ~1/10 in E4 R4,
  where only the placebo has a design. The unscaled column bounds the error.
- F2 The DEV costs of the 10 classic and CI arms came from oversubscribed sessions (an upper estimate). pmrt_nl_eq's
  ran at 4 processes on 4 cores (CPU-s = wall-s).

## Questions
- none.
