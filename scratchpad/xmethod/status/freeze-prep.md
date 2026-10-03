READY-TO-MERGE
# freeze-prep status (2026-10-04): freeze package for Study A (R-1..R-56)

Branch xm/freeze-prep from feat/v2 5a811de; local commits only, not pushed. No EVAL seed, no SEED_REGISTRY change,
no compute, no dependency. Package `scratchpad/xmethod/freeze/`: FREEZE_CHECKLIST.md (launch plan + gating + freeze
steps), FREEZE_NOTE.md (template), eval_projection.py / .json, merge_dev_runs.py.

## HAND-BACK
- PROTOCOL_A (DRAFT v5): every T-item filled except T1's VALUE (aud1's power calc). Main values:
  - T3: none infeasible (max 719 CPU-s, mscr_eq_min n 1000); R-55 calibration aligned with xm-citests exp_c_timing.py.
  - T4: pcorr, pcorr_hac(_fb), rcot2, mscr, cdl admitted; T5 report paths.
  - T6: VALUE 60, i.e. S = clip(S_power, 40, 60) (R-56). T7: freeze contents. T10: gbm. T11: mscr n <= 1000.
  - R-55: 1 process per vCPU, per-record host logging, Exp C timing run, V10 supplementary.
  - R-56: rcot2-type named-arm rule + pre-registered sentence, unfiltered eq table, C2b with / without R1-failing
    partners, C1 without mscr; s.0 discloses R-51..R-56.
  - Freeze step 4: the shas go in FREEZE_NOTE.md (commit messages are one line).
- Spec: pmrt_nl_eq gbm, cdl:CDLMethod (CPU), mscr blocks n 500 / 1000 + R-54 labels, mscr_native
  `c1_sensitivity_drop`, C3-reader E4 seeds [3100000, 3100299]; S 40 = 191 600 units, S 60 = 227 380.
- eval_analysis `xm-eval-analysis/5` (R-56):
  - V4 `C1.sensitivity_without`, `C2b.with_r1_failing_partners`, `eq_arms_table` (every eq arm, unfiltered),
    `C2_named` (+ C2 wording suffix); c1_arm adds `pooled_r1`, `r2_legs`.
  - T1: S_CAP 60, `achieved_power`, `min_power_at_S`. Markdown prints all of these.
- Launch plan: S 40 ~1 240 CPU-h / ~56 h wall; S 60 ~1 520 / ~69 h (4 Kaggle + 3 Colab CPU, 1 process per vCPU,
  no Lightning); cdl 46 %, pmrt_nl_eq 34 %.
- Merge recipe (F21): `merge_dev_runs.py` (dry run / --apply, lineage rule). Dry run vs this branch: TAKE 91, KEEP
  158, DROP 2 (`_ref/`), REVIEW 0. It never commits. Campaign tests use only eval_analysis' seed_row / to_result,
  which are unchanged; I did not run them here (low-memory rule).
- Tests (tests/test_xmethod_eval_analysis.py only): 39 pass + 1 skip (campaign is not on this branch); ruff clean.
  - New: R-56 named arm / eq table / sensitivities; default study; T6 cap + achieved power; mscr n <= 1000 + unit
    counts; T-register has only T1 open.

## Findings
- F19 The freeze is NOT green tonight: gated by checklist B1-B6.
  - The pmrt_nl_eq and cdl DEV runs (cdl holds every Kaggle / Colab slot).
  - aud1's T1.
  - The pmrt_nl_eq cost pilot at n 8000 / 24000.
  - The R-55 calibration block (xm-citests).
  - campaign R-55 logging, 1 process per vCPU, and a lane option (dev-runs).
- F20 cdl on Colab costs ~2x the cdl worker's table (n 4000: 258 vs 129 CPU-s); the projection uses the DEV records.
- F22 DEV preview (descriptive):
  - C1: 5 of 6 D arms FAIL; rcot2_native is INVALID IN R1.
  - C2b = pcorr_eq + granger_eq, both pass.
  - Under R-56, rcot2_eq is the expected named arm.
- F23 exp-c measured that f differs by arm (Lightning 1.65-2.85). R-55's pooled f with error e covers this through the
  rule that an arm within e of 7200 counts as over budget.

## Questions
- Q1 ANSWERED (R-56): keep the C2b rule; name rcot2_eq, add the unfiltered eq table and C2b with / without the
  R1-failing partners. Implemented.
- Q2 ANSWERED (R-56): mscr_native stays in D; C1 is also reported without mscr. Implemented.
- Q3 ANSWERED (R-56): T6 cap 60, S = clip(S_power, 40, 60), achieved power reported. Implemented; projection updated.
- Q4 ANSWERED (orchestrator):
  - anchors as proposed, at n 500 / 1000 / 4000;
  - 3 repeats on E2 R2 = DEV seeds 3_000_000-002, as in exp-c's plan;
  - f pooled per host type, e = max |f_anchor / f - 1|;
  - xm-citests builds and runs the block.
- none open.
