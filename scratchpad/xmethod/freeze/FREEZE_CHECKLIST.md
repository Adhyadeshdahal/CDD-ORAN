# Study A freeze package: EVAL launch plan and freeze checklist (freeze-prep, 2026-10-04)

Protocol `docs/xmethod/PROTOCOL_A.md` (DRAFT v5, rulings R-1..R-56); spec `scratchpad/xmethod/specs/eval/full.json`;
projection `scratchpad/xmethod/freeze/eval_projection.json` (script `eval_projection.py` in the same directory).
Nothing here touches an EVAL seed; EVAL needs the freeze (section C) first.

## A. EVAL launch plan

Method set (25 arms): pmrt_nl_eq (primary, gbm, R-52), pmrt_eq, pmrt_r3, pc_eq / _native, granger_eq / _native,
corr, notears, shap_dag, two_tower, cdl (R-53), mscr x3 (n 500 / 1000 only, R-54), pcorr x3, pcorr_hac / _fb (+ eq_min),
rcot2 x3. Dropped: pdcor (R-48), cmi_knn (R-49 revised). No GPU arm.

Units and cost (DEV costs; tune = DEV tune seeds re-run, R-34; E4 R3 / R4 C3 readers at S_E4 300, T9). T6 cap 60
(R-56): S = clip(S_power, 40, 60), so only the first three rows can occur; S 70-100 are shown for reference only.

| S | units | datasets | tune units | CPU-h | wall h (all slots) | Kaggle process-h | Colab process-h |
|---|---|---|---|---|---|---|---|
| 40 | 191 600 | 16 200 | 43 240 | 1 243 | 56.5 | 904 | 339 |
| 50 | 209 490 | 16 750 | 43 240 | 1 381 | 62.8 | 1 004 | 377 |
| 60 | 227 380 | 17 300 | 43 240 | 1 519 | 69.1 | 1 105 | 414 |
| 70 | 245 270 | 17 850 | 43 240 | 1 657 | 75.3 | 1 205 | 452 |
| 80 | 263 160 | 18 400 | 43 240 | 1 796 | 81.6 | 1 306 | 490 |
| 90 | 281 050 | 18 950 | 43 240 | 1 934 | 87.9 | 1 406 | 527 |
| 100 | 298 940 | 19 500 | 43 240 | 2 072 | 94.2 | 1 507 | 565 |

- By arm at S 40: cdl 574 CPU-h (46 %), pmrt_nl_eq 426 (34 %; 12 000 E4 R3 / R4 units at S_E4 300), mscr_eq_min 52,
  mscr_eq 38, two_tower 33, shap_dag 29, pmrt_eq / pmrt_r3 24 each, the rest < 10 each. By n: 500 43, 1000 134,
  4000 155, 8000 287, 24000 617 CPU-h; generation 7 CPU-h. Tune 312 CPU-h.
- Cost basis (per (arm, world, regime, n) DEV mean): 10 arms results/dev/full (Kaggle); CI arms results/dev/ci_c
  (Kaggle / Colab / Lightning); cdl: its DEV run so far (504 Colab units, n <= 4000; n 8000 / 24000 scaled by
  training steps); pmrt_nl_eq: the R-42 gbm validity run (n <= 4000; n 8000 / 24000 extrapolated n^0.73). The DEV
  CPU-s came from oversubscribed sessions (8 / 4 processes on 4 / 2 vCPU): an upper estimate. Re-run the script
  when the cdl and pmrt_nl_eq DEV runs are merged.
- Platforms (user, tonight): Kaggle at most 4 of the 5 account sessions (one stays for other workers), Colab CPU at
  most 3 jobs, NO Lightning. R-55: 1 process per vCPU (Kaggle 4, Colab 2), so 22 concurrent processes; wall =
  process-h / 22, assuming full slots. Add setup (~10-15 min per session) and Colab reclaims: S 40 ~2.5-3 days.
- Parts: whole-dataset groups, LPT on the cost table (`campaign.partition`, `--cost-table` from the projection
  basis). Proposed part size ~2.5 process-h, so one part fits a Colab job (wall 3 h) and a Kaggle session runs 4
  lanes x 4 parts in sequence (~10 h < 11 h timeout): S 40 ~500 parts, S 60 ~610. This needs a lane option in
  `campaign._cloud_cmd` (today all parts of a chunk start at once); without it, Kaggle 4 parts per session.
- Order: (1) R-55 calibration block, owned and run by xm-citests (`scratchpad/xmethod/exp_c_timing.py calib`: 6
  anchors x n 500 / 1000 / 4000 (mscr <= 1000), E2 R2, DEV seeds 3_000_000-002 = 3 repeats; Kaggle + each Colab CPU
  model; `exp_c_runtime.py calib` -> f, e), (2) EVAL parts by the
  dispatcher (Kaggle first, then Colab), (3) the Exp C paper timing block (xm-citests `exp_c_timing.py
  paper`: every arm x n, E2 R2 (granger E3 R2), seeds 3_000_000-002, Kaggle only, 1 process per vCPU) once EVAL is
  underway; it does not gate EVAL. Error units re-run until none remain (PROTOCOL_A s.11).
- Slot conflict: the cdl DEV run holds 4 Kaggle + 3 Colab slots now (28 / 64 parts still queued at 17:58 UTC); EVAL
  cannot start before it ends, and it gates T1 / T4 anyway.
- Dispatcher for EVAL (`scratchpad/e6_dev/xm_dispatch.py`, owner dev-runs): chunks 8 / 4 -> 4 / 2 (R-55), EVAL output
  directory, KAGGLE_MINE_MAX 4 (= user cap), Lightning never used.

## B. Gating inputs (all must be in before the freeze)

1. pmrt_nl_eq DEV run on the T1 cells (R-43; Kaggle xm-dev-pmrtnl-k1) merged into results/dev/.
2. cdl full DEV run (R-51) merged; its report = T5, its max cost at n 8000 / 24000 = T3.
3. aud1's power calculation (T1) on the T1 input incl. both runs -> S = clip(S_power, 40, 60) (T6 cap 60, R-56);
   if S_power > 60, the achieved power at 60 is reported (`t1` output `min_power_at_S`).
4. T3 cost pilot for pmrt_nl_eq at n 8000 / 24000 (no DEV cost there; DEV seeds 3_000_000-002, 3 costliest
   (world, regime); est. max ~1 000 CPU-s at n 24000).
5. R-55 calibration block (xm-citests) run on Kaggle and on each Colab CPU model -> f and e by host type (T3).
6. campaign.py (xm/dev-runs) R-55 changes, audited: per-record CPU model (/proc/cpuinfo), loadavg, concurrent
   processes; 1 process per vCPU in the launchers; the lane option (A). Tests pass.
7. Questions: Q1-Q4 answered (R-55 details by the orchestrator, R-56); none open.

## C. Freeze steps (in order, on feat/v2 after merging xm/freeze-prep)

1. Merge xm/dev-runs SELECTIVELY with `scratchpad/xmethod/freeze/merge_dev_runs.py` (dry run, then `--apply`; on
   feat/v2 with xm/freeze-prep merged, clean tree). It takes dev-runs' own files (campaign.py, dev_power.py,
   test_xmethod_campaign.py, e6_dev launchers, dev_*.py, specs/dev, results/dev, status/dev-runs.md, colab_run.py
   by lineage), keeps feat/v2's newer copies (eval_analysis.py, specs/eval, PROTOCOL_A, CONTRACT, FIDELITY_CITESTS,
   PMRT_CORE, f4_heavy.jsonl, xm_classic_integration.py, eval_analysis tests), drops `_ref/` and stops (exit 2) on
   any path it cannot classify. Dry run on xm/freeze-prep 2026-10-04: TAKE 91, KEEP 158, DROP 2, REVIEW 0. It never
   commits; run C5's tests, then commit.
2. Spec: replace "TBD" / "TBD_HALF" seeds with [3100000, 3100000+S-1] / [3100000, 3100000+ceil(S/2)-1]; fill
   `pkgs_lock` (uv.lock versions of campaign.PKGS); `status` -> "FROZEN"; drop or mark filled the `tbd` notes.
   No `freeze_commit` key (the protocol sha is verified from the live file / bundled frozen copy).
3. PROTOCOL_A: T1 VALUE (S), T3 VALUE (final maxima, f, e; cdl and pmrt_nl_eq rows), T4 (iii) cdl, T5 paths,
   s.7 unit counts at the chosen S, header line -> "FROZEN: yes (<date>)".
4. Spec `protocol_sha256` = LF sha256 of that PROTOCOL_A (compute after step 3; never edit either file afterwards).
5. Re-run at the freeze tree: `scratchpad/xmethod/estimand_check.py` (R-36), tests `test_xmethod_eval_analysis.py`,
   `test_xmethod_eval_report.py`,
   `test_xmethod_campaign.py`, `test_xmethod_pmrt*.py`, `test_xmethod_cdl.py`, citests / classic tests; the
   campaign self-test bundle; `campaign list` on the frozen spec (validate_spec + EVAL guard pass).
6. Write `FREEZE_NOTE.md` (template beside this file) with every LF sha256, then ONE freeze commit (subject line
   only, < 70 chars, no trailer) containing everything listed in T7.
7. Orchestrator registers 3_100_000-3_100_299 in `docs/benchmark/SEED_REGISTRY.json` (step 5 of the procedure).
8. Launch section A (2): EVAL from a clean checkout of the freeze commit (records must stamp it; dirty false).

Never after the freeze: edit PROTOCOL_A (amendments file only), the spec, eval_analysis.py, eval_report.py or any
adapter.
