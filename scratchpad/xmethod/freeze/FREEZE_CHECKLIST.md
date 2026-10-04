# Study A freeze package: EVAL launch plan and freeze checklist (freeze-prep, 2026-10-04)

Protocol `docs/xmethod/PROTOCOL_A.md` (DRAFT v5, rulings R-1..R-56); spec `scratchpad/xmethod/specs/eval/full.json`;
projection `scratchpad/xmethod/freeze/eval_projection.json` (script `eval_projection.py` in the same directory).
Nothing here touches an EVAL seed; EVAL needs the freeze (section C) first.

## A. EVAL launch plan

Method set (24 arms): pmrt_nl_eq (primary, gbm, R-52), pmrt_eq, pmrt_r3, pc_eq / _native, granger_eq / _native,
corr, notears, shap_dag, two_tower, cdl (R-53; R1 / R2 n <= 4000, R-58(1)), mscr_eq / _native (n 500 / 1000 only,
R-54; mscr_eq_min dropped, R-58(5)), pcorr x3, pcorr_hac / _fb (+ eq_min), rcot2 x3. Dropped: pdcor (R-48),
cmi_knn (R-49 revised). No GPU arm.

Units and cost (R-58 grid; tune = DEV tune seeds re-run, R-34; C3 readers at S_E4 300 in E4 R3 only, E4 R4 at S,
T9 / R-58(4)). T6 cap 60 (R-56): S = clip(S_power, 40, 60). Projection 2026-10-04: pmrt_nl_eq DEV run + T3 cost pilot,
cdl Kaggle costs. Wall assumes full slots at 1 process per vCPU (R-55): Kaggle 4 sessions x 4 = 16 processes, Colab
3 jobs x 2 = 6, the VPS 7 (R-57); host speed differences are ignored (T3 converts costs by the R-55 factors).

| S | units | datasets | tune units | CPU-h | wall h: Kaggle + Colab + VPS | Kaggle + VPS | Kaggle only |
|---|---|---|---|---|---|---|---|
| 40 | 150 960 | 11 000 | 40 520 | 509 | 17.6 | 22.1 | 31.8 |
| 50 | 168 970 | 11 750 | 40 520 | 564 | 19.4 | 24.5 | 35.2 |
| 60 | 186 980 | 12 500 | 40 520 | 618 | 21.3 | 26.9 | 38.6 |

- R-58 expected ~360-450 CPU-h and ~17-22 h on Kaggle + VPS; the measured pmrt_nl_eq cost (pilot: E2 ~480 / ~1 100
  CPU-s per dataset at n 8000 / 24000) puts S 40 at ~510 CPU-h, ~22 h on Kaggle + VPS. Add ~10-15 % for setup
  (~10-15 min per session, sessions <= 11 h) and error re-runs: S 40 ~1 day on Kaggle + VPS (~1.5 days Kaggle
  alone); S 60 ~1.3 / ~1.8 days. Colab may be unavailable (it refused runtimes overnight).
- EVAL interpreter: exactly Python 3.12.14 on every platform (R-57; campaign `EVAL_PYTHON`, pin_check refuses any
  other patch release). The VPS was upgraded to match; it needs its own R-55 calibration (host type 'vps' + CPU
  model) before its records count for T3, and the dispatcher needs a VPS target (owner dev-runs).
- By arm at S 40: pmrt_nl_eq 312 CPU-h (61 %), two_tower 33, shap_dag 29, cdl 29, pmrt_eq / pmrt_r3 24 each, the
  rest smaller. By n: 500 34, 1000 68, 4000 113, 8000 82, 24000 207 CPU-h. Tune 122 CPU-h, measure 382.
- Cost basis (per (arm, world, regime, n) mean CPU-s):
  - 10 arms: results/dev/full (Kaggle). CI arms: results/dev/ci_c (Kaggle / Colab / Lightning; oversubscribed
    sessions, an upper estimate).
  - pmrt_nl_eq: results/dev/pmrt_nl (E1 / E2 / E3 / E5 R1 / R2, n <= 4000) + the T3 cost pilot
    (`freeze/pmrt_nl_cost_pilot.json`: E2 R1 / R2 and E4 R3 at n 8000 / 24000, Kaggle, 1 process per vCPU). Other
    worlds at n 8000 / 24000 = their n 4000 cost x the E2 growth (x1.85 / x4.24). Cells with no cost at all (E4 R1 /
    R2 / R4, E4 R3 n <= 4000) = the per-n mean x pmrt_eq's per-cell / per-n ratio (`--proxy`).
  - cdl: Kaggle CPU-s per n 17 / 34 / 121 (n <= 4000, R-58(1)).
- Command: `eval_projection.py --dev <xm-pmrt results/dev> --extra-agg pmrt_nl/agg.json --pilot
  scratchpad/xmethod/freeze/pmrt_nl_cost_pilot.json --cdl-kaggle 17,34,121,232,487 --proxy pmrt_nl_eq=pmrt_eq
  --vps-procs 7 --out scratchpad/xmethod/freeze/eval_projection.json`.
- Platforms (user): Kaggle at most 4 of the 5 account sessions (one stays for other workers), the VPS (7 processes,
  R-57), Colab CPU at most 3 jobs if available, NO Lightning.
- Parts: whole-dataset groups, LPT on the cost table (`campaign.partition`, `--cost-table` from the projection
  basis). Proposed part size ~2.5 process-h, so one part fits a Colab job (wall 3 h) and a Kaggle session runs 4
  lanes x 4 parts in sequence (~10 h < 11 h timeout): S 40 ~200 parts, S 60 ~250. This needs a lane option in
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
2. cdl DEV run at n <= 4000 (R-51, R-58(7)) merged; its report = T5; the parts at n > 4000 do not gate the freeze
   (descriptive).
3. aud1's power calculation (T1) on the T1 input incl. both runs -> S = clip(S_power, 40, 60) (T6 cap 60, R-56);
   if S_power > 60, the achieved power at 60 is reported (`t1` output `min_power_at_S`).
4. DONE: T3 cost pilot for pmrt_nl_eq at n 8000 / 24000 (dev-runs, Kaggle xm-dev-cdl-p1-040214 / -p2-040523):
   max 1148 CPU-s (E2 R1 n 24000), E4 R3 n 24000 ~100: feasible; R-58(3) fallback not triggered (user). Records
   merged on xm/dev-runs dea0091 (results/dev/pmrt_nl_cost/merged_{e2,e4r3}.jsonl.gz; sha in FREEZE_NOTE, T3).
5. R-55 calibration block (xm-citests) run on Kaggle, on each Colab CPU model and on the VPS (R-57) -> f and e
   by host type (T3).
6. campaign.py (xm/dev-runs) R-55 / R-57 changes, audited: EVAL_PYTHON = "3.12.14" with pin_check on the patch
   release; per-record CPU model (/proc/cpuinfo), loadavg, concurrent processes; 1 process per vCPU in the launchers
   (VPS 7); the lane option (A); a VPS target in the dispatcher. Tests pass.
7. Spec generated by `freeze/r58_spec.py` (R-58; `--e4r3-fallback` only if R-58(3) is ever triggered, then
   re-simulate the C3 F_max and T9 on 16 cells).
8. Open questions in `status/r58-trim.md`.

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
