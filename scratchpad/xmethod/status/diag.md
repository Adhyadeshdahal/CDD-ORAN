WORKING (X5 / X6 declared + built on xm/diag; nothing run yet; paused for compaction)

# diag status (2026-10-05 ~14:10Z): post hoc diagnostics X5 (advisor B) / X6 (advisor C), exploratory, R-60

## Branches
- `xm/diag` from feat/v2 0fe4eb3 (X4 merged): 57cef93 declaration (X5 / X6 after X4, seeds); d897cbe code.
- `xm/exp-b` was reset to 134529a (its READY-TO-MERGE X2 / X3 state). Its earlier copies of these two commits
  (3d683c5, 8e770fb) were made before the orchestrator's branch note.
- Not pushed. feat/v2 was not merged into xm/exp-b.

## Declared (EXTRAS_PROTOCOL.md Amendments, before any p-value was read)
- Before declaring, only these EVAL fields were read: cpu_s, gen_cpu_s, platform, the record structure, and the
  published e4_by_lambda.csv.
- Seeds: XMETHOD_DIAG in SEED_REGISTRY.json: E4 3300000-3301999 (X5), E1 3302000-3302199 (X6).
- Registry pin check:
  - The E6-P pin (`e6p_screen.FROZEN_SHA256`, 5ab935...) was already stale before this edit; feat/v2 is at 2e7f33...
  - Recorded, never enforced (E6-P drivers only check their own block / tags are present): nothing breaks.
- X5 (VPS):
  - Step 1: EVAL records read-only. Uniformity (KS / QQ), tail, concentration, and the chance label count
    (frozen `_boot_mult` bootstrap, 10 000 simulations).
  - Step 2: specs x5_fail (failing cells (lam 0, n 8000), (lam .5, n 8000), (lam 1, n 24000)) and x5_adj (the other
    lambdas at the same n); pmrt_eq + pmrt_r3; 2000 seeds per cell. Primary test: one-sided exact binomial, Holm over
    the 3 failing pairs; outcomes EXCESS / CHANCE / UNRESOLVED.
  - Step 3: E4 R3 has no setpoints, so step 3 = pmrt_r3 on the same data (tested). Exact McNemar, only if an EXCESS
    occurs.
- X6 (Colab): spec x6_gbm, E1 R2 n 1000, 200 seeds, 15 arms.
  - Told-width grid .5 / .8 / 1 / 1.25 / 1.5 / 2, for GBM and Lin.
  - 2 x 2 ablation at told x2: arms `.w200` (TT), `.w200_Tt`, `.w200_tT`, `.w200_tt`.
  - Bias log `x6` in each GBM record.
  - Predictions P1-P4; outcomes SUPPORTED / REFUTED / PARTIAL.

## Built (d897cbe)
- `cdd_oran/xmethod/extras.py`:
  - X5 / X6 experiments with their own seed guard (XMETHOD_DIAG);
  - `x6_hooks` (pmrt_nl.design_law / gbm_profiles / GbmStat wrapped during an X6 gbm unit, restored after) and
    `_x6_run_one`;
  - spec builders; `diag_cost_agg.json` (dev_cost_agg.json unchanged).
- Specs in `scratchpad/xmethod/specs/extras/`: x5_fail.json, x5_adj.json, x6_gbm.json, diag_cost_agg.json. The X2 / X3
  spec files are unchanged.
- Tests `tests/test_xmethod_extras.py`:
  - X5 / X6 specs and seeds; r3 = eq minus lagged actions; X6 TT / exact p-values = frozen; c = 0; hooks restored.
  - On xm/diag (Windows): all pass except `test_freeze_manifest_still_valid`.
  - That failure is NOT ours: feat/v2 db644e0 ("EVAL merged") changed the frozen `scratchpad/e6_dev/xm_dispatch.py`
    after the freeze. Q1 below.

## Cost projection
- X5, from EVAL unit costs per platform:
  - 87.5 VPS CPU-h (172 Kaggle-ref; DEV 203): x5_fail 28 (about 4 h at 7 procs), x5_adj 59 (about 8.5 h).
- X6: 6.5 VPS CPU-h-eq. (about 15 Kaggle-ref; DEV 18.6); Colab 4 procs about 2 h.

## Running
- Nothing of ours. At 13:27Z the VPS ran xm-citests' X4 (xm-x4-v1, nearly done); xm-citests runs X7 there next.

## Next steps
1. Step 1 locally: write `cdd_oran/xmethod/diag.py` (analysis), run X5 step 1 on
   `D:/academia/major-project/CDD-ORAN/scratchpad/xmethod/results/eval/merged.jsonl.gz` (feat/v2 checkout, read-only).
2. X6 on Colab: `uv run python -m cdd_oran.xmethod.extras colab --spec scratchpad/xmethod/specs/extras/x6_gbm.json
   --name xm-diag-x6 --parts 4 --cost-table scratchpad/xmethod/specs/extras/diag_cost_agg.json
   --venv-python 3.12.14 --paths scratchpad/xmethod/EXTRAS_PROTOCOL.md`.
3. X5 on the VPS (free, X7 not waiting): x5_fail then x5_adj, `extras vps --spec ... --parts 7 --procs 7
   --cost-table ...diag_cost_agg.json`; coordinate with xm-citests; a preflight refusal stops the lane.
4. Pull and merge into results/extras/<spec>/, then analyse (X5 tests, X6 rates + bias), then
   results/extras/DIAG_REPORT.md + diag_tables.json, then a commit, then set line 1 to READY-TO-MERGE.

## Questions
- Q1: The freeze manifest fails on feat/v2 0fe4eb3 for `scratchpad/e6_dev/xm_dispatch.py`, which db644e0 changed
  after the freeze. Is that expected? (Not caused by X5 / X6.)
- Q2: VPS order: X5 x5_fail (about 4 h) before or after xm-citests' X7? Default: after X7 unless the VPS is idle and
  X7 is not ready.
