READY-TO-MERGE
# protocol7 status (2026-10-03): R-46..R-50 (pdcor and cmi_knn dropped, CI DEV plan C, cdl placeholder)

Branch xm/protocol7 from feat/v2 8e4cb85 (the R-49 revision / R-50 update came mid-task; no commit existed yet, so
the branch was re-pointed at the new tip, no merge). Local commit, not pushed. No EVAL seed claimed or generated;
SEED_REGISTRY untouched; no dependency added; no compute used (no Lightning credits). The spec sha changes.

## HAND-BACK
- EVAL spec (`specs/eval/full.json`), 30 -> 25 arms:
  - pdcor_eq / _native / _eq_min removed (R-48); cmi_knn_eq / _native / _eq_min removed (R-49 revised), also from the
    E4 R3 / R4 block lists. No GPU arm is left (no `budget_wall_s`).
  - New placeholder arm `cdl` (R-50): ref `cdd_oran.xmethod.methods.cdl:TBD-R-50`, config arm native, declare tau,
    analysis primary, `fixed_threshold: 0.16`; in the "all" blocks and the E4 R3 / R4 native block.
  - tbd: `pdcor` and `cmi_knn` (dropped, reason), `cdl` (placeholder: what the cdl worker fills),
    `large_n_citests` (OPEN, R-47, T11); `citests_arms` now lists mscr / pcorr / rcot2.
  - Units (recomputed with `planned_units`; the old 30-arm numbers reproduce): S 40 = 207 560 units / 16 200
    datasets (46 480 tune); S 100 = 323 180 / 19 500.
- eval_analysis (`xm-eval-analysis/4`):
  - `fixed_declare(result, thr)`: declares iff score >= thr. This matches the conference rule
    (`CDL.get_binary_graph`: mask_CMI >= threshold on refactor/codebase). A NaN score is not declared.
  - An arm with spec `fixed_threshold` gets `like_fixed` (validity over all three declaration rates, untuned) and a
    V0 row rule `fixed`; the placebo is not a tuning column there. V0 legend updated.
  - Comments only: set-D (pdcor gone, R-48) and R-41a ("a GPU shard"). The generic GPU / set_D code paths are kept.
- PROTOCOL_A (DRAFT v4, R-1..R-50, 328 lines):
  - s.0: R-47 plan C, R-48, R-49 revised, R-50 disclosed (history items (f), (g) and R-40 / R-41 kept as seen).
  - s.4: pdcor and cmi_knn rows removed; one "considered and excluded" sentence each, with the reason (pdcor: not CI,
    audit .22 / 5 of 5, O(n^2); cmi_knn: cost, s.0 (g) numbers). New cdl row; T4 gate covers cdl.
  - s.5 / s.6 / s.9: cdl among the native-only methods; fixed .16 as a secondary V0 scoring.
  - s.7: unit counts; GPU budget rule generic (none planned; cdl's device from its cost); platforms R-46.
  - s.10: pdcor sentences removed from C1 / C2b. s.11: shards per R-46.
  - T1: mscr pairs at n 500 / 1000 (plan C). T4: cdl via `status/cdl.md`. New T11: large-n mscr grid OPEN.
- Tests (`tests/test_xmethod_eval_analysis.py` only): 34 pass + 1 skip (campaign.expand needs xm/dev-runs). ruff clean.
  - Removed: `test_eval_spec_cmi_knn_torch_r41`.
  - New: `test_eval_spec_and_protocol_drop_pdcor_cmi_knn_r48_r49`, `test_eval_spec_cdl_placeholder_r50`,
    `test_fixed_threshold_secondary_scoring_r50`; the set-D list in `test_eval_spec_r40` updated.

## Findings
- F15 R-46 vs T3: T3 reads DEV cost on host.platform kaggle only. DEV units run on Lightning / Colab do not count for
  T3, so an arm costed only there needs the Kaggle cost pilot of T3 (Q18).
- F16 The cdl brief has the adapter report the .16 declarations in notes. eval_analysis recomputes them from the
  score (score >= .16), so the score must be the final EMA CMI on the conference scale. The notes are not read.
- F17 The cdl adapter and pdcor / cmi_knn adapters stay in the repo (out of scope); only Study A no longer uses them.

## Coordination
- cdl: fill `ref` (class), version, training-budget rule and device in `tbd.cdl`. The score = final EMA CMI.
- dev-runs: EVAL records must stamp the new spec sha; campaign ignores `fixed_threshold`.

## QUESTIONS
- Q17 cdl and T1 / T4 (iii): T4 (iii) admits an arm only if its DEV run is in the T1 input. The cdl brief plans one
  DEV seed per cell only. Option 1: dev-runs runs cdl on the full DEV grid before the freeze (T1, T5). Option 2:
  waive T4 (iii) for cdl (gate on fidelity only); cdl then joins no T1 pair. Which?
- Q18 T3 host: keep the Kaggle reference host (pilot on Kaggle when the DEV cost came from Lightning / Colab), or
  accept Lightning / Colab DEV costs with a measured speed factor? Default kept as written: Kaggle only.
