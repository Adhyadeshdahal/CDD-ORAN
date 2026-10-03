READY-TO-MERGE
# protocol4 status (2026-10-03): R-41 cmi_knn torch backend + GPU wall budget in the EVAL spec and PROTOCOL_A

Branch xm/protocol4 from feat/v2 acd3dfa. Local commit, not pushed. No EVAL seed claimed or generated;
SEED_REGISTRY untouched; no dependency added; status/protocol*.md of earlier tasks left as is.

## HAND-BACK
- `specs/eval/full.json`: cmi_knn_eq / cmi_knn_native / cmi_knn_eq_min get config `backend: "torch"` (was "cpu")
  plus `budget_cpu_s: null`, `budget_wall_s: 7200`, `device: "gpu"`. These keys and values match the dev-runs
  DEV specs (full_ci, pilot_ci_gpu, smoke_ci; dev-runs Q8). No other arm, block or unit count changes. The spec sha
  changes.
- `scratchpad/xmethod/eval_analysis.py` (R-41: GPU time reported apart from CPU):
  - `cost_field(arm)` returns "wall_s" for an arm with `budget_wall_s`, else "cpu_s".
  - The infeasible cost and the T3 cost are read in that unit. GPU arms use `infeasible_cost_wall_s` and spec
    `t3_cost_wall_s`; CPU arms keep `infeasible_cost_cpu_s` / `t3_cost_cpu_s` unchanged.
  - V10 rows add `budget`, `wall_s_mean` and `wall_s_max`. Wall time is the isolated child's `child_wall_s` when
    recorded, else `wall_s`. T3 rows carry `t3_dev_cost_<field>`.
  - The markdown prints a separate V10 table for the GPU arms (wall-s), and "GPU wall-s" on not-counted cells.
- `docs/xmethod/PROTOCOL_A.md` (DRAFT v3, R-1..R-41; 268 lines):
  - Header: xm/protocol4, feat/v2 acd3dfa.
  - s.0 adds (g), F2-GPU plus the GPU cost seen on one DEV dataset, and R-41 in the post-review choices.
  - Methods table, cmi_knn row: torch (GPU) neighbour search in every arm (R-41); F2-GPU 18/18 bit-identical on a
    Kaggle T4.
  - s.7 has a new "GPU budget (R-41)" item:
    - 2 h wall (7200 s) per (method, dataset) on a Kaggle T4; over it = "infeasible at this n (measured cost X)";
    - EVAL 2x safety cap 14 400 wall-s;
    - GPU wall time reported apart from CPU-s (V10, Experiment C).
  - s.8 cost line and s.12 V10 mention wall-s for the GPU arms.
  - T3 adds the GPU rule: DEV wall-s per unit on a Kaggle T4 > 7200, `t3_cost_wall_s`, cost pilot on a T4.
  - Section 0 was re-wrapped as one paragraph. Only (g) and the R-41 clause are new.
- Tests `tests/test_xmethod_eval_analysis.py`: 26 pass + 1 skip (the campaign.expand test needs xm/dev-runs). Only
  this file was run (memory note). ruff clean. New tests:
  - `test_gpu_arm_cost_is_wall_seconds_r41` (synthetic: infeasible wall cost, T3 wall cost, V10 budget / child
    wall, markdown);
  - `test_eval_spec_cmi_knn_torch_r41` (all three cmi arms torch + wall budget; they are the only GPU arms).
- The pilot plumbing output (`results/eval_analysis_pilot/`) is not regenerated. The DEV pilot spec has no
  `budget_wall_s`, so the only change would be new V10 keys.

## Findings
- F11 campaign (xm/dev-runs `_over`) records an EVAL safety-cap breach as status error ("EVAL safety cap ...
  exceeded"). PROTOCOL_A s.7 (since v2) says "recorded infeasible with its cost, never re-run". eval_analysis
  counts an error as not run. The two should agree before the freeze. I did not touch campaign or change the
  protocol rule.
- F12 F2-GPU is bit-identical, so torch versus cpu changes no statistic or p; only the cost and the host change.
  Spec runs use `default_config()` + the spec config (`_citests_common`). `native_config()` (backend "cpu") is
  used only by the fidelity gates, so it is unaffected.

## Coordination
- dev-runs: EVAL records must stamp the new spec sha. The EVAL GPU shards need a Kaggle T4 kernel; campaign already
  reads `budget_wall_s` / `budget_cpu_s` (`arm_budgets`).
- citests2: nothing needed (cmi_knn default backend is already "torch").

## QUESTIONS
- Q12 (non-blocking) PROTOCOL_A s.7 Platforms says "a dataset's arms in one shard on one platform". Do the cmi_knn
  GPU arms run in the same Kaggle T4 shard as the CPU arms of that dataset, or in a separate GPU shard? The
  eval_analysis check compares only the `host.platform` label ("kaggle" both ways), so it passes either way. The
  text is unchanged until ruled.
