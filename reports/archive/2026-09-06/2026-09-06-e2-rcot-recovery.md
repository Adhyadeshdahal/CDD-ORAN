# E2 RCoT discovery — 10-seed recovery envelope (as-run)

**Date:** 2026-09-06
**Method:** FROZEN E2 RCoT conditional-independence discovery
(`docs/benchmark/E2_RCOT_DISCOVERY_PROTOCOL.md`, `cdd_oran/e2slice/discovery_rcot.py`)
**`protocol_commit`:** `eba381a195c4b341cc35e79b600f6aef815f54ea`
**`null_method`:** `block_perm` (frozen primary; human-approved)
**`score_method`:** `rcot_conditional_independence`
**Seeds:** `r = 0..9` (`env_seed = weight_seed = r`, `sampling_seed = 0`)
**Frozen dataset config:** decoy OFF, `n_rows_per_seed = 4000`, `obs_noise_scale = 0.0`, `sampling_seed = 0`
**True graph:** 16 NCP→KPI edges, 0 KPI→KPI edges (`gt_edge_count = 16`)
**Compute:** single-core (`OMP/MKL/OPENBLAS = 1`); discovery ~1.3–1.8 min/seed, ~14.5 min for the 10-seed
envelope; recovery scoring < 1 s total.

> **Firewall (method validation, NOT decision-value evidence).** This is E2 *discovery-method*
> validation for the nonlinear Gaussian-bump mechanism class. It says **nothing** about whether E2's
> decision gap is real. The recorded E2 decision null (`GATE_CONTRACT_E2.md`, `8d70a4f`,
> `xfail(strict=True)`) stands untouched. A conservative / partial recovery here is a valid recorded
> boundary result, not a defect to tune away. No frozen constant, method, SCM, dataset config, or
> threshold was changed; **no seed was dropped or reweighted.**

## Anti-p-hacking ordering (as executed)

1. **PHASE A (truth-free).** All 10 `discovery_rcot.json` masks were persisted via the canonical
   `write_discovery_rcot` (`frozen_config()` → `block_perm`, no override), content-hashed, and
   fail-closed re-loaded via `load_discovery_rcot` (which re-derives every §11 constant + the
   `protocol_commit`). **10/10 persisted + hashed + loadable BEFORE any ground truth was read.** No
   `true_adj_matrix()` import occurs on the discovery path.
2. **PHASE B (reads truth).** Only after all 10 masks existed and loaded was
   `score_recovery_rcot` run, reading `E2V2Env(env_seed=0).true_adj_matrix()` into a separate
   `recovery_rcot.json` (never `recovery.json`), each verified by `load_recovery_rcot`.

Artifacts are `discovery_rcot.json` / `recovery_rcot.json` under `runs/e2slice-recovery/replicate-NN/`.
The retired pdCor baseline files (`discovery.json` / `recovery.json`) were **not** touched.

## Phase A — discovery selection (truth-free)

No numerical HALT fired: the residual-collapse guard fired on **0 of 84** candidates in every seed
(well below the `ceil(0.05·84) = 5` HALT threshold), and no NaN/inf appeared outside the guard path.

| seed | guarded / 84 | NCP→KPI selected | KPI→KPI selected | total selected | runtime |
|-----:|:------------:|:----------------:|:----------------:|:--------------:|:-------:|
| 0 | 0 | 9 | 1 | 10 | 1.69 min |
| 1 | 0 | 2 | 1 | 3 | 1.70 min |
| 2 | 0 | 0 | 0 | 0 | 1.76 min |
| 3 | 0 | 3 | 0 | 3 | 1.58 min |
| 4 | 0 | 4 | 0 | 4 | 1.29 min |
| 5 | 0 | 0 | 0 | 0 | 1.27 min |
| 6 | 0 | 5 | 1 | 6 | 1.27 min |
| 7 | 0 | 2 | 1 | 3 | 1.26 min |
| 8 | 0 | 3 | 0 | 3 | 1.26 min |
| 9 | 0 | 4 | 2 | 6 | 1.26 min |

## Phase B — recovery vs E2 truth (per seed)

| seed | overall P | overall R | overall F1 | NCP→KPI P | NCP→KPI R | NCP→KPI F1 | KPI→KPI FP / 36 | rejection rate | guarded |
|-----:|:---------:|:---------:|:----------:|:---------:|:---------:|:----------:|:---------------:|:--------------:|:-------:|
| 0 | 0.700 | 0.438 | 0.538 | 0.778 | 0.438 | 0.560 | 1 | 0.972 | 0 |
| 1 | 0.667 | 0.125 | 0.211 | 1.000 | 0.125 | 0.222 | 1 | 0.972 | 0 |
| 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 1.000 | 0 |
| 3 | 1.000 | 0.188 | 0.316 | 1.000 | 0.188 | 0.316 | 0 | 1.000 | 0 |
| 4 | 0.500 | 0.125 | 0.200 | 0.500 | 0.125 | 0.200 | 0 | 1.000 | 0 |
| 5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 1.000 | 0 |
| 6 | 0.667 | 0.250 | 0.364 | 0.800 | 0.250 | 0.381 | 1 | 0.972 | 0 |
| 7 | 0.667 | 0.125 | 0.211 | 1.000 | 0.125 | 0.222 | 1 | 0.972 | 0 |
| 8 | 0.667 | 0.125 | 0.211 | 0.667 | 0.125 | 0.211 | 0 | 1.000 | 0 |
| 9 | 0.667 | 0.250 | 0.364 | 1.000 | 0.250 | 0.400 | 2 | 0.944 | 0 |

## 10-seed envelope (mean / min / max; no seed dropped)

| metric | mean | min | max |
|--------|:----:|:---:|:---:|
| overall precision | 0.553 | 0.000 | 1.000 |
| overall recall | 0.163 | 0.000 | 0.438 |
| overall F1 | 0.241 | 0.000 | 0.538 |
| **NCP→KPI precision** | **0.674** | 0.000 | 1.000 |
| **NCP→KPI recall** | **0.163** | 0.000 | 0.438 |
| **NCP→KPI F1** | **0.251** | 0.000 | 0.560 |
| **KPI→KPI FP / 36** | **0.60** | 0 | 2 |
| **KPI→KPI rejection rate** | **0.983** | 0.944 | 1.000 |
| guarded / 84 | 0.0 | 0 | 0 |

## Comparison vs the retired pdCor baseline (§14 — NOT re-run, read-only)

Retired pdCor **seed-0** (`protocol_commit 828e345`, ~8.65 h): NCP→KPI **P 0.929 / R 0.813 / F1 0.867**,
KPI→KPI **17 FP / 36** (rejection 0.528).

| axis | pdCor seed-0 (retired) | RCoT seed-0 | RCoT 10-seed mean |
|------|:----------------------:|:-----------:|:-----------------:|
| NCP→KPI precision | 0.929 | 0.778 | 0.674 |
| NCP→KPI recall | 0.813 | 0.438 | 0.163 |
| NCP→KPI F1 | 0.867 | 0.560 | 0.251 |
| KPI→KPI FP / 36 | 17 | 1 | 0.60 |
| KPI→KPI rejection rate | 0.528 | 0.972 | 0.983 |
| runtime / seed | ~8.65 h | ~1.7 min | ~1.3–1.8 min |

## Reading of the result (as-run)

- **The calibration property RCoT was built to buy holds.** The block-conditional-permutation null
  crushes the KPI→KPI over-selection the marginal-permutation pdCor null produced: **0.60 mean FP / 36
  (rejection 0.983)** across 10 seeds, versus pdCor's **17 FP / 36 (rejection 0.528)** at seed 0. The
  spurious lagged-KPI→KPI edges are almost entirely rejected, and the ~300× speedup (minutes vs hours)
  is a further win.
- **Recovery is conservative, not complete.** NCP→KPI **recall is low (mean 0.163, max 0.438)** —
  precision stays moderate-to-high where anything is selected (many seeds at 1.000), but many of the 16
  true NCP→KPI parents go unselected, and two seeds (2, 5) select nothing at all. RCoT trades pdCor's
  higher recall for a much better-calibrated, near-zero-false-positive null. This is a **recorded
  partial recovery**, not a green full recovery.
- **Interpretation limits (§16).** Direction is supplied only by the frozen temporal ordering
  `t → t+1`, not by the test; full-conditioning on the other 13 candidates can overcondition; E2 is
  noiseless by design. None of these was altered to improve a number.

## Provenance

- Frozen protocol: `docs/benchmark/E2_RCOT_DISCOVERY_PROTOCOL.md` (`protocol_commit eba381a`).
- Discovery module: `cdd_oran/e2slice/discovery_rcot.py` (`PROTOCOL_COMMIT = eba381a…`).
- Recovery scorer: `cdd_oran/e2slice/evaluate_rcot.py` (mirrors `evaluate.py`; reuses
  `full_graph_from_discovered_mask`, `_per_target_metrics`, `recovery_by_edge_type` / `_prf`).
- Per-seed artifacts: `runs/e2slice-recovery/replicate-{00..09}/discovery_rcot.json` and
  `recovery_rcot.json`.
