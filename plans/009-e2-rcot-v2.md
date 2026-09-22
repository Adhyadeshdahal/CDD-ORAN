# Plan 009: E2 RCoT-v2 — the low-recall fix (`block_perm_reps` 99 → 299)

> **Light iteration note on Plan 008.** This is the v2 iteration of the frozen RCoT E2 discovery method.
> It is a **single-constant** change to Plan 008's v1 and inherits 008's entire firewall, freeze
> discipline, and run/scoring machinery. Read Plan 008 and the v1 protocol for the full context; this note
> records only what v2 adds.

## Status

- **STATUS: DONE (2026-09-22 correction) — RUN COMPLETE + CANONICAL.** RCoT-v2 is the current frozen E2
  discovery method (`cdd_oran/e2slice/discovery_rcot_v2.py`, `PROTOCOL_COMMIT_V2 = 8052e10`); run finished
  all 10 seeds (`runs/e2slice-recovery/replicate-{00..09}/discovery_rcot_v2.json`). RESULT = P 0.964 / R 0.769
  / F1 0.855. Canonical E2 discovery per `docs/ARCHITECTURE.md`. (Was: "RUN = pending execution"; the unchecked
  Done-criteria boxes below are stale.)
- **Priority**: P1 (recall fix for the frozen RCoT E2 discovery method; sibling to Plan 008)
- **Depends on**: Plan 008 (frozen RCoT-v1 method + dataset + evaluate machinery),
  `reports/2026-09-06-e2-rcot-lowrecall-diagnosis.md` (the diagnosis that motivates v2)
- **Registered at**: 2026-09-06

## Why v2 (one paragraph)

Frozen RCoT-v1 (`protocol_commit eba381a`, `block_perm_reps = 99`) fixed the KPI→KPI over-selection but
recovered only ~2–3 of 16 true NCP→KPI edges per seed (recall 0.163, F1 0.251). The diagnosis proved this
is a **BH × permutation-resolution bug, not low power**: the `B = 99` p-value floor `1/(99+1) = 0.010`
sits **above** the per-target BH-FDR leading threshold `q/m = 0.05/14 = 0.00357`, so true edges the test
detects raw (~80%) are discarded at selection. Raising `B` to **299** (floor `1/300 = 0.00333 < 0.00357`)
lifts truth-free BH-power **0.086 → 0.742** while KPI→KPI FP stays flat at **~0.010** (diagnosis §B2). The
looser `analytic_hbe` null was **rejected** (its FP grows to 0.120 at n=4000, §B3). **v2 = v1 with exactly
one constant changed.**

## What changed in code (thin sibling, v1 fully preserved)

- **New**: `cdd_oran/e2slice/discovery_rcot_v2.py` — `PROTOCOL_COMMIT_V2` (FREEZE-PENDING placeholder),
  `FROZEN_BLOCK_PERM_REPS_V2 = 299`, `frozen_config_v2()`, `write_discovery_rcot_v2()` /
  `load_discovery_rcot_v2()`. Imports the v1 numerics + record/persist/load helpers; overrides only the
  config, the artifact filename (`discovery_rcot_v2.json`), and the `protocol_commit`.
- **New**: `cdd_oran/e2slice/evaluate_rcot_v2.py` — `score_recovery_rcot_v2()` / `load_recovery_rcot_v2()`
  writing `recovery_rcot_v2.json`; reuses every shared metric helper.
- **New**: `tests/test_e2slice_discovery_rcot_v2.py`.
- **Minimal additive refactor of v1** (backward-compatible, keyword-only params defaulting to v1
  behaviour; v1 tests still green): `discovery_rcot.py` `build_rcot_discovery_record` gains
  `protocol_commit=`; `write_discovery_rcot` / `load_discovery_rcot` gain
  `config_factory`/`filename`/`descendants`/`protocol_commit` / `expected_protocol_commit`. `evaluate_rcot.py`
  `score_recovery_rcot` gains `discovery_loader`/`out_filename`; `load_recovery_rcot` gains
  `filename`/`expected_protocol_commit`.
- **v1 preserved**: `discovery_rcot.py`, its on-disk `discovery_rcot.json`/`recovery_rcot.json` (B=99,
  `eba381a`) in `runs/e2slice-recovery/replicate-*/` remain valid + loadable; v2 writes DISTINCT filenames
  beside them.

## Freeze ordering (mandatory; identical mechanism to Plan 008)

1. Commit `docs/benchmark/E2_RCOT_DISCOVERY_PROTOCOL_V2.md`. **That commit's SHA is the freeze.**
2. Replace the placeholder `PROTOCOL_COMMIT_V2` in `discovery_rcot_v2.py` with that SHA; commit the code.
   Config is already locked via `frozen_config_v2()` + `load_discovery_rcot_v2` re-derivation.
3. Run discovery for ALL 10 seeds → persist + content-hash every `discovery_rcot_v2.json` FIRST; verify
   fail-closed-loadable.
4. ONLY THEN score recovery vs `E2V2Env().true_adj_matrix()` into `recovery_rcot_v2.json`.

## Run plan / cost

- `block_perm`, `B = 299` is ~3× v1's `B = 99`: **~10 min/seed at n=4000 → ~1.7 h for all 10 seeds**,
  single-core. Trivial vs the retired pdCor's ~8 h/seed.
- Discipline: preserve every replicate; on a guard-fraction HALT (≥5/84) or NaN/inf outside the guard
  path, RECORD and STOP — never silently drop/retune. `block_perm_reps` is now a FROZEN constant too: do
  NOT raise it further to chase a number.

## Truth-free verification (done, pre-freeze)

- Diagnosis §B2 B-sweep: B=299 → BH-power 0.742, KPI→KPI FP 0.010.
- v2-code-path sanity (`discover_graph_rcot` + `frozen_config_v2`, n=1000, 8 reps, single-core): BH-power
  and KPI→KPI FP match the diagnosis B=299 row — see the protocol §12(b) FREEZE-PENDING slot for the
  transcribed numbers.
- v2 + v1 tests all green single-core; ruff clean.

## Done criteria

- [ ] Protocol V2 frozen (committed); SHA recorded as `PROTOCOL_COMMIT_V2`; config locked.
- [ ] Method run on all 10 seeds; `discovery_rcot_v2.json` masks persisted + hashed BEFORE truth; per-seed
      recovery + KPI→KPI FP/rejection reported; no seed dropped.
- [ ] Result doc compares v2 vs v1's recorded first-freeze result and the retired pdCor baseline; restates
      the firewall (method validation, NOT E2 decision-value evidence); v1 + pdCor untouched on disk.
