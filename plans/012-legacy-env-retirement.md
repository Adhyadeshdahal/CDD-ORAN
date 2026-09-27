# 012 — Legacy Env I–IV retirement → hard-delete (sequenced)

**Status:** TRACKED SEQUENCE. The E-series (E1–E5) is the live env system; legacy `Environment I–IV`
(`cdd_oran/envs/env_i.py … env_iv.py`) is already RETIRED + quarantined but **not yet deleted**. This file
records the remaining steps and the gate that must clear before deletion, so the cleanup is a plan item, not
a mental note. See memory `legacy-vs-eseries-boundary`.

## 2026-09-13 status update (re-assessed)
- **GATE (step 1) CLEARED:** the FM-1/FM-2 E-series objective re-validation is DONE (honest NULL) — report
  `reports/archive/2026-09-07/2026-09-07-fm-objective-eseries-revalidation.md`. The principled block on
  deletion is lifted.
- **E-series↔legacy import SEAM already CLOSED:** `tests/test_eseries_import_firewall.py` is 9/9 GREEN — the
  E-series (v2 envs, e1slice/e2slice evaluators, `prf`, `recovery_metrics`) loads NONE of `env_i..iv`. The
  "residual seam" prose in the firewall test docstring (lines ~12-16) and in §"Remaining inbound references"
  below is STALE; the lazy-import fix `f076843` closed it. So the E-series is already legacy-free.
- **The REAL remaining blocker to hard-delete is the BASELINES, not a gate.** Deleting `env_i..iv.py`
  cascades into ~30 files — the whole legacy v1 stack: `envs/base.py`+`XApp`+`statistics.py`, `conflicts.py`,
  `get_env`, `experiments/`, legacy `analysis/*`, legacy `config`/`cli` paths, and the planners
  **QACM/CEM/Joint/MPPI/MCTS** (`planners/cost.py:189`, `joint.py`/`horizon.py` via `get_env`). `plans/011`
  wants to KEEP QACM/CEM/Joint/MPPI as E-series BASELINES, but they are wired to the legacy env API and NOT
  ported to `V2Env` — so they are not E-series-functional as-is. **Correct sequence before deletion: PORT the
  kept baselines onto `V2Env` (or explicitly abandon them), THEN hard-delete the legacy envs + dead v1 code.**
- **USER DECISION 2026-09-13:** deletion DEFERRED ("forgot about the baselines; do the safest thing or
  nothing"). Safest action taken = none needed (the E-series decouple was already complete). Hard-delete
  remains USER-GATED, now blocked on the baseline-porting decision, not on the FM gate.

## Where we are
- **E1–E5 built + LIVE:** E1 done, E2 current frontier, E3/E4 scaffolded, E5 research. E-series envs live
  under `cdd_oran/envs/v2/` (`e2.py`, `e4.py`, …).
- **Legacy already quarantined (committed on `feat/v2`):**
  - `e4447be` — legacy v1 envs marked retired; `cdd_oran/envs/README.md` (legacy-vs-E-series boundary).
  - `fd7249a` — decoupled E-series PRF from retired legacy envs (`_prf` move).
  - `f076843` — lazy-import `env_i..iv` in `envs/__init__` (Option B root fix).
- **Mechanism provenance already COPIED into the E-series:** e.g. `e2.py:110-121` carries the K5/K0..K4
  formulas verbatim; `env_ii.py` is now only a *citation* for E2, not a code dependency of it.

## Why deletion is GATED (do not delete yet)
Legacy Env I–IV is the **sole reference** for experimental history the E-series has not yet re-validated —
above all the **FM-1/FM-2 objective findings** now recorded in `plans/011` (validated on legacy Env I, not
the E-series objective). Deleting legacy before that re-validation destroys the only reference for claims we
still lean on. **Gate = the E-series re-validation in 011 must complete first.**

## Remaining inbound references to decouple (as of 2026-09-06)
Live-ish dependencies (not just comments): `cdd_oran/analysis/{graph_baselines,prf,stats}.py`,
`cdd_oran/config.py`, `cdd_oran/envs/{__init__,statistics}.py`, `cdd_oran/envs/v2/__init__.py`,
`cdd_oran/planners/cost.py`, `scripts/{ablation_sweep,attribution_eval,recovery_gate,run_journal_suite}.py`,
`tests/{test_attribution,test_discovery,test_eseries_import_firewall,test_posterior,test_structure_dynamics}.py`.
Provenance-only citations (rehome, don't need imports): `cdd_oran/envs/v2/{e2,e4}.py`, `envs/README.md`.

## Sequence
1. **[GATE] Re-validate borrowed findings on the E-series** — FM-1/FM-2 objective behavior on the E-series
   objective (`analysis/counterfactual_metrics.py`, `SEMANTICS.md:205-228`); any recovery/attribution
   history still cited from legacy. Until this clears, stop here.
2. **Rehome provenance** — make the E-series mechanism files self-authoritative (or cite a frozen doc, not
   `env_ii.py`/`env_iv.py`); update `envs/README.md`.
3. **Decouple live deps** — remove `env_i..iv` imports/branches from analysis, `config.py`, `cost.py`,
   scripts, and the listed tests (repoint to E-series or delete the legacy-only paths). Keep
   `test_eseries_import_firewall.py` green (it exists to prove the E-series doesn't import legacy).
4. **Hard-delete** `cdd_oran/envs/env_i.py … env_iv.py` and any legacy-only helpers left unreferenced.
5. **Verify** — full test suite green; `git grep env_i..iv` returns only historical/plan docs.

## When
The **LATER** bucket, alongside 011's implementation — after P0→K5 / the E2 spine work and the E-series
objective re-validation. Not before: steps 3–4 depend on the gate in step 1. Doing it now would cut the
reference branch we are still standing on.

## Progress note — 2026-09-28 (step 3, first cut: live V2 path decoupled at import time)
- `cdd_oran/planners/__init__.py` is now lazy (PEP 562 `__getattr__` + imports inside
  `get_planners`/`build_aggregator`). Before, importing any planner submodule (e.g. the live V2
  `cdd_oran.planners.sequence`, used by `scripts/e3_decision_gate.py`, `scripts/d1_decision_study.py`)
  eagerly loaded cem → ensemble → `cost.py` → `cdd_oran.envs.legacy` (+ gymnasium, config).
  `from cdd_oran.planners import get_planners, build_aggregator` (experiments/evaluate.py) unchanged.
- `tests/test_eseries_import_firewall.py` now asserts in a fresh subprocess that
  `cdd_oran.planners.sequence`, `cdd_oran.benchmark.learned_world_model`, `cdd_oran.benchmark.rollout`
  and `cdd_oran.decision.arbiter` load no `cdd_oran.envs.legacy*` module, and that
  `cdd_oran.decision.arbiter` loads no torch.
- Still blocking hard deletion (step 4): the E6 QACM baseline now lives self-contained in
  `cdd_oran/envs/e6/published.py`, so the legacy planners (`planners/{qacm,cem,mppi,mcts,horizon,joint,
  ensemble,cost}.py`; `cost.py` imports `envs.legacy.base.XApp`) serve only the legacy pipeline.
  Remaining legacy consumers: `experiments/{discover,evaluate,train,viz}.py`, `models/__init__.py`,
  `analysis/{attribution,auto_threshold,edge_stability,graph_posterior,threshold_sweep}.py`,
  `scripts/{ablation_sweep,recovery_gate}.py`, plus 12 legacy-referencing test files. Step-1 gate still applies.
