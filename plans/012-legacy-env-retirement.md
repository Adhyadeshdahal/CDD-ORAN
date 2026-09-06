# 012 — Legacy Env I–IV retirement → hard-delete (sequenced)

**Status:** TRACKED SEQUENCE. The E-series (E1–E5) is the live env system; legacy `Environment I–IV`
(`cdd_oran/envs/env_i.py … env_iv.py`) is already RETIRED + quarantined but **not yet deleted**. This file
records the remaining steps and the gate that must clear before deletion, so the cleanup is a plan item, not
a mental note. See memory `legacy-vs-eseries-boundary`.

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
