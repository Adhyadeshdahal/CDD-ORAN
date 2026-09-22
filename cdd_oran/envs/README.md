# `cdd_oran/envs/` — two environment systems (read this first)

There are **two** environment numbering systems in this repo. They are easy to confuse; this note
exists so we never conflate them again.

## 1. LEGACY (v1) — Environment I / II / III / IV — RETIRED

Files: **all under `legacy/`** (quarantined 2026-09-22) — `legacy/base.py` (`BaseORANEnv`,
`XApp`/`Param`/`KPI`), `legacy/env_i.py`..`env_iv.py` (`ORANEnvironment1..4`), `legacy/statistics.py`,
`legacy/stats_cache.py`; factory `get_env()` in `legacy/__init__.py` — import it explicitly:
`from cdd_oran.envs.legacy import get_env`. Configs `configs/env_*_cdl.yaml` / `env_*_mlp.yaml`; results
under `runs/EnvironmentI..IV/`.

- The original v1 benchmark. **Implemented but retired.** All the historical objective / OOD /
  planner decision-value work (e.g. the −0.083→+0.303 numbers) was measured here.
- **Do not build new work on these.** They are kept only for archival and reproduction.
- Status: **quarantined into `cdd_oran/envs/legacy/` (2026-09-22).** Import paths are honest
  (`cdd_oran.envs.legacy.*`); `envs/__init__.py` loads no legacy. Full code removal is deferred (gated on
  porting the kept baselines — QACM et al. — to `V2Env`; see `plans/012`, `plans/014`).

## 2. E-SERIES redesign — E1 / E2 / E3 / E4 / E5 — the LIVE benchmark

Env classes: `v2/base.py` (`V2Env`), `v2/e1.py`..`v2/e4.py` (`E1V2Env`..`E4V2Env`).
Pipelines: `cdd_oran/e1slice/`, `cdd_oran/e2slice/`. Gates: `scripts/e{2,3,4}_*_gate.py`.
Docs & contracts: `docs/benchmark/` (`SEMANTICS.md`, `SPEC.md`, `GATES.md`,
`E2_DISCOVERY_PROTOCOL.md`, `GATE_CONTRACT_E{2,3,4}.md`), plans `plans/006`, `plans/007`.

| Env | Scientific question | Build status |
|-----|---------------------|--------------|
| **E1** | Clean sanity — recover an identifiable graph + correct predictions | **DONE** (recovery gate passed via v2 partial-correlation discovery) |
| **E2** | Shared-control conflict — can complete causal fan-out prevent a harmful shared-knob action? | **Current frontier.** Env built; discovery method banked + tested (`e2slice`); the discovery **run is parked** (~5–18 d on a 16 GiB box — needs a bigger machine). Decision gate = recorded NULL (frozen). |
| **E3** | Temporal decision — avoid immediate reward that creates larger delayed harm | **Scaffolded** — `v2/e3.py` + `scripts/e3_*_gate.py` exist; no `e3slice` discovery/decision pipeline yet |
| **E4** | Confounded decision — recover the right intervention when association has the wrong sign | **Scaffolded** — `v2/e4.py` + `scripts/e4_structural_gate.py` exist; no `e4slice` yet |
| **E5** | Integrated stress — E2–E4 mechanisms combined + moderate observation noise | **Research-only** — design notes; not built |

## Firewall note

The E-series discovery modules import **no** environment ground truth (verified by tests such as
`test_e2slice_discovery.py`). One remaining *import-time* seam still causes the E-series evaluators to
transitively load the legacy `env_i..iv` (via `analysis/threshold_sweep.py` → `recovery_metrics.py`);
decoupling that seam is the active quarantine task. See
`reports/2026-09-05-planner-redesign-and-env-boundary/` for the full boundary map.
