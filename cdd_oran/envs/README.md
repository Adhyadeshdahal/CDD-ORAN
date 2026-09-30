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

## 2. E-SERIES: E1-E5 (abstract structural worlds, closed) and E6 / E6-P (live)

Full map: `docs/ARCHITECTURE.md` sections 0-2.

### 2a. E1-E5: `v2/` (no longer developed; kept for reproduction)

Env classes: `v2/base.py` (`V2Env`), `v2/e1.py`..`v2/e5.py`. Pipelines: `cdd_oran/e1slice/`, `cdd_oran/e2slice/`.
Gates: `scripts/e{2,3,4}_*_gate.py`. Docs & contracts: `docs/benchmark/` (`SEMANTICS.md`, `SPEC.md`, `GATES.md`,
`E2_DISCOVERY_PROTOCOL.md`, `GATE_CONTRACT_E{2,3,4,5}.md`). Discovery method: MSCR (`cdd_oran/discovery/mscr.py`).

| Env | Scientific question | Status |
|-----|---------------------|--------|
| **E1** | Clean sanity: recover an identifiable graph + correct predictions | Done (recovery gate passed) |
| **E2** | Shared-control conflict: can complete causal fan-out prevent a harmful shared-knob action? | Closed; decision gate = recorded NULL (frozen) |
| **E3** | Temporal decision: avoid immediate reward that creates larger delayed harm | Decision DEV only (2026-09-24/25, not pre-registered; `docs/ARCHITECTURE.md` section 8); no `e3slice` pipeline |
| **E4** | Confounded decision: recover the right intervention when association has the wrong sign | Scaffolded only (`v2/e4.py` + `scripts/e4_structural_gate.py`); no `e4slice` pipeline; superseded by E6 |
| **E5** | Integrated stress: E2-E4 mechanisms + observation noise | Frozen structure-only (`GATE_CONTRACT_E5.md`) |

### 2b. E6 / E6-P: `e6/` (live, since 2026-09-26)

System-level near-RT RIC plant with multiple xApps and a WG3 conflict arbiter; E6-P (`E6PConfig`) adds the power /
protected-slice extension used by the discovery and option (a) studies. Discovery method: **PMRT**
(`cdd_oran/decision/pmrt.py`; `docs/benchmark/METHOD_NAMES.md`). Protocols: `docs/benchmark/E6P_*.md`.

### 2c. `xtruce/`

Re-implementation of the xTRUCE plant (`docs/benchmark/XTRUCE_SIM_SPEC.md`); screen done, kept as a cross-check plant.

## Firewall note (as of 2026-09-05)

The E-series discovery modules import **no** environment ground truth (verified by tests such as
`test_e2slice_discovery.py`). One remaining *import-time* seam still causes the E-series evaluators to
transitively load the legacy `env_i..iv` (via `analysis/threshold_sweep.py` → `recovery_metrics.py`);
decoupling that seam is the active quarantine task. See
`reports/2026-09-05-planner-redesign-and-env-boundary/` for the full boundary map.
