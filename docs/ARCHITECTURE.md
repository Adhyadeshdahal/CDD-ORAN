# CDD-ORAN: Architecture and Project State (single source of truth)

**Status: 2026-09-30.** Updated from the code on branch `feat/v2` (PMRT rename included) and the committed documents
listed below. When memory or an older doc disagrees with the code, the code wins; then fix this doc and re-date it.

**Goal:** use causal discovery for conflict mitigation among xApps in O-RAN. Everything below serves that goal.

## 0. Where the project is (short version)

The work has two eras that share one package:

1. **E1-E5 (abstract structural benchmarks, `cdd_oran/envs/v2/`).** Discovery with **MSCR**
   (`cdd_oran/discovery/mscr.py`) on randomized designs, then a learned world model and planners. These results are
   closed; their code is kept and still tested.
2. **E6 / E6-P (system-level near-RT RIC simulator, `cdd_oran/envs/e6/`), since 2026-09-26.** Multi-xApp conflicts in
   a 3GPP-grounded plant, a WG3-style arbiter interface, randomized unit-level logging, and design-based causal
   discovery. The current discovery method is **PMRT** (Predictable Matched-Filter Randomization Test,
   `cdd_oran/decision/pmrt.py` + `cdd_oran/decision/fdr_layer.py`). The current decision-side object is the
   map-driven referee **MapGateV2** (`cdd_oran/decision/mapgate.py`).

Open at the time of writing: the discovery v4 fresh-seed verdict (frozen protocol, analysis run pending), the
Kaggle full-data equivalence rerun for the PMRT rename, and the option (a) confounded-logs study (protocol drafted,
not frozen). No outcome of those runs is recorded here.

Naming rule (`docs/benchmark/METHOD_NAMES.md`): "MSCR" means only the Max-Stratified Correlation-Ratio test of the
E2-E5 studies. PMRT was labelled "MSCR+" in the frozen protocol v4 and in the frozen artifact
`E6P_MSCRPLUS_V4_FROZEN.json`; frozen files keep that old label and are never edited.

## 1. Package layout and module responsibilities

### 1.1 Environments (`cdd_oran/envs/`)

| path | what it is | status |
|---|---|---|
| `envs/legacy/` (`env_i.py`..`env_iv.py`, `base.py`, `statistics.py`, `stats_cache.py`) | Environment I-IV of the conference project | **Legacy.** Quarantined 2026-09-22; import explicitly via `cdd_oran.envs.legacy.get_env`. Used only by the legacy CDL/MLP trainer and legacy planners. |
| `envs/v2/` (`base.py`, `e1.py`..`e5.py`) | E1-E5 abstract structural worlds (coordinate-keyed exogenous tape, `docs/benchmark/SEMANTICS.md`) | Closed studies; kept for reproduction. E5 frozen structure-only (`docs/benchmark/GATE_CONTRACT_E5.md`). |
| `envs/e6/` | E6 system-level RIC plant (below) | Live. |
| `envs/xtruce/` | Re-implementation of the xTRUCE plant (arXiv:2608.28532; 4 cells / 20 UEs, Direct / Clipping / xTRUCE arbiters); spec `docs/benchmark/XTRUCE_SIM_SPEC.md` | Screen done (XTS-v1 verdict: not screenable); kept as a cross-check plant. |

E6 modules:

- `sim.py`: plant. 40 ms ticks, 25 per control second. Hidden per-site log-OU load, mobility, traffic, TR 38.901
  radio with load-coupled interference, handover / RLF, slice-aware scheduler, queues, EARTH-style energy model,
  KPM counters. Stress scenarios (`SurgeScenario`, `MistuneScenario`; `docs/benchmark/E6_STRESS_SCENARIOS.md`).
- `geometry.py`: 7-site hex layout with wrap-around, macro sectors plus co-channel picos, per-seed gain maps.
- `ric.py`: near-RT RIC layer: KPM reporting (granularity, delay, drops), knob registry with hard actuator limits,
  request plumbing (accept / reject / modify / defer, NACK), last-writer-wins default.
- `env.py`: episode loop. Two-phase step (`step_propose` / `step_apply`), WG3 lock and rollback, churn cap,
  `env.copy()` (bit-exact mid-episode copies, used for lookahead rollouts and knockout labels), optional trace.
- `config.py`: parameter registry with provenance tags ([S] sourced, [A] assumption, [V] to verify); `E6PConfig`
  (default off) adds the E6-P power / protected-slice extension.
- `xapps.py`: v1 rule-based xApps (MRO, TS, ES, SliceSLA) with per-seed implementation variants.
  `xapps_v2.py`: v2 stack (MLB, MROv2), drafted after the gate-A kill of v1, never run as a gate (spec
  `docs/benchmark/E6_V2_XAPPS_SPEC.md`, not frozen).
- `xapps_p.py`: E6-P xApps `ESPico`, `PowerES`, `Coverage`, `SliceGuarantee`.
- `baselines.py`: obs-only arbiters: freeze, priority, knob lock, cell-priority lock, subsets, static / tuned-static,
  `SMORestore`.
- `published.py`: re-implementations of published conflict methods as obs-only arbiters (QACM, CMF, PACIFISTA,
  Djidjev; Sharma / two-tower / Graphica interface-only). Spec `docs/benchmark/E6_PUBLISHED_BASELINES.md`.

### 1.2 Discovery (MSCR, E2-E5): `cdd_oran/discovery/`

- `mscr.py`: MSCR-v2. Max over single conditioners of a stratified correlation ratio, B = 2999 row permutations,
  per-target Benjamini-Yekutieli (q = 0.05) over parameter candidates. Valid only for i.i.d. randomized designs.
- `mscr_torch.py`: CUDA / torch permutation bank, bit-identical to the numpy bank (`MSCR_DEVICE`,
  `MSCR_NUM_THREADS`).
- Earlier frozen methods: `cdd_oran/e1slice/` (E1 partial correlation, v1 and v2), `cdd_oran/e2slice/` (E2 RCoT v1
  and v2).

### 1.3 Decision stack and E6 discovery: `cdd_oran/decision/`

Legacy-free (no import of `envs/legacy`). Grouped by role:

**E6 arbiter search and learned policy-effect model (2026-09-27/28; the step-5 ranking test failed, kept for
reproduction):**
`plans.py` (per-region plans, `half` = halve the slew rate), `world_model.py` (`WorldModel` protocol, `TrueSimWM`
privileged reference), `arbiter.py` (`WG3Arbiter`: budgeted receding-horizon plan search), `adapters/e6.py` (region
map: 7 macro sites + 3 picos), `trace.py` (`e6-trace/1` per-second trace), `collect.py` (randomized joint-policy
collectors v1-v3, contract `docs/benchmark/E6_COLLECTION_CONTRACT.md`), `effect_model.py` (`PolicyEffectModel`,
`EffectWM`, support rule, EB shrinkage), `gate.py` (selection-aware conformal gate), `rank_eval.py` (oracle-panel
ranking evaluation), `probe.py` (operator DEV probe campaign, `e6-probe/3`), `features.py` (cell-exchangeable panels,
`build_panel_p` for E6-P), `discovery.py` (template discovery with MSCR + conflict map on panels).

**E6-P unit-level infrastructure (2026-09-29/30):**
- `units_p.py`: decision unit = (cell c, xApp x, t0); opens at x's first request on c, mode held for T = 60 s;
  modes accept / half / reject (+ rollback for PowerES, SliceGuarantee). `UnitArbiter` reads only `obs`.
- `collect_p.py`: logging policies and the KPI tap. `RandomizedUnitPolicy` (high / low eps regimes),
  `PI0_HIGH_NO_RB` (v1-v3 discovery), `PI0_V4` (accept .5 / reject .5, v4), `PlaceboPolicy` (modes drawn and logged,
  accept applied: a sharp null), `IncumbentPolicy` / `PlaceboIncumbent` (option (a) context-dependent confounded
  logging), `CellKPITap` (privileged per-cell per-second series), `run_collection`, record codec `enc` / `dec`.
- `slots_p.py`: fixed-slot units (`SlotArbiter`), an option for fresh collection; not used by any frozen protocol.
- `labels_p.py`: paired lookahead labels via `env.copy(reseed=k)`; `gt_p.py`: CRN knockout ground truth
  (accept vs reject per unit, three reseeds) and the GT classification (TRUE / NULL / INDET).
- `ope.py`: IPS / SNIPS / DR off-policy value estimates with episode-cluster bootstrap.
- `referee_p.py`: step-2 regime-referee pieces (K-L0), energy-saving request classification, honest tree, knapsack.

**E6-P discovery methods:** see section 3. `crt.py`, `crt_units.py`, `crt_units_v2.py`, `pmrt.py`, `fdr_layer.py`,
`eprocess_units.py`, `baselines_disc.py`, `edge_score.py`, `disc_bench.py`. Deprecated shims:
`crt_units_plus.py` -> `pmrt.py`, `mscr_multi.py` -> `fdr_layer.py` (same objects; old version strings kept as
`LEGACY_VERSION`).

**Referee:** `mapgate.py` (`MapGate` v1, `MapGateV2`, `DirectionalUnitArbiter`, map builders and controls).

### 1.4 E2-E5 decision path and legacy model code

- `cdd_oran/benchmark/`: `masked_world_model.py` (true equations restricted to a discovered graph; not deployable),
  `learned_world_model.py` (per-KPI MLP ensembles on discovered parents), `rollout.py` (open-loop rollout kernel).
- `cdd_oran/planners/sequence.py`: V2-native planners (`exhaustive_fh`, `greedy_fm`, `qacm_v2`, `cem_sequence`,
  `mppi_sequence`). Used by the E2/E3 decision studies; nothing in `planners/` talks to E6.
- **Legacy (conference project):** `planners/{cem,mcts,mppi,qacm,horizon,joint,ensemble,cost,base}.py` (torch
  planners on the Env I-IV API; loaded lazily by `planners/__init__.py` so V2 users do not import legacy),
  `cdd_oran/models/` (`cdl.py`, `mlp.py`), `cdd_oran/experiments/` (train / discover / evaluate / sweep / viz on
  `envs.legacy`), `cdd_oran/cli.py`, root `cli.py`, `configs/env_*_{cdl,mlp}.yaml`, `cdd_oran/policies/`. Kept for
  archival reproduction; retirement plan `plans/012-legacy-env-retirement.md`.
- `cdd_oran/analysis/`: metrics shared by E1-E5 (recovery, regret `v2_regret.py`, thresholds, posterior, stats).
  `cdd_oran/viz/`, `cdd_oran/utils/` (seeding, run dirs, logging, sweeps).

### 1.5 Scripts and drivers

- `scripts/`: E1-E5 gates, spines and baselines (`e2_spine.py`, `e5_spine.py`, `e5_structure_gate.py`,
  `stage0_falsifiers.py`, ...).
- `scratchpad/e6_dev/`: E6 / E6-P drivers, analyzers and cloud tooling (tracked; "DEV scratch" by name, but every
  frozen protocol's driver lives here). Main files:
  - screens: `e6p_screen.py` (+ `e6p_stage{0a,0b,0c,1,2,3}.py` wrappers), `e6p_v2.py` (O_tape falsifier),
    `xtruce_screen.py`, `gate_a*.py`, `gate_b*.py`, `oracle.py`, `wg3_oracle.py`;
  - discovery collection: `e6p_discovery.py` (stages dev / eval / placebo / gt / prof and the v4 stages; one wrapper
    per stage, e.g. `e6p_disc_eval_v4.py`);
  - discovery analysis: `e6p_disc_analyze.py` (v1), `e6p_disc_analyze_v2.py` (v2, v3), `e6p_disc_analyze_v4.py` (v4);
  - PMRT research and artifacts: `disc_bench_run.py`, `pmrt_arms_bench.py`, `pmrt_bench.py`, `fdr_layer_bench.py`,
    `pmrt_artifacts.py`, `pmrt_equivalence.py`;
  - step 2 and option (a): `e6p_referee_kl0.py`, `e6p_step2_dev.py`, `e6p_opta_ka.py`, `e6p_opta_ka2.py`,
    `e6p_opta_kb.py` (+ `_run` / `_analyze`), `e6p_conf.py` (study driver; stage wrappers `e6p_conf_{dev,disc,
    placebo,gt,eval}.py`), `e6p_conf_analyze.py`;
  - cloud: `cloud.py`, `kaggle_run.py`, `kaggle_job.py`, `colab_run.py`, `lightning_run.py` (section 6).
- `scratchpad/decision_stack/`: design documents and grids of the E6 learned-arbiter phase (`DESIGN.md`,
  `collect_grid.py`, `probe_grid.py`, `rank_grid.py`, `req_pilot/`).

## 2. E6-P environment and data pipeline

### 2.1 Plant and cell under study

E6-P = E6 with `E6PConfig` on. Screen pairs (`docs/benchmark/E6P_SCREEN_PROTOCOL.md`): P1 = ES x SliceGuarantee,
P2 = PowerES x Coverage (not screenable), P3 = ES + PowerES x SliceGuarantee (three-way indirect conflict). Every
discovery and referee study uses cell **P3 surge-L40**: `e6p_screen.make_cfg("P3", 3, seed, lf)` with the calibrated
L40 load factor from `scratchpad/e6_dev/e6p_state.json`, 120 s warm-up + 600 s scored = 720 s, run as
`E6Env(cfg, log=False, wg3=True, trace=True)`.

Knob families (one writer each): `carrier` and `sleep` (ES), `ptx` (PowerES), `prot_min` (SliceGuarantee).
KPIs per cell: `pv` (protected violated UE-s), `v` (all-UE violated UE-s), `e` (energy J), `rlf`, `load` (UE-s
served). Relations: `own` (the unit's cell), `nbr` (its obs-only exposure set minus itself), `far` (the rest).
Hypothesis space: 4 families x 3 relations x 5 KPIs = **60 hypotheses** (`crt_units.HYPOTHESES`), oriented "dir"
(effect of a knob-value increase). The chain of interest: pico sleep -> nbr load (+) -> nbr pv (+).

### 2.2 Collection stages

`collect_p.run_collection` wraps `units_p.UnitArbiter` around a logging policy; the policy draws each unit's mode
from `default_rng([seed, 6612, c, x_idx, t0])` and logs the propensity row. Stage roles
(`scratchpad/e6_dev/e6p_discovery.py`):

| stage | policy | role |
|---|---|---|
| dev | pi0 | development, allowed before a freeze |
| eval | pi0 | discovery evaluation (frozen protocol required) |
| placebo | `PlaceboPolicy` | K0 null check (logged modes, accept applied) |
| gt | pi0 + knockout labels (`gt_p`) | ground truth by CRN accept-vs-reject rollouts |
| prof | one xApp accepted, others rejected | PACIFISTA-style profiles for a baseline |

v1-v3 used `PI0_HIGH_NO_RB` with an all-accept arbiter warm-up (no units in the first 120 s). v4 uses `PI0_V4` and
randomizes from t = 0 (`arb_warmup_s=0.0`, `count_all=True`); units with t0 < 90 are logged but not tested.
Option (a) logs with `IncumbentPolicy` (confounded, context-dependent) through `mapgate.DirectionalUnitArbiter`.

### 2.3 Records

JSONL, schema **`e6p-disc-rec/1`** (full contract in the `e6p_discovery.py` docstring). Line kinds `header`,
`episode`, `close`. Arrays are base64(zlib(float32)) dicts decoded by `collect_p.dec`. An episode record holds:
`key`, `pi0_table`, `obs_static`, `units` (per unit: c, x, knob, t0, logged `mode` and `p`, `applied_mode`, request
cur / prop / step, exposure set, obs-only `ctx`, n_req / n_applied, privileged `lab_kpi`), `panel` (obs-only
10 s panel, `e6p-panel/1`), and privileged `lab_series` (per-second per-cell pv / v / e / rlf / prb / ue / prot_ue),
`lab_outcome`, `gt_static`, `gt_labels`. Discovery code reads only obs-only fields plus `lab_series` as the outcome;
`gt_static`, `gt_labels` and `lab_outcome` are listed in `crt_units.PRIVILEGED_KEYS` and never read.

Cloud runs write `res_*.jsonl` per shard; `kaggle_run.py pull` merges them into
`scratchpad/e6_dev/runs/<NAME>/all.jsonl`.

### 2.4 Caches

`disc_bench.build_cache` streams records (one episode at a time) into one compressed npz per source file: the unit
table (mode, p, propensity row, step, sgn, family), exposure masks, per-cell KPI sums `post_H` / `pre_H` for
H = 30, 60, 90, 150 (float64, bit-identical to `crt_units.build_unit_data`), ctx, and episode metadata. `load_pool`
rebuilds a `crt_units.UnitData` for any episode subset, and `pmrt.pmrt_data` builds PMRT's time-binned features from
the same cache. The v4 and option (a) analyzers build caches in `--cache-dir` (default `OUT/cache`); on Kaggle the
caches stay in the kernel output and later jobs attach the kernel as a source (`kaggle_job.py pull` skips `*.npz` by
default).

## 3. Discovery methods on E6-P

All E6-P methods test the sharp null "the KPI series of (rel, kpi) is invariant to family f's assignment draws",
using the logged propensities (design-based), never row permutation. Row-permutation MSCR is invalid on E6 traces
(serial dependence, spatial coupling, xApp feedback; `scratchpad/decision_stack/REPORT_H.md`).

| method | module | version string | status |
|---|---|---|---|
| MSCR-CRT (probe blocks) | `crt.py` | `mscr-crt-v1` | MSCR statistic with a design-based null on the probe campaign; superseded |
| MSCR-CRT v1 (units) | `crt_units.py` | `mscr-crt-units-v1` | protocol v1; KILL (indirect recall 3/8, sign error from pooling request directions) |
| "MSCR-CRT v2" | `crt_units_v2.py` | `mscr-crt-units-v2` | protocols v2 (KILL, near miss) and v3 (PARTIAL). Despite the name its statistic is a design-centred linear score with episode x sgn fixed effects, not MSCR. Only approximately valid (see below) |
| **PMRT** | `pmrt.py` + `fdr_layer.py` | `pmrt-v1` + `fdr-layer-v1` (legacy `mscr-crt-units-plus-v0` + `mscr-multi-v1`) | protocol v4 method (frozen artifact); primary arm `loadsp_c` + `wby1s` |
| e-process | `eprocess_units.py` | `mscr-eproc-units-v1` | exact companion test; research, not frozen |
| baselines | `baselines_disc.py` | - | corr, Granger (+BY), SHAP-GBDT, two-tower, INT, QACM-style, MSCR row-perm, PACIFISTA native; declarations uncalibrated on placebo |

**Validity finding (agent V, after v3).** The unit skeleton (which units exist, their t0 and request direction)
depends on the family's own past modes: a rejected request is re-proposed about 60 s later. A CRT that redraws modes
i.i.d. on the realised skeleton and demeans over the realised unit set (v2's fixed effects) is not exact; on null
outcomes v2 rejected sleep hypotheses at .143 (nominal .05). The placebo cannot detect this because it applies accept.

**PMRT** (`docs/benchmark/METHOD_NAMES.md`; `pmrt.py` docstring):
- assignment score v_u = sgn_u (L(mode_u) - E_pi0 L) from the unit's logged propensity row
  (`crt_units_v2.design_regressor`);
- predictable adjustment: ridge prediction of the post-window slots x time bins from pre-window KPI bins, ctx and
  past-only mode history, plus running (past-only) stratum centres; no statistic uses later units or realised-set
  means;
- matched filter K = Sigma^-1 mu over receiving-cell slots x time bins (Ledoit-Wolf Sigma), sign-constrained spatial
  profile (`loadsp`), predictable Huber clip (`_c`);
- inference: CRT redraw of the family's modes from the logged pi0, other families fixed, B = 9999, RNG
  `[0, 6616, 3, family_idx, split]`. Asymptotically valid by the martingale CLT, not exact;
- every learned piece was fitted on ev2 (480 eps) + DEV (20 eps) and is frozen in a pure-JSON artifact (section 5).

**Declaration layer** (`fdr_layer.py`): statistic-agnostic multiplicity over the 60 hypotheses, valid under arbitrary
dependence provided weights / directions are fixed from independent data: `by`, `wby` (weighted BY,
Roeder-Wasserman weights), `by1s` / `wby1s` (one-sided in the prior direction where |z_prior| >= 3), `dagger` /
`dagger1s`, `gholm`, `pfilter_w1s`, `ebh` / `ebh1s` / `webh1s`. `bh_invalid` is reference only. The prior is the
same statistic's signed z on ev2.

**e-process** (`eprocess_units.py`): M_n = prod (1 + lambda_u v_u s_u) with predictable bets; a test martingale under
the sharp null whatever the skeleton does, so exact (Ville), combined across hypotheses with e-BH. Weaker at small n.

**Scoring and bench.** `edge_score.py` scores declarations against the GT reference (`gt_p.summarize`, "dir"):
chain hits, premise edge (sleep -> nbr pv declared +), indirect recall / precision / F1, sign accuracy, far
declarations, P1 / P2 checks. `disc_bench.py` runs any method on disjoint small episode subsets (n = 60 / 120 /
300) of a cache pool plus placebo checks; PMRT selection was done this way on the ev3 records, which the v4 protocol
discloses as selection data.

**Discovery attempt history** (protocols in `docs/benchmark/`):

| protocol | method | verdict |
|---|---|---|
| `E6P_DISCOVERY_PROTOCOL.md` (v1) | MSCR-CRT v1 | KILL |
| `E6P_DISCOVERY_PROTOCOL_V2.md` | MSCR-CRT v2 | KILL (near miss: premise z 2.67 not declared) |
| `E6P_DISCOVERY_PROTOCOL_V3.md` | MSCR-CRT v2, n = 1200 | PARTIAL (P1 pass, P2 vs baselines fail); approximate validity only |
| `E6P_DISCOVERY_PROTOCOL_V4.md` + `E6P_DISCOVERY_PROTOCOL_V4_ADDENDUM_PMRT.md` | PMRT (labelled "MSCR+" in the frozen doc) on 60 / 120 slices of 600 fresh episodes, fresh GT and placebo | frozen 4fc2cd9; verdict pending |

Results of v1-v3 and step-2 DEV: `scratchpad/e6_dev/decision/STEP1_V{1,2,3}_RESULT.md`,
`reports/2026-09-30-step1-discovery-and-step2-dev/`.

## 4. Decision / referee side

- **Screens and oracles (closed).** Gate A v2 screen on P1 / P3 (`e6p_screen.py`; verdict DEAD), O_tape falsifier
  (`e6p_v2.py`; P3 surge-L40 NOISE STOP, oracle sign reliability .676 < .70), xTRUCE screen (not screenable).
  Reports: `reports/2026-09-29-e6p-screen-xtruce-negative/`, `reports/2026-09-29-e6p-v2-otape-and-status/`.
- **Incumbent and reference policies.** Accept-all, freeze, static priority / cell-priority lock, hindsight static
  subsets (`envs/e6/baselines.py`); published methods (`envs/e6/published.py`); static rules such as never-sleep and
  blanket saving maps (`mapgate.blanket_saving_map`) used as referee arms; `collect_p.IncumbentPolicy`, a
  context-dependent logging policy (accept probability by request class and PRB pressure, floored so every unit has
  positive propensity), which produces the confounded logs of option (a).
- **Step-2 regime referee** (`referee_p.py`, `e6p_referee_kl0.py`, `e6p_step2_dev.py`): K-L0 KILL; DEV arms found no
  eligible map-driven gain (`scratchpad/e6_dev/decision/STEP2_DEV_RESULT.md`).
- **MapGate** (`mapgate.py`). A `units_p.UnitArbiter` policy that reads only `unit["ctx"]` and a causal map
  M = {(family, rel, kpi): beta}. `MapGateV2` (option (a) DEV iteration, rule fixed before any v2 run): total signed
  effect per KPI over own / nbr / far, KPI priority (pv, v, rlf), energy class from the `e` edges, defer iff harmful
  and (costly or pressure > theta = .05), guard-conflict duty bound k_conf = 1, and a directional arbiter
  (`DirectionalUnitArbiter`) that defers only the opening direction. Controls: `map_from_gt`, `own_only`,
  `sign_flip`, `random_sized_map`, `blanket_saving_map`.
- **Option (a) study** (`docs/benchmark/E6P_CONFOUNDED_PROTOCOL.md`, FROZEN: no). Question: from confounded incumbent
  logs, does PMRT give maps that drive MapGateV2 safely, where associational baselines give wrong maps? Kill tests
  K-A, K-A2, K-B are summarized in `scratchpad/e6_dev/decision/opta_*.json` and the protocol's section 2. Driver
  `scratchpad/e6_dev/e6p_conf.py`, analyzer `scratchpad/e6_dev/e6p_conf_analyze.py` (stages disc / placebo / gt /
  eval; two freezes: the protocol, then the maps artifact `docs/benchmark/artifacts/E6P_CONF_MAPS.json`, which does
  not exist yet).

## 5. Protocol and freeze discipline

- **Frozen documents.** A protocol is frozen by setting "FROZEN: yes" and recording its LF-normalised sha256 in the
  freeze commit message, in `.tmp/PLAN.md` (gitignored living plan) and in the driver. Drivers refuse to run
  confirmatory stages when the doc's sha256 differs:
  `e6p_screen.FROZEN_SHA256` (E6P_SPEC + screen protocol), `e6p_v2.FROZEN_SHA256`, `xtruce_screen.FROZEN_SHA256`,
  `e6p_discovery.FROZEN_SHA256` (v1) and `FROZEN_SHA256_V4`, `e6p_conf.FROZEN_SHA256_CONF` (None until freeze 1).
  Frozen files are never edited; changes go into addenda or a new protocol version with fresh seeds.
- **Artifacts with sha256 pins** (`docs/benchmark/artifacts/`): `E6P_MSCRPLUS_V4_FROZEN.json` (frozen v4 artifact,
  sha in the `.sha256` file and in `e6p_disc_analyze_v4.ARTIFACT_SHA256`), `E6P_PMRT_V4.json` (label-only successor,
  `PMRT_ARTIFACT_SHA256`, records `supersedes` / `legacy_code_sha256`), `E6P_PMRT_V4_EQUIV_LOCAL.json` (local
  equivalence result). Each artifact embeds the sha256 of the code files it depends on; analyzers refuse on a sha or
  code mismatch. The v4 verdict of record is computed by the frozen code at 4fc2cd9; the renamed code is checked
  against it by `pmrt_equivalence.py` (local: byte-identical; full-data rerun pending). Older frozen artifacts:
  `docs/benchmark/plan003_frozen_artifacts/`, `plan004_envelope_digest/`, `plan006_recovery_digest/`.
- **Seed registry** (`docs/benchmark/SEED_REGISTRY.json`): used seed ranges per world, reserved blocks and RNG stream
  tags. E6 blocks include DEV 0-30, screen 150000-150399, O_tape 155000-155199 (155200-155399 confirmation reserved),
  discovery 180000-183999, referee 184000-185999, option (a) 186000-187999, discovery v4 188000-189999; xTRUCE
  190000-190399. Tags: 6612 (pi0 draws), 6613 (label sampling), 6616 (CRT draws), 6617 (GT bootstrap), 6622-6624
  (option (a)). Drivers assert their seeds lie in a registered block and outside forbidden ones.
- **Selection disclosure.** Every protocol lists what was tuned on which data (e.g. v4 section 0: PMRT arms chosen on
  ev3 and the pooled GT). Claims are scored only on fresh seeds.
- **Numerics.** numpy 2.4.2 / scipy 1.18.1 pinned (`pyproject.toml`, `uv.lock`). Windows (UCRT) and Linux (glibc)
  libm differ in the last bit and trajectories fork, so comparisons never mix platforms; confirmatory stages run on
  Kaggle Linux only.

## 6. Cloud execution

All heavy compute runs in the cloud; local processes stay light.

- `scratchpad/e6_dev/cloud.py`: builds a bundle zip (`cdd_oran` E6 / xTRUCE / decision code, `e6dev/*.py`, protocol
  docs, `SEED_REGISTRY.json`, `docs/benchmark/artifacts/*` for discovery / option (a) scripts, `MANIFEST.json` with
  git HEAD, per-file sha256, local versions and numeric fingerprint) plus private Kaggle script kernels, 4 shards per
  kernel, one single-threaded process per shard. `PIN=match|strict|off` controls numpy / scipy pinning; each kernel
  writes `startup.json` with `fp_match`.
- `kaggle_run.py launch|status|pull NAME SCRIPT KERNELS`: grid runs (stage wrappers that take `run --part i/k --out`);
  pulls to `scratchpad/e6_dev/runs/NAME/all.jsonl`. NAME must be new per launch. Pass bare script names.
- `kaggle_job.py launch|status|pull NAME --cmd ... [--sources owner/kernel,...]`: one-shot analysis jobs; earlier
  kernels' outputs are attached as sources (e.g. the v4 analysis reads the v4 collection kernels). The account's
  5-concurrent-CPU-session limit is handled by retrying.
- `colab_run.py`: Colab counterpart (grid `launch` and one-shot `job` mode), driven by a Windows scheduled task
  (`tick`) instead of a long-lived laptop process. Sessions can be reclaimed; lost runs are relaunched under a new
  NAME. Preferred only for short jobs.
- `lightning_run.py`: Lightning AI Studio runner (CPU only; stop the Studio after pull). Used once (gate B stress).

## 7. Where results and reports live

- `reports/<date>-<topic>/`: archived, tracked reports with figures (`index.html`, `figures_data.json`). Latest:
  `reports/2026-09-30-step1-discovery-and-step2-dev/`.
- `scratchpad/e6_dev/decision/`: plans, verdict notes and summaries (`STEP1_MSCR_PLAN.md`, `STEP1_V*_RESULT.md`,
  `STEP2_*`, `OPTION_A_PLAN.md`, `opta_*.json`, morning briefs, reviews).
- `scratchpad/e6_dev/runs/<NAME>/`: pulled cloud outputs (partly tracked).
- `scratchpad/e6_dev/HEADROOM_RESULT.md`, `GATE_A_RESULT.txt`: E6 v1 headroom and gate results.
- `docs/benchmark/`: specs, protocols, contracts, method names, seed registry, frozen artifacts, and E1-E5 results
  (`E5_STRUCTURE_RESULT.md`, `STAGE0_RESULT.md`, `DECISION_STUDY_D1_RESULT.md`, ...).
- `plans/`: numbered plans of the E1-E5 era (`plans/README.md`); `.tmp/PLAN.md` is the untracked working plan.

## 8. E2-E5 results (closed; still valid for their setting)

Pipeline of that era: randomized data -> MSCR graph -> per-KPI models on graph parents -> planner. The masked true
simulator (`cdd_oran/benchmark/masked_world_model.py`) isolates discovery errors but is not deployable; the learned
world model (`learned_world_model.py`) is the deployable path.

- MSCR-v2, E2, 100 fresh seeds: family FDR 0.008 (UB 0.012), gated P0->K5 100/100, param-edge recall 0.945
  (`scratchpad/p0k5_fp_calibration/VERDICT_v2.md`, main repo scratch; not in this tree).
- E5 noiseless core, 100 fresh seeds: harmful edge 100/100, FDR 0.022 (UB 0.032), recall 1.00. SHAP at tau <= 0.02
  matches MSCR on this corpus (`docs/benchmark/E5_STRUCTURE_RESULT.md`, `docs/benchmark/STAGE0_RESULT.md`).
- Stage 0: on bounded setpoint + dither data MSCR's family FDR is 0.54-0.67 (row-permutation null ignores block
  structure). This is the same failure that ruled row-permutation MSCR out on E6.
- Decision DEV (2026-09-24/25, not pre-registered): E2 learned K5 model cost 0.04-0.13 at n = 24k; E3 CEM / MPPI on
  the MSCR graph near-optimal, QACM (myopic) regret 0.118; E5 makes no decision claim.
- Thread B (closed 2026-09-22): learned action-effect estimators failed off-gate at low occupancy.

## 9. Known inconsistencies and gaps

- `cdd_oran/envs/README.md` predates E6: it calls E2 the "current frontier", E5 "not built", and does not mention
  `envs/e6/` or `envs/xtruce/`.
- `cdd_oran/decision/__init__.py` docstring lists only `plans`, `world_model`, `arbiter`, `adapters.e6`.
- Version strings keep historical MSCR labels where they are part of RNG or result provenance:
  `mscr-crt-units-v2` (not an MSCR statistic), `mscr-eproc-units-v1`, and PMRT's `LEGACY_VERSION`
  `mscr-crt-units-plus-v0`. `e6p_discovery.py` is titled "E6-P MSCR causal discovery" (its v1 plan name).
- The v4 analyzer's default `--artifact` is the frozen `E6P_MSCRPLUS_V4_FROZEN.json`; the option (a) analyzer's is
  `E6P_PMRT_V4.json`. Both are intended.
- `cloud.py` bundles an option (a) maps artifact that exists only after freeze 2.
- No real O-RAN grounding beyond the simulator (documented gap); KPI->KPI discovery on E6 is not attempted.

## Anti-drift protocol

- This doc is canonical for architecture. Verify against code before asserting; code wins ties; then update this doc
  and date it.
- Method names follow `docs/benchmark/METHOD_NAMES.md`.
