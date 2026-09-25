# CDD-ORAN — Architecture & Project State (SINGLE SOURCE OF TRUTH)

**Status: 2026-09-25.** Updated from the code and the committed results listed below.
**Read this before trusting older plans/reports/memory. When memory or an old doc disagrees with the code, the code wins — then fix this doc.**

**Goal:** use **causal discovery** for **conflict mitigation among xApps in O-RAN**. That is the whole project; everything below serves it.

## The one-paragraph truth: the solution is MODEL-BASED, discovery is model-free

The pipeline has three stages, and only the first is model-free:

```
randomized data --MSCR (model-free CI test)--> graph --fit per-KPI models on graph parents--> LEARNED world model
                                                                                                    |
                                   planner(world_model, state, panel) -> action  <------------------+
```

1. **Discovery (model-free).** MSCR is a permutation conditional-independence test. It outputs a binary
   knob→KPI graph plus a p-value per edge, and predicts nothing. RCoT, pdCor and partial correlation are
   likewise tests that output graphs.
2. **World model.** Deciding needs predictions. Three things play this role, and they must not be confused:
   - **Benchmark oracle:** the true simulator. It is the yardstick (regret = `G_oracle − G_planner`), not deployable.
   - **Masked true simulator** (`cdd_oran/benchmark/masked_world_model.py`): the true equations restricted to
     the discovered graph. This was the old "model-free live spine". It is useful for isolating discovery
     errors but is **not deployable** (a real RAN gives no true equations).
   - **Learned world model:** per-KPI models fitted on the discovered parents. This is the **deployable**
     path. DEV evidence (below) says it now works on E2 and E3; it is not yet productized or pre-registered.
3. **Planner** (`cdd_oran/planners/sequence.py`): `(model_factory, snapshot, param_id, grid, H, R) → action
   sequence`. The same planner runs on any of the three world models, which separates "is the planner
   good?" from "is the model good enough?".

## What is built (code-verified)

- **Envs:** `cdd_oran/envs/v2/e{1,2,3,4,5}.py`. E2 = shared-knob trap (P0→{K0,K1,K2,K5}, gated harmful
  P0→K5). E3 = multi-step KPI→KPI chains. E4 = latent confounding. **E5 = composed benchmark, FROZEN
  STRUCTURE-ONLY** (`docs/benchmark/GATE_CONTRACT_E5.md`, `7ab3899`). Legacy Env I–IV modules live in
  `cdd_oran/envs/legacy/` until QACM's legacy coupling is removed (`plans/012`).
- **Discovery (canonical): MSCR-v2** (`cdd_oran/discovery/mscr.py`): max over single conditioners of a
  stratified correlation ratio, B=2999 permutations, per-target Benjamini–Yekutieli (q=0.05) over the
  **parameter** candidates. Evidence:
  - E2, 100 fresh seeds: family FDR 0.008 (UB 0.012), gated P0→K5 100/100, param-edge recall 0.945
    (`scratchpad/p0k5_fp_calibration/VERDICT_v2.md`).
  - E5, 100 fresh seeds: harmful edge 100/100, FDR 0.022 (UB 0.032), recall 1.00, while the SHAP-GBDT and
    two-tower proxies miss it 100/100 (`docs/benchmark/E5_STRUCTURE_RESULT.md`).
  - Scope: randomized designs only; known failure on single-parameter-actuation data (E1); KPI→KPI
    selection is exploratory (no FDR evidence). RCoT-v2 (E2) and partial correlation (E1) remain as frozen
    earlier methods.
- **Spine:** `scripts/e2_spine.py`, `scripts/e5_spine.py` load discovered masks into the masked world model
  and score decisions.
- **Planners on V2:** `sequence.py` has `exhaustive_fh`, `greedy_fm`, **`qacm_v2`** (QACM = exhaustive
  one-step conflict-knob choice under the locked cost; a market baseline, not "ours"), **`cem_sequence`**,
  **`mppi_sequence`** (`0fb64bd`). The legacy torch planners (`cem.py`, `mcts.py`, `mppi.py`, `qacm.py`)
  stay on the legacy API until retirement.

## Decision-phase DEV results (2026-09-24/25, `scratchpad/decision_phase/`, NOT pre-registered)

- **E2** (one knob, H=1, brute-force pick as the measuring stick): missing the harmful edge costs 15.1.
  A learned K5 model (MLP ensemble on MSCR parents) costs 0.04–0.13 at n=24k. With **every** KPI learned
  (lean training), the loss is 0.7–0.8 (about 95% of decision value). No-graph inputs are worse
  (0.9–1.3). Tree models fail (loss about 10) because K5 depends on P0+P7, a diagonal ridge; ensemble
  uncertainty does not flag those failures (bias, not variance).
- **E3** (H=3, exact optimum over 101³ sequences): with the true model, QACM (myopic) regret 0.118
  vs CEM/MPPI 0.000. With a learned model on the MSCR-discovered graph (which matched the true graph,
  KPI→KPI edges included), CEM/MPPI 0.0002 (31/32 optimal) and QACM 0.118. Without a graph, QACM rises to
  0.325.
- **E5 decision layer:** under the locked standardized scoring the decision contrast collapses and a
  pre-declared re-balance failed. The closest trap sits on a constraint boundary where under-estimating harm
  by 5% is costly. E5 makes **no decision claim**; this motivates a later cautious-planner study.

## Gaps

- The learned world model and the planners are not yet in a pre-registered decision study (next step).
- E3 separates myopic from look-ahead planners but not CEM from MPPI: a harder planning testbed (several
  knobs, longer horizons) is needed.
- KPI→KPI discovery has no FDR evidence; discovery under noise, confounding or observational data is untested.
- **No real O-RAN grounding** — abstract benchmark (documented gap).

## History that still matters

- **Thread B (closed 2026-09-22):** learned action-effect estimators (ridge/GCV, honest tree) failed off-gate
  at low occupancy on the injected B-bank design. The 2026-09-25 E2/E3 results use a different estimator (MLP
  ensembles on discovered parents) on the native mechanisms; they do not overturn Thread B for its setting.
- **Binary graphs:** in August the hard threshold was found harmful in the legacy CDL (a near-threshold edge
  became absent). In the current chain the graph only selects inputs and the learned model supplies
  strengths; the remaining risk is a weak true edge being dropped (e.g. P6→K5 at n=4000).

## Anti-drift protocol

- This doc is canonical. `memory/next-session-handoff` points here.
- Verify against code before asserting; code wins ties; then update this doc and date it.
- Keep ONE pipeline story: discovery → learned world model → planner.
