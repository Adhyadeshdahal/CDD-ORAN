# CDD-ORAN — Architecture & Project State (SINGLE SOURCE OF TRUTH)

**Status: 2026-09-22.** Code-verified by a three-lane investigation (plans, code, reports).
**Read this before trusting older plans/reports/memory. When memory or an old doc disagrees with the code, the code wins — then fix this doc.**

**Goal:** use **causal discovery** for **conflict mitigation among xApps in O-RAN**. That is the whole project; everything below serves it.

## The one-paragraph truth (resolves the model-free vs model-based confusion)

THREE distinct things have all been called "the world-model / the decision engine." Conflating them caused repeated circling:

1. **Benchmark oracle / scoring machinery** — true-SCM `do()`-propagation on the ground-truth simulator. The *yardstick* (regret = `G_oracle − G_planner`, `docs/benchmark/GATES.md`). Model-free (it *is* the simulator). Not deployable.
2. **Live E-series decision gates (what runs today in code)** — structural do-propagation / rollouts on the TRUE simulator, restricted to a given structure (`scripts/e2_decision_gate.py`, `e3_decision_gate.py` → `cdd_oran/benchmark/rollout.py`, `cdd_oran/planners/sequence.py`). **Model-free; no learned predictor.**
3. **Intended DEPLOYABLE engine (per plans) & the RETIRED legacy** — a **learned world-model fit** (model-based). Real O-RAN has no true SCM, so deployment needs a *fitted* model. The retired legacy Env I–IV pipeline (`cdd_oran/models/cdl.py` neural net + QACM/CEM/MCTS/MPPI) is this. **Thread A/B tested a learned surrogate and it FAILS off-gate at low occupancy — the deployability blocker.**

**Bottom line:** the LIVE benchmark pipeline is **model-free**; the intended DEPLOYABLE product is **model-based** (a learned world-model), which is exactly the piece Thread A/B showed does not yet generalize. The one demonstrated working spine (`reports/2026-09-06-e2-discovery-method-and-livetrap` §5) is model-free — discovered structure → do-propagation on the *true* mechanism → conflict avoided (regret 0 oracle vs +15 discovered-decoy vs +60 do-nothing). It proves **discovery completeness matters**; it does NOT prove a deployable learned solution.

## What is built (code-verified)

- **Envs:** `cdd_oran/envs/v2/e{1,2,3,4}.py` implemented. **E2 has a live shared-knob trap** (`_TRUE_ADJACENCY` P0→{K0,K1,K2,K5}; `decoy_omit_p0_k5` freezes P0 inside only the K5 mechanism). **E5 is NOT built (no `e5.py`).** E3=chains, E4=deconfounding (scaffolded/partial). Legacy Env I–IV modules **moved to `cdd_oran/envs/legacy/` (2026-09-22)**; the rest of the legacy subsystem (`cli.py`, `experiments/`, `models/cdl.py`, `conflicts.py`, `config.py`, legacy planners/analysis, `configs/`) stays until QACM is ported to `V2Env` (plan 014), then deletes. `runs/EnvironmentI..IV/` (~775MB) archives, not deletes — backs the submitted conference + DoECE papers.
- **Discovery (canonical, model-free):** E1 = partial correlation (`cdd_oran/e1slice/discovery_v2.py`, `FROZEN_SCORE_METHOD="partial_correlation"`); E2 = **RCoT-v2** kernel CI test (`cdd_oran/e2slice/discovery_rcot_v2.py`, `block_perm_reps=299`). Both write masks consumed ONLY by their own precision/recall scorers.
- **Decision gates:** structural rollouts on the true env (`rollout.py`, `sequence.py` `greedy_fm`/`exhaustive_fh`). No learned model in this path.
- **Planners:** legacy `ModelBased*` (QACM/CEM/MCTS/MPPI) query `models/cdl.py`; wired to the LEGACY env API only, **not ported to `V2Env`** (`plans/012`). **No CID planner in code** (deterministically ≡ QACM per `plans/011`). **QACM = third-party market baseline, not "ours."**

## Not wired / gaps

- **Live discovery output is NOT connected to the live decision gates** — they build the true env directly and never load a discovery mask. **No end-to-end spine in live code.**
- Kept baselines not ported to `V2Env` (`plans/012`).
- **No real O-RAN grounding anywhere** — abstract `exp(−x²)` benchmark (documented gap).

## OPEN contradictions — resolve, do not paper over

1. **Discovery method mismatch — INVESTIGATED 2026-09-22 (P1).** M3 = a **stratified operating-point conditional-CI permutation test** (`scratchpad/p0k5_fp_calibration/m3_core.py`; model-free; NC=6/NB=8/B=299/BH_Q=0.05), living ONLY in `scratchpad/` (`m3_core.py`, `m3_superset/m3_general.py`, `design_adaptive_m3/m3_*.py`), NOT in `cdd_oran/`. The frozen PRODUCT discovery is RCoT-v2 (E2) + partial-corr (E1). The headline E2-ID decision results used **M3** because it recovers the **gated** harmful edge P0→K5, which RCoT-v2/partial-corr miss (`discovery-method-no-single-superset`, `e2-harmful-edge-discovery-gap`). **RECOMMENDATION (pending user confirm, per plan 014):** canonical discovery front-end = a **FIXED ENSEMBLE** (RCoT-v2/partial-corr for general edges + an M3-style stratified conditional test for gated/operating-point edges), consistent with the "no single superset" record — and **productize M3 into `cdd_oran/`** so the solution's discovery is the method that produced its results.
2. **"World-model" is two things.** Plans want a *structural, graph-composed multi-intervention* world-model (`plans/011:47`) but the only thing exercised is an *arbitrary learned k-NN* (`E5_DATA_COLLECTION_CONTRACT.md:143`). Unresolved — this IS the model-free/model-based fork.
3. **Thread B CLOSED (2026-09-22, STOP).** Learned action-effect estimators (ridge/GCV, honest tree) fail off-gate at low occupancy = a finite-sample boundary-resolution / **structure** problem, not a bounded-estimator problem. See `reports/2026-09-19-thread-b-worldmodel-offgate/` §8.

## Anti-drift protocol

- This doc is canonical. `memory/next-session-handoff` points here.
- Verify against code before asserting; code wins ties; then update this doc + date it.
- Keep ONE pipeline story — retire/quarantine legacy Env I–IV so the learned-world-model question has a single answer.
