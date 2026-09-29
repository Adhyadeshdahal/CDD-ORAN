# Decision stack redesign: MSCR → local-response world model → WG3 arbiter (DRAFT for sol review, 2026-09-27)

## Why replace the current stack
Survey (`SURVEY.md` summary in chat, 2026-09-27): nothing in `cdd_oran/planners/`, `models/` or `benchmark/` can talk
to E6 (single float knob + KPI-vector panel vs tuple knobs, request lists, WG3 categorical decisions, SLA counters).
`planners/__init__.py` eagerly imports the legacy stack (cost.py:6 → envs.legacy), so any new code under `planners/`
inherits it. The live V2 pieces (`sequence.py`, `v2_regret.py`, masked/learned E-series WMs) are single-knob E1–E5
tools; they stay for reproducing those results but are not the product path.

Evidence that shapes the design:
- **Action class:** veto-only arbitration has ≈ no headroom (E2-CL D0; E6 rounds 1–3). Headroom needs modify / lock /
  rollback and a 60–90 s horizon (E6 veto 30 s 0.98 → 90 s 0.92 × noarb).
- **Response model is the binding constraint** (D1/D2): passively trained level models (incl. true parents) rank
  candidate actions WORSE than accept-all (regret 0.74–1.41 vs 0.59; within-state ρ ≈ 0.3). Controllers were fine.
- **Graph's value** (D1, K2, E3): input selection earns its keep modestly (strongly for myopic planners, with small
  n); SHAP-parents close most of MSCR's gap on randomized data → the graph must earn its keep through what only a
  CAUSAL graph gives: which xApp actions reach which other xApps' KPIs (conflict map), and interventional validity.
- **Objective failures of the old stack:** flat saturating satisfaction cost (FM-1), untrusted OOD predictions
  (FM-2), single-knob world model (FM-3).

## Architecture (new package `cdd_oran/decision/`, legacy-free, env adapters per world)
```
logs (E6 episodes: config, requests, decisions, KPM reports)
  └─ features.py      per-cell panel, cell-exchangeable: own knobs, in/out-neighbour CIO aggregates, neighbour
                      load/carriers, own KPM KPIs, lags; + "applied change" events (Δknob, cell, time)
  └─ discovery.py     MSCR on the pooled per-cell panel → TEMPLATE graph (knob family → KPI family, own vs
                      neighbour) with per-edge p-values; → CONFLICT MAP: xApp X's knobs → KPIs owned by xApp Y
  └─ world_model.py   WorldModel protocol: predict(state, plan, horizon) → per-cell KPI distribution
                        TrueSimWM     privileged (env.copy rollouts) = today's oracle; upper reference
                        LocalResponseWM  learned ensemble, trained on LOCAL EFFECTS of applied changes (Δ KPI over
                                      the next window vs a matched no-change counterfactual), inputs = MSCR parents,
                                      returns mean + member spread
                        ablations: NoGraphWM (all inputs), ShapGraphWM (SHAP parents), CorrGraphWM
  └─ arbiter.py       WG3Arbiter(world_model, conflict_map, objective, budget) — the SAME search as
                      wg3_oracle.py (per-region plans over accept/reject/half/lock + rollback, receding horizon,
                      coordinate descent) but scored by the world model, with:
                        - candidate pruning by the conflict map (only requests whose causal fan-out reaches
                          another xApp's KPIs are ever vetoed/modified; others auto-accept),
                        - CONFIDENCE GATE: deviate from accept-all only if the lower confidence bound of the
                          predicted improvement > 0 (ensemble spread + support/OOD check),
                        - churn-parity budget awareness (rationing changes),
                        - objective = priced SVR (UE-s violated + w_LL·LL + λ·kWh), guardrails as constraints.
  └─ adapters/        e6.py (obs ↔ features, plan ↔ WG3 decisions); later e2cl.py for the E2-CL diagnostics.
```
One search, swappable world model → the headline ablation writes itself: TrueSimWM (ceiling) vs LocalResponseWM
(MSCR graph) vs NoGraph / SHAP-graph / correlation-graph, all on the same tapes, vs published baselines
(`envs/e6/published.py`) and simple ones (`envs/e6/baselines.py`).

## The hard part: what the world model must predict
The plan fixes how the arbiter treats requests for D s; the network then evolves under the xApps' reactions.
Levels:
1. **v1 (build first): one-window local response.** Predict per-cell SLA-relevant KPIs over the next D = 20 s given
   the applied changes the plan implies NOW (accepted/modified requests, rollbacks) and the current state. Future
   requests inside the window are ignored (approximation, same for all candidates). Horizon = 1 decision epoch.
2. **v2: xApp-reaction model.** Learn P(xApp request | KPM state) from logs (xApps are deterministic rule software
   observed through their requests; we never read their code), so the WM can roll 90 s: propose → plan decides →
   response → next state. Needed if v1 shows horizon matters for the learned arbiter as it does for the oracle.
Training data: noarb logs (passive) PLUS a budgeted randomized arbitration policy during DEV collection (random
accept/reject/half/lock per region at low rate — the "excitation for response-surface estimation" pivot from
step 1b), because passive data could not rank actions (D1). Reported as a design choice with its data cost.

## Keep / retire
| component | decision |
|---|---|
| `discovery/mscr*.py` | KEEP (unchanged); called by `decision/discovery.py` |
| `benchmark/learned_world_model.py` (`MLPMember`, `fit_kpi_models`) | REUSE code in `decision/world_model.py` with member spread + tests; old module stays for E-series reproducibility |
| `planners/sequence.py`, `analysis/v2_regret.py`, masked WMs, `benchmark/rollout.py` | FREEZE (E1–E5 results depend on them); not extended |
| `planners/{base,cost,ensemble,qacm,cem,mppi,mcts,horizon,joint}.py`, `models/{base,cdl,mlp}.py`, `experiments/*`, cli evaluate/sweep, legacy tests | RETIRE with plan 012 (QACM re-implemented faithfully for E6 in `envs/e6/published.py`, so the "port QACM" blocker is gone). Step 1: make `planners/__init__.py` lazy so `sequence.py` stops importing legacy |
| `scratchpad/e6_dev/wg3_oracle.py` | becomes `decision/arbiter.py` + `TrueSimWM` |

## Build order (parallel with E6 gates)
1. `decision/` skeleton + E6 adapter + `TrueSimWM` + arbiter search (port of wg3_oracle, bit-for-bit same decisions
   as the scratch oracle on a short episode, verified then the equivalence check deleted) — no dependency on gates.
2. `features.py` + log collection harness (noarb + randomized-arbitration DEV episodes, Kaggle).
3. `discovery.py`: template MSCR on E6 panel; conflict-map sanity (does it recover ES→TS/SLICE util coupling, TS→MRO
   RLF coupling seen in the subset sweeps?).
4. `LocalResponseWM` + ablation WMs; ranking diagnostics (D1-style: within-state ρ and decision regret vs
   accept-all) BEFORE closed-loop runs.
5. Closed-loop learned arbiter vs TrueSimWM vs baselines on gate-B scenarios (only if gate B passes).
6. Legacy retirement (plan 012) once 1–5 no longer need anything legacy.

## Kill / success signals
- Step 4 gate: LocalResponseWM must rank WG3 candidates better than accept-all (regret < accept-all regret,
  within-state ρ > 0.5) on held-out DEV states; else fix data/model before any closed loop.
- Graph earns its keep iff MSCR-graph WM beats NoGraph and SHAP-graph WMs on ranking regret (paired, DEV).
- Closed-loop target = the E6_METRIC split bar, measured against the published baselines.

---
# REVISION v2 (2026-09-28) — answers to SOL_REVIEW.md (all 6 BLOCKING accepted)
1. **Estimand = policy effect at the arbitration boundary.** At decision epoch e (every D = 20 s) with observable
   history h_e, a candidate is a per-region WG3 policy π_r (mode per xApp ∈ accept/reject/half/lock + rollback flag)
   held for D s, then the default continuation (accept-all). Target per region r:
   Δ_r(π | h_e) = E[C_H(π) − C_H(accept-all) | h_e], with C_H over H = 90 s = violated UE-s (by slice) +
   RLF + severe + energy + churn used, measured on the UEs served in the region's cells (and a network-wide spill term).
   The oracle (TrueSimWM) evaluates exactly this quantity on the true tape; the learned model estimates it.
2. **Trace schema** (versioned, `decision/trace.py`): per second — pre-action config, requests, decisions, ACK/NACK
   and applied value per request, rollbacks/locks, churn used/cap, delivered KPM reports (runtime features), and
   per-cell per-slice violated UE-s / RLF / energy increments (LABELS, privileged, never features). Splits by
   episode/tape/scenario.
3. **Data = randomized joint-policy episodes with logged propensities** (`decision/collect.py`): at each epoch each
   region draws π_r from a declared distribution (accept-all with prob 1−ε, else uniform over modes; stratified by
   congestion), recorded with its probability and application success. Effects fit as candidate-minus-incumbent
   returns (IPW / doubly-robust regression), not levels, not "no-change" matching.
4. **xApp reactions are inside the outcome** (the estimand integrates them over H). No separate request model for
   one-epoch decisions; a history-conditioned transition model only if multi-epoch planning is needed. Features
   include report age, pending requests, locks, recent NACKs, time since last change per knob.
5. **Confidence gate = calibrated paired-difference bound**: split-conformal on held-out EPISODES of
   (Δ̂ − Δ_oracle) for the selected candidate (selection-aware: calibrate the argmax pipeline, not single
   predictions), per scenario/support stratum; outside support → fallback accept-all. Report coverage,
   false-improvement rate, guardrail violations. Guardrails bounded separately (per-component effects).
6. **Graph = soft prior + context selector**, never pruning by omission: MSCR (template, cell-exchangeable) chooses
   which neighbouring regions' state/requests enter each region's effect model and weights feature groups; ablate
   vs all-input and SHAP-selected context at equal capacity/samples/search; oracle audit of any pruning.
Non-blocking accepted: observable churn allowance (per-epoch budget = noarb's typical rate from DEV, not the
same-tape count) for all comparators; keep legacy/E-series code until audits pass; update plans 011/012 before
deleting. Declined: persistent oracle≡arbiter equivalence test (user rule: verify then delete golden tests);
instead a behavioural test that TrueSimWM scores the pending batch between phases.
Build order = sol's: (1) gates → (2) freeze estimand/trace/budget → (3) TrueSim reference + short-vs-90 s ranking
test → (4) randomized collection → (5) effect models + calibrated gate on held-out oracle panels → (6) closed loop.

## Timing parameters D = 20 s, H = 90 s (status: declared engineering choices, NOT tuned; user asked 2026-09-28)
- D (policy re-plan period): xApp cadences are SLICE 1 s, TS/ES 10 s, MRO 30 s; D went 10 → 15 → 20 s across oracle
  rounds mainly to cut compute. Never compared. Per-request decisions stay per second; only the policy is re-chosen
  every D (state this vs the near-RT 10 ms–1 s loop: SON-type xApps with 1–30 s cadences).
- H (look-ahead): evidence only for "longer beats 30 s" (veto oracle 30 s backfired up to 1.24× noarb; 90 s gave
  ~6 pts at medium). 90 s covers ≥ 2 MRO report cycles, but is SHORTER than the ES 120 s carrier dwell — known gap.
- TODO before TEST: DEV oracle sensitivity sweep D ∈ {10, 20, 30} × H ∈ {60, 90, 150}; fix D/H in the protocol.
  Learned-model labels are per-second in the trace, so H can be re-chosen without recollection; D affects collection.
