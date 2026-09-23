# Plan 015 — E5: the composed shared-knob benchmark (design spec, for review)

**Status: BUILD IN PROGRESS / DEV, sol-reviewed (2026-09-23).** Read `docs/ARCHITECTURE.md` + `plans/014`
first. `cdd_oran/envs/v2/e5.py`, `scripts/e5_spine.py`, `scripts/e5_baselines.py`, and
`cdd_oran/benchmark/MaskedE5WorldModel` are BUILT (DEV, not frozen; no `GATE_CONTRACT_E5.md` yet).
An adversarial sol review (`scratchpad/sol_review_e5_spine.md`) corrected an overstated first headline;
this spec now reflects the honest, real-methods result. Registration/freeze remain user-gated.

## 0. Honest result so far (real methods, CORE layer only — sol-corrected)

On the CORE layer (chain/confound/noise OFF, H=1), at the harmful edge's natural RARE operating regime
(gate occupancy ≈ 5%, where `base(G1,G2)` is near the K_harm threshold), with **real discovery masks**
run end-to-end through the spine on a 40-state in-gate bank:

| method | recovers P0→K_harm? | in-gate-conditional regret (norm) | K_harm violations |
|---|---|---|---|
| oracle graph | — | 0.000 | 18/40 |
| **SHAP-DAG (Sharma, GBDT proxy)** | **no** (ratio 0.026) | **0.378** (raw 34.8) | 40/40 |
| **two-tower reconstruction (GNN-style)** | **no** | **0.378** | 40/40 |
| pooled \|corr\| | yes | 0.000 | 18/40 |
| stratified operating-point (ours) | yes | 0.000 | 18/40 |

**Supported claim:** the two named learned-importance SOTA competitors (SHAP-DAG, GNN) prune a
subdominant decision-critical edge and walk into the conflict; a truth-free operating-point-stratified
test recovers it. **NOT yet supported:** "beats correlational methods" (pooled `|corr|` recovers it —
needs the confound layer); the chain/confound/noise composition (not yet evaluated); unconditional
(deployment-weighted ≈ 0.019 at 5% occupancy, report both). Reward geometry + occupancy were tuned
during construction, so this is a **constructed stress case**, to be frozen + sensitivity-swept before
any confirmatory comparison. Honest amplitude×occupancy separation: `scratchpad/e5_design/`.

## 1. Why E5 exists (what E2 could NOT show)

E2's decision spine (`scripts/e2_spine.py`, plan 014) ran every discovery method through
discovery→do-propagation→decision. Verified result:

| method | family | P0→K5 recovery | spine regret |
|---|---|---|---|
| oracle | — | — | 0.000 |
| M3 | operating-point stratified CI | 20/20 | 0.000 |
| SHAP-DAG (Sharma 2025) | predictive importance | **10/10** | **0.000** |
| two-tower GNN | correlational graph-learning | **10/10** | **0.000** |
| RCoT-v2 | kernel CI | 3/10 | 0.133 |
| pdCor | distance-corr CI | 0/1 | 0.190 |
| do-nothing | — | — | 0.795 |

**E2 does not separate us from the strongest competitors on discovery.** E2's harmful edge
`K5 = -35·exp(-((P0+P7-25)²)/…)` is *additively* argumented, so P0 carries global marginal variance;
every variance-based method (SHAP-DAG, GNN) recovers it and, through do-propagation, avoids the trap.
E2 separates us from the CI methods on structure, and from SHAP-DAG/GNN only on *decision-relevance*
(their deliverables — a descriptive ATE, a conflict graph — never convert structure into a
conflict-avoiding action; Sharma's linear ATE(P0→K5) ≈ 0 and sign-unstable while the true in-gate harm
is a full 34-unit swing).

**The ownable discovery-superiority claim requires a harmful edge that is invisible to variance-based
methods at the *structure* level.** That is E5.

## 2. The validated E5 recipe (3 scratchpad probes, `scratchpad/e5_design/`)

The harmful edge must be all three of:

1. **Amplitude-subdominant** — its in-gate effect on `K_harm` is small relative to `K_harm`'s dominant
   benign parent. `measure_zero_gate_probe.py`: at occupancy 0.2 vs a benign parent of amplitude 80,
   a P0 slope δ∈{0.1,0.2,0.5} gives SHAP importance ratio **0.043–0.218 < 0.30** → SHAP-DAG/GNN PRUNE
   it (structure miss), while a conditional test still detects it (p<0.01). (Falsified alternative:
   low occupancy *alone* does not hide a strong slope — SHAP still recovers at ratio 0.34.)
2. **Co-parent-gated** — the P0 effect exists only inside a thin `(G1,G2)` operating-point band, zero
   off-gate. Per E2, this defeats the CI tests (RCoT/pdCor miss the gated edge); only an
   operating-point-stratified M3-style test recovers it.
3. **Decision-critical** — a locally-attractive benign KPI pulls P0 into the gate; missing the edge
   tips `K_harm` across its constraint. `e5_decision_criticality_probe.py`: oracle steps P0 back
   (80→74) to keep `K_harm` satisfied (regret 0); the missed-edge planner chases the benign peak
   (P0=80) and tips `K_harm` over → **regret 6.4–7.1 in-gate, 0 off-gate**. The harm flips the discrete
   satisfied-count (`−satisfied²` in the hinge), which is how a *subdominant* KPI change (SHAP-pruned)
   stays decision-critical.

Combined: **subdominant ⇒ SHAP/GNN miss; gated ⇒ RCoT/pdCor miss; only M3-style catches; and missing
it is decision-critical.** Only discovery(stratified)→do-propagation→action-selection avoids the trap.

## 3. The composed E5 SCM (fan-out + chain + confound + noise) — user chose "fully composed"

E5 layers all four E-series stressors around the validated harmful edge so that **each named
competitor fails on at least one axis and only the full causal pipeline handles all**. Draft layout
(constants to be frozen at build time per the `GATE_CONTRACT_E*` discipline; shapes are the design):

**Params.** `P0` = shared knob (the acted control). `G1, G2` = context/gate axes (E2-style, narrow
band). `Pc` = a chain-driving param. (Latent `Z` = confounder, not a param.)

**KPIs / mechanisms** (one-step lag per hop, `k_t = f(p_{t-1}, k_{t-1})`, per `V2Env`):

- **Fan-out (E2-style), shared knob P0 → several KPIs.** `P0` benignly helps `K_ben` and drives a
  chain, *and* has the subdominant-gated harmful edge into `K_harm`. The shared knob is the conflict
  surface.
- **Benign pull.** `K_ben = A_ben·exp(-((P0-μ_ben)²)/(2 w_ben²))` — a broad bump **peaked in the
  harmful P0 zone**, peak just below its own satisfy-above threshold ⇒ ~unsatisfiable ⇒ the planner
  chases the peak (into the gate). This is the "locally-good" pull.
- **Chain (E3-style), multi-hop harmful path.** `P0 → K_mid → K_harm` (unit-ish coefficients, one hop
  each) so the harmful influence is partly *indirect*: discovery must trace a 2-hop path, not just a
  direct edge. The decoy truncates the chain's contribution to `K_harm` (E3 `truncate_fanout` pattern).
- **The harmful KPI (the core).**
  `K_harm = base(G1,G2)  +  γ·K_mid  +  subdom·(P0−C0)·1[G1≤τ1, |G2−c2|≤w]  +  θ·Z`
  where `base(G1,G2)` is the DOMINANT benign parent (large amplitude → subdominance of the P0 term),
  the gate is the thin operating-point band, and `θ·Z` is the confounding term. Satisfy-below
  threshold, positioned so the in-gate P0 push tips it over (decision-critical).
- **Confound (E4-style), FULLY LATENT Z (no observable proxy; §6.3).** `Z → P0_behavior` (obs mode)
  and `Z → K_harm`, so the *observational* P0–K_harm association is confounded; the causal pipeline uses
  the randomized-`do(P0)` corpus, which identifies the true effect **by randomization** (no backdoor set
  needed). E5 exposes both an **obs** and a **do** corpus: correlational methods on the obs corpus get
  the confounded association; ours uses the do corpus.
  **✅ BUILT (2026-09-23, sol-reviewed conditional pass — `scripts/e5_confound.py`, `tests/test_e5_confound.py`).**
  Sol #6 fixed: default `theta` flipped `+2.5 → −2.5` (`Z → K_harm` negative, `lam>0` positive) so the
  confound MASKS the true `+subdom`. The confound demonstration is the OBS-vs-DO contrast of an
  ORACLE-GATE IN-STRATUM OLS estimator (Option A scope — grant true `base(G1,G2)`/gate/form, contest only
  the scalar in-gate P0 coefficient; fitted world model = `E5V2Env(subdom=α̂, theta=0)`, drift-safe):
  OBS slope is confound-biased `+0.117` → regret **0.208** (walks into the trap); randomized `do(P0) ⟂ Z`
  identifies `+0.198` → regret **0.002**. A paired **θ=0 difference-in-differences control** gives
  `R_OBS−R_DO = 0.000` (no confound → both identify), so DiD = **+0.206** is attributable to confounding;
  a 5-seed panel is tight (OBS 0.208, DO 0.0005, gap 0.206–0.208). **Honest scope (sol):** the winning arm
  REQUIRES interventional data (value of intervention capability under a fully-latent Z, NOT a superiority
  claim over observational methods); marginal pooled OLS sign-reverses (−0.073) but adds no decision
  contrast (gate dilution already breaks its DO arm — saturation, not absence of confounding); GBDT is
  diagnostic-only (hyperparameter-sensitive PD slope). Truth-free gate DISCOVERY → effect estimation
  END-TO-END (propagating discovery uncertainty) is DEFERRED (overlaps M3 productization, §6.5).
- **Noise ON.** `obs_noise_scale > 0` and process noise ON (E5 is the realistic-noise env — E1–E4 ran
  largely noise-off). Discovery + decision must be robust to it.

**Decision panel.** `{K_ben (satisfy-above), K_harm (satisfy-below), + chain terminal(s)}`, hinge
objective reused from `cdd_oran/analysis/v2_regret.py` (the frozen `reward`/`PanelXApp`). Conduit /
non-descendant KPIs excluded from the panel (E3 pattern).

**Horizon.** H≥2 (the chain needs ≥2 hops to surface), open-loop rollout via
`cdd_oran/benchmark/rollout.py`.

## 4. Evaluation protocol (reuse the plan-014 spine + the completed baseline harness)

- **Discovery arms** through the spine (`MaskedE2WorldModel`-analogue for E5's structure): M3-style
  stratified ensemble (ours), SHAP-DAG, two-tower GNN, RCoT-v2, pdCor, partial-corr.
- **Planner baselines**: do-nothing floor, true-SCM oracle ceiling, and QACM (market baseline — needs
  the V2Env port, `planners/cost.py:6`; decision-level comparison, not a discovery mask).
- **Metric**: normalized regret = `G_oracle − G_planner` (`docs/benchmark/GATES.md`), plus
  harmful-edge (and harmful-*path*) discovery P/R/F1, and decision accuracy (fraction of seeds
  rejecting the harmful action, matching oracle). Sharma-unit ATE table for legibility.
- **Headline (HYPOTHESIS pending confound + composed eval + H≥2):** on the fully-composed E5, SHAP-DAG/GNN
  prune the subdominant edge; correlational fits on the obs corpus get the confounded association; **only
  stratified discovery → do-propagation on the do corpus avoids the conflict (regret ≈ 0).** STATUS: the
  SHAP-DAG/GNN half is DEMONSTRATED on the CORE layer with real runs (§0); the correlational/confound half
  is NOT yet built (§3 ⚠️); the chain (needs H≥2) and noise layers are not yet evaluated. Mark as a
  hypothesis in any writeup until the composed evaluation exists.

## 5. Build plan (after this spec is approved)

1. Write `docs/benchmark/GATE_CONTRACT_E5.md` (frozen constants: ranges, gate band, subdom, chain
   coeffs, θ, standardization moments at n=1e6, thresholds) — the E2/E3/E4 discipline.
2. Build `cdd_oran/envs/v2/e5.py` (`E5V2Env(V2Env)`): the composed `_update_kpis`, latent-`Z` tape slot
   (E4 pattern), obs/do modes, `decoy`/`truncate` variants, `adjacency_edges`/`true_adj_matrix`, panel.
3. `scripts/e5_decision_gate.py` + `scripts/e5_spine.py` (generalize the E2 spine to E5's structure) +
   `tests/test_e5_*.py` (bit-exact decoy equivalence, latency, feasibility, the 3 validated probes as
   regression fixtures).
4. Register + generate the discovery corpus (obs + do) → run every discovery/baseline arm → the table.
5. RUN is expensive → user-gated 'go'; freeze/pre-reg → user-gated.

## 6. Open decisions

**RESOLVED (user 2026-09-23):**
1. **Build order — INCREMENTAL WITHIN COMPOSED + ablation gates.** Build the full composed `e5.py`, but
   bring layers up one at a time (gated core → +chain → +confound → +noise), each layer individually
   *toggleable* on the constructor (E3 `truncate_fanout` / E4 `lam`,`mode` pattern) and each with an
   ablation gate proving it is what defeats the relevant competitor. Same final env; attributable.
3. **Confounding — Z is FULLY LATENT; the do-corpus identifies.** No observable proxy. Only the
   randomized-`do(P0)` corpus identifies `P0→K_harm`; observational-only baselines get the confounded
   sign/magnitude by design. (E4 latent-Z pattern exactly.)

**STILL OPEN (resolve during build):**
2. **Standardization**: analytic vs n=1e6 empirical moments for the composed mechanism (E3 analytic,
   E2/E4 empirical). Compute at build; freeze into `GATE_CONTRACT_E5.md` — user-gated freeze.
4. **QACM**: port to V2Env now (unblocks legacy delete) or defer; it's a decision-level baseline.
5. **Discovery front-end canonicalization** (ARCHITECTURE.md contradiction #1): E5 makes the
   fixed-ensemble (RCoT-v2/partial-corr + M3-style stratified) concrete — productize M3 into `cdd_oran/`
   as part of this build.

## 7. Risks / honesty

- The headline rests on E5 being a *faithful* stressor, not one reverse-engineered to make competitors
  fail. Each stressor is grounded (E2 fan-out, E3 chain, E4 confound) + the validated subdominant-gated
  edge; the ablation gates (turn each layer off → competitor recovers) are the honesty check.
- Still the abstract `exp(−x²)` benchmark family — no real O-RAN grounding (documented gap; ns-O-RAN is
  future per plan 014).
- Per-edge mechanism identifiability under noise + confounding is the hardest part; the uncertainty-gated
  abstention (plan 014 mechanism substrate) is the safety valve if a layer is unidentifiable.
