# E5 Composed Gate Contract (DRAFT preregistration — NOT YET FROZEN)

**Status: DRAFT. The freeze is USER-GATED** ([[run-gating-explicit-go]]): no value here is binding until
the user says "freeze E5". This draft transcribes the DEV design values now in `cdd_oran/envs/v2/e5.py`
plus the built-and-committed layer results (`feat/v2` `2ef5f09` confound, `fd3c159` chain), the
pre-declared sensitivity sweeps (`scripts/e5_sensitivity.py`), and a proposed confirmatory success rule +
separate seed set. It follows the E2/E3/E4 contract discipline. Once frozen, no equation, coefficient,
gate, threshold, horizon, or tolerance below may change; a failed confirmatory check is recorded as-is.

E5 is the composed shared-knob conflict benchmark: a harmful edge that is amplitude-SUBDOMINANT (defeats
variance/predictive-importance discovery — SHAP-DAG/GNN), co-parent-GATED (defeats CI tests — RCoT/pdCor),
reachable by a 2-hop CHAIN (defeats direct-edge-only discovery), CONFOUNDED by a latent Z (defeats
observational effect estimation), and DECISION-CRITICAL (missing/mis-estimating it flips the action).

---

## Status and anti-tuning rule

**Status: DRAFT / DEV.** Values were tuned during construction (disclosed — see §Sensitivity). The
sensitivity sweeps and the θ=0 / subdom=0 / chain_gamma=0 negative controls are the honesty package; the
confirmatory comparison MUST use the separate untouched seed set in §Confirmatory. After freeze, no value
below changes.

## SCM and mechanism (DEV design values, `cdd_oran/envs/v2/e5.py`)

```text
num_params = 4   # P0 (shared knob), G1, G2 (gate axes), P3 (distractor)
num_kpis   = 4   # K_ben(0), K_harm(1), K_mid(2, conduit/excluded), K_dist(3)
id_ranges  = [(-100,100), (1.5,4.0), (-100,150), (-20,20)]
one-step lag: k_t = f(p_{t-1}, k_{t-1}); latent Z committed alongside params (E4 pattern)

K_ben  = A_BEN * exp(-((P0-MU_BEN)^2)/(2 W_BEN^2))                    # benign local pull
K_mid  = C_CHAIN * P0                                                 # chain conduit (panel-excluded)
K_harm = base(G1,G2)                                                  # DOMINANT benign parent (P0-free)
         + subdom      * (P0 - C0)      * gate(G1,G2)                 # subdominant gated DIRECT edge
         + chain_gamma * K_mid_prev     * gate(G1,G2)                 # gated 2-hop chain edge
         + theta       * Z_prev                                       # latent confounder
K_dist = A_DIST * exp(-((P3-MU_DIST)^2)/(2 W_DIST^2))                 # benign distractor
base(G1,G2) = A_BASE * exp(-((G2-MU_BASE)^2)/(2 W_BASE^2)) + B1*G1
gate(G1,G2) = 1[ G1 <= TAU1  AND  |G2 - C2| <= W_GATE ]

A_BEN, MU_BEN, W_BEN   = 12.0, 80.0, 40.0
C_CHAIN                = 1.0
A_BASE, MU_BASE, W_BASE, B1 = 72.0, 25.0, 25.0, 2.0
TAU1, C2, W_GATE       = 2.75, 25.0, 12.6      # occupancy P(G1<=TAU1)*P(|G2-C2|<=W_GATE) ~= 0.05
C0                     = 0.0
A_DIST, MU_DIST, W_DIST= 70.0, 0.0, 10.0

layer coefficients (each layer individually toggleable):
subdom       = 0.20            # core gated-subdominant direct edge (0 => off)
chain_gamma  = 0.20            # gated 2-hop chain edge (0 => off)
theta        = -2.5            # latent confounder Z->K_harm; NEGATIVE so it MASKS the +subdom
lam          = 1.0             # Z->P0_behavior coupling (obs mode); lam*theta<0 = masking
ETA_SCALE    = 0.5             # behavior-policy noise
_DO_GRID     = linspace(-100,100,201)   # randomized do(P0) grid (P0 ⟂ Z)

panel_kpi_ids   = (0,1,3)      # K_ben, K_harm, K_dist  (K_mid excluded — conduit)
kpi_thresholds  = (55, 80, 0, 55)     directions = (above, below, -, above)
```

The observed adjacency toggles with the layer coefficients. Latent edges: `Z->K_harm` (always when
theta!=0) and `Z->P0_behavior` (obs mode, lam!=0). `Z` is fully latent (no observable proxy) — the
randomized `do(P0)` corpus identifies by randomization, not by a backdoor set.

## Standardization

**PENDING (compute at freeze).** The panel currently scores in RAW units (`scripts/e5_spine.e5_panel`
uses mean=0, std=1). Freeze either analytic moments (E3 pattern) or n=1e6 empirical moments (E2/E4
pattern) for each panel KPI's interventional-mean utility; record the exact values here. Regret magnitudes
below are raw-unit / range-normalized and will shift under standardization — recompute the confirmatory
numbers after the standardization is frozen.

## Decision protocol

- Selection: the frozen H=1 single-control oracle (`cdd_oran/analysis/v2_regret.py`, `score_grid` /
  `argmax_v R`), reused unchanged. `reward = -(Σ w_i·distance_i·s − (Σ ok_i)^2)`, s=10, w=1.
- Horizon per layer: **H=1** scores the direct edge (core, confound); **H=2** (`score_grid_h`, 3 advances)
  is REQUIRED to score the 2-hop chain — H=1 delegates byte-identically to the frozen `score_grid`.
- Scoring integrates the latent Z to its mean (theta=0 at scoring): a risk-neutral mean-SCM surrogate,
  NOT expected nonlinear reward under Z (documented limitation; `R(E[K]) != E[R(K)]`).
- In-gate bank: the first N_B qualifying committed in-gate states (`build_ingate_bank`); regret is
  **in-gate-conditional** (~5% occupancy). Report deployment-weighted regret = in-gate-conditional × occ.
- World models are the true frozen `E5V2Env` mechanism restricted to a discovered structure
  (`MaskedE5WorldModel`) or carrying a fitted scalar effect (`E5V2Env(subdom=α̂, theta=0)`, confound
  layer). Granting the true `base(G1,G2)`/gate/form in the confound layer is a disclosed
  oracle-nuisance isolation (§Layers).

## Layers and DEV results (committed; in-gate-conditional, N_B=40, H per layer)

| layer | knobs | competitor it defeats | DEV result |
|---|---|---|---|
| CORE (structure) | subdom=0.2, gate | SHAP-DAG / GNN (subdominance) + RCoT/pdCor (gating) | SHAP-GBDT/GNN prune (SHAP ratio 0.026) → regret 0.378; pooled\|corr\|/stratified recover → 0.000 |
| CONFOUND (effect) | theta=−2.5, lam=1 | observational effect estimation | oracle-gate in-stratum OLS: OBS +0.117→0.208, DO +0.198→0.002; **DiD +0.206** (θ=0 control 0.000) |
| CHAIN (mediated) | chain_gamma=0.2 | direct-edge-only discovery (KPI→KPI edge) | miss-chain INERT@H1 (0.000) / ACTIVE@H2 (0.311); miss-BOTH@H2 0.637 |

Honest scope of each headline (see the env docstring + `scripts/e5_confound.py`): the CONFOUND winning arm
REQUIRES interventional (randomized-P0) data — value of intervention under a fully-latent Z, NOT a
superiority claim over observational methods; marginal pooled OLS sign-reverses (−0.073) but is
decision-saturated by gate dilution (no added decision contrast, not absence of confounding); GBDT is
diagnostic-only (hyperparameter-sensitive PD slope). The CONFOUND stratified arm uses the TRUE gate to
isolate the estimand (truth-free gate DISCOVERY → effect estimation end-to-end is DEFERRED).

## Sensitivity (pre-declared, `scripts/e5_sensitivity.py`) — the operating region, not a knife-edge

All curves smooth and monotone with correct negative controls (subdom→0, theta→0, chain_gamma→0 give ~0):

- **Edge amplitude (subdom):** miss-direct H=1 regret 0.005 / 0.045 / 0.378 / 0.678 at subdom 0.05/0.1/0.2/0.5
  — decision-critical band ≈ 0.1–0.5; the design 0.2 is mid-band, not an extreme.
- **K_harm threshold:** regret 0.472 / 0.417 / 0.378 / 0.329 / 0.230 at thr 75/78/80/82/85 — smooth; design 80 mid-range.
- **Confound strength (theta):** DiD +0.000 / +0.018 / +0.074 / +0.202 / +0.374 at theta 0/−0.5/−1/−2.5/−5;
  DO slope stays ~+0.20 throughout (identification robust), OBS slope degrades smoothly.
- **Sample size:** OBS in-stratum slope stable at +0.117 across n=2k–20k; DO slope → +0.200 with tightening
  range (±0.015 @2k → ±0.002 @20k) — not a small-sample artifact.
- **Chain strength (chain_gamma):** miss-chain H=2 regret 0.000 / 0.230 / 0.345 / 0.441 at 0/0.1/0.2/0.5.
- **Structure (SHAP P0 ratio, occ × subdom):** at design (occ 0.05, sd 0.2) ratio **0.025** (prunes,
  matches `e5_baselines` 0.026). The ratio scales ~linearly with BOTH occupancy and amplitude:
  occ{0.05,0.1,0.2,0.4} × sd{0.1,0.2,0.5,1.0} → e.g. sd=0.2 gives 0.025/0.050/0.096/0.186; sd=1.0 gives
  0.121/0.250/0.469/0.931. SHAP still PRUNES (ratio<0.30) at occ 0.2–0.4 for the decision-relevant sd
  range (≤0.5), and only RECOVERS at sd≥0.5 & occ≥0.2 — so amplitude-subdominance suppresses SHAP
  INDEPENDENTLY of occupancy; the miss is the combination, not occupancy alone (sol #5). Base-width
  (occ 0.05, sd 0.2): W_BASE 15/25/40/100 → 0.032/0.025/0.023/0.048 (low across the swept dominance).

## Confirmatory success rule (PROPOSED — binds only on freeze)

Bank: freeze N_B = 40 in-gate states. **Confirmatory seed set = a SEPARATE untouched range (proposed
seeds 1000–1999, from which the first 40 in-gate qualify)** — construction/DEV used seeds 0–5, so the
confirmatory bank is disjoint. Report per arm: raw regret, range-normalized regret, K_harm
constraint-violation rate (excess vs the oracle baseline) and magnitude, D(s), and deployment-weighted
(× occupancy) regret, each with a multi-seed CI over the confirmatory seed panel. Proposed PASS:

- CORE: SHAP-DAG and GNN regret ≥ 0.30 (prune the edge) while stratified + pooled\|corr\| ≤ 0.02 (recover).
- CONFOUND: DiD ≥ 0.15 with the θ=0 control ≤ 0.02 and stratified-DO regret ≤ 0.02.
- CHAIN: miss-chain H=2 regret ≥ 0.15 AND miss-chain H=1 regret ≤ 0.001 (inert without the extra hop).

## Explicitly deferred (defaults must not change after freeze)

- End-to-end truth-free gate DISCOVERY → effect estimation (propagating discovery uncertainty), replacing
  the confound layer's oracle-gate estimand — overlaps M3 productization (plan 014 #4, ARCHITECTURE #1).
- NOISE layer (`obs_noise_scale`, `process_noise`) confirmatory eval.
- QACM V2Env port (decision-level market baseline) — and QACM is EXCLUDED from the effect-estimation
  table (no causal effect machinery; category error — literature lane).
- Full whole-mechanism (no base-grant) confound pipeline — Option B, a separate future gate.

## Implementation pointers

Env `cdd_oran/envs/v2/e5.py`; spine + H-rollout `scripts/e5_spine.py`; masked/fitted world models
`cdd_oran/benchmark/masked_world_model.py`; layer harnesses `scripts/e5_{baselines,confound,chain}.py`;
sensitivity `scripts/e5_sensitivity.py`; tests `tests/test_e5_{spine,confound,chain}.py`; design spec
`plans/015-e5-composed-benchmark-design.md`.
