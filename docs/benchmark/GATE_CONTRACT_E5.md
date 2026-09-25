# E5 Composed Gate Contract: STRUCTURE-ONLY (FROZEN preregistration)

> **Scope note (added 2026-09-25, rules and results unchanged):** this E5 evidence is **NOISELESS CORE only**. The agreed E5 design (`plans/015` §3) has noise ON; the noise layer was never applied (the corpus generator bypasses it) and this departure was not flagged before the freeze. Stage 0 later found that on this noiseless corpus the SHAP-GBDT proxy, at its DEV-selected τ = 0.005 (τ = 0.005–0.02 all matched on the swept grid), matched MSCR (`docs/benchmark/STAGE0_RESULT.md`), so the 'competitors miss it' result is specific to τ = 0.10. A synthetic observation-noise stress test of the CORE discovery corpus (not process noise, not the fully composed E5) is pre-declared in `docs/benchmark/E5_NOISE_DETECT.md`.

**Status: FROZEN 2026-09-24 (user: "freeze E5").** The freeze point is the commit that introduced this
status line. No equation, constant, corpus spec, arm, threshold, seed range or gate below may change; a
failed confirmatory check is recorded as-is.

**Scope change (2026-09-24).** The earlier draft (committed `755c0a8`) proposed decision-regret gates in
RAW units. Under the locked SEMANTICS scoring (each panel KPI standardized once) the decision contrast
collapses, and a pre-declared reward-only re-balance failed (0/900; STOP). This contract therefore
freezes E5 as a **structure (discovery) benchmark only**. Decision claims are deferred to a separately
pre-declared study (uncertainty-aware planner, see §Deferred). Evidence:
`scratchpad/e5_design/{std_rerun/,REBALANCE_TARGET.md,REBALANCE_RESULT.md}`, sol review
`scratchpad/sol_review_e5_rebalance.md`.

## What E5 tests

A shared knob P0 fans out to several KPIs. Its edge to K_harm is **amplitude-subdominant** (small
next to K_harm's dominant benign parent) and **co-parent-gated** (active only in a thin operating band,
occupancy ≈ 0.05). Claim under test: the fixed predictive-importance proxies (SHAP-GBDT proxy, two-tower
reconstruction, both at τ = 0.10) miss this edge, while the stratified test MSCR recovers it, with calibrated
false-discovery control on the parameter family.

## SCM and mechanism (unchanged DEV design values, `cdd_oran/envs/v2/e5.py`)

```text
num_params = 4   # P0 (shared knob), G1, G2 (gate axes), P3 (distractor)
num_kpis   = 4   # K_ben(0), K_harm(1), K_mid(2, conduit), K_dist(3)
id_ranges  = [(-100,100), (1.5,4.0), (-100,150), (-20,20)]
one-step lag: k_t = f(p_{t-1}, k_{t-1}); latent Z committed alongside params (E4 pattern)

K_ben  = A_BEN * exp(-((P0-MU_BEN)^2)/(2 W_BEN^2))
K_mid  = C_CHAIN * P0
K_harm = base(G1,G2) + subdom*(P0-C0)*gate(G1,G2) + chain_gamma*K_mid_prev*gate(G1,G2) + theta*Z_prev
K_dist = A_DIST * exp(-((P3-MU_DIST)^2)/(2 W_DIST^2))
base(G1,G2) = A_BASE * exp(-((G2-MU_BASE)^2)/(2 W_BASE^2)) + B1*G1
gate(G1,G2) = 1[ G1 <= TAU1  AND  |G2 - C2| <= W_GATE ]

A_BEN, MU_BEN, W_BEN        = 12.0, 80.0, 40.0
C_CHAIN                     = 1.0
A_BASE, MU_BASE, W_BASE, B1 = 72.0, 25.0, 25.0, 2.0
TAU1, C2, W_GATE            = 2.75, 25.0, 12.6      # occupancy ≈ 0.05
C0                          = 0.0
A_DIST, MU_DIST, W_DIST     = 70.0, 0.0, 10.0
subdom = 0.20, chain_gamma = 0.20, theta = -2.5, lam = 1.0   (layer defaults)
```

## Standardization (recorded; binds any future E5 decision study)

`scripts/e5_standardization.py`: n = 1,000,000, seed 0, params i.i.d. Uniform(id_ranges), lagged K_mid
from an independent P0 draw, subdom = chain_gamma = 0.2, theta = 0 (Z at its mean), no noise.
Per-KPI (mean, std), also `E5_KPI_MOMENTS` in `scripts/e5_spine.py`:

```text
K_ben : (4.161893, 4.634914)
K_harm: (23.514415, 24.662764)
K_mid : (0.075764, 57.680719)   # conduit, not in the panel
K_dist: (41.874536, 20.192141)
```

## Discovery corpus (frozen)

- Layer: **CORE** (subdom = 0.2, chain_gamma = 0, theta = 0, no noise).
- Design: params i.i.d. Uniform(id_ranges), KPIs = `E5V2Env._update_kpis(params, 0)` (the
  `scripts/e5_baselines.corpus` generator). Randomized design, the regime MSCR is validated for.
- **n = 24,000 rows per seed.** Disclosed choice from a DEV power probe (seeds 0–9): P0→K_harm recovered
  7/10 at n=6000, 10/10 at n=12000, 10/10 at the permutation floor at n=24000
  (`scratchpad/e5_design/mscr_e5_probe*`). 24,000 is taken for margin, not tuned against confirmatory
  data. It is a DEV choice, not independent evidence or an unbiased power estimate.
- Truth (param edges, `(kpi, param)`): (0,0) (1,0) (1,1) (1,2) (2,0) (3,3). Harmful edge = (1,0).
  The other 10 param→KPI pairs are exact nulls (independent exogenous params).

## Arms (frozen)

| Arm | Implementation | Declares edge (k, p) if |
|---|---|---|
| **MSCR-v2** (ours) | `cdd_oran.discovery.discover_mscr(X, Y, n_params=4, seed)`, frozen config (NC 6, NB 8, MIN 40, B 2999) | per-target BY q=0.05 over the 4 params |
| SHAP-GBDT proxy | `scripts/e5_baselines.shap_mask` (HistGBR + TreeExplainer mean\|SHAP\|) | importance ≥ 0.10 × max for that KPI |
| Two-tower reconstruction | `scripts/e5_baselines.gnn_mask` (800 epochs, seed 0) | gate score ≥ 0.10 × max for that KPI |
| Pooled \|corr\| (reference) | `scripts/e5_baselines.corr_mask` | \|corr\| ≥ 0.10 × max for that KPI |

τ = 0.10 is the implemented relative threshold: a lower, miss-conservative choice than 0.30 (a lower τ
keeps more edges), but not the lowest possible. The claim is pinned to this fixed rule; no
threshold-robustness claim is made. The earlier draft's "0.30" is withdrawn. Labels:
"SHAP-GBDT proxy" (HistGradientBoosting stands in for XGBoost) and "two-tower reconstruction" (not
message-passing), never "SHAP-DAG" or "GNN" without the qualifier. The two-tower arm is one in-sample fit with fixed
initialization and 800 epochs; results are for that fixed proxy, not neural reconstruction in general. Pooled |corr| is **expected to
recover** the edge (it is not defeated in CORE) and is reported, not gated.

## Pre-freeze de-risk (DEV seeds 0–19, before the freeze; go/no-go, not evidence)

- **D1 MSCR marginal calibration on E5** (B1 pattern from the E2 v2 study; an empirical diagnostic of
  the shared-bank mismatch, not an exact max-null test or a proof of validity): permute one param column
  (64 draws per seed × target × slot) and score its p-value against the bank; GO if at every BY cutoff
  for m=4 (q·k/(4·H_4), k = 1..4) and at 0.02 the seed-cluster one-sided 95% upper bound of
  P(p ≤ α) is ≤ 1.5α.
- **D2 power:** MSCR declares P0→K_harm in ≥ 19/20 DEV seeds.
- NO-GO on either means HALT and report. Nothing is retuned.
- **Result (2026-09-24, `runs/e5-structure-gate/dev.json`): GO.** D1: rate/α 0.91–1.00 at every gated
  α, worst seed-cluster UB95 = 1.13α. D2: P0→K_harm 20/20. Descriptive only: DEV family FDR 0.016,
  total param-edge recall 1.00.
- Confirmatory cost (measured, one seed at n = 24,000): MSCR ≈ 28 s, SHAP-GBDT ≈ 30 s, two-tower ≈ 63 s;
  100 seeds on 8 workers ≈ 30 min, run detached.

## Confirmatory rule (binds on freeze)

**Seeds 900000–900099 (N = 100)**, fresh. Previously used E5 seeds (repo scan 2026-09-24): corpus/rng
seeds 0–19, env/bank seeds 0–5999. The harness asserts disjointness from `USED_E5_SEED_RANGES`. One corpus per seed; all arms
score the same corpus. Truth is used only to score.

| # | Claim | PASS |
|---|---|---|
| S1 | MSCR recovers the harmful edge | P0→K_harm declared in ≥ 95/100 |
| S2 | MSCR controls false discoveries (param family) | family-level FDR = mean over 400 (seed, KPI) families of V/max(R,1); seed-cluster bootstrap one-sided 95% upper bound ≤ 0.05 |
| S3 | MSCR recovers the graph | total param-edge recall ≥ 0.90 (of 600) |
| S4 | SHAP-GBDT proxy misses the harmful edge | P0→K_harm NOT declared in ≥ 95/100 |
| S5 | Two-tower reconstruction misses the harmful edge | P0→K_harm NOT declared in ≥ 95/100 |

All five must pass for the E5 structure claim. Reported, not gated: pooled FDP and FP count per seed
(seed-cluster CIs, never a binomial interval over selected declarations), per-edge recall per arm,
pooled |corr| results, SHAP ratio distribution for P0→K_harm, and a D1-style calibration diagnostic on 20
of the fresh seeds. Wording rule: S2 is empirical evidence consistent with FDR control on this
randomized, noiseless design, not a guarantee.

## Decision layer: DEV record only (no claims)

In raw units the missing-edge planner showed regret 0.378 (CORE), a confound DiD of 0.206 and a
chain-miss regret of 0.311. Standardized, those fall to 0.0095, 0.0008 and 0.020, and the oracle itself
violates K_harm in 40/40 in-gate states. A pre-declared re-balance of the reward-geometry constants
found no point restoring even the trap (0/900 pass M1+M2). At the closest tested point the optimum sits
on the K_harm constraint boundary: under-estimating the harmful slope by 5% loses a constraint step in
18/40 states, while over-estimating by 5% loses none. This is a finite-grid DEV result, not an
impossibility proof. **No decision-regret or "decision-critical" claim is made for E5.**

## Deferred (not claimed; each needs its own pre-declaration)

- **Uncertainty-aware (pessimistic) planner arm:** effect-magnitude estimation with calibrated
  uncertainty and planning against the asymmetric loss. The next study.
- KPI→KPI chain-edge discovery (MSCR makes no KPI→KPI claim; the chain layer is untested for discovery).
- RCoT / pdCor on E5: **not run**. Their gating miss is established on E2 only.
- Discovery under noise, under latent confounding, or from observational data.
- Confound-layer effect estimation (OBS vs DO) as a claim (DEV slopes recorded in `scripts/e5_confound.py`).
- QACM V2Env port (decision-level baseline).

Pre-freeze review: `scratchpad/sol_review_e5_structure_contract.md` (1 BLOCKING seed-guard item,
fixed; wording items applied).

## Implementation pointers

Env `cdd_oran/envs/v2/e5.py`; MSCR `cdd_oran/discovery/mscr.py`; baselines + corpus
`scripts/e5_baselines.py`; moments `scripts/e5_standardization.py`; spine `scripts/e5_spine.py`; design
spec `plans/015-e5-composed-benchmark-design.md`. Structure-gate harness: `scripts/e5_structure_gate.py`
(to be built before the freeze).
