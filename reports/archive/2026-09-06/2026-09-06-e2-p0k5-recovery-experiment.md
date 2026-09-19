# E2 harmful-edge (P0→K5) recovery experiment — DEV prototype, truth-free

**Date:** 2026-09-07 (night session). **Status:** DEV/analysis prototype. Nothing frozen, committed, or
pre-registered. Truth read ONLY at scoring. Env = live-trap E2 (frozen `9d87a60`). Method design is
truth-free; scored against `E2V2Env().true_adj_matrix()` only in `score.py`.

## Motivation
On the live-trap E2, frozen RCoT-v2 discovery recovers the 3 beneficial fan-out arrows P0→{K0,K1,K2} 10/10
but the globally-HARMFUL P0→K5 arrow **0/10** (co-parent-gated edge: `k5 = -35·exp(-((p7+p0-25)²)/(2·safe_exp(p6)²))`,
true parents {P0,P6,P7}; P0 buried additively with P7 inside a Gaussian bump of width P6). The downstream
spine (world-model + decision) is already proven to reject the harmful action **given complete structure**,
so recovering this one edge is the sole remaining blocker to the E5 thesis on E2. Prior diagnosis: the
marginal P0→K5 signal is ~1% variance and fully nonlinear; naive interventional `do(P0)` cannot help because
E2 params are mutually-independent exogenous roots (do(P0) == observational marginal — no confounding to
break); the edge is recovered only by **conditioning on / controlling the co-parents**.

## Methods compared (all truth-free; BH-FDR q=0.05 per target)
- **M1 — frozen RCoT-v2 baseline** (the shipped E2 selector). `baseline_results.json`.
- **M2 — do(P0)-marginal** (control): marginal nonlinear dependence η²(K5|P0) with a global-permutation null.
  Under independent exogenous params this equals the interventional do(P0) marginal test.
- **M3 — STRATIFIED [PRIMARY]:** `S* = max over stratifier g (g over ALL candidates, g≠i) of the conditional
  η²(target | cand_i binned, pooled within g's strata)`, with a **conditional-permutation null** that
  permutes cand_i *within each g-stratum* and **re-maxes over g** (a proper max-statistic CI test;
  multiplicity over the stratifier handled by the max). **Truth-free:** g ranges over all candidates — the
  method is not told P6/P7 are the co-parents; it *discovers* that stratifying by them opens the gate.
- **M4 — recall-first superset + prune:** deliberately over-select on the marginal (BH q=0.20) then intersect
  with the M3 conditional selection.

Data: real E2 DGP (`cdd_oran.e2slice.dataset.generate_rows`, n=4000/seed, 10 seeds, sampling_seed=0),
single-core. Runner `scratchpad/p0k5_full.py`; scorer `scratchpad/score.py`; results `m3_results_full.json`.

## Results (10 seeds, BH q=0.05)

| method | P0→K5 | P0→{K0,K1,K2} | NCP recall /16 | KPI→KPI FP | param FP |
|---|---:|---:|---:|---:|---:|
| M1 RCoT-v2 baseline | **0/10** | 30/30 | ~10–11 | 0.60 | — |
| M2 do(P0)-marginal (control) | **0/10** | 30/30 | 11.2 | 0.70 | 0.20 |
| **M3 stratified [PRIMARY]** | **10/10** | 30/30 | **15.3** | 0.30 | 0.30 |
| M4 superset+prune | **0/10** | 30/30 | 11.2 | 0.00 | 0.00 |

P0→K5 diagnostics (per seed): stratified cond-perm p = **0.0033 on all 10 seeds** (the 1/300 permutation
floor — observed statistic beats all 299 permutations every seed). Marginal η²(K5|P0) = 0.001–0.005
(vanishing); marginal-perm p = 0.08–0.98 (never significant); P0→K5 in the marginal superset (q=0.20) =
**0/10** (too weak to admit even at a loose threshold).

## Interpretation
- **The harmful co-parent-gated edge IS recoverable from purely OBSERVATIONAL data** — no interventional data
  required. The lever is the **conditioning / stratification strategy**, matching the mechanistic prediction
  (condition on the co-parents → η² ≈ 1.0).
- **Both controls fail, as diagnosed:** interventional-marginal (M2) draws the same weak marginal;
  recall-first marginal-superset (M4) can't even admit the edge because the marginal signal is essentially
  zero — superset-then-prune is the wrong frame for a *gated* edge.
- **M3 is not just flagging everything:** it *improves* NCP recall (15.3/16 vs baseline ~10) while keeping
  KPI→KPI FP low (0.30/seed) — the signature of a genuinely more powerful, calibrated test.
- **Closes the E2 loop:** observational stratified discovery → complete structure → (already proven)
  world-model + decision reject the locally-attractive/globally-harmful action. The E5 thesis is now
  demonstrable end-to-end on E2 with *discovered* (not oracle) structure.

## Caveats & what stays USER-GATED
- Truth-free **prototype**, not a frozen method. The max-over-g stratified test is more expensive than a
  single-conditional CI and needs its own **calibration study** (proper FP control across a null at more
  seeds; sensitivity to strata count NC, within-bin NB, min-per-stratum).
- Promoting it to the **live E2 discovery method = fresh pre-registration** (SPEC anti-p-hacking discipline),
  which is a user decision — do NOT freeze from these numbers.
- Robustness: p sits at the permutation floor, so B=299 cannot distinguish "just below floor" seeds; a
  pre-registration should set B and the strata hyper-parameters up front and report exact FP calibration.

## Recipe for a future pre-registration (if the user greenlights)
1. Freeze the stratified statistic + null (NC strata bins, NB within-bin, min-per-stratum=40, B, BH q) as a
   protocol doc → protocol_commit.
2. Pre-declare a truth-free FP-calibration gate on a null/permutation testbed (KPI→KPI true-negatives + a
   no-edge control) proving nominal FP at the chosen hyper-parameters.
3. Only then score recovery on the frozen E2 (and re-check it does not regress the beneficial edges / RCoT-v2
   KPI→KPI rejection).
