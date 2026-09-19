# E3 multi-hop KPI→KPI chain recovery — truth-free de-risk study

**Date:** 2026-09-06 (run 2026-09-07) · **Branch:** feat/v2 · **Status:** DEV-ONLY, nothing
committed, no tracked source modified, no freeze/pre-registration. This is a
calibration/measurement study (same move as plan 008 for RCoT), **not** an E3 discovery
protocol freeze (that is user-gated).

## Why this mattered (the sharpest unbuilt risk toward E5)

E5 composes E3's temporal KPI→KPI chain. Every prior discovery result validated the *wrong*
direction of the KPI→KPI question: E1 recovered only a single-hop KPI→KPI edge; E2's
RCoT/pdCor were validated only to **reject** KPI→KPI true-negatives. **Nothing had ever been
validated to *recover* a KPI→KPI chain, let alone a multi-hop one.** E3 is the first env with a
real multi-hop lagged cascade. This study measures, truth-free, whether candidate discovery can
recover it — before any (user-gated) E3 discovery freeze.

## Confirmed E3 structure (read from source, not paraphrase)

Source: `cdd_oran/envs/v2/e3.py`, `docs/benchmark/GATE_CONTRACT_E3.md`, `scripts/e3_scm_gate.py`.

SCM (one-step lag; all displayed coefficients exactly 1.0; **linear + noiseless**, like E1):

```
K0(t) = P0(t-1)               # P0->K0
K1(t) = K0(t-1)               # K0->K1   (internal conduit)
K2(t) = P1(t-1) + K1(t-1)     # P1->K2, K1->K2
K3(t) = P2(t-1) + K1(t-1)     # P2->K3, K1->K3
K4(t) = P3(t-1)               # P3->K4   (non-descendant control)
```

TRUE adjacency `(kpi_index, source_index)` with KPI sources `4+i` (source 4 = K0, source 5 = K1):
`[(0,0),(1,4),(2,1),(2,5),(3,2),(3,5),(4,3)]`. TRUE graph = **P0→K0→K1→{K2,K3}** plus P1→K2,
P2→K3, P3→K4. **K1 is the intermediate KPI: simultaneously a descendant of K0 and a parent of
{K2,K3}** — the 2-hop chain the study targets.

Candidate layout (matching E1-v2): per-target regression. Inputs `X = [P0,P1,P2,P3,K0,K1,K2,K3,K4]`
(d=9, params 0–3, KPIs 4–8), targets `Y = next[K0..K4]` (k=5). Candidate shape (5, 9) = 45
candidate edges/seed; 7 are true. Because `source_index` already equals the input index, the true
edge set maps directly onto `(target j, input i)`.

**Load-bearing structural fact (verified):** each E3 target `Y[:,j]` is an *exact* linear function
of the *recorded* inputs (`max|Y-pred| = 0.0` for all 5 targets; `smoke.py`). The chain therefore
does **not** live inside a single regression — it is recovered as *separate per-target edges*
(target K1 ← parent K0) and (target K2 ← parent K1) and (target K3 ← parent K1); the multi-hop
chain emerges by *composing* per-target parent sets across targets.

## Method (truth-free until final scoring)

- **Data:** byte-faithful E3 rows via the env's own `apply_action`/`advance` path, mirroring
  E1 `generate_rows` (same `_ACTION_NS=101` action stream, warmup=2, one-param-per-step random
  policy, noise off). **10 seeds** (env_seed 0–9), **n=4000 rows/seed** (250 episodes × 16 steps).
- **FULL conditioning** = the E1-v2 method as-is: `|partial_corr(input i, target j | all other
  inputs)|`.
- **LOW-ORDER / mediator-aware** = PC-stable-style skeleton: `score(i,j) = min` over conditioning
  subsets `S` (|S| ≤ 2, S ⊆ other inputs) of `|pcorr(i,j|S)|`. An edge any subset can separate
  scores ≈ 0; the arg-min records the separating set.
- **Threshold = permutation null** (never truth): permute target rows (breaks X→Y), take the
  family-wise max score per permutation, threshold = 99th-quantile of maxima. Computed per seed,
  per method.
- **Persistence-driven collinearity is real** (the plan-007 hazard's precondition): the
  one-param-per-step policy makes P0(t), K0=P0(t-1), K1=P0(t-2) collinear — input correlations
  **P0–K0 = 0.74, K0–K1 = 0.74, P0–K1 = 0.57** (`smoke.py`). So the over-conditioning hazard is
  genuinely *set up*, not assumed away.

Truth (`E3V2Env.TRUE_ADJACENCY`) is read only in the final scoring block.

## Results

### Per-edge recall (10 seeds, n=4000/seed)

| True edge | target | parent input | recall FULL | recall LOW-ORDER | mean score FULL | mean score LOW-ORDER | STOPs FULL |
|-----------|:------:|:------------:|:-----------:|:----------------:|:---------------:|:--------------------:|:----------:|
| P0→K0     | K0 | P0 | **1.00** | 1.00 | 1.000 | 1.000 | 0 |
| **K0→K1** (hop 1) | K1 | K0 | **1.00** | 1.00 | 1.000 | 1.000 | 0 |
| P1→K2     | K2 | P1 | **1.00** | 1.00 | 1.000 | 0.563 | 0 |
| **K1→K2** (hop 2) | K2 | K1 | **1.00** | 1.00 | 1.000 | 0.461 | 0 |
| P2→K3     | K3 | P2 | **1.00** | 1.00 | 1.000 | 0.569 | 0 |
| **K1→K3** (hop 2) | K3 | K1 | **1.00** | 1.00 | 1.000 | 0.461 | 0 |
| P3→K4     | K4 | P3 | **1.00** | 1.00 | 1.000 | 1.000 | 0 |

**Overall (both methods): recall 1.00, precision 1.00, F1 1.00** (70/70 true edges recovered,
**0 false positives**, 0 collinearity STOPs across all 10 seeds × 45 candidates). Permutation-null
thresholds: FULL ≈ 0.056, LOW-ORDER ≈ 0.045 (99th-quantile family-wise maxima) — far below
true-edge scores and far above decoy scores; threshold choice is not delicate.

### Plan-007 over-conditioning hazard: **REFUTED, with numbers**

Full conditioning does **not** block any chain edge. All three chain KPI→KPI edges (K0→K1, K1→K2,
K1→K3) score **exactly 1.0** under full conditioning *despite* the real P0/K0/K1 collinearity
(0.57–0.74). Mechanism: because each target is an *exact* linear function of its recorded parents,
residualizing the target on the controls removes the mediator's shared variance from the target and
from the input *identically*, so the partial-correlation ratio stays 1.0 (scale/collinearity
invariant). **Zero STOPs** confirms the input residual never vanished — collinearity was strong but
never perfect. Conditioning on the descendant/mediator K1 (present in every downstream target's
control set) never opened a collider or blocked the true edge.

### Direct-vs-ancestry discrimination

Ancestry decoys (skip-mediator non-edges) are the discriminator. Marginal (order-0) correlation is
**fooled**; conditioning **fixes it**:

| Decoy (non-edge) | marginal score | FULL score | LOW-ORDER score | selected? | separating set (seed 0) |
|------------------|:--------------:|:----------:|:---------------:|:---------:|:-----------------------:|
| P0→K1 (skip K0)  | 0.751 | **0.000** | 0.000 | no (0/10) | {K0} — the mediator |
| K0→K2 (skip K1)  | 0.529 | **0.000** | 0.000 | no (0/10) | {P1, K1} — the mediator |
| P0→K2 (2-skip)   | 0.398 | **0.000** | 0.000 | no (0/10) | {P1, K1} |
| K0→K3 (skip K1)  | 0.531 | **0.000** | 0.000 | no (0/10) | {P2, K1} — the mediator |
| P0→K3 (2-skip)   | 0.403 | **0.000** | 0.000 | no (0/10) | {P2, K1} |

Head-to-head at target K2: true **K1→K2 = 1.000** (FULL) vs spurious **K0→K2 = 0.529 marginal →
0.000 FULL/LOW-ORDER**. Conditioning on the mediator K1 correctly separates the ancestry shortcut.
**Full conditioning is protective here, not harmful** — it is the *stronger* discriminator (crisp
1.0 vs 0.0), while the low-order min-statistic is the more attenuated one.

## Verdict

- **Is the E3 multi-hop KPI→KPI chain recoverable?** **Yes — fully.** All 7 true edges,
  including both chain hops K0→K1 and K1→{K2,K3}, at **100% per-edge recall over 10 seeds, 0 FP,
  0 STOPs**.
- **Under which conditioning strategy?** **Both.** FULL conditioning (the E1-v2 method
  unmodified) and the low-order/mediator-aware PC-skeleton each hit recall 1.0 / precision 1.0.
  FULL gives the crispest separation (true 1.0, decoy 0.0).
- **Does full conditioning break any chain edge?** **No** — the plan-007 over-conditioning hazard
  is refuted with numbers, even though its collinearity precondition is genuinely present.

### Concrete working recipe (for a later, user-gated pre-registration)

1. Rows: env's own `apply_action`/`advance`, E1 conventions (warmup 2, one-param-per-step policy,
   noise off), n≈4000/seed, ≥10 seeds. Per-target candidate layout, inputs `[params | prev_KPIs]`.
2. Score: `|partial_corr(i, j | all other inputs)|` (FULL conditioning; E1-v2 as-is). Treat a
   vanishing input residual as a recorded STOP (none occurred here) and a vanishing target residual
   as rho:=0 (edge absent), per the E1-v2 branch order.
3. Threshold: family-wise **permutation null** (permute target rows, 99th-quantile of per-permutation
   maxima). Data-driven, truth-free. The huge score gap (≈1.0 vs ≈0.0) makes it robust.
4. The chain is assembled by composing per-target parent sets across targets — no chain-spanning
   regression is needed.

### Honest caveats

- **Noiselessness carries the crisp separation.** E3 is exact-linear + noiseless, so partial
  correlations are exactly 1.0/0.0 and full conditioning's collinearity-invariance is exact. **E5
  turns noise ON.** With noise, the FULL partial correlation for a chain edge stays near 1 (still
  collinearity-robust), but the **low-order min-statistic already shows attenuation here** (chain
  edges 0.46 vs direct-param 0.56–1.0 vs threshold 0.045) — under noise that low-order margin will
  shrink first. **Recommendation: prefer FULL conditioning for the chain; re-run this calibration
  with E5-level process/observation noise before trusting either at E5.**
- **This validates stage-1 skeleton recovery only** — the *edge set*. It does **not** exercise the
  downstream E5 spine (discovered structure → multi-intervention world-model → planner → rejects a
  locally-good/globally-harmful shared-knob action). Per the E5-direction synthesis, the
  load-bearing blocker remains the *multi-intervention world model on the discovered structure*,
  not discovery. This study removes the "can discovery even find the chain" risk; it does not
  remove that one.
- Orientation is free here (architecture forbids within-step KPI→KPI; all fan-out is lagged /
  time-oriented), so only skeleton recovery + ancestry discrimination were at issue — both pass.
- pooled-across-episode collinearity never became perfect at n=4000 (0 STOPs); a much smaller n or a
  policy that freezes a param entirely could trip the STOP branch — untested here.

## Provenance / reproducibility

- Scratchpad (uncommitted): `…/scratchpad/e3_chain_derisk/` — `smoke.py` (generation + exactness +
  collinearity), `study.py` (full study), `results.json` (raw output).
- Single-core (`OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`); venv `.venv/Scripts/python.exe`. Runtime
  ≈ 94 s for 10 seeds. E2 files/dirs untouched.
