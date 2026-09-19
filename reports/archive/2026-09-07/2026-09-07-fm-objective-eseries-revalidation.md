# FM-1 / FM-2 objective re-validation on the E-series (truth-free)

**Date:** 2026-09-07 · **Branch:** feat/v2 · **Status:** DEV/analysis, gitignored, nothing committed, no tracked source modified, no freeze/pre-registration.
**Closes the gate in:** plans/011 §2 ("re-validate FM-1/FM-2 on the E-series objective") and plans/012 step 1 (legacy hard-delete gate).

Prototype (scratchpad, not committed):
`…/scratchpad/fm_eseries/fm1_eseries.py` (λ-grid realized-value curve) and
`…/scratchpad/fm_eseries/fm1_flatstruct.py` (flat-structure diagnostic).
Both reuse the FROZEN E2 machinery verbatim — `scripts/e2_decision_gate.py` (bank + panel), `cdd_oran/analysis/v2_regret.py` (locked objective `R`, H=1 rollout), `cdd_oran/envs/v2/e2.py` (true + decoy world models). Single-core thermal env set.

---

## Headline

- **FM-1 does NOT transfer to the E-series as a realized-value gain — NULL-to-harmful.** The flat-hinge indifference *structure* the legacy fix targets **does exist** in the E-series objective, but adding the legacy `utility_weight` z-utility term to the E-series decision objective **never improves realized R** on the true env, across the pre-declared λ grid {0, 0.1, 0.25, 0.5}. Under a perfect world model it is neutral then mildly harmful; under the decoy world model it is strictly harmful. The legacy Env I gain (−0.083 → +0.303) **does not reproduce**.
- **FM-2 (OOD/ensemble-disagreement penalty) is NOT measurable in the current E-series decision arm** and would need a new module (an *ensemble* world model). The E-series sim-level gates roll a single **analytic** true/decoy model, so no ensemble-disagreement signal is defined. Feasibility note below; **no module built** (per instructions).

---

## 1. Applicability: does the E-series objective have the FM-1 flat-hinge structure?

**Yes, structurally.** The E-series primary objective `R` (SEMANTICS §2, `v2_regret.reward` / `counterfactual_metrics._objective_for_value`) is the *identical* clamped hinge cost as the legacy `cost.py`:

```
cost(k) = Σ_i w_i · max(θ_i − u_i, 0) · s  −  (Σ_i ok_i)²      (w_i=1, s=10, R = −cost)
```

The `max(·,0)` clamps distance to 0 once an xApp is satisfied, and the `(Σ ok_i)²` bonus counts only the *number* satisfied — neither term rewards how far *above* threshold an xApp sits. So the "planner is indifferent past satisfaction" precondition is present. The `(Σ ok_i)²` term does **not** change this: it is a discrete count, flat with respect to over-satisfaction of a fixed satisfied set.

**Structural diagnostic** (64 banked E2 states, full panel {xApp0,1,2,4}, true-model P0 grid — `fm1_flatstruct.py`):

| quantity | value |
|---|---|
| mean R-optimal tie-set size (of 101 grid pts) | 1.81 (min 1, max 14) |
| states with a FLAT R-optimal region (>1 grid pt tied at max R) | **13 / 64** |
| …of those, tie-set has z-utility spread >1e-6 (fix has something to bite) | **13 / 13** |
| max satisfied-count at the R-optimum | mean 1.77 of 4 (min 0, max 4) |

So a genuine flat R-optimal plateau exists in ~20% of states, and in every such state the z-utility varies across the plateau — the FM-1 fix is *applicable* (it has a tie to break). Note also that mean satisfied-count at the optimum is only 1.77/4: these are genuine conflict states where R is usually distance-dominated, so the flat plateau is exercised but is not the common regime.

## 2. FM-1 λ-grid realized-value curve

Decision objective under test: `R_λ(k) = R(k) + λ·Σ_i sign_i·u_i(k)` (exactly the legacy Option A term, `cost.py:104-112`, ported to the locked `R`). Planner picks `argmax_v R_λ` over the frozen 101-pt P0 grid using a **world model** to predict the scored latent `k2`; **realized value is the locked `R` scored on the TRUE env only** (E-series realized-scoring discipline). 64 states (32 positive + 32 negative bank). Higher realized R is better.

**Arm A — perfect world model (planner scores on the true env):**

| λ | realized_pos | realized_neg | realized_all | norm_regret_pos | off-optimum frac |
|---:|---:|---:|---:|---:|---:|
| 0.0  | −42.3334 | −60.8470 | −51.5902 | 0.00000 | 0.0000 |
| 0.1  | −42.3335 | −60.8470 | −51.5903 | 0.00000 | 0.0625 |
| 0.25 | −42.3518 | −60.8517 | −51.6018 | 0.00026 | 0.1094 |
| 0.5  | −42.3789 | −60.8517 | −51.6153 | 0.00072 | 0.1250 |

**Arm B — decoy world model (planner scores on the P0→K5-omitting decoy; realized on true env):**

| λ | realized_pos | realized_neg | realized_all | norm_regret_pos | off-optimum frac |
|---:|---:|---:|---:|---:|---:|
| 0.0  | −57.4800 | −60.8470 | −59.1635 | 0.19002 | 0.3906 |
| 0.1  | −58.0076 | −60.8470 | −59.4273 | 0.19397 | 0.4219 |
| 0.25 | −58.0076 | −60.8517 | −59.4296 | 0.19397 | 0.4375 |
| 0.5  | −58.0076 | −60.8517 | −59.4296 | 0.19397 | 0.4375 |

**Reading:**
- **Perfect model:** λ=0 already realizes the oracle (norm_regret_pos = 0, off-optimum = 0). Raising λ is flat to 5 dp at 0.1, then **strictly worse** (realized_pos −42.333 → −42.379; off-optimum 0 → 0.125). Breaking the flat R-plateau by z-utility moves the pick to points that are *equal* in realized R at best and *below* the R-optimum once λ is large enough to overpower `R` — because realized R is the same flat `R`, the "utility left on the table" is worth **zero** realized R.
- **Decoy model:** λ=0 already carries large regret (norm 0.19) from the decoy's structural error; adding the utility term makes it **worse**, not better (realized_pos −57.48 → −58.01). Over-satisfaction margin does not buffer a *structural* wrong-model bias — it amplifies it.

## 3. Why the legacy gain doesn't transfer (mechanism)

The legacy FM-1 gain required a condition the E-series decision arm does not have. `utility_weight` helps realized value only when **over-satisfying on the decision model raises realized value** — which needs *either* a realized objective that itself rewards over-satisfaction, *or* a **noisy learned model** whose estimation error a satisfaction margin buffers (over-satisfy on the model → still satisfied on the true env → more `ok_i` → higher `(Σok)²`). The legacy validation was on a *learned ensemble* planner where that margin-vs-noise trade is real.

In the E-series:
1. The **realized** objective is the *identical* flat `R`, so over-satisfaction has **zero realized value by construction** (Arm A proves it: gain = 0 at the plateau, harm beyond).
2. The **decision** arm rolls an **analytic** true/decoy model, not a learned ensemble — there is no estimation noise for a margin to buffer; the decoy's error is structural, and margin amplifies it (Arm B: gain < 0).

So the direction recorded in plans/011 ("flat objective ⇒ leaves utility on the table") is **structurally real** but its *fix does not produce a realized-value gain under the E-series objective + decision arm*. This is a valid, load-bearing NULL for the gate.

## 4. FM-2 (OOD model trust) — feasibility, not measured

FM-2's remedy is an **ensemble-disagreement penalty** (`ood_threshold`), which requires an *ensemble* of world models whose spread flags off-distribution states. The E-series decision gates (`e2_decision_gate.py`, `e3_decision_gate.py`) roll a **single analytic model** per arm (true, decoy, or truncated). There is **no ensemble and therefore no disagreement signal** in the current E-series decision arm, so lifting an OOD-trust penalty is **not measurable here without a new module**. Per instructions I did **not** build one.

Truth-free feasibility note: the decoy-vs-true gap in Arm B *is* the "model confidently wrong off-distribution" phenomenon FM-2 describes (the decoy is confidently wrong about P0→K5, driving norm_regret 0.19 and off-optimum 0.39). An OOD penalty needs ≥2 models that *disagree*; the E-series has two candidate structures (true, decoy) but the planner is **handed one**, never an ensemble to compare. Wiring FM-2 into the E-series would mean giving the decision arm an *ensemble* world model (e.g. bootstrapped/structure-perturbed members) and penalizing member spread — a **user-gated module**, not an ablation of an existing knob. It is the same "world-model capability, not a planner/objective knob" conclusion already recorded for FM-3 (plans/011 §2; memory `e2-harmful-edge-discovery-gap`).

## 5. Bottom line for plans/011 & plans/012

- **FM-1:** transfers **structurally** (flat plateau exists, fix is applicable) but **NOT as a realized gain** — null under a perfect model, harmful under the decoy. Update plans/011 §2 to note the legacy magnitude (+0.303) is legacy-only; on the E-series objective the utility_weight fix is neutral-to-harmful because realized `R` is the same flat hinge and the decision arm is a true/decoy analytic model (no learned-model noise to buffer). The lever survives only if a future E-series arm scores realized value on something that rewards over-satisfaction OR uses a *learned/ensemble* world model.
- **FM-2:** **not testable** in the current E-series decision arm (no ensemble). Needs a user-gated ensemble-world-model module; do not credit the legacy +0.297 to the E-series.
- **plans/012 gate:** the E-series objective re-validation asked for is now DONE and is a NULL for FM-1 / not-applicable for FM-2 — an honest completion of the gate, not a positive port. Whether that is "sufficient to unblock legacy hard-delete" is the user's call; the borrowed-finding reference (legacy Env I magnitudes) is now reproduced/adjudicated on the E-series here.

### What still needs a user-gated decision/module
1. **FM-2 measurement** needs an *ensemble* E-series world model in the decision arm (not built).
2. If FM-1 is still wanted as a lever, it needs an E-series arm with a **learned/imperfect (noisy) world model** — the only regime where a satisfaction margin can pay off. The current sim-level gates deliberately use analytic true/decoy models, so FM-1 cannot be exercised in its beneficial regime there.
