# E2 operating-point redesign — PRE-REGISTRATION (DRAFT, nothing frozen)

**Status:** DRAFT pre-registration for review. Nothing here is frozen, committed, or scored against any
discovery / world-model / planner method yet. On approval it is frozen via the two-commit ordering in §8
BEFORE any method is re-scored.

## 1. Motivation (verified finding)
E2's shared-knob decision is currently a **frozen preregistered NULL**: `mean_positive(gap_norm)=0.0162 <
tau_E2=0.10` (`tests/test_e2_decision_gate.py:53-63`, `xfail(strict=True)`). Mechanism (verified, truth-free):
K5 is *reachable* across its −25 threshold at most committed states, but reachability ≠ decision conflict.
The K5-satisfying P0 band is **too wide** — under the frozen `id_ranges[6]` (P6 width) `(-60, 65)` the band
half-width is 4.7–52.8 P0-units — so it swallows the K0/K1/K2 argmax: satisfying K5 either costs nothing or
costs everything, so the K5-blind (decoy) and K5-aware (oracle) choices coincide at all but 7/32 states. The
trap does not bite → E2 does not pose its own question.

## 2. Pre-registration discipline
- The change is derived **truth-free**: only the env geometry vs the locked objective was examined; **no
  discovery / world-model / planner output was scored** to choose it.
- `tau_E2 = 0.10` is **UNCHANGED**. We do not tune the acceptance threshold; we repair the env so the
  scientific conflict exists. The strict `xfail` re-reddens if τ is ever tuned to pass.
- Acceptance is gated on **degeneracy guards** (§5) proving a genuine trade-off, not a number maximised.

## 3. The change (exactly one tuple)
`cdd_oran/envs/v2/e2.py`, `id_ranges` (`:56-65`):

```
id_ranges[6]:  (-60.0, 65.0)  ->  (1.5, 4.0)      # P6, the K5 Gaussian width
```

Everything else is **frozen-identical**. Justification for minimality: P6 and P7 enter **only** K5's
mechanism (`e2.py:115-120`; verified — K0–K4 reference none of them), so re-ranging P6 moves K5's operating
point and leaves K0–K4 **bit-identical** (empirically confirmed: shared params drawn from coordinate-keyed
tape are unchanged, k0..k4 identical). P7 is **left untouched** — narrowing it destroys the negative-control
class (K5-flat states), which the gate needs for `mean_negative`.

Effect of the change: band half-width becomes 1.2–3.3 P0-units (full width 2.4–6.6). Because the P0 decision
grid is `linspace(-100,100,101)` (step **2.0**, `v2_regret.py:39-40`), a full width ≥ 2.4 keeps the
satisfying band **grid-resolvable at every state** — the reason (1.5, 4.0) is chosen over the sharper
(1.0, 4.0) (whose band dips to 1.6 full width < grid step, risking a silently-unsatisfiable = degenerate
trap).

## 4. Derived standardization constant
K5's frozen `(mean, std)` must be re-derived because K5's distribution changed (K0–K4 constants stay). Derive
via the frozen provenance procedure — `get_env_ii_mean_std(param_ranges = new ID ranges, seed = <frozen
provenance seed>, num_samples = 1_000_000)` over `compute_kpis` — and lock the KPI-5 entry only:

```
KPI_MEAN_STD[5]:  (-10.140367, 12.583744)  ->  (re-derived at freeze; exploration estimate (-0.961945, 4.788258))
KPI_MEAN_STD[0..4]:  UNCHANGED
```

The exact value is produced deterministically at freeze (§8), not taken from the exploration estimate.

## 5. Pre-registered acceptance criteria (τ unchanged)
Run `scripts/e2_decision_gate.py` (32 K5-positive + 32 K5-negative committed states, `env_seed 0..4095`
scan, 2-advance lag, panel {K0,K1,K2,K5}). Accept the redesign iff ALL hold:
- **A. Positive control:** `mean_positive(gap_norm) ≥ tau_E2 = 0.10`.
- **B. Negative control:** `mean_negative(gap_norm) ≤ tol = 0.01`.
- **C. Factor-removal:** dropping xApp4/K5 from the panel collapses the positive gap to `≤ tol` (the gap
  localises to the shared-control factor).
- **D. Feasibility:** 32 positive + 32 negative states found.

Degeneracy guards (per-state, must all pass — a genuine trade-off, not a degenerate trap):
- **Reachable-satisfied:** some P0 gives true K5 ≤ −25 at every scored state.
- **Not free-at-optimum:** at the K0/K1/K2-optimal P0, true K5 is violated at a broad majority of states.
- **Genuine cost:** satisfying K5 requires moving P0 off the K0/K1/K2 optimum (a real utility cost), with
  both a satisfiable and a violated P0 regime coexisting per state.

## 6. Predicted outcomes (truth-free exploration; to be CONFIRMED at scoring, not assumed)
`mean_positive(gap_norm) ≈ 0.204`; conflict at ≈ 26/32 (81%) positive states; `mean_negative = 0`;
factor-removal = 0; feasibility 32/32; all degeneracy guards pass. These are predictions; the frozen run
in §8 is authoritative.

## 7. Consequences
- **New E2 benchmark version.** Datasets change (K5 distribution + standardization), so existing frozen E2
  artifacts (RCoT-v1/v2 discovery + recovery) are on the **old** env and must be **re-run** on the
  redesigned E2 after freeze. Their old records stand as historical, not comparable.
- The sharper K5 bump may change P0→K5 discoverability (co-parent-gated edge, memory
  `e2-harmful-edge-discovery-gap`) — re-measured on the new env.
- On a passing frozen run, `test_e2_decision_gate.py`'s positive-control test flips from `xfail` to an
  expected pass (remove the `xfail`, keeping the same τ).
- **Downstream envs:** E5 composes E2's fan-out trap, so this repair is a prerequisite for E5 being live.

## 8. Freeze procedure (only after approval)
Two-commit ordering (mirrors the discovery-protocol freezes):
1. Commit this doc with §3/§4 finalised → its SHA becomes `protocol_commit`.
2. Derive `KPI_MEAN_STD[5]` at canonical n=1e6; apply the `id_ranges[6]` + `KPI_MEAN_STD[5]` change; record
   `protocol_commit`; run the §5 gate.
3. Commit the env change + as-run gate report. If §5 fails, the redesign is rejected (do NOT re-tune to pass
   — re-enter design at §1 with a fresh pre-registration).
