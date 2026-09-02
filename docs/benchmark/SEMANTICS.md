# Benchmark Semantics Contract (preregistered)

**Status:** LOCKED design contract for the v2 benchmark suite. Preregistered *before* any
evaluator/regret/gate code is written. Implements PD2/PD3/PD4 and resolves blockers
B1/B2/B3 and the RNG/CRN (MAJOR 6/7) and scoring-target (MAJOR 10) findings from
`.herdr/reports/review-tonight-plan.md`.

This document is normative. Where it disagrees with existing code, the code is legacy and
is migrated (or explicitly kept as a legacy diagnostic) per §9. Nothing here permits
post-hoc selection among metrics: the primary objective, the comparator, and the pairing
rule are all fixed here.

Anchor citations use `file:line` against the tree at the current SHA.

---

## 0. Notation and timeline

The environment already delays parameter → KPI response by one step
(`cdd_oran/envs/base.py:192`: `update_kpi` reads `self.prev_params`, `self.prev_kpis`).
We preserve that latency (PD2) and align the horizon, simulator, and oracle to it. We do
**not** remove the latency.

Fix the following indexing, read directly off `BaseORANEnv.step` / `.reset`:

- `p_t` — parameter vector **in effect after step `t`** (`new_params`, `base.py:191`).
  `reset` establishes `p_0` (`base.py:177`) and `k_0 = 0` (`base.py:178`).
- `k_t` — **raw latent** KPI vector produced *at* step `t`:
  `k_t = f(p_{t-1}, k_{t-1})` (`base.py:192`, using `prev_params`, `prev_kpis`). `k_t` is the
  raw `KPI.value` (`base.py:79`), **before** any standardization. Standardization is a
  separate step done *inside* each xApp utility, written `u_i(k_t)` (defined in §2) — never
  fold standardization into `k_t` itself.
- `a_t` — the action applied at step `t`. It writes exactly one param coordinate into
  `p_t` (`base.py:187-190`), so `a_t` first influences KPIs **at step `t+1`**, through
  `k_{t+1} = f(p_t, k_t)`.
- `ŷ_t = k_t + ε_t` — the **observed** KPI, `ε_t ~ N(0, 0.01)` (`base.py:206`). Observation
  noise is added only to the state copy, never to `k_t`.

The one-step actuation latency is therefore the single fact every downstream definition
must respect: **`a_t` is scored by `k_{t+1}`, never by `k_t`.** Because a single
`BaseORANEnv.step(a_t)` call both writes `p_t` *and* returns `k_t = f(p_{t-1}, k_{t-1})`
(`base.py:184-197`), one `step` call can never reveal the KPI that `a_t` influences. The
executable decomposition of `step` that makes the scored window reachable is fixed in §1.1.

---

## 1. Transition timing and the scored trajectory (PD2, B1)

### 1.1 Executable API and the call list (resolves B1)

The scored trajectory is defined against a **split of `step` into two operations** — the
one explicit mechanism this contract commits to (option (a) of the review). Legacy
`step(a)` is exactly their composition, so no legacy behavior changes:

- **`apply_action(a)`** — writes the one param coordinate selected by `a` into the *pending*
  parameter vector (`base.py:187-190`). It does **not** advance KPIs and does **not** move
  `prev_params` / `prev_kpis`.
- **`advance()`** — computes `k = f(prev_params, prev_kpis)` (`base.py:192`), then commits
  `prev_params ← pending params`, `prev_kpis ← k`. Called with no preceding `apply_action`,
  it re-rolls the *same* params forward, so **`advance()` alone never mutates parameters** —
  it only reveals the delayed KPI response to whatever action was last applied.
- **`step(a) ≡ apply_action(a); advance()`** — the legacy call, unchanged.

The planner observes the **committed pre-decision state `s = (p_0, k_0)`** — i.e.
`(prev_params, prev_kpis)` as left by `reset`/the prior advance — *before* emitting any
action. It never observes a KPI produced by its own not-yet-applied action. The primary
action class is **open-loop**: at the decision epoch the planner emits the full sequence
`a_1, …, a_H` (closed-loop feedback is deferred, §10).

**Canonical call list for a horizon-`H` decision** (committed start state `(p_0, k_0)`):

```
for h = 1 … H:
    apply_action(a_h); advance()      # produces k_h = f(p_{h-1}, k_{h-1})
advance()                             # TERMINAL, no action → produces k_{H+1} = f(p_H, k_H)
```

That is `H` `apply_action` calls and `H+1` `advance` calls. The first advance yields
`k_1`, a **warm-up** that reflects only the pre-decision `p_0` and is **unscored**. The
**terminal advance carries no action, so it mutates no parameter**; it exists solely to
reveal `k_{H+1}`, the delayed response to the last applied action `a_H`.

### 1.2 Scored window
The outcomes attributable to a decision made over horizon `H` are the latent KPIs
**`k_2, …, k_{H+1}`** — one per applied action, each the one-step-delayed response to `a_h`
(`k_{h+1} = f(p_h, k_h)`). Equivalently, with the decision epoch at `t`, the window is
`k_{t+1}, …, k_{t+H}`. The warm-up `k_1` is excluded. The oracle, the simulator rollout,
and the planner's regret are all scored on **this same window via the identical call list of
§1.1** — same number and order of `apply_action`/`advance` calls, only the action *values*
differ across arms. This is the alignment that resolves B1: environment return and oracle
return are both delayed by one step and produced by one call timeline, so they are
comparable, and CRN coordinates (§4) line up across arms by construction.

### 1.3 Per-step and cumulative objective
Let `R(k)` be the primary scalar objective on the latent KPI (defined in §2). The episode
return is

```
G = Σ_{h=1..H} γ^{h-1} · R(k_{t+h})            (γ = 1, fixed; see §2)
```

`H = 1` is the special case that reduces to a single delayed transition. It is **not** the
instantaneous static map used by legacy evaluation (`base.py:222-233` rebuilds KPIs from
the proposed params with no delay); that map is retired for scoring (§9).

### 1.4 WORKED EXAMPLE — H = 1 (single delayed decision)

Minimal instance (clean single-KPI mechanism obeying the `base.py` timing exactly;
`R(k)=k`, direction = maximize):

```
k_t = g(p_{t-1}),   g(p) = 1 − (p − 2)²
reset:  p_0 = 0,  k_0 = 0
action candidates (illustrative grid): p ∈ {0, 1, 2, 3}
```

Horizon `H = 1`, committed start `(p_0 = 0, k_0 = 0)`, action sequence `(a_1)`. Call list
per §1.1 (identical for planner and oracle; only `a_1`'s value differs):

```
apply_action(a_1); advance()   → k_1 = g(p_0) = g(0) = −3    [warm-up, UNSCORED]
advance()          (terminal)  → k_2 = g(p_1)                [SCORED]
```

`1` `apply_action` + `2` `advance` calls; the terminal advance applies no action and leaves
`p_1` unchanged. Scored window = `{k_2}`. Scoring the warm-up `k_1` would make `a_1`
irrelevant — the concrete statement of the latency (exactly the mismatch B1 warned about).

| arm      | action `a_1 = p_1` | `k_2 = g(p_1)` | return `G` |
|----------|--------------------|----------------|------------|
| planner  | 1                  | `1−(1−2)² = 0` | 0          |
| oracle   | 2 (argmax over grid)| `1−(2−2)² = 1`| 1          |

- **Regret** = `G_oracle − G_planner = 1 − 0 = 1` (§3).
- **Simulator check:** replaying `a_1 = 1` on a clone from the same `env_seed`/episode
  yields `k_2 = 0`, byte-identical to the value the planner is scored on. Planner-scored,
  simulator-rollout, and oracle all evaluate `k_2 = g(p_1)` under one dynamics → consistent.
- **Observation noise:** `ŷ_2 = k_2 + ε_2`, with `ε_2` keyed to coordinate
  `(env_seed, episode, time=2, variable=kpi)` (§4). It is identical across the two arms and
  cancels in the latent objective; it appears only in the secondary observed metric (§2).

### 1.5 WORKED EXAMPLE — H > 1 (delayed harm through a KPI→KPI chain)

This example uses a **two-KPI chain** so that an action's *downstream penalty* only surfaces
two steps later — the case a myopic per-step oracle cannot see (motivates §3/B2). Env I
already contains such chains (`K2 ← K0`, `K3 ← K1`; `env_i.py:57-61`, `update_kpi3` reads
`prev_kpis[0]`, `counterfactual`/`env_i.py:20-23`). Minimal instance, timing per `base.py`:

```
k^A_t = p_{t-1}                        (KPI A mirrors the previous param)
k^B_t = p_{t-1} − 2 · k^A_{t-1}        (outcome KPI; chain term is the previous k^A)
R(k) = k^B     (maximize)
reset:  p_0 = 0,  k^A_0 = k^B_0 = 0
action candidates: p ∈ {0, 1}
```

Horizon `H = 2`, committed start `(p_0 = 0, k_0 = 0)`, action sequence `(a_1, a_2)`. Call
list per §1.1 — note the terminal advance carries **no `a_3`**; `k_3` is produced by rolling
the already-committed `p_2` forward, not by a third action (all values from `base.py:192`):

```
apply_action(a_1); advance()   → k^A_1 = p_0 = 0   k^B_1 = p_0 − 2·k^A_0 = 0      [warm-up, UNSCORED]
apply_action(a_2); advance()   → k^A_2 = p_1       k^B_2 = p_1 − 2·k^A_1 = p_1    [SCORED]
advance()          (terminal)  → k^A_3 = p_2       k^B_3 = p_2 − 2·k^A_2 = p_2 − 2·p_1  [SCORED]
```

`2` `apply_action` + `3` `advance` calls; the terminal advance applies no action and leaves
`p_2` unchanged. Scored window = `{k_2, k_3}`. `a_1` shows up as a **benefit** in `k^B_2`
(`= p_1`) and, through the chain, as a **penalty** in `k^B_3` (`= p_2 − 2·p_1`) — the second
effect surfaces only because the terminal advance rolls `p_2` (and the chain-carried `k^A_2 =
p_1`) forward. Undiscounted return:

```
G = k^B_2 + k^B_3 = p_1 + (p_2 − 2·p_1) = −p_1 + p_2
```

| arm                         | `(p_1, p_2)` | `k^B_2` | `k^B_3` | return `G`     |
|-----------------------------|--------------|---------|---------|----------------|
| finite-horizon oracle       | (0, 1)       | 0       | 1       | **1** (argmax) |
| planner (illustrative)      | (1, 0)       | 1       | −2      | −1             |
| legacy myopic per-step "oracle" | (1, 1)   | 1       | −1      | 0              |

- **Cumulative regret** (§3) of the planner = `G_oracle − G_planner = 1 − (−1) = 2`.
- **Why summed per-step `decision_regret` is wrong (B2):** the myopic per-step oracle
  maximizes each `k^B` independently — step 1 maximizes `k^B_2 = p_1` → picks `p_1 = 1`;
  step 2 maximizes `k^B_3 = p_2 − 2·p_1` (with `p_1` already fixed) → `p_2 = 1`. Each step's
  static regret is 0, so the **summed** per-step regret is **0**, hiding a true finite-horizon
  regret of `G_oracle − G_myopic = 1 − 0 = 1`. The myopic minima also describe a sequence
  `(1,1)` that the finite-horizon return-maximizing controller would not choose. Hence
  §3 uses a cloned-simulator finite-horizon oracle, and legacy per-step `decision_regret`
  (`cdd_oran/analysis/counterfactual_metrics.py:146`) is retained only as an H = 1 diagnostic.
- **Simulator/oracle identity:** both arms are rolled on clones from the same
  `(env_seed, episode)`; every `k^B` above is produced by the one `f` in `base.py:192`. The
  oracle enumerates the same action grid over the same exogenous stream the planner faced.

---

## 2. Scoring target: latent noiseless outcome (PD3, MAJOR 10)

Four distinct quantities exist today and must not be conflated: env reward = unnormalized
`sum(next_kpis)` (`base.py:181-193`); standardized xApp utility (`base.py:160-167`); observed
noisy KPI (`base.py:199-207`); and the planner/regret hinge-cost
(`counterfactual_metrics.py:118-143`). Preregistered choice:

- **Primary objective `R` is LOCKED — one function, no config switch.** `R` is the
  **negated decision cost** of `_objective_for_value`
  (`counterfactual_metrics.py:118-143`), evaluated on the **raw latent KPI vector** `k_t` (§0)
  **before** the `N(0,0.01)` observation noise of `base.py:206`. Standardization is applied
  once, *inside* each xApp utility `u_i(·)` — never to `k_t` itself, so there is no
  double-standardization. Planner, simulator rollout, and oracle all use this exact `R`; there
  is no alternate primary objective. Fully specified:
  - **xApp set** = exactly the xApps under the evaluated conflict panel, held **identical**
    across all arms (oracle / planner / decoy) — the matched decision population (§4). No arm
    scores a smaller xApp set.
  - **Per-xApp weights** `w_i = 1` for every xApp in the panel (uniform), fixed.
  - **Per-xApp standardized utility** `u_i(k_t)` = the xApp's own utility on the raw latent
    vector, which standardizes internally: `u_i(k_t) = (m_i(k_t) − xapp.mean) / xapp.std`,
    where `m_i(k_t)` is the mean of the raw KPIs the xApp aggregates
    (`XApp.compute_utility`, `base.py:160-167`). The raw `k_t` enters exactly once; the
    division by `xapp.std` is the only standardization.
  - **Directions** = each xApp's declared `direction` (`XApp.direction`): `0` ⇒ satisfy-above
    (`distance_i = max(θ_i − u_i(k_t), 0)`, `ok = u_i(k_t) ≥ θ_i`); `1` ⇒ satisfy-below
    (`distance_i = max(u_i(k_t) − θ_i, 0)`, `ok = u_i(k_t) ≤ θ_i`).
  - **Threshold treatment** = standardized hinge, threshold standardized the same way as the
    utility: `θ_i = (xapp.threshold − xapp.mean) / xapp.std`.
  - **Cost** `= Σ_i w_i · distance_i · s − (Σ_i satisfied_i)²`, scaling term `s = 10.0`
    (fixed). `R(k_t) = −cost(k_t)`, so higher `R` is better.
  - **Discount** `γ = 1` (undiscounted sum over the scored window `k_2 … k_{H+1}`), fixed.
- **Identical exogenous draws.** Oracle and planner returns are computed under the same
  exogenous stream (§4); observation noise, being outside `R`, cancels in the latent
  comparison.
- **Secondary sensitivity metrics only** (labeled, reported, **never** used to select, rank,
  or gate): observed-noisy objective `R(ŷ_t)`, plain latent standardized-utility sum, and
  env-reward-sum `Σ reward` (`base.py:182`). They are diagnostics, not comparators.

The `R(k)=k` / `R=k^B` scoring in §1.4/§1.5 is a **didactic single-KPI reduction** used only
to keep those worked examples' arithmetic legible; it is not a second objective and is not
claimed equivalent to the locked hinge cost. The locked `R` above is the one function every
arm and every gate uses on the real multi-xApp environments.

---

## 3. Regret comparator (B2)

- **Comparator = a cloned-simulator finite-horizon oracle** under the same exogenous stream
  as the planner. The oracle searches the declared action class (open-loop action sequence
  over the horizon, restricted to the same 101-point action grid the evaluator exposes;
  `counterfactual_metrics.py:171` / evaluator sweep) and maximizes the **cumulative** latent
  return `G` of §1.3 by rolling a clone forward — it does **not** minimize per-step static
  objectives.
- **Regret = `G_oracle − G_planner`** (cumulative, over the scored window `k_{t+1..t+H}`),
  both returns produced by cloned rollouts on the same `(env_seed, episode)` under identical
  exogenous draws. Non-negativity is a property to verify, not to clamp (the legacy
  `max(0, …)` clamp at `counterfactual_metrics.py:195` is a per-step diagnostic behavior).
- **Legacy `decision_regret` (`counterfactual_metrics.py:146-215`) is kept only as an H = 1
  static diagnostic.** It independently minimizes a one-step objective over one parameter and
  cannot value delayed harm through a KPI chain (demonstrated in §1.5). It is never summed
  across steps to stand in for cumulative regret, and it is not an acceptance gate.

---

## 4. Trajectory ownership, replication unit, and exogenous pairing (B3)

- **Replication unit = one independently cloned/reset simulator per
  `(env_seed, arm, planner, episode)`.** No planner's applied action ever mutates the state
  another planner or arm will score. "Arm" spans oracle / planner / decoy. This replaces the
  legacy in-place, snapshot-then-neutral-step evaluator (`evaluate.py:710-712`, which never
  applies planner actions) — that evaluator is not mutated in place; a new episodic runner is
  built later (out of scope tonight).
- **Canonical conflict schedule.** The sequence of conflict panels presented in an episode is
  fixed per `(env_seed, episode)` and identical across arms/planners. Simultaneous conflicts
  are handled by a declared **joint action**, or else each panel is declared an **independent
  counterfactual episode**; a step's meaning never changes silently with panel count.
- **Coordinate-keyed exogenous tape for ALL stochasticity (resolves MAJOR 6 + B3).** Every
  exogenous quantity — with **no exceptions** — is a deterministic, pure function of the
  coordinate **`(env_seed, episode, time, variable)`** and nothing else. This explicitly
  covers:
  - **initial parameters** (`base.py:47,173` uniform draws) → keyed at `time = 0`;
  - the **latent confounder `Z`** (E4) → its own `variable` slot;
  - **behavior-policy randomness** (the `π_b` action draws of §7) → a `variable` slot per
    behaved coordinate;
  - **transition noise** (any per-step process noise a mechanism introduces);
  - **observation noise** `ε_t` (`base.py:206`) → `ε` for KPI `v` is
    `tape(env_seed, episode, t, v)`.

  Because arms that take different actions still index the **same coordinates**, they receive
  **byte-identical** exogenous values regardless of how many draws each arm makes or in what
  order. The env-private generator of §5 may only **derive** a value *from* a coordinate
  (e.g. seed a per-coordinate sub-stream); it must **not** define any paired outcome by its
  mutable sequential draw order. **Snapshot/restore of a sequential generator is NOT a
  permitted alternative** — it depends on equal draw counts/order and reopens the
  desynchronization hole under action-dependent branches (differing episode length, behavior
  policy, conflict counts). Reseeding the global NumPy/Torch RNG (`evaluate.py:90-101`) is
  likewise insufficient.
- **Test obligation (contract, code later):** assert byte-identical exogenous values — across
  initial params, `Z`, `π_b` draws, transition noise, and observation noise — for
  oracle/planner/decoy arms whose applied actions differ.

---

## 5. RNG and benchmark version boundary (PD4, MAJOR 6/7)

- **Env-private generator.** Each env instance owns a private `np.random.Generator`
  constructed from the **`env_seed` role** (§6). It replaces all current global-NumPy draws:
  `Param.__init__` uniform (`base.py:47`), `reset` uniform (`base.py:173`), and the
  `get_state` observation noise (`base.py:206`). No env code path may call module-level
  `np.random.*` for stochastic state.
- **New benchmark version.** Because the env-private generator changes which numbers are
  drawn even at unchanged config seeds, the v2 suite is a **NEW benchmark version**. There is
  **no bit-compatible-legacy claim**. Legacy environments (`EnvironmentI..IV`) are left intact
  tonight (PD4); legacy reports reproduce **only** from their recorded git SHA. The single
  atomic cutover happens after E1–E5 exist.
- **CRN pairing** is provided entirely by the §4 coordinate-keyed tape: the env-private
  generator is instantiated/seeded **per coordinate** and consumed only to derive that
  coordinate's value, never as one long sequential stream whose order pairs arms.

---

## 6. Four-way seed roles (PD4)

Today only `seed` (discovery+training, conflated) and `mitigation_seed` (evaluation) exist,
with `training_seed` captured but discovery reusing `seed`
(`evaluate.py:357-362`, `utils/seeding.py:7-14`). The contract fixes **four independent
roles**, each drawn from its own namespace so streams never alias:

| role             | governs                                                            |
|------------------|--------------------------------------------------------------------|
| `discovery_seed` | causal-structure discovery / data collection for discovery         |
| `train_seed`     | model/planner training                                             |
| `env_seed`       | the env-private generator: init, transitions' exogenous draws, obs noise (§5) |
| `mitigation_seed`| evaluation-time planner draws / action sampling                    |

CRN coordinates (§4) are keyed by `env_seed` (+ episode/time/variable). Discovery no longer
reuses the evaluation/training seed.

---

## 7. ID/OOD and observational-vs-interventional split

- **ID vs OOD = ONE common, normalized support-shift construction (resolves MAJOR 10).**
  The legacy per-env `ood` ranges (`env_i.py:32-35` and the sibling `_param_ranges` in
  `env_ii/iii/iv.py`) are **not reused**: each was chosen independently, so their shift
  direction and magnitude differ and cross-env "OOD degradation" would confound the shift
  with the env — it is not a single axis. Instead, OOD is **derived deterministically from
  each env's ID support by one rule applied identically to every env**, mechanisms unchanged
  (covariate/support shift only), evaluated as an evaluation mode via
  `evaluation_param_ranges` (`config.py:108-109,166`):
  - For each parameter `j` with ID range `[lo_j, hi_j]` and width `w_j = hi_j − lo_j`, the OOD
    range is the ID range **shifted by a fixed normalized amount and widened symmetrically**:
    `ood_j = [lo_j + (δ − ρ)·w_j,  hi_j + (δ + ρ)·w_j]`, where `δ` is the normalized shift
    (in ID-width units) and `ρ` the normalized widening. Degenerate params (`lo_j = hi_j`)
    are left unshifted. Because `δ, ρ` are expressed in each param's own ID-width units, the
    **shift severity is identical across envs** and OOD becomes one comparable axis.
  - **Preregistered severity (single value, fixed before any run): `δ = 0.5`, `ρ = 0.25`.**
    (I.e. shift the support by half its ID width and widen each side by a quarter width; this
    yields a partially-overlapping OOD support with a genuine extrapolation region.) Any other
    `(δ, ρ)` is a labeled secondary sensitivity sweep, never the primary OOD axis.
  - **Required reporting, per env:** support-overlap and extrapolation statistics — for each
    param the fraction of the OOD range lying outside `[lo_j, hi_j]`, and the aggregate
    **fraction of sampled OOD support outside the ID convex hull** — logged in the artifact
    (§8) so that measured OOD degradation is interpretable against how far OOD actually lies
    outside ID. No per-env range is hand-tuned.
- **Observational vs interventional data.** The suite must expose a clean split, which does
  **not** exist today (all transitions come from a `RandomPolicy` applying actions;
  `edge_stability.py:36-49`, `train.py:155-157`; no `do()` API). Definition:
  - *Observational* = transitions generated under a declared **behavior policy** `π_b`
    (a fixed stochastic policy over the action space), with **no** interventional override.
  - *Interventional* = transitions where a chosen coordinate is set by an explicit `do(·)`
    (the applied-action path), independent of `π_b`.
  - **E4 requirement:** E4 needs a *real* behavior policy `π_b` so that an observational
    dataset can be confounded. Its spec must exhibit `Z → behavior-action`, `Z → outcome`,
    and `action → outcome`, name the interventional evaluation distribution, and require a
    sign-reversal / action-regret contrast at its gate (MINOR 14). A latent that touches only
    the outcome (the old Env IV shape, `env_iv.py:203-209`) is insufficient.

---

## 8. Version stamp (PD4, MINOR 13)

Every produced artifact (posterior, enumeration graph, evaluation record) is stamped with:

- `benchmark_version` — an unambiguous id for the v2 suite (new env identifiers, distinct
  from legacy `EnvironmentI..IV`), and
- `scm_hash` — a hash of the SCM definition (adjacency edges + update-fn identities +
  param ranges) for the env instance.

New IDs **reject** old posterior/enumeration/checkpoint artifacts (no accidental reuse across
the version boundary). The git SHA is recorded alongside for legacy reproduction.

---

## 9. What is retired vs kept

- **Retired for scoring:** the instantaneous static utility map (`base.py:222-233`) and the
  neutral-action, no-apply evaluation loop (`evaluate.py:710-712`) — replaced by the episodic
  latent-scored runner (built later).
- **Kept as legacy diagnostic:** per-step `decision_regret`
  (`counterfactual_metrics.py:146`), reported at H = 1 only, never summed as cumulative
  regret, never a gate.
- **Kept intact tonight (no deletion, PD4/MINOR 12):** legacy envs, configs, reports, and
  `reports.zip`. One atomic cutover only after E1–E5 exist.

---

## 10. Open question

The action class for the finite-horizon oracle (§3) is fixed as **open-loop over the
101-point grid** to keep the oracle enumerable and the comparator realizable. Whether E3 also
needs a **closed-loop** (state-feedback) oracle — which would be a strictly stronger, harder-
to-enumerate comparator and could enlarge measured regret — is deferred to the E3 spec (WP2);
this contract commits only to the open-loop comparator and flags the closed-loop question as
unresolved.

---

## Implementation pointer (NON-NORMATIVE)

The shared open-loop executor of the §1.1 call list, the §3 exhaustive open-loop oracle, and
§3 unclamped paired regret live in `cdd_oran/benchmark/rollout.py`
(`rollout_open_loop` / `enumerate_open_loop` / `paired_regret`), characterized by
`tests/test_v2_rollout.py`. This paragraph is a convenience pointer only: the definitions above
remain the sole authority, and nothing in that module or its tests amends this contract.
