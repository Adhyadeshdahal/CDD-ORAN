# Benchmark Specification — Environments E1–E5 (Phase 1)

Status: Preregistered (Phase 1 of `.herdr/plans/environment-benchmark-redesign.md`).
Date: 2026-09-01.

## Preamble — scope and framing

This is a **preregistered boundary-condition study**, not a demonstration that the
proposed model wins. Environments are specified so the gates decide, and no env is tuned
to advantage discovered structure. Only E2 is anchored in an already-validated result
(the narrow CFCP shared-control coordination win, `memory/causal-planner-direction-explored.md`);
E3 and E4 are open tests of whether causal decision value appears once decisions are made
genuinely load-bearing along a temporal chain (E3) and under action-relevant confounding (E4).

**Notation.** `Pi` = NCP / control parameter index `i`; `Kj` = KPI index `j`. An SCM edge
`Kj <- Pi` means param `Pi` is a direct cause of `Kj`; `Kj <- Kk` is a KPI->KPI edge. In
code these are declared as `adjacency_edges = [(kpi_index, source_index)]` with
`source_index >= num_params` encoding a KPI source (`cdd_oran/envs/base.py:134-138`).

**Reconciliation stamp (COMPLETE).** All timing, regret-comparator, and latent-outcome
conventions below are owned by `docs/benchmark/SEMANTICS.md` (WP1). Reconciliation against
that contract is **COMPLETE**: this spec is locked against SEMANTICS.md content hash
`sha256:d17871c38567042d2ada5295089c9218084c56978e60daac4ace3136d2e4e406` (the LOCKED v2
contract). References marked **[SEM]** cite that
version; if a later SEMANTICS.md edit changes a [SEM] rule, the hash above must be bumped and
this spec re-reconciled in the same change — so the preregistration line cannot move
silently. The load-bearing shared assumptions (each verified present in the locked
SEMANTICS.md):

- **[SEM] PD2 — one-step actuation latency.** Action `a_t` sets params at `t`; KPIs respond
  at `t+1` (env already does this via `prev_params`, `base.py:192`). The horizon and the
  oracle are defined over this same delayed trajectory.
- **[SEM] PD3 — latent noiseless scoring.** Oracle and regret are computed on the latent
  (pre-observation-noise) KPI under identical exogenous draws. Observed-noisy outcome and
  env-reward-sum are secondary metrics only.
- **[SEM] Regret comparator.** The comparator is a cloned-simulator **finite-horizon
  oracle** searching an **open-loop** action sequence over the 101-point action grid and
  maximizing cumulative latent return; regret = `G_oracle − G_planner`, not clamped
  (SEMANTICS §3). Cumulative regret (E3/E5) uses `H > 1`; single-decision regret (E2, E4)
  is the **same oracle at `H = 1`** — a single delayed transition on the latent state
  (SEMANTICS §1.3/§3), not the legacy per-step `decision_regret`
  (`counterfactual_metrics.py:146`), which is retained only as an H=1 diagnostic and is
  never the comparator or a gate (SEMANTICS §3/§9).

Each E1–E4 varies **one** stress axis; E5 composes already-validated axes. Every decision
env (E2–E5) must carry a **verified simulator-level oracle-vs-decoy regret gap** before any
training (Phase 4 decision-gap gate); no current env has one, so each gap below is a
**requirement to be verified**, not an assumed property.

---

## E1 — Clean sanity (recovery control)

**Scientific hypothesis.** On a small, sparse, identifiable SCM with measurement noise
disabled, the pipeline recovers the true causal graph (skeleton + orientation) and produces
correct one-step and short-horizon KPI predictions. This establishes that recovery and
prediction machinery is sound before any decision claim is tested.

**Causal graph / SCM sketch.** Small sparse DAG, ~4 params / ~4 KPIs, reusing the Env I
shape (`env_i.py:51-62`) as the structural template:

```
P0 ──► K0        P1 ──► K1
P2 ──► K2        P3 ──► K3
K0 ──► K2   (KPI->KPI, direct-child at lag 1)
K1 ──► K3   (KPI->KPI, direct-child at lag 1)
```

- Nodes: params {P0..P3}, KPIs {K0..K3}. Every KPI has exactly one param parent; K2 and K3
  additionally have one KPI parent (K0, K1 respectively), so the KPI->KPI edge is **present
  and load-bearing** for the SCM gate (per Phase-1 depth requirement).
- Edges are sparse and orientable (no cycles, no shared latent).

**Locked SCM (exact — nothing left for WP3 to invent, BLOCKER a).** E1 reuses the Env I
*plumbing* (`env_i.py`) but replaces its non-monotone Gaussian-bump `update_kpi*` with
**monotone linear** mechanisms, so every edge effect is analytic and state-independent. All
values LOCKED:

- **ID parameter ranges:** `P0, P1, P2, P3 ∈ [0, 1]` (identical ranges; the OOD support-shift
  is the common SEMANTICS §7 rule, not defined here).
- **Raw latent update equations** (timing per `base.py:192`, one-step lag; `K*_prev` is the
  previous-step raw latent KPI):

  ```
  K0(t) = P0(t-1)                       # f0: edge P0->K0,   slope a0 = 1
  K1(t) = P1(t-1)                       # f1: edge P1->K1,   slope a1 = 1
  K2(t) = P2(t-1) + 0.5 · K0(t-1)       # g2: edges P2->K2 (a2=1), K0->K2 (b2=0.5)
  K3(t) = P3(t-1) + 0.5 · K1(t-1)       # g3: edges P3->K3 (a3=1), K1->K3 (b3=0.5)
  ```

  All four are strictly increasing in each argument (monotone), acyclic (K2/K3 read only
  prior-step K0/K1). Raw ranges: `K0,K1 ∈ [0,1]`; `K2,K3 ∈ [0, 1.5]`.
- **Standardization constants** (fixed, derived analytically under the ID uniform prior
  `P·~U[0,1]`, used by `get_state`/utility standardization): `μ_K0 = μ_K1 = 0.5`,
  `σ_K0 = σ_K1 = √(1/12) = 0.288675`; `μ_K2 = μ_K3 = 0.75`,
  `σ_K2 = σ_K3 = √(1.25/12) = 0.322749`. Standardized KPI `k̃_j = (K_j − μ_j)/σ_j`.
- **xApp definitions (constructor constants only — E1 has NO shared-control conflict).**
  `kpi_to_xapp = {0:0, 1:1, 2:2, 3:3}`, `xapp_kpi_indices = [(0,),(1,),(2,),(3,)]`,
  `xapp_param_indices = [(0,),(1,),(2,),(3,)]` — a clean one-param/one-KPI-per-xApp map, so
  no two xApps share a control knob. `kpi_thresholds`/`directions` are pinned for
  construction but are **not read by any E1 gate** (E1 has no decision panel).
- **Analytic `min_eff` confirmation (not tuned).** Because the mechanisms are linear, each
  edge's effect on its standardized child is state-independent. For the gate's
  `do(P_i = 0.75)` vs `do(P_i = 0.25)` contrast (Δ_raw_param = 0.5):
  - direct param edges `P0->K0`, `P1->K1`: `0.5/0.288675 = 1.732` standardized units;
  - direct param edges `P2->K2`, `P3->K3`: `0.5/0.322749 = 1.549`;
  - KPI->KPI edges `K0->K2`, `K1->K3` (via `do(P0)`/`do(P1)`, descendant lag `t+2`):
    `0.5·b2/σ_K2 = 0.25/0.322749 = 0.775`.
  The **minimum** edge effect is `0.775`, which is `≥ min_eff = 0.05` at **every** bank state
  by construction (linear ⇒ no zero-derivative states). `min_eff = 0.05` is therefore valid
  analytically, with a ~15× margin, before any code exists.
- **Measurement noise OFF:** requires a real `noise_scale = 0` knob; the base class currently
  adds `N(0, 0.01)` to every KPI unconditionally (`base.py:206`) and must gain a noise-off
  path (built in WP3). Latent outcome == observed outcome when noise is off.

**Controlled characteristic.** Structural cleanliness/identifiability — smallest sparse SCM,
noise disabled. This is the ONE varied factor; nothing else stresses the pipeline.

**Deliberate decoy / trap.** **None.** E1 is the recovery control and is **exempt from the
decision-gap and control gates** (MAJOR 9 / plan Phase 4). **E1 has NO decision/conflict
panel and NO decision-oracle check** — its one-param-per-xApp map shares no control knob, so
there is no conflict to score and no H=1 finite-horizon decision oracle is defined or
required for E1 (MAJOR, r2 finding 2). Its only gate is SCM correctness + analytical
transition/prediction sanity.

**Designated baseline / comparison arm.** Discovered structure vs oracle structure, plus the
capacity-matched dense (MLP) predictor for the prediction sub-claim. No decoy arm.

**Primary metric.** (1) **Graph recovery** — directed cell-wise PRF over `true_adj_matrix`
(`recovery_metrics.py` / `threshold_sweep.py`); (2) **predictive error** — ensemble one-step
MSE + short-horizon rollout RMSE (`attribution.py:72`, `counterfactual_metrics.py:89`);
(3) **analytical transition/prediction sanity** — because the locked mechanisms are known and
noiseless, the closed-form next latent KPI (from the equations above) must match the simulator
rollout to `tol_zero`. No decision-oracle / regret metric is used for E1.

**Expected result.** Near-perfect recovery (PRF ≈ 1 on the identifiable skeleton, correct
orientation on the two KPI->KPI edges) and low predictive error, discovered ≈ oracle. Dense
predictor may match on prediction (no structure needed for a clean identifiable map).

**Failure interpretation.** If recovery or prediction fails **here**, the fault is in the
discovery/prediction pipeline itself (not in any downstream decision claim), and no E2–E5
decision result is interpretable until E1 is green. A discovered ≪ oracle recovery gap here
flags the posterior/bootstrap validity risk (Phase 5) rather than an environment property.

### E1 SCM-gate protocol (preregistered, BLOCKER 3)

E1 stays **exempt from the decision-gap and factor-removal gates**. Its only gate is SCM
correctness: the generated latent transitions match the declared graph. The protocol is a
single fixed rule with no post-hoc selection. All quantities are on the **latent noiseless
state** (`noise_scale = 0`, `base.py:206` path disabled); observation noise is excluded
entirely.

- **State bank (interior intervention points).** `N_states = 8` latent states
  `{s_1..s_8}`, generated deterministically: reset the env with `env_seed ∈ {0..7}` and roll
  forward **3 neutral (no-op) steps** with noise OFF; the resulting `(p, k)` latent state is
  `s_i`. These are interior (not the reset state) and fixed before any run.
- **Intervention values per param.** For each param `P_i`, two preregistered `do(·)` values:
  `low_i` = 25th percentile and `high_i` = 75th percentile of `P_i`'s **ID** range
  (`_param_ranges`). At each state `s` we apply `do(P_i = low_i)` and `do(P_i = high_i)` and
  read the resulting latent trajectory forward.
- **Direct-effect lag.** A direct child `K_c` of `P_i` (edge `K_c <- P_i`) responds at
  **`t+1`**: `k_c(t+1) = f(p_t, k_t)` (`base.py:192`). The direct-effect statistic is
  `Δ_c = |k_c(t+1; do high_i) − k_c(t+1; do low_i)|`.
- **Descendant horizon.** A KPI->KPI descendant `K_d` reached through `h` KPI edges responds
  at **`t+1+h`** (here `K2 <- K0 <- P0` ⇒ `K2` at `t+2`; likewise `K3` via `K1`). The
  descendant horizon is `H_desc = 2`. Descendants that move only at their later lag are
  **correct and MUST NOT be rejected**; they are checked at their own lag, not at `t+1`.
- **Edge-ablation check — DIRECT off-manifold source intervention verifies DIRECT KPI->KPI
  parentage, not mere ancestry (BLOCKER b; repaired).** A param-ancestor drive alone **cannot**
  tell declared `P0->K0->K2` from an undeclared direct `P0->K2` (both move `K2` when `P0`
  moves), because after the bank equilibrates on the manifold `K0 = P0` the two mechanisms
  agree and a shadow that zeroes the KPI coefficient removes **both** paths. The param-ancestor
  drive (`do(P0)` moves `K0`; `do(P1)` moves `K1`; child read at lag `t+2`) is therefore kept
  **only as the clause (a) ancestry** check (necessary — the child must respond to its source's
  ancestor — but insufficient for direct parentage). The **discriminating** clause (c)
  direct-parentage check instead intervenes **directly on the latent source**. For **each**
  KPI->KPI edge — `K0->K2` and `K1->K3` — on the latent noiseless state:
  1. **Directly set the source latent KPI off the manifold.** Set `prev_kpis[src] = P_src ± 0.3`
     (two preregistered offsets, so `K_src ≠ P_src` and the low/high values differ), **holding
     ALL params and every other latent KPI fixed**, then advance **one** step (no warm-up — a
     KPI child reads `prev_kpis` directly) and read the child. A `P_src`-reading impostor is
     now exposed: its child does not move, because no param changed.
  2. Build a **shadow SCM** with that one KPI-parent coefficient zeroed (for `K0->K2`, set
     `b2 = 0` ⇒ `K2 = P2 + 0·K0_prev`; for `K1->K3`, set `b3 = 0`), **holding the child's own
     param-parent fixed** (`P2`, resp. `P3`).
  3. The declared edge is **confirmed** iff: (i) in the **true** SCM the direct off-manifold
     source move moves the child by `≥ min_eff` (the KPI-parent contributes), **and** (ii) in
     the **shadow** SCM that same direct source move moves the child by `≤ tol_zero` (the
     KPI-parent contribution is gone), **while** (iii) the direct param->child effect (`do(P2)`,
     resp. `do(P3)`, at `t+1`) is **unchanged** between true and shadow
     (`|Δ_shadow − Δ_true| ≤ tol_zero`). This isolates the KPI-parent's contribution from any
     direct param->child path.
  4. **Edge-ablation negative control:** a **non-parent** source KPI of the child (e.g. `K1`
     for `K2`), moved by the **same direct off-manifold intervention** (`prev_kpis[K1] = P1 ± 0.3`,
     all else fixed), must produce **no** effect on the child (`|Δ| ≤ tol_zero`) in the true
     SCM — no undeclared KPI->KPI edge exists.
- **Numerical tolerance.** `tol_zero = 1e-6` (deterministic equality tolerance under noise
  OFF).
- **Minimum-effect rule.** A declared edge PASSES only if it moves its child by
  `Δ ≥ min_eff = 0.05` (standardized latent units) at **every** state in the bank, for at
  least the `high_i` intervention (direct edges at `t+1`; KPI->KPI edges at the descendant's
  lag, driven by intervening on the param-ancestor of the source KPI).
- **Non-descendant negative controls.** For each `do(P_i)`, every KPI that is **not** a
  descendant of `P_i` MUST satisfy `|Δ| ≤ tol_zero` at all lags up to `H_desc`. Any movement
  above `tol_zero` at a non-descendant is a gate FAILURE (spurious edge in the generator).
- **Acceptance.** The SCM gate is green iff: (a) every declared edge passes the
  minimum-effect rule at the correct lag, at all 8 states; (b) every non-descendant passes
  the negative control at all 8 states; (c) **every KPI->KPI edge passes the edge-ablation
  check (direct parentage confirmed) and its edge-ablation negative control, at all 8
  states**; (d) the analytical closed-form next-latent-KPI matches the rolled-out latent value
  to `tol_zero`. Failure of any clause blocks WP3.

---

## E2 — Shared-control conflict (anchored in validated CFCP fan-out)

**Scientific hypothesis.** When one shared control knob fans out to multiple xApps, a model
that reasons over the **complete** causal fan-out avoids a locally attractive single-xApp
action that is globally harmful, whereas a model that sees only an **incomplete** fan-out
takes it. This is the already-validated narrow CFCP shared-control coordination win
(+2.22 on TRUE shared-control blocks; `memory/causal-planner-direction-explored.md`),
re-expressed as a preregistered decoy contrast.

**Causal graph / SCM sketch.** Reuse Env II fan-out (`env_ii.py:85-105`). Key structure:

```
P0 ──► K0, K1, K2, K5      (shared NCP: broad fan-out)  # (schematic error corrected to match env_ii.py:88-105 source graph; not an outcome-dependent change)
P1 ──► K0, K1, K3, K4
xApp3 ◄── {K3(=kpi41), K4(=kpi42)}   (two KPIs fan into one xApp)
kpi_to_xapp = {0:0, 1:1, 2:2, 3:3, 4:3, 5:4}   (env_ii.py:87)
```

- No KPI->KPI edge required (this env isolates the **shared-NCP fan-out** axis, not a chain).
- The conflict panel for a decision on `P0` includes every xApp sharing `P0`
  (`detect_conflict_edges`, `conflicts.py:23-59`).

**Controlled characteristic.** Shared-control fan-out breadth — more xApps per shared knob,
graph size and depth held approximately fixed. The one varied axis.

**Deliberate decoy / trap (single preregistered rule, BLOCKER 4).** The decoy is an
**incomplete fan-out world model that omits EXACTLY ONE edge: `P0 -> K5`** (the fan-out edge
into xApp4). This one edge is fixed now, before any simulator search — no "one or more"
choice, no post-hoc edge selection. The decoy therefore **mispredicts** the effect of moving
`P0` on `K5` (it predicts no effect) and picks the locally attractive `P0` move; the
complete-fan-out oracle predicts the `K5` harm and avoids it.

- **Matched decision population (identical across arms).** The full xApp objective and the
  full conflict panel — every xApp sharing `P0`, **including xApp4/K5** — are IDENTICAL for
  the oracle and decoy arms. The decoy only *mispredicts* `K5`; it does **NOT** remove `K5`
  (or xApp4) from **realized** scoring. Regret is computed on the full objective under the
  true SCM for both arms (a smaller objective would be a different decision, not a different
  structure). Enforced via the canonical enumeration (`enum_source`, `evaluate.py:445-467`).

**Required verified gap (Phase 4).** One fixed **state/conflict bank** of `P0`-shared
decisions, containing **positive controls** (states where the true `do(P0)` realizes harm to
`K5`) AND **negative controls** (states where `do(P0)` is inert on `K5`), all fixed before
search. One **regret threshold** `τ_E2` (min normalized realized regret) is set in the WP4
gate contract BEFORE any simulator inspection. Acceptance: mean realized regret gap
`(decoy − oracle) ≥ τ_E2` on the positive controls **and** `≤ tol` (no gap) on the negative
controls. Anchor target for `τ_E2`: at least the validated CFCP shared-control effect on TRUE
blocks. No conflict subset is chosen after inspecting the simulator.

**Designated baseline / comparison arm.** Complete-fan-out oracle structure (reference)
vs incomplete-fan-out (`P0 -> K5` omitted) decoy structure vs discovered structure;
correlational and dense arms reported for context. **Planner: the SAME single-control
planner for BOTH arms** — a one-control `QACM`/`CEM` acting on the single shared param `P0`
(MAJOR 8). `JointMultiNCPPlanner` is **not** used: E2's hypothesis has one shared control, and
joint multi-NCP optimization would add a second stress axis.

**Primary metric.** Regret from incomplete-vs-complete fan-out — single-decision latent
regret difference (decoy regret − oracle regret) over the fixed conflict bank, each regret
computed by the finite-horizon oracle at `H = 1` (SEMANTICS §3, not the legacy per-step
`decision_regret`), on the identical full-objective decision population (`enum_source`,
`evaluate.py:445-467`).

**Expected result.** Complete-fan-out oracle ≈ discovered (if discovery finds the fan-out)
≪ incomplete-fan-out decoy regret on the **positive controls**, and no gap on the **negative
controls**; the win is expected to be narrow, not global (consistent with the validated
finding).

**Failure interpretation.** If the decoy incurs no more regret than the oracle on the
preregistered positive controls, the fan-out is not decision-relevant under current semantics
→ Phase-4 decision-gap gate fails → stop/redesign per Decision Gates (the fixed `P0 -> K5`
decoy and controls are NOT re-tuned toward a positive result; a null is a valid boundary
result). If discovered ≪ oracle, the fault is discovery, not the benchmark.

---

## E3 — Temporal decision (genuinely new)

**Scientific hypothesis.** When an action delivers an **immediate** KPI benefit that causes a
**larger delayed** KPI harm downstream, a horizon-aware causal planner scoring the full
delayed trajectory avoids it, whereas the same planner at H=1 (myopic) takes it.
This tests whether causal decision value appears once the decision is genuinely load-bearing
over time — a case the current benchmark **cannot** express (`evaluate.py` never executes
planner actions and reports a mean of static per-step utilities;
`research-envs-semantics.md` headline).

**Causal graph / SCM sketch (RATIFIED fan-out redesign, `review-e3-redesign.md`;
`GATE_CONTRACT_E3.md`).** A depth-three KPI cascade whose internal conduit **fans out** into two
delayed monitored services, so an early beneficial move propagates to a larger later loss on more
than one SLA. This is the ratified redesign that supersedes the earlier single-harm
`P0->K0->K1->K2` chain (the old single-edge `K1->K2` decoy is obsolete). Load-bearing structure
(`num_params=4`, `num_kpis=5`, every coefficient exactly 1):

```
P0 ──► K0        (immediate: action on P0 improves K0 at t+1)
K0 ──► K1        (internal conduit, EXCLUDED from the panel)
K1 ──► K2        (K1 fans out to a delayed monitored service)
K1 ──► K3        (K1 fans out to a second delayed monitored service)
P1 ──► K2 ,  P2 ──► K3 ,  P3 ──► K4   (K4 is P0's non-descendant structural control)
```

- Each KPI->KPI edge advances one step per **[SEM] PD2** (K0_t -> K1_{t+1} -> {K2,K3}_{t+2}),
  so the immediate gain at K0 and the delayed loss at the fan-out pair {K2,K3} are separated by
  the chain depth in time. The sign structure is set so that maximizing the immediate K0
  objective drives K2 and K3 below target by a larger combined amount later — an immediate
  benefit that creates greater delayed harm across two services.
- Scored panel IDs are `(0,2,3)`; K1 (internal conduit) and K4 (non-descendant control) are
  excluded IDs `(1,4)`, identically for all four cells, the TRUE oracle, realized return, and D3.
- Depth and lag of the chain are fixed to **match the planning horizon** given to the
  horizon-aware arm, so the oracle, simulator, and planner all score the same delayed
  trajectory (no timing mismatch).
- Requires the new common temporal semantics (executed actions + cumulative return), not
  merely a new env file. Uses `RecedingHorizonCEM` (`horizon.py`) as the horizon-aware arm.

**Controlled characteristic.** Temporal depth of the decision — the number of steps between
an action's immediate benefit and its larger delayed harm. The one varied axis (breadth,
confounding, and noise held fixed).

**Deliberate decoy / trap (preregistered 2×2, MAJOR 7).** The trap — immediate benefit
causing larger downstream loss — is probed by a **2×2 design** crossing two binary factors,
**all four cells under the SAME model capacity and objective**:

|                          | **H = 1 (myopic)** | **H = causal-depth horizon** |
|--------------------------|--------------------|------------------------------|
| **Full structure**       | cell FM            | **cell FH (proposed)**       |
| **Truncated structure** (drop the fan-out pair `{K1->K2, K1->K3}`) | cell TM | cell TH |

- **PRIMARY contrast (headline temporal claim): FH vs FM** — full structure held fixed
  (correct), horizon varied H=depth vs H=1. This isolates **temporal decision value**: given
  the correct chain, does planning over the horizon avoid the delayed harm that the myopic
  planner takes? This is the ONE named primary contrast.
- **Mechanism checks (secondary):** FH vs TH isolates **causal structure** (same horizon,
  fan-out pair `{K1->K2, K1->K3}` present vs dropped); TM is the neither-cell floor. Together they separate
  structure from horizon so a positive primary contrast cannot be attributed to horizon
  alone. No "H=1 and/or drop the edge" ambiguity: each factor is its own axis.

**Required verified gap (Phase 4).** The PRIMARY contrast FH vs FM must show meaningful
**cumulative** realized regret over the causal horizon on the latent trajectory (FM incurs
positive cumulative regret vs the FH oracle). One threshold and the causal-depth horizon are
preregistered in the WP4 gate contract before any simulator search.

**Designated baseline / comparison arm.** The four 2×2 cells (FH proposed, FM myopic decoy,
TH structure-ablation, TM floor), plus the discovered-structure planner at H=depth and a
capacity-matched dense arm for context. Same capacity and objective across all cells.

**Primary metric.** **Cumulative regret over the causal horizon** — oracle cumulative
latent return − planner cumulative latent return under the same exogenous stream (B2),
scored on the delayed window `k_{t+1..t+H}` (SEMANTICS §1.2). The oracle comparator searches
the **open-loop** action sequence over the 101-point grid (SEMANTICS §3); whether E3 also
warrants a closed-loop (state-feedback) oracle is the open question SEMANTICS §10 defers to
this spec — E3 commits to the open-loop comparator (a closed-loop oracle would only enlarge
measured regret, so open-loop is the conservative choice). This metric does not yet exist
(`evaluate.py` reports only the per-horizon mean; `research-planner-metrics-validity.md` §3)
and must be built before E3 runs.

**Future-prerequisite note (r2 finding 3 — NOT blocking tonight; E3 is not built tonight).**
The locked open-loop oracle action class emits a full sequence `a1..aH`, but the named
`RecedingHorizonCEM.act` returns only the **first** action and its rollout repeats one held
action across the horizon (`cdd_oran/planners/horizon.py:49-73,115-164`) — so E3 cannot later
claim the locked open-loop comparison by reusing that planner unchanged. **Before any E3
CODE**, either (a) add a sequence-emitting open-loop planner whose action class matches the
oracle's `a1..aH`, or (b) switch E3 to a receding-horizon feedback protocol with a
correspondingly matched (closed-loop) oracle. This is recorded as an E3 implementation
prerequisite, not a tonight blocker.

**Expected result (open).** If temporal causal value is real, horizon-aware oracle ≈
discovered ≪ myopic decoy in cumulative regret. A null (no cumulative gap) is a legitimate,
publishable boundary result — it says temporal structure does not add decision value under
these dynamics.

**Failure interpretation.** No cumulative gap ⇒ the temporal chain is not decision-relevant
(stop/redesign, or report as a negative boundary result); do not deepen the chain merely to
manufacture a gap (that adds a second stress axis, forbidden by Decision Gates). A gap that
vanishes under capacity matching means the effect was parameters, not structure.

---

## E4 — Confounded decision (action-relevant confounding — highest spec risk)

**Scientific hypothesis (narrow claim, BLOCKER 5).** When a latent state `Z` drives **both**
the behavior-policy's actions **and** the outcome, the observational association between
action and outcome is misleading. E4 tests whether a **structural causal inductive bias
improves estimation and decision quality FROM THE SAME PARTIALLY-INTERVENTIONAL DATASET** —
i.e. given a mixture of observational (Z-confounded) and randomized-interventional
transitions, does the structure-aware model recover the true `A -> K_out` effect and choose
the correct intervention better than a correlational model trained on the identical mixture?
Identification comes from the **randomized-interventional subset** (where action assignment
is independent of `Z`); E4 does **NOT** claim the latent-confounded effect is identifiable
from observational data alone (in the general SCM below, `P(K_out | do(A))` is not identified
from `P(A, K_out)`). The genuine observational behavior-policy stream — absent today, all
transitions come from a RandomPolicy (`research-envs-semantics.md` §2) — must be built.

**Causal graph / SCM sketch (must be a real behavior policy — MINOR 14).** The confounder
must sit on all three required edges, not on the outcome alone:

```
        Z (latent)
       ╱   │   ╲
      ▼    │    ▼
 behavior  │  outcome          Z ──► A_behavior   (Z biases the data-collection action)
 action A  │   K_out           Z ──► K_out        (Z directly affects the outcome)
      ╲    │   ╱                A ──► K_out        (the true causal effect to be recovered)
       ▼   ▼  ▼
        A ──► K_out
```

- **Z -> A_behavior:** the latent biases which action the behavior policy tends to take
  during observational data collection (a real policy, e.g. `policies/random_policy.py`
  reweighted by `Z`), inducing spurious action–outcome association.
- **Z -> K_out:** the latent directly moves the outcome KPI.
- **A -> K_out:** the genuine causal effect the model must recover.
- Structure otherwise reuses Env IV (`env_iv.py`), including its `K0->K4->K5` backbone, but
  Env IV's `Z` currently affects **outcomes only** (`env_iv.py:203-209`) — that is the
  **old Env IV null** and must NOT be repeated. The distinguishing new element is the
  `Z -> A_behavior` edge and a behavior-policy observational dataset.

**Controlled characteristic.** Action-relevant confounding strength — the coupling of `Z`
into the behavior policy (and thus the observational association). The one varied axis.

**Named interventional evaluation distribution.** Decisions are scored under the
**interventional** distribution `P(K_out | do(A = a))` — i.e. actions set by the planner
(not by the `Z`-biased behavior policy), latent `Z` drawn from its marginal, outcome on the
latent noiseless KPI (**[SEM] PD3**). The **observational** distribution `P(K_out | A = a)`
(actions from the `Z`-biased behavior policy) is what the correlational arm is trained on and
is deliberately misleading. This obs-vs-interventional split does not exist today and must be
built.

**Matched-data contract (preregistered, BLOCKER 5).** Every **learned** arm — proposed
(structure-aware), correlational, and capacity-matched dense — receives the **IDENTICAL**
training corpus and protocol:

- the **same observational transitions** (generated under the `Z`-confounded behavior policy
  `π_b`) **and the same randomized-interventional transitions** (where the acted coordinate
  is set by `do(·)` with assignment **independent of `Z`**);
- the **same train/validation splits**, the **same intervention indicators** (per-transition
  obs-vs-interventional flag), and the **same action coverage** over the grid.

No arm sees interventions the others do not. The only thing that varies across arms is the
**structural assumption / inductive bias**, never the data. Identification of `A -> K_out`
rests on the randomized-interventional subset shared by all arms.

**Locked confounder parameters (fixed BEFORE any generation, MAJOR 9).** All of the
following are preregistered and are **not** tuned toward a positive result:

- **Behavior-policy equation `π_b(a | Z)`** — the exact functional form and coefficients by
  which `Z` biases the collected action (fixed equation, not "an example policy").
- **Latent `Z` distribution** — the exact marginal `P(Z)`.
- **Confounding coupling level(s)** — the fixed `Z -> A_behavior` / `Z -> K_out` coupling
  strength(s); if multiple levels are declared, the full set is fixed now (a preregistered
  sweep), with the primary level named.
- **Fixed state bank** — the states/decisions at which the interventional contrast is scored.
- **PRIMARY acceptance criterion** — sign-reversal OR positive interventional action regret
  (below), with a fixed magnitude.
- **Minimum action-disagreement prevalence** — the fixed minimum fraction of bank decisions
  on which the correlational arm and the interventional oracle must choose different actions.

These values live in the WP4 gate contract and are frozen before simulator generation.

**Deliberate decoy / trap.** **Correlational structure** trained on the identical mixture
above: it absorbs the `Z`-induced spurious association and recommends the intervention that
looks best observationally but is worse (or sign-reversed) interventionally.

**Required verified gap (Phase 4).** At the simulator gate, on the fixed state bank, the
correlational/decoy action and the interventional oracle action must differ with a **sign
reversal or positive interventional action regret** of at least the locked magnitude, on at
least the locked **minimum action-disagreement prevalence** of decisions (MINOR 14 —
mandatory, not optional). A latent affecting outcomes alone (no `Z -> A_behavior`) reproduces
the old null and **fails** this gate by construction.

**Designated baseline / comparison arm.** Interventional oracle structure (reference) vs
correlational structure (decoy, trained on the shared mixture) vs discovered structure; dense
arm for context. All learned arms share the matched-data contract above. The correlational
arm is **required** here (Phase 3 arm table).

**Primary metric.** **Interventional regret under confounding** — oracle interventional
latent return − planner interventional latent return, evaluated under `P(· | do(A))`. Does
not exist today (`research-planner-metrics-validity.md` §3) and must be built.

**Expected result (open).** If deconfounding has decision value, interventional oracle ≈
discovered ≪ correlational decoy (which suffers positive interventional regret / picks the
sign-reversed action). Memory (`memory/causal-planner-direction-explored.md`) notes
deconfounding was **not** significant in prior 5-seed tests (all CIs included 0), so a null
here is a live and expected possibility — reportable as a boundary result.

**Failure interpretation.** If, at the **preregistered locked** coupling level(s), the
correlational decoy incurs no interventional regret / no action disagreement above the locked
prevalence, then structural inductive bias adds no decision value on this partially-
interventional data under these conditions — **report the null boundary result. Do NOT tune
`Z` (or the behavior policy) upward** toward a positive result; the locked parameters stand.
A gap that requires cranking `Z` into an unlocked regime is not a valid E4 result.

### E4 built minimal SCM + structural design-validation gate (FROZEN, `GATE_CONTRACT_E4.md`)

Everything in this subsection is FROZEN pre-code in `docs/benchmark/GATE_CONTRACT_E4.md` (ruling
`.herdr/reports/rule-e4-design.md`, which CORRECTED two blockers in the `e4-research.md`
proposal). It supersedes, for the built unit, the "reuse Env IV backbone" sketch above: the built
E4 is the **minimal single-outcome** confounded SCM (ruling Q6, Q9), NOT the legacy Env-IV graph
(whose `Z` sits on outcomes only — the old null).

**Minimal confounded SCM (`cdd_oran/envs/v2/e4.py`).** One acted param `A = P0`, one KPI
`K_out = K0`, one latent confounder `Z` on all three required edges:

```
Z ──► A_behavior   (confounder biases the data-collection action)
Z ──► K_out        (confounder moves the outcome)
A ──► K_out        (the true causal effect to recover)

K_out(q+1) = alpha*A_q + theta*Z_q + c   (alpha=-1, theta=+2.5, c=0; one-step lag, [SEM] PD2)
Z ~ Normal(0,1),  eta ~ Normal(0, 0.5^2),  A_behavior = clip(0.5 + lambda*Z + eta, 0, 1)
```

`Z` is **latent** (never a param or KPI, never exposed to a planner), so the *observed* env
adjacency is only `(K0,P0) = (0,0)`; the two `Z`-incident edges are mandatory latent-edge SCM
metadata. Data collection has two modes on a preregistered `rho_obs:rho_do = 90:10` mixture:
**observational** (`A = A_behavior`, the same `Z` enters `K_out`) and **interventional**
(`do(A=v)` set independently of `Z`, `Z` still enters `K_out`). `Z`, `eta`, and the randomized
do-action use dedicated NON-ALIASING coordinate-keyed tape slots; pending + committed `Z` are
carried in snapshot/restore. Decisions are scored on the latent-noiseless interventional-mean
reduction `K_score(a) = E_Z[K_out|do(A=a)] = -a` (`Z` integrated to `E[Z]=0`; ruling Q10).

**Structural design-validation gate + λ=0 control (PRE-REGISTERED gate;
`scripts/e4_structural_gate.py`).** Deterministic, NO training. It checks that the benchmark
*expresses* action-relevant confounding: it recomputes the exact clipped-normal moments and the
exact 90:10 naive pooled line `K_hat_naive(a) = c_pool + b_pool*a` (slope `b_pool = +3.29` —
**sign-reversed** from the true `alpha = -1`), builds the 64-state geometry bank (seeds `0..63`),
has the aware model `K_hat=-a` and the naive pooled model each select the locked hinge `argmax`
over `V`, realizes both on the true interventional-mean SCM, and computes the graded structural
`gap_norm = 0.6695395708852947`. The **factor-removal control** removes exactly `Z -> A_behavior`
(`lambda=0`), leaving everything else fixed; the naive pooler then recovers slope `-1`, agrees
with the oracle, and the control gap collapses to `0`. **PASS iff both**
`mean_bank(gap_norm) >= 0.10` **AND** `mean_bank(abs(gap_norm_control)) <= 0.01` — both clauses
wired into the result and the process exit (the E2 lesson). Deterministic success says the design
contains the intended action-relevant confounding trap.

**Trained-arm decision value is DEFERRED (requires training).** The structural gate + λ=0 control
own benchmark *validity*; they do NOT claim a learned causal arm wins. The structure-aware
(`D=1`-restricted regression), correlational-pooler, discovered, and dense arm comparison on the
matched 90:10 corpus — with `pi_E4=0.50`, `tau_E4_trained=0.10` defaults — is **DEFERRED** and,
per `memory/causal-planner-direction-explored.md`, may honestly be null. Its null must NOT trigger
changes to `Z`, `lambda`, the mixture, or the bank.

---

## E5 — Integrated stress (composition capstone)

**Scientific hypothesis.** When the independently validated mechanisms of E2 (shared-control
fan-out), E3 (temporal chain), and E4 (action-relevant confounding) interact under moderate
observation noise, the complete causal system remains effective — its aggregate and
worst-case decision regret stay bounded relative to the decoy/correlational arms.

**Causal graph / SCM sketch.** **Composition only — no new invented equations** (plan
Phase 8 / Decision Gates). One graph carrying, at moderate density:

- a shared-control fan-out block from E2 (`P_shared -> {multiple xApps}`);
- a validated temporal KPI chain from E3 (`K -> K -> K` at one-step lag, **[SEM] PD2**);
- a validated `Z -> A_behavior`, `Z -> K_out`, `A -> K_out` confounding block from E4;
- **moderate observation noise ON** (unlike E1) — a config sweep value, not a new mechanism.

Each sub-mechanism must have **already passed** its own SCM + decision-gap + control gates in
E2/E3/E4 before entering E5 (E5 begins only after E2–E4 pass independently, plan Phase 4).

**Controlled characteristic.** Integration — the interaction of the three validated axes plus
moderate noise. Not an isolated noise or confounding ablation (Phase 8).

**Deliberate decoy / trap.** **Combined traps:** the incomplete fan-out (E2), the myopic
horizon / truncated chain (E3), and the correlational observational structure (E4) together.
No trap is novel to E5.

**Designated baseline / comparison arm.** Full-causal oracle (reference) vs the combined
decoy/correlational arm vs discovered structure; dense capacity-matched arm; evaluated under
the **common support-shift severity rule owned by SEMANTICS.md §7** (not a per-env ad-hoc
range), so ID→OOD degradation is comparable across environments.

**Primary metric.** **Aggregate and worst-case decision regret** across the composed
conflict/temporal/interventional decisions — mean regret and the max (worst-case) regret over
the decision population, ID and OOD.

**Expected result (open).** If E2–E4 mechanisms are real and compose, discovered ≈ oracle
with bounded aggregate and worst-case regret ≪ combined decoy, with graceful ID->OOD
degradation. A collapse under composition (a mechanism that held alone but fails when
combined) is a legitimate integration-negative result.

**Failure interpretation.** Because E5 introduces no new mechanism, any E5 failure localizes
to an **interaction** between already-validated parts (or to the added noise), not to an
untested equation. If E5 needs a mechanism not validated in E2–E4 to show a gap, it is
rejected outright (Decision Gates: "E5 introduces mechanisms not already validated
independently").

---

## Cross-cutting requirements (apply to all envs)

- **One stress axis per env.** E1 = cleanliness; E2 = fan-out breadth; E3 = temporal depth;
  E4 = action-relevant confounding; E5 = composition. A decoy that needs a second stress axis
  ⇒ redesign (Decision Gates).
- **Matched comparisons.** All arms scored on matched data, matched decision populations
  (`enum_source` canonical enum graph, `evaluate.py:445-467`), matched capacity (dense arm
  must be genuinely capacity/ensemble-matched — currently only reported, not enforced), and
  matched planner settings.
- **Latent noiseless scoring [SEM] PD3** for every oracle/regret computation.
- **Verified gap before training (E2–E5).** No env advances past the Phase-4 decision-gap
  gate without a preregistered, verified oracle-vs-decoy realized-regret gap; thresholds are
  fixed in the WP4 gate contract BEFORE any simulator search (MAJOR 11).
- **ID/OOD = common support-shift rule.** ID→OOD uses the single **common support-shift
  severity rule owned by SEMANTICS.md §7**, applied identically across E1–E5; no env defines
  its own ad-hoc NCP range. Report overlap/extrapolation statistics; mechanisms are preserved.
- **Timing owned by SEMANTICS.md** — E1/E3 chain timing is reconciled against the locked
  `docs/benchmark/SEMANTICS.md` (hash in the reconciliation stamp above); on any divergence
  SEMANTICS.md wins and the stamp is re-bumped.
