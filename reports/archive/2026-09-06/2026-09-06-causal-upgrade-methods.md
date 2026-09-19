# Causal upgrade — structure-learning methods over our RCoT CI oracle (2026-09-06)

**Lane 1 of 3 research report (read-only; informs a FUTURE fresh pre-registration, changes no frozen method).**

**Question we were asked.** Our frozen discovery (RCoT-v2, `cdd_oran/e2slice/discovery_rcot_v2.py`) runs a
Randomized Conditional Correlation Test *per candidate edge* and selects with per-target BH-FDR; direction
comes only from the known temporal order. RCoT is exactly the CI-test *oracle* that constraint-based causal
learners (PC, PC-stable, FCI) consume. **What do we GAIN by wrapping RCoT inside a PC/FCI structure-learner
versus our independent per-edge BH-FDR?**

**One-line answer.** For E2 specifically, wrapping RCoT in a *full CPDAG/PAG learner* (PC → v-structures →
Meek, or FCI) buys **almost nothing on orientation** — the known temporal order already orients every edge,
so the whole orientation/equivalence-class layer is redundant. The **one genuinely valuable idea** in the
constraint-based family is the *skeleton phase's adaptive, level-wise, minimal conditioning sets*, which
directly attacks the known weak point of our current design: a single **maximal 13-dimensional** conditioning
on every edge. The concrete recommendation is therefore a **tiered PC-stable *skeleton-only* pass** (equivalently
the condition-selection stage of PCMCI/PCMCI+) using our existing frozen RCoT as the pluggable CI oracle — **not**
a full PC/FCI. Details, what it buys, honest caveats, and a minimal truth-free prototype below.

---

## 1. E2 structure, restated in causal-graph terms

From the env definitions and the frozen slice layout (`discovery_rcot.py` docstring, memory
`006`/`007`/`e2-method-redesign-decision`):

- **Nodes.** Per target-time `t`: 8 exogenous params `P(t)`, 6 KPIs `K(t)`. Candidates for each current KPI
  target = **14** = {8 params `P(t)`, 6 *lagged* KPIs `K(t-1)`}. Targets = **6** current KPIs `K(t)`.
- **True edges.** `P(t) → K(t)` only, via nonlinear Gaussian-bump **deterministic** maps; **16** true
  `param→KPI` edges out of 48 possible. Params are **mutually independent** and **iid across time**. KPIs are a
  **pure function of the current params** — *no autoregression*, so all **36 lagged `K(t-1)→K(t)` candidates are
  true-negatives**, and the other `48-16 = 32` `param→KPI` cells are true-negatives too.
- **Determinism / noise.** `obs_noise = 0`; `K = f(P)` exactly (near-deterministic in finite sample).
- **Temporal order is KNOWN.** Every candidate (`P(t)` or `K(t-1)`) is temporally prior to every target
  `K(t)`. So the *entire* candidate set sits in one tier and every target in the next.

Three graph facts follow that dominate the whole analysis:

1. **Every edge is already oriented** by time (`candidate → target`). There is nothing left to orient.
2. **There are no latent confounders** among modelled variables (params are observed and exogenous; noiseless).
3. **The deterministic variables (KPIs) are terminal sinks** — no edge *leaves* a determined node in E2.

---

## 2. What a constraint-based learner actually adds — component by component

A PC/FCI pipeline is four stages. Evaluated against the three facts above:

### 2a. Skeleton discovery (adaptive, level-wise conditioning) — **the only real prize**
PC-stable does **not** condition each pair on all other variables at once. It searches by *conditioning order*:
test marginal independence (`|S|=0`), then `|S|=1`, `|S|=2`, … conditioning only on subsets of the pair's
current *adjacencies*, and deletes an edge the moment **any** subset `S` makes the pair independent
(that `S` is the pair's *separating set*)
([causal-learn PC docs](https://causal-learn.readthedocs.io/en/latest/search_methods_index/Constraint-based%20causal%20discovery%20methods/PC.html);
[dodiscover PC tutorial](https://www.pywhy.org/dodiscover/dev/tutorials/markovian/example-pc-algo.html)).

Contrast our frozen method: it conditions **every** candidate on the **full 13-dim Z** (the other 13 candidates)
in a single test. On E2 that maximal conditioning is what strains the kernel CI test (a 13-D RFF regression) and
is what invites the residual-collapse guard.

What a tiered skeleton search does to E2, edge by edge:
- **36 lagged `K(t-1)→K(t)` true-negatives → removed at order 0 (marginal test).** Params are iid across time,
  so `K(t-1) ⊥ K(t)` *marginally*. No conditioning needed — a cheap, well-powered, well-calibrated 1-D-vs-1-D
  RCoT kills them. This sidesteps the exact `block_perm × BH-resolution` artifact that forced RCoT-v1→v2
  (`reports/2026-09-06-e2-rcot-lowrecall-diagnosis.md`), because that artifact only bites when a **strong true
  edge** must clear the per-target BH leading threshold — for a true-negative removed at order 0 it is irrelevant.
- **32 `param→KPI` true-negatives → removed at order 0 (or low order).** A non-parent param is (near-)independent
  of the KPI; the marginal kernel test removes it.
- **16 `param→KPI` true edges → survive all orders.** A genuine parent has **no** separating set (params are
  independent, so no other candidate screens it off), so the edge is never deleted.

**Net structural claim:** on E2 *most* decisions are made with **low-dimensional** conditioning, precisely where
RCoT has power and near-nominal calibration, and where the determinism strain and the residual-collapse guard are
**least** likely to fire. This is a principled reason to expect a skeleton search to reach the same-or-better
recall/precision as our single full-Z test **with far less high-dimensional kernel regression**. That is the gain.

### 2b. v-structure (collider) orientation — **buys nothing for E2**
PC then orients unshielded colliders `A → C ← B` from separating sets, then applies **Meek rules** to propagate.
E2 *is* full of colliders — every KPI with ≥2 parents is `P_i → K ← P_j` — but their orientation is **already
fixed by time** (`param(t) → KPI(t)`). There are no undirected edges left for Meek rules to resolve. The output
CPDAG **collapses to the true DAG** the instant the skeleton is known. So the collider/Meek/CPDAG machinery — the
part that makes PC "causal" rather than "associational" — contributes **zero** on E2.

### 2c. FCI (latent confounders, PAG output) — **buys nothing for E2**
FCI adds possible-d-sep tests and bidirected edges to a **PAG** to stay sound under hidden confounding
([faithfulness/latent survey](https://arxiv.org/pdf/2303.15027)). E2 has **no latent confounders**, so FCI's
extra tests and PAG symbology add cost and ambiguity for no gain here. (FCI/RFCI become relevant only if a *later*
env, E3–E5, hides a common driver — flag for the roadmap, not for E2.)

### 2d. Error control — our BH-FDR is actually the *stronger* story
PC is a **cascade of α-level CI tests with no graph-level error guarantee**, and is order-dependent unless you use
PC-stable ([causal-learn PC](https://causal-learn.readthedocs.io/en/latest/search_methods_index/Constraint-based%20causal%20discovery%20methods/PC.html)).
Our **per-target BH-FDR** is an explicit, clean FDR guarantee per target. PCMCI+ retrofits an FDR-style control
onto the MCI step for time series
([tigramite/PCMCI](https://github.com/jakobrunge/tigramite)), but plain PC does not. So moving to a PC skeleton
**would cost us the tidy BH-FDR guarantee** unless we re-add FDR on the skeleton tests (nontrivial; active
research). This is a genuine trade-off, **not** a strict win — worth stating plainly against the user's
"genuine causality" framing: the causal upgrade is real on *conditioning strategy*, but weaker on *error control*.

---

## 3. How the KNOWN temporal order simplifies everything

The `t→t+1` order is *tiered background knowledge* `K_τ=(R,F)` with `F` = all edges from a later tier to an
earlier one ([tiered-PC completeness, Andrews et al. 2020](https://arxiv.org/pdf/2503.21526);
[exploiting temporal structure](https://arxiv.org/pdf/2406.19503)). Consequences:

- It **forbids** every `target→candidate` and every `candidate–candidate` orientation, so the skeleton reduces,
  per target, to a level-wise separating-set search over its 14 candidates — a strict subset of full PC, and it
  makes stages 2b/2c provably moot (nothing to orient).
- Tiered PC is **complete** under this background knowledge (Andrews et al.), so we lose no soundness by skipping
  orientation. All three mainstream libraries accept it: **causal-learn** `BackgroundKnowledge` (tiers/forbidden
  edges), **pgmpy** `ExpertKnowledge` (temporal tiers), **dodiscover** `Context` (fixed/forbidden edges), and
  **tigramite/PCMCI** is *natively* tiered by construction (lagged parents precede)
  ([causal-learn issue #171](https://github.com/py-why/causal-learn/issues/171);
  [pgmpy expert knowledge](https://pgmpy.org/examples/Expert_Knowledge.html)).

This is why the honest recommendation is a **tiered skeleton search, not a CPDAG learner**: the temporal order
already hands us what PC's expensive second half is supposed to compute.

---

## 4. The hard caveat — determinism breaks the faithfulness PC/FCI assume

This is the part that must be stated honestly. PC/FCI are sound only under **faithfulness**: every CI in the data
corresponds to a d-separation in the graph. **Deterministic relations violate this.** The canonical failure
(NeurIPS 2024, *On Causal Discovery in the Presence of Deterministic Relations*, Li et al.,
[PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/ec52572b9e16b91edff5dc70e2642240-Paper-Conference.pdf)):
for `X → Y → Z` with `Y = f(X)`, one gets the *extra* CI `Z ⊥ Y | X`, because `Y` degenerates to a constant given
`X`; PC then **wrongly deletes the true edge `Y→Z`**. Glymour's DPC and Luo's determinism-aware PC patch this by
restricting which variables enter conditioning sets in a deterministic system; the NeurIPS paper argues exact
**score-based** search under the weaker SMR assumption handles it better and proposes DGES (Determinism-aware GES:
(1) detect minimal deterministic clusters, (2) modified GES, (3) exact search on clusters+neighbors), for linear
*and* nonlinear, continuous *or* discrete data
([also Lemeire, deterministic relations](http://parallel.vub.ac.be/~jan/papers/JanLemeire_CausalInferenceWithDeterministicRelations2008.pdf)).

**How badly does this bite E2? Less than you'd fear — because of fact (3):**

- The classic failure needs a determined node with a **child** (`Y=f(X)` then `Y→Z`). In E2 the deterministic
  nodes (KPIs) are **terminal sinks** — nothing leaves them — so **the canonical "edge-out-of-a-determined-node
  gets wrongly removed" failure does not arise on E2.**
- The residual determinism risk is only the *residual-collapse* case: testing an edge **into** a target `K` while
  the conditioning subset `S` already contains **all** of `K`'s true parents, so `K|S` is constant. PC's search
  reaches that `S` only at *high* order and only for the specific subset = all other true parents; our RCoT already
  maps such cells to `p=1` via its relative residual-collapse guard (`discovery_rcot.py`). And the low-recall
  diagnosis already **empirically falsified** determinism as the E2 blocker: the conditional test rejects true
  edges raw at ~80% (`reports/2026-09-06-e2-rcot-lowrecall-diagnosis.md`, "(iii) Fundamental to noiselessness — NO").
- **Therefore: manageable, not fatal, for E2.** It becomes a real threat in **E3–E5** the moment a determined
  variable gains a child or the graph densifies — at which point a determinism-aware conditioning rule (DPC/DGES)
  or exact score search under SMR is needed, not vanilla PC.

Two more honest notes:
- **Score-based (GES) is *not* a drop-in for noiseless E2.** BIC/likelihood scores assume non-degenerate residual
  variance; a noiseless deterministic child has **zero** residual variance → degenerate/singular score. The
  NeurIPS "exact score search handles determinism" result assumes SMR / some stochasticity, not the pure-noiseless
  limit. So GES/DGES is a *roadmap* idea for denser later envs, **not** an E2 recommendation.
- **Order-dependence.** Use **PC-stable** (adjacency snapshot per level) so the skeleton is deterministic in
  the input order.

---

## 5. Ecosystem assessment (custom RCoT plug-in, license, Windows/torch fit)

| Library | Custom CI test? | Tiered/temporal BK | Output | License | Fit for us |
|---|---|---|---|---|---|
| **causal-learn** (py-why) | **Yes** — `CIT` interface; PC accepts any callable CI oracle (built-ins incl. `kci`) ([docs](https://causal-learn.readthedocs.io/en/latest/independence_tests_index/index.html)) | `BackgroundKnowledge` (forbidden/required, tiers) | CPDAG (skeleton reusable) | **MIT** ([LICENSE](https://github.com/py-why/causal-learn/blob/main/LICENSE)) | **Best general fit.** Mature, pure-NumPy/SciPy, Windows-clean, easy to inject RCoT. |
| **tigramite / PCMCI(+)** | **Yes** — subclass `CondIndTest` | **Native** (time-series tiers built in) | time-series DAG; MCI + FDR-ish control | **GPLv3** (verify — copyleft matters if we ever distribute) | **Best fit for the time-series framing** (`t→t+1`, lagged candidates) — but our whole per-target/lagged layout *is* PCMCI's setting. License caveat. |
| **dodiscover** (pywhy) | **Yes** — cleanest CI API (`ci_estimator=...`), `Context` for fixed/forbidden edges ([tutorial](https://www.pywhy.org/dodiscover/dev/tutorials/markovian/example-pc-algo.html)) | `Context` (fixed edges) | CPDAG | BSD-3 (pywhy convention — verify) | Cleanest API for a custom oracle, but **younger / v0.0.0**, thinner tiered-BK support. |
| **pgmpy** | Partial (`ci_test` callable) | `ExpertKnowledge` tiers ([docs](https://pgmpy.org/examples/Expert_Knowledge.html)) | CPDAG | **MIT** | Fine, but CI-test plug-in less ergonomic than causal-learn. |

None require a GPU; all are NumPy/SciPy and Windows-friendly. Our RFF/RCoT numerics can stay **byte-identical**
and torch-port later independently — the structure-learner only changes *which conditioning sets* are tested, not
the test.

**Verdict:** don't adopt a whole library's PC end-to-end. Reuse the **skeleton/condition-selection algorithm**
(reimplement the ~40-line PC-stable level-wise loop, or call causal-learn's skeleton with our RCoT as `CIT`),
feeding it the frozen RCoT and the temporal tier. causal-learn (MIT, custom CIT, tiered BK) is the reference to
mirror; PCMCI is the conceptual twin worth reading for the time-series framing (mind GPLv3).

---

## 6. Concrete recommendation

1. **Do NOT** wrap RCoT in a full PC/FCI CPDAG/PAG learner for E-series discovery *as it stands*. On E2 the
   orientation half (v-structures, Meek, CPDAG) and FCI's latent machinery are **redundant** (temporal order
   orients everything; no latent confounders; determined nodes are sinks). Selling that as "genuine causality"
   would be overclaiming — E2 gives the causal structure away for free via time.
2. **DO** consider replacing the *conditioning strategy*: move from "one test per edge conditioning on the full
   13-dim Z + per-target BH-FDR" to a **tiered PC-stable *skeleton* search** (marginal → order-1 → order-2 …)
   using the frozen RCoT (block_perm, `B=299`) as the pluggable CI oracle. This is the sole component that
   materially changes the E2 result, for the structural reasons in §2a.
3. **Keep the error-control question open.** Plain PC drops our clean BH-FDR guarantee. Either (a) keep per-target
   BH on the skeleton's tests, or (b) adopt a PCMCI+-style q-value control — decide *before* any fresh
   pre-registration, and measure it truth-free first.

**What it BUYS (grounded in E2 + known temporal order):** adaptive **low-dimensional** conditioning that removes
all 68 true-negative cells at order 0/low order (cheap, well-calibrated, dodging both the 13-D kernel regression
and the `block_perm×BH` resolution artifact) while keeping the 16 true edges (no separating set exists). It does
**not** buy orientation, colliders, or an equivalence class here — those are already handed to us by time.

---

## 7. Minimal first prototype (truth-free, informs a fresh pre-reg — no frozen change)

Build this on the **byte-faithful test-bed** `runs/calib-study/calib_study4_correct` (parents known *by
construction*, decoy OFF) — **not** on E2 truth — exactly as the low-recall diagnosis Part B did:

- **Implement** a tiered **PC-stable skeleton-only** pass: tier 0 = {8 params `P(t)`, 6 lagged `K(t-1)`}, tier 1
  = {6 targets `K(t)`}; only tier0→tier1 edges allowed. Per target, level-wise separating-set search over its 14
  candidates using the **imported, unchanged** RCoT test (`discover_graph_rcot` numerics; block_perm `B=299`).
  Cap conditioning order (e.g. `≤2 or 3`) — E2's parents-are-independent structure means high order is rarely
  needed and is where determinism bites.
- **Compare, truth-free**, against the current design on the test-bed:
  - (a) frozen single-full-Z + per-target BH  vs  (b) PC-stable minimal-Z skeleton.
  - Metrics: raw power to *keep* the true `param→KPI` parents; raw FP on the lagged-`K` and false-`param`
    candidates; **at which conditioning order** each true-negative is removed; how often the residual-collapse
    guard fires per order; and **compute** vs the full-Z baseline.
  - Also run the **order-dependence** check (PC vs PC-stable) to confirm determinism.
- **Success criterion (pre-decide):** skeleton (b) matches-or-beats (a) on recall AND on KPI→KPI FP, using
  strictly *lower-dimensional* conditioning. If yes → it justifies a fresh pre-registration of a "tiered
  RCoT-skeleton" method; if no → the full-Z BH design stands and we've bounded the ceiling honestly.

Everything RCoT-numeric stays frozen; **only the conditioning-set *selection* changes.** No E2 truth is read; no
frozen artifact is touched.

---

## Sources
- dodiscover PC tutorial — https://www.pywhy.org/dodiscover/dev/tutorials/markovian/example-pc-algo.html
- causal-learn PC + CIT docs — https://causal-learn.readthedocs.io/en/latest/search_methods_index/Constraint-based%20causal%20discovery%20methods/PC.html , https://causal-learn.readthedocs.io/en/latest/independence_tests_index/index.html
- causal-learn license (MIT) — https://github.com/py-why/causal-learn/blob/main/LICENSE ; background-knowledge issue — https://github.com/py-why/causal-learn/issues/171
- pgmpy expert/temporal knowledge — https://pgmpy.org/examples/Expert_Knowledge.html
- tigramite / PCMCI — https://github.com/jakobrunge/tigramite
- Li et al., *On Causal Discovery in the Presence of Deterministic Relations*, NeurIPS 2024 (DGES) — https://proceedings.neurips.cc/paper_files/paper/2024/file/ec52572b9e16b91edff5dc70e2642240-Paper-Conference.pdf
- Lemeire, *Causal Inference on Data Containing Deterministic Relations* (DPC) — http://parallel.vub.ac.be/~jan/papers/JanLemeire_CausalInferenceWithDeterministicRelations2008.pdf
- Tiered background knowledge / temporal structure — https://arxiv.org/pdf/2503.21526 , https://arxiv.org/pdf/2406.19503
- Survey (constraint- vs score-based, faithfulness) — https://arxiv.org/pdf/2303.15027
- Internal: `cdd_oran/e2slice/discovery_rcot.py`, `discovery_rcot_v2.py`; `reports/2026-09-06-e2-rcot-lowrecall-diagnosis.md`
