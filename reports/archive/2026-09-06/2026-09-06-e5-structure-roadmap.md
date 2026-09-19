# E5 structure-discovery roadmap — from RCoT (E2) to complete-oriented causal fan-out (2026-09-06)

**Lane 2 of 3 (structure discovery). Read-only research note; changes no frozen method, reads no
env truth.** Companion lanes: E3/E4/E5 *decision-value* framing (main) and the *effects* layer
(what-exactly-is). This note reframes the RCoT/CI-engine work around the **END GOAL E5**:

> *Can recovering the **COMPLETE ORIENTED causal fan-out** prevent a locally-attractive-but-globally-
> harmful **SHARED-KNOB** action, under a real KPI→KPI chain, action-relevant confounding, and
> moderate noise?* (`docs/benchmark/SPEC.md:556-599`)

E2 is a **stepping stone**, not the target. E2's four simplifications — known temporal order, causal
sufficiency (no latents), noiselessness, and *all KPI→KPI candidates are true-negatives* — are exactly
the assumptions E3/E4/E5 are designed to break. Every capability below is judged against **E5**, not
against E2 edge-recovery.

Discipline note: everything here informs a **future fresh pre-registration**. The frozen E2 methods
(pdCor `828e345`; RCoT-v2 `block_perm_reps=299`) and the frozen E3/E4 gate contracts stand untouched;
nothing may be retuned after a recovery number is seen.

---

## 1. What each env adds vs E2 (grounded in specs)

The base v2 dynamics fix one fact that dominates the whole structural analysis:
`k_t = f(p_{t-1}, k_{t-1})` — **every KPI reads the *previous* step's params and KPIs**
(`cdd_oran/envs/v2/base.py:83,108`; legacy `envs/base.py:192`; SEMANTICS §0
`docs/benchmark/SEMANTICS.md:28-35`). So **no contemporaneous (within-step) KPI→KPI edge can exist** in
this benchmark family — every KPI→KPI edge is inherently **one-step-lagged**, hence orientable by time.
Hold this; it recurs in §2/§4.

| Env | Structure it adds vs E2 | Key spec anchor |
|---|---|---|
| **E2** (baseline) | Nonlinear Gaussian-bump `K=f(P)`; **shared-knob fan-out** `P0→{K0,K1,K2,K5}`; **no KPI→KPI edge** ("isolates the shared-NCP fan-out axis, not a chain"); noiseless; causal sufficiency; direction from time. **All 36 lagged KPI→KPI candidates are true-negatives.** | `GATE_CONTRACT_E2.md:22-67`; `SPEC.md:224-236`; `discovery_rcot.py:32-41` |
| **E3** | **Real KPI→KPI chain with fan-out**: `P0→K0→K1→{K2,K3}` (+`P1→K2,P2→K3,P3→K4`). The fan-out point is an **intermediate KPI (K1)**, so the recovery target is a **depth-3 multi-hop oriented structure**, not a one-hop star. Linear coeff=1, still noiseless, still causally sufficient, still time-oriented (all chain edges are lagged). | `GATE_CONTRACT_E3.md:30-51`; `SPEC.md:299-320` |
| **E4** | **Latent confounder Z** on three edges: `Z→A_behavior`, `Z→K_out`, `A→K_out`. **Causal sufficiency FAILS** (Z is never a param/KPI; observed adjacency is only `(K0,P0)` — the Z-edges are `LATENT_EDGES` metadata, `e4.py:73`). Adds a real **behavior policy** and an **observational-vs-interventional data split** (`e4.py:24-32`). Naive pooled association is **sign-reversed** (`b_pool=+3.29` vs true `alpha=-1`). | `GATE_CONTRACT_E4.md:28-63,89-110`; `SPEC.md:390-427` |
| **E5** | **Composition only, no new equations**: E2 fan-out block + E3 KPI chain + E4 `Z`-confounding block + **moderate observation noise ON**. Scored on **aggregate and worst-case regret, ID *and* OOD** (common support-shift rule, SEMANTICS §7). | `SPEC.md:556-599`, `603-621` |

**Net for the structure layer:** toward E5 the recovery target moves from "an oriented edge list on a
noiseless, sufficient, star-shaped graph" (E2) to **"a complete, oriented, multi-hop fan-out that
contains a latent-confounded sub-block, recovered under noise, and correct enough that the downstream
shared-knob decision is right ID and OOD."** Two of E2's four crutches break by E3 (KPI→KPI
true-negative assumption; single-hop fan-out), a third by E4 (causal sufficiency), the fourth by E5
(noiselessness). Only **known temporal order survives all the way to E5** — and that is what keeps
orientation cheap (§2/§4).

---

## 2. Per-env capability audit — which "redundant on E2" machinery becomes NECESSARY toward E5

Read "first env that NEEDS it" as: the earliest env where the capability is *required to recover the
complete oriented fan-out well enough for the decision*, grounded in the specs — not merely "could be
applied."

| Capability | First env that NEEDS it | Why (spec-grounded) | Status |
|---|---|---|---|
| **CI edge test with power on nonlinear conditional dependence** (RCoT RFF kernel + per-target BH-FDR) | **E2** | E2's `K=f(P)` is non-monotone (`P0²`, width-in-denominator); a linear/marginal score is blind to it (`discovery_rcot.py:12-21`). | **Built** (frozen RCoT-v2) — the reusable core. |
| **Temporal orientation (t→t+1)** | **E2** (and sufficient through **E5**) | Every candidate is temporally prior to every target; the base one-step latency makes even KPI→KPI edges lagged (`base.py:83,108`). Params are exogenous roots. Time orients **every** edge in E1–E5. | **Built** (free from the layout). |
| **Selecting *real* KPI→KPI edges** (KPI→KPI true-negative assumption BREAKS) | **E3** | E2 has 0 true KPI→KPI edges (reject all 36); E3 has 3 real ones (`K0→K1,K1→K2,K1→K3`, `GATE_CONTRACT_E3.md:47-51`). The method must now gain **recall** on lagged KPI→KPI edges while keeping precision — the opposite of E2's "reject them all". | **Not built for recovery.** RCoT-v2 has the power (block_perm keeps cand\|Z coupling); needs re-validation where KPI→KPI is a *positive*, not a true-negative. |
| **Multi-hop chain / transitive-closure recovery + direct-vs-ancestry discrimination** | **E3** | The fan-out point is an intermediate KPI; recovering the *complete oriented* structure means telling declared `P0→K0→K1→K2` from a spurious *direct* `P0→K2`, and recovering all hops. The E3 SCM gate encodes exactly this (off-manifold source intervention, one-edge shadow ablation, `GATE_CONTRACT_E3.md:196-206`; mirrors E1 `SPEC.md:169-205`). A per-edge CI screen alone yields an edge list, not a verified oriented chain. | **Not built.** This is the first place "complete oriented fan-out" (vs edge list) has teeth. |
| **Tiered low-order (PC-stable) conditioning** | **E3** (strongly desirable), forced by **E5** | E2 gets away with one full 13-D conditioning per edge. Once real KPI→KPI edges exist (E3) and the graph densifies + noise arrives (E5), a real chain edge must survive conditioning; low-order separating-set search keeps decisions in RCoT's calibrated, powered 1-D/low-D regime and stops a true chain edge being over-conditioned away. Removes true-negatives at order 0 where calibration is trivial. | **Not built.** Prototype-A candidate (causal-upgrade-methods §2a, §7). |
| **Latent-confounder handling** (obs/interventional split + interventional identification; FCI *detection* optional) | **E4** (composed into **E5**) | Causal sufficiency fails; `P(K_out\|do A)` is **not** identified from `P(A,K_out)` — the observational slope is sign-reversed (`GATE_CONTRACT_E4.md:89-110`; `SPEC.md:400-404`). Identification comes from the **randomized-interventional subset**, not from a graph-orientation trick. The structure layer must (a) consume `LATENT_EDGES` metadata / know the block is confounded, (b) route effect estimation through the do() subset. | **Deferred by design.** E4's built gate is *analytic* (no discovery arm runs; `GATE_CONTRACT_E4.md:156-215`). Needed live at E5. |
| **Noise-robust CI calibration + re-tuned BH thresholds** | **E5** | E5 turns observation noise ON (`SPEC.md:569`). RCoT's null calibration and the BH-FDR resolution were validated **noiseless**; they must be re-calibrated under noise before scoring. | **Not built.** A re-calibration, not new machinery. |
| **Genuine orientation: v-structures / colliders / Meek / CPDAG** | **None in E1–E5** | Colliders exist (e.g. `P1→K2←K1` in E3) but are **already oriented by time**; no undirected edge is ever left for Meek to resolve. The one-step latency forbids contemporaneous KPI→KPI structure, so there is never a within-step orientation ambiguity. | **Redundant** unless a future env breaks the latency architecture (§4 flag). |
| **FCI / RFCI PAG discovery of the latent** | **None as *identification*; E4/E5 as *diagnosis* only** | FCI can *detect* the `A–K_out` confounding (bidirected edge) but gives **no sign** (discovery-method-scout `2026-09-05` E4 row). The sign comes from interventions, which E4/E5 already provide. So FCI is a label-free *flag*, never the solver. | **Optional/diagnostic.** Not on the critical path. |
| **Functional / ANM / PNL / LiNGAM / determinism-native (H(Y\|S)=0) discovery** | **None required** | Determinism-native screening is an *E2/E3 optimization* that exploits noiselessness (causal-upgrade-synthesis §"(B)"). At **E5 noise is ON**, so the determinism regime is gone; ANM/PNL need noise but also need *unknown* direction — which we never have (time gives it). `Z~Normal`, obs-noise Gaussian ⇒ the **non-Gaussianity lever (RCD/LiNGAM) is unavailable** at E5 anyway. | **Redundant toward E5.** Orthogonal to the end goal. |

---

## 3. Reusable core vs E5-specific additions, and a concrete build order

### 3.1 The reusable core (E2's blocks genuinely carried forward, not thrown away)

1. **Batched CI engine (RCoT RFF kernel test).** The `_rff_residuals` → `_rcot_statistic` →
   `block_perm` null → residual-collapse guard pipeline (`discovery_rcot.py:267-416`) is the **CI
   oracle for every env**. It is method-agnostic to *which* conditioning set it is handed, so it is
   reused verbatim as the plug-in oracle for a skeleton search (causal-upgrade-methods §5: RCoT is
   exactly the `CIT` interface PC/PC-stable consumes).
2. **Per-target BH-FDR selection** (`bh_fdr_reject`, imported into RCoT from pdCor,
   `discovery_rcot.py:94-95`) — the graph-level error-control layer; keep it (it is a *cleaner* story
   than plain PC's uncontrolled α-cascade, causal-upgrade-methods §2d).
3. **Temporal-order orientation** — reused for E3/E5 (all edges lagged).
4. **The freeze / persist / fail-closed-load / post-hoc recovery-scoring discipline**
   (`discovery_rcot.py:550-706`) — the anti-p-hacking machinery transfers unchanged to every fresh
   pre-registration.
5. **GPU acceleration of the CI engine** (plan `010`) — see the tension flag in §3.3.

### 3.2 E5-specific additions (net-new toward complete-oriented fan-out), in dependency order

- **(S1) Tiered PC-stable skeleton search over the reused RCoT oracle** — low-order conditioning +
  temporal tier + per-target BH-FDR. First materially useful at E3 (real KPI→KPI edges must survive
  conditioning), load-bearing at E5 (denser graph + noise). *New science → fresh pre-reg.*
- **(S2) Chain / transitive-closure recovery + direct-vs-ancestry discrimination** — turns the S1
  edge list into a *verified oriented multi-hop fan-out* (the E3/E1 off-manifold-source + one-edge-
  shadow logic, `GATE_CONTRACT_E3.md:196-206`). This is the operational meaning of "complete oriented
  fan-out" for E5's end goal.
- **(S3) Obs/interventional data split consumption + interventional identification** for the
  confounded block — the structure layer reads `LATENT_EDGES`, marks the block unidentified from
  observation, and hands the effect layer the do()-subset route (coordinate with the effects lane).
  FCI-style latent *detection* optional.
- **(S4) Noise-robust re-calibration** of the CI null and BH resolution under E5's moderate noise.
- **(S5) OOD support-shift evaluation** of recovered structure (SEMANTICS §7 common rule) — recovery
  must hold ID *and* OOD for E5's regret metric.

### 3.3 Concrete build order (each stage reuses the prior; freeze-clean)

1. **Stage 0 — systems only, no science change: GPU-accelerate the frozen RCoT-v2 CI engine** behind
   a `device=` flag, NumPy path as oracle, **equivalence gate** (GPU masks == CPU masks) — behavior-
   identical ⇒ no re-freeze (plan `010` §4-6; causal-upgrade-synthesis "Stage (a)"). This yields the
   reusable batched CI engine every later stage leans on.
   **Flag the tension:** plan `010` §2-3 recommends *against* GPU-porting RCoT (tiny RFF dims,
   overhead-bound, already ~3–10 min/seed) — but that verdict is scoped to E2's **single full-Z, 84-
   test** design. A **tiered skeleton search issues far more, lower-order CI tests**, and E3/E5 will
   want larger n / more reps under noise, which flips the economics toward **batched GPU-RCoT as the
   CI oracle driving a CPU skeleton loop** (causal-upgrade-synthesis "GPU"). Resolve this when S1 is
   prototyped: measure test-count × per-test cost under the skeleton search, not under E2's design.
2. **Stage 1 — E3-driven (fresh pre-reg): S1 + S2.** Prototype truth-free on the byte-faithful testbed
   (`runs/calib-study/calib_study4_correct`, parents known by construction, `discovery_rcot.py:16-18`),
   extended to carry **real** lagged KPI→KPI edges. Bar: match-or-beat RCoT-v2's ~0.7 recall / ~0.01
   KPI→KPI-FP **while KPI→KPI edges are positives**, using strictly lower-dim conditioning, and
   correctly recovering the depth-3 chain with direct-vs-ancestry discrimination
   (causal-upgrade-methods §7). Freeze → score against `E3V2Env().true_adj_matrix()`.
3. **Stage 2 — E4-driven: S3.** Add obs/interventional split consumption + interventional
   identification for the confounded block (E4's live discovery arm is currently *deferred*,
   `GATE_CONTRACT_E4.md:207-215`; this stage is where it gets built). FCI detection as an optional
   label-free confounding flag.
4. **Stage 3 — E5 capstone: S4 + S5 + composition.** Re-calibrate CI null + BH under moderate noise,
   compose the three blocks into one graph, verify complete-oriented-fan-out recovery drives the
   correct shared-knob decision, and evaluate ID→OOD via the SEMANTICS §7 support-shift rule. No new
   mechanism may enter here (`SPEC.md:595-599`).

The critical path to E5 is **S1→S2 (chain + completeness) then S3 (deconfounding) then S4 (noise)** —
i.e. *conditioning strategy, completeness, and confounding*, **not** orientation and **not**
determinism-exploitation. The reusable CI engine + BH-FDR + temporal orientation carry through all of
it unchanged.

---

## 4. Honest flags — genuinely needed vs still redundant

- **Orientation machinery (v-structures / colliders / Meek / CPDAG) stays REDUNDANT through E5** — an
  *architecture-level* fact, not a lucky property of E2. `k_t=f(p_{t-1},k_{t-1})` (`base.py:83,108`)
  forbids contemporaneous KPI→KPI edges, so every edge (params→KPI and KPI→KPI alike) is temporally
  oriented for free. It would only become necessary if a **future** env deliberately broke the one-
  step-latency architecture to introduce *within-step* KPI structure — which no current spec does.
  Answering the task's question directly: **within-step KPI→KPI chains do not exist in E1–E5**, so
  genuine orientation is never the gap; *completeness* of the (already-oriented) fan-out is.
- **FCI/latent handling is REQUIRED as *detection/awareness* but NOT as *identification*.** Needed
  first at E4 (causal sufficiency fails), but the sign is unidentified from association — FCI gives a
  PAG bidirected edge, never the effect direction. The real requirement is the **obs/interventional
  split + do()-subset identification** the benchmark already provides. Do not oversell FCI as the E4/E5
  solution; it is a flag, not a solver (discovery-method-scout E4 row; `SPEC.md:400-404`).
- **Determinism-native / ANM / LiNGAM methods are REDUNDANT toward the E5 goal.** They are an
  E2/E3-only optimization that exploits noiselessness; E5 turns noise ON (killing the determinism
  regime) while temporal order still removes the orientation problem they solve, and the Gaussian
  `Z`/noise removes the non-Gaussianity lever. Real value only if a *future* env has noise **and**
  unknown direction — not the case here.
- **The KPI→KPI true-negative assumption is the sharpest thing that breaks (at E3), and it is not yet
  covered.** RCoT-v2 was frozen and validated in a world where every KPI→KPI candidate is a
  *true-negative* to reject (`discovery_rcot.py:32-41`); the whole v1→v2 fix concerned keeping
  KPI→KPI **FP** low. E3 inverts this: KPI→KPI edges become **positives to recover**. The block_perm
  null keeps the candidate\|Z coupling that gives it power there, but **recall on real lagged KPI→KPI
  edges is unvalidated** and must be measured truth-free before any E3/E5 pre-reg.
- **GPU-accelerate-RCoT verdict is context-dependent — resolve it against the skeleton design, not
  E2's.** Plan `010` says don't; the skeleton search may say do. This is a measurement, not a belief;
  flagged so Stage 0/1 doesn't inherit E2's scoping by accident.
- **E4's live deconfounding arm is *deferred*, so the confounding capability is the least-built and
  highest-risk piece toward E5.** Memory (`causal-planner-direction-explored`) records deconfounding
  as non-significant in prior 5-seed tests — a null at E4/E5 is a live, legitimate boundary outcome,
  and the roadmap must not tune `Z`/λ upward to manufacture a gap (`GATE_CONTRACT_E4.md:196-215`;
  `SPEC.md:497-502`).

---

## Sources

- Specs: `docs/benchmark/SEMANTICS.md` (§0 timing, §7 ID/OOD), `docs/benchmark/SPEC.md`
  (E2 224-285, E3 289-386, E4 390-552, E5 556-599, cross-cutting 603-621).
- Frozen gate contracts: `docs/benchmark/GATE_CONTRACT_E2.md`, `GATE_CONTRACT_E3.md`,
  `GATE_CONTRACT_E4.md`.
- Frozen discovery: `docs/benchmark/E2_RCOT_DISCOVERY_PROTOCOL_V2.md`;
  `cdd_oran/e2slice/discovery_rcot.py` (RCoT engine); `cdd_oran/e1slice/discovery_v2.py`
  (E1 partial-corr, orientation-by-time).
- Env architecture: `cdd_oran/envs/v2/base.py:83,108` (one-step latency), `envs/base.py:192`;
  `cdd_oran/envs/v2/e4.py:24-73` (`LATENT_EDGES`, obs/interventional modes).
- Related lane reports (E2-scoped, directly extended here): `reports/2026-09-06-causal-upgrade-methods.md`
  (tiered PC-stable skeleton over RCoT; orientation redundant on E2; determinism caveat for E3–E5),
  `reports/2026-09-06-causal-upgrade-synthesis.md` (GPU staging rule; per-env re-audit),
  `reports/2026-09-05-discovery-method-scout.md` (RCoT/WGCM/FCI/RCD per-env fit).
- Systems: `plans/010-gpu-acceleration.md` (GPU CI-engine, equivalence gate, RCoT-port caveat).
