# The path to E5 — 3-lane synthesis (2026-09-06, E5-first)

Supersedes the E2-scoped `2026-09-06-causal-upgrade-synthesis.md` (which judged "go causal" against E2
recovery — the wrong yardstick). This judges against the END GOAL **E5**: *does complete causal fan-out
let a planner reject a locally-attractive-but-globally-harmful shared-knob action?* Lane reports:
`2026-09-06-e5-{causal-representation,structure-roadmap,effect-quantification}.md`. No code changed.

## The one finding that reframes everything

**The project's actual thesis has never been exercised.** Every gate built so far (E2/E3/E4) — and every
discovery result including RCoT-v2 — proves only that *the benchmark expresses the trap*. **None of it has
shown that DISCOVERED structure → a world-model → a planner → rejects the harmful shared-knob action.** That
end-to-end claim is the whole point of E5, and it has never been run, not even on E2. **RCoT recall is
stage-1 skeleton discovery, not the deliverable.** (Lane 1.)

## What the shared-knob decision actually needs (all lanes agree)

The decision is definitional (Lane 3, from `SPEC.md:217-221`, `SEMANTICS.md:198-239`): the shared knob is
`P0`, fanning out to `{K0,K1,K2,K5}`; a move that helps K0/K1/K2 locally pushes K5 past a satisfy-below
threshold = globally harmful. To reject it a system must: enumerate `v`, **propagate `do(P0=v)` through the
complete fan-out**, aggregate into the hinge objective `R`, argmax. The decoy trap *is literally an
effect-estimation error* (it omits `P0→K5`, mispredicting `do(P0)`'s effect on K5 as zero).

So E5 demands four things — and only the first exists:
1. **CI edge oracle** — RCoT-v2 (have it, E2 done, solid).
2. **Completeness** — recover the full oriented fan-out incl. **real multi-hop lagged KPI→KPI chains** that
   first appear at **E3** (`P0→K0→K1→{K2,K3}`). RCoT-v2 was validated only to *reject* KPI→KPI, **never to
   recover** it. **This flip is the sharpest unbuilt/unvalidated risk.** (Lane 2.)
3. **Signed interventional effects** — RCoT is a *test*, not an estimator. **E4 proves correlation gets the
   SIGN catastrophically wrong** (observational `+3.29` vs true `−1`). Need structural `do`-propagation → a
   world-model. (Lanes 1+3.)
4. **Deconfounding** — E4's latent `Z`, identifiable only from the interventional (do) subset. Least-built,
   highest-risk; a null is a legitimate boundary result. (Lanes 1+2.)

## The load-bearing blocker: a multi-intervention world model

All three lanes **and three prior memory threads** (`cma-es-gate-falsified`, `planner-set-redesign-
direction`, `causal-planner-direction-explored`) converge: a shared-knob fan-out decision *is* a
multi-intervention prediction problem. **The planner is not the bottleneck — the world-model is.** The
fan-out quantification (do-propagation through the discovered, possibly-confounded, multi-hop graph to
signed KPI effects) **IS** the world-model the planner queries. `rollout.py` already does exactly this on
the TRUE simulator (the oracle); the missing piece is doing it on *discovered/learned* structure.

## What "going causal" does NOT mean here (honest, so we don't chase fashion)

- **Orientation machinery (PC/FCI v-structures, Meek, CPDAG) is redundant THROUGH E5** — architecture-level
  fact: `k_t = f(p_{t-1}, k_{t-1})` (`base.py:83,108`) forbids within-step KPI→KPI edges, so all KPI→KPI
  fan-out is *lagged* and time-orients for free, even at E3+. The gap is **completeness, not orientation**.
- **FCI** matters only as latent *detection* (a PAG bidirected edge, no sign); the **sign comes from the
  interventional subset E4 already provides**, not from FCI.
- **Determinism-native / ANM / LiNGAM / IGCI are all redundant toward E5** — E5 turns noise ON (kills the
  determinism regime and the non-Gaussianity lever), and time already gives direction.
- **Big-data effect-estimation scalers (NVIDIA RAPIDS/DoubleML, Tencent) — right family, WRONG regime:**
  they scale confounded ATE to millions of rows; we have a ≤14-node, known-mechanism, mostly-unconfounded
  SCM. The relevant reference is **DoWhy-GCM** (fit SCM + propagate `do` through the graph = exactly the
  fan-out quantification); **EconML/DML** scoped to E4's confounded block only.

## Build order (convergent)

- **Stage 0 (systems, now):** GPU-accelerate the frozen RCoT (equivalence gate, no re-freeze). NOTE: plan
  010's "don't port RCoT" was E2-single-full-Z-scoped and **likely flips under a many-test skeleton
  search** — resolve by measurement, not assertion.
- **Stage 1 — the actual missing spine:** wire **discovered-structure → world-model → planner → decision
  END-TO-END on E2** (the simplest env), so the thesis is finally exercised. Extend the one-step learned
  world-model (`cdl.py:640`) to **multi-step interventional rollout that chains predicted KPIs through
  discovered KPI→KPI edges** — the bridge between E2's one-step predictor and E3/E5's temporal fan-out.
- **Stage 2 (E3):** build `e3slice`; recover the multi-hop KPI→KPI chain (the true-negative→true-positive
  flip); tiered PC-stable low-order conditioning (carry-forward from the discovery lane).
- **Stage 3 (E4):** deconfounding via the interventional subset (EconML/DML there); signed effects.
- **Stage 4 (E5):** noise-robust re-calibration + composition + aggregate/worst-case ID+OOD scoring.
- Critical path = **completeness + signed-effects/world-model + confounding** — NOT orientation, NOT
  determinism-exploitation. Joint/multi-param intervention closes the OOD gap that keeps the known **+4.82
  coordination win oracle-only** today.

## Bottom line for the user
RCoT-v2 is a solid E2 building block — but the next real move toward E5 is **not** a fancier discovery
algorithm. It's (1) exercise the never-run thesis end-to-end on E2 (discovery → world-model → planner →
decision), and (2) build the **multi-intervention world model** (do-propagation on discovered structure,
signed, later deconfounded) that the shared-knob decision actually needs. Discovery feeds it; it is the
lever three prior threads already named.
