# 013 — E5 finalization: occupancy-conditional decision claim (durable record)

**Status:** DECISION RECORD (not an implementation plan). Codifies the user-gated decision made 2026-09-16
to finalize E5 at the **documentation/decision layer** under an occupancy-conditional scope. The finalized
contract is `docs/benchmark/E5_DATA_COLLECTION_CONTRACT.md`; this file is the tracked decision record.
Detailed evidence is in the (gitignored) working reports named below.

**Scope caveat (read first):** the load-bearing decision evidence is an **E2-identification (E2-ID),
fixed-k-NN benchmark characterization**, NOT evidence for a deployed composed-E5 system (sol, 2026-09-15).
It informs E5's scope and contract; it does not replace running E5. E5 the composed environment
(`e5.py` / gate contract) remains **unbuilt and unfrozen**; no `cdd_oran/` module changed; nothing frozen.

---

## 1. Decision: finalize E5 under the occupancy-conditional decision claim (option A)

The DRAFT E5 data-collection contract's finalization fork offered (a) finalize under a linear estimand,
(b) characterize the gated-effect MDE first then finalize, or (c) keep hardening. The gated decision-MDE
study **is** option (b) — E5's headline harmful edge P0→K5 is co-parent-gated, so a linear-only MDE was too
narrow. That study is complete, re-scored, spot-verified bit-exact, sol-reviewed SOUND-WITH-CAVEATS, with
all five cleanup caveats closed (2026-09-16). **User GO 2026-09-16: finalize now (A); world-model work
next (B).**

**The claim E5 is licensed to make (occupancy-conditional):** on the E2-ID shared-knob atom, a complete
causal fan-out prevents the locally-attractive-but-globally-harmful action **where the harm is active
(in-gate)**; **net all-deployment benefit is occupancy-dependent** and **washes out at realistic low
occupancy** via learned-world-model off-gate collateral; the binding constraint is the **learned
world-model's off-gate generalization / sample efficiency, not discovery.** E5 must NOT be registered for
the unconditional strong claim.

## 2. Evidence (gated decision-MDE, E2-ID, 40 seeds, IPW S_eval)

- **Null at every finite occupancy** q ∈ {2,5,10,20 %}: prevention power 0.00 at all δ; off-gate control
  fails everywhere.
- **First crosses only at q = 1.0:** barG_omit **0.16334**, decision power **0.925 (37/40)** at δ = 0.4 →
  **1.0** at δ ≥ 0.8; discovered arm regret 0.028 → 0.002, matching the oracle. Decision logic is sound
  where the harm is active.
- **Discovery is not the bottleneck:** edge power → 1.0 at every occupancy.
- **Oracle-structure arm ALSO fails at finite q** (q=0.2, δ=0.8: discovered 0.162 vs oracle 0.143) ⇒
  world-model, not M3 selection.
- **Severity honesty:** finite-q barG_omit (~0.13) is dominated by baseline learned-pipeline regret;
  intrinsic exact-decoy severity is 0.001–0.027.
- **Controls persisted:** factor-removal all-pass (max|diff| 6.66e-03 < 0.02; the non-zero cell is the
  bad_disc col-13 nuisance); nuisance diagnostics emitted for all 1600 cells (528 carry a lagged parent).
- Evidence: `reports/2026-09-15-gated-decision-mde-result/` (gitignored); pre-reg + harness + scorer in
  `scratchpad/design_adaptive_m3/`; sol note `sol_result_review.md`. Memory `next-session-handoff`.

## 3. What is registered vs deferred

- **Registered (doc/decision layer):** the data-collection contract (Part I of the contract) as the
  trustworthy-discovery requirement, and the occupancy-conditional decision-claim scope (Part II). ITT +
  gated-effect estimand; sol's required log-fields + scope caveats; the single-frozen-S_eval caveat.
- **NOT done (deliberately):** no `e5.py` / composed-E5 environment, no gate contract code, no method or
  constant freeze, no `cdd_oran/` change, no commit (pending user request).
- **Deferred:** correlated / non-ideal sensor noise; KPI→KPI instruments; O-RAN grounding of the abstract
  exp(−x²) mechanisms (known external-validity gap); gated-effect MDE calibration of the discovery
  power/MDE gate beyond the decision-MDE characterized here.

## 4. Next thread (B): the world-model off-gate generalization bottleneck

The characterized lever. Improve the learned world-model's off-gate generalization / sample efficiency
(learner class, regularization, off-gate coverage) so net all-deployment benefit survives at realistic
occupancy; only then is E5 positioned to test the strong (unconditional) claim. Consistent with the
standing direction that **world-model quality is the lever, not discovery**
(memory `discovery-method-no-single-superset`, `e2-harmful-edge-discovery-gap`; and FM-3 in plan 011).

---

## What this means going forward
E5 is finalized as an **occupancy-conditional** result at the documentation/decision layer: the causal
decision logic works where the harm is active, but net all-deployment benefit is occupancy-gated and the
learned world-model is the bottleneck. Building the composed-E5 environment and any strong-claim test is
gated behind thread B (world-model quality) and remains user-gated. See
`docs/benchmark/E5_DATA_COLLECTION_CONTRACT.md`.
