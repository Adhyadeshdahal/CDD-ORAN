# Morning briefing — overnight of 2026-09-06 → 07

**Role reminder:** everything below is DEV/analysis, **truth-free**, **nothing frozen or committed**. Git
tree is clean at `9d87a60` (the E2 operating-point redesign you approved before sleeping). All the genuinely
new builds and freezes are parked for your decision (§4).

## 1. TL;DR
The E5 thesis is now **demonstrated end-to-end on E2 with *discovered* structure**, and the single hard
blocker — recovering the globally-harmful fan-out edge — is **solved observationally**. Five independent
work-streams ran; all five landed. The picture is coherent: **discovery-completeness (not the planner or the
objective) is the lever, and it is now within reach.**

## 2. What ran (all verified by me, not just the subagents)

| # | Thread | Result |
|---|---|---|
| A | Re-run frozen RCoT-v2 discovery on the redesigned (live-trap) E2 | P0→K5 = **0/10** (baseline CI still can't get the harmful edge); beneficial arrows 10/10. Verified by my own mask check. |
| B | Thesis end-to-end on the **live trap** (do-nothing / oracle / discovered) | **Fires.** Oracle/complete fan-out rejects the harmful move (regret 0, K5 satisfied 26/32); discovered-decoy falls in (regret +15, K5 satisfied 2/32); null control byte-identical on non-trap states. Spine works **given complete structure**. |
| C | **E2 harmful-edge P0→K5 recovery** (the blocker) | **SOLVED — 10/10, purely observational.** See §3. |
| D | E3 multi-hop KPI→KPI chain recovery de-risk | **100% per-edge recall, 0 FP**, both hops; the plan-007 over-conditioning/collider hazard is **refuted** (partial-corr is collinearity-invariant here). Caveat: rides on E3 being noiseless — re-calibrate at E5 noise. |
| E | FM-1/FM-2 objective re-validation on the E-series | **Null.** The legacy `utility_weight` / OOD-trust gains do **not** transfer to the E-series objective — it's the same flat hinge + analytic model. Confirms the objective is not the lever. Completes the plans/011 §2 / plans/012 step-1 gate (as a null). |
| — | E4 deconfounding feasibility (bonus) | **Viable route.** A D=1 (interventional-subset) estimator recovers true α=−1 [CI −1.20,−0.88] where the correlational pooler gives +3.30. The prior "null" 5-seed result was just **under-power**, not a real failure. |

## 3. The headline — the harmful edge is recoverable observationally
On the live-trap E2, frozen RCoT-v2 misses P0→K5 (the globally-harmful arrow) 0/10 because it is
**co-parent-gated**: P0 only moves K5 when co-parent P7 opens the Gaussian gate, so the marginal signal is
~0.3% variance and vanishes globally.

Four methods compared truth-free (BH-FDR q=0.05, 10 seeds, truth read only at scoring):

| method | P0→K5 | NCP recall /16 | KPI→KPI FP |
|---|---:|---:|---:|
| frozen RCoT-v2 (baseline) | **0/10** | ~12 | 0.60 |
| do(P0)-marginal (interventional control) | **0/10** | 11.2 | 0.70 |
| **co-parent-stratified conditional CI [WINNER]** | **10/10** | **15.3** | 0.30 |
| recall-first marginal superset + prune | **0/10** | 11.2 | 0.00 |

**The winner needs no interventional data** — the lever is the *conditioning strategy*: stratify the
observational sample by the arg-max co-parent (found truth-free as P7 every seed), and P0's effect on K5
becomes locally visible (η² 0.003 → 0.19). It also *improves* overall recall while keeping FP low.
**Interventional-marginal fails** (E2 is unconfounded, so do(P0) ≡ observational) and **marginal-superset
fails** (the edge is too weak to even admit) — both exactly as diagnosed.

**So the E2 loop closes:** observational stratified discovery → complete structure → world-model + decision
reject the locally-attractive/globally-harmful action. That is the E5 thesis, on E2, with discovered
structure.

## 4. Decisions awaiting you (all USER-GATED — I did not touch these)
1. **Promote the stratified recovery method to the live E2 discovery method** — needs a fresh
   pre-registration + an FP-calibration study (the max-over-stratifier test is more expensive and its FDR
   isn't formally calibrated yet). This is the natural next freeze.
2. **E3 discovery protocol freeze + `e3slice` module** — the de-risk (D) says the linear method recovers the
   chain cleanly; freezing it is your call. Re-calibrate at E5 noise first.
3. **E4 trained-arm** (scored deconfounding gate + matched-corpus module) — feasibility (bonus row) says the
   route works but is finite-sample fragile; the scored arm is a build decision.
4. **E5 registration** (composition SCM + gate contract) — hard-gated on E2/E3/E4 each passing their
   *discovered/learned* arm, which is now much closer.

### Two generalization caveats to carry into E3/E4 (from the recovery work)
- The single-co-parent `max over g` stratification won't cover a true **≥2-co-parent AND-gate** (would need
  max-over-pairs) — plausible from E3 on.
- The stratified null is sound only because **E2 is unconfounded**; for E4's latent Z it must be combined
  with the deconfounding do-subset route (bonus row).

## 5. Where everything lives
- Reports (gitignored): `reports/2026-09-06-e2-rcot-v2-recovery-REDESIGNED.md`,
  `…-e2-thesis-endtoend-livetrap.md`, `…-e2-p0k5-recovery-experiment.md`,
  `…-e3-chain-recovery-derisk.md`, `2026-09-07-fm-objective-eseries-revalidation.md`,
  `…-e4-deconfound-feasibility.md`.
- Memory: `next-session-handoff.md` (resume anchor, full state), `e2-harmful-edge-discovery-gap.md` (the
  blocker → solved chain), `option-a-utility-weight-validated.md` (now caveated legacy-only).
- Prototypes: session temp scratchpad (`p0k5_full.py`+`score.py`, `e2_thesis_endtoend.py`, `e3_chain_derisk/`,
  `fm_eseries/`, `e4_feas/`). Nothing in `runs/` was touched; `runs/e2slice-recovery/` intact.

**Recommended first move when you're up:** decide on (4.1) — pre-registering the stratified recovery method —
since it converts tonight's prototype win into the first frozen piece of the E2 discovered-arm.
