# E5 finalization — data-collection contract + decision-claim scope

**Status: FINALIZED (2026-09-16).** Supersedes the working draft
`scratchpad/design_adaptive_m3/DRAFT_E5_data_collection_contract.md` (gitignored). This is the
git-tracked, durable record. "Finalized" here means the **documentation / decision layer** is settled:
the data-collection contract for trustworthy discovery, and the **occupancy-conditional scope** of E5's
decision claim. It is **not** a code artifact — E5 the composed environment (`e5.py` / gate contract) is
**not built**, nothing is frozen, no `cdd_oran/` module changed. Detailed evidence lives in the
(gitignored) working reports named below.

**Scope caveat (read first).** The decision-claim evidence in Part II is an **E2-identification (E2-ID),
fixed-k-NN benchmark characterization**, not evidence for a deployed composed-E5 system (sol, 2026-09-15).
It informs the E5 scope and contract; it does not stand in for running E5 itself. The discovery contract
in Part I rests on the 2026-09-09/10 calibration-stress + 2026-09-12 Phase-4b hardening tests.

---

## Part I — Data-collection contract for trustworthy causal discovery

### The one-sentence version
If you want the discovery engine to say "knob X causes problem Y" and be *trusted*, the data it runs on
must be collected as a **logged, (block-)randomized experiment** — on passively-collected or
adaptively-commanded data the same engine gives **guaranteed false alarms** (demonstrated: 40/40).

### Why a contract at all (what the tests showed)
- **Purely observational / non-randomized actuation cannot be fixed by any statistic.** When knobs are
  carried or partially actuated, innocent knobs are *genuinely* correlated with outcomes; no test
  separates that from causation. (Calibration stress: false-discovery 0.44–0.81; no diagnostic rescued it.)
- **Randomized per-step actuation makes it work** — the engine recovers even a faint buried cause and
  refuses look-alike knobs, robustly to heavy noise, imperfect actuation, and thin ranges.
- **BUT if the command depends on a hidden operational state that also drives the outcome, and that state
  is not logged, the engine false-alarms 100% of the time.** Logging the batch/session or the state fixes
  it to 0% (block-CRT / state-conditioned CRT). The fix lives in the *data collection* — this is the list.

### What must be logged (each item tied to the failure it prevents)
For every actuation event on every parameter (knob):
1. **Assignment policy version + the probability/distribution the command was drawn from.** The engine
   re-simulates the null by re-drawing commands from this distribution; without it, no valid test.
2. **Pre-command operational state `S`** the policy adapts to (load, topology, traffic regime, prior
   KPIs…). **REQUIRED.** Prevents the confounding break: unlogged adaptive state ⇒ 100% false alarms;
   logged (state-conditioned CRT) ⇒ ~0%. Primary, robust fix; stays valid even under within-block drift.
3. **Block / session identifiers.** A cheaper *partial* substitute for #2: permuting commands within a
   block fixes validity without modelling the state — **but only if the state is ~constant within the
   block.** (2026-09-12 drift test: under within-block drift, block-only conditioning false-alarms again,
   0%→93%→100% as drift grows; state-conditioning stays ~0%.) Log block IDs, but trust block-only
   conditioning only with evidence the state is ~constant within a block; when in doubt use logged state.
4. **Commanded AND realized values** for each knob — and **the engine analyzes the COMMANDED (randomized)
   value, never the realized one.** (2026-09-12 endogenous-compliance test: a controller that pulls a knob
   back when the KPI runs high makes the REALIZED column look causal — analyzing realized false-alarms
   100%; analyzing the commanded value stays valid at 0% and recovers the true cause 100%.) Log realized
   only to confirm actuation / detect non-compliance; identification rides on the commanded assignment.
5. **Clipping / saturation / actuation-failure indicators**, and **the exact set of levels each knob was
   commanded from.** (2026-09-12 positivity test: with a resample matched to the real assignment, discrete
   actuation to 2 levels and clipping to 10% of range stayed valid AND fully powered; a mismatched
   continuous resampler is what corrupts it — a distance-to-support check catches the mismatch.)
6. **Independent within-block variation of each knob.** Each knob must be varied independently of the
   others within each block/state stratum (positivity). Where it fails, abstain and request more varied
   actuation (factorial dither), not guess. A low distinct-level count is an *advisory*, not an auto-reject.

### The procedure the logged data feeds (the two-route engine)
1. **Admission certificate (design-only, pre-outcome):** independently randomized (not carried)? knobs not
   near-collinear? adequate within-block variation and coverage? If not → **abstain and request
   intervention** (dither / re-collect); never fall back to a different observational detector.
2. **On admitted data → analyze** with the assignment-aware, block/state-conditioned randomization test:
   resample each commanded knob within its block / from `P(command | S)`; recover parameter→KPI edges with
   controlled false-discovery.
3. **Power / MDE gate (outcome-based):** before trusting a *"no edge"* conclusion, compute the minimum
   detectable effect by injection-and-recovery; **trust a null only if MDE ≤ target effect**, else report
   "underpowered — cannot rule out effects up to MDE" and request more range/samples. (A positive
   detection is valid regardless of power; the gate only guards NEGATIVE conclusions.)
4. **Separate the two edge classes:** parameter→KPI is identified by the randomized commands; **KPI→KPI is
   NOT** directly identifiable from parameter randomization (deferred — needs an instrument), marked
   ineligible.

### Sol's adversarial review (2026-09-12) — incorporated
- **Declare the estimand ITT.** Commanded-value analysis identifies intention-to-treat, not the effect of
  realized exposure; a realized-exposure (IV/CACE) analysis must be separately validated.
- **State logging is not automatically sufficient.** Valid only if the FULL adapting state is captured
  (assignment probabilities, treatment history for carryover); require a pre-registered
  state-balance / residual-drift diagnostic; do not condition on any state variable affected by current or
  prior treatment unless the procedure handles that dependence.
- **"Discreteness benign" scoped down.** Holds only when the resampler reproduces the actual support AND
  probability mass. Still-open breaks: rare/imbalanced levels, state/time-drifting assignment probs,
  support-but-not-mass match, sparse state×assignment cells, clipping×nonlinearity, temporal dependence
  resampled as independent. Resample the full joint/conditional assignment LAW, not merely observed support.
- **MDE gate is per-estimand and per-cell.** Linear-injection MDE licenses negatives only for the linear
  estimand; gated/thresholded/localized/sign-changing effects need their own gated-effect MDE (see Part II,
  now characterized).
- **Null-calibration hardening.** Pre-register a binomial acceptance bound; enough null reps to separate
  0.05 from 0.10; stratify; do not average a failing cell away.
- **Additional required log-fields:** timestamp/order; eligibility indicator; assignment probability;
  controller + assignment-policy version/hash; missingness / logging-failure indicators; treatment history
  / washout boundaries. Plus multiplicity rules across KPIs, states, windows, and effect shapes.

---

## Part II — E5 decision-claim scope (occupancy-conditional)

The DRAFT's finalization fork offered (a) finalize under a linear estimand, (b) **characterize the
gated-effect MDE first, then finalize**, or (c) keep hardening. The **gated decision-MDE study
(2026-09-13 → 2026-09-15)** is exactly (b): a co-parent-gated harmful edge is E5's headline shape, so a
linear-only MDE was too narrow. That study is complete, re-scored, spot-verified bit-exact, and reviewed
by sol as SOUND-WITH-CAVEATS with all five cleanup caveats since closed (2026-09-16). It licenses the
following **scoped** decision claim.

### The claim E5 is scoped to make
> On the E2-ID shared-knob atom, a **complete causal fan-out** (discovering the full P0→{K0,K1,K2,K5}
> structure and acting on it) **prevents the locally-attractive-but-globally-harmful shared-knob action
> where that harm is active (in-gate)**. Its **net all-deployment benefit is occupancy-dependent** and is
> **washed out at realistic low occupancy** by the learned world-model's off-gate collateral. The binding
> constraint is the **learned world-model's off-gate generalization / sample efficiency — not discovery.**

E5 must **not** be registered to make the unconditional strong claim ("complete fan-out prevents the harm
across the whole deployment"); the evidence does not support it at realistic occupancy.

### The evidence (gated decision-MDE, E2-ID, 40 seeds, IPW-weighted S_eval)
- **Decision-MDE is NULL at every finite occupancy** q ∈ {2, 5, 10, 20 %}: prevention power = 0.00 at every
  injection strength δ; the off-gate control fails everywhere (`off_ctrl_ok = False`).
- **Decision-MDE first crosses only at q = 1.0 (every state in-gate):** barG_omit = **0.16334**, decision
  power **0.925 (37/40)** at δ = 0.4, rising to **1.0** at δ ≥ 0.8. There the discovered arm cuts regret to
  near-zero (0.028 → 0.002), matching the oracle handed the true structure — the causal decision logic is
  sound where the harm is active.
- **Discovery is not the bottleneck.** Edge power (M3 finds P0→K5) → 1.0 at every occupancy.
- **The oracle-structure arm ALSO fails at finite q** (e.g. q = 0.2, δ = 0.8: discovered 0.162 vs oracle
  0.143 — both far above the ~0.02 target). Handing the learner the *true* structure does not rescue it ⇒
  the failure is world-model generalization / sample efficiency, not M3 selection error.
- **Severity honesty (sol caveat 4).** At finite q the ~0.13 barG_omit is dominated by *baseline*
  learned-pipeline regret; the trap's intrinsic all-deployment severity is small (0.001–0.027). barG_omit
  is the metric label, not "trap severity."
- **Controls persisted.** Factor-removal (K5 removed from scoring + both planners' panels): all cells pass,
  max|diff| = 6.66e-03 < FR_MARGIN 0.02 (exactly 0 on normal cells; 6.66e-03 only on the bad_disc seed,
  where P0 leaks into K2 via a falsely-carried lagged-K5 warmup — a bounded illustration of the col-13
  nuisance). Nuisance diagnostics: 528/1600 cells carry a reward-relevant lagged parent, with per-cell
  selected lagged parents + query-support distances emitted.

### Caveats that ride with this claim (do not drop when citing)
- **E2-ID, not composed-E5 deployment.** Offline characterization on the abstract exp(−x²) benchmark with
  **no O-RAN grounding** (a known, user-flagged external-validity gap). A deployed system would be cheap
  online (offline discovery + world-model fit, per-decision planner lookup), not this sweep.
- **Single frozen off-gate S_eval sample (sol caveat 5).** Wilson CIs condition on one frozen off-gate
  sample shared across seeds/cells; they exclude evaluation-subsample uncertainty. E5 must use its own
  composed-system state law and a full-bank or independently replicated off-gate evaluation.
- **Fixed k-NN world-model.** The bottleneck is specific to the learned k-NN learner as configured; a
  different world-model class is the subject of the next thread (below).

---

## Decision & next threads

- **Decision (2026-09-16, user-gated GO):** finalize E5 at the **documentation/decision layer** under the
  occupancy-conditional scope above. The data-collection contract (Part I) is the trustworthy-discovery
  requirement; the decision-claim scope (Part II) is what E5 is licensed to assert. E5 the composed
  environment remains unbuilt and unfrozen.
- **Next thread (B) — the world-model bottleneck.** Improve the learned world-model's off-gate
  generalization / sample efficiency so net all-deployment benefit survives at realistic occupancy; only
  then would E5 be positioned to test the *strong* (unconditional) claim. This is the substantive lever and
  is consistent with the standing "world-model quality is the lever, not discovery" direction.
- **Deferred regardless:** correlated / non-ideal sensor noise; KPI→KPI instruments; O-RAN grounding of the
  benchmark mechanisms; a gated-effect MDE calibration for the discovery power/MDE gate (Part I §3) beyond
  the decision-MDE characterized here.

## Evidence pointers (working reports, gitignored per repo convention)
- `reports/2026-09-15-gated-decision-mde-result/` — the gated decision-MDE result (index.html + live figs).
- `scratchpad/design_adaptive_m3/` — pre-registrations (`PREDECLARE_phase4b_gated_mde_v3*.md`), harness
  (`gated_mde.py`, `validate_gated_mde.py`, `learner_knn.py`), scorer (`rescore_gated_mde.py`), companion
  post-pass (`emit_diagnostics_gated_mde.py`), outputs (`gated_mde_rescore.json`,
  `gated_mde_diagnostics.json`, `gated_mde_factor_removal.json`), sol note (`sol_result_review.md`).
- `reports/2026-09-09-design-adaptive-m3-calibration-stress/` — the discovery calibration-stress + Phase-4b
  hardening evidence behind Part I.
- Memory: `next-session-handoff`, `discovery-method-no-single-superset` (the world-model-quality lever).
