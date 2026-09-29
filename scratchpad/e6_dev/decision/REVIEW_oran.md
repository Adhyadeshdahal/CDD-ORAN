# E6 decision review: O-RAN / RAN-systems perspective

**Decision: Option 2, redesign the scenario, using standard 3GPP SON stress cases rather than a new invention. Combine it with an action-class and budget fix (Option 4). Keep the current E6 as the "calm network" control and report Pareto results there as a secondary result, never as the headline. Reject Option 3 for now.**

## 1. Is "freeze beats everything" unrealistic?

No. It is what an experienced operator would expect for this scenario. The finding is realistic, and it shows the scenario is missing the conditions under which SON earns its keep.

- The plant is a regular 21-macro hexagonal grid starting at CIO=0 with sensible Hys/TTT. That static configuration is already close to optimal for the average load. MRO has nothing to repair, so it is inert. That matches field experience: MRO pays off after mis-tuned rollouts, new sites and topology changes, and does little on a tuned grid.
- ES is supposed to trade SLA for energy. SLICE takes PRBs away from eMBB/BE, and SVR counts all slices. Two of the four xApps are net SLA-negative by design. TS is the only SLA-positive one, and with 3 picos, a gentle log-OU load and a compressed 0.4→1.0 ramp, it finds little to offload (−2.6 %).
- Operators often switch SON functions off ("SON fatigue") for exactly this reason. Freeze winning on a calm, tuned network is therefore a credible result, and the paper should report it.

The conclusion follows directly: the available headroom equals the harm the xApps cause (noarb→freeze ≈ 9–12 %). No arbiter can cut SVR by 25 % against a baseline that is only 10 % worse than doing nothing. The ≤0.75× bar cannot be met in this regime even in principle.

## 2. Is a surge/hotspot redesign principled or tailoring?

It is principled if the stress cases come from the standard SON use-case catalogue (TS 32.500/32.522, TR 36.902) and are fixed before any arbiter is run:
- **(a) Moving hotspot / event surge.** This is the canonical MLB use case. Pico and macro CRE offloading has well-documented SLA value in HetNets (eICIC/CRE studies).
- **(b) Mobility mis-tuning onset.** Examples: a TTT/CIO mis-set on a cluster after a "software upgrade", or a new pico going live. This is the canonical MRO case.
- **(c) Cell outage.** A neighbour-compensation case, if time allows.

It becomes tailoring if the parameters (surge amplitude, location, timing) are chosen after looking at oracle results. Freeze the parameters from cited ranges and randomise onset and location per seed. A hostile reviewer will also ask for a **tuned-static baseline**, meaning a configuration optimised offline for the scenario's average load. Without it, freeze is weak only because CIO=0 is bad in a hotspot. Include it, and include the trivial subset policy "run only TS+MRO, block ES/SLICE". Our method has to beat both.

## 3. Is the write-budget point fatal?

It is fatal to the *positive* Pareto claim from rounds 2–3, not to the *negative* conclusion. The own-writes were network-wide "macros": one decision changed many knobs, giving 296–583 writes against a budget of about 50. Enforcing the budget can only shrink the headroom, so the "no 25 % headroom on calm E6" finding stands.

The budget as defined is also arbitrary and asymmetric. The xApps churn hundreds of knobs without limit, while the arbiter gets 50 writes across 282 knobs. I propose **churn parity**: the arbiter's total knob changes (accepted + modified + own writes, weighted by step size) must be ≤ the churn of noarb on the same tape. That rule is defensible to a reviewer ("we never change the network more than the xApps would have"), and every re-run should use it.

## 4. Legitimate action classes for a near-RT RIC conflict-mitigation component

- **Clearly legitimate (WG3):**
  - Pre-action resolution of requests: ACCEPT, REJECT (NACK), MODIFY to a value between the current and proposed setting, DEFER.
  - Priority/lease locks.
  - Post-action detection with ROLLBACK to the last-known-good value.
  - Guidance or policy toward the xApps (A1-like).
- **Grey zone:** arbitrary own writes to values no xApp requested. That makes the component a centralised SON optimiser, and reviewers will then demand centralised SON/RL optimiser baselines.

So restrict own writes to rollback/restore toward last-known-good or the default configuration, plus bounded probes. The first budgeted witness should use only these classes.

## 5. The bar

≤0.75× is not a universal bar. It depends on the regime, because the gain is capped by the harm conflicts actually cause. QACM, Adamczyk and similar papers report against no mitigation or against their own variants, never against freeze. I agree with sol that freeze must appear on every plot. The headline should be ≤0.75× every goal-retaining baseline in the stress regimes, with freeze and tuned-static shown and required to be beaten as well (at least ≤0.85×). An energy-saving floor ("ES must deliver ≥X % kWh saving") is a realistic operator framing. It should be a pre-declared secondary endpoint, not a relabelled headline.

## 6. One-week plan

- **Day 1: Specify and freeze the regimes.** Freeze S1 (event surge, from TR 36.902 MLB) and S2 (mis-tuning onset). Write down the parameter ranges and their citations, and the seed split. Implement churn parity and the restricted action class. Commit the frozen spec before running anything.
- **Day 2: Gate A (positive value exists).** Run on DEV, both loads: solo TS and TS+MRO in S1/S2 must reach SVR ≤0.85× freeze. If they fail, kill: the xApps cannot create value, so no arbiter can.
- **Days 3–5: Gate B (budgeted feasibility witness).** Run the per-region lookahead oracle with churn parity and the WG3 action classes only (accept/reject/modify/defer/rollback), on the same tapes. Compare against noarb, freeze, tuned-static, the TS+MRO subset, priority/lock and a QACM-style veto. Report SVR, per-slice violations, RLF, severe incidents, kWh and goal retention.
- **Days 6–7: Decide.**
  - **Kill** if the witness does not reach **≤0.65× every goal-retaining baseline and ≤0.85× freeze and tuned-static in both loads**, with guardrails passing. The 0.65 threshold leaves margin for learning loss; D012 showed passively learned response models ranking worse than accept-all.
  - **Also kill** if passing requires a parameter or seed chosen after inspection.
  - On a kill: publish calm-E6 plus the stress regimes as a boundary study ("coordination value scales with conflict harm"), with the constrained/Pareto result as the contribution.
  - On a pass: build the observed-information arbiter and test it on untouched seeds.

## 7. Agreement with sol

**Agree:**
- Option 2 as a conditional bet, with E6 kept as a control.
- Don't train yet.
- The write budget invalidates the Pareto claim.
- Open-ended CEM/JAX search is a distraction.
- Freeze stays visible.
- No post-hoc parameter shopping.

**Disagree or extend:**
- The 50-write budget is itself unprincipled. Replace it with churn parity rather than enforcing it as is.
- Restrict own writes to the WG3 classes, or the method becomes a centralised optimiser with a larger required baseline set.
- Use standard SON stress cases and a tuned-static baseline to answer the tailoring charge head-on.
- Sol's kill threshold of 0.75× for the witness is too lenient. A privileged oracle at 0.75× leaves nothing for learning loss, so require ≤0.65×.
- An energy-floor constrained endpoint is legitimate operator practice when pre-declared as secondary. It is not goalpost-moving.
