# E6 decision: methodologist's review (hostile reviewer stance)

## 1. Are the headroom conclusions valid?

**The oracles have defects, but none of them changes the no-go.**

Biases that under-estimate headroom:
- **Relative writes compound.** `cand_writes` multiplies the *current* `cio`/`ll_ratio` by 0.5 or 0. So when the oracle re-chooses its current plan every 15–20 s, CIO decays geometrically. It cannot set an absolute value.
- **The copy is taken mid-second.** `env.copy()` runs inside the arbiter call, so the requests being decided now are never applied or answered in the rollout. TS and ES act every 10 s and the oracle decides on a 10/15/20 s grid, so it often scores a mask without its first batch. The copied xApps' escalation state (`pending_target`) also diverges from the real run.
- **Weak search.** Only 8–14 random candidates and one coordinate-descent pass, each decision planned myopically. `ModOracle` does not always include the freeze candidate.

Biases that over-estimate headroom:
- **Unbudgeted writes.** Writes were unbudgeted (`1e9`): 296–583 used vs an allowance of 50 (Sol is right).
- **Clairvoyant selection on the same tape.** Picking the best rollout on the realised tape partly selects favourable noise, which no causal policy can exploit.

**Net:** a deployable policy's headroom is almost certainly *below* what was reported. The search deficits show diminishing returns (0.84 → 0.82 vs noarb). "Low ceiling" holds for deployable arbiters on the probed cells. It is not a bound on the true optimum.

**Is n=3–5 enough?** Enough for a no-go. Every seed of the per-region oracle is ≥ 0.87× freeze at medium load and ≥ 0.97× at high load. The gap to 0.75 is many times the between-seed spread. The bigger gap is coverage: only 2 of the 16 factorial cells were probed. Scope the claims to "E6 as built, M4 / mixed mobility".

**Missing: a controllability audit.** Freeze is the 3GPP-default configuration. Beating it by 25 % means the arbiter's own writes must beat the defaults as a SON optimiser. Nobody has measured how much SVR *any* configuration (e.g. a hindsight-tuned static per-cell one) can remove. If most violations are floor, 0.75 is unreachable.

## 2. How each option would fare with a hostile reviewer

- **Option 1 (Pareto reframe):** reads as post-hoc endpoint switching, and the dominance evidence comes from an over-budget clairvoyant oracle. It is acceptable only as a secondary, pre-declared claim, with a budgeted observed-information policy and multi-objective baselines.
- **Option 2 (redesign the scenario):** the only route to an unqualified SLA claim. Required safeguards:
  - Take the surge model from an external source (3GPP hotspot/event models or a public trace). Commit it by hash before any coordinated arm runs.
  - Tune the xApps on their solo objectives only, never in combination.
  - Keep a registry of every regime built, including failures. Keep original E6 as a control, and hold out some regimes (location, onset, load).
  - Give all arms the same write budget and the same information.
  - Faithfully reimplement published methods (paper parameters, author code where available) with an equal tuning budget. A "QACM-style veto" is not QACM.
- **Options 3 and 4:** fixing the oracle is necessary infrastructure, but open-ended CEM/JAX search raises privilege, not deployability. A new metric or task is legitimate only as a declared new hypothesis.

## 3. The bar

- **Point estimate ≤ 0.75 plus a one-sided test against 0.80** is coherent as "margin plus evidence of a non-trivial effect". It is not evidence that ρ ≤ 0.75, and the paper must say so, or test against 0.75 directly.
- **Freeze is the wrong ≤ 0.75 comparator.** It already fails goal retention by construction, and beating it by 25 % measures the arbiter as a SON optimiser, not as a conflict mitigator. Split the bar:
  - **(a)** ≤ 0.75 (tested vs 0.80) against every **goal-retaining** comparator, including faithful published methods and the best static subset;
  - **(b)** SVR non-inferior to freeze (≤ 1.00×, one-sided), shown on the same plot.
- **E6 as built fails even this relaxed bar:** the oracle reaches only 0.815× / 0.90× noarb (medium / high load). Relaxing the bar does not rescue option 1.
- **Retention guardrail:** replace the ratio of differences with absolute margins. Noarb−freeze energy gaps of 2–8 % make the ratio noise.

## 4. Decision

**Option 2, gated by a controllability audit and a corrected, budgeted oracle. Keep E6 as built as the pre-registered control and negative result.** Reject open-ended option 3. Keep option 1 as a secondary claim only.

**Plan (≤ 1 week, DEV seeds 11–15 only):**
- **Day 1:**
  - Fix the oracle: take the copy at a second boundary and replay the current batch, use absolute-target writes, always include freeze as a candidate, and enforce the 50-write budget.
  - Regression-run the fixed oracle on seeds 11–13 at medium load.
  - Run the controllability audit: a hindsight-tuned static per-cell configuration.
- **Day 2:** Commit the pre-registration: externally sourced surge regime, holdout variants, witness specification, and kill thresholds.
- **Days 3–5:** Paired DEV runs in the new regime and in original E6, over 4 cells × 2 loads.
  - Arms: solo xApps, noarb, freeze, priority, lock, best static subset, and the budgeted witness.
  - Stop before the witness runs if TS or MRO solo does not beat freeze on SVR by ≥ 10 %.
- **Days 6–7:** Freeze the regime or kill it.

**Kill the regime if any of these holds (per seed, not on means):**
1. Less than 35 % of SVR is controllable.
2. The budgeted witness fails to reach ≤ **0.65×** the best goal-retaining baseline and ≤ 1.00× freeze on every DEV seed, in both loads, with guardrails passing. The 10-point allowance below 0.75 is for learning loss: passive response models already rank candidates worse than accept-all.
3. Success depends on a surge parameter chosen after inspection.

After a kill: an explicitly multi-objective contribution, no SLA-edge wording.

## 5. Agreement with Sol

**Agree:** conditional option 2, the write budget, no training yet, freeze kept visible, the 0.75 vs 0.80 distinction, absolute retention margins, and a failed witness as a no-go rather than an impossibility bound.

**Disagree or add:**
1. Sol missed the compounding writes and the mid-second rollout gap. The witness must be fixed first, or it may kill a good regime unjustly.
2. A witness at exactly 0.75 leaves no room for learning loss. Require 0.65.
3. Use non-inferiority for freeze, and ≤ 0.75 only against goal-retaining comparators.
4. Add the controllability audit. It is the cheapest test of whether any regime can clear the bar.
