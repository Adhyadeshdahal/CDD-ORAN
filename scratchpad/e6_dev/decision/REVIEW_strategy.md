# E6 decision: strategy review (research-advisor perspective)

**Decision: option 4, with a strictly time-boxed option-2 probe inside it.** The paper to write in the next weeks is a
headroom and benchmark paper, not a new method: *what conflict mitigation can buy at all, by action class, and why
published veto-style arbiters are capped near "turn everything off"*. Stop chasing ≤0.75× every baseline on E6. The
evidence says that bar is unreachable there, even with perfect knowledge.

## Why the edge chase has negative expected value right now

1. **The ceiling is measured, not guessed.** Across three rounds of oracle strength (veto 30 s, then veto 90 s /
   modify+write, then per-region coordinate descent), the best perfect-knowledge result is 0.815× noarb and 0.924× freeze
   at medium load, and 0.90× / 0.976× at high load. Returns shrink with each round. Option 3 (CEM/MPC, a JAX simulator)
   would buy a few more points of *oracle* headroom. That is weeks of engineering aimed at a number no deployable method
   will reach.
2. **Those ceilings are optimistic.** As sol says, the modify/write oracles used 296–583 writes against a budget of about
   50 per run. And D1/D2 on E2-CL showed that every passively learned response model, including one given the true
   parents, ranks candidates *worse than accept-all*. So a learned arbiter starts below zero and has to climb to a ceiling
   that is already too low. It will not get there in weeks.
3. **The same pattern keeps repeating.** Discovery matched tuned SHAP, PACT was retired, and now E6 headroom is small.
   Each time the answer was "redesign and try again". A thesis-to-journal project with a submitted report and a paper
   under review needs a result that is finished and defensible, not another multi-month bet.

## Is ≤0.75× every baseline the right bar?

It is the wrong bar in two ways.
- **Freeze is not a deployable policy.** It turns off energy saving and LL protection, and operators deploy those
  xApps for exactly those reasons. The correct formulation is a constrained problem: minimise SVR subject to keeping
  the xApps' goals (an absolute energy margin and an LL margin). Freeze then belongs on the plot as a reference point,
  not in the denominator. Here I **disagree with sol**, who keeps freeze as the unqualified denominator. This is not
  moving the goalposts, provided the constrained formulation is stated as the claim up front.
- **Even that corrected bar fails on E6.** Against goal-retaining baselines the perfect-knowledge ceiling is 0.815
  (medium) and 0.90 (high), before any learning loss and before the write budget. So option 1 (Pareto reframe) cannot
  deliver a "strong SLA edge" either. Here I **agree with sol**: do not sell Pareto wording as the strong-SLA claim.

The honest conclusion to give the user: on E6 as built, no method, ours or anyone's, can cut SLA violations by 25%.
That is a finding worth publishing, not a failure to hide.

## Option 2: sound bet or sunk-cost trap?

It is a trap if it is open-ended, and a sound probe if it is capped at 2–3 days. Even if a surge scenario creates large
oracle headroom, D1 shows we still lack a response model that can use it. That is the binding constraint, and fixing it
(excitation-based local response estimation) is a months-long project. Option 2 therefore cannot produce the strong-edge
*method* paper within weeks. Its value is as a **second regime for the headroom paper**: "in regimes where the xApps
create positive SLA value, headroom by action class is X". That result is useful whichever way it comes out.

## Option 4 dominates: the contribution

**"Where is the headroom in xApp conflict mitigation? A privileged-oracle decomposition on a 3GPP-grounded closed-loop
RIC benchmark."** Planned content:
- E6 benchmark (38.901 channel, A3/TTT handover, RLF, slices, EARTH energy, KPM delays, budgeted arbiter API) released
  as open source.
- Headroom broken down by **action class** (veto / modify / own-write), horizon and knowledge, **under the write
  budget**, on two plants (E2-CL and E6).
- Key findings a hostile reviewer cannot wave away:
  - (a) Veto-class arbitration, the class QACM, PACIFISTA-style admission and SHAP→DAG gating operate in, buys 2–4%
    even with perfect foresight. Any policy in that class is approximately bounded by it (state that the oracle is
    myopic).
  - (b) Freeze, a baseline the literature omits, beats no-arbitration by 7–12% on SVR.
  - (c) Coordinated joint actions are where the headroom is.
  - (d) The graph is not the bottleneck; the response model is (D1: learners given the true parents are still worse than
    accept-all).
- MSCR and discovery appear as a component together with its parity result, reported honestly.

This also gives the next project a well-posed target: interventional response estimation for small joint actions.

Hostile reviewer: "negative result, simulator-specific." Answers: two plants, standards-grounded dynamics, released
code, and a precommitted protocol. Discover Telecommunications is a suitable venue for this.

## ≤1-week plan

- **Day 1–2 (Kaggle):** re-run the modify/write and per-region oracles with **write_budget enforced (50 writes per
  600 s)**, seeds 11–15, both loads. Add stronger goal-retaining baselines: best static subset, a QACM-style veto, and a
  SHAP→DAG gate. Adopt the absolute energy/LL retention margins now.
- **Day 2–4 (option-2 probe, capped):** before looking at any outcome, specify one surge regime from an external source
  (e.g. an event hotspot from a 3GPP traffic model) and fix its parameters and seed split. Run solo TS/MRO, noarb,
  freeze, best subset and the *budgeted* modify oracle.
- **Day 5–7:** freeze the headroom tables, write the paper outline and figures (a Pareto plot plus bars by action class),
  and update the E6_METRIC.md claim to the constrained formulation.

**Kill criteria (precommitted):**
- **Surge regime.** If the budgeted privileged oracle does not reach ≤0.75× the best goal-retaining baseline in both
  loads while passing the guardrails, the strong-edge pursuit ends for this journal cycle. The surge regime then goes
  into the paper only as a second headroom data point. If it passes, write it up as future work with a funded plan (the
  response-model learner). Do not start that learner inside this deadline.
- **Write budget.** If the modify/write headroom collapses to veto level under the budget, finding (c) becomes "joint
  actions help only when writes are cheap". That is still reportable.

## Agreement with sol

- **Agree:** do not train a world model yet; enforce the write budget; do not relabel Pareto as a strong-SLA claim;
  precommit the scenario before looking at outcomes; E6 as built stays as the control regime.
- **Disagree:**
  - (1) Freeze should not be the denominator of a goal-retaining claim. It is a reference point and a published-baseline
    gap.
  - (2) Sol makes option 2 the primary route. I make it a 3-day probe inside a paper that is already publishable. Given
    D1, even a passing surge oracle gives no method result within weeks.
  - (3) Sol treats "change the contribution" as the fallback. I think it is the highest-EV primary path.
