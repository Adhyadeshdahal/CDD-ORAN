# E6 decision: synthesis of sol + 3 independent reviewers (2026-09-27)

Inputs: `SOL_OPINION.md`, `REVIEW_oran.md`, `REVIEW_strategy.md`, `REVIEW_method.md`. User preference (2026-09-27):
ground the redesign in **standard 3GPP SON stress cases + O-RAN WG3-allowed conflict-mitigation actions**.

## Votes
| | choice | freeze in the bar | oracle kill threshold |
|---|---|---|---|
| sol | option 2, gated; E6 as control | keep ≤ 0.75 × freeze | ≤ 0.75 × freeze, both loads |
| O-RAN systems | option 2 (3GPP SON stress cases) + WG3 action/budget fix | ≤ 0.85 × freeze + tuned static | ≤ 0.65 × every goal-keeping baseline |
| strategist | option 4 (headroom-by-action-class paper) + capped surge probe | reference only | ≤ 0.75 × best goal-keeping baseline |
| methodologist | option 2, gated; E6 as control | "no worse than freeze" | ≤ 0.65 × best goal-keeping + ≤ 1.00 × freeze |

## Consensus (adopt)
1. **Keep E6 as built** as the pre-declared "calm network" control and a negative result: the only available
   headroom there is the xApps' own harm (noarb → freeze ≈ 9–12 %). Pareto wording is secondary, not the SLA claim.
2. **No world-model training yet; no open-ended oracle search** (option 3 rejected).
3. **Write-budget finding stands**: rounds 2–3 used 296–583 writes vs ~50 allowed → their Pareto result is not
   deployable evidence. The "no headroom on E6" conclusion survives (budgeting can only reduce headroom).
4. **Action class = O-RAN WG3 conflict mitigation**: accept / reject / modify / defer / lock / rollback to
   last-known-good, not arbitrary centralised SON writes. Budget = churn parity (the arbiter may not change knobs
   more often than noarb on the same tape) — O-RAN reviewer's proposal; replaces the arbitrary 50 writes.
5. **Fix oracle defects before any new verdict** (methodologist): absolute-value writes (no compounding CIO/ll
   decay), copy at a second boundary so the decided requests are replayed, always include freeze and accept-all as
   candidates, enforce the budget.
6. **Scenario from standards, frozen before outcomes**: 3GPP SON stress cases — event / moving-hotspot surge (MLB,
   TS 32.500 / 36.902 load-balancing use case) and mobility mis-tuning onset (MRO use case). Parameters cited and
   committed before any oracle run; holdout variants (timing/location) reserved.
7. **Controllability check first**: how much SVR can ANY configuration remove (hindsight-tuned static per-cell
   setting, and TS/TS+MRO alone vs freeze)? If most violations are an uncontrollable floor, no scenario can meet the bar.
8. **Kill threshold leaves room for learning loss**: 3 of 4 put the oracle bar at ≤ 0.65 × the best goal-keeping
   baseline (not 0.75), because learned response models ranked worse than accept-all on E2-CL.

## Disagreement → recommendation
**Freeze in the bar.** sol keeps ≤ 0.75 × freeze; others relax it. Recommendation (methodologist, strategist):
split the claim — **≤ 0.75 × every goal-keeping comparator** (faithful published methods, best static subset,
tuned static) **and SVR no worse than freeze**, with freeze always plotted. Rationale: beating freeze by 25 % tests
network optimisation, not conflict mitigation; freeze fails goal retention by construction. sol's stricter reading is
reported as a secondary, not dropped. **User to confirm.**

## One-week plan (gated)
- **Day 1:** fix oracle defects (item 5); WG3 action set + churn-parity budget in the env/oracle; write the scenario
  spec (item 6) with citations and commit it before running it.
- **Day 2 — gate A (controllability):** in the new scenarios, TS alone or TS+MRO must reach ≤ 0.85 × freeze SVR,
  and ≥ 35 % of SVR must be removable by a hindsight-tuned static configuration. Fail → stop.
- **Days 3–5 — gate B (budgeted WG3 oracle):** fixed per-region oracle, WG3 actions only, churn parity, paired DEV
  seeds, both scenarios + original E6.
- **Days 6–7 — decide.** Kill if the oracle does not reach ≤ 0.65 × the best goal-keeping baseline AND ≤ 1.00 ×
  freeze at both loads with guardrails passing, or if passing needs a parameter/seed chosen after inspection.
  On kill: pivot to the strategist's headroom-by-action-class paper (E2-CL + E6) plus a constrained/Pareto
  contribution; do not relabel it as a strong SLA edge.
