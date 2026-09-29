# Morning brief — 2026-09-28 (overnight autonomous run)

## Bottom line
E6 as built cannot deliver the strong SLA edge, and every independent signal agrees:
1. **Gate A (controllability) — predeclared KILL** (510 jobs). TS / TS+MRO 1.00–1.03× freeze; only 2–10 % of SVR
   removable by any static config / subset (bar: 35 %). TS never improves its own KPI; MRO only cuts RLF in mistune.
2. **Physics diagnosis** (fresh seeds): surge violations are 74–93 % capacity-limited; ≤ 3.5 % relievable by CIO;
   perfect MRO repair ≤ 7.4 %. Better xApps would not pass the gate.
3. **Gate B (budgeted WG3 oracle, perfect knowledge)**: base 0.85–0.94× noarb, stress 0.87–0.96× noarb, 1.00–1.08× freeze,
   no better than the best fixed subset. Kill line was ≤ 0.65× best goal-keeping. (Stress: 17/24 jobs completed before the
   Lightning Studio was stopped at the waiter cap; the 7 unrun are mostly mistune/SLA-only.)
4. **Step-5 ranking test**: the learned policy-effect model (120 randomized episodes, frozen contract) ranks WG3
   candidates at ρ ≈ 0 vs the oracle; raw picks worse than accept-all; the calibrated gate abstains → safe, no gain.
5. **Literature scout**: handover/CIO conflicts show single-digit headroom in every paper (consistent with 1–3).
   Large, recoverable harm appears only in PRB/slice, transmit-power and cell-sleep conflicts (PACIFISTA ~−50 %
   throughput, xTRUCE 65–92 % rate-floor violations, QACM 39 %→3 % shortfall). New competitor: xTRUCE.

## What got built and committed (usable whatever you decide)
E6 WG3 env (lock/rollback/churn cap/two-phase step), 3GPP stress scenarios E6-scn-v1 (frozen), baselines incl.
QACM/CMF/PACIFISTA/Djidjev + SMO restore, the legacy-free decision stack `cdd_oran/decision/` (WG3 arbiter, true-sim
+ learned policy-effect world models, conformal gate, trace, randomized collectors v1–v3, template MSCR, probe
campaign + design-based CRT `mscr-crt-v1`, ranking eval), lazy legacy imports. Contracts: E6_METRIC.md,
E6_SCENARIO_CONTRACT.json, E6_COLLECTION_CONTRACT.md. Not frozen (sol blockers listed): v2 xApps, probe/3.

## Decision needed (sol's ranked paths, `SOL_STRATEGY.md`)
1. **New, externally grounded conflict-rich benchmark** (ground it on PRB/slice/power/sleep conflicts, not handover)
   + deployable WG3 controller. Highest ceiling, highest evidence burden. Must first show: competent xApps, avoidable
   co-deployment loss, and a *budgeted* oracle margin — then a learned controller that actually ranks (step 5 failed
   here, so the model side is also unproven).
2. **Screen existing simulators** (scout shortlist: ns-O-RAN/ns-o-ran-gym TS+ES, NIST ns3-oran with its built-in
   conflict-mitigation module, mobile-env PRB/power as a fast control) with Gate-A semantics; keep only settings with
   ≥ 15 % co-deployment loss and ≥ half recoverable within budget. Cheapest way to find out if the edge exists anywhere.
3. v2 xApps rerun on E6-scn-v1 — clean but predicted to fail.
4. **Pivot**: headroom-by-action-class + constrained/Pareto + "why conflict mitigation has little SLA headroom when
   violations are capacity floors" paper, with the E6 negative results and the learned-model ranking failure as
   findings. Defensible, but it is not the strong edge — it should be called what it is.
My recommendation: **run path 2 as a 1–2 day screen first** (it decides whether path 1 is even worth it), and in
parallel attack the model side (ρ ≈ 0 ranking) since any path needs a controller that can rank actions.

Files: `.tmp/PLAN.md` (full log), `scratchpad/e6_dev/decision/` (all reviews), `scratchpad/decision_stack/`.
