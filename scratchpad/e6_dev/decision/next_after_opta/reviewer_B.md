# Reviewer B (O-RAN / RAN systems, operator view): next step after option (a)

## 1. Recommendation
Option 2b, run as a disclosed follow-up with a new frozen protocol and fresh seeds. MapGate should only treat a guard-KPI edge as safe when PMRT has shown that the harm is small (a non-inferiority / power certificate). Not finding the edge is not enough. Keep the RLF guard at 1.10. Write up option (a) as NOT ELIGIBLE regardless of what 2b shows.

## 2. Strongest deployment argument
No operator deploys an xApp conflict manager that turns pico sleep into a 31% RLF increase. RLF is a red-line KPI (MRO, drop-call stats, regulatory reporting), and energy savings do not buy it back. The failure was not a wrong edge. PMRT made 0 wrong-sign and 0 placebo declarations. The referee read "not declared" as "no effect" on a rare-event KPI where the test had no power (sleep->nbr rlf z 0.2-1.3 against a true +.114). O-RAN WG3 conflict mitigation calls this an implicit conflict: two xApps coupled through a KPI that neither owns. Operator practice for implicit conflicts is fail-safe: hold an action whose side effect on a guarded KPI cannot be bounded. 2b turns that practice into a rule driven by the map, and it fixes the actual failure mode, which is how the referee acts under uncertainty rather than the map itself.

## 3. Against the main alternative (2a, an RLF-specific filter)
2a tunes the detector on the one KPI that just failed, after seeing the failure. A reviewer will call that a forking path. It also may not work: neighbour RLF is sparse, and a better-shaped filter does not create events that are not in the data. Even if it recovers these three edges, the next rare guard KPI (handover failure, ping-pong) brings the same problem back. 2b generalises to every guard KPI and needs no new statistic. 2d (more data) is an honest power fix, but it does not change the referee's logic, and it costs the most compute.

## 4. Concrete design
- For each (action a, guard KPI k, relation rel in {own, nbr, far}), PMRT reports the effect estimate and a one-sided 90% upper bound UB on harm (randomization CI by test inversion over the same matched filter).
- Harm margin delta_k is derived from the guard itself: the per-action effect that would push the pooled ratio to 1.05 at the DEV action rate. This leaves half the 1.10 budget for estimation error.
- Status per edge: DECLARED-HARM, CERTIFIED-SAFE (UB <= delta_k) or UNRESOLVED. MapGateV2 treats UNRESOLVED like DECLARED-HARM, with the same priority order (pv, v, rlf). Nothing else in MapGateV2 changes.
- Fix delta_k and the CI method on the DEV seeds (184200-239) only. Freeze, hash the artifact, then run fresh EVAL seeds (n 160).
- Guards: keep the point-estimate rule of ratio <= 1.10. Add one pre-registered secondary check that the RLF ratio upper 90% CI is <= 1.20, and add an absolute floor (an RLF increase of at most x per 1000 UE-s) so that low base rates cannot pass on ratios alone. Loosening 1.10 after a failure would look like moving the goalposts, and 1.10 is already lenient for RLF.
- Prediction to register: sleep near neighbours ends up UNRESOLVED and is deferred, so the arm keeps the non-sleep wins. The pre-registered target is R > never_sleep (+.221) with LB90 > 0 and eligible. If R only matches never_sleep, report that honestly. In that case the map adds nothing over a static rule.

## 5. Edge over baselines
- QACM, SHAP->DAG, Granger and two-tower produce a point map with no calibrated "unknown" state. Under confounding their nulls are invalid, so they cannot issue a safety certificate. That is the source of granger_by's 29 placebo edges. Run the same certificate rule on their maps as a control. The prediction is that the rule either defers almost everything (R near 0) or certifies false safety.
- PACIFISTA profiles each xApp in a sandbox, separately. It cannot see neighbour/far coupling that only appears with live interaction on the network. PMRT's certificates come from randomization on the real interaction.
- The GNN predictors give no coverage guarantee on rare KPIs.
- The claim becomes: the only map that carries its own uncertainty, so it is the only one an operator can gate on. This is a stronger and more defensible journal contribution than "highest R".
