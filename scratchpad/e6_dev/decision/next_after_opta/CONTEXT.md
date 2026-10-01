# Decision context: what to do after option (a) (2026-10-01)

Goal (user): a strong, clear end-task edge over published xApp-conflict solutions (QACM, Sharma SHAP->DAG,
Djidjev & Kaminski, PACIFISTA, GNN, two-tower), measured on preventing SLA violations in an O-RAN near-RT RIC
(E6-P simulator). Discovery method = PMRT (Predictable Matched-Filter Randomization Test). Venue: Springer
Discover Telecommunications journal paper (in progress), conference paper NaNA 2026 (submitted).

Results (all pre-registered, frozen protocols, fresh seeds):
1. v4 discovery (randomized logs): PARTIAL. K0/K0n/G/K1/P1/P3/S PASS; P2 fails only vs granger_by at n 60 (2/10 wins;
   4/5 at n 120); granger_by declares 13 placebo edges vs PMRT 0. PMRT pooled indirect F1 .95, sign acc 1.00.
2. option (a) (confounded incumbent logs -> causal map -> MapGateV2 referee), discovery: DISC-PASS. PMRT map 22 edges,
   18 TRUE right sign, 0 wrong, 0 placebo edges. granger_by 47 (16 right, 6 wrong, 29 placebo), granger 32/12/6/19,
   corr 10/6/2/8, two_tower 8/5/0/22, SHAP 5/1/0/2.
3. option (a) EVAL (160 seeds): verdict NOT ELIGIBLE.
   R vs accept-all (higher better): MG:PMRT +.470 [.346,.594] (highest of all arms), MG:GT +.244 (eligible),
   never_sleep +.221 (eligible static), blanket2 +.227 (inelig.), corr@dev map +.033, qacm +.011,
   granger/granger_by/two_tower/shap@dev maps -9.8 .. -11.5. PMRT referee cuts SLA viol .86x, nonprot eMBB .86x,
   LL .94x, but RLF ratio 1.31 [1.10,1.56] > 1.10 guard -> ineligible -> R* 0 -> D1/D2 fail by rule.
   PMRT - never_sleep = +.249 [.139,.362].
4. Cause: PMRT map misses all 3 neighbour/far RLF edges of the GT map. GT effects: sleep->own rlf -.137, sleep->nbr rlf
   +.114 [.059,.167], sleep->far rlf +.119, ptx->nbr rlf -.061. PMRT z: sleep own -5.0 (found), sleep nbr .2-1.3,
   sleep far ~.7, ptx nbr -2.2 plain / -.7 loadsp. The loadsp matched filter (shaped for load spillover) adds nothing
   for neighbour RLF; nbr RLF is a rare-event, noisy KPI.

Files (read if needed): scratchpad/e6_dev/decision/MORNING_BRIEF_4.md, conf_verdict.json, conf_map_quality.txt,
docs/benchmark/E6P_CONFOUNDED_PROTOCOL.md (frozen), docs/benchmark/METHOD_NAMES.md, .tmp/PLAN.md (tail).

Candidate options:
 1. Report as is (honest NOT ELIGIBLE + strong descriptive results) and move to writing.
 2. A disclosed follow-up study (new protocol, fresh seeds) attacking RLF power: (2a) RLF-specific spillover filter /
    rare-event statistic in PMRT; (2b) conservative referee that treats an untested guard-KPI edge as possible harm
    (e.g. MapGate defers actions whose guard-KPI edge is untested or under-powered); (2c) both; (2d) more data.
 3. Something else you think is better.
