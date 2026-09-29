# Path-2 scout 1: ns-O-RAN / ns-o-ran-gym (2026-09-28, desk research) — VERDICT MARGINAL (leaning DEAD for strong edge)

Repos: wineslab/ns-o-ran-ns3-mmwave (GPL-2.0, last push 2025-03-18, dormant); wineslab/ns-o-ran-gym (pip nsoran,
GPL-3.0, last 2025-04-04). Build: Ubuntu 20.04, waf/ns-3.36, e2sim, contrib/oran-interface; no prebuilt sim+gym image.
Gym control = CSV file + POSIX semaphores (ns_env.py posix_ipc, action_controller.py); LteEnbNetDevice::ReadControlFile
(lte-enb-net-device.cc L128-375) every 100 ms, dispatch by FILE NAME, mutually exclusive branches: TS
ts_actions_for_ns3.csv (PerformHandoverToTargetCell), ES es_actions_for_ns3.csv (SetSecondaryCellHandoverAllowedStatus
+ EvictUsersFromSecondaryCell), QoS qos_actions.csv (perPckToLTE split, no gym env). TS env scenario-one; ES env
scenario-three (7-bit mask, energy = TB-count proxy). KPIs via SQLite KPM; no native SLA. Scale hardcoded 7 NR gNB + 1
LTE eNB (scenario-one.cc L480). Runtime: minutes per 2 s sim [unit V]; est 30-150x slower than real time at ~60 UEs
[U]; no state fork -> counterfactuals replay from t=0.
Co-deployment: not out of box (one control file/run; TS and ES in different scenarios). Glue ~50 lines C++ + Python
arbiter shim.
CRITICAL [V]: ES "off" = HO bar + eviction only; TurnOff() flag never read by PHY/MAC; PerformHandoverToTargetCell
(lte-enb-rrc.cc ~L4478-4532) ignores m_allowHandoverTo -> TS can steer UEs back onto "off" cell with FULL service ->
conflict = lost energy + churn, NOT SLA harm. SLA harm needs a physical-off patch (our modification, reviewer target),
and then mostly recovered by one rule "reject HO to off cells".
Published: no ns-O-RAN conflict harm numbers (ZODIAC 2604.19610 detection only; QACM ES-MRO 2411.03326 MATLAB 4 gNB
100 UE +17% connection loss, +2.2% HOs; QACM moved to NIST ns3-oran 2602.19758). Lacava DRL-ES 2410.14021: ES alone
costs 23-52% throughput for 0.7-24% energy saving.
Feasibility: WSL2 only on Windows; Lightning best; Kaggle via saved build tree; build 30-60 min [U].
Screen (if ever): energy-matched metric; scenario-three 7 gNB, 5 & 9 UE/gNB, traffic 3 & 0, 10 s; semantics A shipped /
B physical-off; arms freeze, TS, ES-static, ES-dynamic, TS+ES, best static subset, rule oracle, k-grid; 320 runs 30-130
CPU-h [U]; pilot 1 seed 1-3 CPU-h. Kill: K1 >30 min wall per 10 s episode; K2 sem A co-deploy SLA >= ES-alone; K3 sem B
>=90% recovered by one-line rule; K4 energy-matched loss <15% both.
