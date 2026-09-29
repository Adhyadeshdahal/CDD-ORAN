# Decision brief: what to do with E6 after the headroom results (2026-09-27)

## Project goal (user's bar)
Causal discovery → world model → planner/arbiter for xApp conflict mitigation in the O-RAN near-RT RIC. The user
wants a STRONG, clear end-task edge over every published solution (QACM, Sharma SHAP→DAG, Djidjev & Kaminski,
PACIFISTA, GNN conflict classifiers, two-tower), measured on preventing SLA violations. "Slight edge under specific
constraints" is not acceptable; the user will redesign our approach rather than accept parity. Draft decision rule
(`docs/benchmark/E6_METRIC.md`): strong edge = ≤ 0.75 × SVR of EVERY baseline (freeze included), with guardrails
(per-slice violation, RLF, severe incidents, energy, xApp goal retention) passing.

## E6 (built, committed 3695a5e + ccb5481)
`cdd_oran/envs/e6/`: 21 macro + 3 pico co-channel 2 GHz (38.901 gains), load-coupled SINR, A3/TTT handover,
RLF/T310, slice scheduler (LL dedicated PRBs + equal share), EARTH energy, E2SM-KPM-like reports with delay/drops,
282 knobs, arbiter API (accept/reject/modify/defer + own writes). xApps: MRO (RLF/too-late/ping-pong via CIO/hys/TTT),
TS (eMBB offload via CIO), ES (macro capacity-carrier shutdown), SLICE (LL dedicated PRB ratio, PRB-starvation aware).
SVR = violated UE-seconds per UE-hour (plain, all slices). 10-min scored DEV episodes, seeds 11–15.

Solo/subset sweep (seed 11): each xApp improves its own KPI alone (TS eMBB −2.6 %, ES energy −7.5 %, SLICE LL −5 %
/ −20 %; MRO ≈ inert), combinations conflict strongly (medium SVR: none 190.5, all four 199.1, MRO+ES+SLICE 301.4).
Priority and knob-lock arbitration == no arbitration (conflicts are cross-knob / implicit). Freeze (reject all) has
the best SVR but keeps no xApp goal (no energy saving, no LL gain).

## Headroom (perfect-knowledge lookahead oracles on the true future tape; DEV; Kaggle)
SVR ratio (mean over seeds; medium / high load):

| arm | vs noarb | vs freeze | notes |
|---|---|---|---|
| freeze | 0.88–0.91 / 0.92–0.93 | 1 | no xApp goal kept |
| veto-mask oracle, 30 s lookahead, 8 masks every 10 s | 0.975 / 0.96 | 1.08–1.11 / 1.03–1.05 | can backfire (≤1.24× noarb) |
| veto oracle, 90 s | 0.92 / 0.96 | 1.04 / 1.04 | horizon worth ~6 pts |
| modify + own-write oracle (14 random network-wide joint candidates, 60 s) | 0.84–0.89 / 0.91–0.93 | 0.95–1.01 / 0.99–1.00 | with energy/LL price: freeze-level SVR, noarb energy, better LL/RLF |
| per-region oracle (coordinate descent over 10 regions, 90 s), 12/12, pure SLA objective | 0.815 / 0.902 | 0.924 / 0.976 | energy 1.055 / 1.004; LL 0.98 / 1.07 |
| per-region oracle, energy/LL-priced objective | 0.847 / 0.909 | 0.960 / 0.984 | energy 1.005 / 0.985; LL 0.93 / 0.88; RLF 0.83 / 0.82 |

Files: `scratchpad/e6_dev/HEADROOM_RESULT.md`, `headroom*_all.jsonl`, `oracle.py`.
Earlier E2-CL diagnostics (`scratchpad/pact/endtask/D012_SUMMARY.md`): veto-only arbitration ≈ no headroom; bounded
joint nudges large headroom; passively learned response models rank candidates WORSE than accept-all (incl. true
parents) → the response model, not the graph, is the binding constraint.

## Reading
Headroom grows with oracle strength but with diminishing returns. Even perfect knowledge never gets below 0.87×
freeze; vs goal-keeping baselines (noarb/priority/lock/QACM-style veto) the ceiling is ~15–23 % at medium load and
~10 % at high load, before any learning loss. Cause: E6's xApps add little POSITIVE SLA value (TS ≈ −2 %, MRO ≈ 0);
coordination value is mostly harm avoidance, which freeze gets for free.

## Options on the table
1. **Reframe the strong edge as Pareto dominance on E6 as built:** only a coordinated arbiter reaches ≈ freeze SLA
   while retaining xApp goals; baselines keep goals and lose SLA, or keep SLA and lose goals. Plus SVR reduction vs
   goal-keeping baselines (ceiling ~15–23 %).
2. **Redesign the E6 scenario** so coordination has large positive SLA value (e.g. hotspot/event surges where TS/MRO
   offloading gives big SLA gains that ES/SLICE conflicts destroy; freeze becomes weak for a principled reason).
   Risk: accusations of benchmark tailoring; more build time.
3. **Search harder for the true ceiling** (CEM/MPC per cell, longer horizon; possibly a batched torch/JAX simulator for
   GPU rollouts) before deciding.
4. Something else (e.g. change the metric, change the task, change the contribution).

## Questions to answer
- Which option (or combination) and why? What would a hostile IEEE/Springer reviewer say about each?
- Is the ≤ 0.75 × every-baseline bar the right bar at all, given published baselines (QACM etc.) do not include freeze?
- What concrete next step (≤ 1 week), with a kill criterion?
