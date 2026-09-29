# ENV_SCOUT: existing environments with avoidable, coordination-recoverable conflict loss (path 2)

2026-09-28. Desk research only, with no runs. [V] = I checked the number in the paper text or on the arXiv page. [U] = unverified (abstract or secondary source only, or I could not read the full text).

## Bottom line

No existing public environment reports a large *avoidable* SLA loss that a same-rights coordinator then *recovers*, measured closed-loop. The published harm splits sharply by knob class:

- **Handover/CIO conflicts (MLB–MRO, TS–MRO): single-digit headroom everywhere.** Adamczyk & Kliks (arXiv:2305.13464, custom sim) [V]: no-CM vs. best priority gives RLF 204→204, ping-pong 3371→3273 (−2.9 %), call blocks 547→542, satisfaction 63.27→65.50 %. The CMF paper (ComMag 2023, 19-BS custom sim) [V] reports HO −7 % and CB −7 %, while every other KPI worsens. Wadud et al. (INFOCOM, arXiv:2411.03326, MATLAB, MRO–ES on TXP) [V]: QACM gives −17 % connection losses and −2.2 % HOs vs. no-coordination. This matches E6 Gate A (TS/freeze 0.97–1.02). The failure is the knob class, not only our physics.
- **Resource, power and cell-activation conflicts: large harm.** PACIFISTA (arXiv:2405.04395, Colosseum) [V]: TM+ES co-deployed gives about **−50 % throughput** vs. TM alone, −16 % with similar-goal xApps and "up to 30 %" with conflicting ones. Its slicing xApp pairs cut eMBB PRBs by 26 % and eMBB throughput by 31.8 %. The Scheduler paper (arXiv:2504.06867, Python, 4 O-RU/16 UE, power + RBG xApps) [V]: conflict degradation is **16 %** at high load vs. 5 % at low load. QACM (arXiv:2405.07324, synthetic Gaussian tables) [V]: NSWF leaves CCO throughput 39 % below threshold, and QACM cuts that to 3 %. xTRUCE (arXiv:2608.28532, Python + OAI/FlexRIC OTA) [V, abstract/HTML]: a clipping arbiter violates SLA rate floors in 65–92 % of epochs, xTRUCE in about 0 %. COMIX (arXiv:2501.14619) [U]: "up to 60 % energy savings" vs. conflict-unaware.

**Implication:** a path-2 environment is only worth screening if it exposes PRB/slice, TX-power or cell-sleep knobs to at least two xApps. Another CIO/HO environment will very likely reproduce the E6 kill.

## Ranked shortlist

| # | Candidate | Why | Main risk |
|---|---|---|---|
| 1 | **ns-O-RAN + ns-o-ran-gym (TS + ES envs)** | Real E2SM-KPM/RC and 3GPP-grade stack. It ships Traffic Steering and Energy Saving (cell on/off) gym envs on one simulator, and sleep-vs-steering is a high-harm class. Sharma et al. name it as their next target, and ZODIAC uses NS-O-RAN-FlexRIC for multi-xApp conflicts. | No published harm or recovery numbers, so we must screen. Heavy Linux build, and slow at scale [U]. |
| 2 | **NIST ns3-oran (LTE) with its built-in Conflict Mitigation API** | The CM module filters every xApp command, which maps almost 1:1 to our arbiter (accept/reject/modify). The QACM authors moved here: GenC (arXiv:2602.19758) uses ns3-oran with a Dublin OpenCellID topology and an ES–MRO conflict, and arXiv:2509.10978 adds RU energy modelling. This is the best comparability to QACM's current line. | LTE only. Stock actions are handover commands, and ES–MRO on TXP produced only 2–17 % in their MATLAB version. Harm numbers are unpublished [U]. |
| 3 | **Colosseum / OpenRAN Gym (PACIFISTA setting: slicing and scheduling xApps, TM vs ES)** | Largest verified harm (about 50 %). PACIFISTA and GRACE were evaluated natively here, and the DRL slicing xApps are public in OpenRAN Gym. There is a public offline dataset of about 8 GB of KPMs (89 h, up to 7 BS / 42 UE, arXiv:2309.05621). | Closed-loop runs need Colosseum access (application-gated; availability for us [U]). The offline data is not a counterfactual environment. PACIFISTA mitigates at deployment time, so it is not a per-request arbiter. |
| 4 | **Lightweight Python sims with PRB/power knobs: mobile-env; a reimplemented Scheduler-paper sim; the xTRUCE sim** | Laptop-fast, pip-installable (mobile-env: Gymnasium). ZODIAC ran LB vs QoE association, power control vs association, and an implicit power-budget conflict on mobile-env (3 BS / 5 UE). The Scheduler and xTRUCE settings have reported 16 % and 65–92 % harm. | Low fidelity, which is the same "designed to order" criticism E6 got. xTRUCE code is only "upon publication". The Scheduler code is not found [U]. |
| 5 | **MATLAB 5G Toolbox (QACM validation, Wadud INFOCOM, Sharma)** | Same tool as three comparators. | Needs a licence, and there is no MATLAB on Kaggle or Lightning. There is no shipped SON-conflict example, and the comparators' scenario code is not public [U]. Reported harm is 2–17 %. |
| 6 | **Static datasets: QACM conflict tables (github.com/dewanwadud1/QACM), GenC metadata, the two-tower / GRAPHICA / Djidjev–Kaminski synthetic data** | Direct reuse for comparing *detection* against two-tower, GRAPHICA and Djidjev–Kaminski. | Not closed-loop, with no SLA dynamics. Two-tower uses 10 000 Gaussian samples (4 agents, 7 parameters, 4 KPIs) from the Banerjee conflict model [V]. Djidjev–Kaminski arXiv:2606.06663 use synthetic Boolean streams (5 parameters, 4 KPIs) [V], and arXiv:2606.06459 a synthetic closed-loop generator [V]. This cannot show an SLA edge. |

Not suitable: **Sionna** (PHY/system-level, no RIC or conflict model), **ns3-gym** (generic bridge), **O-RAN SC xApps** (TS, QP, AD, KPIMON: a platform that needs an E2 simulator underneath), and the **OTIC testbed** (arXiv:2503.11566: srsRAN + OSC RIC, −78 % DL throughput *variability* [V]; hardware only).

## Per-candidate assessment

**1. ns-O-RAN / ns-o-ran-gym** (github.com/wineslab/ns-o-ran-ns3-mmwave, /ns-o-ran-gym, GPL-3.0 [V]; `pip install nsoran` [V])

- (a) TS: per-UE handover/steering via E2SM-RC. ES: cell on/off. Both run on LTE–mmWave dual-connectivity scenarios. The TS paper used 8 BS and up to 126 UEs and reports +50 % throughput from xApp HO over heuristics [V abstract].
- (b) Conflict harm is not published. We must measure the TS+ES co-deployment against each xApp alone.
- (c) No conflict comparator was evaluated here. ZODIAC (arXiv:2604.19610) used NS-O-RAN-FlexRIC (1 eNB, 5 gNB, 10 UE) but reports only detection metrics [V].
- (d) Open source. It needs Linux or Docker, and runs CPU-only on Kaggle or Lightning. Wall-clock time at 100+ UEs is probably slower than real time [U].
- (e) Moderate effort. The gym env already exposes obs/action per step, and we would add a shim that intercepts both xApps' RC requests before they reach ns-3. About 1–2 weeks.
- (f) Risks: ES may be a trivially dominant fix (just "don't sleep a loaded cell"), which is headroom without a causal edge. Build fragility.

**2. NIST ns3-oran** (github.com/usnistgov/ns3-oran, ns-3 ≥ 3.42, SQLite RIC, ONNX/PyTorch LMs [V])

- (a) LTE handover LMs, plus energy via 2509.10978. The GenC ES–MRO scenario is on a Dublin topology [V abstract].
- (b) Not published. The MATLAB predecessor gave 17 % RLF / 2.2 % HO.
- (c) Only the QACM authors' line, and only as classification (GenC: 3.2× faster than rule-based) [V abstract].
- (d) Lightest ns-3 option: LTE with no mmWave. Fits a 16 GB laptop via WSL.
- (e) Lowest effort. We would implement our arbiter as the CM module (C++ shim, or a Python bridge through the ONNX/PyTorch hooks). About 1 week.
- (f) Risk of HO-class low headroom (see the bottom line). The GenC repo is thin: 3 commits, only `simulation_metadata` [V].

**3. Colosseum / OpenRAN Gym**

- (a) Slicing PRB and scheduling xApps (DRL and rule-based), TM and ES; Rome scenario, 6 UEs [V].
- (b) −50 % (TM+ES), −16 %, and up to −30 % [V].
- (c) PACIFISTA native.
- (d) Access-gated; I could not confirm current status [U].
- (e) High effort: remote SDR testbed.
- (f) Access, and the fact that the harm is xApp-pair incompatibility that deployment-time exclusion (PACIFISTA) already fixes. A per-request arbiter must beat "don't co-deploy".

**4. Python sims**

- (a) Association, power, RBG and cell activation.
- (b) 16 % (Scheduler) and 65–92 % floor violations for naive arbitration (xTRUCE).
- (c) xTRUCE compares only its own Direct/Clipping/Flat variants [V]. Nobody evaluated QACM or PACIFISTA there.
- (d) Trivial.
- (e) Days.
- (f) Fidelity, and xTRUCE is a new direct competitor (a provably safe arbiter with priority certificates) that we should cite and beat.

## Path-1 grounding (externally justified new scenario)

The strongest literature-backed scenario is a **slice-PRB / TX-power / cell-sleep conflict**, not MLB–MRO:

- TM+ES gives about 50 % throughput loss (PACIFISTA).
- Power+RBG conflicts give 16 % at high load (Scheduler).
- Naive arbitration violates rate floors in 65–92 % of epochs (xTRUCE).
- A 39 % threshold shortfall is reduced to 3 % by QACM.

The classical SON MLB–MRO literature (Lobinger et al., VTC 2011; Difference-Based Joint MRO/MLB, IEEE 6240111; policy-based MLB/MRO, IEEE 6912543) describes the conflict qualitatively. I could not read their tables (paywalled) [U]. The O-RAN reproductions show at most about 7 %. Do not ground a new benchmark on MLB–MRO magnitudes.

## Recommended next step (cheap screen, needs your go)

Screen before any integration, using E6 Gate-A semantics: xApp-alone vs. co-deployed vs. an oracle same-rights arbiter, and compute the avoidable fraction.

- (i) ns-o-ran-gym TS+ES on its default scenario.
- (ii) ns3-oran with ES+MRO.
- (iii) A mobile-env PRB/power pair as the fast control.

Proceed only where the co-deployment loss is ≥ 15 % and at least half of it is recoverable under budget.

## Sources

- arXiv: 2305.13464, 2305.07117, 2411.03326, 2405.07324, 2405.04395, 2504.06867, 2608.28532, 2607.22857, 2501.14619, 2602.19758, 2604.19610, 2606.06663, 2606.06459, 2601.13213, 2510.13031, 2503.11566, 2305.06906, 2309.05621, 2509.10978
- github.com/wineslab/ns-o-ran-gym
- github.com/usnistgov/ns3-oran
- github.com/dewanwadud1/QACM
- github.com/dewanwadud1/GenC
