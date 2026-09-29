# Path-2 scout 2: NIST ns3-oran, Colosseum/OpenRAN Gym, public datasets (2026-09-28, desk research)

Key finding: every large recoverable harm in the literature is a DIRECT same-knob conflict (e.g. ORIGAMI IMDEA EuCNC'26
slice-PRB: hi-prio slice 148->28.7 Mbps (-81%), priority rule restores 130 (~85% recovered)) -> passes Gate A, no edge
over QACM/priority. Proposed 3rd Gate-A check: per-request recoverable minus best static-subset/priority recoverable
>= 10 pp. Margin only in INDIRECT state-dependent conflicts (ES power/sleep vs HO/MLB, slicing vs power).

A) NIST ns3-oran (v1.0-1.3, last push 2026-09-05, ns-3>=3.42, NIST/public-domain, LTE only; NR fork stale 2024-12).
E2 terminators -> near-RT RIC + SQLite repo; default LM + any number of extra LMs per cycle (multi-LM native); all
commands pass OranCmm::Filter. ONLY command type: OranCommandLte2LteHandover. LMs: distance/RSRP/ONNX/Torch HO, noop. No
power/sleep/slice LM. CMMs: Noop, Handover (dedupe), SingleCommandPerNode (default LM wins else FCFS) = trivial ->
native baseline, not "already solved". Our arbiter = OranCmm subclass; defer/lock/rollback not native.
Energy ext dewanwadud1/RU-Energy-Model-ns3-oran: power->energy model only, no command/LM. GenC (2602.19758, QACM authors
ES-MRO on ns3-oran): classification speed only, harm numbers [U].
Needs OranCommandLteTxPower (LteEnbPhy::SetTxPower) + ES LM + coverage LM. Indirect pair ES-TXP x RSRP-HO -> coverage
holes/RLF at low-mid load (not capacity floor). Needs energy-capped reference (SLA s.t. E <= E_ES-alone + delta).
Build ~20-30 min on 4 cores [U]; runtime unmeasured [U]. Protocol: 7-cell hex, loads low/mid/high, 300 s, 5 seeds; P1
ES-TXP x RSRP-HO, P2 ES-TXP x CCO-TXP (direct control); arms freeze/each alone/CmmNoop/CmmSingleCommandPerNode/best
static/per-request oracle; ~210 runs 7-18 CPU-h; ~1 week C++. VERDICT: MARGINAL as shipped; PROMISING only if TX-power
ext shows indirect ES x HO harm.

B) Colosseum: team Google form, needs active wireless grant [V snippet], non-US eligibility [U]; remote SDR -> DEAD near
term. PACIFISTA (2405.04395): SCOPE/ColO-RAN, 1 BS 6 UEs 50 PRB 3 slices, Gaussian + PPO xApps, -16% similar goals, up
to -30% conflicting, TM+ES ~-50%; no data/code. xTRUCE (2608.28532): custom Python sim, 4 cells 20 UEs 500 m ISD, 12 RBs
x 360 kHz, UMa + 8 dB shadowing, AR(1) Rayleigh rho 0.349, exp arrivals 6 Mbps/UE, 1 s epochs; knobs RB share/power/
sleep/assoc; OTA OAI+FlexRIC; Direct 99% epochs violate, Clipping 65-92% operational-limit violations; code unreleased.
ColO-RAN dataset (wineslab, GPL-3.0, 1.4 GB): 7 BS 42 static UEs 3 slices, factorial RBG split x scheduler -> zero-cost
lookup-surface screen (mostly zero-sum PRB; recoverable part = over-provisioned light slices, priority may capture);
<1 CPU-h; MARGINAL, cannot carry a paper.

C) QACM tables synthetic; GenC metadata only; no public conflict simulator on GitHub (only a mirror of our repo
Adhyadeshdahal/CDD-ORAN, williamli-15/a1gent-ns3-oran). Reimplementing the xTRUCE sim from its full spec = fastest
closed-loop path (<=2 CPU-h: pairs ES-sleep x assoc/MLB indirect, power-up x ES direct ctrl, RB eMBB x URLLC direct
ctrl, power x RB indirect; kill if indirect <15% or per-request minus static <10 pp). PROMISING as fast screen, MARGINAL
as headline (designed-to-order criticism).
Unverified: ns3-oran runtime; GenC harm numbers; Colosseum eligibility; PACIFISTA traces; 2504.06867/xTRUCE code.
Sources: github usnistgov/ns3-oran (model/oran-cmm*.cc), dewanwadud1/{RU-Energy-Model-ns3-oran,GenC,QACM}; arxiv
2602.19758, 2509.10978, 2608.28532, 2405.04395, 2309.05621; IMDEA eucnc2026_pior; wineslab/colosseum-oran-coloran-dataset.
