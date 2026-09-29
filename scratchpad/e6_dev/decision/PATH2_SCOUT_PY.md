# Path-2 scout 3: lightweight sims vs extending E6 (2026-09-28, desk research, no sims run)

Bottom line: EXTEND E6 (~1-2 d) beats any external sim (1-2 wk); no lightweight public sim has 2 xApps on coupled
power/slice/sleep knobs with interference. Gate A is necessary-not-sufficient: direct same-knob conflicts pass trivially
(static priority recovers ~100%) -> no edge over QACM. ADD criterion: state-dependent headroom = per-request oracle
recovery minus best static subset/priority recovery.

Ranked: 1 extend E6 PROMISING(conditional) | 2 Sionna SYS MARGINAL (physics cross-check) | 3 ns-o-ran-gym TS+ES MARGINAL
(credibility check; no power/slice knob; GPL; speed unknown) | 4 xTRUCE setup MARGINAL as env, key comparator (4 cells,
static CVXPY, code "upon publication"; 65-92% is for a clipping arbiter) | 5 NIST ns3-oran MARGINAL (HO class) |
DEAD: mobile-env (SNR only, no interference), Py5cheSim (single cell), Simu5G (heavy), Vienna/MATLAB, 5G-air-sim, PyLTEs.
Comparator code: QACM repo = 5 synthetic CSVs; PACIFISTA none; ACCoRD (2605.22306) +2.8-4 pp over rule priority (HO
small headroom again); twin-fidelity 2607.22857 MATLAB no code; 2504.06867 16% degradation no code; IMDEA PIOR slice
flipping = direct conflict.

Mechanism: harm is conflict-limited iff (1) floor feasible under a same-rights subset and (2) co-deployment leaves the
feasible set. E6 failed (1) (base freeze SVR 515-656 = capacity). ES conflicts live at low/medium load where capacity is
not binding.
- ES carrier/pico sleep x protected-slice guarantee: est loss 15-40%, recoverable 60-90%, HIGH edge if scored at
  MATCHED ENERGY (else "never sleep" trivially recovers). BEST PAIR.
- ES sleep x MLB/TS: PACIFISTA -50%; needs competent TS (E6 TS broken).
- coverage power-up x neighbour interference/ES power-down: 5-20%, 30-60%, medium.
- same-knob direct (ES vs coverage power; slice quota vs tput-max): ~100% recoverable, NO edge.
Latent E6 coupling: ric.py:142 prb_util includes reserved PRBs; sim.py:210-216 ll_ratio pool dedicated
non-work-conserving (TS 28.541 dedicated not min ratio); carrier-off halves capacity (MACRO_NTRX=2) -> ES x SLICE coupled.

Extend E6 needs: ("ptx",c) dB offset on gain column (geometry.py:173) + EARTH PMAX scaling; protected eMBB slice with
work-conserving min quota; power-ES + coverage power-up xApps; neighbour-list fix if MLB. Risk: "designed to order" ->
ground loads/thresholds/floors in external specs and freeze before outcomes; cross-check direction in Sionna or
ns-o-ran-gym.

Screen 1 E6-P (~1.5 d build, dev-only, needs go): mechanism unit tests first (power step raises neighbour edge SINR;
quota work-conserving; carrier-off halves PRBs; each xApp alone improves own KPI vs freeze). Pairs P1 ES x
SLICE-guarantee, P2 coverage power-up x power-ES, P3 ES x SLICE x power. Loads low+medium (declared from ES literature);
scenarios base + surge in guaranteed-slice cell. Arms: freeze, each alone, accept-all, best static subset, static
priority (QACM-like), budgeted per-request oracle. Primary: protected-slice floor-violation UE-s at MATCHED ENERGY (keep
>=X% of ES-alone saving, X declared). 8 seeds/stratum, fresh DEV range. ~8 CPU-h total. Kill per pair: loss<15% all
strata DEAD; oracle recovers <50% DEAD; static/priority recovers >=80% of oracle recovery -> no-edge STOP.
Screen 2 ns-o-ran-gym ES x TS: 1-day Docker timebox; kill if no build or <0.1x real time; default 63-UE; arms freeze/TS/
ES/both/heuristic arbiter; 5 seeds; kill if both-vs-best-single loss <15%.

Unknowns: all loss magnitudes are estimates; matched-energy reference acceptance; Sionna/ns-o-ran speed; xTRUCE code
timing; reviewer acceptance of self-built plant.
Sources: arxiv 2608.28532, 2605.22306, 2607.22857, 2504.06867, 2601.02240, 2609.27337, 2605.02149;
github stefanbschneider/mobile-env, wineslab/ns-o-ran-gym, dewanwadud1/QACM, ClaudinaRattaro/Py5cheSim;
ieeexplore 11007770 (PACIFISTA); nvlabs sionna sys; IMDEA EuCNC'26 PIOR.
