# Worker H report (E6 template MSCR discovery) — summary written by orchestrator, 2026-09-28
Files: cdd_oran/decision/{features,discovery}.py, tests/test_decision_discovery.py (4 tests).
- features.build_panel(trace, step_s=10): per (episode, step, cell) rows; own knobs (hys, ttt, ll_ratio, carrier,
  sleep, cio_out mean/max), neighbour knobs (cio_in_mean, nbr_carrier), nbr_util_lag; KPI families from delivered KPM
  (prb_util, ll_delay_p95, embb_thp_p5, rlf, too_late, too_early, energy_w) + lags; ownership from logged requests;
  degenerate columns dropped+recorded (real traces: own_hys/ttt/sleep constant).
- discovery.discover_template: per KPI family, tested = own/neighbour knobs + nbr state, conditioners = own lagged
  KPIs + is_macro; rows thinned (≥60 s / report period), seeded shuffle (tie-bug guard), frozen MSCR B=2999, BY q=.05;
  OLS partial sign; soft weight w = floor + (1−floor)·min(1, ln(1/p)/ln(B+1)); diagnostics acf1, Bartlett n_eff,
  dependence_ok (n_eff ≥ 240), power_floor_ok. conflict_map (declared ownership MRO→rlf/too_late/too_early,
  TS→embb_thp_p5, ES→energy_w, SLICE→ll_delay_p95; prb_util unowned), context_mask / region_weight_matrix.
- REAL-DATA FINDING (1 episode, seed 17): persistent targets (prb_util acf .76, energy .98, embb .57; n_eff 20–200,
  dependence_ok False) declare EVERY knob at the p floor = spurious-regression pattern; dependence_ok targets declare
  none. MSCR's null permutes rows freely (no block/episode null).
- OPEN: (1) need ≥10–20 randomized episodes + an EPISODE-LEVEL null (mscr.py change + recalibration, or one row per
  episode×cell); (2) collector randomizes decisions, not knob VALUES → edges are load-confounded associations; needs
  designed bounded knob dither for identification; (3) ownership declared; ES→util→TS conflicts need KPI→KPI step.
- Sample size (heuristic): prb_util ~2–3 episodes, embb ~2, energy ~10 at 60 s stride.
