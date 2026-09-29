# Worker K report — orchestrator summary 2026-09-28
probe.py (e6-probe/1, operator DEV only, wg3=False): seeded schedule; type A 120 s slots (90 obs + 30 washout) over
{cio,hys,ttt,ll_ratio,sham}; type B 240 s slots (180 obs > 120 s dwell) {carrier,sham} macros; unit = region + CIO-
linked neighbourhood, non-overlapping per slot; worst-case changes/hour cap; one actuator step on all family knobs;
feasible-retry 12 s; restore at window end ("superseded" if xApp changed it); KPM-based abort/restore (also sham);
collection_cost vs same-seed no-probe ref (SVR, slices, RLF, severe, energy, churn, p95/p99, changed-knob-seconds).
crt.py (mscr-crt-v1): per (family, KPI) sharp null; outcome = KPI change pre-60 s → obs window; null = re-draw logged
assignment restricted to {f, sham}, outcomes fixed; statistic = MSCR S* (reused _s_star, vectorised copy verified
~1e-17), constants nc=3 nb=3 min_stratum=10; secondary episode-cluster signed contrast; declared (BY, labelled)/
not_detected/undetermined; placebo mode on no-probe refs.
Calibration (synthetic, 40 eps, B=199): type-I ≈ .04–.06 (one .075 at 2 SE); power hys .36/.43 @0.5 SD, .90/.80 @1 SD;
carrier .77 @1 SD. Grid 180 jobs (seeds 110000+30·stratum+j), ~38 s + 0.9 MB per job (probe + ref).
OPEN: (1) strict unit rule → ~1 unit/slot, ~4 blocks/episode, ~100 blocks/family → power .5–.7 @0.5 SD (needs sol:
weaker non-adjacent rule or more episodes); (2) xApps overwrite ll_ratio/CIO probes in seconds → estimand = assigned-
probe effect incl. reactions; (3) placebo needs grid; (4) register RNG tags 8808/5151/6262; (5) abort thresholds [A].
