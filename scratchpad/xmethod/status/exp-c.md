READY (worker exp-c, xm/exp-c from feat/v2 5a811de; no push). R-55 calib DONE (Kaggle ref, VPS 3.12.14, Colab Intel
x 2; Colab AMD provisional). DEV table complete. R-60 X1 paper timing run DONE (all arms feasible). Nothing running.

# exp-c: Experiment C runtime table (R-13, R-35, R-41, R-51, R-55; PROTOCOL_A s.7, T3, V10)
## Deliverables
- `scratchpad/xmethod/exp_c_runtime.py` (xm-exp-c/1): `table` -> EXP_C_RUNTIME.md + exp_c_runtime.json, per (arm, n)
  CPU-s per unit mean / median / min / max in Kaggle-reference s (R-55 Q4), raw stats, per role (tune / measure) in
  the json, per (world, regime), peak RSS median / max, wall / CPU, procs per vCPU (R-55 load fields), T3 vs 7200
  CPU-s tested on cost x f_hi (R-55: within the factor's error = over budget), freeze spec fields. Host type =
  platform + CPU model (R-55 Q2). `calib`: per (arm, host) f = sum ref / sum host, f_lo / f_hi (per-n spread),
  dataset-hash agreement of pairs; arm "*" per host = pooled anchors, f_hi = max anchor f (used by every non-anchor
  arm). `plan`: full T3 201-unit plan (only if spare capacity, R-55). DEV output results/exp_c/dev/.
- `scratchpad/xmethod/exp_c_timing.py` (xm-exp-c-timing/1) + `specs/exp_c/timing.json` + `_bundle_expc/` (uv.lock
  export, hashes): one single-thread process per vCPU (cgroup quota), LPT assignment, per unit load start / end
  (loadavg, running procs, concurrent timing procs, quota), CPU model, commit, pkgs; Python 3.12 venv + pinned lock
  + torch 2.10.0 CPU, pin check; launch gates (Kaggle < 4 running, Colab <= 3 exp-c jobs, never Lightning).
  calib (R-55 Q1): 6 anchors, n 500/1000/4000 (mscr <= 1000), E2 R2, seeds 3_000_000-002, 51 units; paper: below.
- Tests tests/test_xmethod_exp_c_runtime.py: 21 passed. colab_run.py = xm/dev-runs' copy (byte-identical).
  `--slot5` = launch override (orchestrator only); timing records log a CPU family / model / flags fingerprint.
## DEV findings
- Table rebuilt 13:1xZ on full + ci_c + pmrt_nl + cdl DEV (106 840 units, 114 arm x n rows), R-55 factors
  (calib/factors.json; Colab Intel + VPS calibrated, others observational): no arm T3-infeasible. Max unit 1879
  ref CPU-s at f_hi (cdl n 24000, 26 % of 7200). cdl median / max ref s at n 500 / 1k / 4k / 8k / 24k: 20 / 57,
  50 / 122, 147 / 439, 271 / 779, 575 / 1654; peak RSS <= 390 MB. pmrt_nl_eq median / max 30 / 64, 45 / 98, 115 /
  290 (n 500 / 1k / 4k). Notes: cdl DEV summary has dirty_records 2536 (their run, not mine); Exp B E6 cdl at n
  ~10k hit 7200 ref s on the VPS while DEV cdl n 24000 tops out at 1654: E6 is a costlier world (Exp B only).
## R-55 calibration, Kaggle reference (results/exp_c/calib/: kaggle_ref/EXP_C_RUNTIME.md + json, raw/xm-expc-cal-k1/)
- xm-expc-cal-k1 (commit cbfbdad, `--slot5` = 5th account slot by orchestrator instruction): COMPLETE 00:38Z,
  51 / 51 ok, 0 errors, wall 1634 s; 4 procs on cgroup quota 4 (<= 4 timing procs at every unit start), wall / CPU
  1.00 (max 1.02): uncontended. Pin check: 0 mismatches, torch 2.10.0+cpu, Python 3.12.14. Dataset sha = DEV for
  all 33 keys with a DEV record.
- Median ref CPU-s n 500 / 1k / 4k: pcorr_eq .08 / .12 / .55; rcot2_eq .91 / 1.77 / 2.89; shap 2.2 / 3.6 / 22;
  pmrt_nl_eq 43 / 67 / 206; cdl 35 / 66 / 413; mscr_eq_min 363 / 780. Max unit 784 s (10.9 % of budget).
- Findings: (1) Kaggle also assigns AMD EPYC 7B12 sessions: separate host type, uncalibrated -> provisional.
  (2) The reference label varies between sessions (DEV contended / calib uncontended CPU-s .89-1.91); records log
  CPU family / model / flags (7b81488). (3) cdl n 4000 = 413 ref s here vs 128.5 in R-53 (another host).
- Paired factors f = Kaggle / host CPU-s (results/exp_c/calib/factors.json, raw/ per job, identical datasets):
  VPS (cal-v2, Py 3.12.14; vps.md) cdl 2.31, mscr 2.81, pmrt_nl 2.28, shap 2.30, rcot2 1.99, pcorr 1.92, pooled
  2.56 [1.92, 2.81] (3.12.13 cal-v1: 2.69, factors_vps_py3.12.13.json, vps.md Q4); Colab Intel Xeon 2.2 GHz model 79, 2 sessions (cal-c14 + cal-c16, 2 procs each, 51 / 51 ok, Py 3.12.14,
  pins ok; host cost = mean over sessions): cdl 1.05, mscr 1.10, pmrt_nl .99, shap 1.00, rcot2 .98, pcorr .96,
  pooled 1.067 [.96, 1.10]; session f 1.085 / 1.050 (largest arm gap pmrt_nl 1.04 / .94). Colab wall / CPU median
  1.15 (max 1.78) with 2 procs on quota 2 (CPU-s is the measure). Others: provisional (observational).
## R-60 X1: paper timing run DONE (results/exp_c/paper/EXP_C_RUNTIME.md + json, raw/xm-expc-paper-k1/)
- Spec a1153ad: 24 frozen EVAL arms (refs / configs / worlds = feat/v2 specs/eval/full.json; mscr_eq_min dropped,
  R-58(5)) x n 500-24000 (mscr <= 1000 = "not in grid"), E2 R2 (granger E3 R2), seeds 3_000_000-002, 342 units.
- Launched 02:54Z (EVAL all-launched 02:20Z; 3 Kaggle running, 1 slot left free, R-60), commit 147fcc2, one
  session, no campaign parts. Host Intel Xeon 2.20GHz model 79 = reference host (not EPYC 7B12: not provisional);
  Py 3.12.14, pins 0 mismatches. COMPLETE 04:05Z, 4162 s wall, 342 / 342 ok, 0 dup / missing / extra; 4 procs on
  quota 4, wall / CPU median 1.007 (max 1.02): uncontended. 4.47 CPU-h.
- T3: every arm feasible at every grid n. Costliest: cdl n 24000 median / max 1436 / 1443 ref s (20 % of 7200);
  pmrt_nl_eq 1108 / 1140; shap_dag 119 / 121; two_tower 114 / 121; pc_eq 49 / 66; pmrt_eq 52 / 57. mscr_eq 131 /
  132 at n 1000. Peak RSS max 963 MB (pmrt_nl_eq n 24000).
- Same cell vs DEV (E2 R2, DEV median > .5 s, 88 cells): paper / DEV median .70 (.21 pcorr_hac_fb* to 1.15
  pmrt_r3 n 24000): the uncontended run is at or below the contended DEV costs. Pooled DEV rows differ more
  (other worlds), so this table is per method at one cell.
- exp_c_runtime.py: T3 marks n above a spec max_n "not in grid" (R-58 wording); test added.
## Questions
- Q1-Q4 ANSWERED (orchestrator, R-55): trimmed paired calib block; host = platform + CPU model; dedicated paper
  block after EVAL starts; pooled rows median / mean / min / max, tune / measure flag, per world / regime in json.
- Q5 / Q6 ANSWERED: in-session launches only; Colab until both CPU models covered, cap 3 (AMD: provisional).
