READY (worker exp-c, xm/exp-c from feat/v2 5a811de; no push). R-55 calib: Kaggle ref DONE, VPS (3.12.14) + Colab Intel
factors DONE (Colab Intel x 2 sessions), Colab AMD provisional. DEV table complete (cdl in).

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
  * calib (R-55 Q1): 6 anchors (pcorr_eq, rcot2_eq, shap_dag, pmrt_nl_eq, cdl, mscr_eq_min), n 500/1000/4000
    (mscr <= 1000), E2 R2, seeds 3_000_000-002 (= "3 repeats"), 51 units. paper (Q3, after EVAL starts): 25 arms x
    n grid, 3 seeds, 348 units, est. 3.9 CPU-h.
- Tests tests/test_xmethod_exp_c_runtime.py: 20 passed. colab_run.py = xm/dev-runs' copy (byte-identical).
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
- Median ref CPU-s at n 500 / 1000 / 4000: pcorr_eq .08 / .12 / .55; rcot2_eq .91 / 1.77 / 2.89; shap_dag 2.2 /
  3.6 / 22.0; pmrt_nl_eq 43 / 67 / 206; cdl 35 / 66 / 413; mscr_eq_min 363 / 780 (n <= 1000). Max unit 784 s
  (mscr_eq_min n 1000), 10.9 % of the budget.
- Findings: (1) Kaggle also assigns AMD EPYC 7B12 sessions (the DEV cdl Kaggle shards ran there): a separate host
  type (R-55 Q2), uncalibrated -> provisional. (2) The reference label itself varies between sessions: same keys,
  same code and p-values, DEV (contended) / calib (uncontended) CPU-s = .89 (mscr_eq_min n 1000) to 1.91
  (shap_dag n 500). GCP uses "Intel Xeon @ 2.20GHz" for more than one machine type, so the timing records now
  also log CPU family / model / stepping / flags (commit 7b81488; host type key unchanged). (3) cdl n 4000 costs
  413 ref CPU-s here vs 128.5 in R-53 (same 4063 gradient steps). The R-53 figure was measured on another host;
  the T3 check for cdl n 8000 / 24000 must come from the cdl DEV run, converted.
- Paired factors f = Kaggle / host CPU-s (results/exp_c/calib/factors.json, raw/ per job, identical datasets):
  VPS (cal-v2, Py 3.12.14; vps.md) cdl 2.31, mscr 2.81, pmrt_nl 2.28, shap 2.30, rcot2 1.99, pcorr 1.92, pooled
  2.56 [1.92, 2.81] (3.12.13 cal-v1: 2.69, factors_vps_py3.12.13.json, vps.md Q4); Colab Intel Xeon 2.2 GHz model 79, 2 sessions (cal-c14 + cal-c16, 2 procs each, 51 / 51 ok, Py 3.12.14,
  pins ok; host cost = mean over sessions): cdl 1.05, mscr 1.10, pmrt_nl .99, shap 1.00, rcot2 .98, pcorr .96,
  pooled 1.067 [.96, 1.10]; session f 1.085 / 1.050 (largest arm gap pmrt_nl 1.04 / .94). Colab wall / CPU median
  1.15 (max 1.78) with 2 procs on quota 2 (CPU-s is the measure). Others: provisional (observational).
## Running / next
- Colab AMD not covered (Q6): 3-job cap used (c14 Intel, c15 LOST, c16 Intel DONE 06:40Z) -> stays provisional.
## Questions
- Q1-Q4 ANSWERED (orchestrator panel, R-55, 2026-10-04): Q1 trimmed paired calibration block, built as above;
  Q2 platform + CPU model; Q3 dedicated paper block, prepared, launch after EVAL starts; Q4 pooled rows with
  median / mean / min / max, tune / measure flag kept, per world / regime in the json.
- Q5 ANSWERED (orchestrator): launch in-session only (no new Task Scheduler entries); blocked Colab tick -> in-session
  keep-alive ticks. Q6 ANSWERED: Colab until both CPU models covered within the 3-job cap; uncovered = provisional.
