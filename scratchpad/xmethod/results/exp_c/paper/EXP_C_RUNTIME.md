# Experiment C: runtime per method (R-55 paper timing run (R-60 X1): xm-expc-paper-k1, one uncontended Kaggle session, Intel Xeon 2.20GHz model 79 (reference host), 24 frozen arms, E2 R2 (granger E3 R2), seeds 3_000_000-002)

Generated 2026-10-05T04:07:06Z by `scratchpad/xmethod/exp_c_runtime.py` (xm-exp-c/1). Unit = one (method, dataset): record `cpu_s` = process CPU time of `method.run`, single thread (data generation excluded, shown apart). Costs in **Kaggle reference CPU-s** (`kaggle|Intel(R) Xeon(R) CPU @ 2.20GHz`); other hosts x f (R-51). Peak RSS = `peak_rss_mb` (scope per job on Linux; includes interpreter, imports and dataset), not converted. Budget R-13: 7,200 CPU-s per (method, dataset); T3: max over worlds, regimes, lambdas at n > budget makes n and every larger n infeasible; R-55: tested on cost x f_hi (within the speed factor's error of the budget = over). Rows pool tune + measure seeds, kappas and lambdas (per role and per world / regime in the json).

## Inputs

- `D:\academia\major-project\CDD-ORAN-wt\xm-citests\scratchpad\xmethod\results\exp_c\paper\raw\xm-expc-paper-k1\res_0.jsonl` (85 lines, sha256 `4d49c168a7deffa2`)
- `D:\academia\major-project\CDD-ORAN-wt\xm-citests\scratchpad\xmethod\results\exp_c\paper\raw\xm-expc-paper-k1\res_1.jsonl` (85 lines, sha256 `f24945c945c3450d`)
- `D:\academia\major-project\CDD-ORAN-wt\xm-citests\scratchpad\xmethod\results\exp_c\paper\raw\xm-expc-paper-k1\res_2.jsonl` (86 lines, sha256 `69c1855f5afb5733`)
- `D:\academia\major-project\CDD-ORAN-wt\xm-citests\scratchpad\xmethod\results\exp_c\paper\raw\xm-expc-paper-k1\res_3.jsonl` (86 lines, sha256 `2d5a189030c8bc77`)
- 342 lines, 0 duplicate keys (0 on two hosts); specs exp_c_paper; commits 147fcc2

Hosts (units):

- `kaggle|Intel(R) Xeon(R) CPU @ 2.20GHz`: 342

## Speed factors applied (cost_ref = cost x f)

None needed (every unit on the reference host) or none available.

## Summary: median / max CPU-s per unit (Kaggle ref.) by n; T3 status

| arm | n 500 | n 1000 | n 4000 | n 8000 | n 24000 | T3 |
|---|---|---|---|---|---|---|
| cdl | 36.7 / 37.8 | 75.8 / 76.0 | 298 / 386 | 726 / 739 | 1,436 / 1,443 | feasible n <= 24000 |
| corr | 0.055 / 0.086 | 0.057 / 0.085 | 0.064 / 0.098 | 0.075 / 0.11 | 0.13 / 0.18 | feasible n <= 24000 |
| granger_eq | 0.11 / 0.16 | 0.15 / 0.15 | 0.46 / 0.65 | 1.00 / 1.31 | 3.67 / 4.56 | feasible n <= 24000 |
| granger_native | 0.029 / 0.047 | 0.032 / 0.051 | 0.054 / 0.074 | 0.071 / 0.073 | 0.23 / 0.24 | feasible n <= 24000 |
| mscr_eq | 62.6 / 62.7 | 131 / 132 | - | - | - | feasible n <= 1000; not in grid at n 4000, 8000, 24000 (spec max_n) |
| mscr_native | 22.1 / 23.6 | 45.8 / 47.2 | - | - | - | feasible n <= 1000; not in grid at n 4000, 8000, 24000 (spec max_n) |
| notears | 0.44 / 1.35 | 0.60 / 1.69 | 1.80 / 2.56 | 3.48 / 3.87 | 12.0 / 19.0 | feasible n <= 24000 |
| pc_eq | 4.79 / 5.61 | 6.58 / 10.4 | 16.8 / 23.1 | 27.5 / 33.6 | 49.4 / 65.9 | feasible n <= 24000 |
| pc_native | 0.45 / 0.61 | 0.58 / 0.85 | 1.64 / 2.84 | 1.56 / 2.74 | 4.84 / 5.63 | feasible n <= 24000 |
| pcorr_eq | 0.067 / 0.068 | 0.10 / 0.16 | 0.50 / 0.51 | 0.97 / 1.21 | 2.71 / 3.64 | feasible n <= 24000 |
| pcorr_eq_min | 0.068 / 0.11 | 0.10 / 0.16 | 0.32 / 0.46 | 0.91 / 0.93 | 2.37 / 3.16 | feasible n <= 24000 |
| pcorr_hac | 0.18 / 0.19 | 0.26 / 0.28 | 0.71 / 0.71 | 0.91 / 1.47 | 4.25 / 4.33 | feasible n <= 24000 |
| pcorr_hac_eq_min | 0.13 / 0.20 | 0.27 / 0.27 | 0.95 / 0.95 | 0.94 / 1.33 | 3.35 / 4.39 | feasible n <= 24000 |
| pcorr_hac_fb | 0.37 / 0.38 | 0.38 / 0.47 | 0.57 / 0.92 | 1.41 / 1.46 | 3.80 / 4.54 | feasible n <= 24000 |
| pcorr_hac_fb_eq_min | 0.32 / 0.46 | 0.28 / 0.44 | 0.83 / 0.96 | 1.07 / 1.45 | 5.20 / 5.27 | feasible n <= 24000 |
| pcorr_native | 0.013 / 0.020 | 0.018 / 0.018 | 0.054 / 0.054 | 0.10 / 0.10 | 0.39 / 0.53 | feasible n <= 24000 |
| pmrt_eq | 0.35 / 0.48 | 0.82 / 1.07 | 4.49 / 5.77 | 10.9 / 14.1 | 51.9 / 56.9 | feasible n <= 24000 |
| pmrt_nl_eq | 36.5 / 38.6 | 65.3 / 68.2 | 185 / 209 | 353 / 421 | 1,108 / 1,140 | feasible n <= 24000 |
| pmrt_r3 | 0.42 / 0.51 | 0.84 / 1.01 | 3.92 / 5.81 | 14.7 / 15.4 | 51.4 / 52.1 | feasible n <= 24000 |
| rcot2_eq | 1.46 / 1.47 | 1.54 / 1.66 | 3.84 / 3.96 | 6.81 / 6.83 | 19.2 / 21.9 | feasible n <= 24000 |
| rcot2_eq_min | 0.77 / 1.31 | 1.49 / 1.61 | 2.29 / 3.46 | 4.15 / 6.34 | 17.9 / 18.4 | feasible n <= 24000 |
| rcot2_native | 1.26 / 1.28 | 1.49 / 1.71 | 3.40 / 3.40 | 4.09 / 6.02 | 16.3 / 17.7 | feasible n <= 24000 |
| shap_dag | 3.08 / 3.55 | 5.81 / 6.01 | 16.2 / 20.0 | 39.0 / 39.6 | 119 / 121 | feasible n <= 24000 |
| two_tower | 4.31 / 5.15 | 6.04 / 6.68 | 18.4 / 18.8 | 31.0 / 31.4 | 114 / 121 | feasible n <= 24000 |

`-` = no record at that n (above the spec max_n: not in grid; else T3: a cost pilot decides if the n is in the arm's grid); * = some unit unconverted; + = some unit converted with a provisional (observational) factor. Pooled over worlds, regimes, lambdas, kappas (the kappa sweep is at n 1000) and roles (tune + measure).

## Full table

| arm | method | n | units (tune / measure) | non-ref | CPU-s mean | median | min | max | max x f_hi | max / budget | RSS med MB | RSS max MB | wall-s mean | wall / CPU | procs / vCPU | gen CPU-s | multi-thr | T3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cdl | cdl | 500 | 3 (timing 3) | 0 % | 36.4 | 36.7 | 34.7 | 37.8 | 37.8 | 0.53 % | 712 | 745 | 36.7 | 1.01 | 1.00 | 0.018 | 0 | feasible |
| cdl | cdl | 1000 | 3 (timing 3) | 0 % | 73.1 | 75.8 | 67.5 | 76.0 | 76.0 | 1.06 % | 757 | 757 | 73.7 | 1.01 | 1.00 | 0.030 | 0 | feasible |
| cdl | cdl | 4000 | 3 (timing 3) | 0 % | 326 | 298 | 293 | 386 | 386 | 5.36 % | 626 | 628 | 328 | 1.01 | 1.00 | 0.13 | 0 | feasible |
| cdl | cdl | 8000 | 3 (timing 3) | 0 % | 727 | 726 | 717 | 739 | 739 | 10.26 % | 387 | 390 | 734 | 1.01 | 1.00 | 0.24 | 0 | feasible |
| cdl | cdl | 24000 | 3 (timing 3) | 0 % | 1,437 | 1,436 | 1,432 | 1,443 | 1,443 | 20.05 % | 393 | 394 | 1,452 | 1.01 | 1.00 | 0.72 | 0 | feasible |
| corr | corr | 500 | 3 (timing 3) | 0 % | 0.065 | 0.055 | 0.053 | 0.086 | 0.086 | 0.00 % | 829 | 830 | 0.065 | 1.00 | 0.75 | 0.012 | 0 | feasible |
| corr | corr | 1000 | 3 (timing 3) | 0 % | 0.066 | 0.057 | 0.056 | 0.085 | 0.085 | 0.00 % | 830 | 834 | 0.066 | 1.00 | 0.75 | 0.022 | 0 | feasible |
| corr | corr | 4000 | 3 (timing 3) | 0 % | 0.075 | 0.064 | 0.062 | 0.098 | 0.098 | 0.00 % | 830 | 834 | 0.075 | 1.00 | 0.75 | 0.084 | 0 | feasible |
| corr | corr | 8000 | 3 (timing 3) | 0 % | 0.087 | 0.075 | 0.073 | 0.11 | 0.11 | 0.00 % | 830 | 834 | 0.087 | 1.00 | 0.50 | 0.16 | 0 | feasible |
| corr | corr | 24000 | 3 (timing 3) | 0 % | 0.14 | 0.13 | 0.12 | 0.18 | 0.18 | 0.00 % | 830 | 834 | 0.14 | 1.00 | 0.50 | 0.47 | 0 | feasible |
| granger_eq | granger | 500 | 3 (timing 3) | 0 % | 0.13 | 0.11 | 0.11 | 0.16 | 0.16 | 0.00 % | 829 | 829 | 0.13 | 1.00 | 0.75 | 0.007 | 0 | feasible |
| granger_eq | granger | 1000 | 3 (timing 3) | 0 % | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 | 0.00 % | 834 | 834 | 0.15 | 1.00 | 0.50 | 0.010 | 0 | feasible |
| granger_eq | granger | 4000 | 3 (timing 3) | 0 % | 0.52 | 0.46 | 0.44 | 0.65 | 0.65 | 0.01 % | 829 | 835 | 0.52 | 1.00 | 0.50 | 0.050 | 0 | feasible |
| granger_eq | granger | 8000 | 3 (timing 3) | 0 % | 1.10 | 1.00 | 0.98 | 1.31 | 1.31 | 0.02 % | 829 | 835 | 1.10 | 1.00 | 0.50 | 0.099 | 0 | feasible |
| granger_eq | granger | 24000 | 3 (timing 3) | 0 % | 3.95 | 3.67 | 3.63 | 4.56 | 4.56 | 0.06 % | 774 | 834 | 3.98 | 1.00 | 0.50 | 0.28 | 0 | feasible |
| granger_native | granger | 500 | 3 (timing 3) | 0 % | 0.035 | 0.029 | 0.029 | 0.047 | 0.047 | 0.00 % | 830 | 834 | 0.035 | 1.00 | 0.75 | 0.007 | 0 | feasible |
| granger_native | granger | 1000 | 3 (timing 3) | 0 % | 0.038 | 0.032 | 0.032 | 0.051 | 0.051 | 0.00 % | 829 | 830 | 0.038 | 1.00 | 0.75 | 0.014 | 0 | feasible |
| granger_native | granger | 4000 | 3 (timing 3) | 0 % | 0.059 | 0.054 | 0.048 | 0.074 | 0.074 | 0.00 % | 829 | 830 | 0.059 | 1.00 | 0.75 | 0.057 | 0 | feasible |
| granger_native | granger | 8000 | 3 (timing 3) | 0 % | 0.071 | 0.071 | 0.070 | 0.073 | 0.073 | 0.00 % | 829 | 834 | 0.071 | 1.00 | 0.50 | 0.077 | 0 | feasible |
| granger_native | granger | 24000 | 3 (timing 3) | 0 % | 0.21 | 0.23 | 0.16 | 0.24 | 0.24 | 0.00 % | 829 | 834 | 0.21 | 1.00 | 0.75 | 0.35 | 0 | feasible |
| mscr_eq | mscr | 500 | 3 (timing 3) | 0 % | 62.5 | 62.6 | 62.1 | 62.7 | 62.7 | 0.87 % | 765 | 767 | 63.0 | 1.01 | 1.00 | 0.017 | 0 | feasible |
| mscr_eq | mscr | 1000 | 3 (timing 3) | 0 % | 132 | 131 | 131 | 132 | 132 | 1.83 % | 619 | 627 | 133 | 1.01 | 1.00 | 0.033 | 0 | feasible |
| mscr_native | mscr | 500 | 3 (timing 3) | 0 % | 22.5 | 22.1 | 21.9 | 23.6 | 23.6 | 0.33 % | 745 | 751 | 22.7 | 1.01 | 1.00 | 0.017 | 0 | feasible |
| mscr_native | mscr | 1000 | 3 (timing 3) | 0 % | 45.8 | 45.8 | 44.5 | 47.2 | 47.2 | 0.66 % | 674 | 701 | 46.1 | 1.01 | 1.00 | 0.032 | 0 | feasible |
| notears | notears | 500 | 3 (timing 3) | 0 % | 0.73 | 0.44 | 0.40 | 1.35 | 1.35 | 0.02 % | 829 | 835 | 0.73 | 1.00 | 0.50 | 0.014 | 0 | feasible |
| notears | notears | 1000 | 3 (timing 3) | 0 % | 0.94 | 0.60 | 0.52 | 1.69 | 1.69 | 0.02 % | 829 | 835 | 0.94 | 1.00 | 0.50 | 0.023 | 0 | feasible |
| notears | notears | 4000 | 3 (timing 3) | 0 % | 1.86 | 1.80 | 1.22 | 2.56 | 2.56 | 0.04 % | 835 | 835 | 1.86 | 1.00 | 0.50 | 0.084 | 0 | feasible |
| notears | notears | 8000 | 3 (timing 3) | 0 % | 3.24 | 3.48 | 2.35 | 3.87 | 3.87 | 0.05 % | 830 | 835 | 3.25 | 1.00 | 0.50 | 0.17 | 0 | feasible |
| notears | notears | 24000 | 3 (timing 3) | 0 % | 12.7 | 12.0 | 7.02 | 19.0 | 19.0 | 0.26 % | 754 | 758 | 12.8 | 1.01 | 1.00 | 0.58 | 0 | feasible |
| pc_eq | pc | 500 | 3 (timing 3) | 0 % | 5.03 | 4.79 | 4.69 | 5.61 | 5.61 | 0.08 % | 746 | 758 | 5.03 | 1.00 | 0.50 | 0.009 | 0 | feasible |
| pc_eq | pc | 1000 | 3 (timing 3) | 0 % | 7.74 | 6.58 | 6.19 | 10.4 | 10.4 | 0.14 % | 746 | 756 | 7.76 | 1.00 | 0.50 | 0.022 | 0 | feasible |
| pc_eq | pc | 4000 | 3 (timing 3) | 0 % | 18.3 | 16.8 | 15.0 | 23.1 | 23.1 | 0.32 % | 752 | 756 | 18.4 | 1.00 | 1.00 | 0.10 | 0 | feasible |
| pc_eq | pc | 8000 | 3 (timing 3) | 0 % | 27.3 | 27.5 | 20.8 | 33.6 | 33.6 | 0.47 % | 723 | 745 | 27.5 | 1.01 | 1.00 | 0.25 | 0 | feasible |
| pc_eq | pc | 24000 | 3 (timing 3) | 0 % | 50.8 | 49.4 | 37.0 | 65.9 | 65.9 | 0.92 % | 730 | 744 | 51.2 | 1.01 | 1.00 | 0.68 | 0 | feasible |
| pc_native | pc | 500 | 3 (timing 3) | 0 % | 0.46 | 0.45 | 0.30 | 0.61 | 0.61 | 0.01 % | 829 | 835 | 0.46 | 1.00 | 0.50 | 0.012 | 0 | feasible |
| pc_native | pc | 1000 | 3 (timing 3) | 0 % | 0.64 | 0.58 | 0.49 | 0.85 | 0.85 | 0.01 % | 829 | 835 | 0.64 | 1.00 | 0.50 | 0.021 | 0 | feasible |
| pc_native | pc | 4000 | 3 (timing 3) | 0 % | 1.82 | 1.64 | 0.99 | 2.84 | 2.84 | 0.04 % | 829 | 829 | 1.84 | 1.00 | 0.75 | 0.11 | 0 | feasible |
| pc_native | pc | 8000 | 3 (timing 3) | 0 % | 1.89 | 1.56 | 1.37 | 2.74 | 2.74 | 0.04 % | 829 | 835 | 1.89 | 1.00 | 0.50 | 0.13 | 0 | feasible |
| pc_native | pc | 24000 | 3 (timing 3) | 0 % | 5.03 | 4.84 | 4.61 | 5.63 | 5.63 | 0.08 % | 746 | 754 | 5.04 | 1.00 | 1.00 | 0.57 | 0 | feasible |
| pcorr_eq | pcorr | 500 | 3 (timing 3) | 0 % | 0.067 | 0.067 | 0.067 | 0.068 | 0.068 | 0.00 % | 829 | 834 | 0.067 | 1.00 | 0.50 | 0.009 | 0 | feasible |
| pcorr_eq | pcorr | 1000 | 3 (timing 3) | 0 % | 0.12 | 0.10 | 0.10 | 0.16 | 0.16 | 0.00 % | 830 | 834 | 0.12 | 1.00 | 0.75 | 0.022 | 0 | feasible |
| pcorr_eq | pcorr | 4000 | 3 (timing 3) | 0 % | 0.46 | 0.50 | 0.36 | 0.51 | 0.51 | 0.01 % | 829 | 834 | 0.46 | 1.00 | 0.75 | 0.100 | 0 | feasible |
| pcorr_eq | pcorr | 8000 | 3 (timing 3) | 0 % | 0.98 | 0.97 | 0.75 | 1.21 | 1.21 | 0.02 % | 829 | 829 | 0.98 | 1.00 | 0.75 | 0.21 | 0 | feasible |
| pcorr_eq | pcorr | 24000 | 3 (timing 3) | 0 % | 3.02 | 2.71 | 2.71 | 3.64 | 3.64 | 0.05 % | 829 | 835 | 3.02 | 1.00 | 0.50 | 0.47 | 0 | feasible |
| pcorr_eq_min | pcorr | 500 | 3 (timing 3) | 0 % | 0.081 | 0.068 | 0.067 | 0.11 | 0.11 | 0.00 % | 830 | 834 | 0.081 | 1.00 | 0.50 | 0.012 | 0 | feasible |
| pcorr_eq_min | pcorr | 1000 | 3 (timing 3) | 0 % | 0.12 | 0.10 | 0.100 | 0.16 | 0.16 | 0.00 % | 829 | 830 | 0.12 | 1.00 | 0.75 | 0.022 | 0 | feasible |
| pcorr_eq_min | pcorr | 4000 | 3 (timing 3) | 0 % | 0.36 | 0.32 | 0.31 | 0.46 | 0.46 | 0.01 % | 829 | 829 | 0.36 | 1.00 | 0.50 | 0.085 | 0 | feasible |
| pcorr_eq_min | pcorr | 8000 | 3 (timing 3) | 0 % | 0.82 | 0.91 | 0.63 | 0.93 | 0.93 | 0.01 % | 799 | 829 | 0.82 | 1.00 | 0.75 | 0.19 | 0 | feasible |
| pcorr_eq_min | pcorr | 24000 | 3 (timing 3) | 0 % | 2.63 | 2.37 | 2.36 | 3.16 | 3.16 | 0.04 % | 829 | 829 | 2.64 | 1.00 | 0.75 | 0.49 | 0 | feasible |
| pcorr_hac | pcorr_hac | 500 | 3 (timing 3) | 0 % | 0.17 | 0.18 | 0.12 | 0.19 | 0.19 | 0.00 % | 829 | 829 | 0.17 | 1.00 | 0.75 | 0.014 | 0 | feasible |
| pcorr_hac | pcorr_hac | 1000 | 3 (timing 3) | 0 % | 0.24 | 0.26 | 0.17 | 0.28 | 0.28 | 0.00 % | 829 | 834 | 0.25 | 1.00 | 0.75 | 0.027 | 0 | feasible |
| pcorr_hac | pcorr_hac | 4000 | 3 (timing 3) | 0 % | 0.64 | 0.71 | 0.48 | 0.71 | 0.71 | 0.01 % | 829 | 829 | 0.64 | 1.00 | 0.75 | 0.10 | 0 | feasible |
| pcorr_hac | pcorr_hac | 8000 | 3 (timing 3) | 0 % | 1.09 | 0.91 | 0.90 | 1.47 | 1.47 | 0.02 % | 829 | 835 | 1.10 | 1.00 | 0.50 | 0.17 | 0 | feasible |
| pcorr_hac | pcorr_hac | 24000 | 3 (timing 3) | 0 % | 3.92 | 4.25 | 3.19 | 4.33 | 4.33 | 0.06 % | 746 | 754 | 3.97 | 1.02 | 1.00 | 0.57 | 0 | feasible |
| pcorr_hac_eq_min | pcorr_hac | 500 | 3 (timing 3) | 0 % | 0.15 | 0.13 | 0.13 | 0.20 | 0.20 | 0.00 % | 829 | 834 | 0.15 | 1.00 | 0.75 | 0.012 | 0 | feasible |
| pcorr_hac_eq_min | pcorr_hac | 1000 | 3 (timing 3) | 0 % | 0.24 | 0.27 | 0.18 | 0.27 | 0.27 | 0.00 % | 829 | 829 | 0.25 | 1.00 | 0.75 | 0.026 | 0 | feasible |
| pcorr_hac_eq_min | pcorr_hac | 4000 | 3 (timing 3) | 0 % | 0.88 | 0.95 | 0.75 | 0.95 | 0.95 | 0.01 % | 829 | 829 | 0.88 | 1.00 | 1.00 | 0.13 | 0 | feasible |
| pcorr_hac_eq_min | pcorr_hac | 8000 | 3 (timing 3) | 0 % | 1.07 | 0.94 | 0.94 | 1.33 | 1.33 | 0.02 % | 829 | 835 | 1.08 | 1.00 | 0.50 | 0.16 | 0 | feasible |
| pcorr_hac_eq_min | pcorr_hac | 24000 | 3 (timing 3) | 0 % | 3.68 | 3.35 | 3.31 | 4.39 | 4.39 | 0.06 % | 746 | 758 | 3.69 | 1.00 | 0.50 | 0.48 | 0 | feasible |
| pcorr_hac_fb | pcorr_hac | 500 | 3 (timing 3) | 0 % | 0.32 | 0.37 | 0.20 | 0.38 | 0.38 | 0.01 % | 829 | 835 | 0.34 | 1.04 | 0.75 | 0.016 | 0 | feasible |
| pcorr_hac_fb | pcorr_hac | 1000 | 3 (timing 3) | 0 % | 0.37 | 0.38 | 0.27 | 0.47 | 0.47 | 0.01 % | 829 | 829 | 0.37 | 1.00 | 0.75 | 0.029 | 0 | feasible |
| pcorr_hac_fb | pcorr_hac | 4000 | 3 (timing 3) | 0 % | 0.68 | 0.57 | 0.57 | 0.92 | 0.92 | 0.01 % | 829 | 835 | 0.69 | 1.00 | 0.50 | 0.083 | 0 | feasible |
| pcorr_hac_fb | pcorr_hac | 8000 | 3 (timing 3) | 0 % | 1.28 | 1.41 | 0.99 | 1.46 | 1.46 | 0.02 % | 829 | 829 | 1.29 | 1.00 | 0.75 | 0.20 | 0 | feasible |
| pcorr_hac_fb | pcorr_hac | 24000 | 3 (timing 3) | 0 % | 3.87 | 3.80 | 3.26 | 4.54 | 4.54 | 0.06 % | 821 | 834 | 3.89 | 1.00 | 0.50 | 0.47 | 0 | feasible |
| pcorr_hac_fb_eq_min | pcorr_hac | 500 | 3 (timing 3) | 0 % | 0.35 | 0.32 | 0.26 | 0.46 | 0.46 | 0.01 % | 829 | 835 | 0.35 | 1.00 | 0.50 | 0.012 | 0 | feasible |
| pcorr_hac_fb_eq_min | pcorr_hac | 1000 | 3 (timing 3) | 0 % | 0.32 | 0.28 | 0.25 | 0.44 | 0.44 | 0.01 % | 829 | 835 | 0.32 | 1.00 | 0.50 | 0.022 | 0 | feasible |
| pcorr_hac_fb_eq_min | pcorr_hac | 4000 | 3 (timing 3) | 0 % | 0.79 | 0.83 | 0.58 | 0.96 | 0.96 | 0.01 % | 829 | 835 | 0.79 | 1.00 | 0.75 | 0.11 | 0 | feasible |
| pcorr_hac_fb_eq_min | pcorr_hac | 8000 | 3 (timing 3) | 0 % | 1.19 | 1.07 | 1.04 | 1.45 | 1.45 | 0.02 % | 829 | 835 | 1.19 | 1.00 | 0.50 | 0.16 | 0 | feasible |
| pcorr_hac_fb_eq_min | pcorr_hac | 24000 | 3 (timing 3) | 0 % | 4.83 | 5.20 | 4.02 | 5.27 | 5.27 | 0.07 % | 831 | 835 | 4.84 | 1.00 | 1.00 | 0.59 | 0 | feasible |
| pcorr_native | pcorr | 500 | 3 (timing 3) | 0 % | 0.015 | 0.013 | 0.012 | 0.020 | 0.020 | 0.00 % | 829 | 830 | 0.015 | 1.00 | 0.25 | 0.012 | 0 | feasible |
| pcorr_native | pcorr | 1000 | 3 (timing 3) | 0 % | 0.018 | 0.018 | 0.018 | 0.018 | 0.018 | 0.00 % | 829 | 834 | 0.018 | 1.00 | 0.50 | 0.017 | 0 | feasible |
| pcorr_native | pcorr | 4000 | 3 (timing 3) | 0 % | 0.054 | 0.054 | 0.053 | 0.054 | 0.054 | 0.00 % | 834 | 834 | 0.054 | 1.00 | 0.50 | 0.064 | 0 | feasible |
| pcorr_native | pcorr | 8000 | 3 (timing 3) | 0 % | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.00 % | 829 | 834 | 0.10 | 1.00 | 0.50 | 0.13 | 0 | feasible |
| pcorr_native | pcorr | 24000 | 3 (timing 3) | 0 % | 0.44 | 0.39 | 0.39 | 0.53 | 0.53 | 0.01 % | 829 | 834 | 0.44 | 1.00 | 0.50 | 0.47 | 0 | feasible |
| pmrt_eq | pmrt_core | 500 | 3 (timing 3) | 0 % | 0.39 | 0.35 | 0.33 | 0.48 | 0.48 | 0.01 % | 829 | 835 | 0.39 | 1.00 | 0.50 | 0.012 | 0 | feasible |
| pmrt_eq | pmrt_core | 1000 | 3 (timing 3) | 0 % | 0.90 | 0.82 | 0.81 | 1.07 | 1.07 | 0.01 % | 829 | 835 | 0.90 | 1.00 | 0.50 | 0.022 | 0 | feasible |
| pmrt_eq | pmrt_core | 4000 | 3 (timing 3) | 0 % | 4.64 | 4.49 | 3.65 | 5.77 | 5.77 | 0.08 % | 830 | 834 | 4.66 | 1.00 | 1.00 | 0.11 | 0 | feasible |
| pmrt_eq | pmrt_core | 8000 | 3 (timing 3) | 0 % | 11.4 | 10.9 | 9.23 | 14.1 | 14.1 | 0.20 % | 878 | 878 | 11.5 | 1.00 | 0.75 | 0.21 | 0 | feasible |
| pmrt_eq | pmrt_core | 24000 | 3 (timing 3) | 0 % | 50.0 | 51.9 | 41.3 | 56.9 | 56.9 | 0.79 % | 845 | 867 | 50.5 | 1.01 | 1.00 | 0.66 | 0 | feasible |
| pmrt_nl_eq | pmrt_core | 500 | 3 (timing 3) | 0 % | 36.9 | 36.5 | 35.7 | 38.6 | 38.6 | 0.54 % | 701 | 744 | 37.3 | 1.01 | 1.00 | 0.017 | 0 | feasible |
| pmrt_nl_eq | pmrt_core | 1000 | 3 (timing 3) | 0 % | 65.9 | 65.3 | 64.1 | 68.2 | 68.2 | 0.95 % | 704 | 756 | 66.4 | 1.01 | 1.00 | 0.031 | 0 | feasible |
| pmrt_nl_eq | pmrt_core | 4000 | 3 (timing 3) | 0 % | 193 | 185 | 183 | 209 | 209 | 2.91 % | 626 | 667 | 194 | 1.01 | 1.00 | 0.12 | 0 | feasible |
| pmrt_nl_eq | pmrt_core | 8000 | 3 (timing 3) | 0 % | 374 | 353 | 348 | 421 | 421 | 5.84 % | 661 | 663 | 377 | 1.01 | 1.00 | 0.22 | 0 | feasible |
| pmrt_nl_eq | pmrt_core | 24000 | 3 (timing 3) | 0 % | 1,090 | 1,108 | 1,022 | 1,140 | 1,140 | 15.84 % | 950 | 963 | 1,100 | 1.01 | 1.00 | 0.67 | 0 | feasible |
| pmrt_r3 | pmrt_core | 500 | 3 (timing 3) | 0 % | 0.39 | 0.42 | 0.24 | 0.51 | 0.51 | 0.01 % | 829 | 835 | 0.39 | 1.00 | 0.75 | 0.014 | 0 | feasible |
| pmrt_r3 | pmrt_core | 1000 | 3 (timing 3) | 0 % | 0.86 | 0.84 | 0.73 | 1.01 | 1.01 | 0.01 % | 829 | 835 | 0.86 | 1.00 | 0.50 | 0.022 | 0 | feasible |
| pmrt_r3 | pmrt_core | 4000 | 3 (timing 3) | 0 % | 4.35 | 3.92 | 3.33 | 5.81 | 5.81 | 0.08 % | 835 | 852 | 4.38 | 1.00 | 0.50 | 0.084 | 0 | feasible |
| pmrt_r3 | pmrt_core | 8000 | 3 (timing 3) | 0 % | 13.4 | 14.7 | 10.2 | 15.4 | 15.4 | 0.21 % | 868 | 875 | 13.5 | 1.01 | 1.00 | 0.20 | 0 | feasible |
| pmrt_r3 | pmrt_core | 24000 | 3 (timing 3) | 0 % | 46.4 | 51.4 | 35.8 | 52.1 | 52.1 | 0.72 % | 834 | 867 | 46.8 | 1.01 | 1.00 | 0.67 | 0 | feasible |
| rcot2_eq | rcot2 | 500 | 3 (timing 3) | 0 % | 1.24 | 1.46 | 0.78 | 1.47 | 1.47 | 0.02 % | 829 | 835 | 1.24 | 1.00 | 1.00 | 0.017 | 0 | feasible |
| rcot2_eq | rcot2 | 1000 | 3 (timing 3) | 0 % | 1.41 | 1.54 | 1.01 | 1.66 | 1.66 | 0.02 % | 829 | 829 | 1.41 | 1.00 | 0.75 | 0.027 | 0 | feasible |
| rcot2_eq | rcot2 | 4000 | 3 (timing 3) | 0 % | 3.45 | 3.84 | 2.55 | 3.96 | 3.96 | 0.05 % | 829 | 830 | 3.45 | 1.00 | 1.00 | 0.10 | 0 | feasible |
| rcot2_eq | rcot2 | 8000 | 3 (timing 3) | 0 % | 6.09 | 6.81 | 4.64 | 6.83 | 6.83 | 0.09 % | 746 | 754 | 6.14 | 1.01 | 1.00 | 0.20 | 0 | feasible |
| rcot2_eq | rcot2 | 24000 | 3 (timing 3) | 0 % | 19.6 | 19.2 | 17.7 | 21.9 | 21.9 | 0.30 % | 751 | 752 | 19.8 | 1.01 | 1.00 | 0.68 | 0 | feasible |
| rcot2_eq_min | rcot2 | 500 | 3 (timing 3) | 0 % | 0.95 | 0.77 | 0.76 | 1.31 | 1.31 | 0.02 % | 829 | 835 | 0.95 | 1.00 | 0.50 | 0.014 | 0 | feasible |
| rcot2_eq_min | rcot2 | 1000 | 3 (timing 3) | 0 % | 1.34 | 1.49 | 0.93 | 1.61 | 1.61 | 0.02 % | 829 | 835 | 1.34 | 1.00 | 0.75 | 0.026 | 0 | feasible |
| rcot2_eq_min | rcot2 | 4000 | 3 (timing 3) | 0 % | 2.67 | 2.29 | 2.27 | 3.46 | 3.46 | 0.05 % | 829 | 835 | 2.68 | 1.00 | 0.50 | 0.084 | 0 | feasible |
| rcot2_eq_min | rcot2 | 8000 | 3 (timing 3) | 0 % | 4.85 | 4.15 | 4.06 | 6.34 | 6.34 | 0.09 % | 831 | 835 | 4.87 | 1.00 | 0.50 | 0.17 | 0 | feasible |
| rcot2_eq_min | rcot2 | 24000 | 3 (timing 3) | 0 % | 16.1 | 17.9 | 12.0 | 18.4 | 18.4 | 0.26 % | 746 | 754 | 16.2 | 1.01 | 1.00 | 0.67 | 0 | feasible |
| rcot2_native | rcot2 | 500 | 3 (timing 3) | 0 % | 1.12 | 1.26 | 0.81 | 1.28 | 1.28 | 0.02 % | 829 | 835 | 1.12 | 1.00 | 0.75 | 0.015 | 0 | feasible |
| rcot2_native | rcot2 | 1000 | 3 (timing 3) | 0 % | 1.39 | 1.49 | 0.96 | 1.71 | 1.71 | 0.02 % | 829 | 829 | 1.39 | 1.00 | 0.75 | 0.028 | 0 | feasible |
| rcot2_native | rcot2 | 4000 | 3 (timing 3) | 0 % | 3.03 | 3.40 | 2.30 | 3.40 | 3.40 | 0.05 % | 830 | 835 | 3.05 | 1.00 | 1.00 | 0.10 | 0 | feasible |
| rcot2_native | rcot2 | 8000 | 3 (timing 3) | 0 % | 4.73 | 4.09 | 4.08 | 6.02 | 6.02 | 0.08 % | 754 | 758 | 4.74 | 1.00 | 0.50 | 0.16 | 0 | feasible |
| rcot2_native | rcot2 | 24000 | 3 (timing 3) | 0 % | 16.4 | 16.3 | 15.3 | 17.7 | 17.7 | 0.25 % | 746 | 754 | 16.5 | 1.01 | 1.00 | 0.57 | 0 | feasible |
| shap_dag | shap_dag | 500 | 3 (timing 3) | 0 % | 2.92 | 3.08 | 2.12 | 3.55 | 3.55 | 0.05 % | 830 | 835 | 2.92 | 1.00 | 1.00 | 0.014 | 0 | feasible |
| shap_dag | shap_dag | 1000 | 3 (timing 3) | 0 % | 5.17 | 5.81 | 3.68 | 6.01 | 6.01 | 0.08 % | 774 | 830 | 5.19 | 1.00 | 1.00 | 0.026 | 0 | feasible |
| shap_dag | shap_dag | 4000 | 3 (timing 3) | 0 % | 16.5 | 16.2 | 13.2 | 20.0 | 20.0 | 0.28 % | 745 | 756 | 16.6 | 1.00 | 0.75 | 0.13 | 0 | feasible |
| shap_dag | shap_dag | 8000 | 3 (timing 3) | 0 % | 39.1 | 39.0 | 38.7 | 39.6 | 39.6 | 0.55 % | 712 | 751 | 39.4 | 1.01 | 1.00 | 0.24 | 0 | feasible |
| shap_dag | shap_dag | 24000 | 3 (timing 3) | 0 % | 119 | 119 | 116 | 121 | 121 | 1.69 % | 756 | 757 | 120 | 1.01 | 1.00 | 0.68 | 0 | feasible |
| two_tower | two_tower | 500 | 3 (timing 3) | 0 % | 4.12 | 4.31 | 2.90 | 5.15 | 5.15 | 0.07 % | 754 | 758 | 4.16 | 1.01 | 1.00 | 0.014 | 0 | feasible |
| two_tower | two_tower | 1000 | 3 (timing 3) | 0 % | 5.72 | 6.04 | 4.46 | 6.68 | 6.68 | 0.09 % | 754 | 758 | 5.77 | 1.01 | 1.00 | 0.026 | 0 | feasible |
| two_tower | two_tower | 4000 | 3 (timing 3) | 0 % | 17.7 | 18.4 | 16.0 | 18.8 | 18.8 | 0.26 % | 745 | 751 | 17.9 | 1.01 | 1.00 | 0.12 | 0 | feasible |
| two_tower | two_tower | 8000 | 3 (timing 3) | 0 % | 30.6 | 31.0 | 29.3 | 31.4 | 31.4 | 0.44 % | 712 | 719 | 30.8 | 1.01 | 1.00 | 0.23 | 0 | feasible |
| two_tower | two_tower | 24000 | 3 (timing 3) | 0 % | 108 | 114 | 89.7 | 121 | 121 | 1.67 % | 689 | 714 | 109 | 1.01 | 1.00 | 0.66 | 0 | feasible |

Single-thread check: 0 units with cpu_s > 1.15 x wall_s + .05 s (more than one busy thread).

Contention check: 0 units with wall_s > 1.3 x cpu_s + .05 s (CPU-starved: more busy processes than cores; CPU-s on shared hyperthreads runs slower, so the speed factor must be measured under the same load).
