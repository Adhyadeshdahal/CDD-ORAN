# Experiment C: runtime per method (Experiment C runtime table (DEV: 10 arms + CI + pmrt_nl + cdl))

Generated 2026-10-04T13:02:47Z by `scratchpad/xmethod/exp_c_runtime.py` (xm-exp-c/1). Unit = one (method, dataset): record `cpu_s` = process CPU time of `method.run`, single thread (data generation excluded, shown apart). Costs in **Kaggle reference CPU-s** (`kaggle|Intel(R) Xeon(R) CPU @ 2.20GHz`); other hosts x f (R-51). Peak RSS = `peak_rss_mb` (scope per job on Linux; includes interpreter, imports and dataset), not converted. Budget R-13: 7,200 CPU-s per (method, dataset); T3: max over worlds, regimes, lambdas at n > budget makes n and every larger n infeasible; R-55: tested on cost x f_hi (within the speed factor's error of the budget = over). Rows pool tune + measure seeds, kappas and lambdas (per role and per world / regime in the json).

## Inputs

- `D:\academia\major-project\CDD-ORAN-wt\xm-pmrt\scratchpad\xmethod\results\dev\full\merged.jsonl.gz` (41840 lines, sha256 `d8754daa294774b1`)
- `D:\academia\major-project\CDD-ORAN-wt\xm-pmrt\scratchpad\xmethod\results\dev\ci_c\merged.jsonl.gz` (58480 lines, sha256 `f83c7d1a70f05320`)
- `D:\academia\major-project\CDD-ORAN-wt\xm-pmrt\scratchpad\xmethod\results\dev\pmrt_nl\merged.jsonl.gz` (1440 lines, sha256 `d81e3a6c474844ec`)
- `D:\academia\major-project\CDD-ORAN-wt\xm-pmrt\scratchpad\xmethod\results\dev\cdl\merged.jsonl.gz` (5080 lines, sha256 `057c67309238f308`)
- 106840 lines, 0 duplicate keys (0 on two hosts); specs dev_cdl, dev_ci_c, dev_full, dev_pmrt_nl; commits 0843512, 3a251c6, 44b0a31, 558266f, 9fd845b, a7b6e36, bf7b2da, d3d6d2f, d4fd1d9, e645167, e8c7ced, f0af25b

Hosts (units):

- `kaggle|Intel(R) Xeon(R) CPU @ 2.20GHz`: 85720
- `lightning|Intel(R) Xeon(R) Platinum 8488C`: 10052
- `colab|Intel(R) Xeon(R) CPU @ 2.20GHz`: 8296
- `colab|AMD EPYC 7B12`: 2160
- `kaggle|AMD EPYC 7B12`: 612

## Speed factors applied (cost_ref = cost x f)

| arm | host | f | f_hi | source | cells | host units | cell ratio q25 / q50 / q75 |
|---|---|---|---|---|---|---|---|
| cdl | `colab|Intel(R) Xeon(R) CPU @ 2.20GHz` | 1.048 | 1.08 | r55_calibration_block | - | 9 | - |
| cdl | `kaggle|AMD EPYC 7B12` | 1.411 | 1.56 | observational | 83 | 612 | 1.39 / 1.44 / 1.48 |
| cdl | `lightning|Intel(R) Xeon(R) Platinum 8488C` | 1.824 | 2.17 | observational | 76 | 328 | 1.84 / 1.91 / 1.96 |
| mscr_eq | `colab|AMD EPYC 7B12` | 1.572 | 1.65 | observational | 16 | 60 | 1.58 / 1.60 / 1.63 |
| mscr_eq | `colab|Intel(R) Xeon(R) CPU @ 2.20GHz` | 1.067 | 1.10 | r55_calibration_block (pooled anchors) | - | 51 | - |
| mscr_eq | `lightning|Intel(R) Xeon(R) Platinum 8488C` | 2.842 | 3.44 | observational | 41 | 748 | 2.80 / 2.89 / 2.96 |
| mscr_eq_min | `colab|AMD EPYC 7B12` | 1.531 | 1.69 | observational | 16 | 60 | 1.56 / 1.59 / 1.64 |
| mscr_eq_min | `colab|Intel(R) Xeon(R) CPU @ 2.20GHz` | 1.104 | 1.11 | r55_calibration_block | - | 6 | - |
| mscr_eq_min | `lightning|Intel(R) Xeon(R) Platinum 8488C` | 2.884 | 3.41 | observational | 41 | 748 | 2.84 / 2.91 / 3.04 |
| mscr_native | `colab|AMD EPYC 7B12` | 1.587 | 1.72 | observational | 16 | 60 | 1.60 / 1.61 / 1.63 |
| mscr_native | `colab|Intel(R) Xeon(R) CPU @ 2.20GHz` | 1.067 | 1.10 | r55_calibration_block (pooled anchors) | - | 51 | - |
| mscr_native | `lightning|Intel(R) Xeon(R) Platinum 8488C` | 2.861 | 3.36 | observational | 41 | 748 | 2.79 / 2.87 / 3.01 |
| pcorr_eq | `colab|AMD EPYC 7B12` | 1.525 | 1.71 | observational | 38 | 198 | 1.44 / 1.54 / 1.65 |
| pcorr_eq | `colab|Intel(R) Xeon(R) CPU @ 2.20GHz` | 0.960 | 0.99 | r55_calibration_block | - | 9 | - |
| pcorr_eq | `lightning|Intel(R) Xeon(R) Platinum 8488C` | 2.086 | 3.19 | observational | 41 | 748 | 1.98 / 2.08 / 2.27 |
| pcorr_eq_min | `colab|AMD EPYC 7B12` | 1.477 | 1.75 | observational | 38 | 198 | 1.41 / 1.52 / 1.61 |
| pcorr_eq_min | `colab|Intel(R) Xeon(R) CPU @ 2.20GHz` | 1.067 | 1.10 | r55_calibration_block (pooled anchors) | - | 51 | - |
| pcorr_eq_min | `lightning|Intel(R) Xeon(R) Platinum 8488C` | 2.110 | 3.08 | observational | 41 | 748 | 1.83 / 1.99 / 2.20 |
| pcorr_hac | `colab|AMD EPYC 7B12` | 1.486 | 1.79 | observational | 38 | 198 | 1.50 / 1.56 / 1.64 |
| pcorr_hac | `colab|Intel(R) Xeon(R) CPU @ 2.20GHz` | 1.067 | 1.10 | r55_calibration_block (pooled anchors) | - | 51 | - |
| pcorr_hac | `lightning|Intel(R) Xeon(R) Platinum 8488C` | 2.136 | 2.51 | observational | 41 | 748 | 1.87 / 2.02 / 2.21 |
| pcorr_hac_eq_min | `colab|AMD EPYC 7B12` | 1.482 | 1.78 | observational | 38 | 198 | 1.46 / 1.52 / 1.59 |
| pcorr_hac_eq_min | `colab|Intel(R) Xeon(R) CPU @ 2.20GHz` | 1.067 | 1.10 | r55_calibration_block (pooled anchors) | - | 51 | - |
| pcorr_hac_eq_min | `lightning|Intel(R) Xeon(R) Platinum 8488C` | 2.021 | 2.51 | observational | 41 | 748 | 1.85 / 1.93 / 2.12 |
| pcorr_hac_fb | `colab|AMD EPYC 7B12` | 1.512 | 1.68 | observational | 38 | 198 | 1.46 / 1.54 / 1.61 |
| pcorr_hac_fb | `colab|Intel(R) Xeon(R) CPU @ 2.20GHz` | 1.067 | 1.10 | r55_calibration_block (pooled anchors) | - | 51 | - |
| pcorr_hac_fb | `lightning|Intel(R) Xeon(R) Platinum 8488C` | 1.678 | 1.88 | observational | 41 | 748 | 1.59 / 1.65 / 1.71 |
| pcorr_hac_fb_eq_min | `colab|AMD EPYC 7B12` | 1.512 | 1.65 | observational | 38 | 198 | 1.48 / 1.52 / 1.58 |
| pcorr_hac_fb_eq_min | `colab|Intel(R) Xeon(R) CPU @ 2.20GHz` | 1.067 | 1.10 | r55_calibration_block (pooled anchors) | - | 51 | - |
| pcorr_hac_fb_eq_min | `lightning|Intel(R) Xeon(R) Platinum 8488C` | 1.628 | 1.83 | observational | 41 | 748 | 1.56 / 1.59 / 1.67 |
| pcorr_native | `colab|AMD EPYC 7B12` | 1.484 | 1.97 | observational | 38 | 198 | 1.40 / 1.53 / 1.59 |
| pcorr_native | `colab|Intel(R) Xeon(R) CPU @ 2.20GHz` | 1.067 | 1.10 | r55_calibration_block (pooled anchors) | - | 51 | - |
| pcorr_native | `lightning|Intel(R) Xeon(R) Platinum 8488C` | 2.051 | 3.38 | observational | 41 | 748 | 1.86 / 1.95 / 2.33 |
| rcot2_eq | `colab|AMD EPYC 7B12` | 1.486 | 1.81 | observational | 38 | 198 | 1.41 / 1.48 / 1.58 |
| rcot2_eq | `colab|Intel(R) Xeon(R) CPU @ 2.20GHz` | 0.984 | 1.15 | r55_calibration_block | - | 9 | - |
| rcot2_eq | `lightning|Intel(R) Xeon(R) Platinum 8488C` | 2.054 | 2.57 | observational | 41 | 748 | 1.91 / 2.01 / 2.14 |
| rcot2_eq_min | `colab|AMD EPYC 7B12` | 1.466 | 1.78 | observational | 38 | 198 | 1.42 / 1.48 / 1.57 |
| rcot2_eq_min | `colab|Intel(R) Xeon(R) CPU @ 2.20GHz` | 1.067 | 1.10 | r55_calibration_block (pooled anchors) | - | 51 | - |
| rcot2_eq_min | `lightning|Intel(R) Xeon(R) Platinum 8488C` | 2.046 | 2.73 | observational | 41 | 748 | 1.91 / 2.00 / 2.12 |
| rcot2_native | `colab|AMD EPYC 7B12` | 1.486 | 1.71 | observational | 38 | 198 | 1.42 / 1.49 / 1.56 |
| rcot2_native | `colab|Intel(R) Xeon(R) CPU @ 2.20GHz` | 1.067 | 1.10 | r55_calibration_block (pooled anchors) | - | 51 | - |
| rcot2_native | `lightning|Intel(R) Xeon(R) Platinum 8488C` | 2.038 | 2.79 | observational | 41 | 748 | 1.91 / 1.99 / 2.10 |

**observational** = PROVISIONAL within-cell estimate (same cell, different seeds on each host), not the T3 factor (paired calibration units, `calib`). Replace with `--factors` before the freeze.

## Summary: median / max CPU-s per unit (Kaggle ref.) by n; T3 status

| arm | n 500 | n 1000 | n 4000 | n 8000 | n 24000 | T3 |
|---|---|---|---|---|---|---|
| cdl | 20.1 / 56.5+ | 50.4 / 122+ | 147 / 439+ | 271 / 779+ | 575 / 1,654+ | feasible n <= 24000 |
| corr | 0.064 / 0.14 | 0.082 / 0.17 | 0.067 / 0.17 | 0.068 / 0.19 | 0.073 / 0.41 | feasible n <= 24000 |
| granger_eq | 0.25 / 0.27 | 0.34 / 0.39 | 0.87 / 1.03 | 1.61 / 2.32 | 5.79 / 7.44 | feasible n <= 24000 |
| granger_native | 0.10 / 0.11 | 0.11 / 0.13 | 0.14 / 0.16 | 0.19 / 0.21 | 0.39 / 0.43 | feasible n <= 24000 |
| mscr_eq | 9.26 / 93.6+ | 58.0 / 202+ | - | - | - | feasible n <= 1000; no DEV cost at n 4000, 8000, 24000 |
| mscr_eq_min | 3.72 / 363+ | 26.9 / 790+ | - | - | - | feasible n <= 1000; no DEV cost at n 4000, 8000, 24000 |
| mscr_native | 3.21 / 34.0+ | 26.2 / 72.6+ | - | - | - | feasible n <= 1000; no DEV cost at n 4000, 8000, 24000 |
| notears | 0.061 / 2.10 | 0.29 / 2.27 | 0.069 / 4.68 | 0.076 / 7.28 | 0.11 / 24.3 | feasible n <= 24000 |
| pc_eq | 0.15 / 15.4 | 0.49 / 19.6 | 0.19 / 33.6 | 0.21 / 42.8 | 0.36 / 79.5 | feasible n <= 24000 |
| pc_native | 0.080 / 1.04 | 0.18 / 1.77 | 0.092 / 3.95 | 0.10 / 6.96 | 0.15 / 18.9 | feasible n <= 24000 |
| pcorr_eq | 0.020 / 0.18+ | 0.050 / 0.36+ | 0.063 / 0.75+ | 0.10 / 1.81+ | 0.28 / 6.94+ | feasible n <= 24000 |
| pcorr_eq_min | 0.014 / 0.20+ | 0.027 / 0.29+ | 0.030 / 0.71+ | 0.042 / 1.57+ | 0.10 / 5.08+ | feasible n <= 24000 |
| pcorr_hac | 0.046 / 0.33+ | 0.12 / 0.54+ | 0.061 / 1.14+ | 0.069 / 2.09+ | 0.12 / 12.3+ | feasible n <= 24000 |
| pcorr_hac_eq_min | 0.043 / 0.33+ | 0.12 / 0.50+ | 0.061 / 1.43+ | 0.069 / 2.25+ | 0.12 / 8.89+ | feasible n <= 24000 |
| pcorr_hac_fb | 0.94 / 1.79+ | 1.04 / 1.78+ | 1.02 / 2.37+ | 1.05 / 3.31+ | 1.05 / 8.53+ | feasible n <= 24000 |
| pcorr_hac_fb_eq_min | 0.92 / 1.62+ | 1.02 / 1.69+ | 1.02 / 2.35+ | 1.03 / 3.07+ | 1.05 / 7.93+ | feasible n <= 24000 |
| pcorr_native | 0.012 / 0.035+ | 0.019 / 0.077+ | 0.025 / 0.14+ | 0.034 / 0.23+ | 0.084 / 0.91+ | feasible n <= 24000 |
| pmrt_eq | 0.78 / 1.35 | 1.21 / 2.26 | 4.42 / 8.33 | 9.31 / 18.4 | 27.2 / 62.5 | feasible n <= 24000 |
| pmrt_nl_eq | 30.1 / 64.1 | 44.8 / 97.9 | 115 / 290 | - | - | feasible n <= 4000; no DEV cost at n 8000, 24000 |
| pmrt_r3 | 0.77 / 1.35 | 1.17 / 2.25 | 4.31 / 8.53 | 9.14 / 17.3 | 26.8 / 59.9 | feasible n <= 24000 |
| rcot2_eq | 0.088 / 1.69+ | 0.71 / 2.36+ | 0.26 / 5.84+ | 0.40 / 9.90+ | 1.03 / 27.0+ | feasible n <= 24000 |
| rcot2_eq_min | 0.085 / 1.59+ | 0.69 / 2.56+ | 0.25 / 5.27+ | 0.37 / 7.94+ | 0.94 / 23.2+ | feasible n <= 24000 |
| rcot2_native | 0.083 / 1.56+ | 0.68 / 2.18+ | 0.24 / 5.28+ | 0.36 / 8.20+ | 0.92 / 23.3+ | feasible n <= 24000 |
| shap_dag | 0.58 / 4.82 | 3.80 / 7.70 | 3.96 / 25.2 | 8.08 / 48.9 | 25.0 / 148 | feasible n <= 24000 |
| two_tower | 4.10 / 12.9 | 6.99 / 11.4 | 7.62 / 29.8 | 12.1 / 54.0 | 29.9 / 156 | feasible n <= 24000 |

`-` = no record at that n (T3: a cost pilot decides if the n is in the arm's grid); * = some unit unconverted; + = some unit converted with a provisional (observational) factor. Pooled over worlds, regimes, lambdas, kappas (the kappa sweep is at n 1000) and roles (tune + measure).

## Full table

| arm | method | n | units (tune / measure) | non-ref | CPU-s mean | median | min | max | max x f_hi | max / budget | RSS med MB | RSS max MB | wall-s mean | wall / CPU | procs / vCPU | gen CPU-s | multi-thr | T3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cdl | cdl | 500 | 1080 (360 / 720) | 37 % | 25.0 | 20.1 | 8.95 | 56.5 | 59.1 | 0.79 % | 294 | 318 | 47.9 | 2.03 | 0.25 | 0.075 | 0 | feasible |
| cdl | cdl | 1000 | 1480 (560 / 920) | 35 % | 50.9 | 50.4 | 16.3 | 122 | 122 | 1.69 % | 294 | 320 | 97.5 | 2.02 | 0.25 | 0.12 | 0 | feasible |
| cdl | cdl | 4000 | 1080 (360 / 720) | 38 % | 183 | 147 | 56.9 | 439 | 449 | 6.10 % | 295 | 322 | 347 | 2.02 | 0.25 | 0.58 | 0 | feasible |
| cdl | cdl | 8000 | 720 (360 / 360) | 35 % | 338 | 271 | 110 | 779 | 895 | 10.81 % | 298 | 326 | 638 | 2.02 | 0.25 | 1.05 | 0 | feasible |
| cdl | cdl | 24000 | 720 (360 / 360) | 28 % | 654 | 575 | 216 | 1,654 | 1,879 | 22.97 % | 312 | 390 | 1,217 | 2.02 | 0.25 | 3.18 | 0 | feasible |
| corr | corr | 500 | 1080 (360 / 720) | 0 % | 0.079 | 0.064 | 0.054 | 0.14 | 0.14 | 0.00 % | 344 | 344 | 0.084 | 1.05 | - | 0.090 | 0 | feasible |
| corr | corr | 1000 | 1480 (560 / 920) | 0 % | 0.083 | 0.082 | 0.055 | 0.17 | 0.17 | 0.00 % | 344 | 345 | 0.088 | 1.05 | - | 0.14 | 0 | feasible |
| corr | corr | 4000 | 1080 (360 / 720) | 0 % | 0.082 | 0.067 | 0.055 | 0.17 | 0.17 | 0.00 % | 346 | 368 | 0.086 | 1.05 | - | 0.68 | 0 | feasible |
| corr | corr | 8000 | 720 (360 / 360) | 0 % | 0.086 | 0.068 | 0.056 | 0.19 | 0.19 | 0.00 % | 351 | 395 | 0.091 | 1.05 | - | 1.25 | 0 | feasible |
| corr | corr | 24000 | 720 (360 / 360) | 0 % | 0.11 | 0.073 | 0.057 | 0.41 | 0.41 | 0.01 % | 371 | 538 | 0.11 | 1.05 | - | 3.74 | 0 | feasible |
| granger_eq | granger | 500 | 120 (40 / 80) | 0 % | 0.24 | 0.25 | 0.20 | 0.27 | 0.27 | 0.00 % | 341 | 342 | 0.25 | 1.02 | - | 0.014 | 0 | feasible |
| granger_eq | granger | 1000 | 200 (80 / 120) | 0 % | 0.33 | 0.34 | 0.26 | 0.39 | 0.39 | 0.01 % | 345 | 346 | 0.34 | 1.01 | - | 0.025 | 0 | feasible |
| granger_eq | granger | 4000 | 120 (40 / 80) | 0 % | 0.85 | 0.87 | 0.67 | 1.03 | 1.03 | 0.01 % | 349 | 350 | 0.86 | 1.01 | - | 0.086 | 0 | feasible |
| granger_eq | granger | 8000 | 80 (40 / 40) | 0 % | 1.70 | 1.61 | 1.28 | 2.32 | 2.32 | 0.03 % | 355 | 359 | 1.73 | 1.01 | - | 0.17 | 0 | feasible |
| granger_eq | granger | 24000 | 80 (40 / 40) | 0 % | 5.72 | 5.79 | 4.06 | 7.44 | 7.44 | 0.10 % | 383 | 394 | 5.78 | 1.01 | - | 0.49 | 0 | feasible |
| granger_native | granger | 500 | 120 (40 / 80) | 0 % | 0.099 | 0.10 | 0.086 | 0.11 | 0.11 | 0.00 % | 341 | 341 | 0.10 | 1.03 | - | 0.014 | 0 | feasible |
| granger_native | granger | 1000 | 200 (80 / 120) | 0 % | 0.10 | 0.11 | 0.091 | 0.13 | 0.13 | 0.00 % | 345 | 346 | 0.11 | 1.03 | - | 0.025 | 0 | feasible |
| granger_native | granger | 4000 | 120 (40 / 80) | 0 % | 0.14 | 0.14 | 0.12 | 0.16 | 0.16 | 0.00 % | 346 | 348 | 0.14 | 1.03 | - | 0.086 | 0 | feasible |
| granger_native | granger | 8000 | 80 (40 / 40) | 0 % | 0.19 | 0.19 | 0.17 | 0.21 | 0.21 | 0.00 % | 352 | 353 | 0.19 | 1.02 | - | 0.17 | 0 | feasible |
| granger_native | granger | 24000 | 80 (40 / 40) | 0 % | 0.39 | 0.39 | 0.35 | 0.43 | 0.43 | 0.01 % | 372 | 374 | 0.40 | 1.01 | - | 0.49 | 0 | feasible |
| mscr_eq | mscr | 500 | 1080 (360 / 720) | 57 % | 23.8 | 9.26 | 2.33 | 93.6 | 107 | 1.30 % | 217 | 345 | 61.1 | 3.62 | - | 0.056 | 0 | feasible |
| mscr_eq | mscr | 1000 | 1480 (560 / 920) | 35 % | 61.7 | 58.0 | 5.09 | 202 | 241 | 2.81 % | 247 | 348 | 185 | 3.58 | - | 0.12 | 0 | feasible |
| mscr_eq_min | mscr | 500 | 1080 (360 / 720) | 57 % | 48.6 | 3.72 | 0.70 | 363 | 397 | 5.04 % | 215 | 258 | 120 | 3.63 | - | 0.056 | 0 | feasible |
| mscr_eq_min | mscr | 1000 | 1480 (560 / 920) | 35 % | 150 | 26.9 | 2.19 | 790 | 881 | 10.97 % | 220 | 260 | 440 | 3.59 | - | 0.12 | 0 | feasible |
| mscr_native | mscr | 500 | 1080 (360 / 720) | 57 % | 9.48 | 3.21 | 0.68 | 34.0 | 38.1 | 0.47 % | 215 | 254 | 24.4 | 3.62 | - | 0.056 | 0 | feasible |
| mscr_native | mscr | 1000 | 1480 (560 / 920) | 35 % | 24.1 | 26.2 | 1.82 | 72.6 | 83.3 | 1.01 % | 218 | 257 | 71.2 | 3.59 | - | 0.12 | 0 | feasible |
| notears | notears | 500 | 1080 (360 / 720) | 0 % | 0.28 | 0.061 | 0.041 | 2.10 | 2.10 | 0.03 % | 345 | 346 | 0.29 | 1.05 | - | 0.090 | 0 | feasible |
| notears | notears | 1000 | 1480 (560 / 920) | 0 % | 0.40 | 0.29 | 0.049 | 2.27 | 2.27 | 0.03 % | 345 | 347 | 0.40 | 1.04 | - | 0.14 | 0 | feasible |
| notears | notears | 4000 | 1080 (360 / 720) | 0 % | 0.61 | 0.069 | 0.051 | 4.68 | 4.68 | 0.06 % | 347 | 369 | 0.62 | 1.04 | - | 0.68 | 0 | feasible |
| notears | notears | 8000 | 720 (360 / 360) | 0 % | 0.94 | 0.076 | 0.051 | 7.28 | 7.28 | 0.10 % | 353 | 396 | 0.95 | 1.04 | - | 1.25 | 0 | feasible |
| notears | notears | 24000 | 720 (360 / 360) | 0 % | 2.62 | 0.11 | 0.058 | 24.3 | 24.3 | 0.34 % | 374 | 539 | 2.65 | 1.03 | - | 3.74 | 0 | feasible |
| pc_eq | pc | 500 | 1080 (360 / 720) | 0 % | 1.67 | 0.15 | 0.074 | 15.4 | 15.4 | 0.21 % | 345 | 360 | 1.70 | 1.02 | - | 0.090 | 0 | feasible |
| pc_eq | pc | 1000 | 1480 (560 / 920) | 0 % | 2.83 | 0.49 | 0.076 | 19.6 | 19.6 | 0.27 % | 346 | 365 | 2.86 | 1.02 | - | 0.14 | 0 | feasible |
| pc_eq | pc | 4000 | 1080 (360 / 720) | 0 % | 3.07 | 0.19 | 0.087 | 33.6 | 33.6 | 0.47 % | 353 | 386 | 3.10 | 1.02 | - | 0.68 | 0 | feasible |
| pc_eq | pc | 8000 | 720 (360 / 360) | 0 % | 3.31 | 0.21 | 0.10 | 42.8 | 42.8 | 0.59 % | 361 | 404 | 3.35 | 1.02 | - | 1.25 | 0 | feasible |
| pc_eq | pc | 24000 | 720 (360 / 360) | 0 % | 6.86 | 0.36 | 0.18 | 79.5 | 79.5 | 1.10 % | 398 | 539 | 6.93 | 1.01 | - | 3.74 | 0 | feasible |
| pc_native | pc | 500 | 1080 (360 / 720) | 0 % | 0.21 | 0.080 | 0.058 | 1.04 | 1.04 | 0.01 % | 345 | 346 | 0.21 | 1.04 | - | 0.090 | 0 | feasible |
| pc_native | pc | 1000 | 1480 (560 / 920) | 0 % | 0.32 | 0.18 | 0.058 | 1.77 | 1.77 | 0.02 % | 346 | 347 | 0.33 | 1.03 | - | 0.14 | 0 | feasible |
| pc_native | pc | 4000 | 1080 (360 / 720) | 0 % | 0.44 | 0.092 | 0.062 | 3.95 | 3.95 | 0.05 % | 348 | 370 | 0.44 | 1.03 | - | 0.68 | 0 | feasible |
| pc_native | pc | 8000 | 720 (360 / 360) | 0 % | 0.56 | 0.10 | 0.067 | 6.96 | 6.96 | 0.10 % | 353 | 397 | 0.57 | 1.03 | - | 1.25 | 0 | feasible |
| pc_native | pc | 24000 | 720 (360 / 360) | 0 % | 1.20 | 0.15 | 0.092 | 18.9 | 18.9 | 0.26 % | 374 | 539 | 1.21 | 1.02 | - | 3.74 | 0 | feasible |
| pcorr_eq | pcorr | 500 | 1080 (360 / 720) | 57 % | 0.039 | 0.020 | 0.009 | 0.18 | 0.25 | 0.00 % | 203 | 211 | 0.19 | 7.98 | - | 0.056 | 0 | feasible |
| pcorr_eq | pcorr | 1000 | 1480 (560 / 920) | 35 % | 0.064 | 0.050 | 0.010 | 0.36 | 0.54 | 0.00 % | 210 | 220 | 0.31 | 6.31 | - | 0.12 | 0 | feasible |
| pcorr_eq | pcorr | 4000 | 1080 (360 / 720) | 20 % | 0.16 | 0.063 | 0.023 | 0.75 | 0.81 | 0.01 % | 216 | 237 | 0.69 | 4.92 | - | 0.62 | 0 | feasible |
| pcorr_eq | pcorr | 8000 | 720 (360 / 360) | 17 % | 0.29 | 0.10 | 0.038 | 1.81 | 1.81 | 0.03 % | 222 | 254 | 1.09 | 4.34 | - | 1.12 | 0 | feasible |
| pcorr_eq | pcorr | 24000 | 720 (360 / 360) | 18 % | 1.00 | 0.28 | 0.090 | 6.94 | 6.94 | 0.10 % | 255 | 332 | 3.70 | 3.92 | - | 3.37 | 0 | feasible |
| pcorr_eq_min | pcorr | 500 | 1080 (360 / 720) | 57 % | 0.031 | 0.014 | 0.006 | 0.20 | 0.22 | 0.00 % | 203 | 211 | 0.16 | 9.94 | - | 0.056 | 0 | feasible |
| pcorr_eq_min | pcorr | 1000 | 1480 (560 / 920) | 35 % | 0.054 | 0.027 | 0.008 | 0.29 | 0.42 | 0.00 % | 210 | 220 | 0.27 | 7.58 | - | 0.12 | 0 | feasible |
| pcorr_eq_min | pcorr | 4000 | 1080 (360 / 720) | 20 % | 0.11 | 0.030 | 0.014 | 0.71 | 0.84 | 0.01 % | 213 | 237 | 0.54 | 6.69 | - | 0.62 | 0 | feasible |
| pcorr_eq_min | pcorr | 8000 | 720 (360 / 360) | 17 % | 0.18 | 0.042 | 0.022 | 1.57 | 1.57 | 0.02 % | 218 | 254 | 0.73 | 5.47 | - | 1.12 | 0 | feasible |
| pcorr_eq_min | pcorr | 24000 | 720 (360 / 360) | 18 % | 0.57 | 0.10 | 0.046 | 5.08 | 5.08 | 0.07 % | 241 | 332 | 2.25 | 4.46 | - | 3.37 | 0 | feasible |
| pcorr_hac | pcorr_hac | 500 | 1080 (360 / 720) | 57 % | 0.089 | 0.046 | 0.027 | 0.33 | 0.37 | 0.00 % | 208 | 224 | 0.32 | 5.56 | - | 0.056 | 0 | feasible |
| pcorr_hac | pcorr_hac | 1000 | 1480 (560 / 920) | 35 % | 0.13 | 0.12 | 0.027 | 0.54 | 0.64 | 0.01 % | 219 | 231 | 0.52 | 5.11 | - | 0.12 | 0 | feasible |
| pcorr_hac | pcorr_hac | 4000 | 1080 (360 / 720) | 20 % | 0.24 | 0.061 | 0.039 | 1.14 | 1.27 | 0.02 % | 222 | 247 | 0.97 | 4.96 | - | 0.62 | 0 | feasible |
| pcorr_hac | pcorr_hac | 8000 | 720 (360 / 360) | 17 % | 0.40 | 0.069 | 0.047 | 2.09 | 2.09 | 0.03 % | 226 | 264 | 1.46 | 4.60 | - | 1.12 | 0 | feasible |
| pcorr_hac | pcorr_hac | 24000 | 720 (360 / 360) | 18 % | 1.37 | 0.12 | 0.075 | 12.3 | 12.3 | 0.17 % | 243 | 343 | 4.92 | 4.26 | - | 3.37 | 0 | feasible |
| pcorr_hac_eq_min | pcorr_hac | 500 | 1080 (360 / 720) | 57 % | 0.088 | 0.043 | 0.024 | 0.33 | 0.38 | 0.00 % | 208 | 224 | 0.33 | 5.53 | - | 0.056 | 0 | feasible |
| pcorr_hac_eq_min | pcorr_hac | 1000 | 1480 (560 / 920) | 35 % | 0.13 | 0.12 | 0.024 | 0.50 | 0.62 | 0.01 % | 219 | 231 | 0.50 | 5.08 | - | 0.12 | 0 | feasible |
| pcorr_hac_eq_min | pcorr_hac | 4000 | 1080 (360 / 720) | 20 % | 0.24 | 0.061 | 0.039 | 1.43 | 1.43 | 0.02 % | 222 | 247 | 0.99 | 4.94 | - | 0.62 | 0 | feasible |
| pcorr_hac_eq_min | pcorr_hac | 8000 | 720 (360 / 360) | 17 % | 0.41 | 0.069 | 0.048 | 2.25 | 2.25 | 0.03 % | 226 | 264 | 1.50 | 4.61 | - | 1.12 | 0 | feasible |
| pcorr_hac_eq_min | pcorr_hac | 24000 | 720 (360 / 360) | 18 % | 1.39 | 0.12 | 0.073 | 8.89 | 8.89 | 0.12 % | 245 | 343 | 5.05 | 4.23 | - | 3.37 | 0 | feasible |
| pcorr_hac_fb | pcorr_hac | 500 | 1080 (360 / 720) | 57 % | 1.00 | 0.94 | 0.71 | 1.79 | 1.80 | 0.02 % | 328 | 346 | 3.08 | 3.70 | - | 0.056 | 0 | feasible |
| pcorr_hac_fb | pcorr_hac | 1000 | 1480 (560 / 920) | 35 % | 1.08 | 1.04 | 0.80 | 1.78 | 2.00 | 0.02 % | 340 | 352 | 3.73 | 3.69 | - | 0.12 | 0 | feasible |
| pcorr_hac_fb | pcorr_hac | 4000 | 1080 (360 / 720) | 20 % | 1.15 | 1.02 | 0.82 | 2.37 | 2.37 | 0.03 % | 344 | 371 | 4.09 | 3.65 | - | 0.62 | 0 | feasible |
| pcorr_hac_fb | pcorr_hac | 8000 | 720 (360 / 360) | 17 % | 1.32 | 1.05 | 0.82 | 3.31 | 3.31 | 0.05 % | 347 | 381 | 4.45 | 3.59 | - | 1.12 | 0 | feasible |
| pcorr_hac_fb | pcorr_hac | 24000 | 720 (360 / 360) | 18 % | 2.14 | 1.05 | 0.68 | 8.53 | 8.53 | 0.12 % | 362 | 421 | 7.32 | 3.62 | - | 3.37 | 0 | feasible |
| pcorr_hac_fb_eq_min | pcorr_hac | 500 | 1080 (360 / 720) | 57 % | 0.97 | 0.92 | 0.71 | 1.62 | 1.68 | 0.02 % | 328 | 346 | 3.04 | 3.71 | - | 0.056 | 0 | feasible |
| pcorr_hac_fb_eq_min | pcorr_hac | 1000 | 1480 (560 / 920) | 35 % | 1.05 | 1.02 | 0.80 | 1.69 | 1.90 | 0.02 % | 340 | 352 | 3.65 | 3.70 | - | 0.12 | 0 | feasible |
| pcorr_hac_fb_eq_min | pcorr_hac | 4000 | 1080 (360 / 720) | 20 % | 1.15 | 1.02 | 0.82 | 2.35 | 2.35 | 0.03 % | 344 | 371 | 4.07 | 3.65 | - | 0.62 | 0 | feasible |
| pcorr_hac_fb_eq_min | pcorr_hac | 8000 | 720 (360 / 360) | 17 % | 1.31 | 1.03 | 0.82 | 3.07 | 3.07 | 0.04 % | 348 | 381 | 4.42 | 3.59 | - | 1.12 | 0 | feasible |
| pcorr_hac_fb_eq_min | pcorr_hac | 24000 | 720 (360 / 360) | 18 % | 2.15 | 1.05 | 0.77 | 7.93 | 7.93 | 0.11 % | 363 | 421 | 7.30 | 3.59 | - | 3.37 | 0 | feasible |
| pcorr_native | pcorr | 500 | 1080 (360 / 720) | 57 % | 0.014 | 0.012 | 0.006 | 0.035 | 0.051 | 0.00 % | 203 | 211 | 0.12 | 12.5 | - | 0.056 | 0 | feasible |
| pcorr_native | pcorr | 1000 | 1480 (560 / 920) | 35 % | 0.020 | 0.019 | 0.005 | 0.077 | 0.13 | 0.00 % | 210 | 220 | 0.16 | 10.0 | - | 0.12 | 0 | feasible |
| pcorr_native | pcorr | 4000 | 1080 (360 / 720) | 20 % | 0.041 | 0.025 | 0.010 | 0.14 | 0.18 | 0.00 % | 212 | 237 | 0.29 | 7.89 | - | 0.62 | 0 | feasible |
| pcorr_native | pcorr | 8000 | 720 (360 / 360) | 17 % | 0.069 | 0.034 | 0.016 | 0.23 | 0.23 | 0.00 % | 217 | 254 | 0.40 | 6.03 | - | 1.12 | 0 | feasible |
| pcorr_native | pcorr | 24000 | 720 (360 / 360) | 18 % | 0.21 | 0.084 | 0.036 | 0.91 | 0.91 | 0.01 % | 236 | 332 | 1.01 | 4.90 | - | 3.37 | 0 | feasible |
| pmrt_eq | pmrt_core | 500 | 1080 (360 / 720) | 0 % | 0.62 | 0.78 | 0.057 | 1.35 | 1.35 | 0.02 % | 348 | 356 | 0.63 | 1.00 | - | 0.090 | 0 | feasible |
| pmrt_eq | pmrt_core | 1000 | 1480 (560 / 920) | 0 % | 0.98 | 1.21 | 0.080 | 2.26 | 2.26 | 0.03 % | 356 | 373 | 0.99 | 1.00 | - | 0.14 | 0 | feasible |
| pmrt_eq | pmrt_core | 4000 | 1080 (360 / 720) | 0 % | 3.72 | 4.42 | 0.23 | 8.33 | 8.33 | 0.12 % | 405 | 486 | 3.76 | 1.01 | - | 0.68 | 0 | feasible |
| pmrt_eq | pmrt_core | 8000 | 720 (360 / 360) | 0 % | 7.46 | 9.31 | 0.44 | 18.4 | 18.4 | 0.26 % | 472 | 635 | 7.53 | 1.01 | - | 1.25 | 0 | feasible |
| pmrt_eq | pmrt_core | 24000 | 720 (360 / 360) | 0 % | 22.0 | 27.2 | 0.51 | 62.5 | 62.5 | 0.87 % | 501 | 778 | 22.2 | 1.01 | - | 3.74 | 0 | feasible |
| pmrt_nl_eq | pmrt_core | 500 | 480 (160 / 320) | 0 % | 35.1 | 30.1 | 23.7 | 64.1 | 64.1 | 0.89 % | 183 | 190 | 35.5 | 1.01 | - | 0.025 | 0 | feasible |
| pmrt_nl_eq | pmrt_core | 1000 | 480 (160 / 320) | 0 % | 53.4 | 44.8 | 35.5 | 97.9 | 97.9 | 1.36 % | 217 | 228 | 54.0 | 1.01 | - | 0.047 | 0 | feasible |
| pmrt_nl_eq | pmrt_core | 4000 | 480 (160 / 320) | 0 % | 144 | 115 | 52.7 | 290 | 290 | 4.03 % | 258 | 279 | 145 | 1.01 | - | 0.16 | 0 | feasible |
| pmrt_r3 | pmrt_core | 500 | 1080 (360 / 720) | 0 % | 0.61 | 0.77 | 0.055 | 1.35 | 1.35 | 0.02 % | 348 | 356 | 0.62 | 1.00 | - | 0.090 | 0 | feasible |
| pmrt_r3 | pmrt_core | 1000 | 1480 (560 / 920) | 0 % | 0.95 | 1.17 | 0.073 | 2.25 | 2.25 | 0.03 % | 356 | 372 | 0.96 | 1.00 | - | 0.14 | 0 | feasible |
| pmrt_r3 | pmrt_core | 4000 | 1080 (360 / 720) | 0 % | 3.64 | 4.31 | 0.22 | 8.53 | 8.53 | 0.12 % | 403 | 486 | 3.68 | 1.01 | - | 0.68 | 0 | feasible |
| pmrt_r3 | pmrt_core | 8000 | 720 (360 / 360) | 0 % | 7.30 | 9.14 | 0.40 | 17.3 | 17.3 | 0.24 % | 470 | 635 | 7.37 | 1.01 | - | 1.25 | 0 | feasible |
| pmrt_r3 | pmrt_core | 24000 | 720 (360 / 360) | 0 % | 21.4 | 26.8 | 0.45 | 59.9 | 59.9 | 0.83 % | 491 | 778 | 21.6 | 1.01 | - | 3.74 | 0 | feasible |
| rcot2_eq | rcot2 | 500 | 1080 (360 / 720) | 57 % | 0.42 | 0.088 | 0.053 | 1.69 | 2.12 | 0.02 % | 206 | 215 | 1.28 | 4.50 | - | 0.056 | 0 | feasible |
| rcot2_eq | rcot2 | 1000 | 1480 (560 / 920) | 35 % | 0.64 | 0.71 | 0.051 | 2.36 | 2.71 | 0.03 % | 213 | 224 | 2.09 | 4.19 | - | 0.12 | 0 | feasible |
| rcot2_eq | rcot2 | 4000 | 1080 (360 / 720) | 20 % | 1.42 | 0.26 | 0.17 | 5.84 | 6.17 | 0.08 % | 223 | 240 | 5.04 | 3.85 | - | 0.62 | 0 | feasible |
| rcot2_eq | rcot2 | 8000 | 720 (360 / 360) | 17 % | 2.32 | 0.40 | 0.27 | 9.90 | 9.90 | 0.14 % | 235 | 257 | 7.64 | 3.62 | - | 1.12 | 0 | feasible |
| rcot2_eq | rcot2 | 24000 | 720 (360 / 360) | 18 % | 6.39 | 1.03 | 0.66 | 27.0 | 27.0 | 0.37 % | 283 | 335 | 21.5 | 3.55 | - | 3.37 | 0 | feasible |
| rcot2_eq_min | rcot2 | 500 | 1080 (360 / 720) | 57 % | 0.41 | 0.085 | 0.054 | 1.59 | 2.12 | 0.02 % | 206 | 215 | 1.23 | 4.51 | - | 0.056 | 0 | feasible |
| rcot2_eq_min | rcot2 | 1000 | 1480 (560 / 920) | 35 % | 0.61 | 0.69 | 0.050 | 2.56 | 2.75 | 0.04 % | 213 | 224 | 2.00 | 4.25 | - | 0.12 | 0 | feasible |
| rcot2_eq_min | rcot2 | 4000 | 1080 (360 / 720) | 20 % | 1.32 | 0.25 | 0.16 | 5.27 | 5.27 | 0.07 % | 221 | 240 | 4.66 | 3.88 | - | 0.62 | 0 | feasible |
| rcot2_eq_min | rcot2 | 8000 | 720 (360 / 360) | 17 % | 2.13 | 0.37 | 0.26 | 7.94 | 7.94 | 0.11 % | 231 | 257 | 6.99 | 3.64 | - | 1.12 | 0 | feasible |
| rcot2_eq_min | rcot2 | 24000 | 720 (360 / 360) | 18 % | 5.68 | 0.94 | 0.51 | 23.2 | 23.2 | 0.32 % | 276 | 335 | 18.9 | 3.55 | - | 3.37 | 0 | feasible |
| rcot2_native | rcot2 | 500 | 1080 (360 / 720) | 57 % | 0.41 | 0.083 | 0.052 | 1.56 | 2.04 | 0.02 % | 206 | 215 | 1.22 | 4.52 | - | 0.056 | 0 | feasible |
| rcot2_native | rcot2 | 1000 | 1480 (560 / 920) | 35 % | 0.61 | 0.68 | 0.046 | 2.18 | 2.77 | 0.03 % | 213 | 224 | 1.99 | 4.30 | - | 0.12 | 0 | feasible |
| rcot2_native | rcot2 | 4000 | 1080 (360 / 720) | 20 % | 1.35 | 0.24 | 0.15 | 5.28 | 5.28 | 0.07 % | 220 | 240 | 4.78 | 3.91 | - | 0.62 | 0 | feasible |
| rcot2_native | rcot2 | 8000 | 720 (360 / 360) | 17 % | 2.16 | 0.36 | 0.26 | 8.20 | 8.20 | 0.11 % | 229 | 257 | 7.07 | 3.63 | - | 1.12 | 0 | feasible |
| rcot2_native | rcot2 | 24000 | 720 (360 / 360) | 18 % | 5.68 | 0.92 | 0.51 | 23.3 | 23.3 | 0.32 % | 270 | 335 | 18.7 | 3.55 | - | 3.37 | 0 | feasible |
| shap_dag | shap_dag | 500 | 1080 (360 / 720) | 0 % | 1.60 | 0.58 | 0.46 | 4.82 | 4.82 | 0.07 % | 365 | 369 | 1.62 | 1.01 | - | 0.090 | 0 | feasible |
| shap_dag | shap_dag | 1000 | 1480 (560 / 920) | 0 % | 3.07 | 3.80 | 0.75 | 7.70 | 7.70 | 0.11 % | 366 | 369 | 3.10 | 1.01 | - | 0.14 | 0 | feasible |
| shap_dag | shap_dag | 4000 | 1080 (360 / 720) | 0 % | 9.90 | 3.96 | 3.02 | 25.2 | 25.2 | 0.35 % | 369 | 387 | 10.00 | 1.01 | - | 0.68 | 0 | feasible |
| shap_dag | shap_dag | 8000 | 720 (360 / 360) | 0 % | 20.1 | 8.08 | 6.35 | 48.9 | 48.9 | 0.68 % | 371 | 414 | 20.3 | 1.01 | - | 1.25 | 0 | feasible |
| shap_dag | shap_dag | 24000 | 720 (360 / 360) | 0 % | 62.2 | 25.0 | 21.0 | 148 | 148 | 2.06 % | 391 | 556 | 62.7 | 1.01 | - | 3.74 | 0 | feasible |
| two_tower | two_tower | 500 | 1080 (360 / 720) | 0 % | 5.21 | 4.10 | 3.59 | 12.9 | 12.9 | 0.18 % | 426 | 446 | 5.28 | 1.01 | - | 0.090 | 0 | feasible |
| two_tower | two_tower | 1000 | 1480 (560 / 920) | 0 % | 6.41 | 6.99 | 3.88 | 11.4 | 11.4 | 0.16 % | 429 | 436 | 6.47 | 1.01 | - | 0.14 | 0 | feasible |
| two_tower | two_tower | 4000 | 1080 (360 / 720) | 0 % | 12.7 | 7.62 | 6.00 | 29.8 | 29.8 | 0.41 % | 441 | 471 | 12.9 | 1.01 | - | 0.68 | 0 | feasible |
| two_tower | two_tower | 8000 | 720 (360 / 360) | 0 % | 22.1 | 12.1 | 8.60 | 54.0 | 54.0 | 0.75 % | 459 | 523 | 22.3 | 1.01 | - | 1.25 | 0 | feasible |
| two_tower | two_tower | 24000 | 720 (360 / 360) | 0 % | 59.4 | 29.9 | 19.9 | 156 | 156 | 2.17 % | 537 | 723 | 60.0 | 1.01 | - | 3.74 | 0 | feasible |

Single-thread check: 0 units with cpu_s > 1.15 x wall_s + .05 s (more than one busy thread).

Contention check: 62411 units with wall_s > 1.3 x cpu_s + .05 s (CPU-starved: more busy processes than cores; CPU-s on shared hyperthreads runs slower, so the speed factor must be measured under the same load); rows: cdl n 500 (1013), cdl n 1000 (1391), cdl n 4000 (1008), cdl n 8000 (676), cdl n 24000 (668), corr n 24000 (1), mscr_eq n 500 (1080), mscr_eq n 1000 (1480), mscr_eq_min n 500 (1080), mscr_eq_min n 1000 (1480), mscr_native n 500 (1080), mscr_native n 1000 (1480), notears n 500 (1), pcorr_eq n 500 (1035), pcorr_eq n 1000 (1454), pcorr_eq n 4000 (1079), pcorr_eq n 8000 (718), pcorr_eq n 24000 (703), pcorr_eq_min n 500 (1017), pcorr_eq_min n 1000 (1428).
