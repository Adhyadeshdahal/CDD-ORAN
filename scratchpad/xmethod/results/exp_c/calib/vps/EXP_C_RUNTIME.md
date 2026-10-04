# Experiment C: runtime per method (R-55 calib block on the VPS (xm-expc-cal-v2, Python 3.12.14), converted to Kaggle ref with the paired factors)

Generated 2026-10-04T08:11:36Z by `scratchpad/xmethod/exp_c_runtime.py` (xm-exp-c/1). Unit = one (method, dataset): record `cpu_s` = process CPU time of `method.run`, single thread (data generation excluded, shown apart). Costs in **Kaggle reference CPU-s** (`kaggle|Intel(R) Xeon(R) CPU @ 2.20GHz`); other hosts x f (R-51). Peak RSS = `peak_rss_mb` (scope per job on Linux; includes interpreter, imports and dataset), not converted. Budget R-13: 7,200 CPU-s per (method, dataset); T3: max over worlds, regimes, lambdas at n > budget makes n and every larger n infeasible; R-55: tested on cost x f_hi (within the speed factor's error of the budget = over). Rows pool tune + measure seeds, kappas and lambdas (per role and per world / regime in the json).

## Inputs

- `D:\academia\major-project\CDD-ORAN-wt\xm-citests\scratchpad\xmethod\results\exp_c\calib\raw\xm-expc-cal-v2\res_0.jsonl` (1 lines, sha256 `59795a55d2e65cd2`)
- `D:\academia\major-project\CDD-ORAN-wt\xm-citests\scratchpad\xmethod\results\exp_c\calib\raw\xm-expc-cal-v2\res_1.jsonl` (1 lines, sha256 `865621ee1a595fb7`)
- `D:\academia\major-project\CDD-ORAN-wt\xm-citests\scratchpad\xmethod\results\exp_c\calib\raw\xm-expc-cal-v2\res_2.jsonl` (1 lines, sha256 `022de3fe16559a7e`)
- `D:\academia\major-project\CDD-ORAN-wt\xm-citests\scratchpad\xmethod\results\exp_c\calib\raw\xm-expc-cal-v2\res_3.jsonl` (11 lines, sha256 `9d7eb8243eb0e86d`)
- `D:\academia\major-project\CDD-ORAN-wt\xm-citests\scratchpad\xmethod\results\exp_c\calib\raw\xm-expc-cal-v2\res_4.jsonl` (11 lines, sha256 `90ffb9e8e2f6a0d2`)
- `D:\academia\major-project\CDD-ORAN-wt\xm-citests\scratchpad\xmethod\results\exp_c\calib\raw\xm-expc-cal-v2\res_5.jsonl` (12 lines, sha256 `6f972feb4b75cd72`)
- `D:\academia\major-project\CDD-ORAN-wt\xm-citests\scratchpad\xmethod\results\exp_c\calib\raw\xm-expc-cal-v2\res_6.jsonl` (14 lines, sha256 `d880eb05dd615bfe`)
- 51 lines, 0 duplicate keys (0 on two hosts); specs exp_c_calib; commits 1b33ca1

Hosts (units):

- `vps|AMD EPYC-Rome Processor`: 51

## Speed factors applied (cost_ref = cost x f)

| arm | host | f | f_hi | source | cells | host units | cell ratio q25 / q50 / q75 |
|---|---|---|---|---|---|---|---|
| cdl | `vps|AMD EPYC-Rome Processor` | 2.307 | 2.51 | r55_calibration_block | - | 9 | - |
| mscr_eq_min | `vps|AMD EPYC-Rome Processor` | 2.810 | 2.84 | r55_calibration_block | - | 6 | - |
| pcorr_eq | `vps|AMD EPYC-Rome Processor` | 1.915 | 2.09 | r55_calibration_block | - | 9 | - |
| pmrt_nl_eq | `vps|AMD EPYC-Rome Processor` | 2.280 | 2.31 | r55_calibration_block | - | 9 | - |
| rcot2_eq | `vps|AMD EPYC-Rome Processor` | 1.987 | 2.46 | r55_calibration_block | - | 9 | - |
| shap_dag | `vps|AMD EPYC-Rome Processor` | 2.303 | 2.48 | r55_calibration_block | - | 9 | - |

## Summary: median / max CPU-s per unit (Kaggle ref.) by n; T3 status

| arm | n 500 | n 1000 | n 4000 | T3 |
|---|---|---|---|---|
| cdl | 44.0 / 44.4 | 90.6 / 90.8 | 380 / 381 | feasible n <= 4000; no DEV cost at n 8000, 24000 |
| mscr_eq_min | 366 / 366 | 774 / 777 | - | feasible n <= 1000; no DEV cost at n 4000, 8000, 24000 |
| pcorr_eq | 0.094 / 0.097 | 0.14 / 0.14 | 0.46 / 0.48 | feasible n <= 4000; no DEV cost at n 8000, 24000 |
| pmrt_nl_eq | 39.2 / 43.1 | 65.7 / 68.7 | 205 / 210 | feasible n <= 4000; no DEV cost at n 8000, 24000 |
| rcot2_eq | 1.12 / 1.17 | 1.43 / 1.50 | 3.60 / 3.64 | feasible n <= 4000; no DEV cost at n 8000, 24000 |
| shap_dag | 3.34 / 3.43 | 5.45 / 5.48 | 19.4 / 20.0 | feasible n <= 4000; no DEV cost at n 8000, 24000 |

`-` = no record at that n (T3: a cost pilot decides if the n is in the arm's grid); * = some unit unconverted; + = some unit converted with a provisional (observational) factor. Pooled over worlds, regimes, lambdas, kappas (the kappa sweep is at n 1000) and roles (tune + measure).

## Full table

| arm | method | n | units (tune / measure) | non-ref | CPU-s mean | median | min | max | max x f_hi | max / budget | RSS med MB | RSS max MB | wall-s mean | wall / CPU | procs / vCPU | gen CPU-s | multi-thr | T3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cdl | cdl | 500 | 3 (timing 3) | 100 % | 44.1 | 44.0 | 43.7 | 44.4 | 48.3 | 0.62 % | 443 | 443 | 19.1 | 1.00 | 1.00 | 0.009 | 0 | feasible |
| cdl | cdl | 1000 | 3 (timing 3) | 100 % | 90.0 | 90.6 | 88.7 | 90.8 | 98.7 | 1.26 % | 347 | 347 | 39.0 | 1.00 | 0.14 | 0.012 | 0 | feasible |
| cdl | cdl | 4000 | 3 (timing 3) | 100 % | 376 | 380 | 366 | 381 | 414 | 5.29 % | 347 | 347 | 163 | 1.00 | 1.00 | 0.046 | 0 | feasible |
| mscr_eq_min | mscr | 500 | 3 (timing 3) | 100 % | 366 | 366 | 366 | 366 | 370 | 5.08 % | 95.9 | 96.0 | 130 | 1.00 | 1.00 | 0.017 | 0 | feasible |
| mscr_eq_min | mscr | 1000 | 3 (timing 3) | 100 % | 773 | 774 | 767 | 777 | 786 | 10.80 % | 96.4 | 96.4 | 275 | 1.00 | 1.00 | 0.022 | 0 | feasible |
| pcorr_eq | pcorr | 500 | 3 (timing 3) | 100 % | 0.094 | 0.094 | 0.092 | 0.097 | 0.11 | 0.00 % | 542 | 542 | 0.049 | 1.00 | 0.43 | 0.007 | 0 | feasible |
| pcorr_eq | pcorr | 1000 | 3 (timing 3) | 100 % | 0.14 | 0.14 | 0.14 | 0.14 | 0.16 | 0.00 % | 542 | 542 | 0.075 | 1.00 | 0.29 | 0.012 | 0 | feasible |
| pcorr_eq | pcorr | 4000 | 3 (timing 3) | 100 % | 0.46 | 0.46 | 0.46 | 0.48 | 0.52 | 0.01 % | 474 | 545 | 0.24 | 1.00 | 0.57 | 0.045 | 0 | feasible |
| pmrt_nl_eq | pmrt_core | 500 | 3 (timing 3) | 100 % | 40.3 | 39.2 | 38.6 | 43.1 | 43.8 | 0.60 % | 458 | 570 | 17.7 | 1.00 | 1.00 | 0.007 | 0 | feasible |
| pmrt_nl_eq | pmrt_core | 1000 | 3 (timing 3) | 100 % | 66.0 | 65.7 | 63.8 | 68.7 | 69.7 | 0.95 % | 247 | 251 | 29.0 | 1.00 | 1.00 | 0.012 | 0 | feasible |
| pmrt_nl_eq | pmrt_core | 4000 | 3 (timing 3) | 100 % | 207 | 205 | 205 | 210 | 213 | 2.91 % | 290 | 290 | 90.8 | 1.00 | 1.00 | 0.045 | 0 | feasible |
| rcot2_eq | rcot2 | 500 | 3 (timing 3) | 100 % | 1.13 | 1.12 | 1.09 | 1.17 | 1.45 | 0.02 % | 541 | 541 | 0.57 | 1.00 | 0.29 | 0.008 | 0 | feasible |
| rcot2_eq | rcot2 | 1000 | 3 (timing 3) | 100 % | 1.44 | 1.43 | 1.40 | 1.50 | 1.86 | 0.02 % | 301 | 541 | 0.73 | 1.00 | 0.57 | 0.012 | 0 | feasible |
| rcot2_eq | rcot2 | 4000 | 3 (timing 3) | 100 % | 3.58 | 3.60 | 3.50 | 3.64 | 4.49 | 0.05 % | 541 | 541 | 1.80 | 1.00 | 0.43 | 0.043 | 0 | feasible |
| shap_dag | shap_dag | 500 | 3 (timing 3) | 100 % | 3.35 | 3.34 | 3.28 | 3.43 | 3.70 | 0.05 % | 541 | 541 | 1.45 | 1.00 | 0.29 | 0.007 | 0 | feasible |
| shap_dag | shap_dag | 1000 | 3 (timing 3) | 100 % | 5.46 | 5.45 | 5.44 | 5.48 | 5.91 | 0.08 % | 531 | 540 | 2.37 | 1.00 | 0.43 | 0.012 | 0 | feasible |
| shap_dag | shap_dag | 4000 | 3 (timing 3) | 100 % | 19.5 | 19.4 | 19.1 | 20.0 | 21.6 | 0.28 % | 536 | 537 | 8.47 | 1.00 | 0.14 | 0.044 | 0 | feasible |

Single-thread check: 0 units with cpu_s > 1.15 x wall_s + .05 s (more than one busy thread).

Contention check: 0 units with wall_s > 1.3 x cpu_s + .05 s (CPU-starved: more busy processes than cores; CPU-s on shared hyperthreads runs slower, so the speed factor must be measured under the same load).
