# Experiment C: runtime per method (R-55 calib block, Kaggle reference (xm-expc-cal-k1, uncontended))

Generated 2026-10-04T00:40:32Z by `scratchpad/xmethod/exp_c_runtime.py` (xm-exp-c/1). Unit = one (method, dataset): record `cpu_s` = process CPU time of `method.run`, single thread (data generation excluded, shown apart). Costs in **Kaggle reference CPU-s** (`kaggle|Intel(R) Xeon(R) CPU @ 2.20GHz`); other hosts x f (R-51). Peak RSS = `peak_rss_mb` (scope per job on Linux; includes interpreter, imports and dataset), not converted. Budget R-13: 7,200 CPU-s per (method, dataset); T3: max over worlds, regimes, lambdas at n > budget makes n and every larger n infeasible; R-55: tested on cost x f_hi (within the speed factor's error of the budget = over). Rows pool tune + measure seeds, kappas and lambdas (per role and per world / regime in the json).

## Inputs

- `D:\academia\major-project\CDD-ORAN-wt\xm-citests\scratchpad\xmethod\results\exp_c\calib\raw\xm-expc-cal-k1\res_0.jsonl` (14 lines, sha256 `92c1e925da73b0e2`)
- `D:\academia\major-project\CDD-ORAN-wt\xm-citests\scratchpad\xmethod\results\exp_c\calib\raw\xm-expc-cal-k1\res_1.jsonl` (13 lines, sha256 `cea993e5795b83dc`)
- `D:\academia\major-project\CDD-ORAN-wt\xm-citests\scratchpad\xmethod\results\exp_c\calib\raw\xm-expc-cal-k1\res_2.jsonl` (12 lines, sha256 `07281ac2d830a510`)
- `D:\academia\major-project\CDD-ORAN-wt\xm-citests\scratchpad\xmethod\results\exp_c\calib\raw\xm-expc-cal-k1\res_3.jsonl` (12 lines, sha256 `b837329a52f6e6ac`)
- 51 lines, 0 duplicate keys (0 on two hosts); specs exp_c_calib; commits cbfbdad

Hosts (units):

- `kaggle|Intel(R) Xeon(R) CPU @ 2.20GHz`: 51

## Speed factors applied (cost_ref = cost x f)

None needed (every unit on the reference host) or none available.

## Summary: median / max CPU-s per unit (Kaggle ref.) by n; T3 status

| arm | n 500 | n 1000 | n 4000 | T3 |
|---|---|---|---|---|
| cdl | 35.1 / 37.9 | 65.9 / 70.6 | 413 / 415 | feasible n <= 4000; no DEV cost at n 8000, 24000 |
| mscr_eq_min | 363 / 364 | 780 / 784 | - | feasible n <= 1000; no DEV cost at n 4000, 8000, 24000 |
| pcorr_eq | 0.078 / 0.079 | 0.12 / 0.12 | 0.55 / 0.55 | feasible n <= 4000; no DEV cost at n 8000, 24000 |
| pmrt_nl_eq | 43.2 / 48.2 | 67.0 / 71.0 | 206 / 223 | feasible n <= 4000; no DEV cost at n 8000, 24000 |
| rcot2_eq | 0.91 / 1.37 | 1.77 / 1.82 | 2.89 / 4.25 | feasible n <= 4000; no DEV cost at n 8000, 24000 |
| shap_dag | 2.22 / 3.97 | 3.63 / 6.29 | 22.0 / 22.3 | feasible n <= 4000; no DEV cost at n 8000, 24000 |

`-` = no record at that n (T3: a cost pilot decides if the n is in the arm's grid); * = some unit unconverted; + = some unit converted with a provisional (observational) factor. Pooled over worlds, regimes, lambdas, kappas (the kappa sweep is at n 1000) and roles (tune + measure).

## Full table

| arm | method | n | units (tune / measure) | non-ref | CPU-s mean | median | min | max | max x f_hi | max / budget | RSS med MB | RSS max MB | wall-s mean | wall / CPU | procs / vCPU | gen CPU-s | multi-thr | T3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cdl | cdl | 500 | 3 (timing 3) | 0 % | 34.2 | 35.1 | 29.6 | 37.9 | 37.9 | 0.53 % | 506 | 507 | 34.3 | 1.00 | 0.75 | 0.016 | 0 | feasible |
| cdl | cdl | 1000 | 3 (timing 3) | 0 % | 66.9 | 65.9 | 64.4 | 70.6 | 70.6 | 0.98 % | 506 | 507 | 67.1 | 1.00 | 0.75 | 0.025 | 0 | feasible |
| cdl | cdl | 4000 | 3 (timing 3) | 0 % | 409 | 413 | 398 | 415 | 415 | 5.76 % | 388 | 389 | 413 | 1.01 | 1.00 | 0.14 | 0 | feasible |
| mscr_eq_min | mscr | 500 | 3 (timing 3) | 0 % | 357 | 363 | 346 | 364 | 364 | 5.05 % | 101 | 101 | 361 | 1.01 | 1.00 | 0.027 | 0 | feasible |
| mscr_eq_min | mscr | 1000 | 3 (timing 3) | 0 % | 781 | 780 | 779 | 784 | 784 | 10.89 % | 101 | 102 | 789 | 1.01 | 1.00 | 0.054 | 0 | feasible |
| pcorr_eq | pcorr | 500 | 3 (timing 3) | 0 % | 0.078 | 0.078 | 0.077 | 0.079 | 0.079 | 0.00 % | 639 | 640 | 0.078 | 1.00 | 0.25 | 0.010 | 0 | feasible |
| pcorr_eq | pcorr | 1000 | 3 (timing 3) | 0 % | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.00 % | 635 | 635 | 0.12 | 1.00 | 0.75 | 0.018 | 0 | feasible |
| pcorr_eq | pcorr | 4000 | 3 (timing 3) | 0 % | 0.51 | 0.55 | 0.41 | 0.55 | 0.55 | 0.01 % | 554 | 635 | 0.51 | 1.00 | 1.00 | 0.11 | 0 | feasible |
| pmrt_nl_eq | pmrt_core | 500 | 3 (timing 3) | 0 % | 40.9 | 43.2 | 31.4 | 48.2 | 48.2 | 0.67 % | 521 | 654 | 41.1 | 1.00 | 0.75 | 0.017 | 0 | feasible |
| pmrt_nl_eq | pmrt_core | 1000 | 3 (timing 3) | 0 % | 65.0 | 67.0 | 57.2 | 71.0 | 71.0 | 0.99 % | 558 | 559 | 65.2 | 1.00 | 0.75 | 0.033 | 0 | feasible |
| pmrt_nl_eq | pmrt_core | 4000 | 3 (timing 3) | 0 % | 207 | 206 | 192 | 223 | 223 | 3.10 % | 601 | 601 | 208 | 1.00 | 0.75 | 0.10 | 0 | feasible |
| rcot2_eq | rcot2 | 500 | 3 (timing 3) | 0 % | 1.05 | 0.91 | 0.87 | 1.37 | 1.37 | 0.02 % | 639 | 640 | 1.05 | 1.00 | 0.50 | 0.013 | 0 | feasible |
| rcot2_eq | rcot2 | 1000 | 3 (timing 3) | 0 % | 1.78 | 1.77 | 1.77 | 1.82 | 1.82 | 0.03 % | 635 | 635 | 1.79 | 1.00 | 0.75 | 0.033 | 0 | feasible |
| rcot2_eq | rcot2 | 4000 | 3 (timing 3) | 0 % | 3.32 | 2.89 | 2.82 | 4.25 | 4.25 | 0.06 % | 639 | 640 | 3.32 | 1.00 | 0.50 | 0.086 | 0 | feasible |
| shap_dag | shap_dag | 500 | 3 (timing 3) | 0 % | 2.78 | 2.22 | 2.15 | 3.97 | 3.97 | 0.06 % | 639 | 640 | 2.80 | 1.00 | 0.50 | 0.013 | 0 | feasible |
| shap_dag | shap_dag | 1000 | 3 (timing 3) | 0 % | 4.51 | 3.63 | 3.61 | 6.29 | 6.29 | 0.09 % | 629 | 630 | 4.53 | 1.00 | 0.50 | 0.024 | 0 | feasible |
| shap_dag | shap_dag | 4000 | 3 (timing 3) | 0 % | 21.0 | 22.0 | 18.8 | 22.3 | 22.3 | 0.31 % | 341 | 623 | 21.2 | 1.01 | 1.00 | 0.14 | 0 | feasible |

Single-thread check: 0 units with cpu_s > 1.15 x wall_s + .05 s (more than one busy thread).

Contention check: 0 units with wall_s > 1.3 x cpu_s + .05 s (CPU-starved: more busy processes than cores; CPU-s on shared hyperthreads runs slower, so the speed factor must be measured under the same load).
