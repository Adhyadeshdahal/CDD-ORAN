# Experiment B measured cost (Kaggle reference CPU-s)

Factors: `scratchpad/xmethod/results/exp_b/factors_vps_py3.12.13.json` (sha256 1aa2fab42d24...); infeasible units at the 7200 s budget. Total 82.0 ref core-h.

| arm | units | ok | infeasible | ref core-h | max ref CPU-s / unit | non-ref factor |
|---|---|---|---|---|---|---|
| cdl | 76 | 64 | 12 | 53.71 | 7200 | cdl 2.368 |
| corr | 76 | 76 | 0 | 0.00 | 0 | * 2.687 |
| granger_eq | 76 | 76 | 0 | 0.07 | 47 | * 2.687 |
| granger_native | 76 | 76 | 0 | 0.00 | 0 | * 2.687 |
| mscr_eq | 18 | 18 | 0 | 1.17 | 385 | * 2.687 |
| mscr_native | 18 | 18 | 0 | 1.09 | 363 | * 2.687 |
| notears | 76 | 76 | 0 | 12.91 | 5071 | * 2.687 |
| pc_eq | 76 | 76 | 0 | 5.54 | 4701 | * 2.687 |
| pc_native | 76 | 76 | 0 | 1.97 | 345 | * 2.687 |
| pcorr_eq | 76 | 76 | 0 | 0.03 | 18 | pcorr_eq 1.975 |
| pcorr_hac_fb | 76 | 76 | 0 | 0.08 | 36 | * 2.687 |
| pcorr_native | 76 | 76 | 0 | 0.03 | 15 | * 2.687 |
| pmrt_eq | 76 | 76 | 0 | 0.06 | 34 | * 2.687 |
| pmrt_nl_eq | 76 | 76 | 0 | 4.05 | 678 | pmrt_nl_eq 2.331 |
| pmrt_r3 | 76 | 76 | 0 | 0.06 | 33 | * 2.687 |
| rcot2_eq | 76 | 76 | 0 | 0.03 | 12 | rcot2_eq 1.992 |
| rcot2_native | 76 | 76 | 0 | 0.04 | 11 | * 2.687 |
| shap_dag | 76 | 76 | 0 | 0.13 | 45 | shap_dag 2.340 |
| two_tower | 76 | 76 | 0 | 1.04 | 523 | * 2.687 |
