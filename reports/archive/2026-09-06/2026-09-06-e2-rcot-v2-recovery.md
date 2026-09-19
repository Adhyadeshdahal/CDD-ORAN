# E2 RCoT-v2 recovery (block_perm, B=299) — as-run

protocol_commit `8052e10` (frozen); block_perm_reps=299; 10 seeds; single-core; no seed dropped.

## Envelope (mean [min, max])

- NCP->KPI: P 0.964 [0.86, 1.00] | R 0.769 [0.75, 0.81] | F1 0.855 [0.80, 0.90]
- KPI->KPI FP: 6/360 total (mean 0.60/36); rejection 0.983

## Per seed

| seed | NCP P | NCP R | NCP F1 | KPI->KPI FP/36 | rej |
|---|---|---|---|---|---|
| 0 | 0.857 | 0.750 | 0.800 | 2/36 | 0.944 |
| 1 | 1.000 | 0.750 | 0.857 | 1/36 | 0.972 |
| 2 | 1.000 | 0.750 | 0.857 | 0/36 | 1.000 |
| 3 | 1.000 | 0.812 | 0.897 | 0/36 | 1.000 |
| 4 | 0.857 | 0.750 | 0.800 | 0/36 | 1.000 |
| 5 | 1.000 | 0.750 | 0.857 | 0/36 | 1.000 |
| 6 | 1.000 | 0.750 | 0.857 | 0/36 | 1.000 |
| 7 | 1.000 | 0.812 | 0.897 | 1/36 | 0.972 |
| 8 | 0.923 | 0.750 | 0.828 | 0/36 | 1.000 |
| 9 | 1.000 | 0.812 | 0.897 | 2/36 | 0.944 |

## Comparison

| method | NCP R | NCP F1 | KPI->KPI FP | runtime |
|---|---|---|---|---|
| RCoT-v2 (B=299, this) | 0.769 | 0.855 | 6/360 | ~1.7h/10 seeds |
| RCoT-v1 (B=99) | 0.163 | 0.251 | 6/360 | ~14min/10 seeds |
| pdCor (retired, seed-0) | 0.813 | 0.867 | 17/36 | 8.65h/seed |

Firewall: method validation only, NOT E2 decision-value evidence; E2 decision null (8d70a4f) untouched. v1 stands as the recorded first-freeze outcome; pdCor the retired baseline.
