# E2 RCoT-v2 recovery on the REDESIGNED E2 env (2026-09-06)

Frozen method: protocol_commit_v2 `8052e10`, block_perm_reps=299, dz=25, block_perm null. NO RCoT constant changed. Firewall: all Phase A discovery persisted+hash-loaded (fail-closed) before any Phase B truth read.

Redesigned env commit 9d87a60 (E2_OPERATING_POINT_REDESIGN): id_ranges[6]=(1.5,4.0), KPI_MEAN_STD[5]=(-0.956730,4.771898). 10 seeds, n_rows=4000, sampling_seed=0.

## NCP->KPI envelope (mean [min, max])

- Precision: 0.992 [0.917, 1.000]
- Recall:    0.650 [0.625, 0.688]
- F1:        0.785 [0.769, 0.815]
- KPI->KPI FP total (over 10 seeds x 36 candidates = 360): 6

Old-env run (for comparison): NCP P 0.964 / R 0.769 / F1 0.855; KPI->KPI FP 6/360.

## Per-seed table

| seed | P | R | F1 | tp | fp | fn | kpikpi_fp | P0->K5 | P0->K0 | P0->K1 | P0->K2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.917 | 0.688 | 0.786 | 11 | 1 | 5 | 1 | 0 | 1 | 1 | 1 |
| 1 | 1.000 | 0.688 | 0.815 | 11 | 0 | 5 | 1 | 0 | 1 | 1 | 1 |
| 2 | 1.000 | 0.625 | 0.769 | 10 | 0 | 6 | 0 | 0 | 1 | 1 | 1 |
| 3 | 1.000 | 0.625 | 0.769 | 10 | 0 | 6 | 0 | 0 | 1 | 1 | 1 |
| 4 | 1.000 | 0.625 | 0.769 | 10 | 0 | 6 | 1 | 0 | 1 | 1 | 1 |
| 5 | 1.000 | 0.625 | 0.769 | 10 | 0 | 6 | 0 | 0 | 1 | 1 | 1 |
| 6 | 1.000 | 0.625 | 0.769 | 10 | 0 | 6 | 0 | 0 | 1 | 1 | 1 |
| 7 | 1.000 | 0.625 | 0.769 | 10 | 0 | 6 | 0 | 0 | 1 | 1 | 1 |
| 8 | 1.000 | 0.688 | 0.815 | 11 | 0 | 5 | 0 | 0 | 1 | 1 | 1 |
| 9 | 1.000 | 0.688 | 0.815 | 11 | 0 | 5 | 3 | 0 | 1 | 1 | 1 |

## Harmful edge P0->K5 vs beneficial P0->{K0,K1,K2}

- **P0->K5 recovered: 0/10** (OLD env: 3/10)
- P0->K0 recovered: 10/10 (OLD env: 10/10)
- P0->K1 recovered: 10/10 (OLD env: 10/10)
- P0->K2 recovered: 10/10 (OLD env: 10/10)

(Recomputed OLD-env from its own discovery masks: P0->K5 3/10; P0->K0 10/10, P0->K1 10/10, P0->K2 10/10.)
