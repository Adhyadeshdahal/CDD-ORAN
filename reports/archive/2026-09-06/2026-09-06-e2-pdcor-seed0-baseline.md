# E2 pdCor discovery — recorded seed-0 baseline (RETIRED method)

**Date:** 2026-09-06 · **Status:** recorded baseline for a retired method — nothing tuned, n=1 seed.

## Framing

This records the scored seed-0 result of the frozen E2 label-free discovery method
(`protocol_commit 828e3458065bfe27ff7af07b1295b650f89e878e`): **U-centered partial distance
correlation** (`score_method: u_centered_partial_distance_correlation`) with a **per-candidate
permutation null (B=999) plus per-target BH-FDR** (`threshold_method:
per_candidate_permutation_per_target_bh_fdr`, q=0.05). This method has been **RETIRED as the LIVE
E2 method** (decision 2026-09-06) in favor of a future RCoT conditional-independence test, because
it is both ~8h/seed and over-selects edges ~9× from marginal-vs-conditional permutation
miscalibration (see `reports/2026-09-06-e2-improvement-synthesis.md` and
`reports/2026-09-06-cmi-pdcor-rcot-comparison.md`). pdCor is kept on disk as a recorded
protocol/baseline. The numbers below are the already-computed seed-0 artifacts, consolidated
without re-running anything; nothing is tuned. This is a single seed (n=1).

## Run config (from `runs/e2slice-recovery/replicate-00/manifest.json`)

| Field | Value |
|---|---|
| env | `E2V2Env` |
| N (n_rows_per_seed) | 4000 |
| seed | 0 |
| sampling_seed | 0 |
| obs_noise_scale | 0.0 (noiseless, `Y = f(Z)` deterministic) |
| num_params | 8 |
| num_kpis | 6 |
| decoy_omit_p0_k5 | false |
| protocol_commit | `828e3458065bfe27ff7af07b1295b650f89e878e` |
| git_sha (run) | `f0768430a3b4848772e053cdc673ad56f4f5da31` (git_dirty false) |
| dataset_hash | `f223764faeee80653f9015f785d8429bba0e9f31fd8a70e892eaea14600896ea` |
| B_perm / q | 999 / 0.05 |

## Scored recovery (from `runs/e2slice-recovery/replicate-00/recovery.json`, verbatim)

`recovery.json` matches the expected structure: it breaks results out by edge class (`ncp_kpi`,
`kpi_kpi`, `overall`) plus a `per_target` array and top-level KPI→KPI aggregates. Ground truth =
16 edges (`gt_edge_count`).

**NCP→KPI (real edges) — power is fine:**

| metric | value |
|---|---|
| precision | 0.9285714285714286 |
| recall | 0.8125 |
| f1 | 0.8666666666666666 |
| tp / fp / fn | 13 / 1 / 3 |
| accuracy | 0.9642857142857143 |
| edges (selected) | 14 |

3 missed NCP→KPI edges (`missed`): P2→K1, P1→K4, P0→K5.

**KPI→KPI (lagged-KPI candidates) — the isolated failure:**

| metric | value |
|---|---|
| precision | 0.0 |
| recall | 0.0 |
| f1 | 0.0 |
| tp / fp / fn | 0 / 17 / 0 |
| accuracy | 0.7976190476190477 |
| edges (selected) | 17 (all false positives) |
| kpi_kpi_candidates | 36 |
| kpi_kpi_fp | 17 |
| kpi_kpi_rejection_rate | 0.5277777777777778 |

All 17 selected KPI→KPI edges are false positives against 0 true KPI→KPI edges; the rejection rate
of 0.528 means only 19 of 36 KPI→KPI candidates were correctly rejected (~47% falsely selected),
i.e. ~9× the nominal FDR.

**Overall:**

| metric | value |
|---|---|
| precision | 0.41935483870967744 |
| recall | 0.8125 |
| f1 | 0.5531914893617021 |
| tp / fp / fn | 13 / 18 / 3 |
| accuracy | 0.8928571428571429 |
| edges (selected) | 31 |

## Seed-0 wall-clock

Discovery for seed-0 ran **~31,138 s ≈ 8.65 h (8 h 38 m)**, from the epoch timestamps
`.discover00.start` (1788631540) → `.discover00.end` (1788662678). `.overnight.log` records
`seed-0 discovery.json present Sun Sep 6 08:29:53 NST 2026` under a single-core sequential driver.
(`.discover00.timelog` holds only `/usr/bin/time: No such file or directory` — the `time` wrapper
was unavailable, so the start/end epoch files are the timing source.) `.discover00.out` confirms
`discovered 31 edges … guarded 0` and `DISCOVER00_EXIT=0`.

## State of seeds 1–9 (factual)

- **seed 0** — COMPLETE: `discovery.json` + scored `recovery.json` present.
- **seeds 1 & 2** (`replicate-01`, `replicate-02`) — GENERATION-ONLY / discovery FAILED:
  each has `manifest.json` + `rows.npz` (4000 rows generated), but **no `discovery.json` and no
  `recovery.json`**. `.overnight.log` records both discover steps exiting `rc=127` (command not
  found); `.discover01.out` and `.discover02.out` are 0 bytes. Discovery never actually ran for
  these seeds.
- **seeds 3–9** — ABSENT: no replicate directories exist.

No discovery, generation, or scoring was run to produce this report.

## Interpretation

Consistent with the synthesis: the failure is **isolated to the Z-dependent lagged-KPI (KPI→KPI)
candidates** — a null-calibration defect, not an estimator-power defect. Real-edge power on the
independent-parameter NCP→KPI candidates is fine (P 0.929 / R 0.812 / F1 0.867; 13 of 16 true
edges recovered with a single false positive), while the KPI→KPI candidates over-select badly
(17/36 false positives, 0 true positives, ~53% correctly rejected). This matches the diagnosed
mechanism: permuting a candidate column that is a smooth function of a shared parent vector
destroys its dependence on the conditioning set, so the permutation null samples *joint* rather
than *conditional* independence and is too tight, landing Z-dependent true-negatives in its upper
tail — while independently-drawn parameter candidates calibrate cleanly. The `pdCor` denominator
guard fired 0 times, as expected (the failure is invisible to it). This is a single-seed recorded
baseline for a retired method; no claim is made beyond the seed-0 numbers above.
