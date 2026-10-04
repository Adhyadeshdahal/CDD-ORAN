# DEV runs, Study A: pmrt_nl_eq (R-43), cdl (R-51) and the final rule T1

DEV run, spec scratchpad/xmethod/specs/dev/pmrt_nl.json (sha256 68e18a32ef86); 1440 / 1440 units, 1440 ok, 0 errors, 0 infeasible, 0 missing; commits ['e645167']; platforms ['kaggle'].
DEV run, spec scratchpad/xmethod/specs/dev/cdl.json (sha256 66df72c517e3); 5080 / 5080 units, 5080 ok, 0 errors, 0 infeasible, 0 missing; commits ['0843512', '44b0a31', '558266f', 'bf7b2da', 'd3d6d2f', 'd4fd1d9', 'e645167', 'f0af25b']; platforms ['colab', 'kaggle', 'lightning'].
Seeds: tune 3_000_000-019; measure 3_000_100-119, + 3_000_120-159 in the R-21 cells (R2 all worlds, E4 R3; n <= 4000); pmrt_nl_eq on the T1 cells only (n <= 4000, E4 excluded); reserve seeds 3_000_160-189 unused. Definitions: eval_analysis.py (PROTOCOL_A sec 8; R-29 conformal tau from the tune seeds, R-30 three-way validity: INVALID = CI lower bound > .05, VALID = upper <= .075, else INCONCLUSIVE; p arms: raw p <= .05 and BY declarations; tau arms: truth-null and confounded-placebo declarations). Descriptive only: nothing is tuned on these numbers. Files: dev_cells.json, t1.json, t1_pairs.json, agg.json.

Cells: 124; validity {'INCONCLUSIVE': 26, 'VALID': 41, 'INVALID': 45, 'NO_READ': 12} (NO_READ = no truth-null / confounded-placebo candidate: tau arms in E4 R1 / R2).

## 1. Validity (R-30) per world-regime: INVALID / INCONCLUSIVE / VALID cells over n, lam, kappa

| world regime | cdl | pmrt_nl_eq |
|---|---|---|
| E1 R1 | 1/2/2 | 0/3/0 |
| E1 R2 | 5/1/1 | 0/2/1 |
| E2 R1 | 0/1/4 | 0/3/0 |
| E2 R2 | 0/1/6 | 0/1/2 |
| E3 R1 | 0/2/3 | 0/2/1 |
| E3 R2 | 7/0/0 | 0/0/3 |
| E4 R1 l1 | nr | - |
| E4 R2 l1 | nr | - |
| E4 R3 l0 | 1/3/1 | - |
| E4 R3 l0.5 | 5/0/0 | - |
| E4 R3 l1 | 5/0/0 | - |
| E4 R3 l1.5 | 5/0/0 | - |
| E4 R4 l0 | 1/2/2 | - |
| E4 R4 l0.5 | 5/0/0 | - |
| E4 R4 l1 | 5/0/0 | - |
| E4 R4 l1.5 | 5/0/0 | - |
| E5 R1 | 0/1/4 | 0/2/1 |
| E5 R2 | 0/0/7 | 0/0/3 |

Key reads, R2 (setpoint + dither) at n 1000, kappa .25: truth-null raw-p rate (p arms) or declaration rate (tau arms) [95 % seed-cluster CI]; P_placebo raw-p rate (p arms).

| arm | E1 R2 | E2 R2 | E3 R2 | E5 R2 | P_placebo (E2 R2) |
|---|---|---|---|---|---|
| cdl | .092 [.072, .111] | .022 [.016, .029] | .168 [.148, .190] | .017 [.007, .028] | - |
| pmrt_nl_eq | .053 [.037, .069] | .046 [.036, .056] | .035 [.025, .047] | .040 [.023, .060] | .050 [.025, .081] |

## 2. Recall (mean over measurement seeds), kappa .25, n 1000 / 4000 / 24000
* = INVALID cell (not compared, R-39); ~ = INCONCLUSIVE / NO_READ; - = no cell.

| world regime | cdl | pmrt_nl_eq |
|---|---|---|
| E1 R1 | 1.00/1.00/1.00~ | 1.00~/1.00~/- |
| E1 R2 | 1.00*/1.00*/1.00* | 1.00~/1.00/- |
| E2 R1 | .72/.96/.98 | .87~/.94~/- |
| E2 R2 | .52/.79/.97 | .64~/.92/- |
| E3 R1 | 1.00/1.00/1.00~ | 1.00/1.00~/- |
| E3 R2 | 1.00*/1.00*/1.00* | 1.00/1.00/- |
| E4 R1 l1 | .60~/1.00~/1.00~ | - |
| E4 R2 l1 | .22~/.55~/1.00~ | - |
| E4 R3 l0 | 1.00/1.00*/1.00~ | - |
| E4 R3 l0.5 | 1.00*/1.00*/1.00* | - |
| E4 R3 l1 | 1.00*/1.00*/1.00* | - |
| E4 R3 l1.5 | 1.00*/1.00*/1.00* | - |
| E4 R4 l0 | .55~/1.00/1.00 | - |
| E4 R4 l0.5 | 1.00*/1.00*/1.00* | - |
| E4 R4 l1 | 1.00*/1.00*/1.00* | - |
| E4 R4 l1.5 | 1.00*/1.00*/1.00* | - |
| E5 R1 | .69/.69/1.00~ | .79/1.00~/- |
| E5 R2 | .69/.74/.99 | .68/.73/- |

## 3. Cost per (method, dataset): CPU-s mean / max over worlds, regimes, seeds (1 thread, Kaggle / Colab / Lightning CPUs, one platform per dataset)

Budget R-13: 7200 CPU-s.

| arm | n 500 | n 1000 | n 4000 | n 8000 | n 24000 | peak RSS MB |
|---|---|---|---|---|---|---|
| cdl | 26.3 / 57 | 50.2 / 122 | 192.5 / 439 | 351.7 / 779 | 671.7 / 1654 | 390 |
| pmrt_nl_eq | 35.5 / 64 | 54.0 / 98 | 145.2 / 290 | - | - | 279 |

Infeasible (method, dataset): none (no unit near the budget).

## 4. Seed-count proposal (R-12, PROTOCOL_A rule T1, R-34)

- T1 (eval_analysis.t1_seed_count): kappa .25 cells at n 500 / 1000 / 4000 (E4 excluded), pmrt_nl_eq vs each primary-block arm with the same declaration rule, neither INVALID; paired recall gap .15, alpha .05 two-sided, power .8: 115 pairs, S_power = **14** (max over cells), S = min(cap 60, max(40, S_power up to a multiple of 10)) = **40** (floor binds: True; cap binds: False). Minimum detectable gap at S: max .083; minimum power at S over the pairs .999.
- Largest per-pair requirements: pc_eq E5|R2|k0.25|n4000 14 (sd_d .18); notears E5|R2|k0.25|n4000 12 (sd_d .16); pc_eq E5|R1|k0.25|n1000 11 (sd_d .16); pc_eq E5|R1|k0.25|n500 11 (sd_d .16); cdl E5|R2|k0.25|n4000 11 (sd_d .16); notears E5|R1|k0.25|n500 10 (sd_d .15).
- E4 (R3 / R4): S_E4 by rule T9 (R-34 / R-39; not computed here).
- Per cell (max over its pairs; all pairs in t1_pairs.json): E5|R2|k0.25|n4000 14; E5|R1|k0.25|n1000 11; E5|R1|k0.25|n500 11; E2|R2|k0.25|n1000 9; E5|R2|k0.25|n1000 9; E5|R2|k0.25|n500 7; E2|R1|k0.25|n1000 6; E2|R1|k0.25|n500 6; E2|R2|k0.25|n4000 6; E2|R2|k0.25|n500 6; E5|R1|k0.25|n4000 6; E2|R1|k0.25|n4000 4; others smaller.
- Null-rate precision (95 % CI half-width <= .02 at a true rate .05, DEV seed-cluster design effect): seeds needed, median / max over cells:
  null_decl: 39 / 68 (max: cdl|E5|R1|k0.25|n1000)
  null_raw: 39 / 68 (max: pmrt_nl_eq|E5|R1|k0.25|n4000)
  plac_raw: 115 / 291 (max: pmrt_nl_eq|E1|R1|k0.25|n1000)

## 5. Notes

- INVALID cells by regime {'R1': 1, 'R2': 12, 'R3': 16, 'R4': 16}; by arm {'cdl': 45}. Reported, not tuned on (R-21).
- Final T1 (R-56: S = clip(S_power, 40, 60); R-43 focal pmrt_nl_eq; feat/v2 eval_analysis xm-eval-analysis/5, copy in scratchpad/xmethod/_ref/, analysis fields from feat/v2 specs/eval/full.json): every DEV record, i.e. the 10 arms (results/dev/full) + 13 CI arms (results/dev/ci_c) + pmrt_nl_eq (results/dev/pmrt_nl) + cdl (results/dev/cdl); 2086 cells, 106 840 records, 0 unexpected / role mismatch / duplicates. 115 pairs, S_power 14 (pc_eq E5 R2 n 4000; cdl's largest pair 11, E5 R2 n 4000), S = 40 (floor), cap 60 not binding; min power at S over the pairs .999, max MDG .083. Same S as the 10-arm and CI T1 (focal pmrt_eq): S = 40 is robust to the focal choice and to adding cdl.
- pmrt_nl_eq (R-43; gbm, pmrt-core-v2, BY; T1 cells only, dev_full blocks / seeds): 24 cells, 0 INVALID (VALID 11, INCONCLUSIVE 13); R2 n 1000 truth-null raw .035-.053, P_placebo (E2 R2) .050. Run: Kaggle xm-dev-pmrtnl-k1, 1440 / 1440 ok, commit e645167, clean. Cost per dataset CPU-s mean / max 35 / 64 (n 500), 54 / 98 (n 1000), 145 / 290 (n 4000); n 8000 / 24000 from the T3 cost pilot (results/dev/pmrt_nl_cost/): E2 max 500 / 1148, E4 R3 max 37 / 103; all far below 7200.
- cdl (R-51; native, tau declarations): 100 cells, INVALID 45, VALID 30, INCONCLUSIVE 13, NO_READ 12 (E4 R1 / R2: no truth-null / confounded-placebo candidate). E1 R2 and E3 R2 INVALID at every n >= 1000 (E3 R2 also n 500): truth-null declaration rate .09-.18 (E1) and .12-.21 (E3), while the confounded-placebo declaration rate stays .00-.08; E2 R2 and E5 R2 VALID at n <= 4000. E1 R1 n 500 INVALID (.100 [.067, .133]). E4 R3 / R4 at lam > 0 INVALID as every other arm (E4 R3 lam 0 n 4000 and R4 lam 0 n 500 too). Reported, not tuned on (R-21); R-39: INVALID cells are not compared. In EVAL cdl runs R1 / R2 at n <= 4000 only (R-58(1)).
- cdl run: 5080 / 5080 ok on the full DEV grid (incl. n 8000 / 24000, which R-58(7) lets finish as descriptive; they arrived before the freeze), 0 missing / errors / duplicates, one platform per dataset (0 switched, 0 mixed). Platforms: Kaggle k3 / k4 / k6 / k8 / k12 / k13 + requeue r7, Colab c1 / c2 / c3 (c1 / c3 lost runtimes; their unfinished parts re-ran in r7 via --skip-complete-from), Lightning l1 (parts 60-63, user GO, ~0.42 credits est.). Code: cdl.py / classic.py and everything under cdd_oran / configs / scripts except campaign.py (launcher) and the pmrt_nl files identical across the 8 recorded commits. 2536 records carry dirty = true (Kaggle bundles); each Kaggle code bundle was downloaded and compared: k3 / k4 / k6 / k8 / k12 match their recorded commit byte for byte (line endings aside), so dirty = uncommitted scratchpad/e6_dev launcher files only; k13 (656 records, recorded bf7b2da) carries campaign.py of 44b0a31 (bundled 7 s before that commit: the Lightning --min-balance launcher option only).
- Cost: CPU-s per dataset mixes Kaggle, Colab (~1.4x slower) and Lightning CPUs (one platform per dataset). cdl on Kaggle alone ~17 / 34 / 121 / 232 / 487 (n 500 ... 24000); max over all platforms 1654 (n 24000), far below 7200.
