# DEV runs, Study A: CI-test arms (plan C)

DEV run, spec scratchpad/xmethod/specs/dev/ci_c.json (sha256 4bce54235d15); 58480 / 58480 units, 58480 ok, 0 errors, 0 infeasible, 0 missing; commits ['3a251c677c7ec15734e5a8ad3875b9d2305e2134', '9fd845b4eba6bae70f12ffd57283b80a8d7524df', 'e8c7ceda3e27142d97672c04db052e864af83265']; platforms ['colab', 'kaggle', 'lightning'].
Seeds as dev_full (tune 3_000_000-019; measure 3_000_100-119, + 3_000_120-159 in the R-21 cells); mscr arms n <= 1000 only (Q10 plan C). Definitions: eval_analysis.py (PROTOCOL_A sec 8; R-29 conformal tau from the tune seeds, R-30 three-way validity: INVALID = CI lower bound > .05, VALID = upper <= .075, else INCONCLUSIVE; p arms: raw p <= .05 and BY declarations; tau arms: truth-null and confounded-placebo declarations). Descriptive only: nothing is tuned on these numbers. Files: dev_cells.json, t1.json, t1_pairs.json, agg.json.

Cells: 1138; validity {'INCONCLUSIVE': 572, 'INVALID': 444, 'VALID': 122} (NO_READ = no truth-null / confounded-placebo candidate: tau arms in E4 R1 / R2).

## 1. Validity (R-30) per world-regime: INVALID / INCONCLUSIVE / VALID cells over n, lam, kappa

| world regime | mscr_eq | mscr_native | mscr_eq_min | pcorr_eq | pcorr_native | pcorr_eq_min | pcorr_hac | pcorr_hac_fb | pcorr_hac_eq_min | pcorr_hac_fb_eq_min | rcot2_eq | rcot2_native | rcot2_eq_min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 R1 | 0/1/1 | 0/2/0 | 0/2/0 | 0/4/1 | 0/3/2 | 0/3/2 | 0/3/2 | 0/3/2 | 0/3/2 | 0/3/2 | 1/2/2 | 2/2/1 | 0/5/0 |
| E1 R2 | 4/0/0 | 4/0/0 | 4/0/0 | 0/3/4 | 7/0/0 | 0/1/6 | 7/0/0 | 7/0/0 | 0/1/6 | 0/1/6 | 7/0/0 | 7/0/0 | 5/2/0 |
| E2 R1 | 0/2/0 | 0/2/0 | 0/2/0 | 0/5/0 | 0/5/0 | 0/5/0 | 0/5/0 | 0/5/0 | 0/5/0 | 0/5/0 | 2/3/0 | 1/4/0 | 2/2/1 |
| E2 R2 | 4/0/0 | 4/0/0 | 4/0/0 | 0/5/2 | 7/0/0 | 1/4/2 | 7/0/0 | 7/0/0 | 1/4/2 | 1/4/2 | 7/0/0 | 7/0/0 | 7/0/0 |
| E3 R1 | 0/2/0 | 0/2/0 | 0/2/0 | 0/4/1 | 0/5/0 | 0/5/0 | 0/5/0 | 0/5/0 | 0/5/0 | 0/5/0 | 1/4/0 | 3/2/0 | 2/3/0 |
| E3 R2 | 4/0/0 | 4/0/0 | 4/0/0 | 0/5/2 | 7/0/0 | 0/5/2 | 7/0/0 | 7/0/0 | 0/5/2 | 0/5/2 | 7/0/0 | 7/0/0 | 6/1/0 |
| E4 R1 l1 | 0/2/0 | 0/2/0 | 0/2/0 | 0/4/1 | 0/4/1 | 0/4/1 | 0/4/1 | 0/4/1 | 0/4/1 | 0/4/1 | 0/4/1 | 0/3/2 | 0/3/2 |
| E4 R2 l1 | 0/4/0 | 0/4/0 | 0/4/0 | 0/5/2 | 0/4/3 | 0/5/2 | 0/4/3 | 0/4/3 | 0/5/2 | 0/5/2 | 0/6/1 | 0/6/1 | 0/7/0 |
| E4 R3 l0 | 0/2/0 | 0/2/0 | 0/2/0 | 0/4/1 | 0/4/1 | 0/4/1 | 0/4/1 | 0/4/1 | 0/4/1 | 0/4/1 | 0/5/0 | 0/5/0 | 0/4/1 |
| E4 R3 l0.5 | 2/0/0 | 2/0/0 | 2/0/0 | 0/5/0 | 0/5/0 | 0/5/0 | 0/5/0 | 0/5/0 | 0/5/0 | 0/5/0 | 5/0/0 | 1/4/0 | 0/5/0 |
| E4 R3 l1 | 2/0/0 | 2/0/0 | 2/0/0 | 0/5/0 | 0/5/0 | 0/5/0 | 0/5/0 | 0/5/0 | 0/5/0 | 0/5/0 | 5/0/0 | 3/2/0 | 3/2/0 |
| E4 R3 l1.5 | 2/0/0 | 2/0/0 | 2/0/0 | 0/5/0 | 0/5/0 | 0/5/0 | 0/5/0 | 0/5/0 | 0/5/0 | 0/5/0 | 5/0/0 | 3/2/0 | 3/2/0 |
| E4 R4 l0 | 0/1/1 | 0/2/0 | 0/2/0 | 0/5/0 | 0/4/1 | 0/4/1 | 0/4/1 | 0/4/1 | 0/4/1 | 0/4/1 | 0/4/1 | 0/4/1 | 0/4/1 |
| E4 R4 l0.5 | 2/0/0 | 2/0/0 | 2/0/0 | 5/0/0 | 5/0/0 | 5/0/0 | 5/0/0 | 5/0/0 | 5/0/0 | 5/0/0 | 5/0/0 | 5/0/0 | 5/0/0 |
| E4 R4 l1 | 2/0/0 | 2/0/0 | 2/0/0 | 5/0/0 | 5/0/0 | 5/0/0 | 5/0/0 | 5/0/0 | 5/0/0 | 5/0/0 | 5/0/0 | 5/0/0 | 5/0/0 |
| E4 R4 l1.5 | 2/0/0 | 2/0/0 | 2/0/0 | 5/0/0 | 5/0/0 | 5/0/0 | 5/0/0 | 5/0/0 | 5/0/0 | 5/0/0 | 5/0/0 | 5/0/0 | 5/0/0 |
| E5 R1 | 0/2/0 | 0/1/1 | 0/1/1 | 0/5/0 | 0/5/0 | 0/5/0 | 0/5/0 | 0/5/0 | 0/5/0 | 0/5/0 | 3/1/1 | 1/4/0 | 0/5/0 |
| E5 R2 | 4/0/0 | 4/0/0 | 4/0/0 | 0/4/3 | 7/0/0 | 0/5/2 | 7/0/0 | 7/0/0 | 0/3/4 | 0/3/4 | 6/1/0 | 6/1/0 | 5/2/0 |

Key reads, R2 (setpoint + dither) at n 1000, kappa .25: truth-null raw-p rate (p arms) or declaration rate (tau arms) [95 % seed-cluster CI]; P_placebo raw-p rate (p arms).

| arm | E1 R2 | E2 R2 | E3 R2 | E5 R2 | P_placebo (E2 R2) |
|---|---|---|---|---|---|
| mscr_eq | 1.000 [1.000, 1.000] | .948 [.936, .960] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | .944 [.925, .964] |
| mscr_native | 1.000 [1.000, 1.000] | .931 [.918, .944] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | .919 [.897, .942] |
| mscr_eq_min | 1.000 [1.000, 1.000] | .931 [.918, .944] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | .914 [.892, .936] |
| pcorr_eq | .037 [.026, .050] | .060 [.049, .072] | .053 [.040, .069] | .060 [.042, .078] | .061 [.039, .086] |
| pcorr_native | .215 [.201, .229] | .461 [.439, .484] | .274 [.256, .292] | .355 [.317, .392] | .422 [.367, .472] |
| pcorr_eq_min | .042 [.028, .057] | .056 [.047, .067] | .046 [.034, .059] | .058 [.042, .077] | .058 [.036, .083] |
| pcorr_hac | .217 [.203, .232] | .464 [.441, .485] | .277 [.261, .295] | .363 [.325, .400] | .414 [.358, .467] |
| pcorr_hac_fb | .215 [.201, .229] | .461 [.439, .482] | .276 [.260, .294] | .360 [.322, .397] | .414 [.358, .467] |
| pcorr_hac_eq_min | .040 [.025, .056] | .057 [.047, .067] | .044 [.032, .057] | .055 [.038, .072] | .058 [.039, .081] |
| pcorr_hac_fb_eq_min | .040 [.025, .056] | .057 [.047, .067] | .044 [.032, .057] | .055 [.038, .072] | .058 [.039, .081] |
| rcot2_eq | .075 [.054, .097] | .104 [.090, .117] | .067 [.050, .084] | .100 [.082, .122] | .108 [.075, .142] |
| rcot2_native | .147 [.121, .174] | .184 [.168, .202] | .199 [.178, .220] | .120 [.097, .148] | .203 [.158, .250] |
| rcot2_eq_min | .058 [.040, .076] | .095 [.081, .110] | .074 [.057, .090] | .083 [.063, .107] | .089 [.058, .122] |

## 2. Recall (mean over measurement seeds), kappa .25, n 1000 / 4000 / 24000
* = INVALID cell (not compared, R-39); ~ = INCONCLUSIVE / NO_READ; - = no cell.

| world regime | mscr_eq | mscr_native | mscr_eq_min | pcorr_eq | pcorr_native | pcorr_eq_min | pcorr_hac | pcorr_hac_fb | pcorr_hac_eq_min | pcorr_hac_fb_eq_min | rcot2_eq | rcot2_native | rcot2_eq_min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 R1 | 1.00~/-/- | 1.00~/-/- | 1.00~/-/- | 1.00~/1.00~/1.00~ | 1.00~/1.00~/1.00 | 1.00~/1.00~/1.00 | 1.00~/1.00~/1.00 | 1.00~/1.00~/1.00 | 1.00~/1.00~/1.00 | 1.00~/1.00~/1.00 | 1.00~/1.00/1.00 | 1.00~/1.00/1.00* | 1.00~/1.00~/1.00~ |
| E1 R2 | 1.00*/-/- | 1.00*/-/- | 1.00*/-/- | 1.00~/1.00/1.00 | 1.00*/1.00*/1.00* | 1.00/1.00/1.00 | 1.00*/1.00*/1.00* | 1.00*/1.00*/1.00* | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 1.00*/1.00*/1.00* | 1.00*/1.00*/1.00* | .99*/1.00*/1.00* |
| E2 R1 | .93~/-/- | .94~/-/- | .94~/-/- | .32~/.36~/.38~ | .32~/.35~/.38~ | .32~/.35~/.38~ | .32~/.35~/.38~ | .32~/.35~/.38~ | .32~/.35~/.38~ | .32~/.35~/.38~ | .55*/.62~/.68~ | .56~/.62~/.68~ | .56*/.62~/.68 |
| E2 R2 | .97*/-/- | .97*/-/- | .96*/-/- | .36~/.55~/.79 | .55*/.76*/.88* | .36~/.54~/.78 | .53*/.73*/.85* | .53*/.73*/.85* | .28~/.42~/.71 | .28~/.42~/.70 | .49*/.63*/.86* | .54*/.67*/.91* | .50*/.63*/.88* |
| E3 R1 | 1.00~/-/- | 1.00~/-/- | 1.00~/-/- | 1.00/1.00~/1.00~ | 1.00~/1.00~/1.00~ | 1.00~/1.00~/1.00~ | 1.00~/1.00~/1.00~ | 1.00~/1.00~/1.00~ | 1.00~/1.00~/1.00~ | 1.00~/1.00~/1.00~ | 1.00~/1.00~/1.00~ | 1.00*/1.00*/1.00~ | 1.00*/1.00~/1.00~ |
| E3 R2 | 1.00*/-/- | 1.00*/-/- | 1.00*/-/- | 1.00/1.00/1.00~ | 1.00*/1.00*/1.00* | 1.00/1.00/1.00~ | 1.00*/1.00*/1.00* | 1.00*/1.00*/1.00* | 1.00/1.00/1.00~ | 1.00/1.00/1.00~ | .97*/1.00*/1.00* | 1.00*/1.00*/1.00* | .97*/1.00*/1.00* |
| E4 R1 l1 | .15~/-/- | .25~/-/- | .25~/-/- | .85~/1.00/1.00~ | .85~/1.00/1.00~ | .85~/1.00/1.00~ | .85~/1.00/1.00~ | .85~/1.00/1.00~ | .85~/1.00/1.00~ | .85~/1.00/1.00~ | .40/.95~/1.00~ | .40~/.95/1.00~ | .40~/.95/1.00~ |
| E4 R2 l1 | .02~/-/- | .10~/-/- | .03~/-/- | .05~/.18~/.85~ | .30~/.92~/1.00 | .05~/.18~/.85~ | .30~/.92~/1.00 | .28~/.92~/1.00 | .05~/.18~/.85~ | .05~/.18~/.85~ | .02~/.03~/.10~ | .13~/.25~/.95~ | .02~/.05~/.10~ |
| E4 R3 l0 | 1.00~/-/- | 1.00~/-/- | 1.00~/-/- | 1.00~/1.00/1.00~ | 1.00~/1.00/1.00~ | 1.00~/1.00/1.00~ | 1.00~/1.00/1.00~ | 1.00~/1.00/1.00~ | 1.00~/1.00/1.00~ | 1.00~/1.00/1.00~ | .97~/1.00~/1.00~ | .98~/1.00~/1.00~ | .98~/1.00~/1.00~ |
| E4 R3 l0.5 | 1.00*/-/- | 1.00*/-/- | 1.00*/-/- | 1.00~/1.00~/1.00~ | 1.00~/1.00~/1.00~ | 1.00~/1.00~/1.00~ | 1.00~/1.00~/1.00~ | 1.00~/1.00~/1.00~ | 1.00~/1.00~/1.00~ | 1.00~/1.00~/1.00~ | .97*/.98*/1.00* | .98~/1.00~/1.00* | .98~/1.00~/1.00~ |
| E4 R3 l1 | 1.00*/-/- | 1.00*/-/- | 1.00*/-/- | 1.00~/1.00~/1.00~ | 1.00~/1.00~/1.00~ | 1.00~/1.00~/1.00~ | 1.00~/1.00~/1.00~ | 1.00~/1.00~/1.00~ | 1.00~/1.00~/1.00~ | 1.00~/1.00~/1.00~ | .93*/1.00*/1.00* | .98~/1.00*/1.00* | .98~/1.00*/1.00* |
| E4 R3 l1.5 | 1.00*/-/- | 1.00*/-/- | 1.00*/-/- | 1.00~/1.00~/1.00~ | 1.00~/1.00~/1.00~ | 1.00~/1.00~/1.00~ | 1.00~/1.00~/1.00~ | 1.00~/1.00~/1.00~ | 1.00~/1.00~/1.00~ | 1.00~/1.00~/1.00~ | .98*/1.00*/1.00* | .98~/.98*/1.00* | .97~/1.00*/1.00* |
| E4 R4 l0 | .50~/-/- | .40~/-/- | .40~/-/- | 1.00~/1.00~/1.00~ | 1.00/1.00~/1.00~ | 1.00/1.00~/1.00~ | 1.00/1.00~/1.00~ | 1.00/1.00~/1.00~ | 1.00/1.00~/1.00~ | 1.00/1.00~/1.00~ | .65~/1.00~/1.00~ | .70~/1.00~/1.00~ | .70~/1.00~/1.00~ |
| E4 R4 l0.5 | 1.00*/-/- | 1.00*/-/- | 1.00*/-/- | 1.00*/1.00*/1.00* | 1.00*/1.00*/1.00* | 1.00*/1.00*/1.00* | 1.00*/1.00*/1.00* | 1.00*/1.00*/1.00* | 1.00*/1.00*/1.00* | 1.00*/1.00*/1.00* | 1.00*/1.00*/1.00* | 1.00*/1.00*/1.00* | 1.00*/1.00*/1.00* |
| E4 R4 l1 | 1.00*/-/- | 1.00*/-/- | 1.00*/-/- | 1.00*/1.00*/1.00* | 1.00*/1.00*/1.00* | 1.00*/1.00*/1.00* | 1.00*/1.00*/1.00* | 1.00*/1.00*/1.00* | 1.00*/1.00*/1.00* | 1.00*/1.00*/1.00* | 1.00*/1.00*/1.00* | 1.00*/1.00*/1.00* | 1.00*/1.00*/1.00* |
| E4 R4 l1.5 | 1.00*/-/- | 1.00*/-/- | 1.00*/-/- | 1.00*/1.00*/1.00* | 1.00*/1.00*/1.00* | 1.00*/1.00*/1.00* | 1.00*/1.00*/1.00* | 1.00*/1.00*/1.00* | 1.00*/1.00*/1.00* | 1.00*/1.00*/1.00* | 1.00*/1.00*/1.00* | 1.00*/1.00*/1.00* | 1.00*/1.00*/1.00* |
| E5 R1 | .68~/-/- | .67/-/- | .67/-/- | .37~/.47~/.62~ | .37~/.47~/.62~ | .37~/.47~/.62~ | .37~/.47~/.62~ | .37~/.47~/.62~ | .37~/.47~/.62~ | .37~/.47~/.62~ | .71*/.78~/.96* | .72*/.82~/.97~ | .74~/.82~/.97~ |
| E5 R2 | 1.00*/-/- | 1.00*/-/- | 1.00*/-/- | .53~/.61/.70~ | .64*/.80*/.92* | .53~/.61/.70~ | .60*/.74*/.88* | .59*/.74*/.88* | .41/.48/.64~ | .41/.48/.64~ | .68*/.71*/.89* | .70*/.77*/.97* | .67*/.70*/.88* |

## 3. Cost per (method, dataset): CPU-s mean / max over worlds, regimes, seeds (1 thread, Kaggle / Colab / Lightning CPU, platform per dataset)

Budget R-13: 7200 CPU-s.

| arm | n 500 | n 1000 | n 4000 | n 8000 | n 24000 | peak RSS MB |
|---|---|---|---|---|---|---|
| mscr_eq | 22.3 / 88 | 53.1 / 192 | - | - | - | 348 |
| mscr_native | 9.2 / 34 | 21.8 / 69 | - | - | - | 257 |
| mscr_eq_min | 38.9 / 333 | 99.1 / 719 | - | - | - | 260 |
| pcorr_eq | 0.0 / 0.2 | 0.1 / 0.3 | 0.2 / 0.8 | 0.4 / 1.8 | 1.4 / 6.9 | 332 |
| pcorr_native | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.1 | 0.1 / 0.2 | 0.3 / 0.9 | 332 |
| pcorr_eq_min | 0.0 / 0.2 | 0.0 / 0.2 | 0.1 / 0.7 | 0.2 / 1.6 | 0.8 / 5.1 | 332 |
| pcorr_hac | 0.1 / 0.3 | 0.1 / 0.4 | 0.3 / 1.1 | 0.6 / 2.1 | 2.0 / 12 | 343 |
| pcorr_hac_fb | 0.8 / 1.8 | 1.0 / 1.6 | 1.2 / 2.4 | 1.5 / 3.3 | 2.7 / 8.5 | 421 |
| pcorr_hac_eq_min | 0.1 / 0.3 | 0.1 / 0.4 | 0.3 / 1.4 | 0.6 / 2.2 | 2.0 / 8.9 | 343 |
| pcorr_hac_fb_eq_min | 0.8 / 1.6 | 1.0 / 1.6 | 1.2 / 2.3 | 1.5 / 3.1 | 2.7 / 7.9 | 421 |
| rcot2_eq | 0.5 / 1.6 | 0.6 / 2.4 | 2.0 / 5.8 | 3.3 / 9.9 | 9.0 / 27 | 335 |
| rcot2_native | 0.4 / 1.5 | 0.6 / 2.2 | 1.9 / 5.3 | 3.0 / 8.2 | 7.9 / 23 | 335 |
| rcot2_eq_min | 0.4 / 1.5 | 0.6 / 2.6 | 1.8 / 5.3 | 3.0 / 7.9 | 7.9 / 23 | 335 |

Infeasible (method, dataset): none (no unit near the budget).

## 4. Seed-count proposal (R-12, PROTOCOL_A rule T1, R-34)

- T1 (eval_analysis.t1_seed_count): kappa .25 cells at n 500 / 1000 / 4000 (E4 excluded), pmrt_eq vs each primary-block arm with the same declaration rule, neither INVALID; paired recall gap .15, alpha .05 two-sided, power .8: 108 pairs, S_power = **14** (max over cells), S = min(cap 100, max(40, S_power up to a multiple of 10)) = **40** (floor binds; cap binds: False). Minimum detectable gap at S: max .082.
- Largest per-pair requirements: granger_eq E3|R2|k0.25|n1000 14 (sd_d .18); pcorr_eq E3|R2|k0.25|n1000 14 (sd_d .18); notears E5|R2|k0.25|n1000 14 (sd_d .18); notears E5|R2|k0.25|n500 13 (sd_d .18); pc_eq E5|R2|k0.25|n1000 12 (sd_d .16); notears E5|R2|k0.25|n4000 12 (sd_d .17).
- E4 (R3 / R4): S_E4 by rule T9 (R-34 / R-39; not computed here).
- Per cell (max over its pairs; all pairs in t1_pairs.json): E3|R2|k0.25|n1000 14; E5|R2|k0.25|n1000 14; E5|R2|k0.25|n500 13; E5|R2|k0.25|n4000 12; E2|R2|k0.25|n500 10; E3|R2|k0.25|n500 10; E3|R2|k0.25|n4000 9; E5|R1|k0.25|n1000 9; E5|R1|k0.25|n500 9; E2|R2|k0.25|n1000 8; E5|R1|k0.25|n4000 8; E1|R2|k0.25|n500 5; others smaller.
- Null-rate precision (95 % CI half-width <= .02 at a true rate .05, DEV seed-cluster design effect): seeds needed, median / max over cells:
  null_decl: 39 / 93 (max: rcot2_eq_min|E5|R2|k0.125|n1000)
  null_raw: 39 / 113 (max: pcorr_eq|E5|R1|k0.25|n500)
  plac_raw: 457 / 481 (max: rcot2_native|E4|R4|lam1|k0.25|n500)

## 5. Notes

- INVALID cells by regime {'R2': 212, 'R3': 46, 'R4': 168, 'R1': 18}; by arm {'rcot2_eq': 64, 'rcot2_native': 56, 'rcot2_eq_min': 48, 'pcorr_hac_fb': 43, 'pcorr_hac': 43, 'pcorr_native': 43, 'mscr_eq_min': 28, 'mscr_eq': 28, 'mscr_native': 28, 'pcorr_eq_min': 16, 'pcorr_hac_eq_min': 16, 'pcorr_hac_fb_eq_min': 16, 'pcorr_eq': 15}. Reported, not tuned on (R-21).
- Scope (Q10 plan C; R-48 / R-49): pcorr x7 + rcot2 x3 on the full DEV grid, mscr x3 at n <= 1000 only; pdcor and cmi_knn are dropped from Study A (their pilot records, results/dev/pilot_ci_*, stay descriptive). Definitions from feat/v2's eval_analysis.py (xm-eval-analysis/4, copy in scratchpad/xmethod/_ref/) and analysis fields from feat/v2's specs/eval/full.json.
- mscr (all three arms): in R2 the truth-null raw-p rejection rate is .93-1.00 (P_placebo .91-.94 in E2 R2); 36 % / 16 % / 16 % of the eq / native / eq_min R2 records have EVERY candidate at the Monte Carlo floor p = 1e-4 (9 999 draws), e.g. E1 R2 n 1000 seed 3000100: 36 / 36 incl. P_placebo. So R2 and E4 R3 / R4 (lam > 0) cells are INVALID, R1 cells INCONCLUSIVE / VALID. This matches the arm's EVAL-spec label ("single-conditioner max statistic; cannot condition on the joint design set", c2b false): reported, not a run failure; flagged for the orchestrator (Q11).
- pcorr / pcorr_hac / pcorr_hac_fb native arms: INVALID in every R2 cell (truth-null raw .21-.46 at n 1000); the eq and eq_min arms of the same tests are VALID / INCONCLUSIVE there (.04-.06). rcot2 (all arms): INVALID in most R2 cells (.06-.20).
- E4 R4 at lam .5 / 1 / 1.5: every CI arm INVALID (as the 10 arms in results/dev/full); E4 R4 lam 0 is not.
- T1 here = the 10-arm records + these 13 arms, focal pmrt_eq (linear): 108 pairs, S_power 14 (pcorr_eq ties the 10-arm max at E3 R2 n 1000), S = 40 unchanged. R-43 makes pmrt_nl_eq the T1 focal: T1 is recomputed when its DEV run (xm-dev-pmrtnl-k1) is in.
- Run: 58 480 / 58 480 expected keys received, 0 missing, 0 errors; 39 duplicate input lines = Lightning-partial datasets re-run elsewhere, resolved to one platform per dataset (36 datasets switched, 0 mixed, 0 conflicts); merged output has one record per key. Platforms: Kaggle (k1-k5), Colab (c1b / c2 / c3), Lightning (l1-l3, stopped early for credits). Code: method code identical across the three recorded commits (e8c7ced; 3a251c6 / 9fd845b only add the launcher's --skip-complete-from). 17 176 records carry dirty = true: the Lightning and Colab c1b / c2 launches ran with an uncommitted launcher fix (scratchpad/e6_dev/xm_lightning.py argparse) in the tree, nothing under cdd_oran/.
- Cost: CPU-s per dataset mix Kaggle, Colab and Lightning CPUs (one platform per dataset); every unit far below the 7200 CPU-s budget (max 719 CPU-s, mscr_eq_min n 1000).
