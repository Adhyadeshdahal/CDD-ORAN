# X4: random accept / defer referee on the Study 3 EVAL episodes (POST HOC, DESCRIPTIVE)

Declared in `scratchpad/xmethod/EXTRAS_PROTOCOL.md`, Amendments 2026-10-05 (declaration commit 259899d), before
any X4 episode. X4 was added after the Study 3 results were known.
- It tests no hypothesis and has no criterion.
- Study 3's verdict (PARTIAL: E, D1, D2 pass; D3 fails; S1 passes) and every frozen file are unchanged.
- Code: `scratchpad/e6_dev/x4_random_defer.py` (commit 4b6e518), a new file that imports the frozen certsafe driver
  and edits nothing. Test: `tests/test_x4_random_defer.py`.
- Numbers: `x4/x4_tables.json`. Records: `x4/records/`.

## Setup

**Policy X4:rand@p.**
- Each request reaching the referee is deferred independently with probability p, from t = 0, with every family
  and direction treated alike.
- One uniform per request comes from `default_rng([seed, 6640])`. The stream is shared by the three p (common
  random numbers).

**The three p.**
- **p = .352** is MG:PMRT's request-level deferral rate in the certsafe EVAL: 37 890 deferred of 107 525
  requests, 160 seeds.
- For reference, the composition's unit-level rate is 9 253 of 35 058 directional units = .264.
- **.25 and .50.**

**Episodes.**
- The 160 Study 3 EVAL seeds 191100-191259, with the same plant (P3, L40, 120 s warm-up + 600 s scored).
- 480 X4 episodes plus 4 provenance episodes, all present. 0 missing, 0 bad lines, 7 / 7 processes exit 0.

**Platform.**
- VPS (AMD EPYC-Rome), capped systemd scope with 7 processes, 2026-10-05 12:54-13:31Z.
- Python 3.12.14, numpy 2.4.2, scipy 1.18.1, glibc 2.39. Pin check: 0 mismatches.

**Provenance.**
- noarb and never_sleep were re-run on the VPS on seeds 191100 and 191101 with the frozen driver's arbiters.
- All 4 are **bit-identical** to the stored Kaggle records in every outcome field and in the decision counts.
- So X4 and the stored arms share the plant's numerics: the comparison is not cross-platform.

**Scoring.**
- The Study 3 code, unchanged: `e6p_conf_analyze.arm_stats_multi` on the stored 26 arms plus X4.
- One paired seed bootstrap: N 10 000, `default_rng([6624, 20, 160])`, the index matrix of `cs_eval.json`.
- V_AA 175.30, V_ref 116.82 (sub:SliceGuarantee), den 58.48.
- All 26 stored arms reproduce `cs_eval.json`: V, R, R*, retention, guard ratios, eligibility and the R / guard
  90 % CIs.

**Cost.**
- 4.12 VPS CPU-h for X4.
- Converted to Kaggle-reference CPU-h:
  - **10.5 [7.9, 11.6]** with the pooled R-55 factor `factors.json` "*" (2.56 [1.91, 2.81]);
  - **9.7** with this workload's direct factor, 2.34 (the mean of the 4 provenance pairs, 2.16-2.58).

## Table (160 seeds; 90 % CIs, paired seed bootstrap)

Defer rate = deferred / requests reaching the referee. Guard ratios are each arm vs accept-all (noarb).
Eligible = retention >= .90 and every guard ratio <= 1.10 (point).

| arm | defer rate | R [90 % CI] | R* | retention [90 % CI] | svr | nonprot eMBB | LL | RLF [90 % CI] | eligible |
|---|---|---|---|---|---|---|---|---|---|
| noarb (accept-all) | 0.000 | +0.000 [+0.000, +0.000] | +0.000 | 0.995 [0.990, 1.000] | 1.00 | 1.00 | 1.00 | 1.00 [1.00, 1.00] | yes |
| never_sleep | 0.048 | +0.256 [+0.169, +0.339] | +0.256 | 1.146 [1.129, 1.164] | 0.88 | 0.88 | 0.86 | 0.96 [0.93, 0.99] | yes |
| MG:PMRT | 0.352 | +0.201 [+0.025, +0.363] | +0.201 | 1.634 [1.584, 1.686] | 0.89 | 0.88 | 0.96 | 1.05 [0.95, 1.16] | yes |
| CS:PMRT | 0.048 | +0.256 [+0.169, +0.339] | +0.256 | 1.146 [1.129, 1.164] | 0.88 | 0.88 | 0.86 | 0.96 [0.93, 0.99] | yes |
| CS:GT | 0.131 | +0.221 [+0.119, +0.319] | +0.221 | 1.294 [1.270, 1.319] | 0.90 | 0.90 | 0.91 | 0.94 [0.90, 0.97] | yes |
| **X4:rand@0.352** | 0.353 | -0.456 [-0.613, -0.325] | -0.456 | 1.014 [0.999, 1.030] | 1.00 | 0.99 | 0.96 | 0.98 [0.95, 1.00] | yes |
| **X4:rand@0.25** | 0.250 | -0.218 [-0.349, -0.105] | -0.218 | 1.018 [1.003, 1.033] | 0.99 | 0.99 | 0.94 | 0.99 [0.96, 1.01] | yes |
| **X4:rand@0.5** | 0.502 | -0.646 [-0.842, -0.483] | -0.646 | 1.031 [1.013, 1.050] | 0.99 | 0.98 | 0.91 | 0.95 [0.92, 0.98] | yes |

Paired differences in R (X4 minus comparator, 90 % CI):

| X4 arm | vs noarb | vs never_sleep | vs MG:PMRT | vs CS:PMRT | vs CS:GT |
|---|---|---|---|---|---|
| X4:rand@0.352 | -0.456 [-0.613, -0.325] | -0.712 [-0.872, -0.572] | -0.657 [-0.868, -0.461] | -0.712 [-0.872, -0.572] | -0.677 [-0.837, -0.535] |
| X4:rand@0.25 | -0.218 [-0.349, -0.105] | -0.474 [-0.613, -0.354] | -0.418 [-0.603, -0.243] | -0.474 [-0.613, -0.354] | -0.438 [-0.582, -0.311] |
| X4:rand@0.5 | -0.646 [-0.842, -0.483] | -0.902 [-1.100, -0.730] | -0.846 [-1.075, -0.643] | -0.902 [-1.100, -0.730] | -0.866 [-1.061, -0.696] |

**Deferred requests by family**, pre + post warm-up, 160 seeds:

| arm | prot_min | ptx | carrier | sleep |
|---|---|---|---|---|
| X4 .352 | 21 190 | 6 631 | 4 204 | 1 018 |
| X4 .25 | 14 902 | 4 078 | 2 797 | 724 |
| X4 .50 | 31 592 | 11 375 | 6 885 | 1 575 |
| MG:PMRT | 0 | 33 740 | 328 | 3 822 |

MG:GT is not a Study 3 arm and has no record on these seeds, so CS:GT is shown. The option (a) MG:GT ran on other
seeds (187000-187159) and is not comparable here.

## Reading (descriptive)

1. **Volume.** Deferring requests at random, at MG:PMRT's own volume (35 %), stayed eligible but lost: R = -.46
   [-.61, -.33] vs accept-all, and -.66 [-.87, -.46] vs MG:PMRT, paired. R fell further as p grew (-.22, -.46, -.65
   at p .25, .352, .50).
2. **Target.** MG:PMRT's and the CS arms' positive R comes from which requests they defer (ptx-up and sleep-up;
   retention 1.63 for MG:PMRT), not from how many. Random deferral hits mostly prot_min, the most frequent request
   (64 % of its deferrals at p .352), and leaves energy near accept-all (retention 1.01-1.03).
3. **Guards.** No random arm came near a guard limit (every ratio <= 1.00 point; highest upper 90 % bound 1.01). So
   in this plant, deferral volume alone neither breaks the 1.10 guards nor produces the gains.
