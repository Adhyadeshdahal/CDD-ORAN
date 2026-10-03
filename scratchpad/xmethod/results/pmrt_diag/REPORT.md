# pmrt-diag: is pmrt_eq liberal on E2 R2 truth-null edges at n 1000?

Setup:
- Branch xm/pmrt-diag (from feat/v2 79c96ee); pmrt_core is untouched.
- Seeds: DEV 3_000_100-159. Reserve DEV 160-189 was used once, on rep -1 only. No EVAL seed was touched.
- Rate = truth-null raw p <= .05 over 32 action edges/dataset (P_placebo separate); CI = seed-cluster bootstrap.
- Runner `scratchpad/xmethod/pmrt_diag.py`: `prod` matches PmrtCore.run() (same p; z to 1e-14) and Kaggle DEV p.
- "Regen" replicates keep a DEV seed's setpoints and noise. They redraw every dither from its declared design, on
  stream SeedSequence([7801, 99, seed, world, regime, purpose, rep]), so they are draws from the law the CRT assumes.
- Small synthetic analogue: regen plus E2 R1 (i.i.d.). No separate toy model (orchestrator ruling on Q1).

## Verdict
There is no flaw in pmrt_core. On fresh dithers, production is nominal in every cell tested: E2 R2 n1000 k.5 is
.050 [.047, .054] on 600 replicates. The DEV excess comes from the realised datasets (H6). Selection made it look
larger: the cell was the worst of many, and kappa .125 / .5 share seeds 100-119. The theoretical gap (H1) is real
but at most .003. The fix is therefore wording plus an optional exact variant, not a production change.

## Theory
The fixed-W CRT is exact iff W is independent of the focal dither v under H0 and the redraw uses v's true law.
- **Production W.** pmrt_core's W (expanding ridge on lag KPIs, setpoints, every action @t-1/@t-2, concurrent
  actions) is predictable, but it contains the focal action's own past dithers. Under a lag-1 null, the lag KPIs can
  also carry the effect of earlier lags. S = sum v_t W_t is then a martingale sum: valid asymptotically (martingale
  CLT), not exactly.
- **E2.** There is no KPI memory (KPI_{t+1} = f(actions_t) + noise), so a truth-null (a, k) implies the global
  sharp null H0g: Y_k is invariant to every dither of a. Verified: max |dY_k| = 0 (10 seeds x 8 actions; minimum
  3.34 on true edges). A W built only from Y_k's own lags and non-focal actions/setpoints (`inv`) is EXACT in E2.
- **Dynamic worlds (E1/E3/E5).** A lag-1 null does not imply H0g, so no Y-based W is invariant and no exact
  Y-based fixed-W CRT exists.

## Hypotheses (E2 R2 n1000 k.5 unless stated)
| H | test | numbers | holds? |
|---|---|---|---|
| H1 own/KPI lags held fixed | prod vs exact `inv`, paired | DEV 100-159 .064 vs .065, diff -.001 [-.010, .007]; regen .0501 vs .0489, +.0012 [-.0016, .0039]; grid max +.003 [.000, .005] (k.125) | real, negligible |
| H2 redraw law != realised | KS vs U(-w, w); lag-1 corr; var ratio | KS p .32-.93; corr ~0; ratio 1.0002 | no |
| H3 overlapping rows | E2 has no memory, so this acts only through W (= H1); nolagkpi | nolagkpi - prod +.004 (DEV); dither cross-column / lag-1 z sd .98 / .99 | no (is H1) |
| H4 concurrent actions | noconc vs prod | DEV diff 0.0 (identical decisions) | no |
| H5 MC / Besag-Clifford | fixed B 9999 (prodB/invB) vs h 20; 60 DEV + 180 regen datasets | prodB - prod +.0001 [-.0017, .0020]; 74 / 9120 decisions flip (39 vs 35) | no |
| H6 chance + selection | regen distribution of the DEV cell; reserve seeds | below | yes |

H6 detail:
- **The DEV cells are extreme draws.**
  - The 20-seed DEV cell (.084) sits in a regen distribution of .055 +- .009: P(sim >= dev) .0005, P(INVALID) .069.
  - The 60-seed cell (.064) has P .004. The exact `inv` is just as extreme on it (.065, P .0008), so the data are
    high, not the test.
  - D-vs-Y on the null pairs: z sd 1.02, with 5.6 % of |z| > 1.96.
- **Selection.** The cell was the worst of about 140 correlated pmrt DEV cells. kappa .125 / .5 reuse seeds
  100-119, so .070 / .084 are one event.
- **Reserve seeds 160-189:**

  | k | prod | R-30 | inv | r3 | pcorr_eq |
  |---|---|---|---|---|---|
  | .5 | .051 [.039, .065] | VALID | .057 | .046 | .053 |
  | .25 | .060 [.045, .077] | INCONCLUSIVE | .059 | .053 | .050 |
  | .125 | .060 [.044, .078] | INCONCLUSIVE | .065 | .060 | .049 |

  No cell is INVALID. pcorr_eq on DEV 100-159 is .055 [.045, .066].

## Grid (Kaggle grid-1, seeds 100-159; 600 regen datasets per cell, plus the 60 DEV rep -1 datasets)
| cell | regen prod | regen inv | prod - inv (paired) | DEV prod / inv |
|---|---|---|---|---|
| E2 R2 n500 k.5 | .051 [.047, .054] | .051 | -.000 [-.003, .003] | .048 / .053 |
| E2 R2 n1000 k.5 | .050 [.047, .054] | .049 | +.001 [-.002, .004] | .064 / .065 |
| E2 R2 n4000 k.5 | .049 [.045, .052] | .048 | +.000 [-.003, .003] | .051 / .044 |
| E2 R2 n1000 k.25 | .051 [.048, .055] | .050 | +.002 [-.001, .004] | .062 / .063 |
| E2 R2 n1000 k.125 | .052 [.048, .055] | .049 | +.003 [.000, .005] | .059 / .065 |
| E2 R1 n1000 k.5 | .050 [.048, .053] | .051 | -.001 [-.003, .001] | .047 / .048 |
| E1 R2 n1000 k.25 | .049 [.044, .054] | .050 | -.001 [-.004, .002] | .039 / .043 |
| E3 R2 n1000 k.25 | .049 [.044, .055] | .051 | -.002 [-.004, .001] | .046 / .054 |
| E5 R2 n1000 k.25 | .053 [.047, .060] | .052 | +.002 [-.002, .005] | .057 / .058 |

Every regen cell is nominal (P_placebo .044-.056, sd_z .99-1.01, r3 / noown within .003 of prod). There is no n
trend: the DEV n1000 excess is absent at n 500 and n 4000.

## Proposed fix (from theory, not from matching rates)
1. Keep pmrt_core. In PMRT_CORE.md, its validity claim should read "asymptotically valid (martingale CLT); exact
   only when W is invariant to the focal dither", replacing any unqualified "exact".
2. Add `inv` as an optional exact arm for designs where the null implies H0g (no-memory worlds). Per target k, W
   uses Y_k's own lags plus non-focal actions, setpoints and concurrent actions; it drops the focal action's own
   lags and other KPIs' lags. It is exact because W is then a function of quantities invariant to the focal dither
   under H0g.
3. Do not use `inv` in dynamic worlds: it is not exact there, and the grid shows no validity gain.

Power cost of `inv` vs prod (paired, same seeds; BY q .05 recall on true edges):
- **E2 R2 k.5 DEV:** .242 vs .267, -.025 [-.044, -.007]. Raw recall is equal (.421).
- **E2 reserve:** -.017 [-.033, -.002].
- **Other worlds:** E1 -.013 (prod 1.0); E3 0; E5 -.008.
- **Comparators:** noown costs -.035 in E2; r3 gains +.037 [.020, .054].

## Other eq arms
- **H1 is specific to fixed-W randomisation tests** that redraw v while holding a v-dependent W. Other pmrt arms
  (r3, any W with own/KPI lags) share prod's asymptotic-only validity. Regen shows r3 is nominal everywhere.
- **CI-test eq arms (e.g. pcorr_eq) do not redraw.** They rest on regression assumptions, which this diagnosis does
  not test.
- **H6 affects every arm on the same DEV datasets.** pcorr_eq is .055 on DEV 100-159. Read any arm's DEV
  truth-null rate on seeds 100-119 with the common-random-numbers and selection caveat.

Files (results/pmrt_diag/): premise_check, dither_independence, cell_null_dist, *summary.json, power/, grid/, and
raw/*.jsonl (grid_*.jsonl.gz = Kaggle grid-1), next to the scripts that made them.
