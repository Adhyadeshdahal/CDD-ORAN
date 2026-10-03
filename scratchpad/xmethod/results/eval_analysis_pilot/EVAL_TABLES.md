# Study A EVAL tables (PROVISIONAL)

> PLUMBING TEST ONLY (worker protocol, 2026-10-03, `xm-eval-analysis/2`): dev-runs pilot records (xm/dev-runs,
> 1648 units, code 5999ab7), seed 3000100 relabelled tune (`--as-tune`), 1 measurement seed (3000101),
> `--min-flag-seeds 1`. With one seed the bootstrap CI is a point, so validity classes, F_max and verdicts below
> mean nothing. Command: `eval_analysis.py --spec specs/dev/pilot.json --merged pilot/merged.jsonl.gz --out .
> --as-tune 3000100 --min-flag-seeds 1`

`xm-eval-analysis/2`, spec `dev_pilot`, primary kappa 0.25; commits 5999ab729. Protocol: docs/xmethod/PROTOCOL_A.md.

## V4 claim verdicts

- **Claim: NOT SUPPORTED** (components {'C1': 'NOT EVALUABLE', 'C2a': 'SUPPORTED', 'C2b': 'NOT SUPPORTED', 'C3': 'NOT SUPPORTED'})
- C1: **NOT EVALUABLE** (2 / 2 counted arms of D are design-blind failures; |D| = 2). Wording: design-blind tests are invalid on the R2 (setpoint + dither) design tested.
  - corr: **FAILURE**; R2 INVALID 20/25 (counted 25), pooled R2 null_raw .846 [.818, .892] INVALID; plac_raw .780 [.575, .926] INVALID; R1 INVALID 8/25 (F_max 16)
  - granger_native: **FAILURE**; R2 INVALID 5/5 (counted 5), pooled R2 null_raw .738 [.738, .738] INVALID; plac_raw .920 [.920, .920] INVALID; R1 INVALID 3/5 (F_max 5)
- C2a pmrt_eq valid in R1 / R2: **SUPPORTED**
  - cells 50/50, INVALID 18 (F_max 30), VALID 32, INCONCLUSIVE 0; pooled null_raw .031 [.009, .048] VALID; plac_raw .055 [.038, .070] VALID
- C2b eq arms whose native partner is a C1 failure: **NOT SUPPORTED**
  - granger_eq (native granger_native): **NOT SUPPORTED**
    - cells 10/10, INVALID 7 (F_max 9), VALID 3, INCONCLUSIVE 0; pooled null_raw .056 [.056, .056] INVALID; plac_raw .060 [.060, .060] INVALID
- C2 wording: design-based inference restores validity; adding design covariates does not suffice.
- C3 pmrt_eq valid in E4 R3: **NOT SUPPORTED**
  - cells 20/20, INVALID 2 (F_max 7), VALID 18, INCONCLUSIVE 0; pooled plac_raw .000 [.000, .000] VALID; conf_raw .100 [.100, .100] INVALID
  - wording: not supported.
- C3 reported for every eq arm (no claim effect): granger_eq NOT EVALUABLE; pc_eq NOT SUPPORTED

## V1 validity: INVALID / VALID / counted cells per world-regime (all n)

| world regime | pmrt_eq | corr | granger_eq | granger_native | notears | pc_eq | pc_native | pmrt_r3 | shap_dag | two_tower |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 R1 | 0/5/5 | 0/5/5 | - | - | 0/5/5 | **2**/3/5 | **2**/3/5 | **1**/4/5 | **4**/1/5 | **3**/2/5 |
| E1 R2 | **3**/2/5 | **5**/0/5 | - | - | **5**/0/5 | **5**/0/5 | **2**/3/5 | **3**/2/5 | **5**/0/5 | **5**/0/5 |
| E2 R1 | **2**/3/5 | **3**/2/5 | - | - | 0/5/5 | **3**/2/5 | **3**/2/5 | **2**/3/5 | **3**/2/5 | **3**/2/5 |
| E2 R2 | **2**/3/5 | **5**/0/5 | - | - | **5**/0/5 | **4**/1/5 | **4**/1/5 | **3**/2/5 | **3**/2/5 | **1**/4/5 |
| E3 R1 | **4**/1/5 | **3**/2/5 | **4**/1/5 | **3**/2/5 | 0/5/5 | **3**/2/5 | **4**/1/5 | **3**/2/5 | **5**/0/5 | **5**/0/5 |
| E3 R2 | **4**/1/5 | **5**/0/5 | **3**/2/5 | **5**/0/5 | **5**/0/5 | **2**/3/5 | **5**/0/5 | **4**/1/5 | **5**/0/5 | **5**/0/5 |
| E4 R1 l1 | 0/5/5 | 0/5/5 | - | - | 0/0/5 | 0/0/5 | 0/0/5 | 0/5/5 | 0/0/5 | 0/0/5 |
| E4 R2 l1 | 0/5/5 | 0/5/5 | - | - | 0/0/5 | 0/0/5 | 0/0/5 | 0/5/5 | 0/0/5 | 0/0/5 |
| E4 R3 l0 | 0/5/5 | 0/5/5 | - | - | 0/5/5 | **2**/3/5 | **2**/3/5 | 0/5/5 | 0/5/5 | **3**/2/5 |
| E4 R3 l0.5 | **1**/4/5 | **5**/0/5 | - | - | 0/5/5 | **4**/1/5 | **4**/1/5 | **1**/4/5 | **5**/0/5 | **5**/0/5 |
| E4 R3 l1 | **1**/4/5 | **5**/0/5 | - | - | 0/5/5 | **3**/2/5 | **4**/1/5 | **1**/4/5 | **5**/0/5 | **5**/0/5 |
| E4 R3 l1.5 | 0/5/5 | **5**/0/5 | - | - | 0/5/5 | **2**/3/5 | **2**/3/5 | 0/5/5 | **5**/0/5 | **5**/0/5 |
| E4 R4 l0 | **1**/4/5 | **1**/4/5 | - | - | 0/5/5 | **4**/1/5 | **4**/1/5 | **1**/4/5 | **3**/2/5 | **3**/2/5 |
| E4 R4 l0.5 | **1**/4/5 | **5**/0/5 | - | - | **5**/0/5 | **5**/0/5 | **5**/0/5 | **1**/4/5 | **5**/0/5 | **5**/0/5 |
| E4 R4 l1 | **1**/4/5 | **5**/0/5 | - | - | **5**/0/5 | **5**/0/5 | **5**/0/5 | 0/5/5 | **5**/0/5 | **5**/0/5 |
| E4 R4 l1.5 | 0/5/5 | **5**/0/5 | - | - | **5**/0/5 | **5**/0/5 | **5**/0/5 | 0/5/5 | **5**/0/5 | **5**/0/5 |
| E5 R1 | **1**/4/5 | **2**/3/5 | - | - | 0/5/5 | **3**/2/5 | **3**/2/5 | **1**/4/5 | **1**/4/5 | 0/5/5 |
| E5 R2 | **2**/3/5 | **5**/0/5 | - | - | **5**/0/5 | **5**/0/5 | **4**/1/5 | **2**/3/5 | **4**/1/5 | **4**/1/5 |

(Nx) = planned cells not counted (missing, infeasible, T3-infeasible, untuned or < 10 seeds).

INVALID cells (395; first 30):
- corr|E1|R2|k0.25|n1000: null_raw .750 [.750, .750]; null_decl .500 [.500, .500]; plac_raw 1.000 [1.000, 1.000]; plac_decl .750 [.750, .750]
- corr|E1|R2|k0.25|n24000: null_raw 1.000 [1.000, 1.000]; null_decl 1.000 [1.000, 1.000]; plac_raw 1.000 [1.000, 1.000]; plac_decl 1.000 [1.000, 1.000]
- corr|E1|R2|k0.25|n4000: null_raw .833 [.833, .833]; null_decl .833 [.833, .833]; plac_raw 1.000 [1.000, 1.000]; plac_decl 1.000 [1.000, 1.000]
- corr|E1|R2|k0.25|n500: null_raw .583 [.583, .583]; null_decl .333 [.333, .333]; plac_raw 1.000 [1.000, 1.000]; plac_decl .750 [.750, .750]
- corr|E1|R2|k0.25|n8000: null_raw .917 [.917, .917]; null_decl .917 [.917, .917]; plac_raw 1.000 [1.000, 1.000]; plac_decl 1.000 [1.000, 1.000]
- corr|E2|R1|k0.25|n1000: plac_raw .167 [.167, .167]
- corr|E2|R1|k0.25|n4000: null_raw .062 [.062, .062]; plac_raw .167 [.167, .167]
- corr|E2|R1|k0.25|n8000: plac_raw .167 [.167, .167]
- corr|E2|R2|k0.25|n1000: null_raw .750 [.750, .750]; null_decl .594 [.594, .594]; plac_raw .667 [.667, .667]; plac_decl .667 [.667, .667]
- corr|E2|R2|k0.25|n24000: null_raw .969 [.969, .969]; null_decl .969 [.969, .969]; plac_raw .833 [.833, .833]; plac_decl .833 [.833, .833]
- corr|E2|R2|k0.25|n4000: null_raw .812 [.812, .812]; null_decl .781 [.781, .781]; plac_raw .833 [.833, .833]; plac_decl .833 [.833, .833]
- corr|E2|R2|k0.25|n500: null_raw .594 [.594, .594]; null_decl .406 [.406, .406]; plac_raw .500 [.500, .500]; plac_decl .500 [.500, .500]
- corr|E2|R2|k0.25|n8000: null_raw .969 [.969, .969]; null_decl .938 [.938, .938]; plac_raw .833 [.833, .833]; plac_decl .833 [.833, .833]
- corr|E3|R1|k0.25|n1000: null_raw .125 [.125, .125]
- corr|E3|R1|k0.25|n24000: null_raw .062 [.062, .062]
- corr|E3|R1|k0.25|n500: null_raw .188 [.188, .188]
- corr|E3|R2|k0.25|n1000: null_raw .938 [.938, .938]; null_decl .812 [.812, .812]; plac_raw 1.000 [1.000, 1.000]; plac_decl 1.000 [1.000, 1.000]
- corr|E3|R2|k0.25|n24000: null_raw 1.000 [1.000, 1.000]; null_decl 1.000 [1.000, 1.000]; plac_raw 1.000 [1.000, 1.000]; plac_decl 1.000 [1.000, 1.000]
- corr|E3|R2|k0.25|n4000: null_raw 1.000 [1.000, 1.000]; null_decl 1.000 [1.000, 1.000]; plac_raw 1.000 [1.000, 1.000]; plac_decl 1.000 [1.000, 1.000]
- corr|E3|R2|k0.25|n500: null_raw .562 [.562, .562]; null_decl .500 [.500, .500]; plac_raw .800 [.800, .800]; plac_decl .800 [.800, .800]
- corr|E3|R2|k0.25|n8000: null_raw 1.000 [1.000, 1.000]; null_decl 1.000 [1.000, 1.000]; plac_raw 1.000 [1.000, 1.000]; plac_decl 1.000 [1.000, 1.000]
- corr|E4|R3|lam0.5|k0.25|n1000: conf_raw 1.000 [1.000, 1.000]; conf_decl 1.000 [1.000, 1.000]
- corr|E4|R3|lam0.5|k0.25|n24000: conf_raw 1.000 [1.000, 1.000]; conf_decl 1.000 [1.000, 1.000]
- corr|E4|R3|lam0.5|k0.25|n4000: conf_raw 1.000 [1.000, 1.000]; conf_decl 1.000 [1.000, 1.000]
- corr|E4|R3|lam0.5|k0.25|n500: conf_raw 1.000 [1.000, 1.000]; conf_decl 1.000 [1.000, 1.000]
- corr|E4|R3|lam0.5|k0.25|n8000: conf_raw 1.000 [1.000, 1.000]; conf_decl 1.000 [1.000, 1.000]
- corr|E4|R3|lam1.5|k0.25|n1000: conf_raw 1.000 [1.000, 1.000]; conf_decl 1.000 [1.000, 1.000]
- corr|E4|R3|lam1.5|k0.25|n24000: conf_raw 1.000 [1.000, 1.000]; conf_decl 1.000 [1.000, 1.000]
- corr|E4|R3|lam1.5|k0.25|n4000: conf_raw 1.000 [1.000, 1.000]; conf_decl 1.000 [1.000, 1.000]
- corr|E4|R3|lam1.5|k0.25|n500: conf_raw 1.000 [1.000, 1.000]; conf_decl 1.000 [1.000, 1.000]

## V2 recall among cells not INVALID (n 500 / 1000 / 4000 / 8000 / 24000; inv INVALID, inf infeasible, mis missing, unt untuned, few < 10 seeds)

| world regime | pmrt_eq | corr | granger_eq | granger_native | notears | pc_eq | pc_native | pmrt_r3 | shap_dag | two_tower |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 R1 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | - | - | 1.00/1.00/1.00/1.00/1.00 | inv/inv/1.00/1.00/1.00 | inv/inv/1.00/1.00/1.00 | inv/1.00/1.00/1.00/1.00 | inv/inv/inv/inv/1.00 | 1.00/1.00/inv/inv/inv |
| E1 R2 | inv/inv/inv/1.00/1.00 | inv/inv/inv/inv/inv | - | - | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/1.00/1.00/1.00/inv | inv/inv/inv/1.00/1.00 | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv |
| E2 R1 | .25/inv/inv/.38/.38 | .31/inv/inv/inv/.38 | - | - | .25/.25/.25/.25/.25 | .38/.31/inv/inv/inv | .31/.44/inv/inv/inv | .25/inv/inv/.38/.38 | inv/inv/1.00/inv/1.00 | .75/.88/inv/inv/inv |
| E2 R2 | .12/.25/inv/.50/inv | inv/inv/inv/inv/inv | - | - | inv/inv/inv/inv/inv | inv/inv/inv/.56/inv | inv/inv/inv/inv/.00 | inv/inv/inv/.56/.88 | .75/inv/inv/.94/inv | .75/inv/.75/.38/.50 |
| E3 R1 | 1.00/inv/inv/inv/inv | inv/inv/1.00/1.00/inv | inv/inv/inv/1.00/inv | inv/inv/1.00/1.00/inv | 1.00/1.00/1.00/1.00/1.00 | 1.00/inv/inv/1.00/inv | inv/inv/inv/1.00/inv | 1.00/inv/inv/1.00/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv |
| E3 R2 | 1.00/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/1.00/inv/inv/1.00 | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/1.00/inv/1.00/1.00 | inv/inv/inv/inv/inv | inv/inv/1.00/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv |
| E4 R1 l1 | .00/1.00/1.00/1.00/1.00 | .00/1.00/1.00/1.00/1.00 | - | - | .00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | .00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 |
| E4 R2 l1 | .00/.00/.00/.00/.00 | .00/.00/1.00/1.00/1.00 | - | - | .00/.00/.00/.00/.00 | 1.00/.00/1.00/.00/1.00 | 1.00/.00/1.00/1.00/1.00 | .00/.00/.00/.00/.00 | .00/1.00/1.00/1.00/1.00 | .00/1.00/1.00/1.00/1.00 |
| E4 R3 l0 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | - | - | 1.00/1.00/1.00/1.00/1.00 | inv/1.00/1.00/1.00/inv | inv/1.00/1.00/1.00/inv | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/inv/inv/inv/1.00 |
| E4 R3 l0.5 | 1.00/1.00/1.00/1.00/inv | inv/inv/inv/inv/inv | - | - | .00/.00/.00/.00/.00 | inv/inv/1.00/inv/inv | inv/inv/1.00/inv/inv | 1.00/1.00/1.00/1.00/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv |
| E4 R3 l1 | 1.00/1.00/1.00/1.00/inv | inv/inv/inv/inv/inv | - | - | .00/.00/.00/.00/.00 | inv/1.00/1.00/inv/inv | inv/1.00/inv/inv/inv | 1.00/1.00/1.00/1.00/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv |
| E4 R3 l1.5 | 1.00/1.00/1.00/1.00/1.00 | inv/inv/inv/inv/inv | - | - | .00/.00/.00/.00/.00 | 1.00/1.00/1.00/inv/inv | 1.00/1.00/1.00/inv/inv | 1.00/1.00/1.00/1.00/1.00 | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv |
| E4 R4 l0 | .00/.00/.00/inv/.00 | 1.00/1.00/1.00/inv/1.00 | - | - | 1.00/1.00/1.00/1.00/1.00 | inv/inv/inv/inv/1.00 | inv/inv/inv/inv/1.00 | .00/.00/.00/inv/.00 | inv/1.00/inv/inv/1.00 | 1.00/1.00/inv/inv/inv |
| E4 R4 l0.5 | .00/.00/.00/inv/.00 | inv/inv/inv/inv/inv | - | - | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | .00/.00/.00/inv/.00 | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv |
| E4 R4 l1 | .00/.00/.00/inv/.00 | inv/inv/inv/inv/inv | - | - | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | .00/.00/.00/.00/.00 | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv |
| E4 R4 l1.5 | .00/.00/.00/.00/.00 | inv/inv/inv/inv/inv | - | - | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | .00/.00/.00/.00/.00 | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv |
| E5 R1 | .33/.50/.50/inv/.67 | .33/inv/.50/inv/.67 | - | - | .50/.50/.33/.33/.33 | .50/.50/inv/inv/inv | .50/.67/inv/inv/inv | .33/.50/.50/inv/.67 | .83/1.00/inv/1.00/1.00 | .67/.67/.67/.67/.67 |
| E5 R2 | .50/.33/.50/inv/inv | inv/inv/inv/inv/inv | - | - | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/.00 | .50/.33/inv/inv/.50 | inv/inv/.67/inv/inv | .67/inv/inv/inv/inv |

## V3 paired recall difference pmrt_eq - arm (neither INVALID, same declaration rule; per-cell 95 % paired-t CI, descriptive)

| arm | rule | block | cells pmrt higher | pmrt lower | CI covers 0 |
|---|---|---|---|---|---|

## V5 information levels in R2 (pmrt, eq, eq_min, native): recall (* INVALID, ~ INCONCLUSIVE)

| row | pmrt_eq | granger_eq | granger_native | pc_eq | pc_native | pmrt_r3 |
|---|---|---|---|---|---|---|
| E1 n500 | 1.00* | - | - | 1.00* | 1.00* | 1.00* |
| E1 n1000 | 1.00* | - | - | 1.00* | 1.00 | 1.00* |
| E1 n4000 | 1.00* | - | - | 1.00* | 1.00 | 1.00* |
| E1 n8000 | 1.00 | - | - | 1.00* | 1.00 | 1.00 |
| E1 n24000 | 1.00 | - | - | 1.00* | 1.00* | 1.00 |
| E2 n500 | .12 | - | - | .31* | .19* | .25* |
| E2 n1000 | .25 | - | - | .56* | .44* | .25* |
| E2 n4000 | .44* | - | - | .50* | .50* | .50* |
| E2 n8000 | .50 | - | - | .56 | .50* | .56 |
| E2 n24000 | .88* | - | - | .62* | .00 | .88 |
| E3 n500 | 1.00 | 1.00* | 1.00* | 1.00* | 1.00* | 1.00* |
| E3 n1000 | 1.00* | 1.00 | 1.00* | 1.00 | 1.00* | 1.00* |
| E3 n4000 | 1.00* | 1.00* | 1.00* | 1.00* | 1.00* | 1.00 |
| E3 n8000 | 1.00* | 1.00* | 1.00* | 1.00 | 1.00* | 1.00* |
| E3 n24000 | 1.00* | 1.00 | 1.00* | 1.00 | 1.00* | 1.00* |
| E4 n500 | .00 | - | - | 1.00~ | 1.00~ | .00 |
| E4 n1000 | .00 | - | - | .00~ | .00~ | .00 |
| E4 n4000 | .00 | - | - | 1.00~ | 1.00~ | .00 |
| E4 n8000 | .00 | - | - | .00~ | 1.00~ | .00 |
| E4 n24000 | .00 | - | - | 1.00~ | 1.00~ | .00 |
| E5 n500 | .50 | - | - | .33* | .50* | .50 |
| E5 n1000 | .33 | - | - | .33* | .50* | .33 |
| E5 n4000 | .50 | - | - | .67* | .50* | .50* |
| E5 n8000 | .50* | - | - | .50* | .67* | .50* |
| E5 n24000 | .50* | - | - | .67* | .00 | .50 |

## V6 E4: placebo_conf rate (raw p, else declared) / wrong-sign rate of the true edge (* INVALID)

| row | pmrt_eq | corr | notears | pc_eq | pc_native | pmrt_r3 | shap_dag | two_tower |
|---|---|---|---|---|---|---|---|---|
| R1 l1 n1000 | -/.00 | -/.00 | -/.00~ | -/.00~ | -/.00~ | -/.00 | -/.00~ | -/.00~ |
| R1 l1 n4000 | -/.00 | -/.00 | -/.00~ | -/.00~ | -/.00~ | -/.00 | -/.00~ | -/.00~ |
| R1 l1 n24000 | -/.00 | -/.00 | -/.00~ | -/.00~ | -/.00~ | -/.00 | -/.00~ | -/.00~ |
| R2 l1 n1000 | -/.00 | -/.00 | -/.00~ | -/.00~ | -/.00~ | -/.00 | -/.00~ | -/.00~ |
| R2 l1 n4000 | -/.00 | -/.00 | -/.00~ | -/.00~ | -/.00~ | -/.00 | -/.00~ | -/.00~ |
| R2 l1 n24000 | -/.00 | -/.00 | -/.00~ | -/.00~ | -/.00~ | -/.00 | -/.00~ | -/.00~ |
| R3 l0 n1000 | .00/.00 | .00/.00 | .00/.00 | .00/.00 | .00/.00 | .00/.00 | .00/.00 | 1.00/.00* |
| R3 l0 n4000 | .00/.00 | .00/.00 | .00/.00 | .00/.00 | .00/.00 | .00/.00 | .00/.00 | 1.00/.00* |
| R3 l0 n24000 | .00/.00 | .00/.00 | .00/.00 | 1.00/.00* | 1.00/.00* | .00/.00 | .00/.00 | .00/.00 |
| R3 l0.5 n1000 | .00/.00 | 1.00/1.00* | .00/.00 | 1.00/.00* | 1.00/.00* | .00/.00 | 1.00/.00* | 1.00/.00* |
| R3 l0.5 n4000 | .00/.00 | 1.00/1.00* | .00/.00 | .00/.00 | .00/.00 | .00/.00 | 1.00/.00* | 1.00/.00* |
| R3 l0.5 n24000 | 1.00/.00* | 1.00/1.00* | .00/.00 | 1.00/.00* | 1.00/.00* | 1.00/.00* | 1.00/.00* | 1.00/.00* |
| R3 l1 n1000 | .00/.00 | 1.00/1.00* | .00/.00 | .00/.00 | .00/.00 | .00/.00 | 1.00/.00* | 1.00/.00* |
| R3 l1 n4000 | .00/.00 | 1.00/1.00* | .00/.00 | .00/.00 | 1.00/.00* | .00/.00 | 1.00/.00* | 1.00/.00* |
| R3 l1 n24000 | 1.00/.00* | 1.00/1.00* | .00/.00 | 1.00/.00* | 1.00/.00* | 1.00/.00* | 1.00/.00* | 1.00/.00* |
| R3 l1.5 n1000 | .00/.00 | 1.00/1.00* | .00/.00 | .00/.00 | .00/.00 | .00/.00 | 1.00/.00* | 1.00/.00* |
| R3 l1.5 n4000 | .00/.00 | 1.00/1.00* | .00/.00 | .00/.00 | .00/.00 | .00/.00 | 1.00/.00* | 1.00/.00* |
| R3 l1.5 n24000 | .00/.00 | 1.00/1.00* | .00/.00 | 1.00/.00* | 1.00/.00* | .00/.00 | 1.00/.00* | 1.00/.00* |
| R4 l0 n1000 | .00/.00 | .00/.00 | .00/.00 | 1.00/.00* | 1.00/.00* | .00/.00 | .00/.00 | .00/.00 |
| R4 l0 n4000 | .00/.00 | .00/.00 | .00/.00 | 1.00/.00* | 1.00/.00* | .00/.00 | 1.00/.00* | 1.00/.00* |
| R4 l0 n24000 | .00/.00 | .00/.00 | .00/.00 | .00/.00 | .00/.00 | .00/.00 | .00/.00 | 1.00/.00* |
| R4 l0.5 n1000 | .00/.00 | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | .00/.00 | 1.00/1.00* | 1.00/1.00* |
| R4 l0.5 n4000 | .00/.00 | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | .00/.00 | 1.00/1.00* | 1.00/1.00* |
| R4 l0.5 n24000 | .00/.00 | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | .00/.00 | 1.00/1.00* | 1.00/1.00* |
| R4 l1 n1000 | .00/.00 | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | .00/.00 | 1.00/1.00* | 1.00/1.00* |
| R4 l1 n4000 | .00/.00 | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | .00/.00 | 1.00/1.00* | 1.00/1.00* |
| R4 l1 n24000 | .00/.00 | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | .00/.00 | 1.00/1.00* | 1.00/1.00* |
| R4 l1.5 n1000 | .00/.00 | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | .00/.00 | 1.00/1.00* | 1.00/1.00* |
| R4 l1.5 n4000 | .00/.00 | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | .00/.00 | 1.00/1.00* | 1.00/1.00* |
| R4 l1.5 n24000 | .00/.00 | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | .00/.00 | 1.00/1.00* | 1.00/1.00* |

## V7 not testable / not applicable

40 (arm, cell) with counts > 0 (JSON V7).

## V8 secondary KPI -> KPI family

260 (arm, cell) rows (JSON V8).

## V9 kappa sweep (R2, n 1000): recall

| row | pmrt_eq | corr | granger_eq | granger_native | notears | pc_eq | pc_native | pmrt_r3 | shap_dag | two_tower |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 k0.125 | 1.00* | 1.00* | - | - | 1.00 | 1.00* | 1.00* | 1.00* | 1.00* | 1.00* |
| E1 k0.25 | 1.00* | 1.00* | - | - | 1.00* | 1.00* | 1.00 | 1.00* | 1.00* | 1.00* |
| E1 k0.5 | 1.00* | 1.00* | - | - | 1.00* | 1.00* | 1.00* | 1.00* | 1.00* | 1.00* |
| E2 k0.125 | .25* | .81* | - | - | .44* | .44* | .50* | .25* | .88* | .62* |
| E2 k0.25 | .25 | .81* | - | - | .38* | .56* | .44* | .25* | .88* | .62* |
| E2 k0.5 | .25 | .81* | - | - | .38 | .69* | .44* | .25* | .88* | .62* |
| E3 k0.125 | 1.00* | 1.00* | 1.00 | 1.00* | 1.00* | 1.00 | 1.00* | 1.00* | 1.00* | 1.00* |
| E3 k0.25 | 1.00* | 1.00* | 1.00 | 1.00* | 1.00* | 1.00 | 1.00* | 1.00* | 1.00* | 1.00* |
| E3 k0.5 | 1.00* | 1.00* | 1.00 | 1.00* | 1.00* | 1.00* | 1.00* | 1.00* | 1.00* | 1.00* |
| E4 k0.125 | .00 | .00 | - | - | .00~ | .00~ | .00~ | .00 | 1.00~ | 1.00~ |
| E4 k0.25 | .00 | .00 | - | - | .00~ | .00~ | .00~ | .00 | 1.00~ | 1.00~ |
| E4 k0.5 | .00 | .00 | - | - | .00~ | .00~ | .00~ | .00 | 1.00~ | 1.00~ |
| E5 k0.125 | .33 | .83* | - | - | .50* | .33* | .50* | .33 | .83* | .83* |
| E5 k0.25 | .33 | .67* | - | - | .67* | .33* | .50* | .33 | .67* | .83* |
| E5 k0.5 | .33 | .67* | - | - | .67* | .50* | .67* | .33* | .67* | .67 |

## V10 cost (CPU-s per dataset, mean / max over worlds, regimes; peak RSS MB; infeasible units; T3 = infeasible from DEV cost)

| arm | n 500 | n 1000 | n 4000 | n 8000 | n 24000 | RSS | infeasible |
|---|---|---|---|---|---|---|---|
| pmrt_eq | 0.6 / 1.1 | 0.9 / 1.8 | 3.6 / 7.2 | 8.6 / 17.5 | 24.7 / 58.8 | 695 | 0 |
| corr | 0.1 / 0.1 | 0.1 / 0.1 | 0.1 / 0.1 | 0.1 / 0.2 | 0.1 / 0.3 | 454 | 0 |
| granger_eq | 0.2 / 0.2 | 0.3 / 0.3 | 0.7 / 0.8 | 1.7 / 2.0 | 5.1 / 6.5 | 394 | 0 |
| granger_native | 0.1 / 0.1 | 0.1 / 0.1 | 0.1 / 0.1 | 0.2 / 0.2 | 0.3 / 0.3 | 367 | 0 |
| notears | 0.3 / 0.9 | 0.4 / 1.1 | 0.7 / 2.4 | 1.3 / 4.5 | 3.6 / 13.1 | 455 | 0 |
| pc_eq | 1.9 / 12.6 | 2.4 / 15.4 | 3.7 / 25.8 | 5.2 / 36.7 | 9.2 / 60.2 | 472 | 0 |
| pc_native | 0.2 / 0.7 | 0.3 / 1.2 | 0.5 / 2.6 | 0.8 / 4.5 | 1.6 / 8.9 | 455 | 0 |
| pmrt_r3 | 0.6 / 1.0 | 0.9 / 1.7 | 3.5 / 6.2 | 8.2 / 15.9 | 23.9 / 55.9 | 695 | 0 |
| shap_dag | 2.0 / 4.2 | 3.4 / 6.8 | 12.2 / 22.9 | 24.4 / 43.8 | 72.0 / 137.2 | 474 | 0 |
| two_tower | 6.7 / 12.2 | 7.0 / 10.0 | 15.4 / 29.3 | 26.8 / 53.4 | 71.0 / 143.8 | 725 | 0 |

## V11 integrity

- label PROVISIONAL; failed checks: commits, freeze_commit_given, protocol_frozen_sha, stamps
- freeze commit None; amendments []; records used 1648 / expected 1648; missing 0; unexpected 0; role mismatch 0; duplicates 0
- commit violations 1648; not clean 0; stamp violations 1648; candidates incomplete 0; pkgs sets 1
- role:status {'measure:ok': 824, 'tune:ok': 824}; errors 0 (not listed persistent 0); infeasible 0; dataset-hash mismatches 0; multi-platform datasets 0
- BY recheck (p arms): 648 / 648 records agree; tune reproducibility vs DEV: None
