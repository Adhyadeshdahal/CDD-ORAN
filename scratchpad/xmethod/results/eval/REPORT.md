# Study A EVAL report

Generator `xm-eval-report/1` on `xm-eval-analysis/5`; spec `eval_full` (sha256 a02fd14833bb3a51); protocol docs/xmethod/PROTOCOL_A.md (frozen, sha matches); integrity **FINAL**.
Primary PMRT arm (focal): pmrt_nl_eq; primary kappa 0.25; generated 2026-10-05T07:33:00Z.

Inputs:

- `scratchpad/xmethod/specs/eval/full.json`: sha256 a02fd14833bb3a51
- `scratchpad/xmethod/results/eval/merged.jsonl.gz`: sha256 a9f52a77b05ac0a8
- records by arm: cdl 1800, corr 5800, granger_eq 680, granger_native 680, mscr_eq 4240, mscr_native 2160, notears 5800, pc_eq 11000, pc_native 5800, pcorr_eq 11000, pcorr_eq_min 5800, pcorr_hac 5800, pcorr_hac_eq_min 5800, pcorr_hac_fb 5800, pcorr_hac_fb_eq_min 5800, pcorr_native 5800, pmrt_eq 11000, pmrt_nl_eq 11000, pmrt_r3 11000, rcot2_eq 11000, rcot2_eq_min 5800, rcot2_native 5800, shap_dag 5800, two_tower 5800

## 1. Claim (PROTOCOL_A s.10)

**Claim: SUPPORTED**

| component | verdict | wording |
|---|---|---|
| C1 design-blind tests invalid on R2 | SUPPORTED | design-blind tests are invalid on the R2 (setpoint + dither) design tested |
| C2a design-based test valid (R1 + R2) | SUPPORTED | using the design restores validity (design-based inference and design-covariate adjustment); rcot2_eq INVALID in R2 as already in R1 (not restored by design covariates) |
| C2b design covariates restore validity | SUPPORTED | (C2 wording above) |
| C3 valid under the logged confounded policy | SUPPORTED | PMRT stays valid under the logged confounded policy (R3) by construction; so do pcorr_eq, so the property is not specific to PMRT |

## 2. C1: design-blind arms (set D)

**SUPPORTED**: 5 of 6 assessable arms are FAILURES (|D| = 6; >= 3 assessable needed).

| arm | in D | verdict | R2 INVALID / planned (counted) | pooled R2 truth-null raw | pooled R2 placebo raw | R1 INVALID / counted (F_max) |
|---|---|---|---|---|---|---|
| corr | yes | FAILURE | 22/25 (25) | .786 [.772, .802] INVALID | .751 [.726, .776] INVALID | 0/25 (2) |
| granger_native | yes | FAILURE | 5/5 (5) | .686 [.654, .719] INVALID | .578 [.515, .642] INVALID | 0/5 (1) |
| mscr_native [single-conditioner max statistic; reported INVALID arm, EVAL n <= 1000 (R-54)] | yes | FAILURE | 8/10 (10) | .944 [.933, .956] INVALID | .906 [.886, .924] INVALID | 1/10 (1) |
| pcorr_hac_fb | yes | FAILURE | 20/25 (25) | .468 [.435, .497] INVALID | .355 [.312, .397] INVALID | 0/25 (2) |
| pcorr_native | yes | FAILURE | 20/25 (25) | .471 [.438, .500] INVALID | .350 [.307, .393] INVALID | 0/25 (2) |
| rcot2_native | yes | INVALID IN R1 | 20/25 (25) | .327 [.317, .338] INVALID | .309 [.291, .328] INVALID | 6/25 (2) |
| pcorr_hac | no (descriptive) | FAILURE | 20/25 (25) | .469 [.435, .497] INVALID | .356 [.312, .397] INVALID | 0/25 (2) |

C1 sensitivity (R-56, no effect): without mscr_native: **SUPPORTED** (4 of 5 assessable arms fail).

## 3. C2a: the design-based test (R1 + R2)

| arm | verdict | cells counted / planned | INVALID (F_max) | pooled rates |
|---|---|---|---|---|
| pmrt_nl_eq (primary) | SUPPORTED | 50/50 | 1 (3) | null_raw .045 [.043, .048] VALID; plac_raw .048 [.042, .054] VALID |
| pmrt_eq [linear statistic (pmrt-core-v1); secondary PMRT arm (R-42)] (secondary PMRT arm, no effect) | SUPPORTED | 50/50 | 0 (3) | null_raw .049 [.046, .052] VALID; plac_raw .050 [.044, .056] VALID |
| pmrt_r3 (secondary PMRT arm, no effect) | SUPPORTED | 50/50 | 1 (3) | null_raw .049 [.045, .052] VALID; plac_raw .049 [.043, .056] VALID |

## 4. C2b: design-covariate adjustment (eq arms whose native partner is a C1 FAILURE)

**SUPPORTED** (membership rule as frozen); with the eq arms whose native partner is INVALID IN R1 but meets C1 legs (i)-(ii) added (R-56 sensitivity, no effect): **SUPPORTED** (added: rcot2_eq).

| arm | verdict | cells counted / planned | INVALID (F_max) | pooled rates |
|---|---|---|---|---|
| granger_eq (native granger_native) | SUPPORTED | 10/10 | 0 (1) | null_raw .050 [.044, .056] VALID; plac_raw .056 [.041, .071] VALID |
| pcorr_eq (native pcorr_native) | SUPPORTED | 50/50 | 0 (3) | null_raw .048 [.045, .051] VALID; plac_raw .051 [.044, .057] VALID |
| mscr_eq [single-conditioner max statistic; cannot condition on the joint design set; reported INVALID arm, EVAL n <= 1000 (R-54)] EXCLUDED (R-40), no effect | NOT SUPPORTED | 20/20 | 8 (2) | null_raw .503 [.498, .508] INVALID; plac_raw .489 [.479, .500] INVALID |
| rcot2_eq (R-56 sensitivity member) | NOT SUPPORTED | 50/50 | 25 (3) | null_raw .177 [.172, .182] INVALID; plac_raw .163 [.154, .172] INVALID |

C2 wording: using the design restores validity (design-based inference and design-covariate adjustment); rcot2_eq INVALID in R2 as already in R1 (not restored by design covariates).

Pre-registered disclosure (R-56; named eq arms):
- rcot2_eq is INVALID in R2 (pooled truth-null raw rate 0.292 [0.283, 0.301]) as already in R1 (0.061 [0.056, 0.066]): the test is miscalibrated without the design (its native partner rcot2_native is INVALID IN R1), so it is outside C2b's membership rule, and adding design covariates does not make it valid.

Every eq arm, same rule on R1 + R2, unfiltered (R-56):

| eq arm | C2b membership | native C1 | verdict | INVALID / cells (F_max) | R2 INVALID / planned | pooled R1 truth-null raw | pooled R2 truth-null raw |
|---|---|---|---|---|---|---|---|
| granger_eq | member | FAILURE | SUPPORTED | 0/10 (1) | 0/5 | .048 [.040, .058] VALID | .052 [.044, .060] VALID |
| mscr_eq [single-conditioner max statistic; cannot condition on the joint design set; reported INVALID arm, EVAL n <= 1000 (R-54)] | excluded (R-40) | FAILURE | NOT SUPPORTED | 8/20 (2) | 8/10 | .046 [.040, .051] VALID | .960 [.952, .970] INVALID |
| pc_eq | not a member (declare tau) | - | NOT SUPPORTED | 4/50 (2) | - | - | - |
| pcorr_eq | member | FAILURE | SUPPORTED | 0/50 (3) | 0/25 | .046 [.042, .051] VALID | .050 [.045, .054] VALID |
| rcot2_eq | not a member (native rcot2_native: INVALID IN R1) | INVALID IN R1 | NOT SUPPORTED | 25/50 (3) | 20/25 | .061 [.056, .066] INVALID | .292 [.283, .301] INVALID |

## 5. C3: logged confounded policy (E4 R3; C3 readers at S_E4)

| arm | verdict | cells counted / planned | INVALID (F_max) | pooled rates |
|---|---|---|---|---|
| pmrt_nl_eq (primary) | SUPPORTED | 20/20 | 0 (2) | plac_raw .048 [.038, .060] VALID; conf_raw .042 [.034, .050] VALID |
| granger_eq (eq arm, no claim effect) | NOT EVALUABLE | 0/0 | 0 (0) | plac_raw -; conf_raw - |
| mscr_eq [single-conditioner max statistic; cannot condition on the joint design set; reported INVALID arm, EVAL n <= 1000 (R-54)] (eq arm, no claim effect) | NOT SUPPORTED | 8/8 | 6 (1) | plac_raw .108 [.095, .121] INVALID; conf_raw .763 [.759, .768] INVALID |
| pc_eq (eq arm, no claim effect) | NOT SUPPORTED | 20/20 | 7 (1) | conf_decl .075 [.065, .087] INVALID |
| pcorr_eq (eq arm, no claim effect) | SUPPORTED | 20/20 | 2 (2) | plac_raw .056 [.043, .070] VALID; conf_raw .058 [.047, .070] VALID |
| rcot2_eq (eq arm, no claim effect) | NOT SUPPORTED | 20/20 | 17 (2) | plac_raw .054 [.043, .066] VALID; conf_raw .681 [.670, .692] INVALID |
| pmrt_eq [linear statistic (pmrt-core-v1); secondary PMRT arm (R-42)] (secondary PMRT arm, no effect) | NOT SUPPORTED | 20/20 | 3 (2) | plac_raw .051 [.040, .064] VALID; conf_raw .049 [.039, .061] VALID |
| pmrt_r3 (secondary PMRT arm, no effect) | SUPPORTED | 20/20 | 1 (2) | plac_raw .050 [.038, .063] VALID; conf_raw .050 [.040, .062] VALID |

## 6. Power among cells not INVALID (V2) and the R4 'not applicable' rows

1166 (arm, cell) with recall (counted, not INVALID; R-39); the grid is in the appendix (V2) and `csv/recall_vs_n.csv`. PMRT arms in R4 (no known design): power NOT APPLICABLE, never recall 0 (R-42); their placebo rates are still read:

| arm | cell | state | placebo raw (P_placebo) |
|---|---|---|---|
| pmrt_eq | E4 R4 lam0.5 k0.25 n1000 | NA (INCONCLUSIVE) | .050 [.000, .125] INCONCLUSIVE |
| pmrt_eq | E4 R4 lam0.5 k0.25 n24000 | NA (INCONCLUSIVE) | .075 [.000, .175] INCONCLUSIVE |
| pmrt_eq | E4 R4 lam0.5 k0.25 n4000 | NA (INCONCLUSIVE) | .125 [.025, .225] INCONCLUSIVE |
| pmrt_eq | E4 R4 lam0.5 k0.25 n500 | NA (INCONCLUSIVE) | .050 [.000, .125] INCONCLUSIVE |
| pmrt_eq | E4 R4 lam0.5 k0.25 n8000 | NA (INCONCLUSIVE) | .125 [.025, .225] INCONCLUSIVE |
| pmrt_eq | E4 R4 lam0 k0.25 n1000 | NA (INCONCLUSIVE) | .050 [.000, .125] INCONCLUSIVE |
| pmrt_eq | E4 R4 lam0 k0.25 n24000 | NA (INCONCLUSIVE) | .050 [.000, .125] INCONCLUSIVE |
| pmrt_eq | E4 R4 lam0 k0.25 n4000 | NA (INCONCLUSIVE) | .150 [.050, .250] INCONCLUSIVE |
| pmrt_eq | E4 R4 lam0 k0.25 n500 | NA (VALID) | .025 [.000, .075] VALID |
| pmrt_eq | E4 R4 lam0 k0.25 n8000 | NA (INCONCLUSIVE) | .125 [.025, .250] INCONCLUSIVE |
| pmrt_eq | E4 R4 lam1.5 k0.25 n1000 | NA (INCONCLUSIVE) | .050 [.000, .125] INCONCLUSIVE |
| pmrt_eq | E4 R4 lam1.5 k0.25 n24000 | NA (INCONCLUSIVE) | .050 [.000, .125] INCONCLUSIVE |
| pmrt_eq | E4 R4 lam1.5 k0.25 n4000 | NA (INCONCLUSIVE) | .075 [.000, .175] INCONCLUSIVE |
| pmrt_eq | E4 R4 lam1.5 k0.25 n500 | NA (VALID) | .025 [.000, .075] VALID |
| pmrt_eq | E4 R4 lam1.5 k0.25 n8000 | NA (INCONCLUSIVE) | .050 [.000, .125] INCONCLUSIVE |
| pmrt_eq | E4 R4 lam1 k0.25 n1000 | NA (INCONCLUSIVE) | .050 [.000, .125] INCONCLUSIVE |
| pmrt_eq | E4 R4 lam1 k0.25 n24000 | NA (INCONCLUSIVE) | .050 [.000, .125] INCONCLUSIVE |
| pmrt_eq | E4 R4 lam1 k0.25 n4000 | NA (INCONCLUSIVE) | .100 [.025, .200] INCONCLUSIVE |
| pmrt_eq | E4 R4 lam1 k0.25 n500 | NA (VALID) | .025 [.000, .075] VALID |
| pmrt_eq | E4 R4 lam1 k0.25 n8000 | NA (INCONCLUSIVE) | .075 [.000, .150] INCONCLUSIVE |
| pmrt_nl_eq | E4 R4 lam0.5 k0.25 n1000 | NA (INCONCLUSIVE) | .050 [.000, .125] INCONCLUSIVE |
| pmrt_nl_eq | E4 R4 lam0.5 k0.25 n24000 | NA (INCONCLUSIVE) | .075 [.000, .175] INCONCLUSIVE |
| pmrt_nl_eq | E4 R4 lam0.5 k0.25 n4000 | NA (INCONCLUSIVE) | .050 [.000, .125] INCONCLUSIVE |
| pmrt_nl_eq | E4 R4 lam0.5 k0.25 n500 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_nl_eq | E4 R4 lam0.5 k0.25 n8000 | NA (VALID) | .025 [.000, .075] VALID |
| pmrt_nl_eq | E4 R4 lam0 k0.25 n1000 | NA (VALID) | .025 [.000, .075] VALID |
| pmrt_nl_eq | E4 R4 lam0 k0.25 n24000 | NA (INCONCLUSIVE) | .050 [.000, .125] INCONCLUSIVE |
| pmrt_nl_eq | E4 R4 lam0 k0.25 n4000 | NA (INCONCLUSIVE) | .050 [.000, .125] INCONCLUSIVE |
| pmrt_nl_eq | E4 R4 lam0 k0.25 n500 | NA (INCONCLUSIVE) | .050 [.000, .125] INCONCLUSIVE |
| pmrt_nl_eq | E4 R4 lam0 k0.25 n8000 | NA (INCONCLUSIVE) | .075 [.000, .175] INCONCLUSIVE |
| pmrt_nl_eq | E4 R4 lam1.5 k0.25 n1000 | NA (INCONCLUSIVE) | .075 [.000, .175] INCONCLUSIVE |
| pmrt_nl_eq | E4 R4 lam1.5 k0.25 n24000 | NA (INCONCLUSIVE) | .050 [.000, .125] INCONCLUSIVE |
| pmrt_nl_eq | E4 R4 lam1.5 k0.25 n4000 | NA (INCONCLUSIVE) | .075 [.000, .175] INCONCLUSIVE |
| pmrt_nl_eq | E4 R4 lam1.5 k0.25 n500 | NA (VALID) | .025 [.000, .075] VALID |
| pmrt_nl_eq | E4 R4 lam1.5 k0.25 n8000 | NA (INCONCLUSIVE) | .050 [.000, .125] INCONCLUSIVE |
| pmrt_nl_eq | E4 R4 lam1 k0.25 n1000 | NA (VALID) | .025 [.000, .075] VALID |
| pmrt_nl_eq | E4 R4 lam1 k0.25 n24000 | NA (INCONCLUSIVE) | .075 [.000, .175] INCONCLUSIVE |
| pmrt_nl_eq | E4 R4 lam1 k0.25 n4000 | NA (INCONCLUSIVE) | .050 [.000, .125] INCONCLUSIVE |
| pmrt_nl_eq | E4 R4 lam1 k0.25 n500 | NA (VALID) | .025 [.000, .075] VALID |
| pmrt_nl_eq | E4 R4 lam1 k0.25 n8000 | NA (INCONCLUSIVE) | .050 [.000, .125] INCONCLUSIVE |
| pmrt_r3 | E4 R4 lam0.5 k0.25 n1000 | NA (INCONCLUSIVE) | .050 [.000, .125] INCONCLUSIVE |
| pmrt_r3 | E4 R4 lam0.5 k0.25 n24000 | NA (INCONCLUSIVE) | .075 [.000, .175] INCONCLUSIVE |
| pmrt_r3 | E4 R4 lam0.5 k0.25 n4000 | NA (INCONCLUSIVE) | .100 [.025, .200] INCONCLUSIVE |
| pmrt_r3 | E4 R4 lam0.5 k0.25 n500 | NA (VALID) | .025 [.000, .075] VALID |
| pmrt_r3 | E4 R4 lam0.5 k0.25 n8000 | NA (INCONCLUSIVE) | .150 [.050, .275] INCONCLUSIVE |
| pmrt_r3 | E4 R4 lam0 k0.25 n1000 | NA (INCONCLUSIVE) | .050 [.000, .125] INCONCLUSIVE |
| pmrt_r3 | E4 R4 lam0 k0.25 n24000 | NA (INCONCLUSIVE) | .050 [.000, .125] INCONCLUSIVE |
| pmrt_r3 | E4 R4 lam0 k0.25 n4000 | NA (INCONCLUSIVE) | .150 [.050, .250] INCONCLUSIVE |
| pmrt_r3 | E4 R4 lam0 k0.25 n500 | NA (VALID) | .025 [.000, .075] VALID |
| pmrt_r3 | E4 R4 lam0 k0.25 n8000 | NA (INCONCLUSIVE) | .125 [.025, .250] INCONCLUSIVE |
| pmrt_r3 | E4 R4 lam1.5 k0.25 n1000 | NA (INCONCLUSIVE) | .050 [.000, .125] INCONCLUSIVE |
| pmrt_r3 | E4 R4 lam1.5 k0.25 n24000 | NA (INCONCLUSIVE) | .050 [.000, .125] INCONCLUSIVE |
| pmrt_r3 | E4 R4 lam1.5 k0.25 n4000 | NA (INCONCLUSIVE) | .075 [.000, .175] INCONCLUSIVE |
| pmrt_r3 | E4 R4 lam1.5 k0.25 n500 | NA (VALID) | .025 [.000, .075] VALID |
| pmrt_r3 | E4 R4 lam1.5 k0.25 n8000 | NA (INCONCLUSIVE) | .050 [.000, .125] INCONCLUSIVE |
| pmrt_r3 | E4 R4 lam1 k0.25 n1000 | NA (INCONCLUSIVE) | .050 [.000, .125] INCONCLUSIVE |
| pmrt_r3 | E4 R4 lam1 k0.25 n24000 | NA (INCONCLUSIVE) | .050 [.000, .125] INCONCLUSIVE |
| pmrt_r3 | E4 R4 lam1 k0.25 n4000 | NA (INCONCLUSIVE) | .100 [.025, .200] INCONCLUSIVE |
| pmrt_r3 | E4 R4 lam1 k0.25 n500 | NA (VALID) | .025 [.000, .075] VALID |
| pmrt_r3 | E4 R4 lam1 k0.25 n8000 | NA (INCONCLUSIVE) | .075 [.000, .150] INCONCLUSIVE |

## 7. kappa sweep (R2, n 1000; V9)

| arm | world | kappa | status | validity | recall |
|---|---|---|---|---|---|
| cdl | E1 | 0.25 | ok | INVALID | 1.00 |
| cdl | E2 | 0.25 | ok | VALID | .49 |
| cdl | E3 | 0.25 | ok | INVALID | 1.00 |
| cdl | E4 | 0.25 | ok | NO_READ | .17 |
| cdl | E5 | 0.25 | ok | VALID | .70 |
| corr | E1 | 0.125 | ok | INVALID | 1.00 |
| corr | E1 | 0.25 | ok | INVALID | 1.00 |
| corr | E1 | 0.5 | ok | INVALID | 1.00 |
| corr | E2 | 0.125 | ok | INVALID | .68 |
| corr | E2 | 0.25 | ok | INVALID | .65 |
| corr | E2 | 0.5 | ok | INVALID | .63 |
| corr | E3 | 0.125 | ok | INVALID | 1.00 |
| corr | E3 | 0.25 | ok | INVALID | 1.00 |
| corr | E3 | 0.5 | ok | INVALID | 1.00 |
| corr | E4 | 0.125 | ok | INCONCLUSIVE | .40 |
| corr | E4 | 0.25 | ok | INCONCLUSIVE | .33 |
| corr | E4 | 0.5 | ok | INCONCLUSIVE | .45 |
| corr | E5 | 0.125 | ok | INVALID | .77 |
| corr | E5 | 0.25 | ok | INVALID | .78 |
| corr | E5 | 0.5 | ok | INVALID | .74 |
| granger_eq | E3 | 0.125 | ok | INCONCLUSIVE | 1.00 |
| granger_eq | E3 | 0.25 | ok | INCONCLUSIVE | 1.00 |
| granger_eq | E3 | 0.5 | ok | INCONCLUSIVE | 1.00 |
| granger_native | E3 | 0.125 | ok | INVALID | 1.00 |
| granger_native | E3 | 0.25 | ok | INVALID | 1.00 |
| granger_native | E3 | 0.5 | ok | INVALID | 1.00 |
| mscr_eq | E1 | 0.25 | ok | INVALID | 1.00 |
| mscr_eq | E2 | 0.25 | ok | INVALID | .97 |
| mscr_eq | E3 | 0.25 | ok | INVALID | 1.00 |
| mscr_eq | E4 | 0.25 | ok | INCONCLUSIVE | .05 |
| mscr_eq | E5 | 0.25 | ok | INVALID | 1.00 |
| mscr_native | E1 | 0.25 | ok | INVALID | 1.00 |
| mscr_native | E2 | 0.25 | ok | INVALID | .97 |
| mscr_native | E3 | 0.25 | ok | INVALID | 1.00 |
| mscr_native | E4 | 0.25 | ok | VALID | .05 |
| mscr_native | E5 | 0.25 | ok | INVALID | 1.00 |
| notears | E1 | 0.125 | ok | INCONCLUSIVE | 1.00 |
| notears | E1 | 0.25 | ok | INVALID | 1.00 |
| notears | E1 | 0.5 | ok | INVALID | 1.00 |
| notears | E2 | 0.125 | ok | VALID | .31 |
| notears | E2 | 0.25 | ok | VALID | .29 |
| notears | E2 | 0.5 | ok | VALID | .31 |
| notears | E3 | 0.125 | ok | INVALID | 1.00 |
| notears | E3 | 0.25 | ok | INVALID | 1.00 |
| notears | E3 | 0.5 | ok | INVALID | 1.00 |
| notears | E4 | 0.125 | ok | NO_READ | .25 |
| notears | E4 | 0.25 | ok | NO_READ | .12 |
| notears | E4 | 0.5 | ok | NO_READ | .15 |
| notears | E5 | 0.125 | ok | INCONCLUSIVE | .47 |
| notears | E5 | 0.25 | ok | VALID | .45 |
| notears | E5 | 0.5 | ok | VALID | .43 |
| pc_eq | E1 | 0.125 | ok | VALID | 1.00 |
| pc_eq | E1 | 0.25 | ok | INCONCLUSIVE | 1.00 |
| pc_eq | E1 | 0.5 | ok | VALID | 1.00 |
| pc_eq | E2 | 0.125 | ok | VALID | .28 |
| pc_eq | E2 | 0.25 | ok | INVALID | .30 |
| pc_eq | E2 | 0.5 | ok | VALID | .31 |
| pc_eq | E3 | 0.125 | ok | VALID | 1.00 |
| pc_eq | E3 | 0.25 | ok | VALID | 1.00 |
| pc_eq | E3 | 0.5 | ok | VALID | 1.00 |
| pc_eq | E4 | 0.125 | ok | NO_READ | .05 |
| pc_eq | E4 | 0.25 | ok | NO_READ | .07 |
| pc_eq | E4 | 0.5 | ok | NO_READ | .20 |
| pc_eq | E5 | 0.125 | ok | INVALID | .57 |
| pc_eq | E5 | 0.25 | ok | VALID | .50 |
| pc_eq | E5 | 0.5 | ok | VALID | .48 |
| pc_native | E1 | 0.125 | ok | VALID | 1.00 |
| pc_native | E1 | 0.25 | ok | VALID | 1.00 |
| pc_native | E1 | 0.5 | ok | INVALID | 1.00 |
| pc_native | E2 | 0.125 | ok | VALID | .25 |
| pc_native | E2 | 0.25 | ok | VALID | .29 |
| pc_native | E2 | 0.5 | ok | VALID | .35 |
| pc_native | E3 | 0.125 | ok | INVALID | 1.00 |
| pc_native | E3 | 0.25 | ok | INVALID | 1.00 |
| pc_native | E3 | 0.5 | ok | INVALID | 1.00 |
| pc_native | E4 | 0.125 | ok | NO_READ | .50 |
| pc_native | E4 | 0.25 | ok | NO_READ | .47 |
| pc_native | E4 | 0.5 | ok | NO_READ | .50 |
| pc_native | E5 | 0.125 | ok | VALID | .47 |
| pc_native | E5 | 0.25 | ok | VALID | .45 |
| pc_native | E5 | 0.5 | ok | VALID | .43 |
| pcorr_eq | E1 | 0.125 | ok | VALID | 1.00 |
| pcorr_eq | E1 | 0.25 | ok | INCONCLUSIVE | 1.00 |
| pcorr_eq | E1 | 0.5 | ok | VALID | 1.00 |
| pcorr_eq | E2 | 0.125 | ok | INCONCLUSIVE | .40 |
| pcorr_eq | E2 | 0.25 | ok | VALID | .37 |
| pcorr_eq | E2 | 0.5 | ok | VALID | .36 |
| pcorr_eq | E3 | 0.125 | ok | INCONCLUSIVE | 1.00 |
| pcorr_eq | E3 | 0.25 | ok | INCONCLUSIVE | 1.00 |
| pcorr_eq | E3 | 0.5 | ok | INCONCLUSIVE | 1.00 |
| pcorr_eq | E4 | 0.125 | ok | INCONCLUSIVE | .10 |
| pcorr_eq | E4 | 0.25 | ok | INCONCLUSIVE | .05 |
| pcorr_eq | E4 | 0.5 | ok | INCONCLUSIVE | .10 |
| pcorr_eq | E5 | 0.125 | ok | INCONCLUSIVE | .57 |
| pcorr_eq | E5 | 0.25 | ok | INCONCLUSIVE | .55 |
| pcorr_eq | E5 | 0.5 | ok | VALID | .54 |
| pcorr_eq_min | E1 | 0.125 | ok | VALID | 1.00 |
| pcorr_eq_min | E1 | 0.25 | ok | INCONCLUSIVE | 1.00 |
| pcorr_eq_min | E1 | 0.5 | ok | VALID | 1.00 |
| pcorr_eq_min | E2 | 0.125 | ok | INCONCLUSIVE | .41 |
| pcorr_eq_min | E2 | 0.25 | ok | VALID | .38 |
| pcorr_eq_min | E2 | 0.5 | ok | INCONCLUSIVE | .36 |
| pcorr_eq_min | E3 | 0.125 | ok | INCONCLUSIVE | 1.00 |
| pcorr_eq_min | E3 | 0.25 | ok | VALID | 1.00 |
| pcorr_eq_min | E3 | 0.5 | ok | INCONCLUSIVE | 1.00 |
| pcorr_eq_min | E4 | 0.125 | ok | INCONCLUSIVE | .10 |
| pcorr_eq_min | E4 | 0.25 | ok | INCONCLUSIVE | .05 |
| pcorr_eq_min | E4 | 0.5 | ok | INCONCLUSIVE | .10 |
| pcorr_eq_min | E5 | 0.125 | ok | VALID | .58 |
| pcorr_eq_min | E5 | 0.25 | ok | INCONCLUSIVE | .54 |
| pcorr_eq_min | E5 | 0.5 | ok | VALID | .53 |
| pcorr_hac | E1 | 0.125 | ok | INVALID | 1.00 |
| pcorr_hac | E1 | 0.25 | ok | INVALID | 1.00 |
| pcorr_hac | E1 | 0.5 | ok | INVALID | 1.00 |
| pcorr_hac | E2 | 0.125 | ok | INVALID | .50 |
| pcorr_hac | E2 | 0.25 | ok | INVALID | .49 |
| pcorr_hac | E2 | 0.5 | ok | INVALID | .49 |
| pcorr_hac | E3 | 0.125 | ok | INVALID | 1.00 |
| pcorr_hac | E3 | 0.25 | ok | INVALID | 1.00 |
| pcorr_hac | E3 | 0.5 | ok | INVALID | 1.00 |
| pcorr_hac | E4 | 0.125 | ok | INCONCLUSIVE | .40 |
| pcorr_hac | E4 | 0.25 | ok | INCONCLUSIVE | .33 |
| pcorr_hac | E4 | 0.5 | ok | INCONCLUSIVE | .40 |
| pcorr_hac | E5 | 0.125 | ok | INVALID | .55 |
| pcorr_hac | E5 | 0.25 | ok | INVALID | .56 |
| pcorr_hac | E5 | 0.5 | ok | INVALID | .55 |
| pcorr_hac_eq_min | E1 | 0.125 | ok | VALID | 1.00 |
| pcorr_hac_eq_min | E1 | 0.25 | ok | INCONCLUSIVE | 1.00 |
| pcorr_hac_eq_min | E1 | 0.5 | ok | VALID | 1.00 |
| pcorr_hac_eq_min | E2 | 0.125 | ok | INCONCLUSIVE | .31 |
| pcorr_hac_eq_min | E2 | 0.25 | ok | INCONCLUSIVE | .28 |
| pcorr_hac_eq_min | E2 | 0.5 | ok | INCONCLUSIVE | .29 |
| pcorr_hac_eq_min | E3 | 0.125 | ok | INCONCLUSIVE | 1.00 |
| pcorr_hac_eq_min | E3 | 0.25 | ok | VALID | 1.00 |
| pcorr_hac_eq_min | E3 | 0.5 | ok | INCONCLUSIVE | 1.00 |
| pcorr_hac_eq_min | E4 | 0.125 | ok | INCONCLUSIVE | .10 |
| pcorr_hac_eq_min | E4 | 0.25 | ok | INCONCLUSIVE | .05 |
| pcorr_hac_eq_min | E4 | 0.5 | ok | INCONCLUSIVE | .10 |
| pcorr_hac_eq_min | E5 | 0.125 | ok | VALID | .41 |
| pcorr_hac_eq_min | E5 | 0.25 | ok | INCONCLUSIVE | .40 |
| pcorr_hac_eq_min | E5 | 0.5 | ok | VALID | .42 |
| pcorr_hac_fb | E1 | 0.125 | ok | INVALID | 1.00 |
| pcorr_hac_fb | E1 | 0.25 | ok | INVALID | 1.00 |
| pcorr_hac_fb | E1 | 0.5 | ok | INVALID | 1.00 |
| pcorr_hac_fb | E2 | 0.125 | ok | INVALID | .50 |
| pcorr_hac_fb | E2 | 0.25 | ok | INVALID | .49 |
| pcorr_hac_fb | E2 | 0.5 | ok | INVALID | .49 |
| pcorr_hac_fb | E3 | 0.125 | ok | INVALID | 1.00 |
| pcorr_hac_fb | E3 | 0.25 | ok | INVALID | 1.00 |
| pcorr_hac_fb | E3 | 0.5 | ok | INVALID | 1.00 |
| pcorr_hac_fb | E4 | 0.125 | ok | INCONCLUSIVE | .40 |
| pcorr_hac_fb | E4 | 0.25 | ok | INCONCLUSIVE | .33 |
| pcorr_hac_fb | E4 | 0.5 | ok | INCONCLUSIVE | .40 |
| pcorr_hac_fb | E5 | 0.125 | ok | INVALID | .55 |
| pcorr_hac_fb | E5 | 0.25 | ok | INVALID | .56 |
| pcorr_hac_fb | E5 | 0.5 | ok | INVALID | .55 |
| pcorr_hac_fb_eq_min | E1 | 0.125 | ok | VALID | 1.00 |
| pcorr_hac_fb_eq_min | E1 | 0.25 | ok | INCONCLUSIVE | 1.00 |
| pcorr_hac_fb_eq_min | E1 | 0.5 | ok | VALID | 1.00 |
| pcorr_hac_fb_eq_min | E2 | 0.125 | ok | INCONCLUSIVE | .31 |
| pcorr_hac_fb_eq_min | E2 | 0.25 | ok | INCONCLUSIVE | .28 |
| pcorr_hac_fb_eq_min | E2 | 0.5 | ok | INCONCLUSIVE | .29 |
| pcorr_hac_fb_eq_min | E3 | 0.125 | ok | INCONCLUSIVE | 1.00 |
| pcorr_hac_fb_eq_min | E3 | 0.25 | ok | VALID | 1.00 |
| pcorr_hac_fb_eq_min | E3 | 0.5 | ok | INCONCLUSIVE | 1.00 |
| pcorr_hac_fb_eq_min | E4 | 0.125 | ok | INCONCLUSIVE | .10 |
| pcorr_hac_fb_eq_min | E4 | 0.25 | ok | INCONCLUSIVE | .05 |
| pcorr_hac_fb_eq_min | E4 | 0.5 | ok | INCONCLUSIVE | .10 |
| pcorr_hac_fb_eq_min | E5 | 0.125 | ok | VALID | .41 |
| pcorr_hac_fb_eq_min | E5 | 0.25 | ok | INCONCLUSIVE | .40 |
| pcorr_hac_fb_eq_min | E5 | 0.5 | ok | VALID | .42 |
| pcorr_native | E1 | 0.125 | ok | INVALID | 1.00 |
| pcorr_native | E1 | 0.25 | ok | INVALID | 1.00 |
| pcorr_native | E1 | 0.5 | ok | INVALID | 1.00 |
| pcorr_native | E2 | 0.125 | ok | INVALID | .52 |
| pcorr_native | E2 | 0.25 | ok | INVALID | .52 |
| pcorr_native | E2 | 0.5 | ok | INVALID | .50 |
| pcorr_native | E3 | 0.125 | ok | INVALID | 1.00 |
| pcorr_native | E3 | 0.25 | ok | INVALID | 1.00 |
| pcorr_native | E3 | 0.5 | ok | INVALID | 1.00 |
| pcorr_native | E4 | 0.125 | ok | INCONCLUSIVE | .40 |
| pcorr_native | E4 | 0.25 | ok | INCONCLUSIVE | .33 |
| pcorr_native | E4 | 0.5 | ok | INCONCLUSIVE | .40 |
| pcorr_native | E5 | 0.125 | ok | INVALID | .61 |
| pcorr_native | E5 | 0.25 | ok | INVALID | .62 |
| pcorr_native | E5 | 0.5 | ok | INVALID | .58 |
| pmrt_eq | E1 | 0.125 | ok | INCONCLUSIVE | .95 |
| pmrt_eq | E1 | 0.25 | ok | VALID | .97 |
| pmrt_eq | E1 | 0.5 | ok | INCONCLUSIVE | .96 |
| pmrt_eq | E2 | 0.125 | ok | VALID | .29 |
| pmrt_eq | E2 | 0.25 | ok | VALID | .28 |
| pmrt_eq | E2 | 0.5 | ok | VALID | .27 |
| pmrt_eq | E3 | 0.125 | ok | INCONCLUSIVE | 1.00 |
| pmrt_eq | E3 | 0.25 | ok | INCONCLUSIVE | 1.00 |
| pmrt_eq | E3 | 0.5 | ok | INCONCLUSIVE | 1.00 |
| pmrt_eq | E4 | 0.125 | ok | INCONCLUSIVE | .10 |
| pmrt_eq | E4 | 0.25 | ok | INCONCLUSIVE | .03 |
| pmrt_eq | E4 | 0.5 | ok | INCONCLUSIVE | .05 |
| pmrt_eq | E5 | 0.125 | ok | INCONCLUSIVE | .51 |
| pmrt_eq | E5 | 0.25 | ok | INCONCLUSIVE | .47 |
| pmrt_eq | E5 | 0.5 | ok | INCONCLUSIVE | .49 |
| pmrt_nl_eq | E1 | 0.125 | ok | VALID | 1.00 |
| pmrt_nl_eq | E1 | 0.25 | ok | INCONCLUSIVE | 1.00 |
| pmrt_nl_eq | E1 | 0.5 | ok | INCONCLUSIVE | 1.00 |
| pmrt_nl_eq | E2 | 0.125 | ok | VALID | .71 |
| pmrt_nl_eq | E2 | 0.25 | ok | INCONCLUSIVE | .66 |
| pmrt_nl_eq | E2 | 0.5 | ok | VALID | .61 |
| pmrt_nl_eq | E3 | 0.125 | ok | INCONCLUSIVE | 1.00 |
| pmrt_nl_eq | E3 | 0.25 | ok | INCONCLUSIVE | 1.00 |
| pmrt_nl_eq | E3 | 0.5 | ok | INCONCLUSIVE | .99 |
| pmrt_nl_eq | E4 | 0.125 | ok | INCONCLUSIVE | .05 |
| pmrt_nl_eq | E4 | 0.25 | ok | INCONCLUSIVE | .00 |
| pmrt_nl_eq | E4 | 0.5 | ok | INCONCLUSIVE | .00 |
| pmrt_nl_eq | E5 | 0.125 | ok | INCONCLUSIVE | .71 |
| pmrt_nl_eq | E5 | 0.25 | ok | VALID | .68 |
| pmrt_nl_eq | E5 | 0.5 | ok | INCONCLUSIVE | .67 |
| pmrt_r3 | E1 | 0.125 | ok | VALID | 1.00 |
| pmrt_r3 | E1 | 0.25 | ok | VALID | 1.00 |
| pmrt_r3 | E1 | 0.5 | ok | INCONCLUSIVE | 1.00 |
| pmrt_r3 | E2 | 0.125 | ok | INCONCLUSIVE | .35 |
| pmrt_r3 | E2 | 0.25 | ok | VALID | .33 |
| pmrt_r3 | E2 | 0.5 | ok | INCONCLUSIVE | .32 |
| pmrt_r3 | E3 | 0.125 | ok | INCONCLUSIVE | 1.00 |
| pmrt_r3 | E3 | 0.25 | ok | INCONCLUSIVE | 1.00 |
| pmrt_r3 | E3 | 0.5 | ok | INCONCLUSIVE | 1.00 |
| pmrt_r3 | E4 | 0.125 | ok | INCONCLUSIVE | .05 |
| pmrt_r3 | E4 | 0.25 | ok | INCONCLUSIVE | .03 |
| pmrt_r3 | E4 | 0.5 | ok | INCONCLUSIVE | .05 |
| pmrt_r3 | E5 | 0.125 | ok | INCONCLUSIVE | .54 |
| pmrt_r3 | E5 | 0.25 | ok | VALID | .51 |
| pmrt_r3 | E5 | 0.5 | ok | VALID | .51 |
| rcot2_eq | E1 | 0.125 | ok | INVALID | 1.00 |
| rcot2_eq | E1 | 0.25 | ok | INVALID | 1.00 |
| rcot2_eq | E1 | 0.5 | ok | INCONCLUSIVE | .90 |
| rcot2_eq | E2 | 0.125 | ok | INVALID | .50 |
| rcot2_eq | E2 | 0.25 | ok | INVALID | .51 |
| rcot2_eq | E2 | 0.5 | ok | INVALID | .46 |
| rcot2_eq | E3 | 0.125 | ok | INVALID | .96 |
| rcot2_eq | E3 | 0.25 | ok | INVALID | .96 |
| rcot2_eq | E3 | 0.5 | ok | INVALID | .79 |
| rcot2_eq | E4 | 0.125 | ok | INCONCLUSIVE | .05 |
| rcot2_eq | E4 | 0.25 | ok | INCONCLUSIVE | .03 |
| rcot2_eq | E4 | 0.5 | ok | INCONCLUSIVE | .05 |
| rcot2_eq | E5 | 0.125 | ok | INVALID | .67 |
| rcot2_eq | E5 | 0.25 | ok | INVALID | .66 |
| rcot2_eq | E5 | 0.5 | ok | INVALID | .66 |
| rcot2_eq_min | E1 | 0.125 | ok | INVALID | 1.00 |
| rcot2_eq_min | E1 | 0.25 | ok | INCONCLUSIVE | 1.00 |
| rcot2_eq_min | E1 | 0.5 | ok | INVALID | .85 |
| rcot2_eq_min | E2 | 0.125 | ok | INVALID | .53 |
| rcot2_eq_min | E2 | 0.25 | ok | INVALID | .51 |
| rcot2_eq_min | E2 | 0.5 | ok | INVALID | .47 |
| rcot2_eq_min | E3 | 0.125 | ok | INCONCLUSIVE | .95 |
| rcot2_eq_min | E3 | 0.25 | ok | INVALID | .94 |
| rcot2_eq_min | E3 | 0.5 | ok | INCONCLUSIVE | .76 |
| rcot2_eq_min | E4 | 0.125 | ok | INCONCLUSIVE | .05 |
| rcot2_eq_min | E4 | 0.25 | ok | VALID | .03 |
| rcot2_eq_min | E4 | 0.5 | ok | VALID | .05 |
| rcot2_eq_min | E5 | 0.125 | ok | INVALID | .67 |
| rcot2_eq_min | E5 | 0.25 | ok | INVALID | .67 |
| rcot2_eq_min | E5 | 0.5 | ok | INCONCLUSIVE | .65 |
| rcot2_native | E1 | 0.125 | ok | INVALID | 1.00 |
| rcot2_native | E1 | 0.25 | ok | INVALID | 1.00 |
| rcot2_native | E1 | 0.5 | ok | INVALID | .99 |
| rcot2_native | E2 | 0.125 | ok | INVALID | .55 |
| rcot2_native | E2 | 0.25 | ok | INVALID | .54 |
| rcot2_native | E2 | 0.5 | ok | INVALID | .53 |
| rcot2_native | E3 | 0.125 | ok | INVALID | .97 |
| rcot2_native | E3 | 0.25 | ok | INVALID | .99 |
| rcot2_native | E3 | 0.5 | ok | INVALID | .99 |
| rcot2_native | E4 | 0.125 | ok | INCONCLUSIVE | .15 |
| rcot2_native | E4 | 0.25 | ok | INCONCLUSIVE | .15 |
| rcot2_native | E4 | 0.5 | ok | INCONCLUSIVE | .10 |
| rcot2_native | E5 | 0.125 | ok | INVALID | .70 |
| rcot2_native | E5 | 0.25 | ok | INVALID | .68 |
| rcot2_native | E5 | 0.5 | ok | INVALID | .68 |
| shap_dag | E1 | 0.125 | ok | INVALID | 1.00 |
| shap_dag | E1 | 0.25 | ok | INVALID | 1.00 |
| shap_dag | E1 | 0.5 | ok | INVALID | 1.00 |
| shap_dag | E2 | 0.125 | ok | VALID | .90 |
| shap_dag | E2 | 0.25 | ok | VALID | .91 |
| shap_dag | E2 | 0.5 | ok | VALID | .89 |
| shap_dag | E3 | 0.125 | ok | INVALID | 1.00 |
| shap_dag | E3 | 0.25 | ok | INVALID | 1.00 |
| shap_dag | E3 | 0.5 | ok | INVALID | 1.00 |
| shap_dag | E4 | 0.125 | ok | NO_READ | .00 |
| shap_dag | E4 | 0.25 | ok | NO_READ | .03 |
| shap_dag | E4 | 0.5 | ok | NO_READ | .10 |
| shap_dag | E5 | 0.125 | ok | VALID | .75 |
| shap_dag | E5 | 0.25 | ok | VALID | .67 |
| shap_dag | E5 | 0.5 | ok | VALID | .67 |
| two_tower | E1 | 0.125 | ok | INVALID | .97 |
| two_tower | E1 | 0.25 | ok | INVALID | .97 |
| two_tower | E1 | 0.5 | ok | INVALID | .95 |
| two_tower | E2 | 0.125 | ok | VALID | .42 |
| two_tower | E2 | 0.25 | ok | VALID | .42 |
| two_tower | E2 | 0.5 | ok | VALID | .51 |
| two_tower | E3 | 0.125 | ok | INVALID | 1.00 |
| two_tower | E3 | 0.25 | ok | INVALID | .99 |
| two_tower | E3 | 0.5 | ok | INVALID | 1.00 |
| two_tower | E4 | 0.125 | ok | NO_READ | .15 |
| two_tower | E4 | 0.25 | ok | NO_READ | .03 |
| two_tower | E4 | 0.5 | ok | NO_READ | .00 |
| two_tower | E5 | 0.125 | ok | VALID | .72 |
| two_tower | E5 | 0.25 | ok | VALID | .68 |
| two_tower | E5 | 0.5 | ok | VALID | .67 |

## 8. Not in grid (pre-registered grid choices, R-54 / R-58)

Cells outside an arm's grid are not planned: they never count as missing, never cap a verdict at PARTIAL and never make a component NOT EVALUABLE; every table labels them `nig`.

| ruling | arms | where | cells | reason |
|---|---|---|---|---|
| R-58(1) | cdl | regimes ['R3', 'R4'] | 40 | cdl: training cost; feeds no claim (R1 / R2, n <= 4000 only) |
| R-58(1) | cdl | ns [8000, 24000] | 20 | cdl: training cost; feeds no claim (R1 / R2, n <= 4000 only) |
| R-58(1) | cdl | kappas [0.125, 0.5] | 10 | cdl: not in the kappa sweep |
| R-54 | mscr_eq, mscr_native | ns [4000, 8000, 24000] | 108 | mscr: reported-INVALID arm, EVAL n <= 1000 only |
| R-58(5) | mscr_eq, mscr_native | kappas [0.125, 0.5] | 20 | mscr: not in the kappa sweep |

198 cells in total; unexplained gaps: none.

## 9. Figure-ready CSVs

- `csv/cost.csv` (sha256 5f3199a87bbd44a6)
- `csv/e4_by_lambda.csv` (sha256 e19f05c2f9fcc166)
- `csv/eq_arms.csv` (sha256 11289e5a7dd6eea4)
- `csv/information_levels_R2.csv` (sha256 3260766939f74f69)
- `csv/kappa_sweep.csv` (sha256 1ea60f35029bcc12)
- `csv/like_for_like.csv` (sha256 54f34375bd6d341f)
- `csv/not_in_grid.csv` (sha256 15c14aaa4891c5cb)
- `csv/paired_diff.csv` (sha256 e9d9c1490af816ee)
- `csv/recall_vs_n.csv` (sha256 fec13186b5d0454d)
- `csv/validity_cells.csv` (sha256 a01b7a9c80de74bf)

## Appendix: every eval_analysis table (V0-V11)


### V0 like-for-like (headline, R-42): recall / truth-null rate / placebo rate per cell

Every p arm scored at raw p <= .05 per edge (raw_p) and with the conformal placebo tau (tau), next to the score-only arms (tau; cdl also at the conference's fixed threshold, fixed). Rates are declaration rates; * INVALID, ~ INCONCLUSIVE; a tau row's placebo is its tuning column (reported, not in its validity). NA = not applicable (PMRT in R4: no known design, never recall 0); inv INVALID, inf infeasible (EVAL or T3), mis missing, unt untuned, few < 10 seeds; nig not in the arm's grid (pre-registered grid choice, R-54 / R-58).

### E1 R1

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | raw_p | 1.00 / .027 / .075~ | 1.00 / .033 / .050~ | 1.00 / .044 / .069~ | 1.00 / .050 / .050~ | 1.00 / .056~ / .056~ |
| pmrt_nl_eq | tau | 1.00 / .029 / .081~ | 1.00 / .023 / .025 | 1.00 / .048 / .075~ | 1.00 / .025 / .025 | 1.00 / .027 / .019 |
| pmrt_eq | raw_p | 1.00 / .037 / .025 | 1.00 / .033 / .044~ | 1.00 / .052 / .087~ | 1.00 / .037 / .075~ | 1.00 / .044 / .050~ |
| pmrt_eq | tau | 1.00 / .029 / .025 | 1.00 / .087* / .131* | 1.00 / .052 / .100* | 1.00 / .035 / .062~ | 1.00 / .062~ / .062~ |
| pmrt_r3 | raw_p | 1.00 / .044 / .037 | 1.00 / .050 / .050~ | 1.00 / .050 / .087* | 1.00 / .052 / .056~ | 1.00 / .044 / .044 |
| pmrt_r3 | tau | 1.00 / .040 / .031 | 1.00 / .069~ / .075~ | 1.00 / .052 / .087* | 1.00 / .037 / .056~ | 1.00 / .037 / .044 |
| cdl | - | - | - | - | nig | nig |
| cdl | tau | 1.00 / .075* / .069~ | 1.00 / .081* / .087~ | 1.00 / .037 / .025 | nig | nig |
| cdl | fixed | .00 / .000 / .000 | .75 / .000 / .000 | 1.00 / .000 / .000 | nig | nig |
| corr | raw_p | 1.00 / .037 / .044 | 1.00 / .019 / .025 | 1.00 / .056~ / .062~ | 1.00 / .029 / .062~ | 1.00 / .023 / .050~ |
| corr | tau | 1.00 / .058~ / .069~ | 1.00 / .017 / .025 | 1.00 / .031 / .050~ | 1.00 / .027 / .056~ | 1.00 / .004 / .019 |
| mscr_eq | raw_p | 1.00 / .048 / .075~ | 1.00 / .040 / .056~ | nig | nig | nig |
| mscr_eq | tau | 1.00 / .060~ / .075~ | 1.00 / .037 / .056~ | nig | nig | nig |
| mscr_native | raw_p | 1.00 / .040 / .056~ | 1.00 / .054 / .031 | nig | nig | nig |
| mscr_native | tau | 1.00 / .058~ / .069~ | 1.00 / .019 / .025 | nig | nig | nig |
| notears | tau | 1.00 / .000 / .000 | 1.00 / .000 / .000 | 1.00 / .000 / .000 | 1.00 / .000 / .000 | 1.00 / .000 / .000 |
| pc_eq | tau | 1.00 / .092* / .113* | 1.00 / .023 / .019 | 1.00 / .040 / .050~ | 1.00 / .021 / .025 | 1.00 / .048 / .069~ |
| pc_native | tau | 1.00 / .094* / .125* | 1.00 / .025 / .019 | 1.00 / .042 / .044~ | 1.00 / .019 / .031 | 1.00 / .054~ / .069~ |
| pcorr_eq | raw_p | 1.00 / .029 / .031 | 1.00 / .025 / .037 | 1.00 / .046 / .087~ | 1.00 / .037 / .075~ | 1.00 / .050 / .050~ |
| pcorr_eq | tau | 1.00 / .019 / .013 | 1.00 / .077* / .087~ | 1.00 / .044 / .075~ | 1.00 / .031 / .069~ | 1.00 / .065~ / .069~ |
| pcorr_eq_min | raw_p | 1.00 / .040 / .037 | 1.00 / .033 / .037 | 1.00 / .050 / .069~ | 1.00 / .046 / .050~ | 1.00 / .037 / .050~ |
| pcorr_eq_min | tau | 1.00 / .037 / .037 | 1.00 / .048 / .050~ | 1.00 / .054 / .081~ | 1.00 / .033 / .044 | 1.00 / .037 / .050~ |
| pcorr_hac | raw_p | 1.00 / .044 / .056~ | 1.00 / .035 / .037 | 1.00 / .048 / .069~ | 1.00 / .044 / .050~ | 1.00 / .037 / .050~ |
| pcorr_hac | tau | 1.00 / .044 / .044~ | 1.00 / .062~ / .050~ | 1.00 / .054 / .075~ | 1.00 / .033 / .044 | 1.00 / .035 / .050~ |
| pcorr_hac_eq_min | raw_p | 1.00 / .044 / .056~ | 1.00 / .035 / .037 | 1.00 / .048 / .069~ | 1.00 / .044 / .050~ | 1.00 / .037 / .050~ |
| pcorr_hac_eq_min | tau | 1.00 / .044 / .044~ | 1.00 / .062~ / .050~ | 1.00 / .054 / .075~ | 1.00 / .033 / .044 | 1.00 / .035 / .050~ |
| pcorr_hac_fb | raw_p | 1.00 / .044 / .050~ | 1.00 / .035 / .037 | 1.00 / .048 / .069~ | 1.00 / .044 / .050~ | 1.00 / .037 / .050~ |
| pcorr_hac_fb | tau | 1.00 / .044 / .044~ | 1.00 / .062~ / .050~ | 1.00 / .054 / .075~ | 1.00 / .033 / .044 | 1.00 / .035 / .050~ |
| pcorr_hac_fb_eq_min | raw_p | 1.00 / .044 / .050~ | 1.00 / .035 / .037 | 1.00 / .048 / .069~ | 1.00 / .044 / .050~ | 1.00 / .037 / .050~ |
| pcorr_hac_fb_eq_min | tau | 1.00 / .044 / .044~ | 1.00 / .062~ / .050~ | 1.00 / .054 / .075~ | 1.00 / .033 / .044 | 1.00 / .035 / .050~ |
| pcorr_native | raw_p | 1.00 / .040 / .037 | 1.00 / .033 / .037 | 1.00 / .050 / .069~ | 1.00 / .046 / .050~ | 1.00 / .037 / .050~ |
| pcorr_native | tau | 1.00 / .037 / .037 | 1.00 / .048 / .050~ | 1.00 / .054 / .081~ | 1.00 / .033 / .044 | 1.00 / .037 / .050~ |
| rcot2_eq | raw_p | 1.00 / .083* / .087~ | 1.00 / .069~ / .050~ | 1.00 / .060~ / .069~ | 1.00 / .050 / .069~ | 1.00 / .046 / .056~ |
| rcot2_eq | tau | 1.00 / .069* / .050~ | 1.00 / .054 / .044 | 1.00 / .023 / .044 | 1.00 / .050 / .081~ | 1.00 / .062~ / .044~ |
| rcot2_eq_min | raw_p | 1.00 / .102* / .087~ | 1.00 / .092* / .087~ | 1.00 / .065~ / .081~ | 1.00 / .069~ / .081~ | 1.00 / .052 / .050~ |
| rcot2_eq_min | tau | 1.00 / .077* / .094* | 1.00 / .085* / .069~ | 1.00 / .021 / .031 | 1.00 / .042 / .075~ | 1.00 / .087* / .100~ |
| rcot2_native | raw_p | 1.00 / .094* / .113* | 1.00 / .094* / .081~ | 1.00 / .062~ / .075~ | 1.00 / .060~ / .081~ | 1.00 / .035 / .056~ |
| rcot2_native | tau | 1.00 / .040 / .050~ | 1.00 / .048 / .044 | 1.00 / .031 / .037 | 1.00 / .029 / .056~ | 1.00 / .040 / .056~ |
| shap_dag | tau | 1.00 / .035 / .025 | 1.00 / .029 / .019 | 1.00 / .133* / .081~ | 1.00 / .125* / .081~ | 1.00 / .037 / .019 |
| two_tower | tau | 1.00 / .021 / .062~ | 1.00 / .025 / .037 | 1.00 / .027 / .050~ | 1.00 / .033 / .037 | 1.00 / .044 / .044~ |

### E1 R2

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | raw_p | 1.00 / .033 / .069~ | 1.00 / .033 / .050~ | 1.00 / .040 / .056~ | 1.00 / .062~ / .056~ | 1.00 / .075* / .069~ |
| pmrt_nl_eq | tau | 1.00 / .033 / .044~ | 1.00 / .025 / .037 | 1.00 / .058~ / .087~ | 1.00 / .042 / .056~ | 1.00 / .046 / .031 |
| pmrt_eq | raw_p | .97 / .048 / .069~ | .99 / .050 / .037 | .99 / .054~ / .031 | .99 / .050 / .044 | .99 / .048 / .044~ |
| pmrt_eq | tau | .98 / .054~ / .062~ | .99 / .035 / .025 | .99 / .104* / .100* | .99 / .037 / .037 | .99 / .021 / .037 |
| pmrt_r3 | raw_p | 1.00 / .073~ / .050~ | 1.00 / .046 / .025 | 1.00 / .037 / .044 | 1.00 / .052 / .062~ | 1.00 / .046 / .044~ |
| pmrt_r3 | tau | 1.00 / .117* / .081~ | 1.00 / .027 / .025 | 1.00 / .046 / .037 | 1.00 / .046 / .044 | 1.00 / .071~ / .075~ |
| cdl | - | - | - | - | nig | nig |
| cdl | tau | 1.00 / .027 / .013 | 1.00 / .106* / .069~ | 1.00 / .183* / .019 | nig | nig |
| cdl | fixed | .00 / .000 / .000 | .00 / .000 / .000 | .42 / .000 / .000 | nig | nig |
| corr | raw_p | 1.00 / .683* / .644* | 1.00 / .758* / .750* | 1.00 / .887* / .863* | 1.00 / .921* / .912* | 1.00 / .954* / .925* |
| corr | tau | 1.00 / .188* / .100* | 1.00 / .177* / .094~ | 1.00 / .183* / .081~ | 1.00 / .192* / .087~ | 1.00 / .185* / .087~ |
| mscr_eq | raw_p | 1.00 / 1.000* / 1.000* | 1.00 / 1.000* / 1.000* | nig | nig | nig |
| mscr_eq | tau | 1.00 / .104* / .031 | 1.00 / .127* / .056~ | nig | nig | nig |
| mscr_native | raw_p | 1.00 / 1.000* / .969* | 1.00 / 1.000* / 1.000* | nig | nig | nig |
| mscr_native | tau | 1.00 / .221* / .087~ | 1.00 / .194* / .044~ | nig | nig | nig |
| notears | tau | 1.00 / .165* / .000 | 1.00 / .169* / .000 | 1.00 / .167* / .000 | 1.00 / .169* / .000 | 1.00 / .169* / .000 |
| pc_eq | tau | 1.00 / .098* / .087~ | 1.00 / .069~ / .037 | 1.00 / .031 / .025 | 1.00 / .060~ / .025 | 1.00 / .035 / .019 |
| pc_native | tau | 1.00 / .067~ / .087~ | 1.00 / .027 / .013 | 1.00 / .027 / .019 | 1.00 / .065~ / .050~ | 1.00 / .048 / .013 |
| pcorr_eq | raw_p | 1.00 / .056~ / .044 | 1.00 / .042 / .050~ | 1.00 / .054~ / .044 | 1.00 / .056~ / .062~ | 1.00 / .042 / .037 |
| pcorr_eq | tau | 1.00 / .127* / .100~ | 1.00 / .023 / .031 | 1.00 / .058~ / .056~ | 1.00 / .029 / .056~ | 1.00 / .033 / .037 |
| pcorr_eq_min | raw_p | 1.00 / .067~ / .031 | 1.00 / .037 / .037~ | 1.00 / .050 / .044 | 1.00 / .058~ / .062~ | 1.00 / .046 / .050~ |
| pcorr_eq_min | tau | 1.00 / .106* / .062~ | 1.00 / .019 / .019 | 1.00 / .023 / .025 | 1.00 / .042 / .056~ | 1.00 / .065~ / .062~ |
| pcorr_hac | raw_p | 1.00 / .212* / .044 | 1.00 / .198* / .031 | 1.00 / .196* / .081~ | 1.00 / .225* / .100* | 1.00 / .242* / .219* |
| pcorr_hac | tau | 1.00 / .188* / .025 | 1.00 / .177* / .019 | 1.00 / .175* / .037 | 1.00 / .177* / .031 | 1.00 / .169* / .006 |
| pcorr_hac_eq_min | raw_p | 1.00 / .067~ / .031 | 1.00 / .040 / .050~ | 1.00 / .054 / .050~ | 1.00 / .058~ / .062~ | 1.00 / .046 / .050~ |
| pcorr_hac_eq_min | tau | 1.00 / .108* / .056~ | 1.00 / .019 / .019 | 1.00 / .023 / .031 | 1.00 / .044 / .056~ | 1.00 / .060~ / .062~ |
| pcorr_hac_fb | raw_p | 1.00 / .210* / .044 | 1.00 / .198* / .031 | 1.00 / .196* / .081~ | 1.00 / .225* / .100* | 1.00 / .242* / .219* |
| pcorr_hac_fb | tau | 1.00 / .188* / .025 | 1.00 / .177* / .019 | 1.00 / .175* / .037 | 1.00 / .177* / .031 | 1.00 / .169* / .006 |
| pcorr_hac_fb_eq_min | raw_p | 1.00 / .067~ / .031 | 1.00 / .040 / .050~ | 1.00 / .054 / .050~ | 1.00 / .058~ / .062~ | 1.00 / .046 / .050~ |
| pcorr_hac_fb_eq_min | tau | 1.00 / .110* / .056~ | 1.00 / .019 / .019 | 1.00 / .023 / .031 | 1.00 / .044 / .056~ | 1.00 / .060~ / .062~ |
| pcorr_native | raw_p | 1.00 / .210* / .037 | 1.00 / .198* / .025 | 1.00 / .198* / .069~ | 1.00 / .225* / .094~ | 1.00 / .242* / .219* |
| pcorr_native | tau | 1.00 / .202* / .025 | 1.00 / .177* / .013 | 1.00 / .177* / .031 | 1.00 / .181* / .031 | 1.00 / .169* / .006 |
| rcot2_eq | raw_p | 1.00 / .104* / .106* | 1.00 / .083* / .075~ | 1.00 / .237* / .156* | 1.00 / .369* / .325* | 1.00 / .708* / .637* |
| rcot2_eq | tau | .96 / .048 / .056~ | .99 / .054 / .087~ | 1.00 / .133* / .119* | 1.00 / .065~ / .062~ | 1.00 / .073~ / .044 |
| rcot2_eq_min | raw_p | 1.00 / .079* / .062~ | 1.00 / .069~ / .062~ | 1.00 / .135* / .094~ | 1.00 / .215* / .181* | 1.00 / .529* / .425* |
| rcot2_eq_min | tau | .97 / .033 / .050~ | .99 / .058~ / .087~ | 1.00 / .058~ / .044 | 1.00 / .040 / .062~ | 1.00 / .058~ / .037 |
| rcot2_native | raw_p | 1.00 / .156* / .131* | 1.00 / .152* / .106* | 1.00 / .204* / .150* | 1.00 / .344* / .362* | 1.00 / .610* / .619* |
| rcot2_native | tau | 1.00 / .052 / .106* | 1.00 / .067~ / .113* | 1.00 / .081* / .062~ | 1.00 / .046 / .044~ | 1.00 / .042 / .037 |
| shap_dag | tau | 1.00 / .246* / .081~ | 1.00 / .225* / .050~ | 1.00 / .219* / .081~ | 1.00 / .202* / .075~ | 1.00 / .198* / .044~ |
| two_tower | tau | .97 / .198* / .037 | .97 / .204* / .056~ | .99 / .212* / .037 | .99 / .196* / .013 | .99 / .223* / .019 |

### E2 R1

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | raw_p | .83 / .041 / .021 | .88 / .037 / .033 | .95 / .045 / .046 | .97 / .045 / .033 | 1.00 / .045 / .050~ |
| pmrt_nl_eq | tau | .83 / .042 / .017 | .88 / .039 / .033 | .96 / .060 / .050~ | .97 / .056 / .042 | 1.00 / .039 / .042 |
| pmrt_eq | raw_p | .35 / .051 / .050~ | .35 / .046 / .046 | .40 / .050 / .050~ | .39 / .035 / .029 | .40 / .050 / .071~ |
| pmrt_eq | tau | .42 / .112* / .096* | .39 / .094* / .079~ | .40 / .056 / .050~ | .38 / .024 / .017 | .39 / .030 / .037 |
| pmrt_r3 | raw_p | .36 / .044 / .054~ | .36 / .043 / .042 | .40 / .052 / .046~ | .40 / .036 / .029 | .40 / .051 / .071~ |
| pmrt_r3 | tau | .41 / .112* / .083~ | .40 / .123* / .096* | .40 / .052 / .046~ | .38 / .023 / .017 | .39 / .028 / .033 |
| cdl | - | - | - | - | nig | nig |
| cdl | tau | .61 / .073* / .075~ | .70 / .078* / .042 | .96 / .050 / .033 | nig | nig |
| cdl | fixed | .00 / .000 / .000 | .00 / .000 / .000 | .53 / .000 / .000 | nig | nig |
| corr | raw_p | .35 / .055 / .058~ | .36 / .052 / .033 | .40 / .053 / .046 | .40 / .040 / .054~ | .40 / .053 / .046~ |
| corr | tau | .39 / .090* / .079~ | .39 / .091* / .071~ | .40 / .053 / .046 | .38 / .026 / .021 | .39 / .021 / .029 |
| mscr_eq | raw_p | .91 / .052 / .058~ | .95 / .042 / .046 | nig | nig | nig |
| mscr_eq | tau | .92 / .048 / .062~ | .95 / .031 / .033 | nig | nig | nig |
| mscr_native | raw_p | .92 / .054 / .042 | .95 / .046 / .029 | nig | nig | nig |
| mscr_native | tau | .93 / .073* / .058~ | .95 / .027 / .025 | nig | nig | nig |
| notears | tau | .25 / .016 / .008 | .26 / .000 / .000 | .25 / .000 / .000 | .25 / .000 / .000 | .25 / .000 / .000 |
| pc_eq | tau | .31 / .069* / .046 | .32 / .057 / .050~ | .38 / .022 / .013 | .38 / .013 / .029 | .41 / .056 / .062~ |
| pc_native | tau | .32 / .069* / .050~ | .33 / .065* / .058~ | .38 / .022 / .013 | .38 / .012 / .025 | .41 / .055 / .058~ |
| pcorr_eq | raw_p | .35 / .054 / .046~ | .35 / .047 / .037 | .40 / .054 / .037 | .40 / .038 / .042 | .40 / .050 / .062~ |
| pcorr_eq | tau | .39 / .108* / .100* | .37 / .070* / .067~ | .40 / .063* / .046~ | .38 / .019 / .004 | .39 / .031 / .042 |
| pcorr_eq_min | raw_p | .35 / .060 / .054~ | .35 / .046 / .050~ | .40 / .052 / .037 | .40 / .038 / .042 | .40 / .052 / .067~ |
| pcorr_eq_min | tau | .37 / .075* / .075~ | .39 / .102* / .083~ | .40 / .061 / .042 | .38 / .024 / .008 | .39 / .028 / .037 |
| pcorr_hac | raw_p | .35 / .055 / .071~ | .36 / .045 / .050~ | .40 / .052 / .037 | .39 / .040 / .037 | .40 / .052 / .067~ |
| pcorr_hac | tau | .37 / .076* / .083~ | .39 / .100* / .092* | .41 / .064* / .054~ | .38 / .023 / .004 | .39 / .027 / .037 |
| pcorr_hac_eq_min | raw_p | .35 / .055 / .071~ | .36 / .045 / .050~ | .40 / .052 / .037 | .39 / .040 / .037 | .40 / .052 / .067~ |
| pcorr_hac_eq_min | tau | .37 / .076* / .083~ | .39 / .100* / .092* | .41 / .064* / .054~ | .38 / .023 / .004 | .39 / .027 / .037 |
| pcorr_hac_fb | raw_p | .35 / .055 / .067~ | .36 / .044 / .050~ | .40 / .052 / .037 | .39 / .040 / .037 | .40 / .052 / .067~ |
| pcorr_hac_fb | tau | .37 / .077* / .083~ | .39 / .100* / .092* | .41 / .064* / .054~ | .38 / .023 / .004 | .39 / .027 / .037 |
| pcorr_hac_fb_eq_min | raw_p | .35 / .055 / .067~ | .36 / .044 / .050~ | .40 / .052 / .037 | .39 / .040 / .037 | .40 / .052 / .067~ |
| pcorr_hac_fb_eq_min | tau | .37 / .077* / .083~ | .39 / .100* / .092* | .41 / .064* / .054~ | .38 / .023 / .004 | .39 / .027 / .037 |
| pcorr_native | raw_p | .35 / .060 / .054~ | .35 / .046 / .050~ | .40 / .052 / .037 | .40 / .038 / .042 | .40 / .052 / .067~ |
| pcorr_native | tau | .37 / .075* / .075~ | .39 / .102* / .083~ | .40 / .061 / .042 | .38 / .024 / .008 | .39 / .028 / .037 |
| rcot2_eq | raw_p | .60 / .094* / .092* | .61 / .065* / .050 | .66 / .052 / .054~ | .69 / .051 / .042 | .73 / .041 / .042 |
| rcot2_eq | tau | .55 / .056 / .058~ | .58 / .042 / .037 | .63 / .043 / .054~ | .67 / .053 / .037 | .70 / .066* / .083~ |
| rcot2_eq_min | raw_p | .61 / .087* / .092* | .61 / .061 / .067~ | .66 / .053 / .058~ | .69 / .051 / .033 | .72 / .044 / .042 |
| rcot2_eq_min | tau | .55 / .043 / .042 | .59 / .037 / .050 | .62 / .028 / .042 | .66 / .046 / .037 | .70 / .051 / .050 |
| rcot2_native | raw_p | .61 / .097* / .079~ | .62 / .059 / .062~ | .66 / .052 / .046 | .69 / .051 / .029 | .72 / .046 / .042 |
| rcot2_native | tau | .55 / .057 / .054~ | .59 / .037 / .042 | .63 / .026 / .042 | .64 / .027 / .025 | .71 / .069* / .096* |
| shap_dag | tau | .87 / .065* / .046 | .95 / .041 / .042 | .99 / .041 / .042 | 1.00 / .056 / .062~ | 1.00 / .032 / .021 |
| two_tower | tau | .67 / .083* / .079~ | .67 / .041 / .033 | .78 / .051 / .021 | .77 / .081* / .017 | .78 / .176* / .042 |

### E2 R2

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | raw_p | .64 / .052 / .037 | .78 / .049 / .054~ | .93 / .052 / .037 | .94 / .041 / .054~ | .94 / .051 / .042 |
| pmrt_nl_eq | tau | .65 / .083* / .067~ | .80 / .078* / .108* | .94 / .111* / .108* | .94 / .078* / .083~ | .94 / .048 / .033 |
| pmrt_eq | raw_p | .36 / .057 / .033 | .45 / .053 / .029 | .60 / .037 / .042 | .65 / .034 / .046 | .78 / .058 / .033 |
| pmrt_eq | tau | .38 / .066* / .054~ | .44 / .047 / .021 | .62 / .055 / .067~ | .61 / .018 / .029 | .75 / .034 / .017 |
| pmrt_r3 | raw_p | .40 / .058 / .033 | .47 / .047 / .029 | .65 / .045 / .033 | .71 / .041 / .067~ | .81 / .055 / .050~ |
| pmrt_r3 | tau | .41 / .066~ / .042 | .49 / .060 / .058~ | .64 / .033 / .025 | .73 / .056 / .079~ | .81 / .067* / .067~ |
| cdl | - | - | - | - | nig | nig |
| cdl | tau | .40 / .030 / .046 | .49 / .027 / .017 | .80 / .037 / .033 | nig | nig |
| cdl | fixed | .00 / .000 / .000 | .00 / .000 / .000 | .43 / .000 / .000 | nig | nig |
| corr | raw_p | .67 / .506* / .442* | .75 / .606* / .583* | .87 / .795* / .771* | .90 / .858* / .838* | .94 / .918* / .933* |
| corr | tau | .27 / .018 / .017 | .28 / .013 / .021 | .28 / .016 / .017 | .28 / .015 / .021 | .28 / .014 / .021 |
| mscr_eq | raw_p | .95 / .883* / .875* | .99 / .945* / .950* | nig | nig | nig |
| mscr_eq | tau | .57 / .073* / .062~ | .53 / .051 / .046 | nig | nig | nig |
| mscr_native | raw_p | .95 / .841* / .817* | .99 / .924* / .925* | nig | nig | nig |
| mscr_native | tau | .78 / .118* / .092* | .81 / .119* / .104* | nig | nig | nig |
| notears | tau | .28 / .033 / .029 | .29 / .037 / .033 | .30 / .041 / .029 | .29 / .029 / .029 | .28 / .028 / .029 |
| pc_eq | tau | .23 / .030 / .017 | .30 / .073* / .067~ | .40 / .062 / .067~ | .47 / .045 / .087* | .50 / .052 / .042 |
| pc_native | tau | .24 / .023 / .013 | .29 / .035 / .017 | .26 / .021 / .017 | .00 / .000 / .000 | .00 / .000 / .000 |
| pcorr_eq | raw_p | .43 / .055 / .071~ | .52 / .059 / .021 | .69 / .039 / .033 | .74 / .035 / .054~ | .83 / .055 / .042 |
| pcorr_eq | tau | .39 / .027 / .033 | .51 / .049 / .021 | .68 / .027 / .013 | .72 / .028 / .037 | .85 / .080* / .067~ |
| pcorr_eq_min | raw_p | .44 / .050 / .042 | .52 / .059 / .029 | .69 / .041 / .021 | .74 / .038 / .050~ | .83 / .059 / .037 |
| pcorr_eq_min | tau | .39 / .025 / .025 | .53 / .066~ / .042 | .70 / .048 / .021 | .72 / .021 / .037 | .83 / .071* / .054~ |
| pcorr_hac | raw_p | .53 / .341* / .375* | .60 / .471* / .463* | .78 / .691* / .662* | .83 / .780* / .800* | .89 / .862* / .867* |
| pcorr_hac | tau | .25 / .020 / .029 | .28 / .034 / .025 | .29 / .052 / .042 | .28 / .045 / .046 | .28 / .046 / .050~ |
| pcorr_hac_eq_min | raw_p | .37 / .050 / .046~ | .42 / .062~ / .029 | .62 / .044 / .017 | .68 / .038 / .050~ | .78 / .059 / .042 |
| pcorr_hac_eq_min | tau | .33 / .032 / .025 | .45 / .074* / .042 | .62 / .040 / .017 | .66 / .023 / .037 | .79 / .069* / .054~ |
| pcorr_hac_fb | raw_p | .53 / .340* / .371* | .60 / .470* / .463* | .78 / .690* / .662* | .83 / .779* / .800* | .88 / .862* / .867* |
| pcorr_hac_fb | tau | .25 / .020 / .029 | .27 / .033 / .025 | .29 / .054 / .046 | .28 / .045 / .046 | .28 / .049 / .050~ |
| pcorr_hac_fb_eq_min | raw_p | .36 / .049 / .042 | .42 / .062~ / .029 | .62 / .044 / .017 | .68 / .038 / .050~ | .78 / .059 / .042 |
| pcorr_hac_fb_eq_min | tau | .33 / .032 / .029 | .45 / .076* / .042 | .62 / .040 / .017 | .66 / .023 / .037 | .79 / .069* / .054~ |
| pcorr_native | raw_p | .54 / .352* / .346* | .64 / .468* / .450* | .81 / .691* / .667* | .85 / .784* / .800* | .91 / .873* / .879* |
| pcorr_native | tau | .28 / .022 / .029 | .30 / .030 / .021 | .32 / .035 / .042 | .31 / .031 / .029 | .30 / .028 / .025 |
| rcot2_eq | raw_p | .57 / .135* / .121* | .64 / .112* / .121* | .73 / .227* / .275* | .81 / .355* / .329* | .89 / .552* / .554* |
| rcot2_eq | tau | .46 / .073* / .071~ | .54 / .057 / .050~ | .56 / .016 / .021 | .61 / .026 / .029 | .62 / .030 / .037 |
| rcot2_eq_min | raw_p | .58 / .108* / .121* | .66 / .113* / .108* | .73 / .226* / .258* | .82 / .338* / .346* | .90 / .551* / .533* |
| rcot2_eq_min | tau | .45 / .062~ / .079~ | .53 / .048 / .033 | .58 / .033 / .037 | .59 / .020 / .037 | .57 / .018 / .042 |
| rcot2_native | raw_p | .62 / .167* / .212* | .68 / .189* / .179* | .74 / .304* / .350* | .82 / .454* / .475* | .92 / .674* / .658* |
| rcot2_native | tau | .43 / .032 / .062~ | .54 / .038 / .054~ | .55 / .013 / .029 | .57 / .016 / .033 | .58 / .013 / .033 |
| shap_dag | tau | .77 / .030 / .029 | .91 / .046 / .021 | .96 / .032 / .013 | .97 / .054 / .067~ | .98 / .050 / .050~ |
| two_tower | tau | .55 / .062~ / .058~ | .42 / .016 / .008 | .66 / .045 / .037 | .58 / .028 / .008 | .60 / .028 / .004 |

### E3 R1

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | raw_p | 1.00 / .036 / .055~ | 1.00 / .039 / .070~ | 1.00 / .042 / .035 | 1.00 / .048 / .045 | 1.00 / .034 / .020 |
| pmrt_nl_eq | tau | 1.00 / .044 / .065~ | 1.00 / .045 / .080~ | 1.00 / .048 / .055~ | 1.00 / .092* / .095* | 1.00 / .059~ / .050 |
| pmrt_eq | raw_p | 1.00 / .037 / .040 | 1.00 / .059~ / .045 | 1.00 / .047 / .050~ | 1.00 / .045 / .055~ | 1.00 / .058 / .035 |
| pmrt_eq | tau | 1.00 / .044 / .045 | 1.00 / .034 / .035 | 1.00 / .047 / .055~ | 1.00 / .045 / .050~ | 1.00 / .091* / .050~ |
| pmrt_r3 | raw_p | 1.00 / .045 / .070~ | 1.00 / .058~ / .045 | 1.00 / .041 / .050~ | 1.00 / .041 / .050~ | 1.00 / .055 / .050~ |
| pmrt_r3 | tau | 1.00 / .059~ / .075~ | 1.00 / .044 / .045 | 1.00 / .050 / .070~ | 1.00 / .039 / .055~ | 1.00 / .097* / .100* |
| cdl | - | - | - | - | nig | nig |
| cdl | tau | 1.00 / .034 / .060~ | 1.00 / .047 / .065~ | 1.00 / .017 / .055~ | nig | nig |
| cdl | fixed | .00 / .000 / .000 | .57 / .000 / .000 | 1.00 / .000 / .000 | nig | nig |
| corr | raw_p | 1.00 / .044 / .060~ | 1.00 / .052 / .045 | 1.00 / .058~ / .040 | 1.00 / .027 / .075~ | 1.00 / .056~ / .060~ |
| corr | tau | 1.00 / .036 / .055~ | 1.00 / .070~ / .060~ | 1.00 / .061~ / .050~ | 1.00 / .053 / .105* | 1.00 / .066~ / .080~ |
| granger_eq | raw_p | 1.00 / .042 / .055~ | 1.00 / .050 / .070~ | 1.00 / .050 / .060~ | 1.00 / .045 / .040 | 1.00 / .055~ / .035 |
| granger_eq | tau | 1.00 / .039 / .045~ | 1.00 / .027 / .035 | 1.00 / .053 / .060~ | 1.00 / .070* / .070~ | 1.00 / .094* / .045~ |
| granger_native | raw_p | 1.00 / .045 / .065~ | 1.00 / .053 / .040 | 1.00 / .058~ / .040 | 1.00 / .030 / .075~ | 1.00 / .056~ / .060~ |
| granger_native | tau | 1.00 / .036 / .060~ | 1.00 / .073* / .065~ | 1.00 / .064~ / .050~ | 1.00 / .056~ / .110* | 1.00 / .066~ / .080~ |
| mscr_eq | raw_p | 1.00 / .050 / .050~ | 1.00 / .045 / .035 | nig | nig | nig |
| mscr_eq | tau | 1.00 / .061~ / .060~ | 1.00 / .045 / .030 | nig | nig | nig |
| mscr_native | raw_p | 1.00 / .044 / .040 | 1.00 / .056 / .025 | nig | nig | nig |
| mscr_native | tau | 1.00 / .041 / .035 | 1.00 / .048 / .030 | nig | nig | nig |
| notears | tau | 1.00 / .000 / .000 | 1.00 / .000 / .000 | 1.00 / .000 / .000 | 1.00 / .000 / .000 | 1.00 / .000 / .000 |
| pc_eq | tau | 1.00 / .017 / .025 | 1.00 / .061~ / .050~ | 1.00 / .044 / .055~ | 1.00 / .069~ / .070~ | 1.00 / .037 / .040 |
| pc_native | tau | 1.00 / .017 / .025 | 1.00 / .045 / .050~ | 1.00 / .045 / .060~ | 1.00 / .064~ / .080~ | 1.00 / .037 / .050~ |
| pcorr_eq | raw_p | 1.00 / .042 / .055~ | 1.00 / .050 / .070~ | 1.00 / .050 / .060~ | 1.00 / .045 / .040 | 1.00 / .055~ / .035 |
| pcorr_eq | tau | 1.00 / .039 / .045~ | 1.00 / .027 / .035 | 1.00 / .053 / .060~ | 1.00 / .070* / .070~ | 1.00 / .094* / .045~ |
| pcorr_eq_min | raw_p | 1.00 / .037 / .065~ | 1.00 / .050 / .055~ | 1.00 / .048 / .035 | 1.00 / .042 / .050~ | 1.00 / .056 / .060~ |
| pcorr_eq_min | tau | 1.00 / .039 / .080~ | 1.00 / .047 / .055~ | 1.00 / .066~ / .080~ | 1.00 / .025 / .040 | 1.00 / .091* / .075~ |
| pcorr_hac | raw_p | 1.00 / .037 / .060~ | 1.00 / .048 / .060~ | 1.00 / .048 / .040 | 1.00 / .042 / .060~ | 1.00 / .056 / .060~ |
| pcorr_hac | tau | 1.00 / .037 / .055~ | 1.00 / .048 / .060~ | 1.00 / .062~ / .075~ | 1.00 / .023 / .040 | 1.00 / .089* / .070~ |
| pcorr_hac_eq_min | raw_p | 1.00 / .037 / .060~ | 1.00 / .048 / .060~ | 1.00 / .048 / .040 | 1.00 / .042 / .060~ | 1.00 / .056 / .060~ |
| pcorr_hac_eq_min | tau | 1.00 / .037 / .055~ | 1.00 / .048 / .060~ | 1.00 / .062~ / .075~ | 1.00 / .023 / .040 | 1.00 / .089* / .070~ |
| pcorr_hac_fb | raw_p | 1.00 / .037 / .060~ | 1.00 / .048 / .060~ | 1.00 / .048 / .040 | 1.00 / .041 / .060~ | 1.00 / .056 / .060~ |
| pcorr_hac_fb | tau | 1.00 / .037 / .055~ | 1.00 / .048 / .060~ | 1.00 / .062~ / .075~ | 1.00 / .023 / .040 | 1.00 / .089* / .070~ |
| pcorr_hac_fb_eq_min | raw_p | 1.00 / .037 / .060~ | 1.00 / .048 / .060~ | 1.00 / .048 / .040 | 1.00 / .041 / .060~ | 1.00 / .056 / .060~ |
| pcorr_hac_fb_eq_min | tau | 1.00 / .037 / .055~ | 1.00 / .048 / .060~ | 1.00 / .062~ / .075~ | 1.00 / .023 / .040 | 1.00 / .089* / .070~ |
| pcorr_native | raw_p | 1.00 / .037 / .065~ | 1.00 / .050 / .055~ | 1.00 / .048 / .035 | 1.00 / .042 / .050~ | 1.00 / .056 / .060~ |
| pcorr_native | tau | 1.00 / .039 / .080~ | 1.00 / .047 / .055~ | 1.00 / .066~ / .080~ | 1.00 / .025 / .040 | 1.00 / .091* / .075~ |
| rcot2_eq | raw_p | 1.00 / .098* / .075~ | 1.00 / .069~ / .060~ | 1.00 / .055 / .055~ | 1.00 / .048 / .075~ | 1.00 / .045 / .040 |
| rcot2_eq | tau | 1.00 / .075* / .060~ | 1.00 / .048 / .030 | 1.00 / .081* / .110* | 1.00 / .067~ / .115* | 1.00 / .047 / .045 |
| rcot2_eq_min | raw_p | 1.00 / .095* / .105* | 1.00 / .078* / .065~ | 1.00 / .058 / .065~ | 1.00 / .045 / .080~ | 1.00 / .053 / .065~ |
| rcot2_eq_min | tau | 1.00 / .089* / .115* | 1.00 / .062~ / .060~ | 1.00 / .055 / .085~ | 1.00 / .041 / .065~ | 1.00 / .100* / .120* |
| rcot2_native | raw_p | 1.00 / .084* / .105* | 1.00 / .080* / .070~ | 1.00 / .053 / .045 | 1.00 / .047 / .040 | 1.00 / .061~ / .065~ |
| rcot2_native | tau | 1.00 / .069* / .090* | 1.00 / .106* / .115* | 1.00 / .053 / .065~ | 1.00 / .092* / .105* | 1.00 / .095* / .125* |
| shap_dag | tau | 1.00 / .044 / .030 | 1.00 / .091* / .070~ | 1.00 / .048 / .035 | 1.00 / .103* / .100* | 1.00 / .056 / .035 |
| two_tower | tau | 1.00 / .072* / .045 | 1.00 / .053 / .025 | 1.00 / .058 / .020 | 1.00 / .061 / .030 | 1.00 / .125* / .050~ |

### E3 R2

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | raw_p | 1.00 / .034 / .045~ | 1.00 / .045 / .055~ | 1.00 / .048 / .060~ | 1.00 / .044 / .075~ | 1.00 / .058 / .075~ |
| pmrt_nl_eq | tau | 1.00 / .041 / .040 | 1.00 / .050 / .060~ | 1.00 / .045 / .055~ | 1.00 / .036 / .060~ | 1.00 / .097* / .150* |
| pmrt_eq | raw_p | .99 / .072~ / .065~ | 1.00 / .062~ / .055~ | 1.00 / .066~ / .060~ | 1.00 / .064~ / .080~ | 1.00 / .045 / .035 |
| pmrt_eq | tau | .99 / .073~ / .075~ | 1.00 / .048 / .050~ | 1.00 / .034 / .025 | 1.00 / .055 / .075~ | 1.00 / .039 / .040 |
| pmrt_r3 | raw_p | 1.00 / .045 / .055~ | 1.00 / .061~ / .065~ | 1.00 / .056 / .045 | 1.00 / .064~ / .075~ | 1.00 / .041 / .040 |
| pmrt_r3 | tau | 1.00 / .059~ / .050~ | 1.00 / .061~ / .070~ | 1.00 / .050 / .035 | 1.00 / .045 / .055~ | 1.00 / .069* / .050~ |
| cdl | - | - | - | - | nig | nig |
| cdl | tau | 1.00 / .111* / .055~ | 1.00 / .163* / .060~ | 1.00 / .205* / .020 | nig | nig |
| cdl | fixed | .00 / .000 / .000 | .00 / .000 / .000 | .26 / .000 / .000 | nig | nig |
| corr | raw_p | 1.00 / .691* / .570* | 1.00 / .784* / .685* | 1.00 / .903* / .830* | 1.00 / .920* / .875* | 1.00 / .939* / .925* |
| corr | tau | 1.00 / .336* / .120* | 1.00 / .339* / .145* | 1.00 / .325* / .120* | 1.00 / .347* / .135* | 1.00 / .330* / .115* |
| granger_eq | raw_p | 1.00 / .056~ / .075~ | 1.00 / .052 / .060~ | 1.00 / .059 / .070~ | 1.00 / .056 / .055~ | 1.00 / .036 / .035 |
| granger_eq | tau | 1.00 / .039 / .070~ | 1.00 / .036 / .040 | 1.00 / .050 / .065~ | 1.00 / .066~ / .090~ | 1.00 / .045 / .050~ |
| granger_native | raw_p | 1.00 / .439* / .280* | 1.00 / .558* / .395* | 1.00 / .744* / .665* | 1.00 / .798* / .730* | 1.00 / .892* / .820* |
| granger_native | tau | 1.00 / .331* / .115* | 1.00 / .336* / .115* | 1.00 / .358* / .130* | 1.00 / .342* / .115* | 1.00 / .352* / .125* |
| mscr_eq | raw_p | 1.00 / 1.000* / 1.000* | 1.00 / 1.000* / 1.000* | nig | nig | nig |
| mscr_eq | tau | 1.00 / .163* / .050~ | 1.00 / .161* / .035 | nig | nig | nig |
| mscr_native | raw_p | 1.00 / 1.000* / 1.000* | 1.00 / 1.000* / 1.000* | nig | nig | nig |
| mscr_native | tau | 1.00 / .255* / .075~ | 1.00 / .294* / .105~ | nig | nig | nig |
| notears | tau | 1.00 / .180* / .005 | 1.00 / .189* / .000 | 1.00 / .188* / .000 | 1.00 / .188* / .000 | 1.00 / .188* / .000 |
| pc_eq | tau | 1.00 / .031 / .050~ | 1.00 / .034 / .035 | 1.00 / .017 / .010 | 1.00 / .042 / .020 | 1.00 / .027 / .035 |
| pc_native | tau | 1.00 / .130* / .055~ | 1.00 / .114* / .030 | 1.00 / .128* / .010 | 1.00 / .119* / .010 | 1.00 / .158* / .025 |
| pcorr_eq | raw_p | 1.00 / .056~ / .075~ | 1.00 / .052 / .060~ | 1.00 / .059 / .070~ | 1.00 / .056 / .055~ | 1.00 / .036 / .035 |
| pcorr_eq | tau | 1.00 / .039 / .070~ | 1.00 / .036 / .040 | 1.00 / .050 / .065~ | 1.00 / .066~ / .090~ | 1.00 / .045 / .050~ |
| pcorr_eq_min | raw_p | 1.00 / .048 / .055~ | 1.00 / .052 / .050 | 1.00 / .058 / .085~ | 1.00 / .062~ / .070~ | 1.00 / .037 / .040 |
| pcorr_eq_min | tau | 1.00 / .048 / .055~ | 1.00 / .047 / .045 | 1.00 / .044 / .055~ | 1.00 / .039 / .045 | 1.00 / .078* / .060~ |
| pcorr_hac | raw_p | 1.00 / .219* / .050~ | 1.00 / .255* / .050~ | 1.00 / .370* / .100* | 1.00 / .442* / .135* | 1.00 / .483* / .295* |
| pcorr_hac | tau | 1.00 / .219* / .050~ | 1.00 / .275* / .075~ | 1.00 / .302* / .035 | 1.00 / .341* / .010 | 1.00 / .389* / .055~ |
| pcorr_hac_eq_min | raw_p | 1.00 / .047 / .060~ | 1.00 / .053 / .050 | 1.00 / .055 / .085~ | 1.00 / .062~ / .070~ | 1.00 / .037 / .040 |
| pcorr_hac_eq_min | tau | 1.00 / .047 / .060~ | 1.00 / .058~ / .055~ | 1.00 / .044 / .060~ | 1.00 / .042 / .050~ | 1.00 / .081* / .060~ |
| pcorr_hac_fb | raw_p | 1.00 / .217* / .045 | 1.00 / .255* / .050~ | 1.00 / .370* / .100* | 1.00 / .442* / .135* | 1.00 / .483* / .295* |
| pcorr_hac_fb | tau | 1.00 / .220* / .060~ | 1.00 / .275* / .075~ | 1.00 / .302* / .035 | 1.00 / .341* / .010 | 1.00 / .389* / .055~ |
| pcorr_hac_fb_eq_min | raw_p | 1.00 / .047 / .060~ | 1.00 / .052 / .050 | 1.00 / .055 / .085~ | 1.00 / .062~ / .070~ | 1.00 / .037 / .040 |
| pcorr_hac_fb_eq_min | tau | 1.00 / .047 / .060~ | 1.00 / .058~ / .055~ | 1.00 / .044 / .060~ | 1.00 / .042 / .050~ | 1.00 / .081* / .060~ |
| pcorr_native | raw_p | 1.00 / .216* / .040 | 1.00 / .255* / .050~ | 1.00 / .366* / .095* | 1.00 / .436* / .090~ | 1.00 / .480* / .280* |
| pcorr_native | tau | 1.00 / .230* / .060~ | 1.00 / .289* / .090~ | 1.00 / .309* / .030 | 1.00 / .348* / .010 | 1.00 / .403* / .080~ |
| rcot2_eq | raw_p | .99 / .111* / .085~ | .99 / .086* / .100* | 1.00 / .253* / .190* | 1.00 / .423* / .400* | 1.00 / .706* / .645* |
| rcot2_eq | tau | .91 / .045 / .060~ | .94 / .028 / .020 | 1.00 / .109* / .040 | 1.00 / .097* / .045 | 1.00 / .077* / .030 |
| rcot2_eq_min | raw_p | .99 / .109* / .065~ | 1.00 / .072* / .115* | 1.00 / .147* / .120* | 1.00 / .237* / .190* | 1.00 / .527* / .460* |
| rcot2_eq_min | tau | .86 / .028 / .030 | .94 / .027 / .030 | 1.00 / .031 / .030 | 1.00 / .055 / .040 | 1.00 / .037 / .015 |
| rcot2_native | raw_p | 1.00 / .159* / .105* | 1.00 / .184* / .100* | 1.00 / .248* / .240* | 1.00 / .372* / .395* | 1.00 / .647* / .670* |
| rcot2_native | tau | .97 / .023 / .045~ | .99 / .059~ / .070~ | 1.00 / .102* / .110* | 1.00 / .105* / .110* | 1.00 / .092* / .090~ |
| shap_dag | tau | 1.00 / .250* / .065~ | 1.00 / .242* / .070~ | 1.00 / .228* / .055~ | 1.00 / .230* / .025 | 1.00 / .230* / .040 |
| two_tower | tau | .99 / .222* / .030 | .99 / .256* / .060~ | .99 / .242* / .045~ | .99 / .264* / .060~ | .99 / .255* / .045~ |

### E4 R1 l1

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | raw_p | .17 / - / .025 | .28 / - / .050~ | .95 / - / .025 | 1.00 / - / .000 | 1.00 / - / .100~ |
| pmrt_nl_eq | tau | .03 / - / .000 | .07 / - / .025 | .70 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ |
| pmrt_eq | raw_p | .62 / - / .000 | .95 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .025 |
| pmrt_eq | tau | .42 / - / .000 | .68 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ |
| pmrt_r3 | raw_p | .62 / - / .000 | .95 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .025 |
| pmrt_r3 | tau | .40 / - / .000 | .65 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ |
| cdl | - | - | - | - | nig | nig |
| cdl | tau | .15 / - / .000 | .53 / - / .075~ | 1.00 / - / .025 | nig | nig |
| cdl | fixed | .00 / - / .000 | .00 / - / .000 | .00 / - / .000 | nig | nig |
| corr | raw_p | .65 / - / .000 | .90 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .025 |
| corr | tau | .38 / - / .000 | .65 / - / .000 | 1.00 / - / .000 | 1.00 / - / .025 | 1.00 / - / .025 |
| mscr_eq | raw_p | .17 / - / .050~ | .40 / - / .000 | nig | nig | nig |
| mscr_eq | tau | .17 / - / .025 | .15 / - / .000 | nig | nig | nig |
| mscr_native | raw_p | .17 / - / .150~ | .25 / - / .025 | nig | nig | nig |
| mscr_native | tau | .07 / - / .025 | .05 / - / .000 | nig | nig | nig |
| notears | tau | .38 / - / .000 | .57 / - / .000 | .80 / - / .000 | .93 / - / .000 | .97 / - / .000 |
| pc_eq | tau | .35 / - / .000 | .62 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .025 |
| pc_native | tau | .38 / - / .000 | .62 / - / .000 | 1.00 / - / .000 | 1.00 / - / .025 | 1.00 / - / .025 |
| pcorr_eq | raw_p | .65 / - / .025 | .90 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .025 |
| pcorr_eq | tau | .25 / - / .000 | .55 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ |
| pcorr_eq_min | raw_p | .68 / - / .000 | .90 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .025 |
| pcorr_eq_min | tau | .30 / - / .000 | .53 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .025 |
| pcorr_hac | raw_p | .68 / - / .000 | .90 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .025 |
| pcorr_hac | tau | .45 / - / .000 | .62 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ |
| pcorr_hac_eq_min | raw_p | .68 / - / .000 | .90 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .025 |
| pcorr_hac_eq_min | tau | .45 / - / .000 | .62 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ |
| pcorr_hac_fb | raw_p | .68 / - / .000 | .90 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .025 |
| pcorr_hac_fb | tau | .45 / - / .000 | .62 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ |
| pcorr_hac_fb_eq_min | raw_p | .68 / - / .000 | .90 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .025 |
| pcorr_hac_fb_eq_min | tau | .45 / - / .000 | .62 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ |
| pcorr_native | raw_p | .68 / - / .000 | .90 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .025 |
| pcorr_native | tau | .30 / - / .000 | .53 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .025 |
| rcot2_eq | raw_p | .40 / - / .150~ | .68 / - / .100~ | 1.00 / - / .100~ | 1.00 / - / .025 | 1.00 / - / .025 |
| rcot2_eq | tau | .15 / - / .050~ | .38 / - / .075~ | .88 / - / .000 | 1.00 / - / .025 | 1.00 / - / .200* |
| rcot2_eq_min | raw_p | .33 / - / .050~ | .68 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .025 | 1.00 / - / .025 |
| rcot2_eq_min | tau | .12 / - / .025 | .40 / - / .025 | .95 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .225* |
| rcot2_native | raw_p | .33 / - / .050~ | .68 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .000 | 1.00 / - / .025 |
| rcot2_native | tau | .10 / - / .025 | .42 / - / .025 | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .225* |
| shap_dag | tau | .30 / - / .050~ | .23 / - / .025 | .97 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .000 |
| two_tower | tau | .82 / - / .075~ | .65 / - / .025 | 1.00 / - / .000 | 1.00 / - / .025 | 1.00 / - / .075~ |

### E4 R2 l1

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | raw_p | .05 / - / .025 | .03 / - / .075~ | .07 / - / .075~ | .12 / - / .050~ | .12 / - / .025 |
| pmrt_nl_eq | tau | .00 / - / .000 | .10 / - / .150~ | .10 / - / .150~ | .17 / - / .125~ | .17 / - / .050~ |
| pmrt_eq | raw_p | .03 / - / .075~ | .05 / - / .075~ | .30 / - / .125~ | .45 / - / .100~ | .93 / - / .025 |
| pmrt_eq | tau | .00 / - / .000 | .05 / - / .075~ | .55 / - / .275* | .40 / - / .100~ | .97 / - / .175* |
| pmrt_r3 | raw_p | .03 / - / .075~ | .05 / - / .075~ | .30 / - / .125~ | .42 / - / .100~ | .93 / - / .025 |
| pmrt_r3 | tau | .00 / - / .000 | .03 / - / .075~ | .55 / - / .250* | .40 / - / .100~ | .95 / - / .175* |
| cdl | - | - | - | - | nig | nig |
| cdl | tau | .17 / - / .075~ | .17 / - / .025 | .60 / - / .000 | nig | nig |
| cdl | fixed | .00 / - / .000 | .00 / - / .000 | .00 / - / .000 | nig | nig |
| corr | raw_p | .15 / - / .075~ | .42 / - / .150~ | .93 / - / .100~ | 1.00 / - / .325* | 1.00 / - / .400* |
| corr | tau | .20 / - / .125~ | .47 / - / .150~ | .70 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .000 |
| mscr_eq | raw_p | .05 / - / .000 | .07 / - / .050~ | nig | nig | nig |
| mscr_eq | tau | .03 / - / .000 | .07 / - / .075~ | nig | nig | nig |
| mscr_native | raw_p | .07 / - / .025 | .10 / - / .000 | nig | nig | nig |
| mscr_native | tau | .05 / - / .000 | .05 / - / .000 | nig | nig | nig |
| notears | tau | .15 / - / .050~ | .12 / - / .000 | .03 / - / .000 | .00 / - / .000 | .00 / - / .000 |
| pc_eq | tau | .05 / - / .075~ | .07 / - / .000 | .35 / - / .025 | .60 / - / .225* | 1.00 / - / .200* |
| pc_native | tau | .20 / - / .125~ | .47 / - / .100~ | .95 / - / .050~ | 1.00 / - / .250* | 1.00 / - / .175* |
| pcorr_eq | raw_p | .03 / - / .075~ | .07 / - / .100~ | .30 / - / .150~ | .42 / - / .100~ | .93 / - / .025 |
| pcorr_eq | tau | .00 / - / .075~ | .07 / - / .050~ | .57 / - / .275* | .42 / - / .100~ | 1.00 / - / .175* |
| pcorr_eq_min | raw_p | .03 / - / .100~ | .07 / - / .100~ | .33 / - / .150~ | .42 / - / .100~ | .93 / - / .025 |
| pcorr_eq_min | tau | .00 / - / .075~ | .07 / - / .100~ | .57 / - / .275* | .42 / - / .100~ | 1.00 / - / .175* |
| pcorr_hac | raw_p | .15 / - / .100~ | .45 / - / .075~ | .93 / - / .025 | 1.00 / - / .100~ | 1.00 / - / .050~ |
| pcorr_hac | tau | .17 / - / .125~ | .53 / - / .125~ | 1.00 / - / .125~ | 1.00 / - / .125~ | 1.00 / - / .050~ |
| pcorr_hac_eq_min | raw_p | .00 / - / .100~ | .07 / - / .100~ | .30 / - / .150~ | .42 / - / .100~ | .93 / - / .025 |
| pcorr_hac_eq_min | tau | .00 / - / .050~ | .07 / - / .075~ | .60 / - / .275* | .42 / - / .100~ | 1.00 / - / .150~ |
| pcorr_hac_fb | raw_p | .15 / - / .100~ | .45 / - / .075~ | .93 / - / .025 | 1.00 / - / .100~ | 1.00 / - / .050~ |
| pcorr_hac_fb | tau | .17 / - / .125~ | .53 / - / .125~ | 1.00 / - / .125~ | 1.00 / - / .125~ | 1.00 / - / .050~ |
| pcorr_hac_fb_eq_min | raw_p | .00 / - / .100~ | .07 / - / .100~ | .30 / - / .150~ | .42 / - / .100~ | .93 / - / .025 |
| pcorr_hac_fb_eq_min | tau | .00 / - / .050~ | .07 / - / .075~ | .60 / - / .275* | .42 / - / .100~ | 1.00 / - / .150~ |
| pcorr_native | raw_p | .15 / - / .075~ | .45 / - / .075~ | .93 / - / .025 | 1.00 / - / .100~ | 1.00 / - / .050~ |
| pcorr_native | tau | .17 / - / .125~ | .53 / - / .125~ | .97 / - / .100~ | 1.00 / - / .125~ | 1.00 / - / .050~ |
| rcot2_eq | raw_p | .10 / - / .100~ | .05 / - / .100~ | .05 / - / .075~ | .10 / - / .025 | .20 / - / .075~ |
| rcot2_eq | tau | .12 / - / .025 | .05 / - / .075~ | .12 / - / .175* | .05 / - / .000 | .12 / - / .050~ |
| rcot2_eq_min | raw_p | .05 / - / .075~ | .05 / - / .025 | .05 / - / .075~ | .07 / - / .025 | .25 / - / .075~ |
| rcot2_eq_min | tau | .07 / - / .025 | .07 / - / .075~ | .05 / - / .050~ | .07 / - / .000 | .10 / - / .050~ |
| rcot2_native | raw_p | .17 / - / .025 | .28 / - / .075~ | .42 / - / .025 | .68 / - / .025 | .88 / - / .050~ |
| rcot2_native | tau | .30 / - / .075~ | .25 / - / .075~ | .53 / - / .150~ | .72 / - / .100~ | .82 / - / .025 |
| shap_dag | tau | .07 / - / .100~ | .03 / - / .050~ | .15 / - / .025 | .42 / - / .025 | .75 / - / .000 |
| two_tower | tau | .10 / - / .100~ | .03 / - / .075~ | .93 / - / .150~ | 1.00 / - / .025 | 1.00 / - / .000 |

### E4 R3 l0

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | raw_p | 1.00 / - / .050~ | 1.00 / - / .070~ | 1.00 / - / .057~ | 1.00 / - / .060~ | 1.00 / - / .053~ |
| pmrt_nl_eq | tau | 1.00 / - / .017 | 1.00 / - / .090* | 1.00 / - / .033 | 1.00 / - / .070~ | 1.00 / - / .030 |
| pmrt_eq | raw_p | 1.00 / - / .037 | 1.00 / - / .033 | 1.00 / - / .043 | 1.00 / - / .047 | 1.00 / - / .067~ |
| pmrt_eq | tau | 1.00 / - / .043 | 1.00 / - / .000 | 1.00 / - / .003 | 1.00 / - / .013 | 1.00 / - / .010 |
| pmrt_r3 | raw_p | 1.00 / - / .033 | 1.00 / - / .033 | 1.00 / - / .037 | 1.00 / - / .047 | 1.00 / - / .063~ |
| pmrt_r3 | tau | 1.00 / - / .077~ | 1.00 / - / .000 | 1.00 / - / .003 | 1.00 / - / .020 | 1.00 / - / .007 |
| cdl | - | nig | nig | nig | nig | nig |
| corr | raw_p | .85 / - / .075~ | 1.00 / - / .075~ | 1.00 / - / .050~ | 1.00 / - / .025 | 1.00 / - / .000 |
| corr | tau | .68 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .025 | 1.00 / - / .050~ |
| mscr_eq | raw_p | .98 / - / .060~ | 1.00 / - / .057~ | nig | nig | nig |
| mscr_eq | tau | .99 / - / .127* | 1.00 / - / .043 | nig | nig | nig |
| mscr_native | raw_p | 1.00 / - / .000 | 1.00 / - / .075~ | nig | nig | nig |
| mscr_native | tau | .97 / - / .000 | 1.00 / - / .150~ | nig | nig | nig |
| notears | tau | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pc_eq | tau | .90 / - / .030 | 1.00 / - / .020 | 1.00 / - / .003 | 1.00 / - / .017 | 1.00 / - / .063~ |
| pc_native | tau | .88 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .050~ |
| pcorr_eq | raw_p | 1.00 / - / .050~ | 1.00 / - / .047 | 1.00 / - / .040 | 1.00 / - / .043 | 1.00 / - / .067~ |
| pcorr_eq | tau | 1.00 / - / .077~ | 1.00 / - / .010 | 1.00 / - / .003 | 1.00 / - / .040 | 1.00 / - / .010 |
| pcorr_eq_min | raw_p | 1.00 / - / .050~ | 1.00 / - / .075~ | 1.00 / - / .025 | 1.00 / - / .050~ | 1.00 / - / .050~ |
| pcorr_eq_min | tau | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .025 |
| pcorr_hac | raw_p | 1.00 / - / .050~ | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .050~ |
| pcorr_hac | tau | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .025 |
| pcorr_hac_eq_min | raw_p | 1.00 / - / .050~ | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .050~ |
| pcorr_hac_eq_min | tau | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .025 |
| pcorr_hac_fb | raw_p | 1.00 / - / .025 | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .050~ |
| pcorr_hac_fb | tau | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .025 |
| pcorr_hac_fb_eq_min | raw_p | 1.00 / - / .025 | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .050~ |
| pcorr_hac_fb_eq_min | tau | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .025 |
| pcorr_native | raw_p | 1.00 / - / .050~ | 1.00 / - / .075~ | 1.00 / - / .025 | 1.00 / - / .050~ | 1.00 / - / .050~ |
| pcorr_native | tau | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .025 |
| rcot2_eq | raw_p | .91 / - / .073~ | .96 / - / .047 | .99 / - / .043 | .99 / - / .037 | 1.00 / - / .047 |
| rcot2_eq | tau | .85 / - / .063~ | .83 / - / .013 | .99 / - / .067~ | .99 / - / .007 | .99 / - / .003 |
| rcot2_eq_min | raw_p | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .025 | 1.00 / - / .050~ | 1.00 / - / .075~ |
| rcot2_eq_min | tau | .95 / - / .025 | .90 / - / .000 | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .100~ |
| rcot2_native | raw_p | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .025 | 1.00 / - / .025 | 1.00 / - / .025 |
| rcot2_native | tau | .95 / - / .000 | .88 / - / .000 | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .050~ |
| shap_dag | tau | .65 / - / .150~ | .90 / - / .125~ | 1.00 / - / .025 | 1.00 / - / .050~ | 1.00 / - / .050~ |
| two_tower | tau | .90 / - / .075~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .075~ |

### E4 R3 l0.5

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | raw_p | 1.00 / - / .033 | 1.00 / - / .033 | 1.00 / - / .050~ | 1.00 / - / .053~ | 1.00 / - / .050~ |
| pmrt_nl_eq | tau | 1.00 / - / .133* | 1.00 / - / .010 | 1.00 / - / .007 | 1.00 / - / .003 | 1.00 / - / .003 |
| pmrt_eq | raw_p | 1.00 / - / .033 | 1.00 / - / .053~ | 1.00 / - / .040 | 1.00 / - / .047 | 1.00 / - / .067~ |
| pmrt_eq | tau | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .003 | 1.00 / - / .017 | 1.00 / - / .010 |
| pmrt_r3 | raw_p | 1.00 / - / .037 | 1.00 / - / .053~ | 1.00 / - / .040 | 1.00 / - / .053~ | 1.00 / - / .067~ |
| pmrt_r3 | tau | 1.00 / - / .063~ | 1.00 / - / .000 | 1.00 / - / .003 | 1.00 / - / .017 | 1.00 / - / .007 |
| cdl | - | nig | nig | nig | nig | nig |
| corr | raw_p | 1.00 / - / .075~ | 1.00 / - / .075~ | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .025 |
| corr | tau | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .025 | 1.00 / - / .050~ |
| mscr_eq | raw_p | 1.00 / - / .050~ | 1.00 / - / .063~ | nig | nig | nig |
| mscr_eq | tau | 1.00 / - / .063~ | 1.00 / - / .053~ | nig | nig | nig |
| mscr_native | raw_p | 1.00 / - / .100~ | 1.00 / - / .075~ | nig | nig | nig |
| mscr_native | tau | 1.00 / - / .025 | 1.00 / - / .075~ | nig | nig | nig |
| notears | tau | .00 / - / .000 | .00 / - / .000 | .00 / - / .000 | .00 / - / .000 | .00 / - / .000 |
| pc_eq | tau | 1.00 / - / .023 | 1.00 / - / .007 | 1.00 / - / .027 | 1.00 / - / .007 | 1.00 / - / .053~ |
| pc_native | tau | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .025 |
| pcorr_eq | raw_p | 1.00 / - / .043 | 1.00 / - / .053~ | 1.00 / - / .040 | 1.00 / - / .057~ | 1.00 / - / .077~ |
| pcorr_eq | tau | 1.00 / - / .043 | 1.00 / - / .013 | 1.00 / - / .003 | 1.00 / - / .033 | 1.00 / - / .010 |
| pcorr_eq_min | raw_p | 1.00 / - / .025 | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .050~ |
| pcorr_eq_min | tau | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .025 |
| pcorr_hac | raw_p | 1.00 / - / .025 | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .050~ |
| pcorr_hac | tau | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .025 |
| pcorr_hac_eq_min | raw_p | 1.00 / - / .025 | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .050~ |
| pcorr_hac_eq_min | tau | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .025 |
| pcorr_hac_fb | raw_p | 1.00 / - / .025 | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .050~ |
| pcorr_hac_fb | tau | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .025 |
| pcorr_hac_fb_eq_min | raw_p | 1.00 / - / .025 | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .050~ |
| pcorr_hac_fb_eq_min | tau | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .025 |
| pcorr_native | raw_p | 1.00 / - / .025 | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .050~ |
| pcorr_native | tau | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .025 |
| rcot2_eq | raw_p | .93 / - / .073~ | .96 / - / .060~ | .99 / - / .050~ | 1.00 / - / .043 | 1.00 / - / .060~ |
| rcot2_eq | tau | .71 / - / .053~ | .81 / - / .033 | .99 / - / .150* | .99 / - / .013 | .99 / - / .003 |
| rcot2_eq_min | raw_p | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .025 | 1.00 / - / .050~ |
| rcot2_eq_min | tau | .93 / - / .150~ | .93 / - / .025 | 1.00 / - / .100~ | 1.00 / - / .025 | 1.00 / - / .075~ |
| rcot2_native | raw_p | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .050~ |
| rcot2_native | tau | .80 / - / .025 | .93 / - / .025 | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .050~ |
| shap_dag | tau | 1.00 / - / .175* | 1.00 / - / .000 | 1.00 / - / .200* | 1.00 / - / .050~ | 1.00 / - / .050~ |
| two_tower | tau | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .125~ | 1.00 / - / .050~ | 1.00 / - / .000 |

### E4 R3 l1

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | raw_p | 1.00 / - / .030 | 1.00 / - / .050~ | 1.00 / - / .037 | 1.00 / - / .023 | 1.00 / - / .053~ |
| pmrt_nl_eq | tau | 1.00 / - / .087* | 1.00 / - / .033 | 1.00 / - / .020 | 1.00 / - / .003 | 1.00 / - / .093* |
| pmrt_eq | raw_p | 1.00 / - / .037 | 1.00 / - / .057~ | 1.00 / - / .037 | 1.00 / - / .067~ | 1.00 / - / .083* |
| pmrt_eq | tau | 1.00 / - / .030 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .010 | 1.00 / - / .003 |
| pmrt_r3 | raw_p | 1.00 / - / .030 | 1.00 / - / .053~ | 1.00 / - / .030 | 1.00 / - / .067~ | 1.00 / - / .083* |
| pmrt_r3 | tau | 1.00 / - / .030 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .010 | 1.00 / - / .003 |
| cdl | - | nig | nig | nig | nig | nig |
| corr | raw_p | 1.00 / - / .050~ | 1.00 / - / .075~ | 1.00 / - / .050~ | 1.00 / - / .025 | 1.00 / - / .025 |
| corr | tau | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .025 | 1.00 / - / .025 | 1.00 / - / .050~ |
| mscr_eq | raw_p | 1.00 / - / .030 | 1.00 / - / .033 | nig | nig | nig |
| mscr_eq | tau | 1.00 / - / .003 | 1.00 / - / .020 | nig | nig | nig |
| mscr_native | raw_p | 1.00 / - / .050~ | 1.00 / - / .050~ | nig | nig | nig |
| mscr_native | tau | 1.00 / - / .000 | 1.00 / - / .125~ | nig | nig | nig |
| notears | tau | .00 / - / .000 | .00 / - / .000 | .00 / - / .000 | .00 / - / .000 | .00 / - / .000 |
| pc_eq | tau | 1.00 / - / .053~ | 1.00 / - / .010 | 1.00 / - / .060~ | 1.00 / - / .007 | 1.00 / - / .043 |
| pc_native | tau | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .025 |
| pcorr_eq | raw_p | 1.00 / - / .050~ | 1.00 / - / .063~ | 1.00 / - / .040 | 1.00 / - / .057~ | 1.00 / - / .083* |
| pcorr_eq | tau | 1.00 / - / .040 | 1.00 / - / .010 | 1.00 / - / .000 | 1.00 / - / .020 | 1.00 / - / .007 |
| pcorr_eq_min | raw_p | 1.00 / - / .025 | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .025 |
| pcorr_eq_min | tau | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac | raw_p | 1.00 / - / .025 | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .025 | 1.00 / - / .025 |
| pcorr_hac | tau | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_eq_min | raw_p | 1.00 / - / .025 | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .025 | 1.00 / - / .025 |
| pcorr_hac_eq_min | tau | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_fb | raw_p | 1.00 / - / .025 | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .025 | 1.00 / - / .025 |
| pcorr_hac_fb | tau | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_fb_eq_min | raw_p | 1.00 / - / .025 | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .025 | 1.00 / - / .025 |
| pcorr_hac_fb_eq_min | tau | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_native | raw_p | 1.00 / - / .025 | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .025 |
| pcorr_native | tau | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| rcot2_eq | raw_p | .87 / - / .097* | .96 / - / .043 | 1.00 / - / .037 | 1.00 / - / .050~ | 1.00 / - / .050~ |
| rcot2_eq | tau | .45 / - / .060~ | .56 / - / .013 | .98 / - / .097* | .97 / - / .020 | .99 / - / .003 |
| rcot2_eq_min | raw_p | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .075~ | 1.00 / - / .025 | 1.00 / - / .025 |
| rcot2_eq_min | tau | .75 / - / .100~ | .72 / - / .000 | 1.00 / - / .100~ | 1.00 / - / .025 | 1.00 / - / .075~ |
| rcot2_native | raw_p | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .100~ | 1.00 / - / .025 | 1.00 / - / .025 |
| rcot2_native | tau | .72 / - / .050~ | .72 / - / .000 | 1.00 / - / .075~ | 1.00 / - / .050~ | 1.00 / - / .075~ |
| shap_dag | tau | 1.00 / - / .000 | 1.00 / - / .075~ | 1.00 / - / .175* | 1.00 / - / .100~ | 1.00 / - / .075~ |
| two_tower | tau | 1.00 / - / .100~ | 1.00 / - / .025 | 1.00 / - / .175* | 1.00 / - / .000 | 1.00 / - / .075~ |

### E4 R3 l1.5

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | raw_p | .99 / - / .057~ | 1.00 / - / .043 | 1.00 / - / .063~ | 1.00 / - / .043 | 1.00 / - / .057~ |
| pmrt_nl_eq | tau | .99 / - / .037 | 1.00 / - / .037 | 1.00 / - / .060~ | 1.00 / - / .037 | 1.00 / - / .050~ |
| pmrt_eq | raw_p | 1.00 / - / .033 | 1.00 / - / .067~ | 1.00 / - / .033 | 1.00 / - / .070~ | 1.00 / - / .080~ |
| pmrt_eq | tau | 1.00 / - / .033 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .017 | 1.00 / - / .000 |
| pmrt_r3 | raw_p | 1.00 / - / .033 | 1.00 / - / .060~ | 1.00 / - / .027 | 1.00 / - / .067~ | 1.00 / - / .077~ |
| pmrt_r3 | tau | 1.00 / - / .027 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .017 | 1.00 / - / .000 |
| cdl | - | nig | nig | nig | nig | nig |
| corr | raw_p | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .025 | 1.00 / - / .025 |
| corr | tau | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .100~ |
| mscr_eq | raw_p | 1.00 / - / .090* | 1.00 / - / .480* | nig | nig | nig |
| mscr_eq | tau | 1.00 / - / .093* | 1.00 / - / .003 | nig | nig | nig |
| mscr_native | raw_p | 1.00 / - / .150~ | 1.00 / - / .675* | nig | nig | nig |
| mscr_native | tau | 1.00 / - / .050~ | 1.00 / - / .000 | nig | nig | nig |
| notears | tau | .00 / - / .000 | .00 / - / .000 | .00 / - / .000 | .00 / - / .000 | .00 / - / .000 |
| pc_eq | tau | 1.00 / - / .040 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .070~ |
| pc_native | tau | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .075~ |
| pcorr_eq | raw_p | 1.00 / - / .050~ | 1.00 / - / .070~ | 1.00 / - / .040 | 1.00 / - / .067~ | 1.00 / - / .080~ |
| pcorr_eq | tau | 1.00 / - / .043 | 1.00 / - / .007 | 1.00 / - / .000 | 1.00 / - / .033 | 1.00 / - / .007 |
| pcorr_eq_min | raw_p | 1.00 / - / .025 | 1.00 / - / .100~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ |
| pcorr_eq_min | tau | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac | raw_p | 1.00 / - / .025 | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ |
| pcorr_hac | tau | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_eq_min | raw_p | 1.00 / - / .025 | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ |
| pcorr_hac_eq_min | tau | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_fb | raw_p | 1.00 / - / .025 | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ |
| pcorr_hac_fb | tau | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_fb_eq_min | raw_p | 1.00 / - / .025 | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ |
| pcorr_hac_fb_eq_min | tau | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_native | raw_p | 1.00 / - / .025 | 1.00 / - / .100~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ |
| pcorr_native | tau | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| rcot2_eq | raw_p | .90 / - / .083* | .99 / - / .047 | 1.00 / - / .030 | 1.00 / - / .060~ | 1.00 / - / .047 |
| rcot2_eq | tau | .20 / - / .027 | .37 / - / .010 | .98 / - / .127* | .94 / - / .013 | .99 / - / .013 |
| rcot2_eq_min | raw_p | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .075~ | 1.00 / - / .025 | 1.00 / - / .075~ |
| rcot2_eq_min | tau | .65 / - / .075~ | .60 / - / .000 | 1.00 / - / .100~ | 1.00 / - / .025 | 1.00 / - / .100~ |
| rcot2_native | raw_p | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .075~ | 1.00 / - / .050~ | 1.00 / - / .050~ |
| rcot2_native | tau | .50 / - / .075~ | .65 / - / .000 | 1.00 / - / .075~ | 1.00 / - / .025 | 1.00 / - / .100~ |
| shap_dag | tau | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .125~ | 1.00 / - / .050~ | 1.00 / - / .075~ |
| two_tower | tau | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .175* | 1.00 / - / .025 | 1.00 / - / .175* |

### E4 R4 l0

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | raw_p | NA / - / .050~ | NA / - / .025 | NA / - / .050~ | NA / - / .075~ | NA / - / .050~ |
| pmrt_nl_eq | tau | NA / - / .100~ | NA / - / .225* | NA / - / .100~ | NA / - / .050~ | NA / - / .275* |
| pmrt_eq | raw_p | NA / - / .025 | NA / - / .050~ | NA / - / .150~ | NA / - / .125~ | NA / - / .050~ |
| pmrt_eq | tau | NA / - / .025 | NA / - / .050~ | NA / - / .000 | NA / - / .000 | NA / - / .050~ |
| pmrt_r3 | raw_p | NA / - / .025 | NA / - / .050~ | NA / - / .150~ | NA / - / .125~ | NA / - / .050~ |
| pmrt_r3 | tau | NA / - / .025 | NA / - / .050~ | NA / - / .000 | NA / - / .000 | NA / - / .050~ |
| cdl | - | nig | nig | nig | nig | nig |
| corr | raw_p | .90 / - / .050~ | 1.00 / - / .025 | 1.00 / - / .125~ | 1.00 / - / .125~ | 1.00 / - / .050~ |
| corr | tau | .95 / - / .050~ | 1.00 / - / .075~ | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .050~ |
| mscr_eq | raw_p | .25 / - / .050~ | .55 / - / .000 | nig | nig | nig |
| mscr_eq | tau | .25 / - / .025 | .65 / - / .000 | nig | nig | nig |
| mscr_native | raw_p | .25 / - / .025 | .53 / - / .050~ | nig | nig | nig |
| mscr_native | tau | .25 / - / .025 | .55 / - / .100~ | nig | nig | nig |
| notears | tau | .82 / - / .025 | .82 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pc_eq | tau | .93 / - / .050~ | 1.00 / - / .075~ | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .050~ |
| pc_native | tau | .95 / - / .050~ | 1.00 / - / .075~ | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .050~ |
| pcorr_eq | raw_p | .88 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .125~ | 1.00 / - / .050~ |
| pcorr_eq | tau | .88 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .050~ |
| pcorr_eq_min | raw_p | .90 / - / .050~ | 1.00 / - / .025 | 1.00 / - / .100~ | 1.00 / - / .125~ | 1.00 / - / .050~ |
| pcorr_eq_min | tau | .95 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .050~ |
| pcorr_hac | raw_p | .88 / - / .050~ | 1.00 / - / .025 | 1.00 / - / .100~ | 1.00 / - / .125~ | 1.00 / - / .050~ |
| pcorr_hac | tau | .95 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .050~ |
| pcorr_hac_eq_min | raw_p | .88 / - / .050~ | 1.00 / - / .025 | 1.00 / - / .100~ | 1.00 / - / .125~ | 1.00 / - / .050~ |
| pcorr_hac_eq_min | tau | .95 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .050~ |
| pcorr_hac_fb | raw_p | .88 / - / .050~ | 1.00 / - / .025 | 1.00 / - / .100~ | 1.00 / - / .125~ | 1.00 / - / .050~ |
| pcorr_hac_fb | tau | .95 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .050~ |
| pcorr_hac_fb_eq_min | raw_p | .88 / - / .050~ | 1.00 / - / .025 | 1.00 / - / .100~ | 1.00 / - / .125~ | 1.00 / - / .050~ |
| pcorr_hac_fb_eq_min | tau | .95 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .050~ |
| pcorr_native | raw_p | .90 / - / .050~ | 1.00 / - / .025 | 1.00 / - / .100~ | 1.00 / - / .125~ | 1.00 / - / .050~ |
| pcorr_native | tau | .95 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .050~ |
| rcot2_eq | raw_p | .62 / - / .100~ | .80 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .125~ | 1.00 / - / .050~ |
| rcot2_eq | tau | .42 / - / .050~ | .55 / - / .050~ | .97 / - / .075~ | 1.00 / - / .100~ | 1.00 / - / .000 |
| rcot2_eq_min | raw_p | .55 / - / .025 | .85 / - / .000 | 1.00 / - / .150~ | 1.00 / - / .150~ | 1.00 / - / .075~ |
| rcot2_eq_min | tau | .50 / - / .050~ | .53 / - / .025 | 1.00 / - / .100~ | 1.00 / - / .100~ | 1.00 / - / .000 |
| rcot2_native | raw_p | .55 / - / .050~ | .88 / - / .000 | 1.00 / - / .150~ | 1.00 / - / .150~ | 1.00 / - / .075~ |
| rcot2_native | tau | .50 / - / .050~ | .55 / - / .025 | 1.00 / - / .100~ | 1.00 / - / .075~ | 1.00 / - / .000 |
| shap_dag | tau | .45 / - / .000 | .88 / - / .075~ | 1.00 / - / .100~ | 1.00 / - / .000 | 1.00 / - / .025 |
| two_tower | tau | .07 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ |

### E4 R4 l0.5

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | raw_p | NA / - / .000 | NA / - / .050~ | NA / - / .050~ | NA / - / .025 | NA / - / .075~ |
| pmrt_nl_eq | tau | NA / - / .000 | NA / - / .000 | NA / - / .025 | NA / - / .000 | NA / - / .025 |
| pmrt_eq | raw_p | NA / - / .050~ | NA / - / .050~ | NA / - / .125~ | NA / - / .125~ | NA / - / .075~ |
| pmrt_eq | tau | NA / - / .050~ | NA / - / .050~ | NA / - / .000 | NA / - / .000 | NA / - / .050~ |
| pmrt_r3 | raw_p | NA / - / .025 | NA / - / .050~ | NA / - / .100~ | NA / - / .150~ | NA / - / .075~ |
| pmrt_r3 | tau | NA / - / .025 | NA / - / .050~ | NA / - / .000 | NA / - / .000 | NA / - / .050~ |
| cdl | - | nig | nig | nig | nig | nig |
| corr | raw_p | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .125~ | 1.00 / - / .125~ | 1.00 / - / .050~ |
| corr | tau | 1.00 / - / .075~ | 1.00 / - / .075~ | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .050~ |
| mscr_eq | raw_p | 1.00 / - / .000 | 1.00 / - / .025 | nig | nig | nig |
| mscr_eq | tau | 1.00 / - / .000 | 1.00 / - / .025 | nig | nig | nig |
| mscr_native | raw_p | 1.00 / - / .050~ | 1.00 / - / .075~ | nig | nig | nig |
| mscr_native | tau | 1.00 / - / .025 | 1.00 / - / .000 | nig | nig | nig |
| notears | tau | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pc_eq | tau | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .100~ |
| pc_native | tau | 1.00 / - / .050~ | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .100~ |
| pcorr_eq | raw_p | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .125~ | 1.00 / - / .050~ |
| pcorr_eq | tau | 1.00 / - / .075~ | 1.00 / - / .075~ | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .050~ |
| pcorr_eq_min | raw_p | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .075~ | 1.00 / - / .100~ | 1.00 / - / .075~ |
| pcorr_eq_min | tau | 1.00 / - / .050~ | 1.00 / - / .075~ | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .075~ |
| pcorr_hac | raw_p | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .075~ | 1.00 / - / .100~ | 1.00 / - / .050~ |
| pcorr_hac | tau | 1.00 / - / .025 | 1.00 / - / .075~ | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .075~ |
| pcorr_hac_eq_min | raw_p | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .075~ | 1.00 / - / .100~ | 1.00 / - / .050~ |
| pcorr_hac_eq_min | tau | 1.00 / - / .025 | 1.00 / - / .075~ | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .075~ |
| pcorr_hac_fb | raw_p | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .075~ | 1.00 / - / .100~ | 1.00 / - / .050~ |
| pcorr_hac_fb | tau | 1.00 / - / .025 | 1.00 / - / .075~ | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .075~ |
| pcorr_hac_fb_eq_min | raw_p | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .075~ | 1.00 / - / .100~ | 1.00 / - / .050~ |
| pcorr_hac_fb_eq_min | tau | 1.00 / - / .025 | 1.00 / - / .075~ | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .075~ |
| pcorr_native | raw_p | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .075~ | 1.00 / - / .100~ | 1.00 / - / .075~ |
| pcorr_native | tau | 1.00 / - / .050~ | 1.00 / - / .075~ | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .075~ |
| rcot2_eq | raw_p | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .100~ | 1.00 / - / .150~ | 1.00 / - / .075~ |
| rcot2_eq | tau | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .075~ | 1.00 / - / .100~ | 1.00 / - / .025 |
| rcot2_eq_min | raw_p | 1.00 / - / .025 | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .025 |
| rcot2_eq_min | tau | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .000 |
| rcot2_native | raw_p | 1.00 / - / .025 | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .025 |
| rcot2_native | tau | 1.00 / - / .025 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .000 |
| shap_dag | tau | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .100~ |
| two_tower | tau | 1.00 / - / .075~ | 1.00 / - / .100~ | 1.00 / - / .025 | 1.00 / - / .025 | 1.00 / - / .025 |

### E4 R4 l1

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | raw_p | NA / - / .025 | NA / - / .025 | NA / - / .050~ | NA / - / .050~ | NA / - / .075~ |
| pmrt_nl_eq | tau | NA / - / .025 | NA / - / .025 | NA / - / .050~ | NA / - / .025 | NA / - / .050~ |
| pmrt_eq | raw_p | NA / - / .025 | NA / - / .050~ | NA / - / .100~ | NA / - / .075~ | NA / - / .050~ |
| pmrt_eq | tau | NA / - / .050~ | NA / - / .050~ | NA / - / .025 | NA / - / .000 | NA / - / .025 |
| pmrt_r3 | raw_p | NA / - / .025 | NA / - / .050~ | NA / - / .100~ | NA / - / .075~ | NA / - / .050~ |
| pmrt_r3 | tau | NA / - / .050~ | NA / - / .050~ | NA / - / .025 | NA / - / .000 | NA / - / .025 |
| cdl | - | nig | nig | nig | nig | nig |
| corr | raw_p | 1.00 / - / .025 | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .050~ |
| corr | tau | 1.00 / - / .075~ | 1.00 / - / .075~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .025 |
| mscr_eq | raw_p | 1.00 / - / .025 | 1.00 / - / .025 | nig | nig | nig |
| mscr_eq | tau | 1.00 / - / .050~ | 1.00 / - / .025 | nig | nig | nig |
| mscr_native | raw_p | 1.00 / - / .000 | 1.00 / - / .025 | nig | nig | nig |
| mscr_native | tau | 1.00 / - / .025 | 1.00 / - / .025 | nig | nig | nig |
| notears | tau | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pc_eq | tau | 1.00 / - / .075~ | 1.00 / - / .050~ | 1.00 / - / .075~ | 1.00 / - / .025 | 1.00 / - / .025 |
| pc_native | tau | 1.00 / - / .075~ | 1.00 / - / .050~ | 1.00 / - / .075~ | 1.00 / - / .025 | 1.00 / - / .025 |
| pcorr_eq | raw_p | 1.00 / - / .025 | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .050~ |
| pcorr_eq | tau | 1.00 / - / .075~ | 1.00 / - / .075~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .025 |
| pcorr_eq_min | raw_p | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .050~ |
| pcorr_eq_min | tau | 1.00 / - / .025 | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .050~ |
| pcorr_hac | raw_p | 1.00 / - / .050~ | 1.00 / - / .025 | 1.00 / - / .050~ | 1.00 / - / .125~ | 1.00 / - / .050~ |
| pcorr_hac | tau | 1.00 / - / .025 | 1.00 / - / .125~ | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .050~ |
| pcorr_hac_eq_min | raw_p | 1.00 / - / .050~ | 1.00 / - / .025 | 1.00 / - / .050~ | 1.00 / - / .125~ | 1.00 / - / .050~ |
| pcorr_hac_eq_min | tau | 1.00 / - / .025 | 1.00 / - / .125~ | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .050~ |
| pcorr_hac_fb | raw_p | 1.00 / - / .050~ | 1.00 / - / .025 | 1.00 / - / .050~ | 1.00 / - / .125~ | 1.00 / - / .050~ |
| pcorr_hac_fb | tau | 1.00 / - / .025 | 1.00 / - / .125~ | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .050~ |
| pcorr_hac_fb_eq_min | raw_p | 1.00 / - / .050~ | 1.00 / - / .025 | 1.00 / - / .050~ | 1.00 / - / .125~ | 1.00 / - / .050~ |
| pcorr_hac_fb_eq_min | tau | 1.00 / - / .025 | 1.00 / - / .125~ | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .050~ |
| pcorr_native | raw_p | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .050~ |
| pcorr_native | tau | 1.00 / - / .025 | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .050~ |
| rcot2_eq | raw_p | 1.00 / - / .125~ | 1.00 / - / .025 | 1.00 / - / .100~ | 1.00 / - / .150~ | 1.00 / - / .050~ |
| rcot2_eq | tau | 1.00 / - / .150~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .050~ |
| rcot2_eq_min | raw_p | 1.00 / - / .075~ | 1.00 / - / .025 | 1.00 / - / .050~ | 1.00 / - / .025 | 1.00 / - / .050~ |
| rcot2_eq_min | tau | 1.00 / - / .150~ | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .050~ |
| rcot2_native | raw_p | 1.00 / - / .075~ | 1.00 / - / .025 | 1.00 / - / .050~ | 1.00 / - / .025 | 1.00 / - / .050~ |
| rcot2_native | tau | 1.00 / - / .150~ | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .050~ |
| shap_dag | tau | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .100~ | 1.00 / - / .000 | 1.00 / - / .025 |
| two_tower | tau | 1.00 / - / .025 | 1.00 / - / .025 | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .225* |

### E4 R4 l1.5

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | raw_p | NA / - / .025 | NA / - / .075~ | NA / - / .075~ | NA / - / .050~ | NA / - / .050~ |
| pmrt_nl_eq | tau | NA / - / .025 | NA / - / .025 | NA / - / .275* | NA / - / .050~ | NA / - / .100~ |
| pmrt_eq | raw_p | NA / - / .025 | NA / - / .050~ | NA / - / .075~ | NA / - / .050~ | NA / - / .050~ |
| pmrt_eq | tau | NA / - / .050~ | NA / - / .075~ | NA / - / .050~ | NA / - / .000 | NA / - / .025 |
| pmrt_r3 | raw_p | NA / - / .025 | NA / - / .050~ | NA / - / .075~ | NA / - / .050~ | NA / - / .050~ |
| pmrt_r3 | tau | NA / - / .050~ | NA / - / .075~ | NA / - / .050~ | NA / - / .000 | NA / - / .025 |
| cdl | - | nig | nig | nig | nig | nig |
| corr | raw_p | 1.00 / - / .025 | 1.00 / - / .050~ | 1.00 / - / .075~ | 1.00 / - / .075~ | 1.00 / - / .050~ |
| corr | tau | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .025 |
| mscr_eq | raw_p | 1.00 / - / .025 | 1.00 / - / .025 | nig | nig | nig |
| mscr_eq | tau | 1.00 / - / .025 | 1.00 / - / .075~ | nig | nig | nig |
| mscr_native | raw_p | 1.00 / - / .175* | 1.00 / - / .600* | nig | nig | nig |
| mscr_native | tau | 1.00 / - / .000 | 1.00 / - / .000 | nig | nig | nig |
| notears | tau | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pc_eq | tau | 1.00 / - / .050~ | 1.00 / - / .075~ | 1.00 / - / .050~ | 1.00 / - / .075~ | 1.00 / - / .000 |
| pc_native | tau | 1.00 / - / .050~ | 1.00 / - / .075~ | 1.00 / - / .075~ | 1.00 / - / .075~ | 1.00 / - / .000 |
| pcorr_eq | raw_p | 1.00 / - / .025 | 1.00 / - / .050~ | 1.00 / - / .075~ | 1.00 / - / .075~ | 1.00 / - / .050~ |
| pcorr_eq | tau | 1.00 / - / .075~ | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .025 |
| pcorr_eq_min | raw_p | 1.00 / - / .050~ | 1.00 / - / .075~ | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .075~ |
| pcorr_eq_min | tau | 1.00 / - / .100~ | 1.00 / - / .125~ | 1.00 / - / .125~ | 1.00 / - / .100~ | 1.00 / - / .025 |
| pcorr_hac | raw_p | 1.00 / - / .075~ | 1.00 / - / .075~ | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .075~ |
| pcorr_hac | tau | 1.00 / - / .100~ | 1.00 / - / .125~ | 1.00 / - / .125~ | 1.00 / - / .100~ | 1.00 / - / .025 |
| pcorr_hac_eq_min | raw_p | 1.00 / - / .075~ | 1.00 / - / .075~ | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .075~ |
| pcorr_hac_eq_min | tau | 1.00 / - / .100~ | 1.00 / - / .125~ | 1.00 / - / .125~ | 1.00 / - / .100~ | 1.00 / - / .025 |
| pcorr_hac_fb | raw_p | 1.00 / - / .050~ | 1.00 / - / .075~ | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .075~ |
| pcorr_hac_fb | tau | 1.00 / - / .100~ | 1.00 / - / .125~ | 1.00 / - / .125~ | 1.00 / - / .100~ | 1.00 / - / .025 |
| pcorr_hac_fb_eq_min | raw_p | 1.00 / - / .050~ | 1.00 / - / .075~ | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .075~ |
| pcorr_hac_fb_eq_min | tau | 1.00 / - / .100~ | 1.00 / - / .125~ | 1.00 / - / .125~ | 1.00 / - / .100~ | 1.00 / - / .025 |
| pcorr_native | raw_p | 1.00 / - / .050~ | 1.00 / - / .075~ | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .075~ |
| pcorr_native | tau | 1.00 / - / .100~ | 1.00 / - / .125~ | 1.00 / - / .125~ | 1.00 / - / .100~ | 1.00 / - / .025 |
| rcot2_eq | raw_p | 1.00 / - / .100~ | 1.00 / - / .025 | 1.00 / - / .075~ | 1.00 / - / .150~ | 1.00 / - / .050~ |
| rcot2_eq | tau | 1.00 / - / .200* | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .050~ |
| rcot2_eq_min | raw_p | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .025 | 1.00 / - / .050~ |
| rcot2_eq_min | tau | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .025 | 1.00 / - / .100~ | 1.00 / - / .025 |
| rcot2_native | raw_p | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .025 | 1.00 / - / .050~ |
| rcot2_native | tau | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .150~ | 1.00 / - / .025 |
| shap_dag | tau | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .025 |
| two_tower | tau | 1.00 / - / .025 | 1.00 / - / .025 | 1.00 / - / .075~ | 1.00 / - / .000 | 1.00 / - / .125~ |

### E5 R1

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | raw_p | .78 / .033 / .062~ | .92 / .037 / .037 | 1.00 / .035 / .056~ | 1.00 / .043 / .050~ | 1.00 / .058~ / .037 |
| pmrt_nl_eq | tau | .82 / .075~ / .094~ | .93 / .068~ / .069~ | 1.00 / .060~ / .081~ | 1.00 / .048 / .044 | 1.00 / .075~ / .087~ |
| pmrt_eq | raw_p | .40 / .040 / .025 | .46 / .058~ / .037 | .55 / .060~ / .069~ | .60 / .050 / .050~ | .67 / .040 / .069~ |
| pmrt_eq | tau | .40 / .028 / .019 | .40 / .033 / .006 | .53 / .040 / .031 | .57 / .022 / .025 | .68 / .043 / .075~ |
| pmrt_r3 | raw_p | .40 / .043 / .037 | .46 / .055~ / .031 | .55 / .055~ / .069~ | .60 / .043 / .056~ | .67 / .040 / .069~ |
| pmrt_r3 | tau | .40 / .035 / .025 | .39 / .028 / .019 | .53 / .037 / .037 | .57 / .015 / .025 | .68 / .043 / .069~ |
| cdl | - | - | - | - | nig | nig |
| cdl | tau | .67 / .013 / .019 | .68 / .040 / .037 | .68 / .013 / .019 | nig | nig |
| cdl | fixed | .00 / .000 / .000 | .27 / .000 / .000 | .67 / .000 / .000 | nig | nig |
| corr | raw_p | .41 / .030 / .025 | .45 / .040 / .037 | .55 / .048 / .050~ | .61 / .043 / .025 | .67 / .037 / .037 |
| corr | tau | .40 / .022 / .019 | .42 / .033 / .019 | .52 / .028 / .031 | .61 / .048~ / .025 | .68 / .040 / .044 |
| mscr_eq | raw_p | .70 / .037 / .050~ | .72 / .043 / .081~ | nig | nig | nig |
| mscr_eq | tau | .72 / .062~ / .069~ | .72 / .048 / .094~ | nig | nig | nig |
| mscr_native | raw_p | .70 / .048 / .069~ | .71 / .050 / .094* | nig | nig | nig |
| mscr_native | tau | .70 / .035 / .069~ | .74 / .052 / .113* | nig | nig | nig |
| notears | tau | .38 / .007 / .013 | .35 / .003 / .000 | .33 / .000 / .000 | .33 / .000 / .000 | .33 / .000 / .000 |
| pc_eq | tau | .39 / .013 / .019 | .40 / .010 / .013 | .50 / .013 / .031 | .53 / .020 / .056~ | .53 / .007 / .037 |
| pc_native | tau | .40 / .013 / .019 | .39 / .010 / .013 | .50 / .013 / .031 | .54 / .025 / .050~ | .53 / .010 / .037 |
| pcorr_eq | raw_p | .41 / .045 / .031 | .45 / .052~ / .031 | .56 / .045 / .062~ | .61 / .048 / .062~ | .68 / .037 / .069~ |
| pcorr_eq | tau | .41 / .043 / .031 | .41 / .020 / .031 | .52 / .035 / .050~ | .57 / .010 / .025 | .68 / .037 / .069~ |
| pcorr_eq_min | raw_p | .41 / .048 / .044 | .46 / .058~ / .031 | .55 / .050 / .069~ | .61 / .048 / .050~ | .68 / .037 / .069~ |
| pcorr_eq_min | tau | .41 / .048 / .044 | .40 / .015 / .025 | .53 / .035 / .037 | .57 / .015 / .025 | .67 / .037 / .069~ |
| pcorr_hac | raw_p | .41 / .048 / .044 | .46 / .060~ / .031 | .55 / .052~ / .062~ | .60 / .048 / .056~ | .67 / .037 / .069~ |
| pcorr_hac | tau | .41 / .050 / .044 | .42 / .022 / .025 | .53 / .035 / .050~ | .57 / .015 / .025 | .67 / .037 / .069~ |
| pcorr_hac_eq_min | raw_p | .41 / .048 / .044 | .46 / .060~ / .031 | .55 / .052~ / .062~ | .60 / .048 / .056~ | .67 / .037 / .069~ |
| pcorr_hac_eq_min | tau | .41 / .050 / .044 | .42 / .022 / .025 | .53 / .035 / .050~ | .57 / .015 / .025 | .67 / .037 / .069~ |
| pcorr_hac_fb | raw_p | .41 / .048 / .044 | .46 / .060~ / .031 | .55 / .052~ / .062~ | .60 / .048 / .056~ | .67 / .037 / .069~ |
| pcorr_hac_fb | tau | .41 / .048 / .044 | .41 / .022 / .025 | .53 / .035 / .050~ | .57 / .015 / .025 | .67 / .037 / .069~ |
| pcorr_hac_fb_eq_min | raw_p | .41 / .048 / .044 | .46 / .060~ / .031 | .55 / .052~ / .062~ | .60 / .048 / .056~ | .67 / .037 / .069~ |
| pcorr_hac_fb_eq_min | tau | .41 / .048 / .044 | .41 / .022 / .025 | .53 / .035 / .050~ | .57 / .015 / .025 | .67 / .037 / .069~ |
| pcorr_native | raw_p | .41 / .048 / .044 | .46 / .058~ / .031 | .55 / .050 / .069~ | .61 / .048 / .050~ | .68 / .037 / .069~ |
| pcorr_native | tau | .41 / .048 / .044 | .40 / .015 / .025 | .53 / .035 / .037 | .57 / .015 / .025 | .67 / .037 / .069~ |
| rcot2_eq | raw_p | .73 / .072~ / .075~ | .77 / .085* / .056~ | .89 / .062~ / .087~ | .95 / .045 / .044~ | 1.00 / .037 / .037 |
| rcot2_eq | tau | .73 / .025 / .081~ | .75 / .030 / .031 | .92 / .072~ / .119* | .95 / .060~ / .062~ | .99 / .060~ / .075~ |
| rcot2_eq_min | raw_p | .72 / .077* / .056~ | .80 / .065~ / .100* | .93 / .068~ / .087~ | .97 / .058~ / .050~ | 1.00 / .060~ / .056~ |
| rcot2_eq_min | tau | .70 / .022 / .037 | .77 / .020 / .075~ | .94 / .050 / .113* | .96 / .045 / .100* | .99 / .040 / .050~ |
| rcot2_native | raw_p | .73 / .090* / .094~ | .79 / .052~ / .075~ | .92 / .065~ / .094~ | .97 / .060~ / .062~ | 1.00 / .045 / .031 |
| rcot2_native | tau | .72 / .030 / .056~ | .77 / .025 / .069~ | .93 / .048 / .094~ | .95 / .033 / .056~ | .99 / .035 / .062~ |
| shap_dag | tau | .92 / .052~ / .050~ | .94 / .010 / .013 | 1.00 / .007 / .025 | 1.00 / .015 / .019 | 1.00 / .045 / .050~ |
| two_tower | tau | .70 / .083* / .050~ | .71 / .052 / .056~ | .74 / .102* / .062~ | .77 / .120* / .056~ | .74 / .113* / .062~ |

### E5 R2

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | raw_p | .70 / .050 / .044~ | .70 / .045 / .019 | .82 / .052 / .025 | .88 / .068~ / .019 | .95 / .058~ / .037 |
| pmrt_nl_eq | tau | .74 / .140* / .163* | .70 / .025 / .013 | .82 / .050 / .031 | .89 / .075* / .037 | .94 / .043 / .013 |
| pmrt_eq | raw_p | .53 / .043 / .087~ | .57 / .048 / .075~ | .63 / .045 / .062~ | .65 / .060~ / .025 | .73 / .058~ / .075~ |
| pmrt_eq | tau | .55 / .060~ / .094~ | .56 / .040 / .062~ | .64 / .055~ / .062~ | .64 / .018 / .006 | .74 / .092* / .106* |
| pmrt_r3 | raw_p | .56 / .045 / .075~ | .60 / .050 / .037 | .66 / .025 / .025 | .69 / .065~ / .031 | .77 / .070~ / .056~ |
| pmrt_r3 | tau | .60 / .090* / .119* | .58 / .022 / .013 | .63 / .007 / .006 | .69 / .052 / .037 | .80 / .100* / .069~ |
| cdl | - | - | - | - | nig | nig |
| cdl | tau | .57 / .000 / .006 | .70 / .013 / .031 | .73 / .025 / .037 | nig | nig |
| cdl | fixed | .00 / .000 / .000 | .00 / .000 / .000 | .54 / .000 / .000 | nig | nig |
| corr | raw_p | .79 / .580* / .675* | .83 / .688* / .750* | .90 / .840* / .887* | .92 / .875* / .938* | .97 / .932* / .975* |
| corr | tau | .37 / .007 / .044~ | .36 / .003 / .025 | .35 / .005 / .025 | .36 / .005 / .031 | .36 / .005 / .031 |
| mscr_eq | raw_p | 1.00 / .998* / 1.000* | 1.00 / 1.000* / 1.000* | nig | nig | nig |
| mscr_eq | tau | .67 / .003 / .019 | .67 / .003 / .031 | nig | nig | nig |
| mscr_native | raw_p | 1.00 / .968* / .969* | 1.00 / 1.000* / 1.000* | nig | nig | nig |
| mscr_native | tau | .67 / .005 / .006 | .67 / .003 / .000 | nig | nig | nig |
| notears | tau | .45 / .022 / .025 | .45 / .033 / .056~ | .44 / .028 / .044 | .45 / .033 / .050~ | .43 / .028 / .044 |
| pc_eq | tau | .47 / .013 / .013 | .50 / .028 / .037 | .53 / .028 / .031 | .57 / .050 / .081~ | .65 / .025 / .019 |
| pc_native | tau | .45 / .018 / .031 | .45 / .035 / .069~ | .00 / .000 / .000 | .00 / .000 / .000 | .00 / .000 / .000 |
| pcorr_eq | raw_p | .56 / .033 / .069~ | .60 / .043 / .056~ | .66 / .045 / .037 | .71 / .052 / .037 | .80 / .070~ / .056~ |
| pcorr_eq | tau | .59 / .075* / .106* | .56 / .005 / .019 | .65 / .030 / .013 | .71 / .048 / .031 | .81 / .095* / .087~ |
| pcorr_eq_min | raw_p | .55 / .037 / .075~ | .60 / .050 / .044~ | .66 / .035 / .031 | .70 / .043 / .044 | .80 / .068~ / .050~ |
| pcorr_eq_min | tau | .61 / .083* / .094~ | .57 / .022 / .025 | .65 / .030 / .013 | .70 / .043 / .037 | .82 / .083* / .062~ |
| pcorr_hac | raw_p | .63 / .270* / .356* | .67 / .328* / .475* | .80 / .485* / .631* | .84 / .542* / .650* | .88 / .598* / .681* |
| pcorr_hac | tau | .37 / .040 / .056~ | .37 / .052 / .069~ | .36 / .040 / .056~ | .37 / .080* / .094~ | .36 / .060~ / .075~ |
| pcorr_hac_eq_min | raw_p | .45 / .035 / .075~ | .50 / .048 / .044~ | .55 / .033 / .031 | .63 / .045 / .050~ | .74 / .068~ / .050~ |
| pcorr_hac_eq_min | tau | .49 / .070~ / .100~ | .46 / .020 / .025 | .55 / .028 / .013 | .62 / .043 / .044 | .76 / .083* / .062~ |
| pcorr_hac_fb | raw_p | .63 / .268* / .356* | .67 / .323* / .475* | .80 / .485* / .631* | .84 / .542* / .650* | .88 / .598* / .681* |
| pcorr_hac_fb | tau | .37 / .040 / .056~ | .37 / .052 / .062~ | .36 / .040 / .056~ | .37 / .080* / .087~ | .36 / .060~ / .075~ |
| pcorr_hac_fb_eq_min | raw_p | .45 / .035 / .075~ | .50 / .048 / .044~ | .55 / .033 / .031 | .63 / .045 / .050~ | .74 / .068~ / .050~ |
| pcorr_hac_fb_eq_min | tau | .49 / .070~ / .100~ | .46 / .020 / .025 | .55 / .028 / .013 | .62 / .043 / .044 | .76 / .083* / .062~ |
| pcorr_native | raw_p | .69 / .270* / .356* | .72 / .335* / .500* | .82 / .490* / .631* | .86 / .557* / .650* | .93 / .613* / .694* |
| pcorr_native | tau | .43 / .043 / .050~ | .42 / .048 / .050~ | .40 / .033 / .044 | .43 / .065~ / .081~ | .42 / .045 / .044 |
| rcot2_eq | raw_p | .71 / .107* / .069~ | .71 / .140* / .094* | .84 / .230* / .225* | .88 / .383* / .375* | .95 / .623* / .562* |
| rcot2_eq | tau | .66 / .033 / .025 | .67 / .075~ / .044~ | .72 / .037 / .062~ | .72 / .037 / .031 | .73 / .045 / .050~ |
| rcot2_eq_min | raw_p | .71 / .090* / .081~ | .69 / .102* / .081~ | .81 / .135* / .169* | .87 / .198* / .244* | .97 / .465* / .581* |
| rcot2_eq_min | tau | .66 / .035 / .044 | .67 / .070~ / .050~ | .71 / .035 / .025 | .73 / .045 / .087~ | .78 / .040 / .069~ |
| rcot2_native | raw_p | .73 / .133* / .094~ | .75 / .152* / .144* | .85 / .207* / .306* | .91 / .310* / .438* | .97 / .595* / .619* |
| rcot2_native | tau | .69 / .065~ / .062~ | .68 / .048 / .025 | .69 / .007 / .025 | .72 / .045 / .075~ | .75 / .033 / .056~ |
| shap_dag | tau | .68 / .060~ / .031 | .67 / .020 / .006 | .69 / .050~ / .025 | .73 / .028 / .025 | .85 / .010 / .013 |
| two_tower | tau | .68 / .055~ / .025 | .68 / .028 / .044~ | .77 / .085* / .056~ | .75 / .030 / .037 | .77 / .020 / .025 |

### V4 claim verdicts

- **Claim: SUPPORTED** (components {'C1': 'SUPPORTED', 'C2a': 'SUPPORTED', 'C2b': 'SUPPORTED', 'C3': 'SUPPORTED'})
- C1: **SUPPORTED** (5 / 6 counted arms of D are design-blind failures; |D| = 6). Wording: design-blind tests are invalid on the R2 (setpoint + dither) design tested.
- C1 sensitivity without mscr_native (R-56, no effect): **SUPPORTED** (4 / 5 counted arms fail)
- Arm labels (R-40; every table): mscr_eq: single-conditioner max statistic; cannot condition on the joint design set; reported INVALID arm, EVAL n <= 1000 (R-54)
- Arm labels (R-40; every table): mscr_native: single-conditioner max statistic; reported INVALID arm, EVAL n <= 1000 (R-54)
- Arm labels (R-40; every table): pmrt_eq: linear statistic (pmrt-core-v1); secondary PMRT arm (R-42)
  - corr: **FAILURE**; R2 INVALID 22/25 (counted 25), pooled R2 null_raw .786 [.772, .802] INVALID; plac_raw .751 [.726, .776] INVALID; R1 INVALID 0/25 (F_max 2)
  - granger_native: **FAILURE**; R2 INVALID 5/5 (counted 5), pooled R2 null_raw .686 [.654, .719] INVALID; plac_raw .578 [.515, .642] INVALID; R1 INVALID 0/5 (F_max 1)
  - mscr_native [single-conditioner max statistic; reported INVALID arm, EVAL n <= 1000 (R-54)]: **FAILURE**; R2 INVALID 8/10 (counted 10), pooled R2 null_raw .944 [.933, .956] INVALID; plac_raw .906 [.886, .924] INVALID; R1 INVALID 1/10 (F_max 1)
  - pcorr_hac_fb: **FAILURE**; R2 INVALID 20/25 (counted 25), pooled R2 null_raw .468 [.435, .497] INVALID; plac_raw .355 [.312, .397] INVALID; R1 INVALID 0/25 (F_max 2)
  - pcorr_native: **FAILURE**; R2 INVALID 20/25 (counted 25), pooled R2 null_raw .471 [.438, .500] INVALID; plac_raw .350 [.307, .393] INVALID; R1 INVALID 0/25 (F_max 2)
  - rcot2_native: **INVALID IN R1**; R2 INVALID 20/25 (counted 25), pooled R2 null_raw .327 [.317, .338] INVALID; plac_raw .309 [.291, .328] INVALID; R1 INVALID 6/25 (F_max 2)
  - pcorr_hac (not in D, descriptive): **FAILURE**; R2 INVALID 20/25 (counted 25), pooled R2 null_raw .469 [.435, .497] INVALID; plac_raw .356 [.312, .397] INVALID; R1 INVALID 0/25 (F_max 2)
- C2a pmrt_nl_eq valid in R1 / R2: **SUPPORTED**
  - cells 50/50, INVALID 1 (F_max 3), VALID 18, INCONCLUSIVE 31; pooled null_raw .045 [.043, .048] VALID; plac_raw .048 [.042, .054] VALID
- C2b eq arms whose native partner is a C1 failure: **SUPPORTED**
  - granger_eq (native granger_native): **SUPPORTED**
    - cells 10/10, INVALID 0 (F_max 1), VALID 2, INCONCLUSIVE 8; pooled null_raw .050 [.044, .056] VALID; plac_raw .056 [.041, .071] VALID
  - pcorr_eq (native pcorr_native): **SUPPORTED**
    - cells 50/50, INVALID 0 (F_max 3), VALID 19, INCONCLUSIVE 31; pooled null_raw .048 [.045, .051] VALID; plac_raw .051 [.044, .057] VALID
  - mscr_eq EXCLUDED from C2b (R-40) [single-conditioner max statistic; cannot condition on the joint design set; reported INVALID arm, EVAL n <= 1000 (R-54)], same rule reported (no effect): **NOT SUPPORTED**
    - cells 20/20, INVALID 8 (F_max 2), VALID 4, INCONCLUSIVE 8; pooled null_raw .503 [.498, .508] INVALID; plac_raw .489 [.479, .500] INVALID
- C2b with the R1-failing partners (R-56, sensitivity, no effect): **SUPPORTED** (added: rcot2_eq)
- C2 wording: using the design restores validity (design-based inference and design-covariate adjustment); rcot2_eq INVALID in R2 as already in R1 (not restored by design covariates).
  - rcot2_eq is INVALID in R2 (pooled truth-null raw rate 0.292 [0.283, 0.301]) as already in R1 (0.061 [0.056, 0.066]): the test is miscalibrated without the design (its native partner rcot2_native is INVALID IN R1), so it is outside C2b's membership rule, and adding design covariates does not make it valid.

Every eq arm, same rule on R1 + R2, unfiltered (R-56):

| eq arm | C2b membership | native C1 | verdict | INVALID / cells (F_max) | R2 INVALID / planned | pooled R1 truth-null | pooled R2 truth-null |
|---|---|---|---|---|---|---|---|
| granger_eq | member | FAILURE | SUPPORTED | 0/10 (1) | 0/5 | 0.048 [0.040, 0.058] | 0.052 [0.044, 0.060] |
| mscr_eq [single-conditioner max statistic; cannot condition on the joint design set; reported INVALID arm, EVAL n <= 1000 (R-54)] | excluded (R-40) | FAILURE | NOT SUPPORTED | 8/20 (2) | 8/10 | 0.046 [0.040, 0.051] | 0.960 [0.952, 0.970] |
| pc_eq | not a member (declare tau) | - | NOT SUPPORTED | 4/50 (2) | - | - | - |
| pcorr_eq | member | FAILURE | SUPPORTED | 0/50 (3) | 0/25 | 0.046 [0.042, 0.051] | 0.050 [0.045, 0.054] |
| rcot2_eq | not a member (native rcot2_native: INVALID IN R1) | INVALID IN R1 | NOT SUPPORTED | 25/50 (3) | 20/25 | 0.061 [0.056, 0.066] | 0.292 [0.283, 0.301] |

- C3 pmrt_nl_eq valid in E4 R3: **SUPPORTED**
  - cells 20/20, INVALID 0 (F_max 2), VALID 6, INCONCLUSIVE 14; pooled plac_raw .048 [.038, .060] VALID; conf_raw .042 [.034, .050] VALID
  - wording: PMRT stays valid under the logged confounded policy (R3) by construction; so do pcorr_eq, so the property is not specific to PMRT.
- pmrt_eq [linear statistic (pmrt-core-v1); secondary PMRT arm (R-42)] (secondary PMRT arm, same rules reported, no claim effect): C2a: **SUPPORTED**
  - cells 50/50, INVALID 0 (F_max 3), VALID 19, INCONCLUSIVE 31; pooled null_raw .049 [.046, .052] VALID; plac_raw .050 [.044, .056] VALID
  - pmrt_eq: C3: **NOT SUPPORTED**
    - cells 20/20, INVALID 3 (F_max 2), VALID 5, INCONCLUSIVE 12; pooled plac_raw .051 [.040, .064] VALID; conf_raw .049 [.039, .061] VALID
- pmrt_r3 (secondary PMRT arm, same rules reported, no claim effect): C2a: **SUPPORTED**
  - cells 50/50, INVALID 1 (F_max 3), VALID 19, INCONCLUSIVE 30; pooled null_raw .049 [.045, .052] VALID; plac_raw .049 [.043, .056] VALID
  - pmrt_r3: C3: **SUPPORTED**
    - cells 20/20, INVALID 1 (F_max 2), VALID 4, INCONCLUSIVE 15; pooled plac_raw .050 [.038, .063] VALID; conf_raw .050 [.040, .062] VALID
- C3 reported for every eq arm (no claim effect): granger_eq NOT EVALUABLE; mscr_eq [single-conditioner max statistic; cannot condition on the joint design set; reported INVALID arm, EVAL n <= 1000 (R-54)] NOT SUPPORTED; pc_eq NOT SUPPORTED; pcorr_eq SUPPORTED; rcot2_eq NOT SUPPORTED

### V1 validity: INVALID / VALID / counted cells per world-regime (all n)

| world regime | pmrt_nl_eq | pmrt_eq | pmrt_r3 | cdl | corr | granger_eq | granger_native | mscr_eq | mscr_native | notears | pc_eq | pc_native | pcorr_eq | pcorr_eq_min | pcorr_hac | pcorr_hac_eq_min | pcorr_hac_fb | pcorr_hac_fb_eq_min | pcorr_native | rcot2_eq | rcot2_eq_min | rcot2_native | shap_dag | two_tower |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 R1 | 0/0/5 | 0/1/5 | **1**/2/5 | **2**/1/3 | 0/2/5 | - | - | 0/0/2 | 0/1/2 | 0/5/5 | **1**/4/5 | **1**/3/5 | 0/2/5 | 0/2/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/2/5 | **1**/0/5 | **2**/0/5 | **2**/0/5 | **2**/3/5 | 0/5/5 |
| E1 R2 | **1**/0/5 | 0/2/5 | 0/2/5 | **2**/1/3 | **5**/0/5 | - | - | **2**/0/2 | **2**/0/2 | **5**/0/5 | **1**/2/5 | 0/3/5 | 0/1/5 | 0/1/5 | **5**/0/5 | 0/0/5 | **5**/0/5 | 0/0/5 | **5**/0/5 | **5**/0/5 | **4**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 |
| E2 R1 | 0/4/5 | 0/2/5 | 0/2/5 | **2**/1/3 | 0/2/5 | - | - | 0/1/2 | 0/2/2 | 0/5/5 | **1**/4/5 | **2**/3/5 | 0/3/5 | 0/2/5 | 0/2/5 | 0/2/5 | 0/2/5 | 0/2/5 | 0/2/5 | **2**/2/5 | **1**/2/5 | **1**/3/5 | **1**/4/5 | **3**/2/5 |
| E2 R2 | 0/3/5 | 0/5/5 | 0/3/5 | 0/3/3 | **5**/0/5 | - | - | **2**/0/2 | **2**/0/2 | 0/5/5 | **1**/4/5 | 0/5/5 | 0/3/5 | 0/4/5 | **5**/0/5 | 0/2/5 | **5**/0/5 | 0/3/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | 0/5/5 | 0/4/5 |
| E3 R1 | 0/3/5 | 0/2/5 | 0/0/5 | 0/3/3 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/2 | 0/2/2 | 0/5/5 | 0/3/5 | 0/4/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | **1**/1/5 | **2**/0/5 | **2**/2/5 | **2**/3/5 | **2**/3/5 |
| E3 R2 | 0/0/5 | 0/1/5 | 0/2/5 | **3**/0/3 | **5**/0/5 | 0/1/5 | **5**/0/5 | **2**/0/2 | **2**/0/2 | **5**/0/5 | 0/5/5 | **5**/0/5 | 0/1/5 | 0/2/5 | **5**/0/5 | 0/2/5 | **5**/0/5 | 0/2/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 |
| E4 R1 l1 | 0/3/5 | 0/4/5 | 0/4/5 | 0/0/3 | 0/4/5 | - | - | 0/1/2 | 0/1/2 | 0/0/5 | 0/0/5 | 0/0/5 | 0/4/5 | 0/4/5 | 0/4/5 | 0/4/5 | 0/4/5 | 0/4/5 | 0/4/5 | 0/2/5 | 0/2/5 | 0/2/5 | 0/0/5 | 0/0/5 |
| E4 R2 l1 | 0/2/5 | 0/1/5 | 0/1/5 | 0/0/3 | **2**/0/5 | - | - | 0/1/2 | 0/2/2 | 0/0/5 | 0/0/5 | 0/0/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/2/5 | 0/3/5 | 0/0/5 | 0/0/5 |
| E4 R3 l0 | 0/0/5 | **1**/1/5 | 0/1/5 | nig | 0/0/5 | - | - | 0/0/2 | 0/0/2 | 0/5/5 | 0/4/5 | 0/4/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/2/5 | 0/2/5 | 0/1/5 | **2**/1/5 | 0/1/5 | 0/2/5 | 0/5/5 | 0/5/5 |
| E4 R3 l0.5 | 0/2/5 | **1**/1/5 | 0/1/5 | nig | **5**/0/5 | - | - | **2**/0/2 | **2**/0/2 | 0/5/5 | **2**/1/5 | **1**/2/5 | **1**/1/5 | 0/0/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/0/5 | **5**/0/5 | **2**/0/5 | **2**/0/5 | **5**/0/5 | **5**/0/5 |
| E4 R3 l1 | 0/3/5 | **1**/1/5 | **1**/1/5 | nig | **5**/0/5 | - | - | **2**/0/2 | **2**/0/2 | 0/5/5 | **3**/0/5 | 0/0/5 | **1**/0/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | **5**/0/5 | **3**/0/5 | **3**/0/5 | **5**/0/5 | **5**/0/5 |
| E4 R3 l1.5 | 0/1/5 | 0/2/5 | 0/1/5 | nig | **5**/0/5 | - | - | **2**/0/2 | **2**/0/2 | 0/5/5 | **2**/3/5 | 0/1/5 | 0/0/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | **5**/0/5 | **3**/0/5 | **4**/0/5 | **5**/0/5 | **5**/0/5 |
| E4 R4 l0 | 0/1/5 | 0/1/5 | 0/1/5 | nig | 0/1/5 | - | - | 0/0/2 | 0/0/2 | 0/5/5 | 0/4/5 | 0/4/5 | 0/0/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/0/5 | 0/1/5 | 0/0/5 | 0/5/5 | 0/5/5 |
| E4 R4 l0.5 | 0/2/5 | 0/0/5 | 0/1/5 | nig | **5**/0/5 | - | - | **2**/0/2 | **2**/0/2 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 |
| E4 R4 l1 | 0/2/5 | 0/1/5 | 0/1/5 | nig | **5**/0/5 | - | - | **2**/0/2 | **2**/0/2 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 |
| E4 R4 l1.5 | 0/1/5 | 0/1/5 | 0/1/5 | nig | **5**/0/5 | - | - | **2**/0/2 | **2**/0/2 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 |
| E5 R1 | 0/1/5 | 0/1/5 | 0/1/5 | 0/3/3 | 0/4/5 | - | - | 0/0/2 | **1**/0/2 | 0/5/5 | 0/5/5 | 0/5/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | **1**/1/5 | **2**/0/5 | **1**/1/5 | 0/4/5 | **4**/1/5 |
| E5 R2 | 0/2/5 | 0/0/5 | 0/2/5 | 0/3/3 | **5**/0/5 | - | - | **2**/0/2 | **2**/0/2 | 0/5/5 | 0/5/5 | 0/5/5 | 0/2/5 | 0/2/5 | **5**/0/5 | 0/1/5 | **5**/0/5 | 0/1/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | 0/3/5 | **1**/3/5 |

(Nx) = planned cells not counted (missing, infeasible, T3-infeasible, untuned or < 10 seeds); nig = not in the arm's grid (pre-registered grid choice, R-54 / R-58: not planned, never missing / PARTIAL).

Not in grid (pre-registered grid choices):

- R-58(1): cdl (regimes ['R3', 'R4']): 40 cells; cdl: training cost; feeds no claim (R1 / R2, n <= 4000 only)
- R-58(1): cdl (ns [8000, 24000]): 20 cells; cdl: training cost; feeds no claim (R1 / R2, n <= 4000 only)
- R-58(1): cdl (kappas [0.125, 0.5]): 10 cells; cdl: not in the kappa sweep
- R-54: mscr_eq, mscr_native (ns [4000, 8000, 24000]): 108 cells; mscr: reported-INVALID arm, EVAL n <= 1000 only
- R-58(5): mscr_eq, mscr_native (kappas [0.125, 0.5]): 20 cells; mscr: not in the kappa sweep

INVALID cells (606; first 30):
- cdl|E1|R1|k0.25|n1000: null_decl .081 [.056, .108]
- cdl|E1|R1|k0.25|n500: null_decl .075 [.054, .098]
- cdl|E1|R2|k0.25|n1000: null_decl .106 [.079, .135]
- cdl|E1|R2|k0.25|n4000: null_decl .183 [.173, .198]
- cdl|E2|R1|k0.25|n1000: null_decl .078 [.066, .091]
- cdl|E2|R1|k0.25|n500: null_decl .073 [.059, .087]
- cdl|E3|R2|k0.25|n1000: null_decl .163 [.138, .191]
- cdl|E3|R2|k0.25|n4000: null_decl .205 [.191, .220]
- cdl|E3|R2|k0.25|n500: null_decl .111 [.086, .136]
- corr|E1|R2|k0.25|n1000: null_raw .758 [.713, .806]; null_decl .685 [.640, .733]; plac_raw .750 [.681, .812]; plac_decl .644 [.569, .719]
- corr|E1|R2|k0.25|n24000: null_raw .954 [.925, .977]; null_decl .940 [.908, .965]; plac_raw .925 [.881, .963]; plac_decl .912 [.869, .950]
- corr|E1|R2|k0.25|n4000: null_raw .887 [.856, .917]; null_decl .852 [.819, .885]; plac_raw .863 [.819, .906]; plac_decl .831 [.775, .887]
- corr|E1|R2|k0.25|n500: null_raw .683 [.633, .731]; null_decl .569 [.508, .627]; plac_raw .644 [.575, .706]; plac_decl .550 [.469, .631]
- corr|E1|R2|k0.25|n8000: null_raw .921 [.887, .952]; null_decl .892 [.856, .925]; plac_raw .912 [.869, .950]; plac_decl .875 [.825, .925]
- corr|E2|R2|k0.25|n1000: null_raw .606 [.580, .634]; null_decl .473 [.441, .507]; plac_raw .583 [.521, .646]; plac_decl .487 [.425, .554]
- corr|E2|R2|k0.25|n24000: null_raw .918 [.903, .932]; null_decl .898 [.883, .912]; plac_raw .933 [.900, .963]; plac_decl .896 [.850, .933]
- corr|E2|R2|k0.25|n4000: null_raw .795 [.778, .814]; null_decl .727 [.702, .751]; plac_raw .771 [.708, .829]; plac_decl .700 [.642, .758]
- corr|E2|R2|k0.25|n500: null_raw .506 [.476, .539]; null_decl .338 [.305, .373]; plac_raw .442 [.375, .508]; plac_decl .283 [.229, .342]
- corr|E2|R2|k0.25|n8000: null_raw .858 [.840, .876]; null_decl .802 [.780, .822]; plac_raw .838 [.787, .883]; plac_decl .767 [.704, .825]
- corr|E3|R2|k0.25|n1000: null_raw .784 [.748, .819]; null_decl .720 [.680, .761]; plac_raw .685 [.600, .765]; plac_decl .630 [.545, .710]
- corr|E3|R2|k0.25|n24000: null_raw .939 [.917, .961]; null_decl .931 [.908, .953]; plac_raw .925 [.880, .965]; plac_decl .905 [.855, .950]
- corr|E3|R2|k0.25|n4000: null_raw .903 [.877, .930]; null_decl .866 [.836, .895]; plac_raw .830 [.770, .885]; plac_decl .775 [.710, .835]
- corr|E3|R2|k0.25|n500: null_raw .691 [.647, .733]; null_decl .617 [.564, .669]; plac_raw .570 [.480, .660]; plac_decl .475 [.380, .560]
- corr|E3|R2|k0.25|n8000: null_raw .920 [.895, .944]; null_decl .903 [.878, .927]; plac_raw .875 [.825, .920]; plac_decl .840 [.785, .895]
- corr|E4|R2|lam1|k0.25|n24000: plac_raw .400 [.250, .550]; plac_decl .350 [.200, .500]
- corr|E4|R2|lam1|k0.25|n8000: plac_raw .325 [.175, .475]; plac_decl .325 [.175, .475]
- corr|E4|R3|lam0.5|k0.25|n1000: conf_raw 1.000 [1.000, 1.000]; conf_decl 1.000 [1.000, 1.000]
- corr|E4|R3|lam0.5|k0.25|n24000: conf_raw 1.000 [1.000, 1.000]; conf_decl 1.000 [1.000, 1.000]
- corr|E4|R3|lam0.5|k0.25|n4000: conf_raw 1.000 [1.000, 1.000]; conf_decl 1.000 [1.000, 1.000]
- corr|E4|R3|lam0.5|k0.25|n500: conf_raw 1.000 [1.000, 1.000]; conf_decl 1.000 [1.000, 1.000]

### V2 recall among cells not INVALID (n 500 / 1000 / 4000 / 8000 / 24000; inv INVALID, inf infeasible, mis missing, unt untuned, few < 10 seeds; NA not applicable, R-42)

| world regime | pmrt_nl_eq | pmrt_eq | pmrt_r3 | cdl | corr | granger_eq | granger_native | mscr_eq | mscr_native | notears | pc_eq | pc_native | pcorr_eq | pcorr_eq_min | pcorr_hac | pcorr_hac_eq_min | pcorr_hac_fb | pcorr_hac_fb_eq_min | pcorr_native | rcot2_eq | rcot2_eq_min | rcot2_native | shap_dag | two_tower |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 R1 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/inv/1.00/1.00 | inv/inv/1.00/nig/nig | 1.00/1.00/1.00/1.00/1.00 | - | - | 1.00/1.00/nig/nig/nig | 1.00/1.00/nig/nig/nig | 1.00/1.00/1.00/1.00/1.00 | inv/1.00/1.00/1.00/1.00 | inv/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | inv/1.00/1.00/1.00/1.00 | inv/inv/1.00/1.00/1.00 | inv/inv/1.00/1.00/1.00 | 1.00/1.00/inv/inv/1.00 | 1.00/1.00/1.00/1.00/1.00 |
| E1 R2 | 1.00/1.00/1.00/1.00/inv | .96/.97/.99/.99/.99 | 1.00/1.00/1.00/1.00/1.00 | 1.00/inv/inv/nig/nig | inv/inv/inv/inv/inv | - | - | inv/inv/nig/nig/nig | inv/inv/nig/nig/nig | inv/inv/inv/inv/inv | inv/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | inv/inv/inv/inv/inv | 1.00/1.00/1.00/1.00/1.00 | inv/inv/inv/inv/inv | 1.00/1.00/1.00/1.00/1.00 | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/1.00/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv |
| E2 R1 | .79/.83/.94/.95/.99 | .29/.32/.34/.37/.38 | .28/.32/.34/.37/.38 | inv/inv/.96/nig/nig | .30/.32/.35/.36/.38 | - | - | .86/.94/nig/nig/nig | .88/.94/nig/nig/nig | .25/.26/.25/.25/.25 | inv/.32/.38/.38/.41 | inv/inv/.38/.38/.41 | .28/.32/.34/.37/.38 | .29/.32/.34/.36/.38 | .31/.31/.34/.36/.38 | .31/.31/.34/.36/.38 | .31/.31/.34/.36/.38 | .31/.31/.34/.36/.38 | .29/.32/.34/.36/.38 | inv/inv/.60/.62/.67 | inv/.56/.61/.62/.67 | inv/.56/.60/.62/.67 | inv/.95/.99/1.00/1.00 | inv/.67/.78/inv/inv |
| E2 R2 | .49/.66/.93/.93/.94 | .19/.28/.45/.55/.68 | .23/.33/.51/.61/.72 | .40/.49/.80/nig/nig | inv/inv/inv/inv/inv | - | - | inv/inv/nig/nig/nig | inv/inv/nig/nig/nig | .28/.29/.30/.29/.28 | .23/inv/.40/.47/.50 | .24/.29/.26/.00/.00 | .30/.37/.54/.63/.76 | .30/.38/.54/.64/.76 | inv/inv/inv/inv/inv | .23/.28/.44/.53/.68 | inv/inv/inv/inv/inv | .23/.28/.44/.53/.68 | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | .77/.91/.96/.97/.98 | .55/.42/.66/.58/.60 |
| E3 R1 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/nig/nig | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/nig/nig/nig | 1.00/1.00/nig/nig/nig | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | inv/1.00/1.00/1.00/1.00 | inv/inv/1.00/1.00/1.00 | inv/inv/1.00/1.00/1.00 | 1.00/inv/1.00/inv/1.00 | inv/1.00/1.00/1.00/inv |
| E3 R2 | 1.00/1.00/1.00/1.00/1.00 | .98/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | inv/inv/inv/nig/nig | inv/inv/inv/inv/inv | 1.00/1.00/1.00/1.00/1.00 | inv/inv/inv/inv/inv | inv/inv/nig/nig/nig | inv/inv/nig/nig/nig | inv/inv/inv/inv/inv | 1.00/1.00/1.00/1.00/1.00 | inv/inv/inv/inv/inv | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | inv/inv/inv/inv/inv | 1.00/1.00/1.00/1.00/1.00 | inv/inv/inv/inv/inv | 1.00/1.00/1.00/1.00/1.00 | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv |
| E4 R1 l1 | .05/.17/.85/1.00/1.00 | .47/.85/1.00/1.00/1.00 | .47/.85/1.00/1.00/1.00 | .15/.53/1.00/nig/nig | .50/.85/1.00/1.00/1.00 | - | - | .17/.20/nig/nig/nig | .07/.20/nig/nig/nig | .38/.57/.80/.93/.97 | .35/.62/1.00/1.00/1.00 | .38/.62/1.00/1.00/1.00 | .55/.85/1.00/1.00/1.00 | .50/.85/1.00/1.00/1.00 | .55/.85/1.00/1.00/1.00 | .55/.85/1.00/1.00/1.00 | .53/.85/1.00/1.00/1.00 | .53/.85/1.00/1.00/1.00 | .50/.85/1.00/1.00/1.00 | .25/.55/1.00/1.00/1.00 | .25/.55/1.00/1.00/1.00 | .23/.55/1.00/1.00/1.00 | .30/.23/.97/1.00/1.00 | .82/.65/1.00/1.00/1.00 |
| E4 R2 l1 | .00/.00/.07/.03/.05 | .00/.03/.17/.35/.90 | .00/.03/.17/.38/.88 | .17/.17/.60/nig/nig | .10/.33/.85/inv/inv | - | - | .03/.05/nig/nig/nig | .05/.05/nig/nig/nig | .15/.12/.03/.00/.00 | .05/.07/.35/.60/1.00 | .20/.47/.95/1.00/1.00 | .00/.05/.15/.40/.90 | .00/.05/.15/.40/.90 | .10/.33/.82/1.00/1.00 | .00/.05/.17/.40/.90 | .10/.33/.82/1.00/1.00 | .00/.05/.17/.38/.90 | .10/.33/.82/1.00/1.00 | .00/.03/.03/.03/.10 | .03/.03/.05/.03/.07 | .03/.15/.30/.50/.75 | .07/.03/.15/.42/.75 | .10/.03/.93/1.00/1.00 |
| E4 R3 l0 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/inv/1.00 | 1.00/1.00/1.00/1.00/1.00 | nig/nig/nig/nig/nig | .75/1.00/1.00/1.00/1.00 | - | - | .96/1.00/nig/nig/nig | .95/1.00/nig/nig/nig | 1.00/1.00/1.00/1.00/1.00 | .90/1.00/1.00/1.00/1.00 | .88/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | inv/inv/.99/.99/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | .65/.90/1.00/1.00/1.00 | .90/1.00/1.00/1.00/1.00 |
| E4 R3 l0.5 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/inv/1.00 | 1.00/1.00/1.00/1.00/1.00 | nig/nig/nig/nig/nig | inv/inv/inv/inv/inv | - | - | inv/inv/nig/nig/nig | inv/inv/nig/nig/nig | .00/.00/.00/.00/.00 | 1.00/1.00/inv/1.00/inv | 1.00/1.00/1.00/1.00/inv | 1.00/1.00/1.00/inv/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | inv/inv/inv/inv/inv | 1.00/1.00/1.00/inv/inv | 1.00/1.00/1.00/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv |
| E4 R3 l1 | .99/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/inv | 1.00/1.00/1.00/1.00/inv | nig/nig/nig/nig/nig | inv/inv/inv/inv/inv | - | - | inv/inv/nig/nig/nig | inv/inv/nig/nig/nig | .00/.00/.00/.00/.00 | inv/1.00/inv/1.00/inv | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/inv | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | inv/inv/inv/inv/inv | 1.00/1.00/inv/inv/inv | 1.00/1.00/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv |
| E4 R3 l1.5 | .99/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | nig/nig/nig/nig/nig | inv/inv/inv/inv/inv | - | - | inv/inv/nig/nig/nig | inv/inv/nig/nig/nig | .00/.00/.00/.00/.00 | inv/1.00/1.00/1.00/inv | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | inv/inv/inv/inv/inv | 1.00/1.00/inv/inv/inv | 1.00/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv |
| E4 R4 l0 | NA/NA/NA/NA/NA | NA/NA/NA/NA/NA | NA/NA/NA/NA/NA | nig/nig/nig/nig/nig | .78/1.00/1.00/1.00/1.00 | - | - | .17/.33/nig/nig/nig | .12/.35/nig/nig/nig | .82/.82/1.00/1.00/1.00 | .93/1.00/1.00/1.00/1.00 | .95/1.00/1.00/1.00/1.00 | .78/1.00/1.00/1.00/1.00 | .75/1.00/1.00/1.00/1.00 | .75/1.00/1.00/1.00/1.00 | .75/1.00/1.00/1.00/1.00 | .75/1.00/1.00/1.00/1.00 | .75/1.00/1.00/1.00/1.00 | .75/1.00/1.00/1.00/1.00 | .47/.62/1.00/1.00/1.00 | .45/.62/1.00/1.00/1.00 | .42/.65/1.00/1.00/1.00 | .45/.88/1.00/1.00/1.00 | .07/1.00/1.00/1.00/1.00 |
| E4 R4 l0.5 | NA/NA/NA/NA/NA | NA/NA/NA/NA/NA | NA/NA/NA/NA/NA | nig/nig/nig/nig/nig | inv/inv/inv/inv/inv | - | - | inv/inv/nig/nig/nig | inv/inv/nig/nig/nig | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv |
| E4 R4 l1 | NA/NA/NA/NA/NA | NA/NA/NA/NA/NA | NA/NA/NA/NA/NA | nig/nig/nig/nig/nig | inv/inv/inv/inv/inv | - | - | inv/inv/nig/nig/nig | inv/inv/nig/nig/nig | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv |
| E4 R4 l1.5 | NA/NA/NA/NA/NA | NA/NA/NA/NA/NA | NA/NA/NA/NA/NA | nig/nig/nig/nig/nig | inv/inv/inv/inv/inv | - | - | inv/inv/nig/nig/nig | inv/inv/nig/nig/nig | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv |
| E5 R1 | .70/.84/.99/1.00/1.00 | .35/.36/.45/.53/.62 | .35/.36/.45/.53/.62 | .67/.68/.68/nig/nig | .35/.36/.45/.53/.62 | - | - | .67/.68/nig/nig/nig | .67/inv/nig/nig/nig | .38/.35/.33/.33/.33 | .39/.40/.50/.53/.53 | .40/.39/.50/.54/.53 | .35/.36/.45/.52/.62 | .35/.36/.45/.53/.62 | .34/.36/.45/.53/.62 | .34/.36/.45/.53/.62 | .34/.36/.45/.53/.62 | .34/.36/.45/.53/.62 | .35/.36/.45/.53/.62 | .68/inv/.80/.87/.97 | inv/inv/.83/.90/.99 | inv/.70/.83/.90/.99 | .92/.94/1.00/1.00/1.00 | inv/.71/inv/inv/inv |
| E5 R2 | .68/.68/.76/.80/.93 | .42/.47/.56/.60/.64 | .47/.51/.60/.62/.70 | .57/.70/.73/nig/nig | inv/inv/inv/inv/inv | - | - | inv/inv/nig/nig/nig | inv/inv/nig/nig/nig | .45/.45/.44/.45/.43 | .47/.50/.53/.57/.65 | .45/.45/.00/.00/.00 | .48/.55/.61/.64/.74 | .47/.54/.60/.64/.72 | inv/inv/inv/inv/inv | .38/.40/.48/.53/.63 | inv/inv/inv/inv/inv | .37/.40/.47/.53/.63 | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | .68/.67/.69/.73/.85 | .68/.68/inv/.75/.77 |

### V3 paired recall difference pmrt_nl_eq - arm (neither INVALID, same declaration rule; per-cell 95 % paired-t CI, descriptive)

| arm | rule | block | cells pmrt higher | pmrt lower | CI covers 0 |
|---|---|---|---|---|---|
| cdl | tau | primary | 4 | 4 | 9 |
| corr | BY | primary | 11 | 6 | 16 |
| granger_eq | BY | primary | 0 | 0 | 10 |
| granger_native | BY | secondary | 0 | 0 | 5 |
| mscr_eq | BY | primary | 3 | 2 | 9 |
| mscr_native | BY | secondary | 1 | 2 | 10 |
| notears | tau | primary | 29 | 3 | 18 |
| pc_eq | tau | primary | 14 | 6 | 29 |
| pc_native | tau | secondary | 13 | 8 | 29 |
| pcorr_eq | BY | primary | 20 | 6 | 41 |
| pcorr_eq_min | BY | secondary | 20 | 5 | 44 |
| pcorr_hac | BY | secondary | 10 | 8 | 32 |
| pcorr_hac_eq_min | BY | secondary | 20 | 5 | 44 |
| pcorr_hac_fb | BY | secondary | 10 | 8 | 32 |
| pcorr_hac_fb_eq_min | BY | secondary | 20 | 5 | 44 |
| pcorr_native | BY | secondary | 10 | 8 | 32 |
| pmrt_eq | BY | secondary | 20 | 6 | 40 |
| pmrt_r3 | BY | secondary | 20 | 6 | 41 |
| rcot2_eq | BY | primary | 7 | 3 | 18 |
| rcot2_eq_min | BY | secondary | 6 | 3 | 27 |
| rcot2_native | BY | secondary | 7 | 7 | 21 |
| shap_dag | tau | primary | 4 | 9 | 20 |
| two_tower | tau | primary | 6 | 7 | 14 |

### V5 information levels in R2 (pmrt, eq, eq_min, native): recall (* INVALID, ~ INCONCLUSIVE)

| row | pmrt_nl_eq | pmrt_eq | pmrt_r3 | granger_eq | granger_native | mscr_eq | mscr_native | pc_eq | pc_native | pcorr_eq | pcorr_eq_min | pcorr_hac | pcorr_hac_eq_min | pcorr_hac_fb | pcorr_hac_fb_eq_min | pcorr_native | rcot2_eq | rcot2_eq_min | rcot2_native |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 n500 | 1.00~ | .96~ | 1.00~ | - | - | 1.00* | 1.00* | 1.00* | 1.00~ | 1.00~ | 1.00~ | 1.00* | 1.00~ | 1.00* | 1.00~ | 1.00* | .99* | .99* | 1.00* |
| E1 n1000 | 1.00~ | .97 | 1.00 | - | - | 1.00* | 1.00* | 1.00~ | 1.00 | 1.00~ | 1.00~ | 1.00* | 1.00~ | 1.00* | 1.00~ | 1.00* | 1.00* | 1.00~ | 1.00* |
| E1 n4000 | 1.00~ | .99~ | 1.00 | - | - | - | - | 1.00 | 1.00 | 1.00~ | 1.00 | 1.00* | 1.00~ | 1.00* | 1.00~ | 1.00* | 1.00* | 1.00* | 1.00* |
| E1 n8000 | 1.00~ | .99 | 1.00~ | - | - | - | - | 1.00~ | 1.00~ | 1.00~ | 1.00~ | 1.00* | 1.00~ | 1.00* | 1.00~ | 1.00* | 1.00* | 1.00* | 1.00* |
| E1 n24000 | 1.00* | .99~ | 1.00~ | - | - | - | - | 1.00 | 1.00 | 1.00 | 1.00~ | 1.00* | 1.00~ | 1.00* | 1.00~ | 1.00* | 1.00* | 1.00* | 1.00* |
| E2 n500 | .49 | .19 | .23 | - | - | .94* | .93* | .23 | .24 | .30~ | .30 | .38* | .23~ | .38* | .23 | .42* | .41* | .40* | .46* |
| E2 n1000 | .66~ | .28 | .33 | - | - | .97* | .97* | .30* | .29 | .37 | .38 | .49* | .28~ | .49* | .28~ | .52* | .51* | .51* | .54* |
| E2 n4000 | .93 | .45 | .51 | - | - | - | - | .40 | .26 | .54 | .54 | .70* | .44 | .70* | .44 | .74* | .64* | .64* | .65* |
| E2 n8000 | .93~ | .55 | .61~ | - | - | - | - | .47 | .00 | .63~ | .64~ | .77* | .53~ | .77* | .53~ | .80* | .72* | .72* | .75* |
| E2 n24000 | .94 | .68 | .72~ | - | - | - | - | .50 | .00 | .76 | .76 | .86* | .68 | .86* | .68 | .89* | .84* | .85* | .88* |
| E3 n500 | 1.00~ | .98~ | 1.00~ | 1.00~ | 1.00* | 1.00* | 1.00* | 1.00 | 1.00* | 1.00~ | 1.00~ | 1.00* | 1.00~ | 1.00* | 1.00~ | 1.00* | .93* | .91* | .99* |
| E3 n1000 | 1.00~ | 1.00~ | 1.00~ | 1.00~ | 1.00* | 1.00* | 1.00* | 1.00 | 1.00* | 1.00~ | 1.00 | 1.00* | 1.00 | 1.00* | 1.00 | 1.00* | .96* | .94* | .99* |
| E3 n4000 | 1.00~ | 1.00~ | 1.00 | 1.00~ | 1.00* | - | - | 1.00 | 1.00* | 1.00~ | 1.00~ | 1.00* | 1.00~ | 1.00* | 1.00~ | 1.00* | 1.00* | 1.00* | .99* |
| E3 n8000 | 1.00~ | 1.00~ | 1.00~ | 1.00~ | 1.00* | - | - | 1.00 | 1.00* | 1.00~ | 1.00~ | 1.00* | 1.00~ | 1.00* | 1.00~ | 1.00* | 1.00* | 1.00* | 1.00* |
| E3 n24000 | 1.00~ | 1.00 | 1.00 | 1.00 | 1.00* | - | - | 1.00 | 1.00* | 1.00 | 1.00 | 1.00* | 1.00 | 1.00* | 1.00 | 1.00* | 1.00* | 1.00* | 1.00* |
| E4 n500 | .00 | .00~ | .00~ | - | - | .03 | .05 | .05~ | .20~ | .00~ | .00~ | .10~ | .00~ | .10~ | .00~ | .10~ | .00~ | .03~ | .03 |
| E4 n1000 | .00~ | .03~ | .03~ | - | - | .05~ | .05 | .07~ | .47~ | .05~ | .05~ | .33~ | .05~ | .33~ | .05~ | .33~ | .03~ | .03 | .15~ |
| E4 n4000 | .07~ | .17~ | .17~ | - | - | - | - | .35~ | .95~ | .15~ | .15~ | .82 | .17~ | .82 | .17~ | .82 | .03~ | .05~ | .30 |
| E4 n8000 | .03~ | .35~ | .38~ | - | - | - | - | .60~ | 1.00~ | .40~ | .40~ | 1.00~ | .40~ | 1.00~ | .38~ | 1.00~ | .03 | .03 | .50 |
| E4 n24000 | .05 | .90 | .88 | - | - | - | - | 1.00~ | 1.00~ | .90 | .90 | 1.00~ | .90 | 1.00~ | .90 | 1.00~ | .10~ | .07~ | .75~ |
| E5 n500 | .68~ | .42~ | .47~ | - | - | 1.00* | .99* | .47 | .45 | .48~ | .47~ | .48* | .38~ | .47* | .37~ | .56* | .68* | .66* | .68* |
| E5 n1000 | .68 | .47~ | .51 | - | - | 1.00* | 1.00* | .50 | .45 | .55~ | .54~ | .56* | .40~ | .56* | .40~ | .62* | .66* | .67* | .68* |
| E5 n4000 | .76 | .56~ | .60 | - | - | - | - | .53 | .00 | .61 | .60 | .71* | .48 | .71* | .47 | .76* | .74* | .71* | .77* |
| E5 n8000 | .80~ | .60~ | .62~ | - | - | - | - | .57 | .00 | .64 | .64 | .78* | .53~ | .78* | .53~ | .82* | .80* | .77* | .85* |
| E5 n24000 | .93~ | .64~ | .70~ | - | - | - | - | .65 | .00 | .74~ | .72~ | .85* | .63~ | .85* | .63~ | .89* | .94* | .93* | .95* |

### V6 E4: placebo_conf rate (raw p, else declared) / wrong-sign rate of the true edge (* INVALID)

| row | pmrt_nl_eq | pmrt_eq | pmrt_r3 | cdl | corr | mscr_eq | mscr_native | notears | pc_eq | pc_native | pcorr_eq | pcorr_eq_min | pcorr_hac | pcorr_hac_eq_min | pcorr_hac_fb | pcorr_hac_fb_eq_min | pcorr_native | rcot2_eq | rcot2_eq_min | rcot2_native | shap_dag | two_tower |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 l1 n1000 | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00 | -/.00 | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ |
| R1 l1 n4000 | -/.00 | -/.00 | -/.00 | -/.00~ | -/.00 | - | - | -/.00~ | -/.00~ | -/.00~ | -/.00 | -/.00 | -/.00 | -/.00 | -/.00 | -/.00 | -/.00 | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ |
| R1 l1 n24000 | -/.00~ | -/.00 | -/.00 | - | -/.00 | - | - | -/.00~ | -/.00~ | -/.00~ | -/.00 | -/.00 | -/.00 | -/.00 | -/.00 | -/.00 | -/.00 | -/.00 | -/.00 | -/.00 | -/.00~ | -/.00~ |
| R2 l1 n1000 | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.05~ | -/.00 | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.03~ | -/.03 | -/.00~ | -/.00~ | -/.00~ |
| R2 l1 n4000 | -/.03~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | - | - | -/.00~ | -/.03~ | -/.00~ | -/.00~ | -/.00~ | -/.00 | -/.00~ | -/.00 | -/.00~ | -/.00 | -/.00~ | -/.00~ | -/.00 | -/.00~ | -/.00~ |
| R2 l1 n24000 | -/.00 | -/.00 | -/.00 | - | -/.00* | - | - | -/.00~ | -/.00~ | -/.00~ | -/.00 | -/.00 | -/.00~ | -/.00 | -/.00~ | -/.00 | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ |
| R3 l0 n1000 | .05/.00~ | .05/.00~ | .05/.00~ | - | .05/.00~ | .04/.00~ | .03/.00~ | .00/.00 | .03/.00 | .00/.00 | .05/.00 | .03/.00~ | .03/.00~ | .03/.00~ | .03/.00~ | .03/.00~ | .03/.00~ | .08/.00* | .05/.00~ | .07/.00~ | .00/.00 | .00/.00 |
| R3 l0 n4000 | .05/.00~ | .06/.00~ | .06/.00~ | - | .03/.00~ | - | - | .00/.00 | .00/.00 | .00/.00 | .06/.00~ | .00/.00 | .00/.00 | .00/.00 | .00/.00 | .00/.00 | .00/.00 | .07/.00~ | .00/.00 | .00/.00 | .00/.00 | .00/.00 |
| R3 l0 n24000 | .03/.00~ | .04/.00~ | .04/.00~ | - | .07/.00~ | - | - | .00/.00 | .06/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00 | .07/.00~ | .10/.00~ | .00/.00 | .03/.00 |
| R3 l0.5 n1000 | .04/.00 | .05/.00~ | .04/.00~ | - | 1.00/1.00* | 1.00/.00* | 1.00/.00* | .00/.00 | .04/.00 | .12/.00~ | .05/.00~ | .07/.00~ | .07/.00~ | .07/.00~ | .07/.00~ | .07/.00~ | .07/.00~ | .53/.00* | .07/.00~ | .10/.00~ | 1.00/.00* | 1.00/.00* |
| R3 l0.5 n4000 | .04/.00~ | .06/.00~ | .07/.00~ | - | 1.00/1.00* | - | - | .00/.00 | .10/.00* | .12/.00~ | .07/.00~ | .07/.00~ | .10/.00~ | .10/.00~ | .10/.00~ | .10/.00~ | .07/.00~ | .92/.00* | .15/.00~ | .12/.00~ | 1.00/.00* | 1.00/.00* |
| R3 l0.5 n24000 | .04/.00~ | .06/.00~ | .07/.00~ | - | 1.00/1.00* | - | - | .00/.00 | .20/.00* | .17/.00* | .06/.00~ | .12/.00~ | .12/.00~ | .12/.00~ | .12/.00~ | .12/.00~ | .12/.00~ | 1.00/.00* | .35/.00* | .40/.00* | 1.00/.00* | 1.00/.00* |
| R3 l1 n1000 | .04/.00~ | .04/.00~ | .05/.00~ | - | 1.00/1.00* | 1.00/.00* | 1.00/.00* | .00/.00 | .05/.00~ | .07/.00~ | .07/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .93/.00* | .15/.00~ | .12/.00~ | 1.00/.00* | 1.00/.00* |
| R3 l1 n4000 | .02/.00 | .05/.00~ | .06/.00~ | - | 1.00/1.00* | - | - | .00/.00 | .14/.00* | .10/.00~ | .06/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .98/.00* | .38/.00* | .45/.00* | 1.00/.00* | 1.00/.00* |
| R3 l1 n24000 | .04/.00~ | .05/.00* | .05/.00* | - | 1.00/1.00* | - | - | .00/.00 | .12/.00* | .15/.00~ | .05/.00* | .10/.00~ | .10/.00~ | .10/.00~ | .10/.00~ | .10/.00~ | .10/.00~ | 1.00/.00* | .70/.00* | .75/.00* | 1.00/.00* | 1.00/.00* |
| R3 l1.5 n1000 | .05/.00~ | .06/.00~ | .05/.00~ | - | 1.00/1.00* | 1.00/.00* | 1.00/.00* | .00/.00 | .03/.00 | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .92/.00* | .10/.00~ | .17/.00* | 1.00/.00* | 1.00/.00* |
| R3 l1.5 n4000 | .05/.00~ | .04/.00 | .05/.00~ | - | 1.00/1.00* | - | - | .00/.00 | .04/.00 | .07/.00~ | .07/.00~ | .07/.00~ | .07/.00~ | .07/.00~ | .07/.00~ | .07/.00~ | .07/.00~ | .99/.00* | .30/.00* | .30/.00* | 1.00/.00* | 1.00/.00* |
| R3 l1.5 n24000 | .03/.00~ | .03/.00~ | .03/.00~ | - | 1.00/1.00* | - | - | .00/.00 | .13/.00* | .15/.00~ | .06/.00~ | .07/.00~ | .07/.00~ | .07/.00~ | .07/.00~ | .07/.00~ | .07/.00~ | 1.00/.00* | .47/.00* | .65/.00* | 1.00/.00* | 1.00/.00* |
| R4 l0 n1000 | .00/NA | .00/NA~ | .00/NA~ | - | .03/.00 | .05/.00~ | .05/.00~ | .00/.00 | .10/.00~ | .10/.00~ | .00/.00~ | .03/.00 | .03/.00 | .03/.00 | .03/.00 | .03/.00 | .03/.00 | .03/.00~ | .03/.00 | .05/.00~ | .00/.00 | .00/.00 |
| R4 l0 n4000 | .00/NA~ | .00/NA~ | .00/NA~ | - | .07/.00~ | - | - | .00/.00 | .03/.00 | .03/.00 | .07/.00~ | .05/.00~ | .07/.00~ | .07/.00~ | .07/.00~ | .07/.00~ | .05/.00~ | .07/.00~ | .05/.00~ | .05/.00~ | .00/.00 | .00/.00 |
| R4 l0 n24000 | .00/NA~ | .00/NA~ | .00/NA~ | - | .03/.00~ | - | - | .00/.00 | .03/.00 | .03/.00 | .03/.00~ | .03/.00~ | .03/.00~ | .03/.00~ | .03/.00~ | .03/.00~ | .03/.00~ | .10/.00~ | .07/.00~ | .07/.00~ | .00/.00 | .03/.00 |
| R4 l0.5 n1000 | .00/NA~ | .00/NA~ | .00/NA~ | - | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* |
| R4 l0.5 n4000 | .00/NA~ | .00/NA~ | .00/NA~ | - | 1.00/1.00* | - | - | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* |
| R4 l0.5 n24000 | .00/NA~ | .00/NA~ | .00/NA~ | - | 1.00/1.00* | - | - | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* |
| R4 l1 n1000 | .00/NA | .00/NA~ | .00/NA~ | - | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* |
| R4 l1 n4000 | .00/NA~ | .00/NA~ | .00/NA~ | - | 1.00/1.00* | - | - | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* |
| R4 l1 n24000 | .00/NA~ | .00/NA~ | .00/NA~ | - | 1.00/1.00* | - | - | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* |
| R4 l1.5 n1000 | .00/NA~ | .00/NA~ | .00/NA~ | - | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* |
| R4 l1.5 n4000 | .00/NA~ | .00/NA~ | .00/NA~ | - | 1.00/1.00* | - | - | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* |
| R4 l1.5 n24000 | .00/NA~ | .00/NA~ | .00/NA~ | - | 1.00/1.00* | - | - | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* |

### V7 not testable / not applicable

60 (arm, cell) with counts > 0 (JSON V7).

### V8 secondary KPI -> KPI family

632 (arm, cell) rows (JSON V8).

### V9 kappa sweep (R2, n 1000): recall

| row | pmrt_nl_eq | pmrt_eq | pmrt_r3 | cdl | corr | granger_eq | granger_native | mscr_eq | mscr_native | notears | pc_eq | pc_native | pcorr_eq | pcorr_eq_min | pcorr_hac | pcorr_hac_eq_min | pcorr_hac_fb | pcorr_hac_fb_eq_min | pcorr_native | rcot2_eq | rcot2_eq_min | rcot2_native | shap_dag | two_tower |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 k0.125 | 1.00 | .95~ | 1.00 | - | 1.00* | - | - | - | - | 1.00~ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00* | 1.00 | 1.00* | 1.00 | 1.00* | 1.00* | 1.00* | 1.00* | 1.00* | .97* |
| E1 k0.25 | 1.00~ | .97 | 1.00 | 1.00* | 1.00* | - | - | 1.00* | 1.00* | 1.00* | 1.00~ | 1.00 | 1.00~ | 1.00~ | 1.00* | 1.00~ | 1.00* | 1.00~ | 1.00* | 1.00* | 1.00~ | 1.00* | 1.00* | .97* |
| E1 k0.5 | 1.00~ | .96~ | 1.00~ | - | 1.00* | - | - | - | - | 1.00* | 1.00 | 1.00* | 1.00 | 1.00 | 1.00* | 1.00 | 1.00* | 1.00 | 1.00* | .90~ | .85* | .99* | 1.00* | .95* |
| E2 k0.125 | .71 | .29 | .35~ | - | .68* | - | - | - | - | .31 | .28 | .25 | .40~ | .41~ | .50* | .31~ | .50* | .31~ | .52* | .50* | .53* | .55* | .90 | .42 |
| E2 k0.25 | .66~ | .28 | .33 | .49 | .65* | - | - | .97* | .97* | .29 | .30* | .29 | .37 | .38 | .49* | .28~ | .49* | .28~ | .52* | .51* | .51* | .54* | .91 | .42 |
| E2 k0.5 | .61 | .27 | .32~ | - | .63* | - | - | - | - | .31 | .31 | .35 | .36 | .36~ | .49* | .29~ | .49* | .29~ | .50* | .46* | .47* | .53* | .89 | .51 |
| E3 k0.125 | 1.00~ | 1.00~ | 1.00~ | - | 1.00* | 1.00~ | 1.00* | - | - | 1.00* | 1.00 | 1.00* | 1.00~ | 1.00~ | 1.00* | 1.00~ | 1.00* | 1.00~ | 1.00* | .96* | .95~ | .97* | 1.00* | 1.00* |
| E3 k0.25 | 1.00~ | 1.00~ | 1.00~ | 1.00* | 1.00* | 1.00~ | 1.00* | 1.00* | 1.00* | 1.00* | 1.00 | 1.00* | 1.00~ | 1.00 | 1.00* | 1.00 | 1.00* | 1.00 | 1.00* | .96* | .94* | .99* | 1.00* | .99* |
| E3 k0.5 | .99~ | 1.00~ | 1.00~ | - | 1.00* | 1.00~ | 1.00* | - | - | 1.00* | 1.00 | 1.00* | 1.00~ | 1.00~ | 1.00* | 1.00~ | 1.00* | 1.00~ | 1.00* | .79* | .76~ | .99* | 1.00* | 1.00* |
| E4 k0.125 | .05~ | .10~ | .05~ | - | .40~ | - | - | - | - | .25~ | .05~ | .50~ | .10~ | .10~ | .40~ | .10~ | .40~ | .10~ | .40~ | .05~ | .05~ | .15~ | .00~ | .15~ |
| E4 k0.25 | .00~ | .03~ | .03~ | .17~ | .33~ | - | - | .05~ | .05 | .12~ | .07~ | .47~ | .05~ | .05~ | .33~ | .05~ | .33~ | .05~ | .33~ | .03~ | .03 | .15~ | .03~ | .03~ |
| E4 k0.5 | .00~ | .05~ | .05~ | - | .45~ | - | - | - | - | .15~ | .20~ | .50~ | .10~ | .10~ | .40~ | .10~ | .40~ | .10~ | .40~ | .05~ | .05 | .10~ | .10~ | .00~ |
| E5 k0.125 | .71~ | .51~ | .54~ | - | .77* | - | - | - | - | .47~ | .57* | .47 | .57~ | .58 | .55* | .41 | .55* | .41 | .61* | .67* | .67* | .70* | .75 | .72 |
| E5 k0.25 | .68 | .47~ | .51 | .70 | .78* | - | - | 1.00* | 1.00* | .45 | .50 | .45 | .55~ | .54~ | .56* | .40~ | .56* | .40~ | .62* | .66* | .67* | .68* | .67 | .68 |
| E5 k0.5 | .67~ | .49~ | .51 | - | .74* | - | - | - | - | .43 | .48 | .43 | .54 | .53 | .55* | .42 | .55* | .42 | .58* | .66* | .65~ | .68* | .67 | .67 |

### V10 cost (CPU-s per dataset; mean / max over worlds, regimes; peak RSS MB; infeasible units; T3 = infeasible from DEV cost)

| arm | n 500 | n 1000 | n 4000 | n 8000 | n 24000 | RSS | infeasible |
|---|---|---|---|---|---|---|---|
| pmrt_nl_eq | 15.8 / 67.0 | 23.8 / 96.7 | 66.8 / 307.5 | 125.1 / 569.4 | 352.1 / 1380.0 | 835 | 0 |
| pmrt_eq | 0.4 / 2.8 | 0.7 / 2.3 | 2.4 / 8.0 | 5.6 / 19.8 | 17.1 / 69.8 | 746 | 0 |
| pmrt_r3 | 0.4 / 1.5 | 0.7 / 2.4 | 2.4 / 8.0 | 5.6 / 18.6 | 17.1 / 59.6 | 746 | 0 |
| cdl | 20.5 / 54.8 | 39.2 / 109.2 | 144.1 / 422.1 | - | - | 466 | 0 |
| corr | 0.1 / 0.2 | 0.1 / 0.3 | 0.1 / 0.2 | 0.1 / 0.2 | 0.1 / 0.5 | 504 | 0 |
| granger_eq | 0.1 / 0.3 | 0.2 / 0.4 | 0.5 / 1.1 | 1.0 / 2.1 | 4.5 / 9.9 | 368 | 0 |
| granger_native | 0.1 / 0.1 | 0.1 / 0.1 | 0.1 / 0.1 | 0.1 / 0.2 | 0.3 / 0.4 | 344 | 0 |
| mscr_eq | 17.8 / 85.9 | 39.7 / 195.1 | - | - | - | 454 | 0 |
| mscr_native | 7.4 / 30.7 | 16.2 / 69.6 | - | - | - | 366 | 0 |
| notears | 0.2 / 1.4 | 0.3 / 1.9 | 0.5 / 4.1 | 0.8 / 5.7 | 2.7 / 28.4 | 504 | 0 |
| pc_eq | 1.2 / 12.2 | 1.6 / 18.0 | 2.3 / 27.2 | 3.3 / 35.2 | 7.0 / 92.8 | 506 | 0 |
| pc_native | 0.2 / 0.9 | 0.2 / 1.5 | 0.3 / 4.3 | 0.5 / 5.8 | 1.2 / 10.9 | 506 | 0 |
| pcorr_eq | 0.0 / 0.2 | 0.0 / 0.3 | 0.1 / 0.9 | 0.3 / 1.6 | 0.9 / 7.2 | 503 | 0 |
| pcorr_eq_min | 0.0 / 0.2 | 0.0 / 0.4 | 0.1 / 0.7 | 0.2 / 1.4 | 0.6 / 5.9 | 503 | 0 |
| pcorr_hac | 0.1 / 0.3 | 0.1 / 0.6 | 0.2 / 1.2 | 0.4 / 2.1 | 1.3 / 8.9 | 507 | 0 |
| pcorr_hac_eq_min | 0.1 / 0.4 | 0.1 / 0.6 | 0.2 / 1.2 | 0.4 / 2.1 | 1.4 / 9.7 | 507 | 0 |
| pcorr_hac_fb | 0.8 / 1.6 | 0.8 / 1.8 | 0.9 / 2.4 | 1.1 / 3.4 | 1.9 / 9.0 | 530 | 0 |
| pcorr_hac_fb_eq_min | 0.8 / 1.7 | 0.8 / 1.8 | 0.9 / 2.4 | 1.1 / 3.6 | 2.0 / 9.5 | 530 | 0 |
| pcorr_native | 0.0 / 0.1 | 0.0 / 0.1 | 0.0 / 0.1 | 0.1 / 0.3 | 0.2 / 1.1 | 503 | 0 |
| rcot2_eq | 0.5 / 2.0 | 0.5 / 2.4 | 1.2 / 6.4 | 2.4 / 10.0 | 6.7 / 29.5 | 506 | 0 |
| rcot2_eq_min | 0.4 / 1.8 | 0.5 / 2.3 | 1.1 / 5.2 | 2.2 / 8.8 | 6.0 / 25.7 | 506 | 0 |
| rcot2_native | 0.4 / 1.7 | 0.5 / 2.1 | 1.1 / 5.1 | 2.2 / 9.0 | 6.0 / 25.9 | 506 | 0 |
| shap_dag | 1.5 / 4.9 | 2.6 / 7.7 | 8.9 / 25.5 | 18.5 / 49.1 | 57.6 / 159.6 | 524 | 0 |
| two_tower | 4.7 / 10.4 | 5.6 / 13.7 | 11.7 / 31.8 | 21.2 / 64.0 | 60.5 / 208.4 | 662 | 0 |

### V11 integrity

- label FINAL; failed checks: none
- freeze commit 93856c230d84abda2ff3b894218dd6cb0ecd47f9; amendments ['A-1']; records used 150960 / expected 150960; missing 0; unexpected 0; role mismatch 0; duplicates 0
- commit violations 0; not clean 0; stamp violations 0; candidates incomplete 0; pkgs sets 1
- role:status {'measure:ok': 110440, 'tune:ok': 40520}; errors 0 (not listed persistent 0); infeasible 0; dataset-hash mismatches 0 (R-41a: equal across all arms of a dataset); ok records without a dataset hash 0 (not-ok 0); multi-platform datasets 0
- BY recheck (p arms): 114960 / 114960 records agree; tune reproducibility vs DEV: {'compared': 39000, 'dataset_sha_equal': 38992, 'declarations_equal': 39000}
