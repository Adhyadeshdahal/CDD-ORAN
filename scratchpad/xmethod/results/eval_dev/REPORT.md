# [DEV, NOT EVAL] Study A report rehearsal

> **DEV DATA, NOT EVAL: development seeds 3_000_000-3_000_199 on the DEV specs; a rehearsal of the report, not the pre-registered result**

Generator `xm-eval-report/1` on `xm-eval-analysis/5`; spec `DEV:dev_full+dev_ci_c+dev_pmrt_nl` (sha256 e1a7e846de563e16); protocol docs/xmethod/PROTOCOL_A.md (NOT verified: frozen False, spec protocol_sha256 TBD-at-freeze vs file 5acfbcff1226e180); integrity **PROVISIONAL** (failed: clean, commits, dataset_hash, freeze_commit_given, one_platform_per_dataset, pkgs_uniform, protocol_frozen_sha, stamps, unexpected).
Primary PMRT arm (focal): pmrt_nl_eq; primary kappa 0.25; generated 2026-10-04T06:15:24Z.

Inputs:

- `D:/academia/major-project/CDD-ORAN-wt/xm-harness/scratchpad/xmethod/specs/eval/full.json`: sha256 81ec2718e17c9ae4
- `D:/academia/major-project/CDD-ORAN-wt/xm-pmrt/scratchpad/xmethod/specs/dev/full.json`: sha256 5ed77b4d3c7edeeb
- `D:/academia/major-project/CDD-ORAN-wt/xm-pmrt/scratchpad/xmethod/specs/dev/ci_c.json`: sha256 4bce54235d15c5dd
- `D:/academia/major-project/CDD-ORAN-wt/xm-pmrt/scratchpad/xmethod/specs/dev/pmrt_nl.json`: sha256 68e18a32ef86e0e1
- `D:/academia/major-project/CDD-ORAN-wt/xm-pmrt/scratchpad/xmethod/results/dev/full/merged.jsonl.gz`: sha256 6a74cb85e3ebb1f6
- `D:/academia/major-project/CDD-ORAN-wt/xm-pmrt/scratchpad/xmethod/results/dev/ci_c/merged.jsonl.gz`: sha256 1b0484798f5dfd2f
- `D:/academia/major-project/CDD-ORAN-wt/xm-pmrt/scratchpad/xmethod/results/dev/pmrt_nl/merged.jsonl.gz`: sha256 d8894fd3a1dee29f
- records by arm: corr 5080, granger_eq 600, granger_native 600, mscr_eq 2560, mscr_eq_min 2560, mscr_native 2560, notears 5080, pc_eq 5080, pc_native 5080, pcorr_eq 5080, pcorr_eq_min 5080, pcorr_hac 5080, pcorr_hac_eq_min 5080, pcorr_hac_fb 5080, pcorr_hac_fb_eq_min 5080, pcorr_native 5080, pmrt_eq 5080, pmrt_nl_eq 1440, pmrt_r3 5080, rcot2_eq 5080, rcot2_eq_min 5080, rcot2_native 5080, shap_dag 5080, two_tower 5080
- note: DEV arm mscr_eq_min is not in the EVAL spec: left out
- note: EVAL arms without DEV records (absent from this rendering): cdl

## 1. Claim (PROTOCOL_A s.10)

> **DEV DATA, NOT EVAL: development seeds 3_000_000-3_000_199 on the DEV specs; a rehearsal of the report, not the pre-registered result**

**Claim: NOT SUPPORTED**

| component | verdict | wording |
|---|---|---|
| C1 design-blind tests invalid on R2 | SUPPORTED | design-blind tests are invalid on the R2 (setpoint + dither) design tested |
| C2a design-based test valid (R1 + R2) | SUPPORTED | using the design restores validity (design-based inference and design-covariate adjustment); rcot2_eq INVALID in R2 as already in R1 (not restored by design covariates) |
| C2b design covariates restore validity | SUPPORTED | (C2 wording above) |
| C3 valid under the logged confounded policy | NOT EVALUABLE | not supported |

## 2. C1: design-blind arms (set D)

> **DEV DATA, NOT EVAL: development seeds 3_000_000-3_000_199 on the DEV specs; a rehearsal of the report, not the pre-registered result**

**SUPPORTED**: 5 of 6 assessable arms are FAILURES (|D| = 6; >= 3 assessable needed).

| arm | in D | verdict | R2 INVALID / planned (counted) | pooled R2 truth-null raw | pooled R2 placebo raw | R1 INVALID / counted (F_max) |
|---|---|---|---|---|---|---|
| corr | yes | FAILURE | 21/25 (25) | .750 [.734, .766] INVALID | .709 [.683, .734] INVALID | 1/25 (2) |
| granger_native | yes | FAILURE | 5/5 (5) | .640 [.599, .676] INVALID | .516 [.444, .580] INVALID | 1/5 (1) |
| mscr_native [single-conditioner max statistic; reported INVALID arm, EVAL n <= 1000 (R-54)] | yes | FAILURE | 8/10 (10) | .945 [.935, .955] INVALID | .908 [.892, .923] INVALID | 0/10 (1) |
| pcorr_hac_fb | yes | FAILURE | 20/25 (25) | .425 [.399, .448] INVALID | .297 [.264, .329] INVALID | 0/25 (2) |
| pcorr_native | yes | FAILURE | 20/25 (25) | .424 [.399, .447] INVALID | .295 [.261, .327] INVALID | 0/25 (2) |
| rcot2_native | yes | INVALID IN R1 | 20/25 (25) | .252 [.238, .265] INVALID | .238 [.219, .256] INVALID | 7/25 (2) |
| pcorr_hac | no (descriptive) | FAILURE | 20/25 (25) | .426 [.400, .449] INVALID | .297 [.264, .330] INVALID | 0/25 (2) |

C1 sensitivity (R-56, no effect): without mscr_native: **SUPPORTED** (4 of 5 assessable arms fail).

## 3. C2a: the design-based test (R1 + R2)

> **DEV DATA, NOT EVAL: development seeds 3_000_000-3_000_199 on the DEV specs; a rehearsal of the report, not the pre-registered result**

| arm | verdict | cells counted / planned | INVALID (F_max) | pooled rates |
|---|---|---|---|---|
| pmrt_nl_eq (primary) | SUPPORTED | 24/24 | 0 (2) | null_raw .043 [.040, .047] VALID; plac_raw .046 [.039, .053] VALID |
| pmrt_eq [linear statistic (pmrt-core-v1); secondary PMRT arm (R-42)] (secondary PMRT arm, no effect) | SUPPORTED | 50/50 | 0 (3) | null_raw .048 [.044, .052] VALID; plac_raw .048 [.042, .054] VALID |
| pmrt_r3 (secondary PMRT arm, no effect) | SUPPORTED | 50/50 | 1 (3) | null_raw .050 [.046, .053] VALID; plac_raw .049 [.043, .056] VALID |

## 4. C2b: design-covariate adjustment (eq arms whose native partner is a C1 FAILURE)

> **DEV DATA, NOT EVAL: development seeds 3_000_000-3_000_199 on the DEV specs; a rehearsal of the report, not the pre-registered result**

**SUPPORTED** (membership rule as frozen); with the eq arms whose native partner is INVALID IN R1 but meets C1 legs (i)-(ii) added (R-56 sensitivity, no effect): **SUPPORTED** (added: rcot2_eq).

| arm | verdict | cells counted / planned | INVALID (F_max) | pooled rates |
|---|---|---|---|---|
| granger_eq (native granger_native) | SUPPORTED | 10/10 | 0 (1) | null_raw .045 [.038, .052] VALID; plac_raw .046 [.034, .057] VALID |
| pcorr_eq (native pcorr_native) | SUPPORTED | 50/50 | 0 (3) | null_raw .049 [.046, .053] VALID; plac_raw .047 [.041, .053] VALID |
| mscr_eq [single-conditioner max statistic; cannot condition on the joint design set; reported INVALID arm, EVAL n <= 1000 (R-54)] EXCLUDED (R-40), no effect | NOT SUPPORTED | 20/20 | 8 (2) | null_raw .730 [.695, .767] INVALID; plac_raw .707 [.677, .737] INVALID |
| rcot2_eq (R-56 sensitivity member) | NOT SUPPORTED | 50/50 | 27 (3) | null_raw .164 [.158, .169] INVALID; plac_raw .159 [.148, .169] INVALID |

C2 wording: using the design restores validity (design-based inference and design-covariate adjustment); rcot2_eq INVALID in R2 as already in R1 (not restored by design covariates).

Pre-registered disclosure (R-56; named eq arms):
- rcot2_eq is INVALID in R2 (pooled truth-null raw rate 0.208 [0.196, 0.220]) as already in R1 (0.067 [0.060, 0.073]): the test is miscalibrated without the design (its native partner rcot2_native is INVALID IN R1), so it is outside C2b's membership rule, and adding design covariates does not make it valid.

Every eq arm, same rule on R1 + R2, unfiltered (R-56):

| eq arm | C2b membership | native C1 | verdict | INVALID / cells (F_max) | R2 INVALID / planned | pooled R1 truth-null raw | pooled R2 truth-null raw |
|---|---|---|---|---|---|---|---|
| granger_eq | member | FAILURE | SUPPORTED | 0/10 (1) | 0/5 | .043 [.033, .052] VALID | .046 [.037, .055] VALID |
| mscr_eq [single-conditioner max statistic; cannot condition on the joint design set; reported INVALID arm, EVAL n <= 1000 (R-54)] | excluded (R-40) | FAILURE | NOT SUPPORTED | 8/20 (2) | 8/10 | .048 [.039, .056] VALID | .958 [.950, .967] INVALID |
| pc_eq | not a member (declare tau) | - | NOT SUPPORTED | 3/50 (2) | - | - | - |
| pcorr_eq | member | FAILURE | SUPPORTED | 0/50 (3) | 0/25 | .051 [.044, .059] VALID | .048 [.045, .052] VALID |
| rcot2_eq | not a member (native rcot2_native: INVALID IN R1) | INVALID IN R1 | NOT SUPPORTED | 27/50 (3) | 20/25 | .067 [.060, .073] INVALID | .208 [.196, .220] INVALID |

## 5. C3: logged confounded policy (E4 R3; C3 readers at S_E4)

> **DEV DATA, NOT EVAL: development seeds 3_000_000-3_000_199 on the DEV specs; a rehearsal of the report, not the pre-registered result**

| arm | verdict | cells counted / planned | INVALID (F_max) | pooled rates |
|---|---|---|---|---|
| pmrt_nl_eq (primary) | NOT EVALUABLE | 0/0 | 0 (0) | plac_raw -; conf_raw - |
| granger_eq (eq arm, no claim effect) | NOT EVALUABLE | 0/0 | 0 (0) | plac_raw -; conf_raw - |
| mscr_eq [single-conditioner max statistic; cannot condition on the joint design set; reported INVALID arm, EVAL n <= 1000 (R-54)] (eq arm, no claim effect) | NOT SUPPORTED | 8/8 | 6 (0) | plac_raw .115 [.079, .154] INVALID; conf_raw .758 [.752, .767] INVALID |
| pc_eq (eq arm, no claim effect) | NOT SUPPORTED | 20/20 | 2 (0) | conf_decl .064 [.043, .088] INCONCLUSIVE |
| pcorr_eq (eq arm, no claim effect) | NOT SUPPORTED | 20/20 | 0 (1) | plac_raw .052 [.023, .088] INCONCLUSIVE; conf_raw .040 [.020, .065] VALID |
| rcot2_eq (eq arm, no claim effect) | NOT SUPPORTED | 20/20 | 15 (1) | plac_raw .043 [.019, .073] VALID; conf_raw .658 [.620, .691] INVALID |
| pmrt_eq [linear statistic (pmrt-core-v1); secondary PMRT arm (R-42)] (secondary PMRT arm, no effect) | NOT SUPPORTED | 20/20 | 0 (1) | plac_raw .045 [.018, .082] INCONCLUSIVE; conf_raw .039 [.019, .062] VALID |
| pmrt_r3 (secondary PMRT arm, no effect) | NOT SUPPORTED | 20/20 | 0 (1) | plac_raw .048 [.020, .083] INCONCLUSIVE; conf_raw .040 [.019, .064] VALID |

## 6. Power among cells not INVALID (V2) and the R4 'not applicable' rows

> **DEV DATA, NOT EVAL: development seeds 3_000_000-3_000_199 on the DEV specs; a rehearsal of the report, not the pre-registered result**

1118 (arm, cell) with recall (counted, not INVALID; R-39); the grid is in the appendix (V2) and `csv/recall_vs_n.csv`. PMRT arms in R4 (no known design): power NOT APPLICABLE, never recall 0 (R-42); their placebo rates are still read:

| arm | cell | state | placebo raw (P_placebo) |
|---|---|---|---|
| pmrt_eq | E4 R4 lam0.5 k0.25 n1000 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_eq | E4 R4 lam0.5 k0.25 n24000 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_eq | E4 R4 lam0.5 k0.25 n4000 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_eq | E4 R4 lam0.5 k0.25 n500 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_eq | E4 R4 lam0.5 k0.25 n8000 | NA (INCONCLUSIVE) | .100 [.000, .250] INCONCLUSIVE |
| pmrt_eq | E4 R4 lam0 k0.25 n1000 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_eq | E4 R4 lam0 k0.25 n24000 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_eq | E4 R4 lam0 k0.25 n4000 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_eq | E4 R4 lam0 k0.25 n500 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_eq | E4 R4 lam0 k0.25 n8000 | NA (INCONCLUSIVE) | .100 [.000, .250] INCONCLUSIVE |
| pmrt_eq | E4 R4 lam1.5 k0.25 n1000 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_eq | E4 R4 lam1.5 k0.25 n24000 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_eq | E4 R4 lam1.5 k0.25 n4000 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_eq | E4 R4 lam1.5 k0.25 n500 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_eq | E4 R4 lam1.5 k0.25 n8000 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_eq | E4 R4 lam1 k0.25 n1000 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_eq | E4 R4 lam1 k0.25 n24000 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_eq | E4 R4 lam1 k0.25 n4000 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_eq | E4 R4 lam1 k0.25 n500 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_eq | E4 R4 lam1 k0.25 n8000 | NA (INCONCLUSIVE) | .050 [.000, .150] INCONCLUSIVE |
| pmrt_r3 | E4 R4 lam0.5 k0.25 n1000 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_r3 | E4 R4 lam0.5 k0.25 n24000 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_r3 | E4 R4 lam0.5 k0.25 n4000 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_r3 | E4 R4 lam0.5 k0.25 n500 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_r3 | E4 R4 lam0.5 k0.25 n8000 | NA (INCONCLUSIVE) | .100 [.000, .250] INCONCLUSIVE |
| pmrt_r3 | E4 R4 lam0 k0.25 n1000 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_r3 | E4 R4 lam0 k0.25 n24000 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_r3 | E4 R4 lam0 k0.25 n4000 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_r3 | E4 R4 lam0 k0.25 n500 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_r3 | E4 R4 lam0 k0.25 n8000 | NA (INCONCLUSIVE) | .100 [.000, .250] INCONCLUSIVE |
| pmrt_r3 | E4 R4 lam1.5 k0.25 n1000 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_r3 | E4 R4 lam1.5 k0.25 n24000 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_r3 | E4 R4 lam1.5 k0.25 n4000 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_r3 | E4 R4 lam1.5 k0.25 n500 | NA (INCONCLUSIVE) | .050 [.000, .150] INCONCLUSIVE |
| pmrt_r3 | E4 R4 lam1.5 k0.25 n8000 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_r3 | E4 R4 lam1 k0.25 n1000 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_r3 | E4 R4 lam1 k0.25 n24000 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_r3 | E4 R4 lam1 k0.25 n4000 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_r3 | E4 R4 lam1 k0.25 n500 | NA (VALID) | .000 [.000, .000] VALID |
| pmrt_r3 | E4 R4 lam1 k0.25 n8000 | NA (VALID) | .000 [.000, .000] VALID |

## 7. kappa sweep (R2, n 1000; V9)

> **DEV DATA, NOT EVAL: development seeds 3_000_000-3_000_199 on the DEV specs; a rehearsal of the report, not the pre-registered result**

| arm | world | kappa | status | validity | recall |
|---|---|---|---|---|---|
| corr | E1 | 0.125 | ok | INVALID | 1.00 |
| corr | E1 | 0.25 | ok | INVALID | 1.00 |
| corr | E1 | 0.5 | ok | INVALID | 1.00 |
| corr | E2 | 0.125 | ok | INVALID | .64 |
| corr | E2 | 0.25 | ok | INVALID | .68 |
| corr | E2 | 0.5 | ok | INVALID | .61 |
| corr | E3 | 0.125 | ok | INVALID | 1.00 |
| corr | E3 | 0.25 | ok | INVALID | 1.00 |
| corr | E3 | 0.5 | ok | INVALID | 1.00 |
| corr | E4 | 0.125 | ok | INCONCLUSIVE | .20 |
| corr | E4 | 0.25 | ok | INCONCLUSIVE | .27 |
| corr | E4 | 0.5 | ok | VALID | .25 |
| corr | E5 | 0.125 | ok | INVALID | .76 |
| corr | E5 | 0.25 | ok | INVALID | .79 |
| corr | E5 | 0.5 | ok | INVALID | .71 |
| granger_eq | E3 | 0.125 | ok | INCONCLUSIVE | 1.00 |
| granger_eq | E3 | 0.25 | ok | VALID | 1.00 |
| granger_eq | E3 | 0.5 | ok | INCONCLUSIVE | 1.00 |
| granger_native | E3 | 0.125 | ok | INVALID | 1.00 |
| granger_native | E3 | 0.25 | ok | INVALID | 1.00 |
| granger_native | E3 | 0.5 | ok | INVALID | 1.00 |
| mscr_eq | E1 | 0.125 | ok | INVALID | 1.00 |
| mscr_eq | E1 | 0.25 | ok | INVALID | 1.00 |
| mscr_eq | E1 | 0.5 | ok | INVALID | 1.00 |
| mscr_eq | E2 | 0.125 | ok | INVALID | .97 |
| mscr_eq | E2 | 0.25 | ok | INVALID | .97 |
| mscr_eq | E2 | 0.5 | ok | INVALID | .97 |
| mscr_eq | E3 | 0.125 | ok | INVALID | 1.00 |
| mscr_eq | E3 | 0.25 | ok | INVALID | 1.00 |
| mscr_eq | E3 | 0.5 | ok | INVALID | 1.00 |
| mscr_eq | E4 | 0.125 | ok | INCONCLUSIVE | .00 |
| mscr_eq | E4 | 0.25 | ok | INCONCLUSIVE | .02 |
| mscr_eq | E4 | 0.5 | ok | INCONCLUSIVE | .00 |
| mscr_eq | E5 | 0.125 | ok | INVALID | 1.00 |
| mscr_eq | E5 | 0.25 | ok | INVALID | 1.00 |
| mscr_eq | E5 | 0.5 | ok | INVALID | 1.00 |
| mscr_native | E1 | 0.125 | ok | INVALID | 1.00 |
| mscr_native | E1 | 0.25 | ok | INVALID | 1.00 |
| mscr_native | E1 | 0.5 | ok | INVALID | 1.00 |
| mscr_native | E2 | 0.125 | ok | INVALID | .96 |
| mscr_native | E2 | 0.25 | ok | INVALID | .97 |
| mscr_native | E2 | 0.5 | ok | INVALID | .95 |
| mscr_native | E3 | 0.125 | ok | INVALID | 1.00 |
| mscr_native | E3 | 0.25 | ok | INVALID | 1.00 |
| mscr_native | E3 | 0.5 | ok | INVALID | 1.00 |
| mscr_native | E4 | 0.125 | ok | INCONCLUSIVE | .00 |
| mscr_native | E4 | 0.25 | ok | INCONCLUSIVE | .10 |
| mscr_native | E4 | 0.5 | ok | INCONCLUSIVE | .00 |
| mscr_native | E5 | 0.125 | ok | INVALID | 1.00 |
| mscr_native | E5 | 0.25 | ok | INVALID | 1.00 |
| mscr_native | E5 | 0.5 | ok | INVALID | 1.00 |
| notears | E1 | 0.125 | ok | INCONCLUSIVE | 1.00 |
| notears | E1 | 0.25 | ok | INVALID | 1.00 |
| notears | E1 | 0.5 | ok | INVALID | 1.00 |
| notears | E2 | 0.125 | ok | VALID | .31 |
| notears | E2 | 0.25 | ok | VALID | .33 |
| notears | E2 | 0.5 | ok | VALID | .31 |
| notears | E3 | 0.125 | ok | INVALID | 1.00 |
| notears | E3 | 0.25 | ok | INVALID | 1.00 |
| notears | E3 | 0.5 | ok | INVALID | 1.00 |
| notears | E4 | 0.125 | ok | NO_READ | .05 |
| notears | E4 | 0.25 | ok | NO_READ | .12 |
| notears | E4 | 0.5 | ok | NO_READ | .05 |
| notears | E5 | 0.125 | ok | VALID | .46 |
| notears | E5 | 0.25 | ok | VALID | .46 |
| notears | E5 | 0.5 | ok | VALID | .46 |
| pc_eq | E1 | 0.125 | ok | INCONCLUSIVE | 1.00 |
| pc_eq | E1 | 0.25 | ok | INVALID | 1.00 |
| pc_eq | E1 | 0.5 | ok | VALID | 1.00 |
| pc_eq | E2 | 0.125 | ok | VALID | .28 |
| pc_eq | E2 | 0.25 | ok | INVALID | .30 |
| pc_eq | E2 | 0.5 | ok | VALID | .30 |
| pc_eq | E3 | 0.125 | ok | VALID | 1.00 |
| pc_eq | E3 | 0.25 | ok | VALID | 1.00 |
| pc_eq | E3 | 0.5 | ok | VALID | 1.00 |
| pc_eq | E4 | 0.125 | ok | NO_READ | .00 |
| pc_eq | E4 | 0.25 | ok | NO_READ | .05 |
| pc_eq | E4 | 0.5 | ok | NO_READ | .00 |
| pc_eq | E5 | 0.125 | ok | INCONCLUSIVE | .51 |
| pc_eq | E5 | 0.25 | ok | VALID | .50 |
| pc_eq | E5 | 0.5 | ok | VALID | .47 |
| pc_native | E1 | 0.125 | ok | VALID | 1.00 |
| pc_native | E1 | 0.25 | ok | VALID | 1.00 |
| pc_native | E1 | 0.5 | ok | INVALID | 1.00 |
| pc_native | E2 | 0.125 | ok | VALID | .24 |
| pc_native | E2 | 0.25 | ok | VALID | .31 |
| pc_native | E2 | 0.5 | ok | VALID | .35 |
| pc_native | E3 | 0.125 | ok | INVALID | 1.00 |
| pc_native | E3 | 0.25 | ok | INVALID | 1.00 |
| pc_native | E3 | 0.5 | ok | INVALID | 1.00 |
| pc_native | E4 | 0.125 | ok | NO_READ | .50 |
| pc_native | E4 | 0.25 | ok | NO_READ | .50 |
| pc_native | E4 | 0.5 | ok | NO_READ | .40 |
| pc_native | E5 | 0.125 | ok | VALID | .46 |
| pc_native | E5 | 0.25 | ok | VALID | .47 |
| pc_native | E5 | 0.5 | ok | VALID | .43 |
| pcorr_eq | E1 | 0.125 | ok | VALID | 1.00 |
| pcorr_eq | E1 | 0.25 | ok | INCONCLUSIVE | 1.00 |
| pcorr_eq | E1 | 0.5 | ok | VALID | 1.00 |
| pcorr_eq | E2 | 0.125 | ok | INCONCLUSIVE | .35 |
| pcorr_eq | E2 | 0.25 | ok | INCONCLUSIVE | .36 |
| pcorr_eq | E2 | 0.5 | ok | INCONCLUSIVE | .32 |
| pcorr_eq | E3 | 0.125 | ok | INCONCLUSIVE | 1.00 |
| pcorr_eq | E3 | 0.25 | ok | VALID | 1.00 |
| pcorr_eq | E3 | 0.5 | ok | INCONCLUSIVE | 1.00 |
| pcorr_eq | E4 | 0.125 | ok | INCONCLUSIVE | .00 |
| pcorr_eq | E4 | 0.25 | ok | INCONCLUSIVE | .05 |
| pcorr_eq | E4 | 0.5 | ok | VALID | .00 |
| pcorr_eq | E5 | 0.125 | ok | INCONCLUSIVE | .54 |
| pcorr_eq | E5 | 0.25 | ok | INCONCLUSIVE | .53 |
| pcorr_eq | E5 | 0.5 | ok | INCONCLUSIVE | .52 |
| pcorr_eq_min | E1 | 0.125 | ok | VALID | 1.00 |
| pcorr_eq_min | E1 | 0.25 | ok | VALID | 1.00 |
| pcorr_eq_min | E1 | 0.5 | ok | VALID | 1.00 |
| pcorr_eq_min | E2 | 0.125 | ok | INVALID | .35 |
| pcorr_eq_min | E2 | 0.25 | ok | INCONCLUSIVE | .36 |
| pcorr_eq_min | E2 | 0.5 | ok | INCONCLUSIVE | .32 |
| pcorr_eq_min | E3 | 0.125 | ok | INCONCLUSIVE | 1.00 |
| pcorr_eq_min | E3 | 0.25 | ok | VALID | 1.00 |
| pcorr_eq_min | E3 | 0.5 | ok | INCONCLUSIVE | 1.00 |
| pcorr_eq_min | E4 | 0.125 | ok | INCONCLUSIVE | .00 |
| pcorr_eq_min | E4 | 0.25 | ok | INCONCLUSIVE | .05 |
| pcorr_eq_min | E4 | 0.5 | ok | VALID | .00 |
| pcorr_eq_min | E5 | 0.125 | ok | INCONCLUSIVE | .54 |
| pcorr_eq_min | E5 | 0.25 | ok | INCONCLUSIVE | .53 |
| pcorr_eq_min | E5 | 0.5 | ok | INCONCLUSIVE | .51 |
| pcorr_hac | E1 | 0.125 | ok | INVALID | 1.00 |
| pcorr_hac | E1 | 0.25 | ok | INVALID | 1.00 |
| pcorr_hac | E1 | 0.5 | ok | INVALID | 1.00 |
| pcorr_hac | E2 | 0.125 | ok | INVALID | .51 |
| pcorr_hac | E2 | 0.25 | ok | INVALID | .53 |
| pcorr_hac | E2 | 0.5 | ok | INVALID | .51 |
| pcorr_hac | E3 | 0.125 | ok | INVALID | 1.00 |
| pcorr_hac | E3 | 0.25 | ok | INVALID | 1.00 |
| pcorr_hac | E3 | 0.5 | ok | INVALID | 1.00 |
| pcorr_hac | E4 | 0.125 | ok | INCONCLUSIVE | .35 |
| pcorr_hac | E4 | 0.25 | ok | INCONCLUSIVE | .30 |
| pcorr_hac | E4 | 0.5 | ok | VALID | .30 |
| pcorr_hac | E5 | 0.125 | ok | INVALID | .57 |
| pcorr_hac | E5 | 0.25 | ok | INVALID | .60 |
| pcorr_hac | E5 | 0.5 | ok | INVALID | .58 |
| pcorr_hac_eq_min | E1 | 0.125 | ok | VALID | 1.00 |
| pcorr_hac_eq_min | E1 | 0.25 | ok | VALID | 1.00 |
| pcorr_hac_eq_min | E1 | 0.5 | ok | VALID | 1.00 |
| pcorr_hac_eq_min | E2 | 0.125 | ok | INVALID | .26 |
| pcorr_hac_eq_min | E2 | 0.25 | ok | INCONCLUSIVE | .28 |
| pcorr_hac_eq_min | E2 | 0.5 | ok | INCONCLUSIVE | .25 |
| pcorr_hac_eq_min | E3 | 0.125 | ok | INCONCLUSIVE | 1.00 |
| pcorr_hac_eq_min | E3 | 0.25 | ok | VALID | 1.00 |
| pcorr_hac_eq_min | E3 | 0.5 | ok | INCONCLUSIVE | 1.00 |
| pcorr_hac_eq_min | E4 | 0.125 | ok | INCONCLUSIVE | .00 |
| pcorr_hac_eq_min | E4 | 0.25 | ok | INCONCLUSIVE | .05 |
| pcorr_hac_eq_min | E4 | 0.5 | ok | VALID | .00 |
| pcorr_hac_eq_min | E5 | 0.125 | ok | INCONCLUSIVE | .42 |
| pcorr_hac_eq_min | E5 | 0.25 | ok | VALID | .41 |
| pcorr_hac_eq_min | E5 | 0.5 | ok | INCONCLUSIVE | .42 |
| pcorr_hac_fb | E1 | 0.125 | ok | INVALID | 1.00 |
| pcorr_hac_fb | E1 | 0.25 | ok | INVALID | 1.00 |
| pcorr_hac_fb | E1 | 0.5 | ok | INVALID | 1.00 |
| pcorr_hac_fb | E2 | 0.125 | ok | INVALID | .51 |
| pcorr_hac_fb | E2 | 0.25 | ok | INVALID | .53 |
| pcorr_hac_fb | E2 | 0.5 | ok | INVALID | .51 |
| pcorr_hac_fb | E3 | 0.125 | ok | INVALID | 1.00 |
| pcorr_hac_fb | E3 | 0.25 | ok | INVALID | 1.00 |
| pcorr_hac_fb | E3 | 0.5 | ok | INVALID | 1.00 |
| pcorr_hac_fb | E4 | 0.125 | ok | INCONCLUSIVE | .35 |
| pcorr_hac_fb | E4 | 0.25 | ok | INCONCLUSIVE | .28 |
| pcorr_hac_fb | E4 | 0.5 | ok | VALID | .30 |
| pcorr_hac_fb | E5 | 0.125 | ok | INVALID | .57 |
| pcorr_hac_fb | E5 | 0.25 | ok | INVALID | .59 |
| pcorr_hac_fb | E5 | 0.5 | ok | INVALID | .58 |
| pcorr_hac_fb_eq_min | E1 | 0.125 | ok | VALID | 1.00 |
| pcorr_hac_fb_eq_min | E1 | 0.25 | ok | VALID | 1.00 |
| pcorr_hac_fb_eq_min | E1 | 0.5 | ok | VALID | 1.00 |
| pcorr_hac_fb_eq_min | E2 | 0.125 | ok | INVALID | .25 |
| pcorr_hac_fb_eq_min | E2 | 0.25 | ok | INCONCLUSIVE | .28 |
| pcorr_hac_fb_eq_min | E2 | 0.5 | ok | INCONCLUSIVE | .25 |
| pcorr_hac_fb_eq_min | E3 | 0.125 | ok | INCONCLUSIVE | 1.00 |
| pcorr_hac_fb_eq_min | E3 | 0.25 | ok | VALID | 1.00 |
| pcorr_hac_fb_eq_min | E3 | 0.5 | ok | INCONCLUSIVE | 1.00 |
| pcorr_hac_fb_eq_min | E4 | 0.125 | ok | INCONCLUSIVE | .00 |
| pcorr_hac_fb_eq_min | E4 | 0.25 | ok | INCONCLUSIVE | .05 |
| pcorr_hac_fb_eq_min | E4 | 0.5 | ok | VALID | .00 |
| pcorr_hac_fb_eq_min | E5 | 0.125 | ok | INCONCLUSIVE | .42 |
| pcorr_hac_fb_eq_min | E5 | 0.25 | ok | VALID | .41 |
| pcorr_hac_fb_eq_min | E5 | 0.5 | ok | INCONCLUSIVE | .42 |
| pcorr_native | E1 | 0.125 | ok | INVALID | 1.00 |
| pcorr_native | E1 | 0.25 | ok | INVALID | 1.00 |
| pcorr_native | E1 | 0.5 | ok | INVALID | 1.00 |
| pcorr_native | E2 | 0.125 | ok | INVALID | .52 |
| pcorr_native | E2 | 0.25 | ok | INVALID | .55 |
| pcorr_native | E2 | 0.5 | ok | INVALID | .52 |
| pcorr_native | E3 | 0.125 | ok | INVALID | 1.00 |
| pcorr_native | E3 | 0.25 | ok | INVALID | 1.00 |
| pcorr_native | E3 | 0.5 | ok | INVALID | 1.00 |
| pcorr_native | E4 | 0.125 | ok | INCONCLUSIVE | .35 |
| pcorr_native | E4 | 0.25 | ok | INCONCLUSIVE | .30 |
| pcorr_native | E4 | 0.5 | ok | VALID | .30 |
| pcorr_native | E5 | 0.125 | ok | INVALID | .62 |
| pcorr_native | E5 | 0.25 | ok | INVALID | .64 |
| pcorr_native | E5 | 0.5 | ok | INVALID | .61 |
| pmrt_eq | E1 | 0.125 | ok | INCONCLUSIVE | .97 |
| pmrt_eq | E1 | 0.25 | ok | VALID | .99 |
| pmrt_eq | E1 | 0.5 | ok | VALID | 1.00 |
| pmrt_eq | E2 | 0.125 | ok | INVALID | .27 |
| pmrt_eq | E2 | 0.25 | ok | VALID | .28 |
| pmrt_eq | E2 | 0.5 | ok | INVALID | .24 |
| pmrt_eq | E3 | 0.125 | ok | INCONCLUSIVE | .97 |
| pmrt_eq | E3 | 0.25 | ok | INCONCLUSIVE | .95 |
| pmrt_eq | E3 | 0.5 | ok | INCONCLUSIVE | .97 |
| pmrt_eq | E4 | 0.125 | ok | VALID | .00 |
| pmrt_eq | E4 | 0.25 | ok | INCONCLUSIVE | .08 |
| pmrt_eq | E4 | 0.5 | ok | VALID | .00 |
| pmrt_eq | E5 | 0.125 | ok | INCONCLUSIVE | .47 |
| pmrt_eq | E5 | 0.25 | ok | VALID | .46 |
| pmrt_eq | E5 | 0.5 | ok | VALID | .48 |
| pmrt_nl_eq | E1 | 0.25 | ok | INCONCLUSIVE | 1.00 |
| pmrt_nl_eq | E2 | 0.25 | ok | INCONCLUSIVE | .64 |
| pmrt_nl_eq | E3 | 0.25 | ok | VALID | 1.00 |
| pmrt_nl_eq | E5 | 0.25 | ok | VALID | .68 |
| pmrt_r3 | E1 | 0.125 | ok | VALID | 1.00 |
| pmrt_r3 | E1 | 0.25 | ok | INCONCLUSIVE | 1.00 |
| pmrt_r3 | E1 | 0.5 | ok | VALID | 1.00 |
| pmrt_r3 | E2 | 0.125 | ok | INCONCLUSIVE | .31 |
| pmrt_r3 | E2 | 0.25 | ok | INVALID | .33 |
| pmrt_r3 | E2 | 0.5 | ok | INVALID | .30 |
| pmrt_r3 | E3 | 0.125 | ok | INCONCLUSIVE | 1.00 |
| pmrt_r3 | E3 | 0.25 | ok | INCONCLUSIVE | 1.00 |
| pmrt_r3 | E3 | 0.5 | ok | INCONCLUSIVE | 1.00 |
| pmrt_r3 | E4 | 0.125 | ok | INCONCLUSIVE | .00 |
| pmrt_r3 | E4 | 0.25 | ok | INCONCLUSIVE | .08 |
| pmrt_r3 | E4 | 0.5 | ok | VALID | .00 |
| pmrt_r3 | E5 | 0.125 | ok | INCONCLUSIVE | .54 |
| pmrt_r3 | E5 | 0.25 | ok | INCONCLUSIVE | .50 |
| pmrt_r3 | E5 | 0.5 | ok | INCONCLUSIVE | .51 |
| rcot2_eq | E1 | 0.125 | ok | INVALID | 1.00 |
| rcot2_eq | E1 | 0.25 | ok | INVALID | 1.00 |
| rcot2_eq | E1 | 0.5 | ok | INVALID | .94 |
| rcot2_eq | E2 | 0.125 | ok | INVALID | .53 |
| rcot2_eq | E2 | 0.25 | ok | INVALID | .49 |
| rcot2_eq | E2 | 0.5 | ok | INVALID | .44 |
| rcot2_eq | E3 | 0.125 | ok | INVALID | .99 |
| rcot2_eq | E3 | 0.25 | ok | INVALID | .97 |
| rcot2_eq | E3 | 0.5 | ok | INVALID | .84 |
| rcot2_eq | E4 | 0.125 | ok | INCONCLUSIVE | .05 |
| rcot2_eq | E4 | 0.25 | ok | INCONCLUSIVE | .02 |
| rcot2_eq | E4 | 0.5 | ok | INCONCLUSIVE | .10 |
| rcot2_eq | E5 | 0.125 | ok | INVALID | .67 |
| rcot2_eq | E5 | 0.25 | ok | INVALID | .68 |
| rcot2_eq | E5 | 0.5 | ok | INCONCLUSIVE | .67 |
| rcot2_eq_min | E1 | 0.125 | ok | INCONCLUSIVE | 1.00 |
| rcot2_eq_min | E1 | 0.25 | ok | INVALID | .99 |
| rcot2_eq_min | E1 | 0.5 | ok | INCONCLUSIVE | .95 |
| rcot2_eq_min | E2 | 0.125 | ok | INVALID | .54 |
| rcot2_eq_min | E2 | 0.25 | ok | INVALID | .50 |
| rcot2_eq_min | E2 | 0.5 | ok | INVALID | .44 |
| rcot2_eq_min | E3 | 0.125 | ok | INVALID | .97 |
| rcot2_eq_min | E3 | 0.25 | ok | INVALID | .97 |
| rcot2_eq_min | E3 | 0.5 | ok | INCONCLUSIVE | .81 |
| rcot2_eq_min | E4 | 0.125 | ok | INCONCLUSIVE | .05 |
| rcot2_eq_min | E4 | 0.25 | ok | INCONCLUSIVE | .02 |
| rcot2_eq_min | E4 | 0.5 | ok | INCONCLUSIVE | .10 |
| rcot2_eq_min | E5 | 0.125 | ok | INCONCLUSIVE | .67 |
| rcot2_eq_min | E5 | 0.25 | ok | INVALID | .67 |
| rcot2_eq_min | E5 | 0.5 | ok | INCONCLUSIVE | .67 |
| rcot2_native | E1 | 0.125 | ok | INVALID | .99 |
| rcot2_native | E1 | 0.25 | ok | INVALID | 1.00 |
| rcot2_native | E1 | 0.5 | ok | INVALID | .99 |
| rcot2_native | E2 | 0.125 | ok | INVALID | .55 |
| rcot2_native | E2 | 0.25 | ok | INVALID | .54 |
| rcot2_native | E2 | 0.5 | ok | INVALID | .50 |
| rcot2_native | E3 | 0.125 | ok | INVALID | 1.00 |
| rcot2_native | E3 | 0.25 | ok | INVALID | 1.00 |
| rcot2_native | E3 | 0.5 | ok | INVALID | .99 |
| rcot2_native | E4 | 0.125 | ok | INCONCLUSIVE | .15 |
| rcot2_native | E4 | 0.25 | ok | INCONCLUSIVE | .13 |
| rcot2_native | E4 | 0.5 | ok | INCONCLUSIVE | .25 |
| rcot2_native | E5 | 0.125 | ok | INVALID | .70 |
| rcot2_native | E5 | 0.25 | ok | INVALID | .70 |
| rcot2_native | E5 | 0.5 | ok | INCONCLUSIVE | .68 |
| shap_dag | E1 | 0.125 | ok | INVALID | 1.00 |
| shap_dag | E1 | 0.25 | ok | INVALID | 1.00 |
| shap_dag | E1 | 0.5 | ok | INVALID | 1.00 |
| shap_dag | E2 | 0.125 | ok | VALID | .92 |
| shap_dag | E2 | 0.25 | ok | VALID | .90 |
| shap_dag | E2 | 0.5 | ok | VALID | .89 |
| shap_dag | E3 | 0.125 | ok | INVALID | 1.00 |
| shap_dag | E3 | 0.25 | ok | INVALID | 1.00 |
| shap_dag | E3 | 0.5 | ok | INVALID | 1.00 |
| shap_dag | E4 | 0.125 | ok | NO_READ | .10 |
| shap_dag | E4 | 0.25 | ok | NO_READ | .10 |
| shap_dag | E4 | 0.5 | ok | NO_READ | .20 |
| shap_dag | E5 | 0.125 | ok | INCONCLUSIVE | .79 |
| shap_dag | E5 | 0.25 | ok | VALID | .67 |
| shap_dag | E5 | 0.5 | ok | VALID | .67 |
| two_tower | E1 | 0.125 | ok | INVALID | 1.00 |
| two_tower | E1 | 0.25 | ok | INVALID | .99 |
| two_tower | E1 | 0.5 | ok | INVALID | 1.00 |
| two_tower | E2 | 0.125 | ok | VALID | .45 |
| two_tower | E2 | 0.25 | ok | VALID | .42 |
| two_tower | E2 | 0.5 | ok | VALID | .49 |
| two_tower | E3 | 0.125 | ok | INVALID | .99 |
| two_tower | E3 | 0.25 | ok | INVALID | .99 |
| two_tower | E3 | 0.5 | ok | INVALID | .99 |
| two_tower | E4 | 0.125 | ok | NO_READ | .30 |
| two_tower | E4 | 0.25 | ok | NO_READ | .18 |
| two_tower | E4 | 0.5 | ok | NO_READ | .00 |
| two_tower | E5 | 0.125 | ok | VALID | .78 |
| two_tower | E5 | 0.25 | ok | VALID | .70 |
| two_tower | E5 | 0.5 | ok | VALID | .67 |

## 8. Not in grid (pre-registered grid choices, R-54 / R-58)

> **DEV DATA, NOT EVAL: development seeds 3_000_000-3_000_199 on the DEV specs; a rehearsal of the report, not the pre-registered result**

Cells outside an arm's grid are not planned: they never count as missing, never cap a verdict at PARTIAL and never make a component NOT EVALUABLE; every table labels them `nig`.

| ruling | arms | where | cells | reason |
|---|---|---|---|---|
| R-58(1) | cdl | regimes ['R3', 'R4'] | 0 | cdl: training cost; feeds no claim (R1 / R2, n <= 4000 only) |
| R-58(1) | cdl | ns [8000, 24000] | 0 | cdl: training cost; feeds no claim (R1 / R2, n <= 4000 only) |
| R-58(1) | cdl | kappas [0.125, 0.5] | 0 | cdl: not in the kappa sweep |
| R-54 | mscr_eq, mscr_native | ns [4000, 8000, 24000] | 108 | mscr: reported-INVALID arm, EVAL n <= 1000 only |
| R-58(5) | mscr_eq, mscr_native | kappas [0.125, 0.5] | 0 | mscr: not in the kappa sweep |
| DEV scope | pmrt_eq, pmrt_r3, pc_eq, pc_native, granger_eq, granger_native, corr, notears, shap_dag, two_tower, pcorr_eq, pcorr_native, pcorr_hac, pcorr_hac_fb, rcot2_eq, rcot2_native, pcorr_eq_min, rcot2_eq_min, pcorr_hac_eq_min, pcorr_hac_fb_eq_min, mscr_eq, mscr_native, pmrt_nl_eq | all | 76 | the DEV specs ran a subset of the EVAL grid (DEV rendering only) |

184 cells in total; unexplained gaps: none.

## 9. Figure-ready CSVs

> **DEV DATA, NOT EVAL: development seeds 3_000_000-3_000_199 on the DEV specs; a rehearsal of the report, not the pre-registered result**

- `csv/cost.csv` (sha256 e88f54164d62e75b)
- `csv/e4_by_lambda.csv` (sha256 1b466987908eb56a)
- `csv/eq_arms.csv` (sha256 57d68ced1532ea9a)
- `csv/information_levels_R2.csv` (sha256 7f5c38742c84b691)
- `csv/kappa_sweep.csv` (sha256 cb3b857563fa0ae2)
- `csv/like_for_like.csv` (sha256 2446c556817ac9fa)
- `csv/not_in_grid.csv` (sha256 cac84ffd9a4a0e07)
- `csv/paired_diff.csv` (sha256 c354212629b6d733)
- `csv/recall_vs_n.csv` (sha256 f2d1e12972ba320e)
- `csv/validity_cells.csv` (sha256 4001b31a04e36cdb)

## Appendix: every eval_analysis table (V0-V11)

> **DEV DATA, NOT EVAL: development seeds 3_000_000-3_000_199 on the DEV specs; a rehearsal of the report, not the pre-registered result**


### V0 like-for-like (headline, R-42): recall / truth-null rate / placebo rate per cell

Every p arm scored at raw p <= .05 per edge (raw_p) and with the conformal placebo tau (tau), next to the score-only arms (tau; cdl also at the conference's fixed threshold, fixed). Rates are declaration rates; * INVALID, ~ INCONCLUSIVE; a tau row's placebo is its tuning column (reported, not in its validity). NA = not applicable (PMRT in R4: no known design, never recall 0); inv INVALID, inf infeasible (EVAL or T3), mis missing, unt untuned, few < 10 seeds; nig not in the arm's grid (pre-registered grid choice, R-54 / R-58).

### E1 R1

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | raw_p | 1.00 / .029 / .050~ | 1.00 / .037 / .050~ | 1.00 / .029 / .050~ | nig | nig |
| pmrt_nl_eq | tau | 1.00 / .033 / .050~ | 1.00 / .017 / .050~ | 1.00 / .029 / .050~ | nig | nig |
| pmrt_eq | raw_p | 1.00 / .046 / .025 | 1.00 / .037 / .050~ | 1.00 / .037 / .087~ | 1.00 / .046 / .062~ | 1.00 / .029 / .050~ |
| pmrt_eq | tau | 1.00 / .025 / .013 | 1.00 / .100* / .138* | 1.00 / .037 / .075~ | 1.00 / .046 / .062~ | 1.00 / .033 / .087~ |
| pmrt_r3 | raw_p | 1.00 / .071~ / .050~ | 1.00 / .046 / .050~ | 1.00 / .029 / .100~ | 1.00 / .054~ / .062~ | 1.00 / .037 / .025 |
| pmrt_r3 | tau | 1.00 / .054 / .050~ | 1.00 / .071~ / .125~ | 1.00 / .033 / .113~ | 1.00 / .042 / .050~ | 1.00 / .037 / .025 |
| corr | raw_p | 1.00 / .050~ / .025 | 1.00 / .037 / .062~ | 1.00 / .029 / .050~ | 1.00 / .046 / .025 | 1.00 / .087~ / .025 |
| corr | tau | 1.00 / .075~ / .037 | 1.00 / .037 / .062~ | 1.00 / .025 / .037~ | 1.00 / .046 / .025 | 1.00 / .029 / .000 |
| mscr_eq | raw_p | 1.00 / .042 / .025 | 1.00 / .037 / .050~ | nig | nig | nig |
| mscr_eq | tau | 1.00 / .050 / .050~ | 1.00 / .054~ / .050~ | nig | nig | nig |
| mscr_native | raw_p | 1.00 / .046~ / .025 | 1.00 / .033 / .075~ | nig | nig | nig |
| mscr_native | tau | 1.00 / .050~ / .050~ | 1.00 / .029 / .025 | nig | nig | nig |
| notears | tau | 1.00 / .000 / .000 | 1.00 / .000 / .000 | 1.00 / .000 / .000 | 1.00 / .000 / .000 | 1.00 / .000 / .000 |
| pc_eq | tau | 1.00 / .087~ / .075~ | 1.00 / .037 / .013 | 1.00 / .054~ / .062~ | 1.00 / .013 / .025 | 1.00 / .050~ / .062~ |
| pc_native | tau | 1.00 / .092* / .050~ | 1.00 / .033 / .013 | 1.00 / .054~ / .075~ | 1.00 / .017 / .025 | 1.00 / .058~ / .062~ |
| pcorr_eq | raw_p | 1.00 / .037 / .025 | 1.00 / .029 / .050~ | 1.00 / .025 / .075~ | 1.00 / .033 / .075~ | 1.00 / .025 / .050~ |
| pcorr_eq | tau | 1.00 / .021 / .025 | 1.00 / .062~ / .100~ | 1.00 / .021 / .075~ | 1.00 / .033 / .075~ | 1.00 / .037 / .062~ |
| pcorr_eq_min | raw_p | 1.00 / .033 / .037 | 1.00 / .046 / .050~ | 1.00 / .021 / .100~ | 1.00 / .033 / .075~ | 1.00 / .033 / .025 |
| pcorr_eq_min | tau | 1.00 / .033 / .025 | 1.00 / .058~ / .062~ | 1.00 / .029 / .125* | 1.00 / .025 / .050~ | 1.00 / .029 / .025 |
| pcorr_hac | raw_p | 1.00 / .037 / .025 | 1.00 / .046 / .050~ | 1.00 / .021 / .100~ | 1.00 / .033 / .087~ | 1.00 / .033 / .025 |
| pcorr_hac | tau | 1.00 / .037 / .025 | 1.00 / .058~ / .062~ | 1.00 / .025 / .100~ | 1.00 / .025 / .050~ | 1.00 / .029 / .025 |
| pcorr_hac_eq_min | raw_p | 1.00 / .037 / .025 | 1.00 / .046 / .050~ | 1.00 / .021 / .100~ | 1.00 / .033 / .087~ | 1.00 / .033 / .025 |
| pcorr_hac_eq_min | tau | 1.00 / .037 / .025 | 1.00 / .058~ / .062~ | 1.00 / .025 / .100~ | 1.00 / .025 / .050~ | 1.00 / .029 / .025 |
| pcorr_hac_fb | raw_p | 1.00 / .037 / .025 | 1.00 / .046 / .050~ | 1.00 / .021 / .100~ | 1.00 / .033 / .087~ | 1.00 / .033 / .025 |
| pcorr_hac_fb | tau | 1.00 / .037 / .025 | 1.00 / .058~ / .062~ | 1.00 / .025 / .100~ | 1.00 / .025 / .050~ | 1.00 / .029 / .025 |
| pcorr_hac_fb_eq_min | raw_p | 1.00 / .037 / .025 | 1.00 / .046 / .050~ | 1.00 / .021 / .100~ | 1.00 / .033 / .087~ | 1.00 / .033 / .025 |
| pcorr_hac_fb_eq_min | tau | 1.00 / .037 / .025 | 1.00 / .058~ / .062~ | 1.00 / .025 / .100~ | 1.00 / .025 / .050~ | 1.00 / .029 / .025 |
| pcorr_native | raw_p | 1.00 / .033 / .037 | 1.00 / .046 / .050~ | 1.00 / .021 / .100~ | 1.00 / .033 / .075~ | 1.00 / .033 / .025 |
| pcorr_native | tau | 1.00 / .033 / .025 | 1.00 / .058~ / .062~ | 1.00 / .029 / .125* | 1.00 / .025 / .050~ | 1.00 / .029 / .025 |
| rcot2_eq | raw_p | 1.00 / .096* / .062~ | 1.00 / .079~ / .087~ | 1.00 / .033 / .037 | 1.00 / .054 / .075~ | 1.00 / .037 / .025 |
| rcot2_eq | tau | 1.00 / .062~ / .037~ | 1.00 / .042 / .037 | 1.00 / .037 / .013 | 1.00 / .071~ / .025 | 1.00 / .037 / .037 |
| rcot2_eq_min | raw_p | 1.00 / .079~ / .062~ | 1.00 / .062~ / .037~ | 1.00 / .013 / .062~ | 1.00 / .054~ / .062~ | 1.00 / .054~ / .087~ |
| rcot2_eq_min | tau | 1.00 / .087* / .025 | 1.00 / .083* / .013 | 1.00 / .025 / .025 | 1.00 / .071~ / .025 | 1.00 / .083* / .100~ |
| rcot2_native | raw_p | 1.00 / .100* / .075~ | 1.00 / .075~ / .037 | 1.00 / .025 / .025 | 1.00 / .058~ / .062~ | 1.00 / .071~ / .125* |
| rcot2_native | tau | 1.00 / .042 / .000 | 1.00 / .033 / .013 | 1.00 / .033 / .013 | 1.00 / .050 / .025 | 1.00 / .054~ / .062~ |
| shap_dag | tau | 1.00 / .062~ / .050~ | 1.00 / .037 / .013 | 1.00 / .104* / .050~ | 1.00 / .100* / .050~ | 1.00 / .046 / .075~ |
| two_tower | tau | 1.00 / .021 / .062~ | 1.00 / .033 / .025 | 1.00 / .025 / .062~ | 1.00 / .037 / .037 | 1.00 / .113* / .025 |

### E1 R2

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | raw_p | 1.00 / .042 / .058~ | 1.00 / .053 / .067~ | 1.00 / .051 / .029 | nig | nig |
| pmrt_nl_eq | tau | 1.00 / .028 / .033 | 1.00 / .036 / .054~ | 1.00 / .068~ / .058~ | nig | nig |
| pmrt_eq | raw_p | .99 / .039 / .050~ | .99 / .039 / .046 | 1.00 / .050 / .042 | 1.00 / .054~ / .075~ | 1.00 / .025 / .025 |
| pmrt_eq | tau | .99 / .043 / .071~ | .99 / .024 / .033 | 1.00 / .103* / .129* | 1.00 / .037 / .050~ | 1.00 / .013 / .000 |
| pmrt_r3 | raw_p | 1.00 / .044 / .037 | 1.00 / .050 / .050~ | 1.00 / .056 / .046 | 1.00 / .042 / .050~ | 1.00 / .017 / .025 |
| pmrt_r3 | tau | 1.00 / .064~ / .071~ | 1.00 / .029 / .037 | 1.00 / .056 / .042 | 1.00 / .042 / .037 | 1.00 / .033 / .037~ |
| corr | raw_p | 1.00 / .681* / .617* | 1.00 / .764* / .733* | 1.00 / .883* / .896* | 1.00 / .925* / .988* | 1.00 / .958* / .988* |
| corr | tau | 1.00 / .219* / .129* | 1.00 / .214* / .096* | 1.00 / .201* / .100* | 1.00 / .221* / .100~ | 1.00 / .212* / .087~ |
| mscr_eq | raw_p | 1.00 / 1.000* / 1.000* | 1.00 / 1.000* / 1.000* | nig | nig | nig |
| mscr_eq | tau | 1.00 / .085* / .050~ | 1.00 / .128* / .092* | nig | nig | nig |
| mscr_native | raw_p | 1.00 / .997* / .996* | 1.00 / 1.000* / 1.000* | nig | nig | nig |
| mscr_native | tau | 1.00 / .249* / .096* | 1.00 / .211* / .042 | nig | nig | nig |
| notears | tau | 1.00 / .157* / .000 | 1.00 / .168* / .000 | 1.00 / .167* / .000 | 1.00 / .167* / .000 | 1.00 / .167* / .000 |
| pc_eq | tau | 1.00 / .099* / .100* | 1.00 / .071* / .054~ | 1.00 / .028 / .017 | 1.00 / .033 / .013 | 1.00 / .042 / .037 |
| pc_native | tau | 1.00 / .101* / .067~ | 1.00 / .028 / .017 | 1.00 / .042 / .017 | 1.00 / .067~ / .075~ | 1.00 / .042 / .037 |
| pcorr_eq | raw_p | 1.00 / .031 / .046~ | 1.00 / .037 / .050~ | 1.00 / .046 / .033 | 1.00 / .062~ / .025 | 1.00 / .021 / .037 |
| pcorr_eq | tau | 1.00 / .087* / .113* | 1.00 / .018 / .021 | 1.00 / .051 / .042 | 1.00 / .033 / .025 | 1.00 / .021 / .025 |
| pcorr_eq_min | raw_p | 1.00 / .035 / .046 | 1.00 / .042 / .046 | 1.00 / .044 / .033 | 1.00 / .054~ / .062~ | 1.00 / .021 / .037 |
| pcorr_eq_min | tau | 1.00 / .081* / .087* | 1.00 / .019 / .013 | 1.00 / .029 / .017 | 1.00 / .042 / .037 | 1.00 / .021 / .050~ |
| pcorr_hac | raw_p | 1.00 / .210* / .054~ | 1.00 / .217* / .054~ | 1.00 / .225* / .117* | 1.00 / .233* / .200* | 1.00 / .271* / .300* |
| pcorr_hac | tau | 1.00 / .181* / .025 | 1.00 / .194* / .033 | 1.00 / .178* / .042 | 1.00 / .179* / .075~ | 1.00 / .167* / .013 |
| pcorr_hac_eq_min | raw_p | 1.00 / .032 / .042 | 1.00 / .040 / .046 | 1.00 / .044 / .033 | 1.00 / .054~ / .062~ | 1.00 / .021 / .037 |
| pcorr_hac_eq_min | tau | 1.00 / .083* / .083* | 1.00 / .022 / .021 | 1.00 / .032 / .017 | 1.00 / .042 / .037 | 1.00 / .025 / .037 |
| pcorr_hac_fb | raw_p | 1.00 / .208* / .054~ | 1.00 / .215* / .054~ | 1.00 / .225* / .117* | 1.00 / .233* / .200* | 1.00 / .271* / .300* |
| pcorr_hac_fb | tau | 1.00 / .181* / .025 | 1.00 / .194* / .033 | 1.00 / .178* / .042 | 1.00 / .179* / .075~ | 1.00 / .167* / .013 |
| pcorr_hac_fb_eq_min | raw_p | 1.00 / .032 / .042 | 1.00 / .040 / .046 | 1.00 / .044 / .033 | 1.00 / .054~ / .062~ | 1.00 / .021 / .037 |
| pcorr_hac_fb_eq_min | tau | 1.00 / .082* / .083* | 1.00 / .022 / .021 | 1.00 / .032 / .017 | 1.00 / .042 / .037 | 1.00 / .025 / .037 |
| pcorr_native | raw_p | 1.00 / .206* / .058~ | 1.00 / .215* / .042 | 1.00 / .225* / .121* | 1.00 / .233* / .200* | 1.00 / .267* / .300* |
| pcorr_native | tau | 1.00 / .193* / .033 | 1.00 / .192* / .029 | 1.00 / .185* / .042 | 1.00 / .183* / .062~ | 1.00 / .167* / .013 |
| rcot2_eq | raw_p | 1.00 / .101* / .117* | 1.00 / .075* / .113* | 1.00 / .243* / .225* | 1.00 / .354* / .375* | 1.00 / .637* / .637* |
| rcot2_eq | tau | .97 / .037 / .075~ | .99 / .053 / .054~ | 1.00 / .153* / .158* | 1.00 / .050 / .037~ | 1.00 / .054~ / .037~ |
| rcot2_eq_min | raw_p | 1.00 / .094* / .092* | 1.00 / .058~ / .083* | 1.00 / .121* / .121* | 1.00 / .192* / .188* | 1.00 / .404* / .475* |
| rcot2_eq_min | tau | .97 / .029 / .046 | .99 / .060~ / .062~ | 1.00 / .067~ / .079~ | 1.00 / .037 / .025 | 1.00 / .037 / .037 |
| rcot2_native | raw_p | 1.00 / .133* / .121* | 1.00 / .147* / .121* | 1.00 / .196* / .200* | 1.00 / .275* / .362* | 1.00 / .517* / .662* |
| rcot2_native | tau | 1.00 / .054 / .058~ | 1.00 / .047 / .075~ | 1.00 / .082* / .108* | 1.00 / .025 / .037~ | 1.00 / .025 / .037~ |
| shap_dag | tau | 1.00 / .260* / .113* | 1.00 / .236* / .087* | 1.00 / .235* / .087* | 1.00 / .237* / .125* | 1.00 / .217* / .062~ |
| two_tower | tau | .99 / .207* / .025 | .99 / .201* / .029 | 1.00 / .229* / .050~ | 1.00 / .217* / .013 | 1.00 / .254* / .050~ |

### E2 R1

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | raw_p | .83 / .030 / .067~ | .90 / .025 / .067~ | .96 / .034 / .075~ | nig | nig |
| pmrt_nl_eq | tau | .83 / .022 / .058~ | .90 / .027 / .067~ | .97 / .044 / .083~ | nig | nig |
| pmrt_eq | raw_p | .38 / .070~ / .042~ | .38 / .056~ / .050~ | .42 / .058~ / .075~ | .41 / .048 / .075~ | .41 / .059~ / .075~ |
| pmrt_eq | tau | .40 / .130* / .150* | .42 / .098* / .100* | .42 / .064~ / .075~ | .40 / .028 / .067~ | .40 / .041 / .067~ |
| pmrt_r3 | raw_p | .36 / .067~ / .058~ | .38 / .058~ / .042 | .42 / .056~ / .075~ | .41 / .048 / .075~ | .41 / .059~ / .075~ |
| pmrt_r3 | tau | .39 / .116* / .108* | .42 / .122* / .108* | .42 / .062~ / .058~ | .40 / .028 / .042~ | .40 / .041 / .058~ |
| corr | raw_p | .36 / .069~ / .083~ | .38 / .052 / .058~ | .42 / .064~ / .050~ | .42 / .042 / .092~ | .41 / .056~ / .100~ |
| corr | tau | .38 / .091* / .100~ | .41 / .092* / .067~ | .42 / .062~ / .050~ | .41 / .033 / .067~ | .39 / .030 / .067~ |
| mscr_eq | raw_p | .92 / .042 / .042~ | .95 / .053 / .050~ | nig | nig | nig |
| mscr_eq | tau | .93 / .034 / .025 | .95 / .028 / .033 | nig | nig | nig |
| mscr_native | raw_p | .93 / .055 / .050~ | .95 / .059~ / .042 | nig | nig | nig |
| mscr_native | tau | .94 / .077* / .083~ | .95 / .041 / .025 | nig | nig | nig |
| notears | tau | .28 / .017 / .008 | .26 / .000 / .000 | .25 / .000 / .000 | .25 / .000 / .000 | .25 / .000 / .000 |
| pc_eq | tau | .32 / .055 / .067~ | .35 / .050 / .042 | .40 / .027 / .017 | .40 / .017 / .033 | .41 / .052 / .092~ |
| pc_native | tau | .31 / .055 / .058~ | .36 / .062~ / .050~ | .40 / .025 / .008 | .40 / .016 / .033 | .42 / .052 / .083~ |
| pcorr_eq | raw_p | .35 / .069~ / .050~ | .38 / .064~ / .050~ | .41 / .058~ / .058~ | .42 / .044 / .083~ | .41 / .050~ / .083~ |
| pcorr_eq | tau | .40 / .120* / .083~ | .40 / .081* / .075~ | .42 / .067~ / .075~ | .40 / .025 / .033 | .40 / .041 / .067~ |
| pcorr_eq_min | raw_p | .35 / .067~ / .033 | .38 / .064~ / .050~ | .42 / .067~ / .058~ | .42 / .045 / .075~ | .41 / .052~ / .083~ |
| pcorr_eq_min | tau | .37 / .095* / .075~ | .42 / .105* / .083~ | .43 / .069~ / .075~ | .40 / .030 / .050~ | .40 / .041 / .058~ |
| pcorr_hac | raw_p | .35 / .070~ / .050~ | .38 / .066~ / .033 | .43 / .064~ / .067~ | .42 / .044 / .075~ | .42 / .052~ / .083~ |
| pcorr_hac | tau | .37 / .092* / .067~ | .42 / .102* / .083~ | .44 / .069~ / .083~ | .40 / .025 / .042~ | .40 / .039 / .058~ |
| pcorr_hac_eq_min | raw_p | .35 / .070~ / .050~ | .38 / .066~ / .033 | .43 / .064~ / .067~ | .42 / .044 / .075~ | .42 / .052~ / .083~ |
| pcorr_hac_eq_min | tau | .37 / .092* / .067~ | .42 / .102* / .083~ | .44 / .069~ / .083~ | .40 / .025 / .042~ | .40 / .039 / .058~ |
| pcorr_hac_fb | raw_p | .35 / .070~ / .050~ | .38 / .066~ / .033 | .43 / .064~ / .067~ | .42 / .044 / .075~ | .42 / .052~ / .083~ |
| pcorr_hac_fb | tau | .37 / .092* / .067~ | .42 / .102* / .083~ | .44 / .069~ / .083~ | .40 / .025 / .042~ | .40 / .039 / .058~ |
| pcorr_hac_fb_eq_min | raw_p | .35 / .070~ / .050~ | .38 / .066~ / .033 | .43 / .064~ / .067~ | .42 / .044 / .075~ | .42 / .052~ / .083~ |
| pcorr_hac_fb_eq_min | tau | .37 / .092* / .067~ | .42 / .102* / .083~ | .44 / .069~ / .083~ | .40 / .025 / .042~ | .40 / .039 / .058~ |
| pcorr_native | raw_p | .35 / .067~ / .033 | .38 / .064~ / .050~ | .42 / .067~ / .058~ | .42 / .045 / .075~ | .41 / .052~ / .083~ |
| pcorr_native | tau | .37 / .095* / .075~ | .42 / .105* / .083~ | .43 / .069~ / .075~ | .40 / .030 / .050~ | .40 / .041 / .058~ |
| rcot2_eq | raw_p | .61 / .109* / .092~ | .64 / .086* / .083~ | .68 / .059~ / .025 | .68 / .050 / .050~ | .75 / .059 / .067~ |
| rcot2_eq | tau | .56 / .075~ / .033 | .61 / .058~ / .058~ | .66 / .030 / .025 | .67 / .041 / .033 | .73 / .050 / .033 |
| rcot2_eq_min | raw_p | .64 / .100* / .100~ | .63 / .073* / .058~ | .67 / .067~ / .033 | .67 / .056 / .050~ | .73 / .061 / .033 |
| rcot2_eq_min | tau | .56 / .053~ / .033 | .62 / .041 / .042~ | .65 / .019 / .033 | .68 / .042 / .017 | .72 / .039 / .017 |
| rcot2_native | raw_p | .63 / .094* / .067~ | .64 / .064~ / .050~ | .68 / .069~ / .033 | .68 / .053 / .075~ | .74 / .061~ / .058~ |
| rcot2_native | tau | .57 / .070~ / .017 | .61 / .042 / .033 | .66 / .019 / .033 | .67 / .013 / .008 | .75 / .067~ / .033 |
| shap_dag | tau | .90 / .073* / .042 | .95 / .039 / .058~ | .99 / .041 / .025 | 1.00 / .045 / .025 | 1.00 / .034 / .033 |
| two_tower | tau | .69 / .052 / .067~ | .68 / .031 / .033 | .78 / .036 / .025 | .79 / .075* / .025 | .79 / .178* / .067~ |

### E2 R2

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | raw_p | .62 / .051 / .047 | .78 / .046 / .050~ | .93 / .041 / .053 | nig | nig |
| pmrt_nl_eq | tau | .66 / .081* / .081* | .80 / .069* / .072~ | .94 / .110* / .111* | nig | nig |
| pmrt_eq | raw_p | .35 / .042 / .039 | .44 / .061 / .042 | .61 / .049 / .033 | .68 / .064~ / .050~ | .83 / .039 / .042~ |
| pmrt_eq | tau | .37 / .054 / .047 | .43 / .049 / .031 | .64 / .077* / .053~ | .64 / .039 / .025 | .78 / .019 / .008 |
| pmrt_r3 | raw_p | .38 / .050 / .053~ | .46 / .063* / .056~ | .65 / .046 / .047 | .71 / .050~ / .025 | .88 / .041 / .008 |
| pmrt_r3 | tau | .40 / .062* / .061~ | .49 / .084* / .064~ | .63 / .035 / .028 | .74 / .069~ / .025 | .88 / .058 / .017 |
| corr | raw_p | .68 / .503* / .536* | .76 / .627* / .619* | .87 / .799* / .797* | .90 / .872* / .883* | .96 / .919* / .942* |
| corr | tau | .28 / .028 / .031 | .28 / .022 / .033 | .28 / .026 / .025 | .30 / .027 / .025 | .30 / .028 / .025 |
| mscr_eq | raw_p | .95 / .867* / .869* | .99 / .948* / .944* | nig | nig | nig |
| mscr_eq | tau | .56 / .074* / .075~ | .53 / .050 / .072~ | nig | nig | nig |
| mscr_native | raw_p | .95 / .834* / .803* | .97 / .931* / .919* | nig | nig | nig |
| mscr_native | tau | .77 / .104* / .097* | .80 / .094* / .114* | nig | nig | nig |
| notears | tau | .29 / .035 / .019 | .33 / .042 / .050 | .33 / .039 / .039 | .30 / .033 / .025 | .29 / .030 / .017 |
| pc_eq | tau | .21 / .027 / .022 | .30 / .065* / .058~ | .42 / .051 / .033 | .49 / .039 / .067~ | .54 / .048 / .042~ |
| pc_native | tau | .27 / .033 / .025 | .31 / .038 / .050 | .27 / .024 / .033 | .00 / .000 / .000 | .00 / .000 / .000 |
| pcorr_eq | raw_p | .43 / .050 / .050 | .51 / .060 / .061~ | .68 / .057 / .056~ | .74 / .048 / .050~ | .90 / .047 / .033 |
| pcorr_eq | tau | .37 / .024 / .017 | .49 / .049 / .056~ | .66 / .046 / .044 | .72 / .031 / .025 | .91 / .070* / .042 |
| pcorr_eq_min | raw_p | .43 / .054 / .056~ | .51 / .056 / .058~ | .68 / .057 / .058~ | .73 / .039 / .033 | .90 / .058 / .033 |
| pcorr_eq_min | tau | .38 / .025 / .031 | .52 / .064* / .067~ | .68 / .061 / .064~ | .72 / .017 / .008 | .90 / .061~ / .050~ |
| pcorr_hac | raw_p | .57 / .358* / .319* | .63 / .464* / .414* | .80 / .683* / .672* | .83 / .759* / .792* | .89 / .842* / .883* |
| pcorr_hac | tau | .24 / .021 / .019 | .26 / .038 / .042 | .28 / .055 / .039 | .28 / .052~ / .050~ | .28 / .045 / .042~ |
| pcorr_hac_eq_min | raw_p | .36 / .056 / .064~ | .43 / .057 / .058~ | .60 / .057 / .061~ | .68 / .041 / .042 | .83 / .058 / .033 |
| pcorr_hac_eq_min | tau | .33 / .034 / .039 | .44 / .073* / .067~ | .60 / .056 / .061~ | .65 / .019 / .008 | .84 / .061~ / .050~ |
| pcorr_hac_fb | raw_p | .57 / .357* / .317* | .63 / .461* / .414* | .80 / .683* / .672* | .83 / .759* / .792* | .89 / .842* / .883* |
| pcorr_hac_fb | tau | .24 / .023 / .019 | .26 / .037 / .042 | .28 / .056 / .039 | .28 / .050 / .050~ | .28 / .045 / .042~ |
| pcorr_hac_fb_eq_min | raw_p | .35 / .056 / .058~ | .43 / .057 / .058~ | .60 / .057 / .061~ | .68 / .041 / .042 | .83 / .058 / .033 |
| pcorr_hac_fb_eq_min | tau | .33 / .034 / .042 | .45 / .074* / .067~ | .60 / .056 / .061~ | .65 / .019 / .008 | .84 / .061~ / .050~ |
| pcorr_native | raw_p | .61 / .347* / .319* | .67 / .461* / .422* | .83 / .691* / .661* | .86 / .764* / .817* | .91 / .852* / .892* |
| pcorr_native | tau | .29 / .028 / .025 | .31 / .035 / .036 | .33 / .042 / .025 | .32 / .036 / .050~ | .33 / .027 / .050~ |
| rcot2_eq | raw_p | .57 / .123* / .117* | .64 / .104* / .108* | .75 / .222* / .181* | .83 / .356* / .342* | .90 / .583* / .542* |
| rcot2_eq | tau | .48 / .062* / .058~ | .51 / .042 / .061~ | .55 / .022 / .003 | .57 / .027 / .033 | .62 / .037 / .058~ |
| rcot2_eq_min | raw_p | .57 / .129* / .108* | .63 / .095* / .089* | .76 / .227* / .214* | .85 / .377* / .408* | .92 / .569* / .617* |
| rcot2_eq_min | tau | .46 / .065* / .089* | .52 / .036 / .044 | .57 / .032 / .033 | .58 / .023 / .025 | .57 / .019 / .025 |
| rcot2_native | raw_p | .59 / .166* / .158* | .66 / .184* / .203* | .77 / .308* / .297* | .86 / .448* / .517* | .93 / .670* / .667* |
| rcot2_native | tau | .43 / .029 / .019 | .53 / .039 / .042 | .54 / .019 / .017 | .55 / .019 / .042 | .56 / .023 / .050~ |
| shap_dag | tau | .77 / .037 / .022 | .90 / .049 / .044 | .95 / .024 / .033 | .97 / .056 / .067~ | .98 / .037 / .042 |
| two_tower | tau | .56 / .062~ / .047 | .42 / .014 / .006 | .63 / .052 / .064~ | .57 / .034 / .017 | .59 / .031 / .033 |

### E3 R1

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | raw_p | 1.00 / .066~ / .030 | 1.00 / .050 / .020 | 1.00 / .050 / .040~ | nig | nig |
| pmrt_nl_eq | tau | 1.00 / .081* / .040~ | 1.00 / .056~ / .020 | 1.00 / .078* / .050~ | nig | nig |
| pmrt_eq | raw_p | 1.00 / .050 / .020 | 1.00 / .044 / .030~ | 1.00 / .041 / .050~ | 1.00 / .050 / .080~ | 1.00 / .041 / .090~ |
| pmrt_eq | tau | 1.00 / .047 / .020 | 1.00 / .022 / .020 | 1.00 / .047 / .040~ | 1.00 / .050 / .060~ | 1.00 / .062~ / .130~ |
| pmrt_r3 | raw_p | 1.00 / .053~ / .060~ | 1.00 / .041 / .030 | 1.00 / .059~ / .080~ | 1.00 / .044 / .070~ | 1.00 / .041 / .080~ |
| pmrt_r3 | tau | 1.00 / .059~ / .070~ | 1.00 / .031 / .030 | 1.00 / .062~ / .070~ | 1.00 / .053~ / .070~ | 1.00 / .075~ / .140* |
| corr | raw_p | 1.00 / .066~ / .060~ | 1.00 / .087* / .050~ | 1.00 / .037 / .040~ | 1.00 / .047~ / .050~ | 1.00 / .041 / .040~ |
| corr | tau | 1.00 / .062~ / .060~ | 1.00 / .113* / .090~ | 1.00 / .044~ / .040~ | 1.00 / .066~ / .120* | 1.00 / .050 / .050~ |
| granger_eq | raw_p | 1.00 / .047 / .050~ | 1.00 / .034 / .030 | 1.00 / .044 / .050~ | 1.00 / .044 / .060~ | 1.00 / .044 / .090~ |
| granger_eq | tau | 1.00 / .041 / .040~ | 1.00 / .028 / .030 | 1.00 / .050 / .050~ | 1.00 / .078~ / .100~ | 1.00 / .072~ / .130* |
| granger_native | raw_p | 1.00 / .069~ / .060~ | 1.00 / .087* / .050~ | 1.00 / .037 / .040~ | 1.00 / .047~ / .050~ | 1.00 / .041 / .040~ |
| granger_native | tau | 1.00 / .059~ / .060~ | 1.00 / .113* / .080~ | 1.00 / .044~ / .040~ | 1.00 / .066~ / .120* | 1.00 / .050 / .050~ |
| mscr_eq | raw_p | 1.00 / .062~ / .060~ | 1.00 / .041 / .060~ | nig | nig | nig |
| mscr_eq | tau | 1.00 / .072~ / .070~ | 1.00 / .041 / .060~ | nig | nig | nig |
| mscr_native | raw_p | 1.00 / .056~ / .020 | 1.00 / .044 / .060~ | nig | nig | nig |
| mscr_native | tau | 1.00 / .050~ / .020 | 1.00 / .047 / .050~ | nig | nig | nig |
| notears | tau | 1.00 / .000 / .000 | 1.00 / .000 / .000 | 1.00 / .000 / .000 | 1.00 / .000 / .000 | 1.00 / .000 / .000 |
| pc_eq | tau | 1.00 / .025 / .060~ | 1.00 / .056~ / .040~ | 1.00 / .056~ / .040~ | 1.00 / .050~ / .100* | 1.00 / .050~ / .040~ |
| pc_native | tau | 1.00 / .016 / .050~ | 1.00 / .050 / .050~ | 1.00 / .059~ / .050~ | 1.00 / .047~ / .100* | 1.00 / .047~ / .040~ |
| pcorr_eq | raw_p | 1.00 / .047 / .050~ | 1.00 / .034 / .030 | 1.00 / .044 / .050~ | 1.00 / .044 / .060~ | 1.00 / .044 / .090~ |
| pcorr_eq | tau | 1.00 / .041 / .040~ | 1.00 / .028 / .030 | 1.00 / .050 / .050~ | 1.00 / .078~ / .100~ | 1.00 / .072~ / .130* |
| pcorr_eq_min | raw_p | 1.00 / .053 / .070~ | 1.00 / .053 / .040~ | 1.00 / .044 / .070~ | 1.00 / .053~ / .060~ | 1.00 / .044 / .080~ |
| pcorr_eq_min | tau | 1.00 / .053 / .070~ | 1.00 / .041 / .040~ | 1.00 / .069~ / .090~ | 1.00 / .028 / .050~ | 1.00 / .069~ / .130* |
| pcorr_hac | raw_p | 1.00 / .047 / .080~ | 1.00 / .047 / .040~ | 1.00 / .044 / .070~ | 1.00 / .056~ / .060~ | 1.00 / .044 / .070~ |
| pcorr_hac | tau | 1.00 / .044 / .060~ | 1.00 / .044 / .040~ | 1.00 / .069~ / .080~ | 1.00 / .028 / .050~ | 1.00 / .066~ / .120* |
| pcorr_hac_eq_min | raw_p | 1.00 / .047 / .080~ | 1.00 / .047 / .040~ | 1.00 / .044 / .070~ | 1.00 / .056~ / .060~ | 1.00 / .044 / .070~ |
| pcorr_hac_eq_min | tau | 1.00 / .044 / .060~ | 1.00 / .044 / .040~ | 1.00 / .069~ / .080~ | 1.00 / .028 / .050~ | 1.00 / .066~ / .120* |
| pcorr_hac_fb | raw_p | 1.00 / .047 / .080~ | 1.00 / .044 / .040~ | 1.00 / .041 / .070~ | 1.00 / .056~ / .060~ | 1.00 / .044 / .070~ |
| pcorr_hac_fb | tau | 1.00 / .047 / .060~ | 1.00 / .044 / .040~ | 1.00 / .069~ / .080~ | 1.00 / .028 / .050~ | 1.00 / .066~ / .120* |
| pcorr_hac_fb_eq_min | raw_p | 1.00 / .047 / .080~ | 1.00 / .044 / .040~ | 1.00 / .041 / .070~ | 1.00 / .056~ / .060~ | 1.00 / .044 / .070~ |
| pcorr_hac_fb_eq_min | tau | 1.00 / .047 / .060~ | 1.00 / .044 / .040~ | 1.00 / .069~ / .080~ | 1.00 / .028 / .050~ | 1.00 / .066~ / .120* |
| pcorr_native | raw_p | 1.00 / .053 / .070~ | 1.00 / .053 / .040~ | 1.00 / .044 / .070~ | 1.00 / .053~ / .060~ | 1.00 / .044 / .080~ |
| pcorr_native | tau | 1.00 / .053 / .070~ | 1.00 / .041 / .040~ | 1.00 / .069~ / .090~ | 1.00 / .028 / .050~ | 1.00 / .069~ / .130* |
| rcot2_eq | raw_p | 1.00 / .087* / .140* | 1.00 / .062~ / .100~ | 1.00 / .050~ / .070~ | 1.00 / .050 / .050~ | 1.00 / .047 / .090~ |
| rcot2_eq | tau | 1.00 / .056~ / .110~ | 1.00 / .056~ / .040~ | 1.00 / .091* / .150* | 1.00 / .078~ / .080~ | 1.00 / .037 / .030 |
| rcot2_eq_min | raw_p | 1.00 / .084* / .110~ | 1.00 / .084* / .070~ | 1.00 / .059~ / .100~ | 1.00 / .053 / .070~ | 1.00 / .059~ / .120~ |
| rcot2_eq_min | tau | 1.00 / .097* / .100~ | 1.00 / .047~ / .050~ | 1.00 / .059~ / .040~ | 1.00 / .056~ / .030 | 1.00 / .078* / .110~ |
| rcot2_native | raw_p | 1.00 / .116* / .050~ | 1.00 / .091* / .070~ | 1.00 / .062~ / .120* | 1.00 / .059~ / .080~ | 1.00 / .062~ / .080~ |
| rcot2_native | tau | 1.00 / .091* / .060~ | 1.00 / .116* / .080~ | 1.00 / .056~ / .110~ | 1.00 / .106* / .080~ | 1.00 / .128* / .130* |
| shap_dag | tau | 1.00 / .034 / .060~ | 1.00 / .109* / .070~ | 1.00 / .034 / .030 | 1.00 / .103* / .080~ | 1.00 / .066~ / .040~ |
| two_tower | tau | 1.00 / .072* / .010 | 1.00 / .062~ / .010 | 1.00 / .053 / .010 | 1.00 / .066~ / .020 | 1.00 / .144* / .040~ |

### E3 R2

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | raw_p | 1.00 / .047 / .043 | 1.00 / .035 / .033 | 1.00 / .043 / .043 | nig | nig |
| pmrt_nl_eq | tau | 1.00 / .050 / .043 | 1.00 / .036 / .037 | 1.00 / .041 / .053~ | nig | nig |
| pmrt_eq | raw_p | .97 / .044 / .047~ | .96 / .046 / .073~ | .97 / .044 / .027 | 1.00 / .050 / .050~ | 1.00 / .047 / .070~ |
| pmrt_eq | tau | .97 / .052 / .067~ | .96 / .037 / .057~ | .97 / .020 / .010 | 1.00 / .059~ / .050~ | .99 / .047 / .060~ |
| pmrt_r3 | raw_p | 1.00 / .046 / .060~ | 1.00 / .047 / .053~ | 1.00 / .043 / .027 | 1.00 / .044 / .060~ | 1.00 / .041 / .070~ |
| pmrt_r3 | tau | 1.00 / .052 / .067~ | 1.00 / .049 / .053~ | 1.00 / .037 / .020 | 1.00 / .034 / .050~ | 1.00 / .084* / .110* |
| corr | raw_p | 1.00 / .680* / .593* | 1.00 / .786* / .707* | 1.00 / .917* / .877* | 1.00 / .953* / .900* | 1.00 / .959* / .940* |
| corr | tau | 1.00 / .317* / .170* | 1.00 / .325* / .163* | 1.00 / .315* / .153* | 1.00 / .356* / .160* | 1.00 / .344* / .170* |
| granger_eq | raw_p | 1.00 / .037 / .053~ | 1.00 / .053 / .037 | 1.00 / .046 / .017 | 1.00 / .053~ / .060~ | 1.00 / .044 / .070~ |
| granger_eq | tau | 1.00 / .033 / .047 | 1.00 / .037 / .033 | 1.00 / .040 / .017 | 1.00 / .072~ / .070~ | 1.00 / .059~ / .090~ |
| granger_native | raw_p | 1.00 / .445* / .297* | 1.00 / .548* / .390* | 1.00 / .750* / .640* | 1.00 / .884* / .800* | 1.00 / .928* / .900* |
| granger_native | tau | 1.00 / .325* / .180* | 1.00 / .318* / .177* | 1.00 / .328* / .167* | 1.00 / .359* / .160* | 1.00 / .366* / .170* |
| mscr_eq | raw_p | 1.00 / 1.000* / 1.000* | 1.00 / 1.000* / 1.000* | nig | nig | nig |
| mscr_eq | tau | 1.00 / .192* / .030 | 1.00 / .171* / .030 | nig | nig | nig |
| mscr_native | raw_p | 1.00 / .996* / .997* | 1.00 / 1.000* / 1.000* | nig | nig | nig |
| mscr_native | tau | 1.00 / .252* / .083~ | 1.00 / .265* / .057~ | nig | nig | nig |
| notears | tau | 1.00 / .180* / .000 | 1.00 / .189* / .003 | 1.00 / .188* / .000 | 1.00 / .188* / .000 | 1.00 / .188* / .000 |
| pc_eq | tau | 1.00 / .048 / .040 | 1.00 / .040 / .023 | 1.00 / .014 / .017 | 1.00 / .034 / .030 | 1.00 / .034 / .010 |
| pc_native | tau | 1.00 / .136* / .043 | 1.00 / .114* / .047 | 1.00 / .143* / .060~ | 1.00 / .131* / .040~ | 1.00 / .166* / .100~ |
| pcorr_eq | raw_p | 1.00 / .037 / .053~ | 1.00 / .053 / .037 | 1.00 / .046 / .017 | 1.00 / .053~ / .060~ | 1.00 / .044 / .070~ |
| pcorr_eq | tau | 1.00 / .033 / .047 | 1.00 / .037 / .033 | 1.00 / .040 / .017 | 1.00 / .072~ / .070~ | 1.00 / .059~ / .090~ |
| pcorr_eq_min | raw_p | 1.00 / .040 / .063~ | 1.00 / .046 / .037 | 1.00 / .044 / .027 | 1.00 / .041 / .060~ | 1.00 / .044 / .070~ |
| pcorr_eq_min | tau | 1.00 / .041 / .063~ | 1.00 / .044 / .030 | 1.00 / .033 / .027 | 1.00 / .034 / .050~ | 1.00 / .119* / .120* |
| pcorr_hac | raw_p | 1.00 / .226* / .043 | 1.00 / .277* / .057~ | 1.00 / .348* / .100* | 1.00 / .434* / .160* | 1.00 / .509* / .360* |
| pcorr_hac | tau | 1.00 / .228* / .043 | 1.00 / .292* / .083~ | 1.00 / .281* / .043 | 1.00 / .319* / .050~ | 1.00 / .412* / .110~ |
| pcorr_hac_eq_min | raw_p | 1.00 / .042 / .067~ | 1.00 / .044 / .037 | 1.00 / .046 / .033 | 1.00 / .041 / .070~ | 1.00 / .047 / .070~ |
| pcorr_hac_eq_min | tau | 1.00 / .042 / .067~ | 1.00 / .048 / .040 | 1.00 / .033 / .027 | 1.00 / .034 / .050~ | 1.00 / .113* / .130* |
| pcorr_hac_fb | raw_p | 1.00 / .225* / .040 | 1.00 / .276* / .057~ | 1.00 / .348* / .100* | 1.00 / .434* / .160* | 1.00 / .509* / .360* |
| pcorr_hac_fb | tau | 1.00 / .229* / .043 | 1.00 / .292* / .083~ | 1.00 / .281* / .043 | 1.00 / .319* / .050~ | 1.00 / .412* / .110~ |
| pcorr_hac_fb_eq_min | raw_p | 1.00 / .042 / .067~ | 1.00 / .044 / .033 | 1.00 / .046 / .033 | 1.00 / .041 / .070~ | 1.00 / .047 / .070~ |
| pcorr_hac_fb_eq_min | tau | 1.00 / .042 / .067~ | 1.00 / .048 / .040 | 1.00 / .033 / .027 | 1.00 / .034 / .050~ | 1.00 / .113* / .130* |
| pcorr_native | raw_p | 1.00 / .224* / .037 | 1.00 / .274* / .043 | 1.00 / .347* / .087~ | 1.00 / .438* / .160* | 1.00 / .506* / .330* |
| pcorr_native | tau | 1.00 / .230* / .043 | 1.00 / .303* / .083~ | 1.00 / .292* / .043 | 1.00 / .331* / .070~ | 1.00 / .425* / .130~ |
| rcot2_eq | raw_p | .98 / .106* / .130* | 1.00 / .067~ / .103* | 1.00 / .231* / .200* | 1.00 / .375* / .360* | 1.00 / .697* / .670* |
| rcot2_eq | tau | .93 / .051 / .037 | .97 / .033 / .023 | 1.00 / .093* / .063~ | 1.00 / .069~ / .060~ | 1.00 / .069~ / .010 |
| rcot2_eq_min | raw_p | .98 / .084* / .087* | 1.00 / .074* / .063~ | 1.00 / .124* / .127* | 1.00 / .222* / .250* | 1.00 / .528* / .520* |
| rcot2_eq_min | tau | .86 / .022 / .030 | .96 / .031 / .027 | .99 / .036 / .053~ | 1.00 / .056~ / .060~ | 1.00 / .041 / .030 |
| rcot2_native | raw_p | 1.00 / .159* / .110* | 1.00 / .199* / .133* | 1.00 / .221* / .257* | 1.00 / .409* / .420* | 1.00 / .691* / .640* |
| rcot2_native | tau | .97 / .032 / .023 | 1.00 / .061 / .050~ | 1.00 / .098* / .110* | 1.00 / .097* / .080~ | 1.00 / .087* / .080~ |
| shap_dag | tau | 1.00 / .271* / .083* | 1.00 / .252* / .040 | 1.00 / .232* / .047 | 1.00 / .237* / .050~ | 1.00 / .228* / .020 |
| two_tower | tau | .99 / .229* / .043 | .99 / .246* / .053~ | .99 / .244* / .047~ | .97 / .287* / .050~ | .97 / .275* / .040~ |

### E4 R1 l1

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | - | nig | nig | nig | nig | nig |
| pmrt_eq | raw_p | .70 / - / .050~ | .90 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .100~ |
| pmrt_eq | tau | .45 / - / .000 | .65 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .100~ |
| pmrt_r3 | raw_p | .75 / - / .050~ | .90 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .100~ |
| pmrt_r3 | tau | .45 / - / .000 | .65 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .100~ |
| corr | raw_p | .75 / - / .050~ | .95 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .100~ |
| corr | tau | .40 / - / .000 | .60 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .050~ |
| mscr_eq | raw_p | .15 / - / .050~ | .20 / - / .100~ | nig | nig | nig |
| mscr_eq | tau | .15 / - / .050~ | .10 / - / .050~ | nig | nig | nig |
| mscr_native | raw_p | .30 / - / .050~ | .35 / - / .100~ | nig | nig | nig |
| mscr_native | tau | .20 / - / .050~ | .00 / - / .000 | nig | nig | nig |
| notears | tau | .40 / - / .000 | .60 / - / .000 | .90 / - / .000 | .85 / - / .000 | .95 / - / .000 |
| pc_eq | tau | .40 / - / .000 | .60 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .100~ |
| pc_native | tau | .40 / - / .000 | .60 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .100~ |
| pcorr_eq | raw_p | .70 / - / .100~ | .95 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .100~ |
| pcorr_eq | tau | .45 / - / .000 | .55 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .100~ |
| pcorr_eq_min | raw_p | .75 / - / .100~ | .95 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .100~ |
| pcorr_eq_min | tau | .40 / - / .000 | .60 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .100~ |
| pcorr_hac | raw_p | .65 / - / .100~ | .95 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .100~ |
| pcorr_hac | tau | .55 / - / .000 | .60 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .100~ |
| pcorr_hac_eq_min | raw_p | .65 / - / .100~ | .95 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .100~ |
| pcorr_hac_eq_min | tau | .55 / - / .000 | .60 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .100~ |
| pcorr_hac_fb | raw_p | .65 / - / .100~ | .95 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .100~ |
| pcorr_hac_fb | tau | .55 / - / .000 | .60 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .100~ |
| pcorr_hac_fb_eq_min | raw_p | .65 / - / .100~ | .95 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .100~ |
| pcorr_hac_fb_eq_min | tau | .55 / - / .000 | .60 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .100~ |
| pcorr_native | raw_p | .75 / - / .100~ | .95 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .100~ |
| pcorr_native | tau | .40 / - / .000 | .60 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .100~ |
| rcot2_eq | raw_p | .45 / - / .050~ | .55 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .050~ |
| rcot2_eq | tau | .30 / - / .000 | .45 / - / .050~ | .95 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .200~ |
| rcot2_eq_min | raw_p | .45 / - / .000 | .50 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .100~ | 1.00 / - / .050~ |
| rcot2_eq_min | tau | .30 / - / .000 | .55 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .150~ |
| rcot2_native | raw_p | .45 / - / .000 | .50 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .100~ | 1.00 / - / .050~ |
| rcot2_native | tau | .30 / - / .000 | .55 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .150~ |
| shap_dag | tau | .30 / - / .100~ | .40 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .000 | 1.00 / - / .000 |
| two_tower | tau | .75 / - / .050~ | .75 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ |

### E4 R2 l1

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | - | nig | nig | nig | nig | nig |
| pmrt_eq | raw_p | .10 / - / .067~ | .12 / - / .100~ | .25 / - / .067~ | .30 / - / .000 | .95 / - / .100~ |
| pmrt_eq | tau | .03 / - / .000 | .10 / - / .050~ | .53 / - / .233* | .25 / - / .000 | 1.00 / - / .200~ |
| pmrt_r3 | raw_p | .10 / - / .050~ | .12 / - / .100~ | .23 / - / .067~ | .30 / - / .000 | .95 / - / .100~ |
| pmrt_r3 | tau | .03 / - / .000 | .08 / - / .067~ | .48 / - / .183* | .30 / - / .000 | 1.00 / - / .200~ |
| corr | raw_p | .20 / - / .017 | .45 / - / .033~ | .98 / - / .083~ | 1.00 / - / .050~ | 1.00 / - / .300* |
| corr | tau | .30 / - / .067~ | .50 / - / .050~ | .82 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| mscr_eq | raw_p | .05 / - / .050~ | .10 / - / .083~ | nig | nig | nig |
| mscr_eq | tau | .02 / - / .033~ | .17 / - / .083~ | nig | nig | nig |
| mscr_native | raw_p | .05 / - / .033~ | .13 / - / .083~ | nig | nig | nig |
| mscr_native | tau | .03 / - / .033~ | .05 / - / .033~ | nig | nig | nig |
| notears | tau | .17 / - / .017 | .12 / - / .017 | .03 / - / .000 | .00 / - / .000 | .00 / - / .000 |
| pc_eq | tau | .08 / - / .050~ | .05 / - / .000 | .18 / - / .017 | .60 / - / .250~ | 1.00 / - / .100~ |
| pc_native | tau | .30 / - / .067~ | .50 / - / .050~ | .98 / - / .033~ | 1.00 / - / .150~ | 1.00 / - / .100~ |
| pcorr_eq | raw_p | .12 / - / .050~ | .10 / - / .117~ | .23 / - / .067~ | .35 / - / .000 | .95 / - / .050~ |
| pcorr_eq | tau | .05 / - / .033~ | .10 / - / .117~ | .50 / - / .167* | .25 / - / .000 | 1.00 / - / .200~ |
| pcorr_eq_min | raw_p | .12 / - / .050~ | .10 / - / .117~ | .23 / - / .067~ | .35 / - / .000 | .95 / - / .100~ |
| pcorr_eq_min | tau | .05 / - / .017 | .10 / - / .117~ | .47 / - / .183* | .30 / - / .000 | 1.00 / - / .200~ |
| pcorr_hac | raw_p | .28 / - / .050~ | .45 / - / .050~ | .97 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac | tau | .37 / - / .083~ | .50 / - / .083~ | 1.00 / - / .117~ | 1.00 / - / .050~ | 1.00 / - / .050~ |
| pcorr_hac_eq_min | raw_p | .12 / - / .050~ | .10 / - / .117~ | .22 / - / .083~ | .30 / - / .000 | .95 / - / .100~ |
| pcorr_hac_eq_min | tau | .05 / - / .017 | .10 / - / .100~ | .48 / - / .183* | .30 / - / .000 | 1.00 / - / .200~ |
| pcorr_hac_fb | raw_p | .28 / - / .050~ | .45 / - / .050~ | .97 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_fb | tau | .37 / - / .083~ | .50 / - / .083~ | 1.00 / - / .117~ | 1.00 / - / .050~ | 1.00 / - / .050~ |
| pcorr_hac_fb_eq_min | raw_p | .12 / - / .050~ | .10 / - / .100~ | .22 / - / .083~ | .30 / - / .000 | .95 / - / .100~ |
| pcorr_hac_fb_eq_min | tau | .07 / - / .017 | .10 / - / .100~ | .48 / - / .183* | .30 / - / .000 | 1.00 / - / .200~ |
| pcorr_native | raw_p | .28 / - / .050~ | .47 / - / .050~ | .97 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_native | tau | .40 / - / .083~ | .50 / - / .117~ | 1.00 / - / .117~ | 1.00 / - / .050~ | 1.00 / - / .050~ |
| rcot2_eq | raw_p | .17 / - / .083~ | .07 / - / .100~ | .08 / - / .067~ | .25 / - / .000 | .20 / - / .150~ |
| rcot2_eq | tau | .17 / - / .100~ | .07 / - / .067~ | .17 / - / .217* | .20 / - / .100~ | .15 / - / .100~ |
| rcot2_eq_min | raw_p | .12 / - / .050~ | .05 / - / .100~ | .10 / - / .050~ | .15 / - / .050~ | .20 / - / .100~ |
| rcot2_eq_min | tau | .02 / - / .067~ | .05 / - / .067~ | .15 / - / .100~ | .15 / - / .150~ | .20 / - / .050~ |
| rcot2_native | raw_p | .25 / - / .067~ | .27 / - / .050~ | .37 / - / .050~ | .65 / - / .000 | .95 / - / .150~ |
| rcot2_native | tau | .32 / - / .167* | .35 / - / .100~ | .58 / - / .117~ | .70 / - / .150~ | .95 / - / .100~ |
| shap_dag | tau | .05 / - / .050~ | .10 / - / .017 | .08 / - / .017 | .40 / - / .000 | .60 / - / .000 |
| two_tower | tau | .13 / - / .100~ | .18 / - / .000 | .98 / - / .100~ | .95 / - / .050~ | 1.00 / - / .050~ |

### E4 R3 l0

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | - | nig | nig | nig | nig | nig |
| pmrt_eq | raw_p | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .100~ |
| pmrt_eq | tau | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pmrt_r3 | raw_p | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .100~ |
| pmrt_r3 | tau | 1.00 / - / .083~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| corr | raw_p | .87 / - / .033~ | 1.00 / - / .033~ | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .050~ |
| corr | tau | .58 / - / .000 | .95 / - / .033~ | 1.00 / - / .017 | 1.00 / - / .050~ | 1.00 / - / .100~ |
| mscr_eq | raw_p | 1.00 / - / .050~ | 1.00 / - / .017 | nig | nig | nig |
| mscr_eq | tau | 1.00 / - / .067~ | 1.00 / - / .017 | nig | nig | nig |
| mscr_native | raw_p | 1.00 / - / .067~ | 1.00 / - / .017 | nig | nig | nig |
| mscr_native | tau | 1.00 / - / .017 | 1.00 / - / .033~ | nig | nig | nig |
| notears | tau | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pc_eq | tau | .92 / - / .017 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ |
| pc_native | tau | .87 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ |
| pcorr_eq | raw_p | 1.00 / - / .083~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .100~ |
| pcorr_eq | tau | 1.00 / - / .150* | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .000 |
| pcorr_eq_min | raw_p | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .017 | 1.00 / - / .050~ | 1.00 / - / .100~ |
| pcorr_eq_min | tau | 1.00 / - / .133~ | 1.00 / - / .033~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .000 |
| pcorr_hac | raw_p | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .017 | 1.00 / - / .050~ | 1.00 / - / .100~ |
| pcorr_hac | tau | 1.00 / - / .150* | 1.00 / - / .033~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .000 |
| pcorr_hac_eq_min | raw_p | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .017 | 1.00 / - / .050~ | 1.00 / - / .100~ |
| pcorr_hac_eq_min | tau | 1.00 / - / .150* | 1.00 / - / .033~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .000 |
| pcorr_hac_fb | raw_p | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .017 | 1.00 / - / .050~ | 1.00 / - / .100~ |
| pcorr_hac_fb | tau | 1.00 / - / .150* | 1.00 / - / .033~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .000 |
| pcorr_hac_fb_eq_min | raw_p | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .017 | 1.00 / - / .050~ | 1.00 / - / .100~ |
| pcorr_hac_fb_eq_min | tau | 1.00 / - / .150* | 1.00 / - / .033~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .000 |
| pcorr_native | raw_p | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .017 | 1.00 / - / .050~ | 1.00 / - / .100~ |
| pcorr_native | tau | 1.00 / - / .133~ | 1.00 / - / .033~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .000 |
| rcot2_eq | raw_p | .97 / - / .050~ | .98 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 |
| rcot2_eq | tau | .92 / - / .050~ | .87 / - / .000 | 1.00 / - / .067~ | 1.00 / - / .000 | 1.00 / - / .000 |
| rcot2_eq_min | raw_p | .98 / - / .067~ | .98 / - / .017 | 1.00 / - / .033~ | 1.00 / - / .000 | 1.00 / - / .000 |
| rcot2_eq_min | tau | .97 / - / .050~ | .93 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ |
| rcot2_native | raw_p | .98 / - / .067~ | .98 / - / .017 | 1.00 / - / .033~ | 1.00 / - / .000 | 1.00 / - / .000 |
| rcot2_native | tau | .97 / - / .050~ | .93 / - / .000 | 1.00 / - / .017 | 1.00 / - / .000 | 1.00 / - / .050~ |
| shap_dag | tau | .70 / - / .067~ | .67 / - / .017 | 1.00 / - / .100~ | 1.00 / - / .000 | 1.00 / - / .000 |
| two_tower | tau | .53 / - / .033~ | .87 / - / .067~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .150~ |

### E4 R3 l0.5

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | - | nig | nig | nig | nig | nig |
| pmrt_eq | raw_p | 1.00 / - / .033~ | 1.00 / - / .067~ | 1.00 / - / .033~ | 1.00 / - / .000 | 1.00 / - / .050~ |
| pmrt_eq | tau | 1.00 / - / .067~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pmrt_r3 | raw_p | 1.00 / - / .050~ | 1.00 / - / .083~ | 1.00 / - / .017 | 1.00 / - / .000 | 1.00 / - / .050~ |
| pmrt_r3 | tau | 1.00 / - / .083~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| corr | raw_p | 1.00 / - / .033~ | 1.00 / - / .033~ | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .050~ |
| corr | tau | 1.00 / - / .000 | 1.00 / - / .017 | 1.00 / - / .033~ | 1.00 / - / .050~ | 1.00 / - / .100~ |
| mscr_eq | raw_p | 1.00 / - / .033~ | 1.00 / - / .067~ | nig | nig | nig |
| mscr_eq | tau | 1.00 / - / .033~ | 1.00 / - / .050~ | nig | nig | nig |
| mscr_native | raw_p | 1.00 / - / .083~ | 1.00 / - / .033~ | nig | nig | nig |
| mscr_native | tau | 1.00 / - / .033~ | 1.00 / - / .033~ | nig | nig | nig |
| notears | tau | .00 / - / .000 | .00 / - / .000 | .00 / - / .000 | .00 / - / .000 | .00 / - / .000 |
| pc_eq | tau | 1.00 / - / .033~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ |
| pc_native | tau | 1.00 / - / .017 | 1.00 / - / .033~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ |
| pcorr_eq | raw_p | 1.00 / - / .050~ | 1.00 / - / .067~ | 1.00 / - / .033~ | 1.00 / - / .000 | 1.00 / - / .050~ |
| pcorr_eq | tau | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_eq_min | raw_p | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ |
| pcorr_eq_min | tau | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac | raw_p | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ |
| pcorr_hac | tau | 1.00 / - / .100~ | 1.00 / - / .033~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_eq_min | raw_p | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ |
| pcorr_hac_eq_min | tau | 1.00 / - / .100~ | 1.00 / - / .033~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_fb | raw_p | 1.00 / - / .083~ | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ |
| pcorr_hac_fb | tau | 1.00 / - / .100~ | 1.00 / - / .033~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_fb_eq_min | raw_p | 1.00 / - / .083~ | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ |
| pcorr_hac_fb_eq_min | tau | 1.00 / - / .100~ | 1.00 / - / .033~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_native | raw_p | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ |
| pcorr_native | tau | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| rcot2_eq | raw_p | .95 / - / .100~ | .97 / - / .033~ | .98 / - / .017 | 1.00 / - / .000 | 1.00 / - / .000 |
| rcot2_eq | tau | .82 / - / .067~ | .88 / - / .050~ | 1.00 / - / .133~ | 1.00 / - / .000 | 1.00 / - / .000 |
| rcot2_eq_min | raw_p | .98 / - / .050~ | .98 / - / .017 | 1.00 / - / .033~ | 1.00 / - / .000 | 1.00 / - / .000 |
| rcot2_eq_min | tau | .93 / - / .083~ | .93 / - / .000 | 1.00 / - / .183* | 1.00 / - / .000 | 1.00 / - / .000 |
| rcot2_native | raw_p | .98 / - / .067~ | .98 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 |
| rcot2_native | tau | .92 / - / .050~ | .93 / - / .000 | 1.00 / - / .150* | 1.00 / - / .000 | 1.00 / - / .050~ |
| shap_dag | tau | 1.00 / - / .133~ | 1.00 / - / .000 | 1.00 / - / .017 | 1.00 / - / .050~ | 1.00 / - / .200~ |
| two_tower | tau | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .100~ |

### E4 R3 l1

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | - | nig | nig | nig | nig | nig |
| pmrt_eq | raw_p | 1.00 / - / .067~ | 1.00 / - / .067~ | 1.00 / - / .033~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pmrt_eq | tau | 1.00 / - / .067~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pmrt_r3 | raw_p | 1.00 / - / .083~ | 1.00 / - / .067~ | 1.00 / - / .033~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pmrt_r3 | tau | 1.00 / - / .067~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| corr | raw_p | 1.00 / - / .017 | 1.00 / - / .033~ | 1.00 / - / .067~ | 1.00 / - / .050~ | 1.00 / - / .050~ |
| corr | tau | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .050~ |
| mscr_eq | raw_p | 1.00 / - / .083~ | 1.00 / - / .017 | nig | nig | nig |
| mscr_eq | tau | 1.00 / - / .000 | 1.00 / - / .017 | nig | nig | nig |
| mscr_native | raw_p | 1.00 / - / .050~ | 1.00 / - / .017 | nig | nig | nig |
| mscr_native | tau | 1.00 / - / .000 | 1.00 / - / .033~ | nig | nig | nig |
| notears | tau | .00 / - / .000 | .00 / - / .000 | .00 / - / .000 | .00 / - / .000 | .00 / - / .000 |
| pc_eq | tau | 1.00 / - / .067~ | 1.00 / - / .000 | 1.00 / - / .067~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pc_native | tau | 1.00 / - / .067~ | 1.00 / - / .000 | 1.00 / - / .100~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_eq | raw_p | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .033~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_eq | tau | 1.00 / - / .100~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_eq_min | raw_p | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .067~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_eq_min | tau | 1.00 / - / .083~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac | raw_p | 1.00 / - / .117~ | 1.00 / - / .050~ | 1.00 / - / .067~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac | tau | 1.00 / - / .083~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_eq_min | raw_p | 1.00 / - / .117~ | 1.00 / - / .050~ | 1.00 / - / .067~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_eq_min | tau | 1.00 / - / .083~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_fb | raw_p | 1.00 / - / .117~ | 1.00 / - / .050~ | 1.00 / - / .067~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_fb | tau | 1.00 / - / .083~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_fb_eq_min | raw_p | 1.00 / - / .117~ | 1.00 / - / .050~ | 1.00 / - / .067~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_fb_eq_min | tau | 1.00 / - / .083~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_native | raw_p | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .067~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_native | tau | 1.00 / - / .083~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| rcot2_eq | raw_p | .88 / - / .100~ | .98 / - / .033~ | 1.00 / - / .033~ | 1.00 / - / .000 | 1.00 / - / .000 |
| rcot2_eq | tau | .57 / - / .133~ | .75 / - / .083~ | 1.00 / - / .083~ | 1.00 / - / .000 | 1.00 / - / .000 |
| rcot2_eq_min | raw_p | .97 / - / .050~ | .98 / - / .017 | 1.00 / - / .033~ | 1.00 / - / .000 | 1.00 / - / .050~ |
| rcot2_eq_min | tau | .80 / - / .050~ | .83 / - / .000 | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .050~ |
| rcot2_native | raw_p | .98 / - / .050~ | .98 / - / .017 | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ |
| rcot2_native | tau | .80 / - / .050~ | .87 / - / .000 | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .050~ |
| shap_dag | tau | 1.00 / - / .000 | 1.00 / - / .033~ | 1.00 / - / .117~ | 1.00 / - / .000 | 1.00 / - / .100~ |
| two_tower | tau | 1.00 / - / .133~ | 1.00 / - / .017 | 1.00 / - / .200* | 1.00 / - / .050~ | 1.00 / - / .050~ |

### E4 R3 l1.5

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | - | nig | nig | nig | nig | nig |
| pmrt_eq | raw_p | 1.00 / - / .067~ | 1.00 / - / .083~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pmrt_eq | tau | 1.00 / - / .067~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pmrt_r3 | raw_p | 1.00 / - / .083~ | 1.00 / - / .067~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pmrt_r3 | tau | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| corr | raw_p | 1.00 / - / .017 | 1.00 / - / .033~ | 1.00 / - / .083~ | 1.00 / - / .050~ | 1.00 / - / .050~ |
| corr | tau | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ |
| mscr_eq | raw_p | 1.00 / - / .167* | 1.00 / - / .483* | nig | nig | nig |
| mscr_eq | tau | 1.00 / - / .217* | 1.00 / - / .000 | nig | nig | nig |
| mscr_native | raw_p | 1.00 / - / .150* | 1.00 / - / .567* | nig | nig | nig |
| mscr_native | tau | 1.00 / - / .083~ | 1.00 / - / .000 | nig | nig | nig |
| notears | tau | .00 / - / .000 | .00 / - / .000 | .00 / - / .000 | .00 / - / .000 | .00 / - / .000 |
| pc_eq | tau | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pc_native | tau | 1.00 / - / .117~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_eq | raw_p | 1.00 / - / .117~ | 1.00 / - / .067~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_eq | tau | 1.00 / - / .100~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_eq_min | raw_p | 1.00 / - / .100~ | 1.00 / - / .067~ | 1.00 / - / .067~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_eq_min | tau | 1.00 / - / .083~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac | raw_p | 1.00 / - / .100~ | 1.00 / - / .067~ | 1.00 / - / .067~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac | tau | 1.00 / - / .083~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_eq_min | raw_p | 1.00 / - / .100~ | 1.00 / - / .067~ | 1.00 / - / .067~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_eq_min | tau | 1.00 / - / .083~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_fb | raw_p | 1.00 / - / .100~ | 1.00 / - / .067~ | 1.00 / - / .067~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_fb | tau | 1.00 / - / .083~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_fb_eq_min | raw_p | 1.00 / - / .100~ | 1.00 / - / .067~ | 1.00 / - / .067~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_fb_eq_min | tau | 1.00 / - / .083~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_native | raw_p | 1.00 / - / .100~ | 1.00 / - / .067~ | 1.00 / - / .067~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_native | tau | 1.00 / - / .083~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| rcot2_eq | raw_p | .93 / - / .083~ | 1.00 / - / .033~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 |
| rcot2_eq | tau | .35 / - / .083~ | .43 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .000 | 1.00 / - / .000 |
| rcot2_eq_min | raw_p | .97 / - / .050~ | 1.00 / - / .017 | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ |
| rcot2_eq_min | tau | .67 / - / .067~ | .68 / - / .000 | 1.00 / - / .183* | 1.00 / - / .050~ | 1.00 / - / .000 |
| rcot2_native | raw_p | .98 / - / .033~ | .98 / - / .017 | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ |
| rcot2_native | tau | .55 / - / .067~ | .75 / - / .000 | .98 / - / .150* | 1.00 / - / .000 | 1.00 / - / .050~ |
| shap_dag | tau | 1.00 / - / .017 | 1.00 / - / .017 | 1.00 / - / .033~ | 1.00 / - / .000 | 1.00 / - / .050~ |
| two_tower | tau | 1.00 / - / .217* | 1.00 / - / .017 | 1.00 / - / .017 | 1.00 / - / .000 | 1.00 / - / .000 |

### E4 R4 l0

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | - | nig | nig | nig | nig | nig |
| pmrt_eq | raw_p | NA / - / .000 | NA / - / .000 | NA / - / .000 | NA / - / .100~ | NA / - / .000 |
| pmrt_eq | tau | NA / - / .000 | NA / - / .000 | NA / - / .000 | NA / - / .000 | NA / - / .000 |
| pmrt_r3 | raw_p | NA / - / .000 | NA / - / .000 | NA / - / .000 | NA / - / .100~ | NA / - / .000 |
| pmrt_r3 | tau | NA / - / .000 | NA / - / .000 | NA / - / .000 | NA / - / .000 | NA / - / .000 |
| corr | raw_p | .95 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .100~ | 1.00 / - / .000 |
| corr | tau | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| mscr_eq | raw_p | .35 / - / .000 | .65 / - / .100~ | nig | nig | nig |
| mscr_eq | tau | .30 / - / .000 | .70 / - / .100~ | nig | nig | nig |
| mscr_native | raw_p | .25 / - / .050~ | .45 / - / .050~ | nig | nig | nig |
| mscr_native | tau | .20 / - / .000 | .75 / - / .050~ | nig | nig | nig |
| notears | tau | .95 / - / .000 | .85 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pc_eq | tau | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pc_native | tau | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_eq | raw_p | .95 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .100~ | 1.00 / - / .000 |
| pcorr_eq | tau | .95 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_eq_min | raw_p | .95 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .100~ | 1.00 / - / .000 |
| pcorr_eq_min | tau | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac | raw_p | .95 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .100~ | 1.00 / - / .000 |
| pcorr_hac | tau | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_eq_min | raw_p | .95 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .100~ | 1.00 / - / .000 |
| pcorr_hac_eq_min | tau | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_fb | raw_p | .95 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .100~ | 1.00 / - / .000 |
| pcorr_hac_fb | tau | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_fb_eq_min | raw_p | .95 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .100~ | 1.00 / - / .000 |
| pcorr_hac_fb_eq_min | tau | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_native | raw_p | .95 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .100~ | 1.00 / - / .000 |
| pcorr_native | tau | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| rcot2_eq | raw_p | .80 / - / .050~ | .90 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ |
| rcot2_eq | tau | .70 / - / .000 | .80 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .000 |
| rcot2_eq_min | raw_p | .70 / - / .050~ | .90 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ |
| rcot2_eq_min | tau | .80 / - / .100~ | .80 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .000 |
| rcot2_native | raw_p | .65 / - / .050~ | .90 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ |
| rcot2_native | tau | .80 / - / .100~ | .80 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .000 |
| shap_dag | tau | .70 / - / .000 | .90 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .200~ | 1.00 / - / .000 |
| two_tower | tau | .45 / - / .000 | .95 / - / .150~ | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .000 |

### E4 R4 l0.5

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | - | nig | nig | nig | nig | nig |
| pmrt_eq | raw_p | NA / - / .000 | NA / - / .000 | NA / - / .000 | NA / - / .100~ | NA / - / .000 |
| pmrt_eq | tau | NA / - / .000 | NA / - / .000 | NA / - / .000 | NA / - / .000 | NA / - / .000 |
| pmrt_r3 | raw_p | NA / - / .000 | NA / - / .000 | NA / - / .000 | NA / - / .100~ | NA / - / .000 |
| pmrt_r3 | tau | NA / - / .000 | NA / - / .000 | NA / - / .000 | NA / - / .000 | NA / - / .000 |
| corr | raw_p | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .100~ | 1.00 / - / .000 |
| corr | tau | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| mscr_eq | raw_p | 1.00 / - / .000 | 1.00 / - / .000 | nig | nig | nig |
| mscr_eq | tau | 1.00 / - / .000 | 1.00 / - / .000 | nig | nig | nig |
| mscr_native | raw_p | 1.00 / - / .100~ | 1.00 / - / .050~ | nig | nig | nig |
| mscr_native | tau | 1.00 / - / .050~ | 1.00 / - / .000 | nig | nig | nig |
| notears | tau | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pc_eq | tau | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pc_native | tau | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_eq | raw_p | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .100~ | 1.00 / - / .000 |
| pcorr_eq | tau | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_eq_min | raw_p | 1.00 / - / .100~ | 1.00 / - / .100~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_eq_min | tau | 1.00 / - / .100~ | 1.00 / - / .150~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac | raw_p | 1.00 / - / .100~ | 1.00 / - / .100~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac | tau | 1.00 / - / .100~ | 1.00 / - / .150~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_eq_min | raw_p | 1.00 / - / .100~ | 1.00 / - / .100~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_eq_min | tau | 1.00 / - / .100~ | 1.00 / - / .150~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_fb | raw_p | 1.00 / - / .100~ | 1.00 / - / .100~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_fb | tau | 1.00 / - / .100~ | 1.00 / - / .150~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_fb_eq_min | raw_p | 1.00 / - / .100~ | 1.00 / - / .100~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_fb_eq_min | tau | 1.00 / - / .100~ | 1.00 / - / .150~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_native | raw_p | 1.00 / - / .100~ | 1.00 / - / .100~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_native | tau | 1.00 / - / .100~ | 1.00 / - / .150~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| rcot2_eq | raw_p | 1.00 / - / .000 | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .000 |
| rcot2_eq | tau | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .100~ | 1.00 / - / .000 |
| rcot2_eq_min | raw_p | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .000 |
| rcot2_eq_min | tau | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .150~ | 1.00 / - / .000 |
| rcot2_native | raw_p | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .000 |
| rcot2_native | tau | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .150~ | 1.00 / - / .000 |
| shap_dag | tau | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .000 |
| two_tower | tau | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .150~ | 1.00 / - / .150~ |

### E4 R4 l1

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | - | nig | nig | nig | nig | nig |
| pmrt_eq | raw_p | NA / - / .000 | NA / - / .000 | NA / - / .000 | NA / - / .050~ | NA / - / .000 |
| pmrt_eq | tau | NA / - / .050~ | NA / - / .000 | NA / - / .000 | NA / - / .000 | NA / - / .000 |
| pmrt_r3 | raw_p | NA / - / .000 | NA / - / .000 | NA / - / .000 | NA / - / .000 | NA / - / .000 |
| pmrt_r3 | tau | NA / - / .050~ | NA / - / .000 | NA / - / .000 | NA / - / .000 | NA / - / .000 |
| corr | raw_p | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .000 |
| corr | tau | 1.00 / - / .150~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| mscr_eq | raw_p | 1.00 / - / .100~ | 1.00 / - / .050~ | nig | nig | nig |
| mscr_eq | tau | 1.00 / - / .100~ | 1.00 / - / .000 | nig | nig | nig |
| mscr_native | raw_p | 1.00 / - / .050~ | 1.00 / - / .050~ | nig | nig | nig |
| mscr_native | tau | 1.00 / - / .050~ | 1.00 / - / .050~ | nig | nig | nig |
| notears | tau | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pc_eq | tau | 1.00 / - / .100~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pc_native | tau | 1.00 / - / .150~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_eq | raw_p | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .000 |
| pcorr_eq | tau | 1.00 / - / .150~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_eq_min | raw_p | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_eq_min | tau | 1.00 / - / .050~ | 1.00 / - / .250* | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac | raw_p | 1.00 / - / .100~ | 1.00 / - / .100~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac | tau | 1.00 / - / .050~ | 1.00 / - / .250* | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_eq_min | raw_p | 1.00 / - / .100~ | 1.00 / - / .100~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_eq_min | tau | 1.00 / - / .050~ | 1.00 / - / .250* | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_fb | raw_p | 1.00 / - / .100~ | 1.00 / - / .100~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_fb | tau | 1.00 / - / .050~ | 1.00 / - / .250* | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_fb_eq_min | raw_p | 1.00 / - / .100~ | 1.00 / - / .100~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_fb_eq_min | tau | 1.00 / - / .050~ | 1.00 / - / .250* | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_native | raw_p | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_native | tau | 1.00 / - / .050~ | 1.00 / - / .250* | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 |
| rcot2_eq | raw_p | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .050~ |
| rcot2_eq | tau | 1.00 / - / .150~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .100~ | 1.00 / - / .000 |
| rcot2_eq_min | raw_p | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 |
| rcot2_eq_min | tau | 1.00 / - / .150~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .100~ | 1.00 / - / .050~ |
| rcot2_native | raw_p | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 |
| rcot2_native | tau | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .100~ | 1.00 / - / .050~ |
| shap_dag | tau | 1.00 / - / .200~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .100~ |
| two_tower | tau | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .150~ | 1.00 / - / .000 |

### E4 R4 l1.5

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | - | nig | nig | nig | nig | nig |
| pmrt_eq | raw_p | NA / - / .000 | NA / - / .000 | NA / - / .000 | NA / - / .000 | NA / - / .000 |
| pmrt_eq | tau | NA / - / .050~ | NA / - / .000 | NA / - / .000 | NA / - / .000 | NA / - / .000 |
| pmrt_r3 | raw_p | NA / - / .050~ | NA / - / .000 | NA / - / .000 | NA / - / .000 | NA / - / .000 |
| pmrt_r3 | tau | NA / - / .050~ | NA / - / .000 | NA / - / .000 | NA / - / .000 | NA / - / .000 |
| corr | raw_p | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| corr | tau | 1.00 / - / .150~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| mscr_eq | raw_p | 1.00 / - / .000 | 1.00 / - / .000 | nig | nig | nig |
| mscr_eq | tau | 1.00 / - / .000 | 1.00 / - / .000 | nig | nig | nig |
| mscr_native | raw_p | 1.00 / - / .250~ | 1.00 / - / .650* | nig | nig | nig |
| mscr_native | tau | 1.00 / - / .000 | 1.00 / - / .050~ | nig | nig | nig |
| notears | tau | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pc_eq | tau | 1.00 / - / .100~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pc_native | tau | 1.00 / - / .150~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .000 |
| pcorr_eq | raw_p | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .000 |
| pcorr_eq | tau | 1.00 / - / .100~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_eq_min | raw_p | 1.00 / - / .050~ | 1.00 / - / .150~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ |
| pcorr_eq_min | tau | 1.00 / - / .050~ | 1.00 / - / .150~ | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .000 |
| pcorr_hac | raw_p | 1.00 / - / .050~ | 1.00 / - / .150~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac | tau | 1.00 / - / .050~ | 1.00 / - / .150~ | 1.00 / - / .100~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_eq_min | raw_p | 1.00 / - / .050~ | 1.00 / - / .150~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_eq_min | tau | 1.00 / - / .050~ | 1.00 / - / .150~ | 1.00 / - / .100~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_fb | raw_p | 1.00 / - / .050~ | 1.00 / - / .150~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_fb | tau | 1.00 / - / .050~ | 1.00 / - / .150~ | 1.00 / - / .100~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_fb_eq_min | raw_p | 1.00 / - / .050~ | 1.00 / - / .150~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_hac_fb_eq_min | tau | 1.00 / - / .050~ | 1.00 / - / .150~ | 1.00 / - / .100~ | 1.00 / - / .000 | 1.00 / - / .000 |
| pcorr_native | raw_p | 1.00 / - / .050~ | 1.00 / - / .150~ | 1.00 / - / .000 | 1.00 / - / .000 | 1.00 / - / .050~ |
| pcorr_native | tau | 1.00 / - / .050~ | 1.00 / - / .150~ | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .000 |
| rcot2_eq | raw_p | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ |
| rcot2_eq | tau | 1.00 / - / .150~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .000 |
| rcot2_eq_min | raw_p | 1.00 / - / .000 | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .050~ |
| rcot2_eq_min | tau | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .100~ | 1.00 / - / .050~ |
| rcot2_native | raw_p | 1.00 / - / .000 | 1.00 / - / .100~ | 1.00 / - / .050~ | 1.00 / - / .100~ | 1.00 / - / .050~ |
| rcot2_native | tau | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .000 | 1.00 / - / .150~ | 1.00 / - / .050~ |
| shap_dag | tau | 1.00 / - / .050~ | 1.00 / - / .050~ | 1.00 / - / .150~ | 1.00 / - / .000 | 1.00 / - / .000 |
| two_tower | tau | 1.00 / - / .150~ | 1.00 / - / .000 | 1.00 / - / .150~ | 1.00 / - / .050~ | 1.00 / - / .000 |

### E5 R1

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | raw_p | .78 / .045 / .062~ | .90 / .040 / .025 | 1.00 / .030 / .062~ | nig | nig |
| pmrt_nl_eq | tau | .82 / .075~ / .100~ | .92 / .050~ / .062~ | 1.00 / .060~ / .087~ | nig | nig |
| pmrt_eq | raw_p | .41 / .050~ / .050~ | .47 / .050~ / .000 | .56 / .065~ / .013 | .61 / .060~ / .037 | .67 / .045~ / .062~ |
| pmrt_eq | tau | .39 / .045~ / .025 | .42 / .025 / .000 | .54 / .055~ / .000 | .59 / .025 / .013 | .67 / .045~ / .050~ |
| pmrt_r3 | raw_p | .42 / .055~ / .037~ | .46 / .055~ / .000 | .56 / .075~ / .025 | .61 / .065~ / .025 | .67 / .055~ / .062~ |
| pmrt_r3 | tau | .39 / .050~ / .025 | .42 / .030 / .000 | .53 / .055~ / .000 | .59 / .015 / .013 | .67 / .045~ / .050~ |
| corr | raw_p | .44 / .040~ / .100~ | .47 / .080~ / .000 | .57 / .070~ / .025 | .61 / .060~ / .025 | .67 / .055~ / .062~ |
| corr | tau | .42 / .025 / .050~ | .46 / .035 / .000 | .55 / .060~ / .000 | .61 / .065~ / .025 | .67 / .060~ / .062~ |
| mscr_eq | raw_p | .69 / .055~ / .062~ | .70 / .050~ / .037 | nig | nig | nig |
| mscr_eq | tau | .69 / .075~ / .087~ | .72 / .040 / .037 | nig | nig | nig |
| mscr_native | raw_p | .69 / .040~ / .050~ | .75 / .040 / .025 | nig | nig | nig |
| mscr_native | tau | .70 / .035 / .050~ | .75 / .040 / .037 | nig | nig | nig |
| notears | tau | .37 / .005 / .025 | .37 / .000 / .000 | .33 / .000 / .000 | .33 / .000 / .000 | .33 / .000 / .000 |
| pc_eq | tau | .42 / .030 / .050~ | .44 / .020 / .000 | .50 / .020 / .000 | .52 / .060~ / .025 | .52 / .015 / .025 |
| pc_native | tau | .42 / .030 / .050~ | .44 / .025 / .000 | .50 / .020 / .000 | .52 / .060~ / .025 | .52 / .015 / .025 |
| pcorr_eq | raw_p | .43 / .085~ / .050~ | .48 / .075~ / .013 | .56 / .080~ / .000 | .61 / .080~ / .037 | .67 / .050~ / .050~ |
| pcorr_eq | tau | .43 / .085~ / .050~ | .44 / .045~ / .000 | .55 / .065~ / .000 | .59 / .015 / .013 | .67 / .055~ / .050~ |
| pcorr_eq_min | raw_p | .43 / .075~ / .050~ | .47 / .060~ / .000 | .57 / .080~ / .000 | .61 / .085~ / .037 | .67 / .050~ / .050~ |
| pcorr_eq_min | tau | .45 / .075~ / .050~ | .42 / .030 / .000 | .54 / .060~ / .000 | .59 / .015 / .013 | .67 / .040~ / .050~ |
| pcorr_hac | raw_p | .42 / .085~ / .062~ | .47 / .060~ / .000 | .57 / .080~ / .000 | .62 / .080~ / .037 | .70 / .060~ / .050~ |
| pcorr_hac | tau | .43 / .085~ / .062~ | .42 / .050~ / .000 | .55 / .060~ / .000 | .60 / .020 / .025 | .69 / .040~ / .050~ |
| pcorr_hac_eq_min | raw_p | .42 / .085~ / .062~ | .47 / .060~ / .000 | .57 / .080~ / .000 | .62 / .080~ / .037 | .70 / .060~ / .050~ |
| pcorr_hac_eq_min | tau | .43 / .085~ / .062~ | .42 / .050~ / .000 | .55 / .060~ / .000 | .60 / .020 / .025 | .69 / .040~ / .050~ |
| pcorr_hac_fb | raw_p | .42 / .080~ / .062~ | .47 / .060~ / .000 | .57 / .080~ / .000 | .62 / .080~ / .037 | .70 / .060~ / .050~ |
| pcorr_hac_fb | tau | .42 / .085~ / .062~ | .42 / .050~ / .000 | .55 / .060~ / .000 | .60 / .020 / .025 | .69 / .040~ / .050~ |
| pcorr_hac_fb_eq_min | raw_p | .42 / .080~ / .062~ | .47 / .060~ / .000 | .57 / .080~ / .000 | .62 / .080~ / .037 | .70 / .060~ / .050~ |
| pcorr_hac_fb_eq_min | tau | .42 / .085~ / .062~ | .42 / .050~ / .000 | .55 / .060~ / .000 | .60 / .020 / .025 | .69 / .040~ / .050~ |
| pcorr_native | raw_p | .43 / .075~ / .050~ | .47 / .060~ / .000 | .57 / .080~ / .000 | .61 / .085~ / .037 | .67 / .050~ / .050~ |
| pcorr_native | tau | .45 / .075~ / .050~ | .42 / .030 / .000 | .54 / .060~ / .000 | .59 / .015 / .013 | .67 / .040~ / .050~ |
| rcot2_eq | raw_p | .72 / .105* / .087~ | .75 / .090* / .062~ | .84 / .050~ / .025 | .93 / .040 / .037 | .98 / .055~ / .125* |
| rcot2_eq | tau | .78 / .070~ / .062~ | .78 / .040~ / .037~ | .90 / .100* / .087~ | .91 / .075~ / .050~ | .98 / .080~ / .113~ |
| rcot2_eq_min | raw_p | .74 / .070~ / .087~ | .82 / .035 / .100~ | .90 / .050~ / .075~ | .94 / .080~ / .062~ | .99 / .055~ / .050~ |
| rcot2_eq_min | tau | .77 / .040 / .050~ | .82 / .020 / .062~ | .91 / .065~ / .050~ | .93 / .050~ / .050~ | .98 / .065~ / .113* |
| rcot2_native | raw_p | .72 / .045~ / .125~ | .79 / .040 / .113* | .90 / .045 / .062~ | .93 / .070~ / .062~ | .99 / .050~ / .050~ |
| rcot2_native | tau | .77 / .045 / .075~ | .82 / .015 / .062~ | .91 / .075~ / .062~ | .91 / .025 / .025 | .98 / .075~ / .100~ |
| shap_dag | tau | .93 / .050~ / .113* | .97 / .030 / .025 | .99 / .010 / .050~ | 1.00 / .020 / .062~ | 1.00 / .020 / .100~ |
| two_tower | tau | .77 / .035 / .087~ | .75 / .050 / .075~ | .76 / .095~ / .087~ | .78 / .140* / .113* | .77 / .115* / .100~ |

### E5 R2

| arm | rule | n 500 | n 1000 | n 4000 | n 8000 | n 24000 |
|---|---|---|---|---|---|---|
| pmrt_nl_eq | raw_p | .71 / .053 / .037 | .72 / .040 / .029 | .81 / .045 / .037 | nig | nig |
| pmrt_nl_eq | tau | .76 / .143* / .096* | .71 / .012 / .021 | .81 / .052 / .037 | nig | nig |
| pmrt_eq | raw_p | .52 / .025 / .037 | .54 / .057 / .037 | .62 / .045 / .050 | .68 / .050~ / .050~ | .77 / .045~ / .037 |
| pmrt_eq | tau | .54 / .055 / .071~ | .53 / .047 / .029 | .62 / .048 / .050 | .64 / .035 / .025 | .81 / .060~ / .050~ |
| pmrt_r3 | raw_p | .54 / .025 / .042 | .58 / .055~ / .033 | .66 / .045 / .046 | .69 / .045 / .075~ | .81 / .060~ / .013 |
| pmrt_r3 | tau | .59 / .062~ / .096* | .54 / .022 / .021 | .62 / .012 / .013 | .68 / .040 / .062~ | .82 / .085~ / .062~ |
| corr | raw_p | .78 / .602* / .596* | .86 / .708* / .696* | .91 / .862* / .833* | .89 / .920* / .838* | .94 / .965* / .900* |
| corr | tau | .38 / .005 / .000 | .36 / .002 / .000 | .37 / .002 / .000 | .38 / .005 / .000 | .38 / .005 / .000 |
| mscr_eq | raw_p | 1.00 / 1.000* / 1.000* | 1.00 / 1.000* / 1.000* | nig | nig | nig |
| mscr_eq | tau | .67 / .017 / .013 | .67 / .028 / .017 | nig | nig | nig |
| mscr_native | raw_p | .99 / .983* / .979* | 1.00 / 1.000* / 1.000* | nig | nig | nig |
| mscr_native | tau | .67 / .005 / .008 | .67 / .007 / .004 | nig | nig | nig |
| notears | tau | .43 / .012 / .029 | .46 / .023 / .042 | .45 / .022 / .037 | .47 / .035 / .037 | .46 / .035 / .037 |
| pc_eq | tau | .46 / .022 / .037 | .50 / .043 / .025 | .55 / .032 / .021 | .60 / .035 / .000 | .67 / .030 / .013 |
| pc_native | tau | .44 / .015 / .033 | .47 / .020 / .054~ | .00 / .000 / .000 | .00 / .000 / .000 | .00 / .000 / .000 |
| pcorr_eq | raw_p | .58 / .033 / .042 | .61 / .060~ / .042 | .67 / .043 / .021 | .72 / .040 / .037 | .82 / .055~ / .000 |
| pcorr_eq | tau | .59 / .062~ / .075~ | .56 / .005 / .008 | .65 / .020 / .017 | .71 / .035 / .025 | .85 / .080~ / .050~ |
| pcorr_eq_min | raw_p | .58 / .025 / .046 | .61 / .058~ / .042 | .67 / .043 / .037 | .72 / .035 / .050~ | .82 / .055~ / .000 |
| pcorr_eq_min | tau | .61 / .068~ / .083* | .57 / .023 / .017 | .65 / .025 / .025 | .72 / .030 / .025 | .82 / .075~ / .000 |
| pcorr_hac | raw_p | .62 / .283* / .254* | .69 / .363* / .383* | .82 / .513* / .550* | .85 / .555* / .613* | .92 / .620* / .650* |
| pcorr_hac | tau | .37 / .025 / .046 | .37 / .038 / .058~ | .37 / .030 / .050~ | .38 / .050~ / .062~ | .36 / .045 / .050~ |
| pcorr_hac_eq_min | raw_p | .46 / .030 / .050 | .49 / .055 / .037 | .58 / .045 / .037 | .67 / .030 / .037 | .78 / .055~ / .000 |
| pcorr_hac_eq_min | tau | .50 / .067~ / .079~ | .45 / .022 / .017 | .56 / .023 / .025 | .67 / .030 / .025 | .78 / .075~ / .000 |
| pcorr_hac_fb | raw_p | .61 / .283* / .254* | .69 / .360* / .383* | .82 / .512* / .550* | .85 / .555* / .613* | .92 / .620* / .650* |
| pcorr_hac_fb | tau | .37 / .028 / .046 | .37 / .040 / .054~ | .36 / .028 / .050~ | .37 / .050~ / .062~ | .36 / .045 / .050~ |
| pcorr_hac_fb_eq_min | raw_p | .46 / .028 / .050 | .49 / .055 / .037 | .57 / .045 / .037 | .67 / .030 / .037 | .78 / .055~ / .000 |
| pcorr_hac_fb_eq_min | tau | .50 / .067~ / .079~ | .45 / .022 / .017 | .56 / .023 / .025 | .67 / .030 / .025 | .78 / .075~ / .000 |
| pcorr_native | raw_p | .64 / .273* / .254* | .75 / .355* / .371* | .86 / .513* / .558* | .87 / .565* / .613* | .92 / .630* / .675* |
| pcorr_native | tau | .45 / .033 / .046 | .44 / .033 / .046 | .44 / .030 / .042 | .45 / .060~ / .037 | .43 / .040 / .037 |
| rcot2_eq | raw_p | .69 / .100* / .104* | .72 / .100* / .071~ | .79 / .243* / .237* | .87 / .360* / .375* | .95 / .655* / .700* |
| rcot2_eq | tau | .66 / .025 / .042 | .70 / .045 / .037 | .71 / .048 / .037 | .72 / .055~ / .013 | .71 / .050 / .025 |
| rcot2_eq_min | raw_p | .71 / .077* / .083~ | .73 / .083* / .058~ | .79 / .162* / .171* | .84 / .255* / .287* | .95 / .470* / .550* |
| rcot2_eq_min | tau | .66 / .015 / .017 | .69 / .043 / .029 | .71 / .047 / .037 | .74 / .045 / .125~ | .78 / .045~ / .113~ |
| rcot2_native | raw_p | .74 / .113* / .129* | .76 / .120* / .121* | .88 / .235* / .279* | .93 / .305* / .362* | .98 / .620* / .675* |
| rcot2_native | tau | .68 / .050 / .058~ | .70 / .038 / .029 | .70 / .013 / .021 | .76 / .040 / .050~ | .75 / .030 / .037 |
| shap_dag | tau | .68 / .075* / .046~ | .67 / .023 / .004 | .71 / .047~ / .025 | .76 / .040~ / .050~ | .91 / .045~ / .025 |
| two_tower | tau | .68 / .063~ / .058~ | .70 / .015 / .029 | .81 / .103* / .083~ | .80 / .035 / .013 | .81 / .050~ / .000 |

### V4 claim verdicts

- **Claim: NOT SUPPORTED** (components {'C1': 'SUPPORTED', 'C2a': 'SUPPORTED', 'C2b': 'SUPPORTED', 'C3': 'NOT EVALUABLE'})
- C1: **SUPPORTED** (5 / 6 counted arms of D are design-blind failures; |D| = 6). Wording: design-blind tests are invalid on the R2 (setpoint + dither) design tested.
- C1 sensitivity without mscr_native (R-56, no effect): **SUPPORTED** (4 / 5 counted arms fail)
- Arm labels (R-40; every table): mscr_eq: single-conditioner max statistic; cannot condition on the joint design set; reported INVALID arm, EVAL n <= 1000 (R-54)
- Arm labels (R-40; every table): mscr_native: single-conditioner max statistic; reported INVALID arm, EVAL n <= 1000 (R-54)
- Arm labels (R-40; every table): pmrt_eq: linear statistic (pmrt-core-v1); secondary PMRT arm (R-42)
  - corr: **FAILURE**; R2 INVALID 21/25 (counted 25), pooled R2 null_raw .750 [.734, .766] INVALID; plac_raw .709 [.683, .734] INVALID; R1 INVALID 1/25 (F_max 2)
  - granger_native: **FAILURE**; R2 INVALID 5/5 (counted 5), pooled R2 null_raw .640 [.599, .676] INVALID; plac_raw .516 [.444, .580] INVALID; R1 INVALID 1/5 (F_max 1)
  - mscr_native [single-conditioner max statistic; reported INVALID arm, EVAL n <= 1000 (R-54)]: **FAILURE**; R2 INVALID 8/10 (counted 10), pooled R2 null_raw .945 [.935, .955] INVALID; plac_raw .908 [.892, .923] INVALID; R1 INVALID 0/10 (F_max 1)
  - pcorr_hac_fb: **FAILURE**; R2 INVALID 20/25 (counted 25), pooled R2 null_raw .425 [.399, .448] INVALID; plac_raw .297 [.264, .329] INVALID; R1 INVALID 0/25 (F_max 2)
  - pcorr_native: **FAILURE**; R2 INVALID 20/25 (counted 25), pooled R2 null_raw .424 [.399, .447] INVALID; plac_raw .295 [.261, .327] INVALID; R1 INVALID 0/25 (F_max 2)
  - rcot2_native: **INVALID IN R1**; R2 INVALID 20/25 (counted 25), pooled R2 null_raw .252 [.238, .265] INVALID; plac_raw .238 [.219, .256] INVALID; R1 INVALID 7/25 (F_max 2)
  - pcorr_hac (not in D, descriptive): **FAILURE**; R2 INVALID 20/25 (counted 25), pooled R2 null_raw .426 [.400, .449] INVALID; plac_raw .297 [.264, .330] INVALID; R1 INVALID 0/25 (F_max 2)
- C2a pmrt_nl_eq valid in R1 / R2: **SUPPORTED**
  - cells 24/24, INVALID 0 (F_max 2), VALID 11, INCONCLUSIVE 13; pooled null_raw .043 [.040, .047] VALID; plac_raw .046 [.039, .053] VALID
- C2b eq arms whose native partner is a C1 failure: **SUPPORTED**
  - granger_eq (native granger_native): **SUPPORTED**
    - cells 10/10, INVALID 0 (F_max 1), VALID 3, INCONCLUSIVE 7; pooled null_raw .045 [.038, .052] VALID; plac_raw .046 [.034, .057] VALID
  - pcorr_eq (native pcorr_native): **SUPPORTED**
    - cells 50/50, INVALID 0 (F_max 3), VALID 13, INCONCLUSIVE 37; pooled null_raw .049 [.046, .053] VALID; plac_raw .047 [.041, .053] VALID
  - mscr_eq EXCLUDED from C2b (R-40) [single-conditioner max statistic; cannot condition on the joint design set; reported INVALID arm, EVAL n <= 1000 (R-54)], same rule reported (no effect): **NOT SUPPORTED**
    - cells 20/20, INVALID 8 (F_max 2), VALID 1, INCONCLUSIVE 11; pooled null_raw .730 [.695, .767] INVALID; plac_raw .707 [.677, .737] INVALID
- C2b with the R1-failing partners (R-56, sensitivity, no effect): **SUPPORTED** (added: rcot2_eq)
- C2 wording: using the design restores validity (design-based inference and design-covariate adjustment); rcot2_eq INVALID in R2 as already in R1 (not restored by design covariates).
  - rcot2_eq is INVALID in R2 (pooled truth-null raw rate 0.208 [0.196, 0.220]) as already in R1 (0.067 [0.060, 0.073]): the test is miscalibrated without the design (its native partner rcot2_native is INVALID IN R1), so it is outside C2b's membership rule, and adding design covariates does not make it valid.

Every eq arm, same rule on R1 + R2, unfiltered (R-56):

| eq arm | C2b membership | native C1 | verdict | INVALID / cells (F_max) | R2 INVALID / planned | pooled R1 truth-null | pooled R2 truth-null |
|---|---|---|---|---|---|---|---|
| granger_eq | member | FAILURE | SUPPORTED | 0/10 (1) | 0/5 | 0.043 [0.033, 0.052] | 0.046 [0.037, 0.055] |
| mscr_eq [single-conditioner max statistic; cannot condition on the joint design set; reported INVALID arm, EVAL n <= 1000 (R-54)] | excluded (R-40) | FAILURE | NOT SUPPORTED | 8/20 (2) | 8/10 | 0.048 [0.039, 0.056] | 0.958 [0.950, 0.967] |
| pc_eq | not a member (declare tau) | - | NOT SUPPORTED | 3/50 (2) | - | - | - |
| pcorr_eq | member | FAILURE | SUPPORTED | 0/50 (3) | 0/25 | 0.051 [0.044, 0.059] | 0.048 [0.045, 0.052] |
| rcot2_eq | not a member (native rcot2_native: INVALID IN R1) | INVALID IN R1 | NOT SUPPORTED | 27/50 (3) | 20/25 | 0.067 [0.060, 0.073] | 0.208 [0.196, 0.220] |

- C3 pmrt_nl_eq valid in E4 R3: **NOT EVALUABLE**
  - cells 0/0, INVALID 0 (F_max 0), VALID 0, INCONCLUSIVE 0; pooled 
  - wording: not supported.
- pmrt_eq [linear statistic (pmrt-core-v1); secondary PMRT arm (R-42)] (secondary PMRT arm, same rules reported, no claim effect): C2a: **SUPPORTED**
  - cells 50/50, INVALID 0 (F_max 3), VALID 14, INCONCLUSIVE 36; pooled null_raw .048 [.044, .052] VALID; plac_raw .048 [.042, .054] VALID
  - pmrt_eq: C3: **NOT SUPPORTED**
    - cells 20/20, INVALID 0 (F_max 1), VALID 1, INCONCLUSIVE 19; pooled plac_raw .045 [.018, .082] INCONCLUSIVE; conf_raw .039 [.019, .062] VALID
- pmrt_r3 (secondary PMRT arm, same rules reported, no claim effect): C2a: **SUPPORTED**
  - cells 50/50, INVALID 1 (F_max 3), VALID 12, INCONCLUSIVE 37; pooled null_raw .050 [.046, .053] VALID; plac_raw .049 [.043, .056] VALID
  - pmrt_r3: C3: **NOT SUPPORTED**
    - cells 20/20, INVALID 0 (F_max 1), VALID 2, INCONCLUSIVE 18; pooled plac_raw .048 [.020, .083] INCONCLUSIVE; conf_raw .040 [.019, .064] VALID
- C3 reported for every eq arm (no claim effect): granger_eq NOT EVALUABLE; mscr_eq [single-conditioner max statistic; cannot condition on the joint design set; reported INVALID arm, EVAL n <= 1000 (R-54)] NOT SUPPORTED; pc_eq NOT SUPPORTED; pcorr_eq NOT SUPPORTED; rcot2_eq NOT SUPPORTED

### V1 validity: INVALID / VALID / counted cells per world-regime (all n)

| world regime | pmrt_nl_eq | pmrt_eq | pmrt_r3 | corr | granger_eq | granger_native | mscr_eq | mscr_native | notears | pc_eq | pc_native | pcorr_eq | pcorr_eq_min | pcorr_hac | pcorr_hac_eq_min | pcorr_hac_fb | pcorr_hac_fb_eq_min | pcorr_native | rcot2_eq | rcot2_eq_min | rcot2_native | shap_dag | two_tower |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 R1 | 0/0/3 | 0/1/5 | 0/1/5 | 0/1/5 | - | - | 0/1/2 | 0/0/2 | 0/5/5 | 0/2/5 | **1**/2/5 | 0/1/5 | 0/2/5 | 0/2/5 | 0/2/5 | 0/2/5 | 0/2/5 | 0/2/5 | **1**/2/5 | 0/0/5 | **2**/1/5 | **2**/2/5 | **1**/4/5 |
| E1 R2 | 0/1/3 | 0/3/5 | 0/3/5 | **5**/0/5 | - | - | **2**/0/2 | **2**/0/2 | **5**/0/5 | **2**/3/5 | **1**/3/5 | 0/2/5 | 0/4/5 | **5**/0/5 | 0/4/5 | **5**/0/5 | 0/4/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 |
| E2 R1 | 0/0/3 | 0/0/5 | 0/0/5 | 0/0/5 | - | - | 0/0/2 | 0/0/2 | 0/5/5 | 0/5/5 | 0/4/5 | 0/0/5 | 0/0/5 | 0/0/5 | 0/0/5 | 0/0/5 | 0/0/5 | 0/0/5 | **2**/0/5 | **2**/1/5 | **1**/0/5 | **1**/4/5 | **2**/3/5 |
| E2 R2 | 0/2/3 | 0/3/5 | **1**/2/5 | **5**/0/5 | - | - | **2**/0/2 | **2**/0/2 | 0/5/5 | **1**/4/5 | 0/5/5 | 0/2/5 | 0/2/5 | **5**/0/5 | 0/2/5 | **5**/0/5 | 0/2/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | 0/5/5 | 0/4/5 |
| E3 R1 | 0/1/3 | 0/1/5 | 0/1/5 | **1**/0/5 | 0/1/5 | **1**/0/5 | 0/0/2 | 0/0/2 | 0/5/5 | 0/1/5 | 0/2/5 | 0/1/5 | 0/0/5 | 0/0/5 | 0/0/5 | 0/0/5 | 0/0/5 | 0/0/5 | **1**/0/5 | **2**/0/5 | **3**/0/5 | **2**/2/5 | **2**/1/5 |
| E3 R2 | 0/3/3 | 0/1/5 | 0/1/5 | **5**/0/5 | 0/2/5 | **5**/0/5 | **2**/0/2 | **2**/0/2 | **5**/0/5 | 0/5/5 | **5**/0/5 | 0/2/5 | 0/2/5 | **5**/0/5 | 0/2/5 | **5**/0/5 | 0/2/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 |
| E4 R1 l1 | nig | 0/1/5 | 0/1/5 | 0/1/5 | - | - | 0/0/2 | 0/0/2 | 0/0/5 | 0/0/5 | 0/0/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/2/5 | 0/2/5 | 0/0/5 | 0/0/5 |
| E4 R2 l1 | nig | 0/1/5 | 0/1/5 | **1**/1/5 | - | - | 0/0/2 | 0/0/2 | 0/0/5 | 0/0/5 | 0/0/5 | 0/1/5 | 0/1/5 | 0/2/5 | 0/1/5 | 0/2/5 | 0/1/5 | 0/2/5 | 0/1/5 | 0/0/5 | 0/1/5 | 0/0/5 | 0/0/5 |
| E4 R3 l0 | nig | 0/1/5 | 0/1/5 | 0/0/5 | - | - | 0/0/2 | 0/0/2 | 0/5/5 | 0/3/5 | 0/4/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/0/5 | 0/1/5 | 0/0/5 | 0/3/5 | 0/1/5 |
| E4 R3 l0.5 | nig | 0/0/5 | 0/1/5 | **5**/0/5 | - | - | **2**/0/2 | **2**/0/2 | 0/5/5 | **1**/2/5 | **1**/1/5 | 0/0/5 | 0/0/5 | 0/0/5 | 0/0/5 | 0/0/5 | 0/0/5 | 0/0/5 | **5**/0/5 | 0/0/5 | **1**/0/5 | **5**/0/5 | **5**/0/5 |
| E4 R3 l1 | nig | 0/0/5 | 0/0/5 | **5**/0/5 | - | - | **2**/0/2 | **2**/0/2 | 0/5/5 | **1**/1/5 | **1**/1/5 | 0/0/5 | 0/0/5 | 0/0/5 | 0/0/5 | 0/0/5 | 0/0/5 | 0/0/5 | **5**/0/5 | **3**/0/5 | **3**/0/5 | **5**/0/5 | **5**/0/5 |
| E4 R3 l1.5 | nig | 0/0/5 | 0/0/5 | **5**/0/5 | - | - | **2**/0/2 | **2**/0/2 | 0/5/5 | 0/0/5 | 0/1/5 | 0/0/5 | 0/0/5 | 0/0/5 | 0/0/5 | 0/0/5 | 0/0/5 | 0/0/5 | **5**/0/5 | **3**/0/5 | **3**/0/5 | **5**/0/5 | **5**/0/5 |
| E4 R4 l0 | nig | 0/4/5 | 0/4/5 | 0/1/5 | - | - | 0/1/2 | 0/0/2 | 0/4/5 | 0/2/5 | 0/2/5 | 0/0/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/1/5 | 0/2/5 | 0/3/5 |
| E4 R4 l0.5 | nig | 0/4/5 | 0/4/5 | **5**/0/5 | - | - | **2**/0/2 | **2**/0/2 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 |
| E4 R4 l1 | nig | 0/4/5 | 0/5/5 | **5**/0/5 | - | - | **2**/0/2 | **2**/0/2 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 |
| E4 R4 l1.5 | nig | 0/5/5 | 0/4/5 | **5**/0/5 | - | - | **2**/0/2 | **2**/0/2 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 |
| E5 R1 | 0/1/3 | 0/0/5 | 0/0/5 | 0/0/5 | - | - | 0/0/2 | 0/1/2 | 0/5/5 | 0/4/5 | 0/4/5 | 0/0/5 | 0/0/5 | 0/0/5 | 0/0/5 | 0/0/5 | 0/0/5 | 0/0/5 | **3**/1/5 | 0/0/5 | **1**/0/5 | 0/4/5 | **2**/2/5 |
| E5 R2 | 0/3/3 | 0/3/5 | 0/2/5 | **5**/0/5 | - | - | **2**/0/2 | **2**/0/2 | 0/5/5 | 0/5/5 | 0/5/5 | 0/3/5 | 0/2/5 | **5**/0/5 | 0/4/5 | **5**/0/5 | 0/4/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **5**/0/5 | **1**/1/5 | **1**/2/5 |

(Nx) = planned cells not counted (missing, infeasible, T3-infeasible, untuned or < 10 seeds); nig = not in the arm's grid (pre-registered grid choice, R-54 / R-58: not planned, never missing / PARTIAL).

Not in grid (pre-registered grid choices):

- R-58(1): cdl (regimes ['R3', 'R4']): 0 cells; cdl: training cost; feeds no claim (R1 / R2, n <= 4000 only)
- R-58(1): cdl (ns [8000, 24000]): 0 cells; cdl: training cost; feeds no claim (R1 / R2, n <= 4000 only)
- R-58(1): cdl (kappas [0.125, 0.5]): 0 cells; cdl: not in the kappa sweep
- R-54: mscr_eq, mscr_native (ns [4000, 8000, 24000]): 108 cells; mscr: reported-INVALID arm, EVAL n <= 1000 only
- R-58(5): mscr_eq, mscr_native (kappas [0.125, 0.5]): 0 cells; mscr: not in the kappa sweep
- DEV scope: pmrt_eq, pmrt_r3, pc_eq, pc_native, granger_eq, granger_native, corr, notears, shap_dag, two_tower, pcorr_eq, pcorr_native, pcorr_hac, pcorr_hac_fb, rcot2_eq, rcot2_native, pcorr_eq_min, rcot2_eq_min, pcorr_hac_eq_min, pcorr_hac_fb_eq_min, mscr_eq, mscr_native, pmrt_nl_eq (): 76 cells; the DEV specs ran a subset of the EVAL grid (DEV rendering only)

INVALID cells (578; first 30):
- corr|E1|R2|k0.25|n1000: null_raw .764 [.733, .794]; null_decl .704 [.668, .738]; plac_raw .733 [.679, .788]; plac_decl .629 [.571, .692]
- corr|E1|R2|k0.25|n24000: null_raw .958 [.933, .983]; null_decl .950 [.921, .975]; plac_raw .988 [.963, 1.000]; plac_decl .988 [.963, 1.000]
- corr|E1|R2|k0.25|n4000: null_raw .883 [.864, .903]; null_decl .846 [.821, .871]; plac_raw .896 [.858, .933]; plac_decl .850 [.796, .900]
- corr|E1|R2|k0.25|n500: null_raw .681 [.646, .714]; null_decl .588 [.546, .629]; plac_raw .617 [.562, .671]; plac_decl .479 [.421, .546]
- corr|E1|R2|k0.25|n8000: null_raw .925 [.900, .950]; null_decl .904 [.871, .938]; plac_raw .988 [.963, 1.000]; plac_decl .975 [.938, 1.000]
- corr|E2|R2|k0.25|n1000: null_raw .627 [.602, .653]; null_decl .495 [.468, .524]; plac_raw .619 [.561, .675]; plac_decl .517 [.458, .572]
- corr|E2|R2|k0.25|n24000: null_raw .919 [.897, .941]; null_decl .898 [.872, .922]; plac_raw .942 [.900, .983]; plac_decl .942 [.900, .983]
- corr|E2|R2|k0.25|n4000: null_raw .799 [.779, .820]; null_decl .727 [.704, .751]; plac_raw .797 [.756, .836]; plac_decl .722 [.675, .769]
- corr|E2|R2|k0.25|n500: null_raw .503 [.478, .527]; null_decl .350 [.318, .383]; plac_raw .536 [.478, .594]; plac_decl .369 [.311, .425]
- corr|E2|R2|k0.25|n8000: null_raw .872 [.845, .897]; null_decl .820 [.786, .855]; plac_raw .883 [.825, .942]; plac_decl .842 [.775, .908]
- corr|E3|R1|k0.25|n1000: null_raw .087 [.053, .125]
- corr|E3|R2|k0.25|n1000: null_raw .786 [.752, .818]; null_decl .706 [.666, .746]; plac_raw .707 [.637, .777]; plac_decl .627 [.547, .703]
- corr|E3|R2|k0.25|n24000: null_raw .959 [.931, .984]; null_decl .953 [.922, .981]; plac_raw .940 [.890, .980]; plac_decl .930 [.880, .980]
- corr|E3|R2|k0.25|n4000: null_raw .917 [.889, .941]; null_decl .880 [.851, .905]; plac_raw .877 [.827, .923]; plac_decl .810 [.753, .867]
- corr|E3|R2|k0.25|n500: null_raw .680 [.643, .718]; null_decl .582 [.537, .627]; plac_raw .593 [.520, .667]; plac_decl .457 [.377, .533]
- corr|E3|R2|k0.25|n8000: null_raw .953 [.925, .978]; null_decl .941 [.909, .969]; plac_raw .900 [.820, .970]; plac_decl .890 [.800, .970]
- corr|E4|R2|lam1|k0.25|n24000: plac_raw .300 [.100, .500]; plac_decl .250 [.100, .450]
- corr|E4|R3|lam0.5|k0.25|n1000: conf_raw 1.000 [1.000, 1.000]; conf_decl 1.000 [1.000, 1.000]
- corr|E4|R3|lam0.5|k0.25|n24000: conf_raw 1.000 [1.000, 1.000]; conf_decl 1.000 [1.000, 1.000]
- corr|E4|R3|lam0.5|k0.25|n4000: conf_raw 1.000 [1.000, 1.000]; conf_decl 1.000 [1.000, 1.000]
- corr|E4|R3|lam0.5|k0.25|n500: conf_raw 1.000 [1.000, 1.000]; conf_decl 1.000 [1.000, 1.000]
- corr|E4|R3|lam0.5|k0.25|n8000: conf_raw 1.000 [1.000, 1.000]; conf_decl 1.000 [1.000, 1.000]
- corr|E4|R3|lam1.5|k0.25|n1000: conf_raw 1.000 [1.000, 1.000]; conf_decl 1.000 [1.000, 1.000]
- corr|E4|R3|lam1.5|k0.25|n24000: conf_raw 1.000 [1.000, 1.000]; conf_decl 1.000 [1.000, 1.000]
- corr|E4|R3|lam1.5|k0.25|n4000: conf_raw 1.000 [1.000, 1.000]; conf_decl 1.000 [1.000, 1.000]
- corr|E4|R3|lam1.5|k0.25|n500: conf_raw 1.000 [1.000, 1.000]; conf_decl 1.000 [1.000, 1.000]
- corr|E4|R3|lam1.5|k0.25|n8000: conf_raw 1.000 [1.000, 1.000]; conf_decl 1.000 [1.000, 1.000]
- corr|E4|R3|lam1|k0.25|n1000: conf_raw 1.000 [1.000, 1.000]; conf_decl 1.000 [1.000, 1.000]
- corr|E4|R3|lam1|k0.25|n24000: conf_raw 1.000 [1.000, 1.000]; conf_decl 1.000 [1.000, 1.000]
- corr|E4|R3|lam1|k0.25|n4000: conf_raw 1.000 [1.000, 1.000]; conf_decl 1.000 [1.000, 1.000]

### V2 recall among cells not INVALID (n 500 / 1000 / 4000 / 8000 / 24000; inv INVALID, inf infeasible, mis missing, unt untuned, few < 10 seeds; NA not applicable, R-42)

| world regime | pmrt_nl_eq | pmrt_eq | pmrt_r3 | corr | granger_eq | granger_native | mscr_eq | mscr_native | notears | pc_eq | pc_native | pcorr_eq | pcorr_eq_min | pcorr_hac | pcorr_hac_eq_min | pcorr_hac_fb | pcorr_hac_fb_eq_min | pcorr_native | rcot2_eq | rcot2_eq_min | rcot2_native | shap_dag | two_tower |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 R1 | 1.00/1.00/1.00/nig/nig | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | - | - | 1.00/1.00/nig/nig/nig | 1.00/1.00/nig/nig/nig | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | inv/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | inv/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | inv/1.00/1.00/1.00/inv | 1.00/1.00/inv/inv/1.00 | 1.00/1.00/1.00/1.00/inv |
| E1 R2 | 1.00/1.00/1.00/nig/nig | .98/.99/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | inv/inv/inv/inv/inv | - | - | inv/inv/nig/nig/nig | inv/inv/nig/nig/nig | inv/inv/inv/inv/inv | inv/inv/1.00/1.00/1.00 | inv/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | inv/inv/inv/inv/inv | 1.00/1.00/1.00/1.00/1.00 | inv/inv/inv/inv/inv | 1.00/1.00/1.00/1.00/1.00 | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv |
| E2 R1 | .80/.87/.94/nig/nig | .28/.32/.35/.37/.38 | .29/.32/.35/.37/.38 | .30/.33/.36/.37/.38 | - | - | .87/.93/nig/nig/nig | .88/.94/nig/nig/nig | .28/.26/.25/.25/.25 | .32/.35/.40/.40/.41 | .31/.36/.40/.40/.42 | .29/.32/.36/.37/.38 | .29/.32/.35/.37/.38 | .31/.32/.35/.37/.38 | .31/.32/.35/.37/.38 | .31/.32/.35/.37/.38 | .31/.32/.35/.37/.38 | .29/.32/.35/.37/.38 | inv/inv/.62/.63/.68 | inv/inv/.62/.63/.68 | inv/.56/.62/.63/.68 | inv/.95/.99/1.00/1.00 | .69/.68/.78/inv/inv |
| E2 R2 | .49/.64/.92/nig/nig | .17/.28/.45/.55/.70 | .21/inv/.50/.63/.75 | inv/inv/inv/inv/inv | - | - | inv/inv/nig/nig/nig | inv/inv/nig/nig/nig | .29/.33/.33/.30/.29 | .21/inv/.42/.49/.54 | .27/.31/.27/.00/.00 | .29/.36/.55/.64/.79 | .29/.36/.54/.64/.78 | inv/inv/inv/inv/inv | .22/.28/.42/.53/.71 | inv/inv/inv/inv/inv | .22/.28/.42/.53/.70 | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | .77/.90/.95/.97/.98 | .56/.42/.63/.57/.59 |
| E3 R1 | 1.00/1.00/1.00/nig/nig | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/inv/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/inv/1.00/1.00/1.00 | 1.00/1.00/nig/nig/nig | 1.00/1.00/nig/nig/nig | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | inv/1.00/1.00/1.00/1.00 | inv/inv/1.00/1.00/1.00 | inv/inv/inv/1.00/1.00 | 1.00/inv/1.00/inv/1.00 | inv/1.00/1.00/1.00/inv |
| E3 R2 | 1.00/1.00/1.00/nig/nig | .96/.95/.97/.99/.99 | 1.00/1.00/1.00/1.00/1.00 | inv/inv/inv/inv/inv | 1.00/1.00/1.00/1.00/1.00 | inv/inv/inv/inv/inv | inv/inv/nig/nig/nig | inv/inv/nig/nig/nig | inv/inv/inv/inv/inv | 1.00/1.00/1.00/1.00/1.00 | inv/inv/inv/inv/inv | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | inv/inv/inv/inv/inv | 1.00/1.00/1.00/1.00/1.00 | inv/inv/inv/inv/inv | 1.00/1.00/1.00/1.00/1.00 | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv |
| E4 R1 l1 | nig/nig/nig/nig/nig | .60/.80/1.00/1.00/1.00 | .60/.80/1.00/1.00/1.00 | .55/.85/1.00/1.00/1.00 | - | - | .05/.15/nig/nig/nig | .20/.25/nig/nig/nig | .40/.60/.90/.85/.95 | .40/.60/1.00/1.00/1.00 | .40/.60/1.00/1.00/1.00 | .55/.85/1.00/1.00/1.00 | .55/.85/1.00/1.00/1.00 | .55/.85/1.00/1.00/1.00 | .55/.85/1.00/1.00/1.00 | .55/.85/1.00/1.00/1.00 | .55/.85/1.00/1.00/1.00 | .55/.85/1.00/1.00/1.00 | .30/.40/.95/1.00/1.00 | .20/.40/.95/1.00/1.00 | .20/.40/.95/1.00/1.00 | .30/.40/1.00/1.00/1.00 | .75/.75/1.00/1.00/1.00 |
| E4 R2 l1 | nig/nig/nig/nig/nig | .07/.08/.18/.20/.80 | .07/.08/.18/.15/.80 | .12/.27/.92/1.00/inv | - | - | .02/.02/nig/nig/nig | .02/.10/nig/nig/nig | .17/.12/.03/.00/.00 | .08/.05/.18/.60/1.00 | .30/.50/.98/1.00/1.00 | .07/.05/.18/.25/.85 | .07/.05/.18/.20/.85 | .17/.30/.92/1.00/1.00 | .07/.05/.18/.20/.85 | .17/.28/.92/1.00/1.00 | .07/.05/.18/.20/.85 | .17/.30/.92/1.00/1.00 | .07/.02/.03/.10/.10 | .03/.02/.05/.15/.10 | .05/.13/.25/.50/.95 | .05/.10/.08/.40/.60 | .13/.18/.98/.95/1.00 |
| E4 R3 l0 | nig/nig/nig/nig/nig | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | .70/.97/1.00/1.00/1.00 | - | - | .98/1.00/nig/nig/nig | .98/1.00/nig/nig/nig | 1.00/1.00/1.00/1.00/1.00 | .92/1.00/1.00/1.00/1.00 | .87/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | .92/.97/1.00/1.00/1.00 | .98/.98/1.00/1.00/1.00 | .98/.98/1.00/1.00/1.00 | .70/.67/1.00/1.00/1.00 | .53/.87/1.00/1.00/1.00 |
| E4 R3 l0.5 | nig/nig/nig/nig/nig | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | inv/inv/inv/inv/inv | - | - | inv/inv/nig/nig/nig | inv/inv/nig/nig/nig | .00/.00/.00/.00/.00 | 1.00/1.00/1.00/1.00/inv | 1.00/1.00/1.00/1.00/inv | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | inv/inv/inv/inv/inv | .98/.98/1.00/1.00/1.00 | .98/.98/1.00/1.00/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv |
| E4 R3 l1 | nig/nig/nig/nig/nig | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | inv/inv/inv/inv/inv | - | - | inv/inv/nig/nig/nig | inv/inv/nig/nig/nig | .00/.00/.00/.00/.00 | inv/1.00/1.00/1.00/1.00 | inv/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | inv/inv/inv/inv/inv | .95/.98/inv/inv/inv | .97/.98/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv |
| E4 R3 l1.5 | nig/nig/nig/nig/nig | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | inv/inv/inv/inv/inv | - | - | inv/inv/nig/nig/nig | inv/inv/nig/nig/nig | .00/.00/.00/.00/.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | inv/inv/inv/inv/inv | .95/.97/inv/inv/inv | .98/.98/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv |
| E4 R4 l0 | nig/nig/nig/nig/nig | NA/NA/NA/NA/NA | NA/NA/NA/NA/NA | .95/1.00/1.00/1.00/1.00 | - | - | .20/.50/nig/nig/nig | .15/.40/nig/nig/nig | .95/.85/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | 1.00/1.00/1.00/1.00/1.00 | .95/1.00/1.00/1.00/1.00 | .95/1.00/1.00/1.00/1.00 | .95/1.00/1.00/1.00/1.00 | .95/1.00/1.00/1.00/1.00 | .95/1.00/1.00/1.00/1.00 | .95/1.00/1.00/1.00/1.00 | .95/1.00/1.00/1.00/1.00 | .40/.65/1.00/1.00/1.00 | .40/.70/1.00/1.00/1.00 | .40/.70/1.00/1.00/1.00 | .70/.90/1.00/1.00/1.00 | .45/.95/1.00/1.00/1.00 |
| E4 R4 l0.5 | nig/nig/nig/nig/nig | NA/NA/NA/NA/NA | NA/NA/NA/NA/NA | inv/inv/inv/inv/inv | - | - | inv/inv/nig/nig/nig | inv/inv/nig/nig/nig | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv |
| E4 R4 l1 | nig/nig/nig/nig/nig | NA/NA/NA/NA/NA | NA/NA/NA/NA/NA | inv/inv/inv/inv/inv | - | - | inv/inv/nig/nig/nig | inv/inv/nig/nig/nig | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv |
| E4 R4 l1.5 | nig/nig/nig/nig/nig | NA/NA/NA/NA/NA | NA/NA/NA/NA/NA | inv/inv/inv/inv/inv | - | - | inv/inv/nig/nig/nig | inv/inv/nig/nig/nig | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv |
| E5 R1 | .72/.79/1.00/nig/nig | .34/.37/.46/.55/.63 | .34/.37/.45/.55/.63 | .33/.37/.47/.55/.63 | - | - | .67/.68/nig/nig/nig | .67/.67/nig/nig/nig | .37/.37/.33/.33/.33 | .42/.44/.50/.52/.52 | .42/.44/.50/.52/.52 | .33/.37/.47/.55/.62 | .33/.37/.47/.55/.62 | .33/.37/.47/.55/.62 | .33/.37/.47/.55/.62 | .33/.37/.47/.55/.62 | .33/.37/.47/.55/.62 | .33/.37/.47/.55/.62 | inv/inv/.78/.85/inv | .67/.74/.82/.88/.97 | .67/inv/.82/.87/.97 | .93/.97/.99/1.00/1.00 | .77/.75/.76/inv/inv |
| E5 R2 | .67/.68/.73/nig/nig | .42/.46/.56/.61/.65 | .46/.50/.60/.62/.68 | inv/inv/inv/inv/inv | - | - | inv/inv/nig/nig/nig | inv/inv/nig/nig/nig | .43/.46/.45/.47/.46 | .46/.50/.55/.60/.67 | .44/.47/.00/.00/.00 | .48/.53/.61/.62/.70 | .49/.53/.61/.62/.70 | inv/inv/inv/inv/inv | .39/.41/.48/.54/.64 | inv/inv/inv/inv/inv | .38/.41/.48/.54/.64 | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/inv/inv/inv/inv | inv/.67/.71/.76/.91 | .68/.70/inv/.80/.81 |

### V3 paired recall difference pmrt_nl_eq - arm (neither INVALID, same declaration rule; per-cell 95 % paired-t CI, descriptive)

| arm | rule | block | cells pmrt higher | pmrt lower | CI covers 0 |
|---|---|---|---|---|---|
| corr | BY | primary | 6 | 0 | 5 |
| granger_eq | BY | primary | 0 | 0 | 6 |
| granger_native | BY | secondary | 0 | 0 | 2 |
| mscr_eq | BY | primary | 2 | 2 | 4 |
| mscr_native | BY | secondary | 2 | 2 | 4 |
| notears | tau | primary | 8 | 0 | 4 |
| pc_eq | tau | primary | 8 | 0 | 8 |
| pc_native | tau | secondary | 8 | 0 | 5 |
| pcorr_eq | BY | primary | 12 | 0 | 12 |
| pcorr_eq_min | BY | secondary | 12 | 0 | 12 |
| pcorr_hac | BY | secondary | 6 | 0 | 6 |
| pcorr_hac_eq_min | BY | secondary | 12 | 0 | 12 |
| pcorr_hac_fb | BY | secondary | 6 | 0 | 6 |
| pcorr_hac_fb_eq_min | BY | secondary | 12 | 0 | 12 |
| pcorr_native | BY | secondary | 6 | 0 | 6 |
| pmrt_eq | BY | secondary | 13 | 0 | 11 |
| pmrt_r3 | BY | secondary | 11 | 0 | 12 |
| rcot2_eq | BY | primary | 2 | 0 | 4 |
| rcot2_eq_min | BY | secondary | 2 | 0 | 6 |
| rcot2_native | BY | secondary | 4 | 0 | 2 |
| shap_dag | tau | primary | 2 | 3 | 4 |
| two_tower | tau | primary | 5 | 0 | 6 |

### V5 information levels in R2 (pmrt, eq, eq_min, native): recall (* INVALID, ~ INCONCLUSIVE)

| row | pmrt_nl_eq | pmrt_eq | pmrt_r3 | granger_eq | granger_native | mscr_eq | mscr_native | pc_eq | pc_native | pcorr_eq | pcorr_eq_min | pcorr_hac | pcorr_hac_eq_min | pcorr_hac_fb | pcorr_hac_fb_eq_min | pcorr_native | rcot2_eq | rcot2_eq_min | rcot2_native |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 n500 | 1.00~ | .98~ | 1.00 | - | - | 1.00* | 1.00* | 1.00* | 1.00* | 1.00~ | 1.00 | 1.00* | 1.00 | 1.00* | 1.00 | 1.00* | 1.00* | .97* | 1.00* |
| E1 n1000 | 1.00~ | .99 | 1.00~ | - | - | 1.00* | 1.00* | 1.00* | 1.00 | 1.00~ | 1.00 | 1.00* | 1.00 | 1.00* | 1.00 | 1.00* | 1.00* | .99* | 1.00* |
| E1 n4000 | 1.00 | 1.00 | 1.00 | - | - | - | - | 1.00 | 1.00 | 1.00 | 1.00 | 1.00* | 1.00 | 1.00* | 1.00 | 1.00* | 1.00* | 1.00* | 1.00* |
| E1 n8000 | - | 1.00~ | 1.00~ | - | - | - | - | 1.00 | 1.00~ | 1.00~ | 1.00~ | 1.00* | 1.00~ | 1.00* | 1.00~ | 1.00* | 1.00* | 1.00* | 1.00* |
| E1 n24000 | - | 1.00 | 1.00 | - | - | - | - | 1.00 | 1.00 | 1.00 | 1.00 | 1.00* | 1.00 | 1.00* | 1.00 | 1.00* | 1.00* | 1.00* | 1.00* |
| E2 n500 | .49 | .17 | .21~ | - | - | .94* | .93* | .21 | .27 | .29 | .29~ | .42* | .22~ | .42* | .22~ | .45* | .41* | .39* | .44* |
| E2 n1000 | .64~ | .28 | .33* | - | - | .97* | .97* | .30* | .31 | .36~ | .36~ | .53* | .28~ | .53* | .28~ | .55* | .49* | .50* | .54* |
| E2 n4000 | .92 | .45 | .50 | - | - | - | - | .42 | .27 | .55~ | .54~ | .73* | .42~ | .73* | .42~ | .76* | .63* | .63* | .67* |
| E2 n8000 | - | .55~ | .63~ | - | - | - | - | .49 | .00 | .64~ | .64 | .79* | .53 | .79* | .53 | .83* | .75* | .73* | .77* |
| E2 n24000 | - | .70~ | .75 | - | - | - | - | .54 | .00 | .79 | .78 | .85* | .71 | .85* | .70 | .88* | .86* | .88* | .91* |
| E3 n500 | 1.00 | .96~ | 1.00~ | 1.00~ | 1.00* | 1.00* | 1.00* | 1.00 | 1.00* | 1.00~ | 1.00~ | 1.00* | 1.00~ | 1.00* | 1.00~ | 1.00* | .93* | .90* | .99* |
| E3 n1000 | 1.00 | .95~ | 1.00~ | 1.00 | 1.00* | 1.00* | 1.00* | 1.00 | 1.00* | 1.00 | 1.00 | 1.00* | 1.00 | 1.00* | 1.00 | 1.00* | .97* | .97* | 1.00* |
| E3 n4000 | 1.00 | .97 | 1.00 | 1.00 | 1.00* | - | - | 1.00 | 1.00* | 1.00 | 1.00 | 1.00* | 1.00 | 1.00* | 1.00 | 1.00* | 1.00* | 1.00* | 1.00* |
| E3 n8000 | - | .99~ | 1.00~ | 1.00~ | 1.00* | - | - | 1.00 | 1.00* | 1.00~ | 1.00~ | 1.00* | 1.00~ | 1.00* | 1.00~ | 1.00* | 1.00* | 1.00* | 1.00* |
| E3 n24000 | - | .99~ | 1.00~ | 1.00~ | 1.00* | - | - | 1.00 | 1.00* | 1.00~ | 1.00~ | 1.00* | 1.00~ | 1.00* | 1.00~ | 1.00* | 1.00* | 1.00* | 1.00* |
| E4 n500 | - | .07~ | .07~ | - | - | .02~ | .02~ | .08~ | .30~ | .07~ | .07~ | .17~ | .07~ | .17~ | .07~ | .17~ | .07~ | .03~ | .05~ |
| E4 n1000 | - | .08~ | .08~ | - | - | .02~ | .10~ | .05~ | .50~ | .05~ | .05~ | .30~ | .05~ | .28~ | .05~ | .30~ | .02~ | .02~ | .13~ |
| E4 n4000 | - | .18~ | .18~ | - | - | - | - | .18~ | .98~ | .18~ | .18~ | .92~ | .18~ | .92~ | .18~ | .92~ | .03~ | .05~ | .25~ |
| E4 n8000 | - | .20 | .15 | - | - | - | - | .60~ | 1.00~ | .25 | .20 | 1.00 | .20 | 1.00 | .20 | 1.00 | .10 | .15~ | .50 |
| E4 n24000 | - | .80~ | .80~ | - | - | - | - | 1.00~ | 1.00~ | .85~ | .85~ | 1.00 | .85~ | 1.00 | .85~ | 1.00 | .10~ | .10~ | .95~ |
| E5 n500 | .67 | .42 | .46 | - | - | 1.00* | .98* | .46 | .44 | .48 | .49 | .50* | .39 | .50* | .38 | .55* | .67* | .66* | .69* |
| E5 n1000 | .68 | .46 | .50~ | - | - | 1.00* | 1.00* | .50 | .47 | .53~ | .53~ | .60* | .41 | .59* | .41 | .64* | .68* | .67* | .70* |
| E5 n4000 | .73 | .56 | .60 | - | - | - | - | .55 | .00 | .61 | .61 | .74* | .48 | .74* | .48 | .80* | .71* | .70* | .77* |
| E5 n8000 | - | .61~ | .62~ | - | - | - | - | .60 | .00 | .62 | .62~ | .78* | .54 | .78* | .54 | .84* | .77* | .76* | .84* |
| E5 n24000 | - | .65~ | .68~ | - | - | - | - | .67 | .00 | .70~ | .70~ | .88* | .64~ | .88* | .64~ | .92* | .89* | .88* | .97* |

### V6 E4: placebo_conf rate (raw p, else declared) / wrong-sign rate of the true edge (* INVALID)

| row | pmrt_eq | pmrt_r3 | corr | mscr_eq | mscr_native | notears | pc_eq | pc_native | pcorr_eq | pcorr_eq_min | pcorr_hac | pcorr_hac_eq_min | pcorr_hac_fb | pcorr_hac_fb_eq_min | pcorr_native | rcot2_eq | rcot2_eq_min | rcot2_native | shap_dag | two_tower |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 l1 n1000 | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00 | -/.00~ | -/.00~ | -/.00~ | -/.00~ |
| R1 l1 n4000 | -/.00 | -/.00 | -/.00 | - | - | -/.00~ | -/.00~ | -/.00~ | -/.00 | -/.00 | -/.00 | -/.00 | -/.00 | -/.00 | -/.00 | -/.00~ | -/.00 | -/.00 | -/.00~ | -/.00~ |
| R1 l1 n24000 | -/.00~ | -/.00~ | -/.00~ | - | - | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ |
| R2 l1 n1000 | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.02~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ |
| R2 l1 n4000 | -/.00~ | -/.00~ | -/.00~ | - | - | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ |
| R2 l1 n24000 | -/.00~ | -/.00~ | -/.00* | - | - | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00 | -/.00~ | -/.00 | -/.00~ | -/.00 | -/.00~ | -/.00~ | -/.00~ | -/.00~ | -/.00~ |
| R3 l0 n1000 | .03/.00~ | .03/.00~ | .03/.00~ | .03/.00~ | .08/.00~ | .00/.00 | .00/.00 | .00/.00 | .03/.00~ | .03/.00~ | .03/.00~ | .03/.00~ | .03/.00~ | .03/.00~ | .03/.00~ | .13/.00~ | .07/.00~ | .07/.00~ | .02/.00 | .03/.00~ |
| R3 l0 n4000 | .00/.00 | .00/.00 | .03/.00~ | - | - | .00/.00 | .00/.00 | .00/.00 | .02/.00 | .02/.00 | .02/.00 | .02/.00 | .02/.00 | .02/.00 | .02/.00 | .03/.00~ | .07/.00~ | .05/.00~ | .12/.00~ | .03/.00~ |
| R3 l0 n24000 | .05/.00~ | .05/.00~ | .05/.00~ | - | - | .00/.00 | .15/.00~ | .15/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .15/.00~ | .05/.00~ | .05/.00~ | .00/.00 | .15/.00~ |
| R3 l0.5 n1000 | .05/.00~ | .07/.00~ | 1.00/1.00* | 1.00/.00* | 1.00/.00* | .00/.00 | .02/.00 | .05/.00~ | .05/.00~ | .07/.00~ | .07/.00~ | .07/.00~ | .07/.00~ | .07/.00~ | .07/.00~ | .52/.00* | .08/.00~ | .08/.00~ | 1.00/.00* | 1.00/.00* |
| R3 l0.5 n4000 | .00/.00~ | .00/.00 | 1.00/1.00* | - | - | .00/.00 | .05/.00~ | .10/.00~ | .00/.00~ | .00/.00~ | .00/.00~ | .00/.00~ | .00/.00~ | .00/.00~ | .00/.00~ | .95/.00* | .03/.00~ | .07/.00~ | 1.00/.00* | 1.00/.00* |
| R3 l0.5 n24000 | .10/.00~ | .10/.00~ | 1.00/1.00* | - | - | .00/.00 | .25/.00* | .25/.00* | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .95/.00* | .25/.00~ | .40/.00* | 1.00/.00* | 1.00/.00* |
| R3 l1 n1000 | .07/.00~ | .05/.00~ | 1.00/1.00* | 1.00/.00* | 1.00/.00* | .00/.00 | .02/.00 | .02/.00 | .03/.00~ | .03/.00~ | .03/.00~ | .03/.00~ | .03/.00~ | .03/.00~ | .03/.00~ | .92/.00* | .10/.00~ | .07/.00~ | 1.00/.00* | 1.00/.00* |
| R3 l1 n4000 | .02/.00~ | .03/.00~ | 1.00/1.00* | - | - | .00/.00 | .13/.00~ | .13/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .97/.00* | .40/.00* | .43/.00* | 1.00/.00* | 1.00/.00* |
| R3 l1 n24000 | .10/.00~ | .10/.00~ | 1.00/1.00* | - | - | .00/.00 | .20/.00~ | .15/.00~ | .05/.00~ | .10/.00~ | .10/.00~ | .10/.00~ | .10/.00~ | .10/.00~ | .10/.00~ | 1.00/.00* | .50/.00* | .60/.00* | 1.00/.00* | 1.00/.00* |
| R3 l1.5 n1000 | .07/.00~ | .07/.00~ | 1.00/1.00* | 1.00/.00* | 1.00/.00* | .00/.00 | .03/.00~ | .02/.00 | .03/.00~ | .03/.00~ | .03/.00~ | .03/.00~ | .03/.00~ | .03/.00~ | .03/.00~ | .93/.00* | .10/.00~ | .12/.00~ | 1.00/.00* | 1.00/.00* |
| R3 l1.5 n4000 | .02/.00~ | .02/.00~ | 1.00/1.00* | - | - | .00/.00 | .03/.00~ | .03/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .98/.00* | .30/.00* | .32/.00* | 1.00/.00* | 1.00/.00* |
| R3 l1.5 n24000 | .05/.00~ | .05/.00~ | 1.00/1.00* | - | - | .00/.00 | .15/.00~ | .15/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | 1.00/.00* | .55/.00* | .55/.00* | 1.00/.00* | 1.00/.00* |
| R4 l0 n1000 | .00/NA | .00/NA | .00/.00 | .05/.00~ | .00/.00~ | .00/.00 | .05/.00~ | .05/.00~ | .05/.00~ | .00/.00 | .00/.00 | .00/.00 | .00/.00 | .00/.00 | .00/.00 | .00/.00~ | .00/.00~ | .00/.00~ | .00/.00 | .20/.00~ |
| R4 l0 n4000 | .00/NA | .00/NA | .05/.00~ | - | - | .00/.00 | .00/.00 | .00/.00 | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .05/.00~ | .10/.00~ |
| R4 l0 n24000 | .00/NA | .00/NA | .15/.00~ | - | - | .00/.00 | .10/.00~ | .10/.00~ | .15/.00~ | .10/.00~ | .10/.00~ | .10/.00~ | .10/.00~ | .10/.00~ | .10/.00~ | .10/.00~ | .10/.00~ | .10/.00~ | .00/.00 | .00/.00 |
| R4 l0.5 n1000 | .00/NA | .00/NA | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* |
| R4 l0.5 n4000 | .00/NA | .00/NA | 1.00/1.00* | - | - | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* |
| R4 l0.5 n24000 | .00/NA | .00/NA | 1.00/1.00* | - | - | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* |
| R4 l1 n1000 | .00/NA | .00/NA | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* |
| R4 l1 n4000 | .00/NA | .00/NA | 1.00/1.00* | - | - | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* |
| R4 l1 n24000 | .00/NA | .00/NA | 1.00/1.00* | - | - | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* |
| R4 l1.5 n1000 | .00/NA | .00/NA | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* |
| R4 l1.5 n4000 | .00/NA | .00/NA | 1.00/1.00* | - | - | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* |
| R4 l1.5 n24000 | .00/NA | .00/NA | 1.00/1.00* | - | - | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* | 1.00/1.00* |

### V7 not testable / not applicable

40 (arm, cell) with counts > 0 (JSON V7).

### V8 secondary KPI -> KPI family

602 (arm, cell) rows (JSON V8).

### V9 kappa sweep (R2, n 1000): recall

| row | pmrt_nl_eq | pmrt_eq | pmrt_r3 | corr | granger_eq | granger_native | mscr_eq | mscr_native | notears | pc_eq | pc_native | pcorr_eq | pcorr_eq_min | pcorr_hac | pcorr_hac_eq_min | pcorr_hac_fb | pcorr_hac_fb_eq_min | pcorr_native | rcot2_eq | rcot2_eq_min | rcot2_native | shap_dag | two_tower |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 k0.125 | - | .97~ | 1.00 | 1.00* | - | - | 1.00* | 1.00* | 1.00~ | 1.00~ | 1.00 | 1.00 | 1.00 | 1.00* | 1.00 | 1.00* | 1.00 | 1.00* | 1.00* | 1.00~ | .99* | 1.00* | 1.00* |
| E1 k0.25 | 1.00~ | .99 | 1.00~ | 1.00* | - | - | 1.00* | 1.00* | 1.00* | 1.00* | 1.00 | 1.00~ | 1.00 | 1.00* | 1.00 | 1.00* | 1.00 | 1.00* | 1.00* | .99* | 1.00* | 1.00* | .99* |
| E1 k0.5 | - | 1.00 | 1.00 | 1.00* | - | - | 1.00* | 1.00* | 1.00* | 1.00 | 1.00* | 1.00 | 1.00 | 1.00* | 1.00 | 1.00* | 1.00 | 1.00* | .94* | .95~ | .99* | 1.00* | 1.00* |
| E2 k0.125 | - | .27* | .31~ | .64* | - | - | .97* | .96* | .31 | .28 | .24 | .35~ | .35* | .51* | .26* | .51* | .25* | .52* | .53* | .54* | .55* | .92 | .45 |
| E2 k0.25 | .64~ | .28 | .33* | .68* | - | - | .97* | .97* | .33 | .30* | .31 | .36~ | .36~ | .53* | .28~ | .53* | .28~ | .55* | .49* | .50* | .54* | .90 | .42 |
| E2 k0.5 | - | .24* | .30* | .61* | - | - | .97* | .95* | .31 | .30 | .35 | .32~ | .32~ | .51* | .25~ | .51* | .25~ | .52* | .44* | .44* | .50* | .89 | .49 |
| E3 k0.125 | - | .97~ | 1.00~ | 1.00* | 1.00~ | 1.00* | 1.00* | 1.00* | 1.00* | 1.00 | 1.00* | 1.00~ | 1.00~ | 1.00* | 1.00~ | 1.00* | 1.00~ | 1.00* | .99* | .97* | 1.00* | 1.00* | .99* |
| E3 k0.25 | 1.00 | .95~ | 1.00~ | 1.00* | 1.00 | 1.00* | 1.00* | 1.00* | 1.00* | 1.00 | 1.00* | 1.00 | 1.00 | 1.00* | 1.00 | 1.00* | 1.00 | 1.00* | .97* | .97* | 1.00* | 1.00* | .99* |
| E3 k0.5 | - | .97~ | 1.00~ | 1.00* | 1.00~ | 1.00* | 1.00* | 1.00* | 1.00* | 1.00 | 1.00* | 1.00~ | 1.00~ | 1.00* | 1.00~ | 1.00* | 1.00~ | 1.00* | .84* | .81~ | .99* | 1.00* | .99* |
| E4 k0.125 | - | .00 | .00~ | .20~ | - | - | .00~ | .00~ | .05~ | .00~ | .50~ | .00~ | .00~ | .35~ | .00~ | .35~ | .00~ | .35~ | .05~ | .05~ | .15~ | .10~ | .30~ |
| E4 k0.25 | - | .08~ | .08~ | .27~ | - | - | .02~ | .10~ | .12~ | .05~ | .50~ | .05~ | .05~ | .30~ | .05~ | .28~ | .05~ | .30~ | .02~ | .02~ | .13~ | .10~ | .18~ |
| E4 k0.5 | - | .00 | .00 | .25 | - | - | .00~ | .00~ | .05~ | .00~ | .40~ | .00 | .00 | .30 | .00 | .30 | .00 | .30 | .10~ | .10~ | .25~ | .20~ | .00~ |
| E5 k0.125 | - | .47~ | .54~ | .76* | - | - | 1.00* | 1.00* | .46 | .51~ | .46 | .54~ | .54~ | .57* | .42~ | .57* | .42~ | .62* | .67* | .67~ | .70* | .79~ | .78 |
| E5 k0.25 | .68 | .46 | .50~ | .79* | - | - | 1.00* | 1.00* | .46 | .50 | .47 | .53~ | .53~ | .60* | .41 | .59* | .41 | .64* | .68* | .67* | .70* | .67 | .70 |
| E5 k0.5 | - | .48 | .51~ | .71* | - | - | 1.00* | 1.00* | .46 | .47 | .43 | .52~ | .51~ | .58* | .42~ | .58* | .42~ | .61* | .67~ | .67~ | .68~ | .67 | .67 |

### V10 cost (CPU-s per dataset; mean / max over worlds, regimes; peak RSS MB; infeasible units; T3 = infeasible from DEV cost)

| arm | n 500 | n 1000 | n 4000 | n 8000 | n 24000 | RSS | infeasible |
|---|---|---|---|---|---|---|---|
| pmrt_nl_eq | 35.5 / 64.1 | 54.0 / 97.9 | 145.2 / 289.9 | - | - | 279 | 0 |
| pmrt_eq | 0.6 / 1.3 | 1.0 / 2.3 | 3.9 / 8.3 | 8.5 / 18.4 | 25.3 / 62.5 | 778 | 0 |
| pmrt_r3 | 0.6 / 1.4 | 1.0 / 2.3 | 3.8 / 8.5 | 8.3 / 17.3 | 24.5 / 59.9 | 778 | 0 |
| corr | 0.1 / 0.1 | 0.1 / 0.2 | 0.1 / 0.2 | 0.1 / 0.2 | 0.1 / 0.4 | 538 | 0 |
| granger_eq | 0.2 / 0.3 | 0.3 / 0.4 | 0.8 / 1.0 | 1.7 / 2.3 | 5.7 / 7.4 | 394 | 0 |
| granger_native | 0.1 / 0.1 | 0.1 / 0.1 | 0.1 / 0.2 | 0.2 / 0.2 | 0.4 / 0.4 | 374 | 0 |
| mscr_eq | 22.3 / 87.8 | 53.1 / 192.2 | - | - | - | 348 | 0 |
| mscr_native | 9.2 / 34.0 | 21.8 / 68.6 | - | - | - | 257 | 0 |
| notears | 0.4 / 2.1 | 0.4 / 2.3 | 0.8 / 4.7 | 1.4 / 7.3 | 3.9 / 24.3 | 539 | 0 |
| pc_eq | 2.0 / 15.4 | 2.4 / 19.6 | 3.7 / 33.6 | 4.9 / 42.8 | 10.1 / 79.5 | 539 | 0 |
| pc_native | 0.2 / 1.0 | 0.3 / 1.8 | 0.5 / 3.9 | 0.8 / 7.0 | 1.7 / 18.9 | 539 | 0 |
| pcorr_eq | 0.0 / 0.2 | 0.1 / 0.3 | 0.2 / 0.8 | 0.4 / 1.8 | 1.4 / 6.9 | 332 | 0 |
| pcorr_eq_min | 0.0 / 0.2 | 0.0 / 0.2 | 0.1 / 0.7 | 0.2 / 1.6 | 0.8 / 5.1 | 332 | 0 |
| pcorr_hac | 0.1 / 0.3 | 0.1 / 0.4 | 0.3 / 1.1 | 0.6 / 2.1 | 2.0 / 12.3 | 343 | 0 |
| pcorr_hac_eq_min | 0.1 / 0.3 | 0.1 / 0.4 | 0.3 / 1.4 | 0.6 / 2.2 | 2.0 / 8.9 | 343 | 0 |
| pcorr_hac_fb | 0.8 / 1.8 | 1.0 / 1.6 | 1.2 / 2.4 | 1.5 / 3.3 | 2.7 / 8.5 | 421 | 0 |
| pcorr_hac_fb_eq_min | 0.8 / 1.6 | 1.0 / 1.6 | 1.2 / 2.3 | 1.5 / 3.1 | 2.7 / 7.9 | 421 | 0 |
| pcorr_native | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.1 | 0.1 / 0.2 | 0.3 / 0.9 | 332 | 0 |
| rcot2_eq | 0.5 / 1.6 | 0.6 / 2.4 | 2.0 / 5.8 | 3.3 / 9.9 | 9.0 / 27.0 | 335 | 0 |
| rcot2_eq_min | 0.4 / 1.5 | 0.6 / 2.6 | 1.8 / 5.3 | 3.0 / 7.9 | 7.9 / 23.2 | 335 | 0 |
| rcot2_native | 0.4 / 1.5 | 0.6 / 2.2 | 1.9 / 5.3 | 3.0 / 8.2 | 7.9 / 23.3 | 335 | 0 |
| shap_dag | 2.2 / 4.8 | 3.7 / 7.7 | 13.2 / 25.2 | 26.4 / 48.9 | 81.3 / 148.1 | 556 | 0 |
| two_tower | 5.9 / 12.9 | 6.9 / 11.4 | 15.5 / 29.8 | 27.4 / 54.0 | 75.2 / 155.9 | 723 | 0 |

### V11 integrity

- label PROVISIONAL; failed checks: clean, commits, dataset_hash, freeze_commit_given, one_platform_per_dataset, pkgs_uniform, protocol_frozen_sha, stamps, unexpected
- freeze commit None; amendments []; records used 99200 / expected 99200; missing 0; unexpected 2560; role mismatch 0; duplicates 0
- commit violations 99200; not clean 16104; stamp violations 99200; candidates incomplete 0; pkgs sets 2
- role:status {'measure:ok': 60400, 'tune:ok': 38800}; errors 0 (not listed persistent 0); infeasible 0; dataset-hash mismatches 1 (R-41a: equal across all arms of a dataset); ok records without a dataset hash 0 (not-ok 0); multi-platform datasets 20
- BY recheck (p arms): 73800 / 73800 records agree; tune reproducibility vs DEV: None


> **DEV DATA, NOT EVAL: development seeds 3_000_000-3_000_199 on the DEV specs; a rehearsal of the report, not the pre-registered result**
