READY-TO-MERGE (EVAL report)
# freeze status (2026-10-05): Study A EVAL analysed with the frozen eval_analysis + eval_report, FINAL

Branch xm/freeze = feat/v2 db644e0 (6b720a6; SEED_REGISTRY taken from feat/v2) + A-1 (65cef51) + the EVAL report.
Local only, not pushed. No frozen file edited (LF sha256 at run time = FREEZE_NOTE, all seven checked).

## Amendment A-1 (orchestrator ruling Q1 (a))
- docs/xmethod/PROTOCOL_A_AMENDMENTS.md + specs/eval/amendments.json (A-1, commit 5a95186, 150 960 keys; LF sha
  43be9758...). Written and committed before the analysis ran.

## Run
- `eval_report.py --spec specs/eval/full.json --merged results/eval/merged.jsonl.gz --freeze-commit 93856c2...
  --amendments specs/eval/amendments.json --dev-merged results/dev/{full,ci_c,pmrt_nl,cdl}/merged.jsonl.gz
  --out results/eval` (chunked path of eval_analysis; ~3 GB RAM free, so not the one-shot CLI). Exit 0.
- Outputs: results/eval/REPORT.md, report.json, csv/ (10 files), n1_check.json.

## Results (pre-registered wording only)
- V11 FINAL, failed checks none: 150 960 / 150 960 used; commit violations 0 (A-1 listed); clean; stamps ok;
  0 missing / unexpected / role / duplicates / errors / infeasible; dataset hash equal across arms; one platform per
  dataset; one pkgs set; BY re-check 114 960 / 114 960 agree.
- Claim SUPPORTED: C1, C2a, C2b, C3 all SUPPORTED.
  - C1: 5 of 6 assessable D arms FAILURE; rcot2_native INVALID IN R1 (6/25 R1, F_max 2); pcorr_hac_fb (set D HAC)
    FAILURE. R-56 without mscr_native: SUPPORTED (4 of 5).
  - C2a: pmrt_nl_eq 1 INVALID / 50 (F_max 3), null_raw .045, plac_raw .048 VALID. pmrt_eq, pmrt_r3 also pass.
  - C2b: granger_eq, pcorr_eq SUPPORTED. mscr_eq excluded (R-40), NOT SUPPORTED. R-56 named arm rcot2_eq (R2 .292,
    R1 .061, both INVALID); sensitivity with rcot2_eq added: SUPPORTED (2 of 3).
  - C3: pmrt_nl_eq 0 INVALID / 20 (F_max 2), VALID. pcorr_eq also passes -> wording "so do pcorr_eq; the property
    is not specific to PMRT", PMRT "by construction". pmrt_eq (secondary) NOT SUPPORTED (3 > 2), no effect.
- N1: raw shards (21 jobs, 192 files) list 0 keys with both infeasible and ok (0 infeasible at all).
- T3: 0 infeasible units; max CPU-s per dataset pmrt_nl_eq 1380 (n 24000) < 7200. 12 680 units ran on the
  uncalibrated Kaggle AMD EPYC 7B12 (N2), none near the cap.
- T6: EVAL used 316 host CPU-h (projection 509 Kaggle-ref CPU-h); kaggle 85 199 units, vps 65 761.

## Unexpected (descriptive, no check fails)
- Tune reproducibility vs DEV: dataset sha equal for 38 992 / 39 000 tune records. The 8 others are one dataset,
  E2 R2 n 24000 seed 3000016 (8 arms), DEV commit a7b6e36, same host type (Kaggle Xeon). Declarations equal for all
  39 000. Not investigated further (DEV side).

## Questions
- None open. Q1 answered (orchestrator): amendment A-1.
