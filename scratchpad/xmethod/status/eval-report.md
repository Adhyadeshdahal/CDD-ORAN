READY-TO-MERGE
# eval-report status (2026-10-04): Study A EVAL report generator

Branch xm/eval-report from feat/v2 e087160; local only, not pushed. No EVAL seed, no compute, no dependency.

## HAND-BACK
- `scratchpad/xmethod/eval_report.py` (`xm-eval-report/1`): one command, merged records -> `REPORT.md`,
  `report.json` (meta + every table) and `csv/` (9 figure-ready long tables). Usage is in the docstring.
  - EVAL mode needs the EVAL spec; exit 0 only if V11 is FINAL.
  - DEV mode (`--dev --dev-spec ...`) takes arms and blocks from the DEV specs and analysis fields from the EVAL spec.
    Without pmrt_nl_eq records it applies the T10 fallback (pmrt_eq focal) for that rendering only, noted in the
    report. DEV always exits 2.
- REPORT.md is ordered by protocol section:
  1. claim and component wordings;
  2. C1 per D arm (legs, pooled R2 rates, R1 / F_max) + the R-56 sensitivity without mscr;
  3. C2a + the secondary PMRT arms;
  4. C2b as frozen + with the R1-failing partners, the C2 wording, the pre-registered named-arm sentence, and the
     unfiltered eq-arm table;
  5. C3 for the focal arm, every eq arm and the secondary PMRT arms;
  6. power among cells not INVALID + the R4 "not applicable" rows (placebo still read);
  7. kappa sweep;
  8. CSV list with sha256.
  An appendix carries every eval_analysis table (V0-V11).
- DEV watermark:
  - REPORT.md: title, after every section heading, and at the end;
  - report.json: `meta.mode` / `meta.watermark`;
  - every CSV row: first column `data_mode` = `DEV-NOT-EVAL`.
- Memory: one streaming pass splits records by arm into a temp dir; cells are built per arm; verdicts, tables and V11
  run once (record-content checks per arm, cross-arm checks on slim records). DEV, 100 320 records: ~2 min, under
  1 GB. Bit-equal to `eval_analysis.analyse` (tested, incl. defects).
- eval_analysis.py: `analyse` split into `integrity` + new `assemble` (no behaviour change; tests unchanged). Fixed
  a misplaced comment from my R-56 edit (C3_TAU_KEYS).
- DEV rehearsal output: `scratchpad/xmethod/results/eval_dev/` from xm-pmrt `results/dev/{full,ci_c}`; cdl and
  pmrt_nl_eq are not in it yet. Rerun with their merged files and DEV specs when they land.
  - DEV verdicts: C1 SUPPORTED (5 of 6; 4 of 5 without mscr), C2a SUPPORTED, C2b SUPPORTED (pcorr_eq, granger_eq;
    also with rcot2_eq added), C3 NOT SUPPORTED; rcot2_eq is named (R2 .208 vs R1 .067).
- Tests: `tests/test_xmethod_eval_report.py`, 7 pass:
  - chunked == analyse (two shards, gzip, duplicates; and with defects);
  - DEV spec fields + T10 fallback;
  - end-to-end DEV report (watermarks, sections, CSV set, sensitivity line);
  - R4 NA rows;
  - EVAL mode refuses DEV seeds.
  With tests/test_xmethod_eval_analysis.py: 46 pass + 1 skip. ruff clean.

## Findings
- F1 C3 on DEV is NOT SUPPORTED by design: 60 seeds per cell give an exactly valid arm a .26 chance to pass. The
  same simulation sets S_E4 = 300 (T9). It is not a validity signal: pmrt_eq has 0 INVALID cells there and a pooled
  conf_raw of .039.
- F2 Dataset-hash mismatch on DEV: E2 R2 n 24000, seed 3000016 differs between the full run (numpy 2.0.2, pins off)
  and ci_c (numpy 2.4.2, locked); same Kaggle CPU model. 1 of ~16k datasets.
  - EVAL is unaffected: everything is pinned and one platform per dataset.
  - The R-34 tune reproducibility check (`--dev-merged`) against the full DEV run may show such dataset-sha
    differences; read them as a numpy-version effect.
- F3 DEV V11 fails as expected: unfrozen protocol, mixed commits / platforms / numpy, 17 176 dirty records (CI run
  launcher). Every record's candidate set is complete; the BY re-check agrees on 74 920 / 74 920.

## Questions
- Q1 Should eval_report.py be frozen with eval_analysis.py (PROTOCOL_A T7 and FREEZE_NOTE sha list)? It computes
  nothing new, but it renders the reported text. Default: yes, add it to T7 at the freeze; not edited here.
