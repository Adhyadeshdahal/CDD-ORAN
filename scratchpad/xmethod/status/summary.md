READY-TO-MERGE
# summary status (2026-10-05): docs/xmethod/STUDY_SUMMARY.md written (end-of-study record)

Branch xm/freeze = feat/v2 bb87bf2 merged (018cb93) + this commit. Local only, not pushed. No frozen file, report or
manuscript edited.

## What
- docs/xmethod/STUDY_SUMMARY.md (114 lines, ~2 pages), sections:
  1. what each part asked;
  2. protocol, freeze, A-1, audits (commits 93856c2, 5a4ef0f, 6f04f02, 86a9767, 8efa8fd, b278d71, 6ee82f4);
  3. headline results;
  4. artifact locations (results dirs, cddfig figure code, journal figures/ and tables/);
  5. platforms and compute;
  6. seed blocks;
  7. open items.
- Every number is copied from REPORT.md, EXP_B.md s.9, EXP_B_COSTS.md, EXP_C_RUNTIME.md, EXTRAS_REPORT.md, FREEZE_NOTE
  or SEED_REGISTRY, with the file and section cited.

## Compute
- Host CPU-h = sum of record cpu_s per merged file and platform; computed here, read-only.
- Kaggle-reference CPU-h are quoted only where a cost table gives them (Exp B 82.0, X2/X3 63.9, EVAL projection 509).

## Open items listed
- Deferred: E6 effect injection, PMRT-GBM recall-gap diagnostic, Exp B seed replication. VPS /opt/cdd-xm 3.1 GB kept.
- Also listed as facts already reported: the 8 / 39 000 tune dataset-sha differences vs DEV (REPORT V11), and the
  Windows vs Linux generator reproducibility note (extras provenance_platform.json).

## Questions
- None.
