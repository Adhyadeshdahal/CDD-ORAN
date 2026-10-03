READY-TO-MERGE (branch xm/citests3 from feat/v2 acd3dfa)
# citests3 status  (2026-10-02 ~23:40Z)

## Done
- Kaggle xm-citests-f4h-k1 COMPLETE (exit 0, 12 382 s, 750 records, 0 duplicates; arm native, primary candidates,
  B 9999 Besag-Clifford). Results: scratchpad/xmethod/results/citests/f4/f4_heavy.jsonl, F4_heavy_summary.json.
- docs/xmethod/FIDELITY_CITESTS.md F4 rows filled (no PENDING left):
  - mscr PASS: null .054 [.045, .064] n 500, .049 [.040, .059] n 1000; planted P0 / P1 found (96-100 / 100),
    gated P2 21 / 74, P3 9 / 43 of 100 (n 500 / 1000).
  - pdcor PASS on the null: .048 / .048; planted-data null edges .060 (CI includes .05, borderline); P0 / P1 all
    found, gated P2 45/100, 41/50, P3 0 (n 500 / 1000).
- Disclosed: mscr null n 500 reps 0-83 come from the earlier Colab run (same arm-native code path); numba /
  llvmlite were Kaggle's preinstalled versions (job predates the R-35 pins).
- No code changes on this branch.

## QUESTIONS
- none.
