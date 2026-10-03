# F7 audit: corr (auditor aud1, 2026-10-02, feat/v2 b7456c2)

**Verdict: PASS.**

Sources used: scipy 1.18 `stats.pearsonr` (t-test p), `cdd_oran/xmethod/methods/corr.py`, `docs/xmethod/FIDELITY_CLASSIC.md`.

## Findings

1. **INFO: the p-value assumes i.i.d. rows.** Under R2 (20 setpoint blocks) the rows are serially dependent and the p-values are anti-conservative. In integration the placebo was declared 2-4x, with FDP .5-.7 in R2. This is a property of the method, disclosed and correctly not patched. Report it in the paper as "corr is invalid under R2 by construction".
2. **INFO: corr is marginal,** so it has the wrong sign under confounding (E4 R3/R4), as expected (R-1). That is part of the method.
3. **OK:**
   - Score |r|, sign sign(r) (R-4), p from pearsonr.
   - BY per family through `_classic_common.declare` (R-2/R-6/R-10: primary, kpi and diagnostic families separate).
   - Degenerate columns give NaN, never declared.
   - No Truth access.
   - F4: raw-p level .050 [.044, .057], power .92 (R1, n 1000, 200 reps).

## Recommended fix

None.
