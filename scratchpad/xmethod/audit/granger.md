# F7 audit: granger (auditor aud1, 2026-10-02, feat/v2 b7456c2)

**Verdict: PASS-WITH-NOTES.**

Sources used:
- Granger (1969);
- statsmodels 0.15 `grangercausalitytests` (ssr_ftest), the F2 reference;
- `cdd_oran/xmethod/methods/granger.py`, `docs/xmethod/FIDELITY_CLASSIC.md`.

## Findings

1. **LOW: the test is bivariate (pairwise).**
   - The restricted model holds only the target's own lag (`granger.py:31-42`, `:59`), as in statsmodels. A conditional (multivariate) Granger test that also conditions on the other actions and lags would be the stronger textbook variant, and the closer analogue of R-3's conditioning set.
   - The pairwise form is the standard "simple" baseline, but name it "pairwise Granger" in the paper.
2. **LOW: one lag only.** It is fixed by the one-step row format, as documented. E3's delayed effects beyond one step cannot be seen by any row-based method, so that is not specific to Granger.
3. **INFO: the source can be the target's own lag** (`:66-67`). It is then tested against an intercept-only model, a marginal F-test equal to corr's t-test. That is correct for that candidate.
4. **INFO: the p-value is invalid under R2 serial dependence,** as for corr (placebo declared 2-4x in integration). Disclosed.
5. **OK:**
   - F2: p equals statsmodels ssr_ftest (rel 1e-8).
   - Sign = sign of the coefficient (R-4).
   - BY per family (R-2/R-6).
   - Degenerate fits give NaN.
   - No Truth access.
   - F4: level .049 [.043, .056], power .93 (R1).
   - E3 only, as the contract says.

## Recommended fix

Paper wording only ("pairwise one-lag Granger F-test").
