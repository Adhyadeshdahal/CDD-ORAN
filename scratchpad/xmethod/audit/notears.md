# F7 audit: notears (auditor aud1, 2026-10-02, feat/v2 b7456c2)

**Verdict: PASS-WITH-NOTES.** Recommendation for the open user decision: include NOTEARS, with the F3 result disclosed.

Sources used:
- Zheng et al. (NeurIPS 2018); the authors' repository xunzheng/notears at 4a9ab19 (`notears/linear.py`, `README.md`, fetched 2026-10-02);
- `cdd_oran/xmethod/methods/notears.py`, `_vendor/notears/`, `docs/xmethod/FIDELITY_CLASSIC.md`.

## Findings (most severe first)

1. **MEDIUM (decision): the borderline F3 is not evidence of a porting error.**
   - **Byte-identity.** I fetched upstream `notears/linear.py` at 4a9ab19 independently. Its sha256 (39644acd...a426f) equals the vendored file.
   - **The adapter's changes.** It changes only the bounds line (`notears.py:79-80`) and calls with w_threshold 0 (`:140`). F2 shows it is array-equal to the vendored function when nothing is forbidden.
   - **What the README promises.** It says "you should see output **like** this" (README l.94). It shows the output of the `__main__` demo as an illustration, not a pinned regression value. No version stack is given.
   - **The miss.** The edge is missed by .004 (W -.296 vs threshold .3; true -.598). The result is the same on 4 scipy and 4 igraph versions. That points to an environment or optimiser-path difference against the authors' unrecorded stack, not to the code.
   - **What I could not do.** I could not rerun the README demo myself: running the upstream `utils.py` was blocked by this session's permission policy. The rerun is left to the user (`scripts/xm_classic_fidelity.py` f3_notears).
   - **Reading.** The gate's purpose ("the implementation is the authors'") is met by F1/F2. Disclose the result as "README example 19/20, one edge .004 below the threshold".
   - **Optional stronger F3.** Report tpr/shd over several seeds of the same README setting, which shows the example is typical rather than exact.

2. **MEDIUM: standardisation is a real deviation, and it matters for E4.**
   - Columns are z-scored (`notears.py:126-127`); the authors only centre. With z-scored l2 loss and lambda1 .1, an edge stays exactly 0 unless |corr| is above about .1.
   - Measured here (E4 R1, n 1000, seed 3_000_000): corr(P0, K0 t+1) = -.092. W(P0, K0) is exactly 0, so the recall 0 in E4 (integration table) is the authors' default L1 penalty on standardised data, not a bug.
   - Without standardisation the result would depend on the arbitrary units of P0 and K0. Z-scoring is the defensible choice: Reisach et al. 2021 recommend it, and it removes var-sortability.
   - It should be stated in the paper as "NOTEARS at authors' defaults on standardised data; edges with |r| < lambda1 are shrunk to 0 by design".

3. **LOW: the score has exact zeros.**
   - The unthresholded |W| is 0 for most nulls because the L-BFGS-B bounds hold the doubled variables at 0. The placebo tau is then often 0, so the rule becomes "declare any nonzero weight". That is reasonable, but note it.
   - Ties at 0 are never declared (`score.placebo_tau`: ties fall below tau).

4. **LOW: background knowledge as (0,0) bounds.**
   - It reuses the authors' own diagonal mechanism, and the forbidden pattern matches pc's (Y->t, into actions except context, action->lag/context).
   - It does change the optimisation problem relative to the authors' unconstrained run. The doc states this; `background_knowledge=False` is available for a sensitivity run.

5. **OK:**
   - Authors' defaults: lambda1 .1, l2, max_iter 100, h_tol 1e-8, rho_max 1e16, w_threshold .3 used for native only (R-13).
   - Sign = sign(W) (R-4).
   - Families and tau rule via `_classic_common` (R-2, R-6).
   - No Truth access; tune() raises if truth_free=False.
   - F4: power .89/.88 (R1), FPR <= .002.

## Recommended fix

- No code change.
- Paper text: disclose the F3 result (finding 1) and the standardisation and lambda1 shrinkage (finding 2).
- Optionally add the multi-seed README-setting check as supporting F3 evidence.
