# F7 audit: pc (auditor aud1, 2026-10-02, feat/v2 b7456c2)

**Verdict: PASS-WITH-NOTES.**

Sources used:
- Spirtes, Glymour, Scheines (2000);
- causal-learn 0.1.4.8 source (`utils/PCUtils/SkeletonDiscovery.py`, read in the venv);
- pcalg (Kalisch et al. 2012, JSS 47(11)) for the precedent of the `pMax` edge score;
- `cdd_oran/xmethod/methods/pc.py`, `docs/xmethod/FIDELITY_CLASSIC.md`.

## Findings (most severe first)

1. **MEDIUM: score definition.**
   - **What it is.** The score is -log10 of the largest CI p-value recorded for the pair (`pc.py:42-80`, `:150-151`). This is pcalg's `pMax` ("maximal p-value over all CI tests"), so it has a standard precedent. Cite pcalg in F6 rather than calling it ad hoc.
   - **What it means in stable mode.** I read causal-learn's PC-stable loop (`SkeletonDiscovery.py`, no break at a removal):
     - a removed pair's pMax is the maximum over every test up to and including its removal depth, so it is >= alpha;
     - a kept pair's pMax is the maximum over all its tests, so it is <= alpha.

     So native-at-alpha is the same as score > -log10(alpha), as documented.
   - **Caveat.** The tau rule is not PC at another alpha. At a looser tau, a removed pair's pMax covers only the conditioning sets tried before its removal. At a stricter tau, the conditioning sets come from the alpha = .05 adjacencies. The doc states this (F6 (2)); keep it in the paper.
   - **Consequence.** F4 R1 null-edge FPR is .121 [.107, .138], because the tau from 30 placebo scores fell to p < .13 (see the cross-cutting note in the status file).

2. **MEDIUM: Fisher-z with the pseudo-inverse in deterministic worlds.**
   - In E1/E3, up to 494 of 2348 tests hit a singular correlation sub-matrix. They are recomputed with pinv (`pc.py:55-66`). For an exactly singular set the partial correlation is not defined, and the pinv value is one arbitrary choice: |r| near 1 gives p near 0.
   - This does not touch the candidate readout directly, but it changes which conditioning sets remove edges.
   - **Recommendation.** Report the fallback fraction per cell (it is already in `notes`), and treat E1/E3 PC results as conditional on this choice. The alternative, causal-learn raising an error, would leave PC unscorable there.

3. **LOW: causal-learn quirks with no effect on results.**
   - `len(Neigh_x) < depth - 1` should be `< depth`, but the combinations are empty anyway.
   - In stable mode, pairs forbidden in both directions are still CI-tested before removal. That costs CPU only, and those pairs (A-L, A-A) are never candidates.

4. **LOW: KCI variant.** Infeasible at n >= 4000 by projection only. If it enters the protocol, measure it as R-13 requires (the xm-classic hand-back says so).

5. **OK:**
   - F2: the adapter pipeline is array-equal to `causallearn ... PC.pc` with the same background knowledge.
   - Defaults are causal-learn's: alpha .05, stable, uc_rule 0, uc_priority 2, fisherz (R-13).
   - Background knowledge (tiers t -> t+1, nothing but context into an action, no action -> lag/context) is consistent with notears.
   - Sign is `pcorr_given_Z` with Z = all other actions incl. placebos, lags and context (R-3/R-4).
   - No Truth access. F3 textbook CPDAG passes, but it is weak evidence beyond F2 since it is a library call.

## Recommended fix

- Doc only:
  - cite pcalg `pMax` as the source of the score;
  - add the stable-mode semantics (finding 1);
  - report the per-cell pinv-fallback fraction for E1/E3 (finding 2).
- No code change.
