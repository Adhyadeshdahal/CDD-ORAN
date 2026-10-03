# F7 audit: two_tower (auditor aud1, 2026-10-02, feat/v2 b7456c2)

**Verdict: FAIL (as "the published two-tower method"). As our own adaptation, it is fixable (finding 2).**

Sources used:
- the paper, arXiv:2601.13213, `archive/reference-papers/Two-tower.md` (sec IV-A, V, VI-A);
- `cdd_oran/xmethod/methods/two_tower.py`, `scripts/e2_baseline_gnn.py`, `docs/xmethod/FIDELITY_CLASSIC.md` (F6 row, F3, F4).

## Findings (most severe first)

1. **HIGH: this is not the paper's method, and the paper's method cannot be run in this study.**
   - The paper encodes each variable's whole sample column with a 2-layer ReLU MLP per tower (eq. 6a/6b).
   - It scores pairs with a learned, scaled cosine similarity of L2-normed embeddings (eq. 10).
   - It trains with BCE against the ground-truth label matrix Y in {0,1}^{Np x Nk} (sec VI-A, eq. 14), using Adam with lr 1e-3 and H 16.
   - It binarises with row-wise sparsemax at tau 0 (eq. 12-13).

   The port (`scripts/e2_baseline_gnn.py:62-115`) differs in every one of these:
   - per-sample scalar encoder psi plus id embeddings;
   - gate a = softplus(U V^T) (`:84`);
   - per-KPI reconstruction heads trained by MSE on the data (`:108`), with no labels;
   - lr 1e-2;
   - no cosine similarity, no sparsemax.

   The faithful model is supervised by the true graph, so it is not a discovery method under CONTRACT sec 5 ("Truth is never used"). Training it on the study's truth would break the contract.

   The F6 row discloses (1) correctly, but the method sits in the "published O-RAN" family (CONTRACT sec 3). Reporting it under the paper's name would misattribute our model to the authors.

2. **HIGH: one absolute tau across KPI rows, while the gate's scale differs from row to row.**
   - The score is the raw gate a[k,p] (`two_tower.py:69-72`). Each per-KPI head can absorb a rescaling of its row of a, so absolute gate values are not comparable across KPIs. The port's own native rule was row-relative (.10 x row max) for this reason, and so is the paper's sparsemax.
   - The placebo tau (`_classic_common.py:213-220`) pools the placebo's score over all KPIs and applies one cut to every row.
   - Measured here: one fit, E2 R1, n 1000, seed 3_000_000, adapter defaults.

     | Row | Null-action gates | Placebo | True gates |
     |---|---|---|---|
     | K4 | 0.82 to 7.6 | 2.33 | 0.36 and 8.3 |
     | K0-K3, K5 | at most .69 | (not noted) | as low as .001 to .9 |

     E1 R1 rows are better behaved: nulls .09 to .53, trues 4 to 5.6.
   - So one inflated row sets tau, or contributes false positives above it, and true edges in the other rows fall below it.
   - Effect on the comparison: it is unfair to this method's recall and inflates its FDP in some cells. That works in PMRT's favour, but artificially.

3. **MEDIUM: the F3 "PASS" is a regression check, not a reproduction.**
   - It replays our own earlier E2 port (`scratchpad/gnn_baseline/gnn_results.json`): ratio .856 vs .847.
   - That shows the wrapper is unchanged. It says nothing about fidelity to the paper. No result from the paper (the CAN dataset of its ref [15]) was reproduced.

4. **LOW: settings.**
   - epochs 500 is the script's CLI default (`e2_baseline_gnn.py:143`); the function default is 800 (`:95`).
   - l1 1e-3 was chosen on synthetic data with planted truth. It is not study DEV data, and the default was kept, so it is acceptable under F5. Record this.

5. **OK:**
   - No Truth or privileged access. The fit sees X_action and Y only, and the meta lists are used only for families.
   - Sign is `pcorr_given_Z` (R-4).
   - KPI->KPI is NaN by design.
   - Seed from tag 7803.
   - Tests pass (29/29 in the classic suite).

## Recommended fix (decision for the user; do not implement the supervised model)

- Move it out of "published O-RAN". Name it e.g. `neural_gate` ("self-supervised neural relevance gate inspired by the two-tower architecture of arXiv:2601.13213; ours"). Or drop it.
- If it is kept, make the score row-relative before the placebo tau. Two options:
  - score = sparsemax(a[k,:])[p], the paper's own row normalisation, which is truth-free;
  - or a[k,p] / max_p' a[k,p'].

  Then rerun F4 and integration.
- An honest "published two-tower" entry would need labels, so it can only be described in the related work, not benchmarked.
