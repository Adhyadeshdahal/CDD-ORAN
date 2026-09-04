# Frozen E1 discovery protocol V2 — partial-correlation selection (contract)

This contract is **frozen before it is executed**. Once committed, this document's commit SHA is
recorded as `protocol_commit` inside every artifact this method produces (`discovery_v2.json` and
its descendants). The ordering is the anti-p-hacking guarantee: the discovery *method* is fixed on
disk before it is run against E1 truth — that is, before it produces any recovery number. E1 ground
truth is read **only after** the mask has been persisted and hashed.

Nothing in this protocol may be revised to improve a recovery number after this method's recovery
result has been inspected. If the first frozen execution recovers the graph imperfectly (or a guard
fires), that outcome is **recorded and reported as-is** — a null or partial recovery is a valid
scientific result. This V2 supersedes the score and threshold of
`docs/benchmark/E1_DISCOVERY_PROTOCOL.md` (v1); the freeze discipline, inputs/layout, persistence
pattern, and fail-closed loading are inherited unchanged. V2 changes **only** the edge score and the
threshold rule; everything downstream (arm, recovery scoring) is structurally identical to v1.

Every design choice below is justified in **structural** terms — the geometry of the transition
map and the algebra of standardization — never by any coefficient value already observed in a frozen
result.

---

## 1. Freeze preamble

- This contract is frozen before execution. Its git commit SHA is the freeze point and is recorded
  as `protocol_commit` in every produced artifact.
- Discovery reads **no environment truth** and imports no true-adjacency symbol. Dimensions come
  from the persisted row shapes, exactly as in v1.
- E1 ground truth is read only after `discovery_v2.json` is persisted and hashed. Recovery scoring
  lives in a separate record (§7) that can never feed back into selection.
- Imperfect recovery is a recorded result, not a defect to tune away. Revising the method after its
  recovery number is seen is forbidden.

## 2. Inputs and layout (unchanged from v1)

- Discovery consumes the persisted `E1Rows` and the **training episodes only** (the `train_episodes`
  IDs from `split.json`). It reads **no test rows** and **no environment truth**. The discovery
  module must not import `E1V2Env` or any true-adjacency symbol.
- Per row, the input vector is `x = [x_params | x_kpis]` (the committed state `s_t`) and the target
  is `y = y_kpis` (`k_{t+1}`). With `num_params = P` and `num_kpis = K`, the input dimension is
  `P + K` and there are `K` outputs.
- The candidate directed graph has shape `(K, P + K)`: every earlier-state input (param or KPI) may
  be a parent, and every next-state KPI is a child. There are **no output-to-output same-time edges**
  (the graph is over `s_t -> k_{t+1}`).

## 3. Score — the core change: per-target partial correlation

The v1 edge score was the absolute **standardized OLS coefficient** `|beta|`. Standardization scales
each edge by `sd(x_i) / sd(y_j)`. Because a downstream KPI's `sd(y_j)` is inflated by the combined
variance of **all** of its parents, every incoming edge score for a multi-parent target is deflated
together; a target with more parents (or higher-variance parents) has its whole score row pushed
down. A single pooled cut then straddles targets that live on different score scales and buries
weak-but-real edges under a scale artefact — a purely structural consequence of standardization, not
of any particular coefficient.

**V2 replaces the score with the magnitude of the partial correlation**, which normalizes each edge
by *residual* variance rather than total output variance, removing that inflation.

### 3.1 Quantity persisted

For a child KPI `j` and a candidate input `i`, let the **control set** be *all other candidate
inputs* `X_{-i}` (the remaining `P + K - 1` inputs), augmented with an intercept column. Define the
residual-maker onto the orthogonal complement of the control column space:

```
M_{-i} = I - X_{-i} (X_{-i}^T X_{-i})^{+} X_{-i}^T
```

(computed in practice by residualizing with `numpy.linalg.lstsq(X_{-i}, ., rcond=None)`, not by
forming the inverse). Then:

```
r_i     = M_{-i} X_i        # input i, residualized on all other inputs
r_{j|i} = M_{-i} y_j        # target j, residualized on all other inputs
rho_{ji} = (r_i . r_{j|i}) / (||r_i|| * ||r_{j|i}||)     # signed partial correlation in [-1, 1]
```

- **Persisted signed underlying quantity**: `partial_corr[j, i] = rho_{ji}` (the signed partial
  correlation, `(K, P+K)`).
- **Persisted edge score**: `scores[j, i] = |rho_{ji}|`, the partial-correlation **magnitude** in
  `[0, 1]`, `(K, P+K)`. The equivalent partial `R^2` (squared semi-partial for the full-input
  regression) is exactly `scores[j, i]^2` and is not stored separately — it is a monotone transform
  of the persisted score, so any threshold expressed on one is equivalent on the other.

We persist and threshold the **magnitude** `|rho|`, a bounded effect size. We deliberately do **not**
use a t-statistic: in a noiseless linear SCM the residual standard error collapses toward zero, so
`t = rho * sqrt((n - dof) / (1 - rho^2))` diverges and any t-based rule degenerates. `|rho|` stays
bounded in `[0, 1]` and is well-defined in exactly the noiseless regime E1 lives in.

### 3.2 Why partial correlation, structurally

Partial correlation measures the association between input `i` and target `j` that **survives after
projecting out every other candidate input**. In a noiseless linear SCM this is the decisive
structural signal: a **true direct parent** carries variance in `y_j` that no other input can
account for, so its residual association is complete and `|rho| -> 1`; a **non-parent** (and any
input whose apparent association is fully mediated by the true parents already in the control set)
has no residual association once the parents are held fixed, so `|rho| -> 0`. Crucially, `rho` is
invariant to the per-target output scale — it is normalized by residual variance, not by `sd(y_j)` —
so the multi-parent deflation of §3 is removed and every target's row is placed on the **same fixed
`[0, 1]` scale**.

### 3.3 Standardized or raw inputs

The partial correlation is **invariant** to any per-column affine transform (centering and scaling)
of the inputs and of the target, so raw and standardized data yield **identical** `rho`. We
standardize inputs and outputs using **training-row moments** (mean and population standard
deviation, `ddof = 0`), matching v1, purely for numerical conditioning and to persist the same
moment block; this choice does not change any persisted `rho` or score.

### 3.4 Fit and guards (label-free, training rows only)

- Gather training rows via `train_episodes`; require a non-empty row set and finite values
  throughout. Reject any input or output column with zero training variance (non-identifiable), as
  in v1.
- Require the full augmented input design `[X | 1]` to be full column rank; a rank-deficient design
  is non-identifiable and is a recorded **STOP**, not a worked-around case (v1 pattern).
- **Near-collinearity guard (recorded STOP, never a silent workaround).** For each input `i`, the
  quantity `||r_i||^2 / n` equals `1 - R^2_i`, where `R^2_i` is input `i` regressed on all other
  inputs (standardized, so `var(X_i) = 1`). This is target-independent — compute it once per input.
  If `min_i (||r_i||^2 / n) < EPS_COND`, input `i` is (near-)linearly determined by the other
  inputs, the partial-correlation denominator `||r_i||` is numerically unstable, and discovery
  records a **STOP** naming the offending input(s). Likewise, if for any `(j, i)` the target residual
  norm `||r_{j|i}||^2 / n < EPS_COND`, target `j` is (near-)deterministically explained by the other
  inputs and its partial correlations are ill-posed — a recorded **STOP**. `EPS_COND` is a fixed
  numerical near-singularity tolerance (documented constant at machine-precision scale, `1e-8`); it
  is a rank/conditioning guard that only ever *rejects*, and it never *selects* an edge.

## 4. Threshold and mask — label-free, per-target

The threshold is applied **within each target's own score row and never pooled across targets**.
Pooling is structurally wrong: even after partial correlation removes the output-scale artefact, each
target defines an independent selection problem over its own candidate parents, and mixing rows
re-imports exactly the cross-target confound V2 exists to remove.

**Primary, persisted per-target rule.** For each target row `j`, the threshold is
`largest_gap(scores[j, :], floor = FROZEN_FLOOR)` from `cdd_oran.analysis.auto_threshold` — the same
label-free gap finder used in v1, now applied **per row instead of on the pooled vector**. This is
justified a-priori by the noiseless-control structure of §3.2: within a single target's row the true
edges cluster near `1` and the non-edges near `0`, so the row's own largest gap is the separating
boundary between "residual-association complete" and "residual-association absent". No boundary
location is assumed or imported; the within-row bimodal structure chooses it. This deliberately
avoids committing to any fixed cut value (e.g. a midpoint of the `[0, 1]` range) that a skeptic could
read as tuned — the gap rule assumes only *that* two clusters separate, not *where*.

- **Tie rule**: an edge is kept iff `score >= threshold[j]` (per-target threshold). The persisted
  `binary_mask` is the row-wise stack of `(scores[j, :] >= threshold[j])`, shape `(K, P + K)` of
  `{0, 1}`.
- **Per-target threshold vector**: the `K` selected thresholds are persisted as `threshold` (length
  `K`), one per target row.
- **Degenerate row (recorded STOP).** If any row's scores are all equal or non-finite, `largest_gap`
  has no meaningful split for that target: persist the diagnostic and **STOP** for that row. Do not
  substitute a truth-informed threshold and do not pool that row into another.
- No threshold sweep may replace this primary result. An oracle/labelled sweep may appear only as a
  clearly labelled diagnostic and can never alter the persisted mask.

## 5. Persistence (before any ground truth is read)

`discovery_v2.json` is written **atomically** and is the point of no return. **Ground truth only
after persistence.** Loading recomputes `content_hash` and refuses to load on any parent-hash,
shape, numeric, or `protocol_commit` mismatch (v1 fail-closed pattern). It records:

- `schema_version`;
- parent hashes `dataset_hash` and `split_hash` (fail-closed binding);
- `train_episodes` and `n_train_rows`;
- standardization moments (`mean_x`, `std_x`, `mean_y`, `std_y`);
- `partial_corr` `(K, P+K)` — signed partial correlations (the signed underlying quantity);
- `scores` `(K, P+K)` — `|partial_corr|`, the persisted edge scores;
- `score_method` (`"partial_correlation"`) and `threshold_method`
  (`"per_target_largest_gap"`), and `floor` (`FROZEN_FLOOR`);
- `threshold` — the per-target threshold vector (length `K`);
- `tie_rule` (`"score >= threshold[target]"`);
- `binary_mask` `(K, P+K)` of `{0, 1}`;
- `candidate_shape` `[K, P+K]`;
- `protocol_commit` — the git SHA of this frozen-protocol commit;
- `git_sha` / `git_dirty` — provenance of the producing checkout;
- `content_hash` — a canonical SHA-256 over all of the above (every field except `content_hash`).

Loading `discovery_v2.json` fails closed unless: `content_hash` recomputes; `scores == |partial_corr|`
elementwise; each row satisfies `binary_mask[j] == (scores[j] >= threshold[j])`; every row's
`threshold[j] == largest_gap(scores[j], FROZEN_FLOOR)`; `protocol_commit`, `score_method`,
`threshold_method`, and `floor` equal the frozen constants; shapes equal the frozen candidate shape;
and the parent `dataset_hash` / `split_hash` match. These are re-derivations of values already in the
file, so a correctly generated artifact is byte-unaffected while a silently retuned one is rejected.

## 6. Discovered prediction arm (unchanged from v1)

The discovered arm uses the **frozen V2 binary mask** and the **exact same per-output MLP, rows,
train IDs, model config, and weight seed** as the oracle and dense arms — the identical architecture
and parameter count; only the fixed mask differs. This is a **registered-parameter** match:
architecture and parameter budget are controlled, while a masked-out input contributes zero signal
and its weights are never trained, so the discovered and oracle arms have strictly fewer *effective*
free parameters than the dense arm. The arm's `arm_meta.json` records the `discovery_v2.json`
`content_hash`; evaluation and verification bind the arm to the frozen graph, so a changed discovery
artifact invalidates eval/verify.

## 7. Recovery scoring (post-freeze only)

After `discovery_v2.json` exists, and only then, map the `(K, P+K)` discovered mask into the full
`(P+K, P+K)` graph expected by `recovery_by_edge_type`: only the child KPI rows `P..P+K-1` carry
predicted edges; param-child rows are empty. Ground truth is `E1V2Env().true_adj_matrix()`. Report
precision / recall / F1 split **overall / NCP->KPI / KPI->KPI** plus the list of missed edges. These
scores live in a **separate** `recovery.json`-style record so they can never alter discovery.

## 8. Stability-selection cross-check (reported diagnostic, NOT the selector)

As a reported heuristic only, compute each edge's **selection frequency** across the Plan 004
10-seed envelope (`env_seed = weight_seed = r`, `split_seed = 0`): the fraction of the `B = 10` seeds
in which the per-target rule of §4 selects that edge (Meinshausen–Bühlmann 2010; CPSS
Shah–Samworth 2013). With `B = 10` the frequency CI is coarse, so this is a **reported** confirmation
that the per-target selection is stable across seeds — it is **never** the selector and **never**
alters any persisted mask. Each seed's mask is produced and persisted by the identical frozen §3–§5
pipeline; no seed is dropped or reweighted, and per-seed recovery is reported alongside.

## 9. Failure interpretation and the noiseless caveat

- Imperfect recovery is an empirical result about this temporally ordered, randomized, linear,
  noiseless E1 control — not a defect to be tuned away. A KPI->KPI recall floor is a real structural
  property (KPIs are never intervened), distinct from an NCP->KPI floor.
- **Stated plainly, not dressed up:** in a noiseless linear SCM, partial correlation separates true
  from false edges almost perfectly — true direct parents reach `|rho| ~ 1` and non-parents `|rho| ~ 0`
  — *because* controlling for the other inputs removes all mediated association and leaves only the
  direct structural signal. This near-perfect separation is a **property of the control** (a
  noiseless linear system inspected with the right conditioning), not a hard-won empirical victory.
  Accepting it is E1's intended role: green E1 means the label-free pipeline recovers structure when
  it is recoverable, which unblocks interpreting E2+. **Do not add noise to E1** or otherwise modify
  the locked benchmark to manufacture difficulty; noise and nonlinearity belong to E2+.
- If a reviewer rejects this contract **before** its first execution, revise it and record why.
  **After** first execution, do not revise it to improve recovery.
