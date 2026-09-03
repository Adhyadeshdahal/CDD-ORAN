# Frozen E1 discovery protocol (contract)

This contract is **frozen before it is executed**. It is committed on branch
`agent/003-e1-learned-discovery` in the commit `docs: freeze E1 discovery protocol`, and that
commit SHA is recorded inside every `discovery.json` this method produces. The ordering is the
anti-p-hacking guarantee: the discovery *method* is fixed on disk before it is run against E1
truth — that is, before it produces any recovery number. (E1's ground-truth graph was defined and
tested earlier, when the SCM was built; the guarantee is the narrower, defensible one — that this
method was not revised after its own recovery result was seen.)

Nothing in this protocol may be revised to improve a recovery number after this method's recovery
result has been inspected. If the first frozen execution recovers the graph imperfectly (or `largest_gap` finds
no valid split), that outcome is **recorded and reported as-is** — a null or partial recovery is
a valid scientific result. Revising the method after seeing truth is forbidden.

## 1. Inputs and layout

- Discovery consumes the persisted `E1Rows` and the **training episodes only** (the
  `train_episodes` IDs from `split.json`). It reads **no test rows** and **no environment
  truth**. The discovery module must not import `E1V2Env` or any true-adjacency symbol.
- Per row, the input vector is `x = [x_params | x_kpis]` (the committed state `s_t`) and the
  target is `y = y_kpis` (`k_{t+1}`). With `num_params = P` and `num_kpis = K`, the input
  dimension is `P + K` and there are `K` outputs.
- The candidate directed graph has shape `(K, P + K)`: every earlier-state input (param or KPI)
  may be a parent, and every next-state KPI is a child. There are **no output-to-output
  same-time edges** (the graph is over `s_t -> k_{t+1}`).

## 2. Fit (training rows only)

- Gather the training rows via the split's `train_episodes`. Require a non-empty row set and
  finite values throughout.
- Compute standardization moments (mean and population standard deviation, `ddof = 0`) for every
  input column and every output column **from the training rows only**. Reject any column whose
  standard deviation is zero (a zero-variance feature or degenerate output is non-identifiable).
- Standardize inputs and outputs with those training moments. Fit **one ordinary least-squares
  linear transition model per output KPI, with an intercept**, using `numpy.linalg.lstsq`
  (`rcond=None`). Do not add a dependency. Append a constant column for the intercept and require
  the augmented design matrix to be full column rank; a rank-deficient design is non-identifiable
  and is a STOP condition (recorded, not worked around).
- The **edge score** for `(child KPI j, input i)` is the **absolute value of the standardized
  coefficient** on input `i` in output `j`'s fit (the intercept term is not an edge and is
  dropped from the score matrix).

## 3. Threshold and mask (label-free)

- The primary, and only persisted, threshold is `largest_gap(scores.ravel(), floor=1e-3)` from
  `cdd_oran.analysis.auto_threshold`. It is selected **without labels**, from the score
  distribution alone.
- **Tie rule**: an edge is kept iff `score >= threshold`. The binary mask is
  `(scores >= threshold)` as `(K, P + K)` of `{0, 1}`.
- If all scores are equal or non-finite, `largest_gap` has no meaningful split: persist the
  diagnostic and STOP. Do not substitute a truth-informed threshold.
- No threshold sweep may replace this primary result. An oracle/labelled sweep may be reported
  only as a clearly labelled diagnostic and can never alter the persisted mask.

## 4. Persistence (before any ground truth is read)

`discovery.json` is written **atomically** and is the point of no return.
**Ground truth only after persistence**: E1 ground truth is read only after `discovery.json` is
persisted and hashed. `discovery.json` records:

- `schema_version`;
- parent hashes `dataset_hash` and `split_hash` (fail-closed binding, per Plan 001 patterns);
- `train_episodes` and `n_train_rows`;
- standardization moments (`mean_x`, `std_x`, `mean_y`, `std_y`);
- signed `coefficients` `(K, P+K)` and absolute `scores` `(K, P+K)`;
- `method` (`"largest_gap"`), `floor`, and the selected `threshold`;
- `tie_rule` (`"score >= threshold"`);
- `binary_mask` `(K, P+K)` of `{0, 1}`;
- `protocol_commit` — the git SHA of this frozen-protocol commit;
- `content_hash` — a canonical SHA-256 over all of the above (every field except
  `content_hash`).

Loading `discovery.json` recomputes `content_hash` from the bytes and refuses to load if any
parent hash, shape, numeric field, or the content hash differs.

## 5. Discovered prediction arm

The discovered arm uses the **frozen binary mask** and the **exact same per-output MLP, rows,
train IDs, model config, and weight seed** as the oracle and dense arms — the identical
644-parameter architecture; only the fixed mask differs. This is a **registered-parameter** match:
architecture and parameter count are identical, but a masked-out input contributes zero signal and
its weights are never trained, so the oracle and discovered arms have strictly fewer *effective*
free parameters than the dense arm. The comparison controls architecture and parameter budget, not
effective capacity. Its `arm_meta.json` records the
`discovery.json` content hash, and evaluation/verification bind the discovered arm to the frozen
graph (a changed discovery artifact invalidates eval/verify).

## 6. Recovery scoring (post-freeze only)

After `discovery.json` exists, and only then, map the `(K, P+K)` discovered mask into the full
`(P+K, P+K)` graph expected by `recovery_by_edge_type`: only the child KPI rows `P..P+K-1` carry
predicted edges; param-child rows are empty. Ground truth is `E1V2Env().true_adj_matrix()`.
Report precision/recall/F1 split **overall / NCP->KPI / KPI->KPI** and the list of missed edges.
These scores live in a separate `recovery.json` record so they can never alter discovery.

## 7. Failure interpretation

- Imperfect recovery is an empirical result about this temporally ordered, randomized, linear
  noiseless E1 control — not a defect to be tuned away. In particular, a KPI->KPI recall floor is
  a real property (KPIs are never intervened), distinct from an NCP->KPI floor.
- If a reviewer rejects this protocol **before** its first execution, revise the contract and
  record why. **After** first execution, do not revise it to improve recovery.
