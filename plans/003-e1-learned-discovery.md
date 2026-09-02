# Plan 003: Add the learned E1 discovery arm

> **Executor instructions**: Begin only after Plans 001 and 002 are reviewed and merged into
> your base. Freeze the discovery protocol in code/tests before looking at its recovery result.
> A null/failure is a valid output; do not tune the method after inspecting E1 truth.
>
> **Drift check (run first)**: `git diff --stat 57bcd46..HEAD -- cdd_oran/e1slice cdd_oran/analysis/auto_threshold.py cdd_oran/analysis/recovery_metrics.py scripts/e1_slice.py tests/test_e1slice_*.py docs/benchmark`

## Status

- **Priority**: P1
- **Effort**: L (multi-day if the first frozen method fails)
- **Risk**: HIGH
- **Depends on**: `plans/001-e1-artifact-integrity.md`, `plans/002-discovery-no-target-leakage.md`
- **Category**: direction / scientific pipeline
- **Planned at**: commit `57bcd46`, 2026-09-02

## Why This Matters

The current E1 result compares a supplied true mask with a dense mask. It validates prediction
and persistence but not the E1 hypothesis in `docs/benchmark/SPEC.md:52-143`, which requires
learned directed graph recovery and discovered-versus-oracle prediction. This plan adds that
missing experiment without using ground truth to select the learned graph.

## Current State

- `cdd_oran/e1slice/model.py:44-56`: masks are only `oracle` or `dense` and the oracle reads
  `E1V2Env.true_adj_matrix()` directly.
- `scripts/e1_slice.py:75-86`: only those two arms train.
- `cdd_oran/e1slice/dataset.py`: persisted rows contain temporally oriented candidates
  `[P_t, K_t] -> K_{t+1}`, so edge direction is fixed by time rather than inferred from a
  contemporaneous covariance graph.
- E1 is exactly linear and noiseless: `K0=P0`, `K1=P1`, `K2=P2+0.5*K0_prev`,
  `K3=P3+0.5*K1_prev` (`cdd_oran/envs/v2/e1.py:64-71`).
- `cdd_oran/analysis/auto_threshold.py:36-125` provides deterministic, label-free threshold
  selection; `largest_gap` is the primary existing method.
- `cdd_oran/analysis/recovery_metrics.py:25-64` already computes directed overall,
  NCP-to-KPI, and KPI-to-KPI PRF after a graph is frozen.

## Frozen Protocol

Write this protocol into a short committed contract under `docs/benchmark/` before executing it:

- Fit one ordinary least-squares linear transition model per output KPI using only training
  episodes and an intercept. Use `numpy.linalg.lstsq`; do not add a dependency.
- Standardize every input and output using moments computed from training rows only before fit.
  Edge score is absolute standardized coefficient. Reject zero-variance features.
- Candidate graph shape is `(num_kpis, num_params + num_kpis)`; every earlier-state input may be
  a parent and every next-state KPI is a child. There are no output-to-output same-time edges.
- Primary threshold is `largest_gap(scores.ravel(), floor=1e-3)`, selected without labels.
- Ground truth is read only after `discovery.json` is atomically persisted and hashed.
- The discovered prediction arm uses the frozen binary mask and the exact same per-output MLP,
  rows, train IDs, model config, and weight seed as oracle/dense.
- No threshold sweep may replace the primary result. An oracle sweep may be reported only as a
  labeled diagnostic and cannot alter the persisted mask.

If a reviewer rejects this protocol before first execution, revise the contract and record why.
After first execution, do not revise it to improve recovery.

## Commands

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Focused tests | `uv run pytest tests/test_e1slice_discovery.py tests/test_e1slice_model.py tests/test_e1slice_evaluate.py` | all pass |
| Lint | `uv run ruff check cdd_oran/e1slice scripts/e1_slice.py tests/test_e1slice_*.py` | exit 0 |
| Types | `uv run ty check` | exit 0 |
| Full regression | `uv run pytest` | all pass; existing xfail remains |

## Scope

**In scope**:
- New `cdd_oran/e1slice/discovery.py`
- `cdd_oran/e1slice/model.py`
- `cdd_oran/e1slice/evaluate.py`
- `scripts/e1_slice.py`
- New `tests/test_e1slice_discovery.py`
- Existing E1 slice model/evaluation tests as needed
- New frozen E1 discovery contract under `docs/benchmark/`

**Out of scope**:
- Reusing target labels for threshold/calibration.
- Modifying the E1 SCM, rows, split, oracle mask, or model capacity.
- Integrating legacy `ExperimentConfig`/`get_env`/CDL training.
- Generalizing to E2-E4.
- Claiming causal identification beyond this temporally ordered randomized E1 control.

## Git Workflow

- Branch: `agent/003-e1-learned-discovery`
- Commit the protocol first, implementation second, result artifact/report separately.
- Suggested messages: `docs: freeze E1 discovery protocol`, then
  `e1slice: add learned graph and discovered arm`.

## Steps

### Step 1: Commit the protocol before any E1 result

Create the contract described above. Include input/output layout, train-only moments, exact solver,
score, threshold, tie rule (`score >= threshold`), artifact fields, recovery metrics, and failure
interpretation. Record the commit SHA in the eventual discovery artifact.

**Verify**: grep confirms the contract contains `numpy.linalg.lstsq`, `largest_gap`,
`training episodes only`, and `ground truth only after persistence`.

### Step 2: Implement train-only linear discovery

Add a frozen config dataclass and a pure function that accepts `E1Rows` plus train episode IDs.
Return coefficients, standardized scores, selected threshold, and binary mask. Validate shapes,
finite values, rank, and nonempty rows. The implementation must not import `E1V2Env` or any true
adjacency symbol.

Unit tests use synthetic equations with known sparse coefficients, shuffled row order, and a
false feature. Test deterministic recovery and prove changing held-out rows cannot change the
mask.

**Verify**: `uv run pytest tests/test_e1slice_discovery.py -q` passes.

### Step 3: Persist and bind `discovery.json`

Persist schema version, dataset/split hashes, train episode IDs, train-row count, standardization
moments, signed coefficients, absolute scores, method/threshold, binary mask, protocol git SHA,
and a canonical content hash. Plan 001 validation patterns apply. Refuse to load if any parent
hash, shape, numeric field, or content hash differs.

The file must be written before any recovery score is computed.

**Verify**: corruption tests reject modified coefficients, threshold, mask, and parent hashes.

### Step 4: Add a discovered prediction arm

Allow `OneStepPredictor` construction from a validated explicit mask. Do not make
`arm_mask("discovered")` read environment truth. Train discovered, oracle, and dense on the same
rows/IDs/config. All three retain the identical 644-parameter architecture; only fixed masks
differ. Arm metadata for discovered includes the discovery artifact hash.

Evaluation and verification must require all three arms and bind discovered to the frozen graph.

**Verify**: tests show equal parameter counts, discovered mask equals persisted mask, and changing
the discovery artifact after training invalidates eval/verify.

### Step 5: Add post-freeze recovery scoring

After persistence, map the `(4,8)` discovered parent mask into the full `(8,8)` graph expected by
`recovery_by_edge_type`: only child rows 4..7 contain predicted edges. Score overall,
NCP-to-KPI, and KPI-to-KPI PRF and missed edges. Store scores in a separate evaluation record so
they cannot alter discovery.

Do not make “perfect recovery” a test assertion. Tests may assert arithmetic against a fixture;
the real E1 outcome is empirical.

**Verify**: a deliberately imperfect fixture reports the expected false positive and missed
KPI-to-KPI edge.

### Step 6: Run the frozen E1 slice once

Use a fresh unique run directory and the report configuration: 48 episodes, 16 recorded steps,
warmup 2, env seed 0, split fraction 0.25, split seed 0, hidden 16, learning rate 0.01, 300 epochs,
batch 64, weight seed 0. Run generate, split, discover, train, eval, verify in that order.

Archive command lines, stdout/stderr, all hashes, recovery metrics, and three-arm prediction
metrics. If recovery is imperfect, record it and stop; do not tune.

**Verify**: CLI exits 0 through verification and the result chain validates from disk.

## Test Plan

- New synthetic discovery tests cover exact sparse recovery, row-order determinism, train-only
  behavior, invalid/rank-deficient data, and artifact corruption.
- Existing model/eval tests expand from two to three arms.
- Real E1 recovery quality is a result, not a unit-test constant.

## Done Criteria

- [ ] Discovery does not import/read E1 true adjacency before artifact persistence.
- [ ] Learned mask is selected from training rows with the frozen label-free rule.
- [ ] Discovered/oracle/dense arms use identical rows, IDs, architecture, and parameter count.
- [ ] Recovery and prediction metrics are both emitted with complete hashes.
- [ ] A failed recovery result is preserved without retuning.
- [ ] Focused tests, Ruff, ty, and full pytest pass.

## STOP Conditions

- Plans 001 or 002 are not merged and green.
- The protocol was already executed before its contract commit exists.
- `largest_gap` has no valid split because all scores are equal/nonfinite. Persist the diagnostic
  and stop; do not substitute a truth-informed threshold.
- Rank deficiency makes coefficients non-identifiable on the frozen training data.
- Achieving recovery appears to require changing E1 generation after seeing truth.

## Maintenance Notes

This is intentionally E1-specific. E4's observational/interventional indicators and hidden
confounding invalidate direct generalization of this OLS discovery rule.
