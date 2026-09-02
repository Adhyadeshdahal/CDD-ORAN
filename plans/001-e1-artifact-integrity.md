# Plan 001: Make E1 artifacts fail closed

> **Executor instructions**: Use a dedicated worktree and branch. Follow this plan in order,
> run every verification command, and update `plans/README.md` when complete. Do not weaken a
> validation to make an old artifact load.
>
> **Drift check (run first)**: `git diff --stat 57bcd46..HEAD -- cdd_oran/e1slice scripts/e1_slice.py tests/test_e1slice_dataset.py tests/test_e1slice_split.py tests/test_e1slice_model.py tests/test_e1slice_evaluate.py`
> If these files changed, compare the cited current-state behavior with the live code. Stop if
> another change already altered artifact schemas or overwrite semantics.

## Status

- **Priority**: P1
- **Effort**: M (one focused day)
- **Risk**: MED
- **Depends on**: none
- **Category**: correctness / reproducibility
- **Planned at**: commit `57bcd46`, 2026-09-02

## Why This Matters

The report describes a provenance-bound held-out pipeline, but the implementation can score a
model against a different split from the one used for training. Loaders also trust hashes in
sidecars without checking the bytes they describe. A rerun, interrupted stage, or edited file
can therefore produce plausible metrics that are not attached to the claimed experiment.

## Current State

- `cdd_oran/e1slice/dataset.py:197-209`: `load_dataset` reads arrays and the manifest but does
  not validate schema, shapes, counts, or `dataset_hash(rows)`.
- `cdd_oran/e1slice/split.py:82-93`: split files overwrite in place; `load_split` accepts JSON
  without recomputing `split_hash` or validating coverage/disjointness.
- `cdd_oran/e1slice/evaluate.py:82-85`: evaluation compares arm and dataset hashes, but never
  compares `arm_meta["split_hash"]` with the current split hash.
- `cdd_oran/e1slice/evaluate.py:111-122`: verification skips all provenance checks and trusts
  an unbound reference prediction file.
- `cdd_oran/e1slice/dataset.py:179-194`, `model.py:142-163`, and `evaluate.py:68-101` write
  directly into live paths and leave descendants from older stages in place.
- `scripts/e1_slice.py:115-145`: numeric CLI inputs permit empty splits, zero epochs, invalid
  batch sizes, NaN fractions, and negative tolerances.
- `dataset.py:83-100`: `obs_noise_scale` is passed to the env, but both features and labels are
  latent arrays. Nonzero noise therefore changes `scm_hash` without changing rows.

## Commands

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Focused tests | `uv run pytest tests/test_e1slice_dataset.py tests/test_e1slice_split.py tests/test_e1slice_model.py tests/test_e1slice_evaluate.py` | all pass |
| Lint | `uv run ruff check cdd_oran/e1slice scripts/e1_slice.py tests/test_e1slice_*.py` | exit 0 |
| Types | `uv run ty check` | exit 0 |
| Full regression | `uv run pytest` | all pass; existing xfail remains |

## Scope

**In scope**:
- `cdd_oran/e1slice/dataset.py`
- `cdd_oran/e1slice/split.py`
- `cdd_oran/e1slice/model.py`
- `cdd_oran/e1slice/evaluate.py`
- `scripts/e1_slice.py`
- `tests/test_e1slice_dataset.py`
- `tests/test_e1slice_split.py`
- `tests/test_e1slice_model.py`
- `tests/test_e1slice_evaluate.py`

**Out of scope**:
- Legacy `cdd_oran/experiments/` training/evaluation code.
- Any E1 SCM equation, graph edge, reported metric, or benchmark threshold.
- Adding observation-noise experiments. This recovery control remains latent/noiseless.
- Repository-wide formatting.

## Git Workflow

- Branch: `agent/001-e1-artifact-integrity`
- Commit logical increments using the existing style, for example
  `e1slice: validate persisted artifact chain`.
- Do not push, merge, or rewrite another branch unless the operator requests it.

## Steps

### Step 1: Add strict validation helpers

Add validation at each load boundary. Use canonical JSON (`sort_keys=True` and compact
separators for hashes) consistently.

- Dataset validation must require `SCHEMA_VERSION`, exact column names, expected dtypes,
  rank/shape agreement, finite numeric values, `n_rows`, episode/config consistency, and a
  recomputed `dataset_hash(rows)` equal to the manifest.
- Split validation must recompute the hash from all fields except `split_hash`, require train
  and test to be nonempty/disjoint, and require their union to equal the dataset episode IDs.
- Arm validation must require arm name, schema, dataset hash, split hash, architecture metadata,
  and a SHA-256 digest for `model.pt` and `test_pred_ref.npz`.
- Reference predictions must have exactly `(n_test_rows, num_kpis)`, finite values, and a digest
  matching arm metadata.

Keep hash helpers small and local unless genuinely reused. Error messages must name the file
and mismatched field.

**Verify**: focused tests including corruption of one dataset byte, split membership, split
hash, model digest, and reference digest all reject with `ValueError`.

### Step 2: Bind every consumer to the complete chain

Change training, evaluation, and verification so they all validate:

```text
rows bytes -> manifest.dataset_hash
split payload -> split.split_hash -> manifest.dataset_hash
arm metadata -> manifest.dataset_hash + split.split_hash
model/reference bytes -> arm metadata digests
```

Specifically add the missing arm/current-split comparison in both `evaluate_dataset` and
`verify_dataset`. Add a regression test that prepares a run, rewrites the split with a different
seed, and proves both evaluation and verification reject it before producing results.

**Verify**: `uv run pytest tests/test_e1slice_evaluate.py -q` passes, including the changed-split
regression.

### Step 3: Make writes atomic and refuse ambiguous overwrite

Write each file to a same-directory temporary path, flush/close it, then publish with
`Path.replace`. Refuse to run `generate`, `split`, or `train` when descendant artifacts already
exist unless an explicit `--force` is supplied. With `--force`, delete only known descendants
of that stage before publishing; never recursively delete an arbitrary user directory.

Known dependency order:

```text
rows/manifest -> split -> arms/model/reference -> metrics
```

`eval` must validate references/reload fidelity before atomically writing `metrics.json`.
Metrics must include model/reference digests and hash the complete canonical record excluding
only the `metrics_hash` field.

**Verify**: tests simulate stale descendants and a failed temporary write; no mixed final state
or stale metrics remains.

### Step 4: Reject invalid configurations and inert noise

Validate at the library boundary, not only argparse:

- `n_episodes >= 2`, `steps_per_episode > 0`, `warmup >= 0`.
- For this E1 latent recovery slice, require `obs_noise_scale == 0.0`; explain in the error that
  observed-noise datasets need a separately specified contract.
- `0 < test_fraction < 1` and finite.
- Nonempty positive hidden widths, finite `lr > 0`, `epochs > 0`, `batch_size > 0`.
- Verification tolerance finite and nonnegative.

Ensure `regression_metrics` rejects shape mismatch, empty arrays, non-2D arrays, and non-finite
inputs rather than writing NaN JSON.

**Verify**: parameterized tests cover each rejected boundary; all existing valid paths pass.

### Step 5: Correct reproducibility wording

The numeric rows are byte-regenerable on the same platform; the full manifest is not because it
contains `created_utc` and live git state. Update module docstrings/tests to state the narrower
truth. Do not remove useful timestamp/git provenance solely to force whole-directory byte
identity.

**Verify**: no source or test claims the entire persisted artifact set is byte-identical.

## Test Plan

- Extend the four existing `test_e1slice_*` files; do not create a parallel test harness.
- Test each hash and binding independently so a failure identifies the broken boundary.
- Preserve happy-path generation, capacity matching, deterministic training, and reload tests.
- Add CLI/library invalid-input tests only where they prove library enforcement.

## Done Criteria

- [ ] Replacing `split.json` after training makes `eval` and `verify` fail.
- [ ] Tampering with rows, model weights, or reference predictions is detected.
- [ ] Invalid/empty configurations fail before training or metric calculation.
- [ ] Nonzero observation noise is explicitly rejected.
- [ ] Metrics are not written unless the artifact chain verifies.
- [ ] Focused tests, Ruff, ty, and full pytest pass.
- [ ] Only in-scope files and the plan status row changed.

## STOP Conditions

- Supporting old `e1slice.v1` artifacts requires bypassing validation. Stop and report; there is
  no external compatibility requirement for these new local artifacts.
- Atomic publication cannot be implemented without recursively deleting user-provided paths.
- The fix would change E1 equations, training hyperparameters, or report numbers.
- An in-scope file drifted in a way that changes the artifact schema before work begins.

## Maintenance Notes

Every new stage must add its digest and parent hashes to the chain. Reviewers should scrutinize
whether hashes are recomputed from bytes, not merely copied from another mutable sidecar.
