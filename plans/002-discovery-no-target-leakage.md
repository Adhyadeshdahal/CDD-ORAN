# Plan 002: Remove target leakage from discovery calibration

> **Executor instructions**: Work in an isolated branch. This is a scientific-validity fix,
> not a compatibility exercise. Never use the target environment's true adjacency to choose,
> calibrate, threshold, or persist the graph that will be called discovered.
>
> **Drift check (run first)**: `git diff --stat 57bcd46..HEAD -- cdd_oran/experiments/discover.py cdd_oran/analysis/graph_posterior.py cdd_oran/analysis/auto_threshold.py cdd_oran/cli.py tests/test_discovery.py tests/test_posterior.py`

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED
- **Depends on**: none
- **Category**: correctness / scientific validity
- **Planned at**: commit `57bcd46`, 2026-09-02

## Why This Matters

`run_discovery` currently calibrates bootstrap edge probabilities against the same environment's
true adjacency when no calibration run is provided. That is target leakage: a downstream arm
sampling that posterior has indirectly seen the answer. The next v2 milestone explicitly needs
a learned E1 graph, so this behavior must be impossible before that result is produced.

## Current State

- `cdd_oran/experiments/discover.py:123-151`: `_calibration_source` returns target-environment
  `env.true_adj_matrix` in `within_environment` mode.
- `discover.py:229-246`: those labels fit an isotonic calibrator and the output is persisted as
  a calibrated posterior suitable for downstream sampling.
- `tests/test_discovery.py:73-81`: the test requires that leaked posterior to be calibrated but
  does not assert independence from target ground truth.
- `cdd_oran/analysis/auto_threshold.py:1-18`: label-free threshold methods already exist and
  correctly reserve ground truth for validation only.
- `cdd_oran/models/__init__.py:46-58`: downstream samplers correctly refuse raw, uncalibrated
  posterior artifacts; retain this safety property.

## Commands

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Focused tests | `uv run pytest tests/test_discovery.py tests/test_posterior.py` | all pass |
| Lint | `uv run ruff check cdd_oran/experiments/discover.py cdd_oran/analysis cdd_oran/cli.py tests/test_discovery.py tests/test_posterior.py` | exit 0 |
| Types | `uv run ty check` | exit 0 |
| Full regression | `uv run pytest` | all pass; existing xfail remains |

## Scope

**In scope**:
- `cdd_oran/experiments/discover.py`
- `cdd_oran/analysis/graph_posterior.py` only if metadata/API support is required
- `cdd_oran/analysis/auto_threshold.py` only for reuse, not algorithm changes
- `cdd_oran/cli.py`
- `tests/test_discovery.py`
- `tests/test_posterior.py`
- README/help text directly describing discovery outputs

**Out of scope**:
- Changing the CDL/CMI training algorithm.
- Tuning thresholds against any target true graph.
- Relaxing the downstream requirement that sampled posteriors be independently calibrated.
- Modifying v2 E1 files; Plan 003 owns that integration.

## Git Workflow

- Branch: `agent/002-discovery-no-target-leakage`
- Suggested commit: `discovery: forbid target-environment calibration`
- Do not push or merge unless instructed.

## Steps

### Step 1: Define two honest output modes

Implement these semantics:

1. Without `calibration_run`, discovery may persist a raw bootstrap diagnostic clearly named
   and marked `calibrated=false`, plus the crisp enumeration graph chosen using the preconfigured
   or label-free threshold. It must not emit a file accepted as a calibrated posterior.
2. With `calibration_run`, calibration labels must come only from that distinct run/environment.
   Reject a calibration path resolving to the newly produced target run. Record calibration run
   identity and source hash in metadata.

Keep `PosteriorStructureSampler.from_artifact` rejecting raw posteriors. Return values and CLI
output must distinguish `raw_posterior` from `calibrated_posterior`; do not call both `posterior`.

**Verify**: focused unit tests assert both modes and metadata.

### Step 2: Prove target truth is not read on the selection path

Add a test double/environment whose `true_adj_matrix` accessor raises. In no-calibration mode,
the training, thresholding, crisp graph, and raw diagnostic must still complete. Ground truth may
be read only by a separately invoked validation/report function after the artifact is frozen.

Add a test that same-target/self calibration is rejected with a precise error.

**Verify**: `uv run pytest tests/test_discovery.py -q` passes.

### Step 3: Update CLI and documentation contracts

CLI help must state:

- `--calibration-run` is required to produce a downstream-sampleable calibrated posterior.
- Without it, the output is an uncalibrated diagnostic and cannot be used by robust structure
  sampling.
- True adjacency is allowed for post-hoc recovery scoring only.

Do not silently invent a default calibration environment.

**Verify**: `uv run python -m cdd_oran.cli discover --help` exits 0 and includes the distinction.

### Step 4: Preserve artifact validation

Confirm raw posterior loading remains possible for analysis, while staging or sampling it raises.
Confirm calibrated artifacts carry enough provenance to establish that calibration was held out.

**Verify**: `uv run pytest tests/test_posterior.py tests/test_structure_dynamics.py -q` passes.

## Test Plan

- Replace the current assertion that default discovery is calibrated.
- Add no-target-truth-read, self-calibration rejection, raw-posterior rejection by sampler, and
  held-out calibration acceptance tests.
- Keep the existing nonempty/right-shape graph assertions.

## Done Criteria

- [ ] No default code path fits calibration against `env.true_adj_matrix` of the target.
- [ ] A no-calibration output cannot pass downstream calibrated-posterior validation.
- [ ] Target truth is used only in explicitly post-hoc scoring code.
- [ ] Focused tests, Ruff, ty, and full pytest pass.
- [ ] Only in-scope files and the plan status row changed.

## STOP Conditions

- Existing production consumers are proven to require the leaked default behavior. Stop and ask
  for an explicit migration decision rather than preserving it silently.
- A proposed fix marks raw probabilities calibrated merely to satisfy current validators.
- Held-out calibration cannot be distinguished in persisted provenance.

## Maintenance Notes

Calibration answers “does this score correspond to an inclusion probability?” It is not the
same as label-free graph selection. Keep both provenance paths separate in future v2 work.
