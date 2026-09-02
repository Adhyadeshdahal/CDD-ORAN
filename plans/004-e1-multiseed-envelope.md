# Plan 004: Measure the E1 multi-seed envelope

> **Executor instructions**: This plan includes a potentially long unattended run. Start only
> from a clean reviewed commit containing Plan 003. Never discard failed seeds, change settings
> mid-run, or select a favorable subset.
>
> **Drift check (run first)**: `git diff --stat 57bcd46..HEAD -- cdd_oran/e1slice scripts/e1_slice.py tests/test_e1slice_*.py docs/benchmark`

## Status

- **Priority**: P2
- **Effort**: M plus compute time
- **Risk**: LOW
- **Depends on**: `plans/003-e1-learned-discovery.md`
- **Category**: tests / scientific measurement
- **Planned at**: commit `57bcd46`, 2026-09-02

## Why This Matters

The current report is one dataset seed and one weight seed. It establishes a zero point but not
the stability of graph recovery or the small oracle-dense gap. A fixed paired seed envelope is
the correct overnight workload after the three-arm pipeline is trustworthy.

## Frozen Run Matrix

- Replicates: `0..9`, all retained.
- For replicate `r`: `env_seed=r`, `weight_seed=r`, `split_seed=0`.
- Dataset: 48 episodes, 16 recorded steps, warmup 2, noise 0.
- Split: test fraction 0.25.
- Model: hidden `(16,)`, lr `1e-2`, 300 epochs, batch size 64.
- Arms: discovered, oracle, dense on identical per-replicate rows and split.
- Primary outputs: graph precision/recall/F1 overall and by edge type; per-arm test MSE/MAE;
  paired MSE differences `dense-oracle`, `discovered-oracle`, `dense-discovered`.
- Summary: all ten values, mean, median, sample standard deviation, min/max, and deterministic
  bootstrap 95% CI of the paired mean using resample seed 0 and 10,000 resamples.
- No hypothesis-significance claim from ten seeds; report effect envelopes and failures.

## Scope

**In scope**:
- New `scripts/e1_slice_sweep.py`
- New `tests/test_e1slice_sweep.py`
- Minimal reusable aggregation helper under `cdd_oran/e1slice/` if needed
- A new dated report directory only after all results exist

**Out of scope**:
- Changing any frozen run value after execution starts.
- Parallel writes into one run directory.
- E2-E4 experiments or legacy model training.
- Deleting/rerunning only unfavorable or failed replicates.

## Commands

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Dry run | `uv run python -m scripts.e1_slice_sweep --out runs/e1slice-v2-envelope --dry-run` | prints exactly 10 unique run dirs and frozen settings |
| Tests | `uv run pytest tests/test_e1slice_sweep.py` | all pass |
| Full run | `uv run python -m scripts.e1_slice_sweep --out runs/e1slice-v2-envelope` | exit 0 or explicit partial-failure exit with journal |
| Full verification | `uv run ruff check . && uv run ty check && uv run pytest` | all pass |

## Git Workflow

- Branch: `agent/004-e1-multiseed-envelope`
- Commit runner/tests before starting compute. Commit the report separately only if reports are
  intentionally force-added under the repository's existing ignored-report convention.

## Steps

### Step 1: Build a resumable, collision-free runner

Each replicate writes to `<out>/replicate-00` through `replicate-09`. Write an atomic journal
after every stage with command, start/end UTC, return code, log path, and artifact hashes. On
resume, skip only a stage whose full artifact chain validates. Never infer completion from file
existence alone.

Capture stdout/stderr in per-stage logs while streaming useful progress. A failed replicate is
marked failed and later replicates continue; process exit is nonzero if any failed.

**Verify**: tests interrupt after a stage, resume without recomputing valid work, and reject an
invalid existing artifact.

### Step 2: Implement deterministic aggregation

Read only verified completed replicate records. Require exactly the frozen seed matrix for a
successful summary. Emit `replicates.jsonl` and canonical `summary.json`; include code SHA,
dirty state, runtime versions, run settings, every child metrics hash, and failed-replicate list.

Unit-test bootstrap CI and pairing against a small fixed array. Do not add SciPy.

**Verify**: rerunning aggregation produces byte-identical numeric summary content.

### Step 3: Execute all ten replicates unattended

Before launch record `git status --short`, commit SHA, `uv.lock` SHA-256, Python, NumPy, and Torch
versions. Use CPU unless the implementation has a tested deterministic GPU mode. Do not mutate
the branch during execution.

**Verify**: all ten `verify` stages pass and `summary.json` lists ten unique replicates.

### Step 4: Write the delta report

Report graph recovery first, then prediction errors and paired differences. State any failed
seed prominently. Compare with the single-seed 2026-09-02 report without replacing it. Avoid
claims of significance; discuss whether the zero point is stable and whether learned recovery
is consistently green.

**Verify**: every number in the report is generated from `summary.json`; no manually transcribed
untracked intermediate is the sole source.

## Done Criteria

- [ ] Dry run names exactly ten collision-free jobs.
- [ ] Runner resumes only validated stages and preserves complete logs.
- [ ] All ten planned replicates are represented, including failures.
- [ ] Summary includes full child hashes and paired effect envelopes.
- [ ] Full verification passes after runner code changes.

## STOP Conditions

- Plan 003 result is not frozen, reviewed, and green enough to run unchanged.
- Any replicate requires different hyperparameters or a different threshold.
- Two processes would write the same output directory.
- Determinism checks fail for a resumed stage.

## Maintenance Notes

Keep this runner E1-specific until E2-E4 data contracts exist. A future general sweep tool must
model observational/interventional mode and decision trajectories explicitly.
