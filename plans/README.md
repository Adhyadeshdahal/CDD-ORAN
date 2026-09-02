# Implementation Plans

Generated on 2026-09-02 against commit `57bcd46`. These plans are the recommended
overnight queue. Every executor must use its own git worktree and branch, read its plan
fully, run every verification gate, and stop rather than silently changing a preregistered
scientific contract.

## Execution Order And Status

| Plan | Title | Priority | Effort | Depends on | Status |
|------|-------|----------|--------|------------|--------|
| 001 | Make E1 artifacts fail closed | P1 | M | - | DONE (merged c5d3663) |
| 002 | Remove target leakage from discovery calibration | P1 | M | - | DONE (merged c5d3663) |
| 003 | Add the learned E1 discovery arm | P1 | L | 001, 002 | DONE (merged afc88de) |
| 004 | Measure the E1 multi-seed envelope | P2 | M | 003 | IN PROGRESS |
| 005 | Build the v2 episodic rollout kernel | P2 | M | - | DONE (merged c5d3663) |

Status values: `TODO`, `IN PROGRESS`, `DONE`, `BLOCKED: <reason>`, `REJECTED: <reason>`.

## Dependency Notes

- Plans 001 and 002 are independent and should start in parallel.
- Plan 003 must not start from an unreviewed branch. It consumes the validated artifact
  contract from 001 and the no-ground-truth discovery rule from 002.
- Plan 004 is the overnight compute job. Start it only after 003 passes all tests; preserve
  every failed replicate rather than rerunning only failures with changed settings.
- Plan 005 is code-parallel with 001 and 002. It must not be presented as an E2/E3 result;
  it only builds and tests the common execution semantics.
- Merge order: 001, 002, 005, 003, 004. Rebase each branch on the reviewed predecessor and
  rerun the full gate before merge.

## Canonical Verification

The current verified baseline is:

```text
uv run ruff check .    -> pass
uv run ty check        -> pass
uv run pytest          -> 190 passed, 1 xfailed
```

`uv run ruff format --check .` is not currently a valid repository gate: 53 existing files
would be reformatted. Executors must format only files they modify and must not create a
repository-wide formatting diff tonight.

## Findings Considered And Rejected

- Repository-wide formatting: real baseline debt, but a 53-file mechanical change has poor
  overnight leverage and high merge-conflict cost. Defer it.
- Generalize `cdd_oran/e1slice` to all v2 environments now: rejected as premature. E4 needs
  observation/intervention indicators and E2/E3 need different dimensions and decision
  semantics. Prove the learned E1 slice before extracting a shared abstraction.
- Run E2/E3/E4 trained-arm experiments now: rejected. E1 discovery is not yet green, and the
  locked benchmark says downstream scientific results are uninterpretable until it is.
- Add periodic checkpoints to the legacy training stack tonight: useful, but outside the
  current v2 critical path. Revisit before long CDL sweeps.
