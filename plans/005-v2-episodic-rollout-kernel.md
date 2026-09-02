# Plan 005: Build the v2 episodic rollout kernel

> **Executor instructions**: Build only the common execution kernel and characterization tests.
> Do not claim an E2/E3 decision result and do not modify frozen gate thresholds or environments.
>
> **Drift check (run first)**: `git diff --stat 57bcd46..HEAD -- cdd_oran/envs/v2 cdd_oran/benchmark tests/test_v2_crn.py docs/benchmark/SEMANTICS.md`

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: MED
- **Depends on**: none
- **Category**: direction / architecture
- **Planned at**: commit `57bcd46`, 2026-09-02

## Why This Matters

The legacy evaluator never executes planner actions. `SEMANTICS.md` already freezes the exact
v2 action/advance timeline, scored window, latent objective, replication unit, and CRN pairing.
A small tested rollout kernel is the next independent substrate for E2/E3 decision evaluation,
and can be built without waiting for E1 learned discovery.

## Current State

- `docs/benchmark/SEMANTICS.md:49-100` requires `H` action+advance pairs, one terminal advance,
  exclusion of the first warm-up KPI, and scoring exactly `k2..k(H+1)`.
- `SEMANTICS.md:243-258` defines cumulative regret against a cloned-simulator open-loop oracle.
- `SEMANTICS.md:262-297` requires independent clones and coordinate-keyed exogenous pairing.
- `cdd_oran/envs/v2/base.py:105-125` supplies `apply_action`, `advance`, and `neutral_step`.
- `V2Env.snapshot/restore` at lines 154-179 preserves state and coordinates.
- Existing E2/E3 gate scripts implement local rollout variants; this plan must not refactor them
  before the shared kernel is independently characterized.

## Target API

Create a small `cdd_oran/benchmark/` package with typed immutable records:

```python
Action = tuple[int, float]

rollout_open_loop(env_factory, start_snapshot, actions, score_fn) -> RolloutResult
enumerate_open_loop(env_factory, start_snapshot, action_options, horizon, score_fn) -> OracleResult
paired_regret(env_factory, start_snapshot, planner_actions, action_options, score_fn) -> RegretResult
```

`RolloutResult` records actions, warm-up latent KPI, scored latent KPI sequence, per-step returns,
and cumulative return. Tie-breaking in oracle enumeration is lexicographically smallest action
sequence. `score_fn` receives raw latent KPI and returns finite float. Do not bake one environment's
xApp panel into the kernel.

## Commands

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Focused tests | `uv run pytest tests/test_v2_rollout.py tests/test_v2_crn.py` | all pass |
| Gate regressions | `uv run pytest tests/test_e2_decision_gate.py tests/test_e3_decision_gate.py` | all pass unchanged |
| Lint/types | `uv run ruff check cdd_oran/benchmark tests/test_v2_rollout.py && uv run ty check` | exit 0 |
| Full regression | `uv run pytest` | all pass; existing xfail remains |

## Scope

**In scope**:
- New `cdd_oran/benchmark/__init__.py`
- New `cdd_oran/benchmark/rollout.py`
- New `tests/test_v2_rollout.py`
- `docs/benchmark/SEMANTICS.md` only for a non-normative implementation pointer

**Out of scope**:
- Modifying E1-E4 equations, gate contracts, or gate thresholds.
- Refactoring existing decision gates onto the kernel in the same change.
- Integrating legacy planners or `BaseORANEnv`.
- Closed-loop planning, pruning, CEM, GPU optimization, or full 101^H production search.
- Writing experiment results.

## Git Workflow

- Branch: `agent/005-v2-episodic-rollout-kernel`
- Suggested commit: `benchmark: add open-loop episodic rollout kernel`

## Steps

### Step 1: Implement the exact timeline

For a horizon-H action sequence:

```text
repeat H times: apply_action(action[h]); advance()
advance() once with no action
discard first produced KPI; score the remaining H latent KPI vectors
```

Create a fresh env from `env_factory`, restore the supplied snapshot, and never mutate an env
owned by another arm. Validate nonempty actions, finite values/scores, valid parameter IDs, and
consistent KPI shape. Return copies, not mutable env arrays.

**Verify**: a minimal test SCM reproduces the numeric H=1 and H=2 worked examples in
`SEMANTICS.md:106-194` exactly.

### Step 2: Add exhaustive open-loop enumeration

Enumerate the Cartesian product of supplied per-step action options. This first implementation
is a correctness oracle, not an optimized planner. Validate a finite, nonempty option set and an
explicit `max_sequences` guard before allocating/iterating. Choose maximum cumulative return;
break exact ties lexicographically.

**Verify**: tiny grids find the known optimum and tie rule; exceeding the guard fails before any
rollout.

### Step 3: Add paired regret

Run planner and oracle from independent clones restored to the same snapshot. Compute
`oracle_return - planner_return` without clamping. Record both trajectories and actions. Assert
the same start coordinate and scored length.

**Verify**: tests reproduce regret 1 for the H=1 example and regret 2 for the H=2 illustrative
planner in the semantics contract.

### Step 4: Characterize CRN under divergent actions

Use a v2 test environment exposing process/observation tape values to prove different planner and
oracle actions still consume identical exogenous values at matching coordinates. Also prove the
caller env and its snapshot are unchanged after paired evaluation.

**Verify**: focused CRN tests pass byte equality checks.

### Step 5: Cross-check, but do not migrate, E2/E3 gates

In tests only, compare the shared kernel's return on a small subset of fixed E2/E3 states and
short action grids with the existing gate-local rollout arithmetic. Do not alter gate output or
replace its implementation tonight.

**Verify**: cross-check values agree to `1e-12`; existing full gate tests pass unchanged.

## Test Plan

- Exact H=1/H=2 timing, terminal advance, scored-window length, and no extra action.
- Independent clones, caller non-mutation, CRN equality under divergent actions.
- Oracle optimum/tie rule, max-sequence guard, non-clamped regret.
- Invalid empty/NaN/out-of-range inputs fail clearly.
- E2/E3 subset cross-checks with no production migration.

## Done Criteria

- [ ] Timeline exactly matches `SEMANTICS.md` worked examples.
- [ ] Oracle and planner run on independent restored envs.
- [ ] Regret is cumulative and unclamped.
- [ ] CRN remains paired when actions differ.
- [ ] E2/E3 existing gate tests remain unchanged and green.
- [ ] Focused tests, Ruff, ty, and full pytest pass.

## STOP Conditions

- Implementing the contract requires changing a frozen environment or semantics document.
- E2/E3 local rollout behavior disagrees with the normative semantics. Preserve evidence and
  report the disagreement; do not choose one silently.
- Full 101^H performance work becomes necessary to pass unit tests.
- The implementation shares one mutable env instance across arms.

## Maintenance Notes

After this kernel is reviewed, migrate one gate at a time with characterization tests. Planner
integration and scalable E3 oracle search are separate plans.
