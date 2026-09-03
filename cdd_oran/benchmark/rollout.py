"""Open-loop episodic rollout kernel (SEMANTICS.md §1/§3/§4).

A small, env-agnostic execution substrate for v2 decision evaluation. It executes the exact
``SEMANTICS.md`` §1.1 call list, scores the exact §1.2 window, enumerates an open-loop
correctness oracle (§3), and computes unclamped cumulative regret (§3) — all on **independent
env clones**, one per arm, restored to the same committed start snapshot (§4). The
coordinate-keyed exogenous tape of the env base (``cdd_oran/envs/v2/base.py``) keeps arms
CRN-paired *by construction* even when their applied actions differ: the kernel only issues the
same number/order of ``apply_action``/``advance`` calls, so matching coordinates are consumed by
every arm regardless of action values.

Scope discipline (Plan 005): this kernel bakes in **no** environment's xApp panel. The scalar
objective ``R`` is supplied by the caller as ``score_fn`` and evaluated on the **raw latent KPI
vector** ``k_t`` (SEMANTICS §2 — the noiseless latent, standardization lives inside ``score_fn``).
This is a correctness reference, not an optimized planner: the oracle is exhaustive Cartesian
enumeration guarded by an explicit ``max_sequences`` limit.

Timeline executed for a horizon-``H`` action sequence ``a_1 … a_H`` from committed start
``(p_0, k_0)`` (``SEMANTICS.md`` §1.1)::

    for h = 1 … H:  apply_action(a_h); advance()   # produces k_h
    advance()                                      # TERMINAL, no action → k_{H+1}

``H`` ``apply_action`` calls and ``H+1`` ``advance`` calls. The first advance yields the warm-up
``k_1`` (reflects only the pre-decision ``p_0``) and is **discarded**; the scored window is exactly
``{k_2 … k_{H+1}}`` (one delayed response per applied action). The undiscounted (``γ = 1``) episode
return is ``G = Σ_{h=1..H} R(k_{h+1})``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

# One open-loop action: ``(param_id, value)`` — writes one param coordinate (base.py apply_action).
Action = tuple[int, float]

# A zero-arg factory returning a FRESH, independent env exposing the v2 rollout interface:
# ``restore(snapshot) / snapshot() / apply_action(param_id, value) / advance()`` plus
# ``num_params`` / ``num_kpis``. Never share one mutable env across arms — the factory MUST mint a
# new instance each call (SEMANTICS §4 replication unit).
EnvFactory = Callable[[], Any]
# Scalar objective on ONE raw latent KPI vector; higher is better; must return a finite float.
ScoreFn = Callable[[np.ndarray], float]

DEFAULT_MAX_SEQUENCES = 1_000_000


def _readonly(arr: np.ndarray) -> np.ndarray:
    """Return a defensive, read-only float copy so results never alias a mutable env array."""
    out = np.array(arr, dtype=float, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class RolloutResult:
    """Immutable record of one open-loop rollout (arrays are read-only copies)."""

    actions: tuple[Action, ...]
    warmup_kpi: np.ndarray                 # k_1 (discarded from scoring), kept for inspection
    scored_kpis: tuple[np.ndarray, ...]    # k_2 … k_{H+1}, one per applied action
    step_returns: tuple[float, ...]        # R(k_2) … R(k_{H+1})
    cumulative_return: float               # G = Σ step_returns (γ = 1)


@dataclass(frozen=True)
class OracleResult:
    """Immutable record of the exhaustive open-loop oracle (§3 correctness comparator)."""

    best_actions: tuple[Action, ...]
    best_return: float
    best_rollout: RolloutResult
    num_sequences: int                     # number of enumerated (and scored) sequences


@dataclass(frozen=True)
class RegretResult:
    """Immutable record of paired planner-vs-oracle regret (§3, unclamped)."""

    planner_actions: tuple[Action, ...]
    oracle_actions: tuple[Action, ...]
    planner_return: float
    oracle_return: float
    regret: float                          # oracle_return − planner_return (NOT clamped)
    planner_rollout: RolloutResult
    oracle_rollout: RolloutResult


def _validate_action(action: Action, num_params: int) -> Action:
    """Coerce/validate one ``(param_id, value)``: integral param id in range, finite value."""
    if not (isinstance(action, tuple) and len(action) == 2):
        raise ValueError(f"action must be a (param_id, value) pair, got {action!r}")
    param_id, value = action
    pid = int(param_id)
    if pid != param_id:
        raise ValueError(f"param_id must be an integer, got {param_id!r}")
    if not (0 <= pid < num_params):
        raise ValueError(f"param_id {pid} out of range [0, {num_params})")
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"action value must be finite, got {value!r}")
    return (pid, val)


def rollout_open_loop(
    env_factory: EnvFactory,
    start_snapshot: Any,
    actions: Sequence[Action],
    score_fn: ScoreFn,
) -> RolloutResult:
    """Execute the SEMANTICS §1.1 call list on a fresh clone and score the §1.2 window.

    Mints a new env from ``env_factory``, restores ``start_snapshot``, then for each action applies
    it and advances (producing ``k_1 … k_H``), and finally issues one action-free terminal advance
    (``k_{H+1}``). The warm-up ``k_1`` is discarded; ``k_2 … k_{H+1}`` are scored with ``score_fn``.
    The supplied env/snapshot owned by the caller are never mutated — a fresh clone is used, and all
    returned arrays are read-only copies.
    """
    action_list = list(actions)
    if not action_list:
        raise ValueError("actions must be non-empty (horizon H >= 1)")

    env = env_factory()
    num_params = int(env.num_params)
    validated: list[Action] = [_validate_action(a, num_params) for a in action_list]

    env.restore(start_snapshot)
    num_kpis = int(env.num_kpis)

    latent: list[np.ndarray] = []
    for pid, val in validated:
        env.apply_action(pid, val)
        latent.append(_readonly(env.advance()))          # k_1 … k_H
    latent.append(_readonly(env.advance()))              # terminal k_{H+1}

    for i, k in enumerate(latent):
        if k.shape != (num_kpis,):
            raise ValueError(
                f"KPI shape mismatch at index {i}: got {k.shape}, expected ({num_kpis},)"
            )
        if not np.all(np.isfinite(k)):
            raise ValueError(f"non-finite latent KPI produced at index {i}: {k}")

    warmup = latent[0]                                    # k_1, unscored warm-up
    scored = tuple(latent[1:])                            # k_2 … k_{H+1}

    step_returns: list[float] = []
    for k in scored:
        r = float(score_fn(k))
        if not math.isfinite(r):
            raise ValueError(f"score_fn returned a non-finite value {r} on KPI {k}")
        step_returns.append(r)

    cumulative = float(sum(step_returns))
    if not math.isfinite(cumulative):
        raise ValueError(
            f"cumulative_return is non-finite ({cumulative}): finite per-step returns summed to a "
            f"non-finite total (overflow)"
        )

    return RolloutResult(
        actions=tuple(validated),
        warmup_kpi=warmup,
        scored_kpis=scored,
        step_returns=tuple(step_returns),
        cumulative_return=cumulative,
    )


def enumerate_open_loop(
    env_factory: EnvFactory,
    start_snapshot: Any,
    action_options: Sequence[Action],
    horizon: int,
    score_fn: ScoreFn,
    max_sequences: int = DEFAULT_MAX_SEQUENCES,
) -> OracleResult:
    """Exhaustive open-loop oracle over ``action_options`` applied at each of ``horizon`` steps.

    Enumerates the Cartesian product ``action_options ** horizon`` (the same per-step option set at
    every step), rolls each candidate on its OWN fresh clone, and returns the maximum-cumulative-
    return sequence. Options are first sorted ascending by ``(param_id, value)``, so exact ties
    break to the **lexicographically smallest** action sequence regardless of caller ordering: the
    sorted ``itertools.product`` yields sequences in ascending order and the incumbent is replaced
    only on a strict improvement, so the first (lex-smallest) maximizer is kept. This is a
    correctness oracle, not an optimized planner. The ``max_sequences`` guard is checked BEFORE the
    option set is even materialized: an option set whose product exceeds it raises without listing
    the options, allocating the product, or executing a single rollout.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    n_options = len(action_options)
    if n_options == 0:
        raise ValueError("action_options must be a finite, non-empty set")

    # Bound the enumeration size BEFORE materializing/iterating the option set, so an oversized
    # option set is rejected without listing the options or allocating the Cartesian product.
    num_sequences = n_options**horizon
    if num_sequences > max_sequences:
        raise ValueError(
            f"enumeration of {num_sequences} sequences exceeds max_sequences={max_sequences} "
            f"({n_options} options ** horizon {horizon})"
        )

    options = list(action_options)
    for opt in options:
        # Value finiteness / arity is checked up front; param-id range is checked per-rollout since
        # it needs the env. Enumeration size is bounded before any rollout is ever executed.
        if not (isinstance(opt, tuple) and len(opt) == 2):
            raise ValueError(f"each option must be a (param_id, value) pair, got {opt!r}")
        if not math.isfinite(float(opt[1])):
            raise ValueError(f"option value must be finite, got {opt!r}")

    # Sort ascending by (param_id, value) so that keeping only STRICT improvers below yields the
    # lexicographically SMALLEST maximizing sequence intrinsically — independent of the caller's
    # option ordering (SEMANTICS §3 tie-break).
    options = sorted(options, key=lambda o: (int(o[0]), float(o[1])))

    best_return: float | None = None
    best_rollout: RolloutResult | None = None
    for combo in itertools.product(options, repeat=horizon):
        result = rollout_open_loop(env_factory, start_snapshot, list(combo), score_fn)
        if best_return is None or result.cumulative_return > best_return:  # strict → lex-smallest
            best_return = result.cumulative_return
            best_rollout = result

    assert best_rollout is not None and best_return is not None
    return OracleResult(
        best_actions=best_rollout.actions,
        best_return=best_return,
        best_rollout=best_rollout,
        num_sequences=num_sequences,
    )


def paired_regret(
    env_factory: EnvFactory,
    start_snapshot: Any,
    planner_actions: Sequence[Action],
    action_options: Sequence[Action],
    score_fn: ScoreFn,
    max_sequences: int = DEFAULT_MAX_SEQUENCES,
) -> RegretResult:
    """Cumulative unclamped regret of a planner against the open-loop oracle (§3).

    The planner and the oracle are each rolled on independent clones restored to the SAME committed
    ``start_snapshot`` (§4) — no arm ever mutates state another arm scores, and CRN pairing holds
    even though the arms take different actions. Regret is ``oracle_return − planner_return`` with
    **no clamping** (non-negativity is a property to verify, never to enforce). The oracle horizon
    is the planner's horizon, so the scored windows match in length by construction.

    Domain guard (SEMANTICS §3): the planner and the oracle must search the SAME action class /
    grid, so every ``planner_actions[h]`` must be one of ``action_options``. An out-of-domain
    planner action would let the planner escape the oracle's search and manufacture a spurious
    NEGATIVE regret, so it is rejected up front with ``ValueError``. Regret itself is NEVER
    clamped: on a matched grid a genuinely sub-optimal oracle must still be able to surface as
    ``regret < 0`` ("Non-negativity is a property to verify, not to clamp", SEMANTICS §3).
    """
    # Reject out-of-domain planner actions before any rollout. Options/actions are normalized to
    # ``(int(param_id), float(value))`` -- the same coercion the rollout kernel applies -- so the
    # comparison is against the exact grid points the oracle enumerates. Malformed entries are left
    # for rollout_open_loop / enumerate_open_loop to report with their own canonical errors.
    option_set: set[tuple[int, float]] = set()
    for opt in action_options:
        if isinstance(opt, tuple) and len(opt) == 2:
            try:
                option_set.add((int(opt[0]), float(opt[1])))
            except (TypeError, ValueError):
                continue
    for h, action in enumerate(planner_actions):
        if not (isinstance(action, tuple) and len(action) == 2):
            continue
        try:
            key = (int(action[0]), float(action[1]))
        except (TypeError, ValueError):
            continue
        if key not in option_set:
            raise ValueError(
                f"planner action {action!r} at step {h} is outside the oracle action grid "
                f"({len(option_set)} options); regret requires the planner and the oracle to share "
                f"one action class/grid (SEMANTICS §3)"
            )

    planner = rollout_open_loop(env_factory, start_snapshot, planner_actions, score_fn)
    horizon = len(planner.actions)
    oracle = enumerate_open_loop(
        env_factory, start_snapshot, action_options, horizon, score_fn, max_sequences
    )

    # Same start coordinate (identical snapshot) and matching scored length (identical horizon).
    if len(oracle.best_rollout.scored_kpis) != len(planner.scored_kpis):
        raise ValueError(
            f"scored-length mismatch: planner {len(planner.scored_kpis)} vs oracle "
            f"{len(oracle.best_rollout.scored_kpis)}"
        )

    return RegretResult(
        planner_actions=planner.actions,
        oracle_actions=oracle.best_actions,
        planner_return=planner.cumulative_return,
        oracle_return=oracle.best_return,
        regret=oracle.best_return - planner.cumulative_return,  # unclamped (§3)
        planner_rollout=planner,
        oracle_rollout=oracle.best_rollout,
    )
