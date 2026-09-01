"""General open-loop sequence planners (FH exhaustive / FM greedy) — V2-native, additive.

This is the GENERAL open-loop planner whose interface is FROZEN in
``.herdr/reports/rule-e3-forks.md`` item 1 and ``e3-planner-research.md`` §3. It is env-agnostic
and independent of the E3 SCM (still under review), so it is safe to build ahead of E3.

**Additive only.** No legacy planner (``horizon.py``, ``cem.py``, ``qacm.py``, ``mcts.py``,
``joint.py``, ``cost.py``, ``ensemble.py``) is imported or edited; E1/E2 gates depend on those
staying byte-identical. This module targets the minimal **V2-native rollout interface** (ruling
item 1), not the legacy ``predict_next_state`` / bins / ``action_space`` path::

    model_factory() -> a fresh rollout model exposing
        restore(snapshot) / snapshot() / apply_action(param_id, value) / advance()

and reuses the locked H=1 rollout call list from ``cdd_oran.analysis.v2_regret``
(``rollout_scored_latent``): ``apply_action; advance`` (warm-up ``k1`` UNSCORED) then ``advance``
(terminal, SCORED ``k2``).

**Call list (SEMANTICS §1.1) for a horizon-H sequence** ``a_1..a_H`` from a committed start::

    for h = 1..H:  apply_action(param_id, a_h); advance()   # produces k_h
    advance()                                               # TERMINAL, no action -> k_{H+1}

The first advance ``k_1`` is the unscored warm-up; the scored window is ``{k_2 .. k_{H+1}}`` (one
delayed response per applied action, since ``a_h`` first influences ``k_{h+1}``). Cumulative
return is ``G = Σ_{h=1..H} γ^{h-1} R(k_{h+1})`` with ``γ = 1`` (ruling item 4). Scoring is latent
noiseless — the model itself must be constructed with observation noise off.

**Correctness-first reference implementation.** ``exhaustive_fh`` enumerates ALL ``len(grid)**H``
sequences by rolling an actual model clone per candidate. This is the deterministic reference the
gate cross-checks against; the vectorized ``H = 3 × 101`` (``1,030,301``-sequence) enumeration is
the E3 gate harness's responsibility, NOT this module.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from cdd_oran.analysis.v2_regret import rollout_scored_latent

# A world-model factory returns a fresh rollout model to be restored to a start snapshot.
ModelFactory = Callable[[], Any]
# A scorer maps one raw latent KPI vector to R (higher is better). Callers pass e.g.
# ``lambda k: reward(k, panel)`` from v2_regret, so the locked hinge R is reused verbatim.
Scorer = Callable[[np.ndarray], float]


def _clone(model_factory: ModelFactory, snapshot) -> Any:
    """clone_from(snapshot): a fresh independent model restored to the committed start state."""
    model = model_factory()
    model.restore(snapshot)
    return model


def _rollout_sequence_scored(
    model_factory: ModelFactory, snapshot, param_id: int, actions: Sequence[float]
) -> list[np.ndarray]:
    """Roll one open-loop sequence on a fresh clone; return the SCORED window ``{k2..k_{H+1}}``.

    ``H`` ``apply_action; advance`` pairs then one action-free terminal ``advance``; the warm-up
    ``k1`` is dropped, so the returned list has exactly ``len(actions)`` scored KPI vectors.
    """
    model = _clone(model_factory, snapshot)
    kpis: list[np.ndarray] = []
    for a in actions:
        model.apply_action(param_id, float(a))
        kpis.append(np.asarray(model.advance(), dtype=float))  # k_1 .. k_H
    kpis.append(np.asarray(model.advance(), dtype=float))  # terminal k_{H+1}
    return kpis[1:]  # {k_2 .. k_{H+1}} — the warm-up k_1 is unscored


def _cumulative_return(scored: Sequence[np.ndarray], R: Scorer) -> float:
    """Undiscounted (γ=1) sum of R over the scored window (ruling item 4)."""
    return float(sum(R(k) for k in scored))


def exhaustive_fh(
    model_factory: ModelFactory,
    snapshot,
    param_id: int,
    grid: Sequence[float],
    H: int,
    R: Scorer,
) -> list[float]:
    """FH arm: exhaustive open-loop sequence search — ``argmax_{a_1..a_H} Σ_h R(k_{h+1})``.

    Enumerates all ``len(grid)**H`` grid-index sequences, rolls each on a fresh clone restored to
    ``snapshot``, and returns the cumulative-``R`` argmax sequence (as grid VALUES). Ties break to
    the **lexicographically smallest grid-index tuple** (ruling item 3): ``itertools.product``
    yields indices in ascending lexicographic order and the best is updated only on a STRICT
    improvement, so the first (smallest) maximizer is kept.
    """
    if H < 1:
        raise ValueError("H must be >= 1")
    grid = list(grid)
    best_indices: tuple[int, ...] | None = None
    best_G: float | None = None
    for indices in itertools.product(range(len(grid)), repeat=H):
        actions = [grid[j] for j in indices]
        scored = _rollout_sequence_scored(model_factory, snapshot, param_id, actions)
        g = _cumulative_return(scored, R)
        if best_G is None or g > best_G:  # strict > => lexicographically smallest tie-break
            best_G, best_indices = g, indices
    assert best_indices is not None
    return [float(grid[j]) for j in best_indices]


def greedy_fm(
    model_factory: ModelFactory,
    snapshot,
    param_id: int,
    grid: Sequence[float],
    H: int,
    R: Scorer,
) -> list[float]:
    """FM arm: greedy one-step-at-a-time open-loop sequence, on PREDICTED model state only.

    For ``h = 1..H``: exhaustive 1-step ``argmax_v R(k_{h+1})`` over the grid, evaluated from the
    arm's OWN current predicted-model state via the locked H=1 rollout (``rollout_scored_latent``:
    ``apply; advance`` warm-up then ``advance``). Commit the choice, roll the model exactly ONE
    real step forward (``apply; advance``) to obtain the next predicted decision state, and
    re-decide from THAT state. The TRUE env is never read between steps (ruling item 7), so this
    stays strictly open-loop. Same lexicographic (smallest grid index) tie-break.
    """
    if H < 1:
        raise ValueError("H must be >= 1")
    grid = list(grid)
    sequence: list[float] = []
    current = snapshot
    for _ in range(H):
        best_j: int | None = None
        best_score: float | None = None
        for j, v in enumerate(grid):
            model = _clone(model_factory, current)
            k = rollout_scored_latent(model, param_id, float(v))  # one-step delayed response
            score = R(np.asarray(k, dtype=float))
            if best_score is None or score > best_score:  # strict > => smallest-index tie-break
                best_score, best_j = score, j
        assert best_j is not None
        chosen = float(grid[best_j])
        sequence.append(chosen)
        # Roll the PREDICTED model one real step forward: apply the committed action, advance once.
        model = _clone(model_factory, current)
        model.apply_action(param_id, chosen)
        model.advance()
        current = model.snapshot()
    return sequence


class OpenLoopSequencePlanner:
    """Thin, duck-typed convenience wrapper over :func:`exhaustive_fh`.

    NOT a subclass of the legacy ``cdd_oran.planners.base.Planner``: that ABC's ``act`` signature
    is torch/xapps-specific (the learned-model path), incompatible with this V2-native
    ``R``-callable reference planner. ``act`` here returns the H=1 first (and only) action, per
    ``e3-planner-research.md`` §3.
    """

    name = "OpenLoopSequenceFH"

    def act_sequence(
        self,
        model_factory: ModelFactory,
        snapshot,
        param_id: int,
        grid: Sequence[float],
        H: int,
        R: Scorer,
    ) -> list[float]:
        return exhaustive_fh(model_factory, snapshot, param_id, grid, H, R)

    def act(
        self,
        model_factory: ModelFactory,
        snapshot,
        param_id: int,
        grid: Sequence[float],
        R: Scorer,
    ) -> float:
        """Return the H=1 first action (single-decision selection)."""
        return self.act_sequence(model_factory, snapshot, param_id, grid, 1, R)[0]
