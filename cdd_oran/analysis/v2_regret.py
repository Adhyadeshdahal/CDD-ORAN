"""Locked v2 regret machinery (SEMANTICS §2/§3) — shared by the v2 decision-gap gates.

ONE implementation of the locked objective ``R`` and the H=1 single-control finite-horizon
selection oracle, so E2/E3/E4 reuse the same code (orchestrator disposition Q7).

**Objective ``R`` (SEMANTICS §2, locked).** On the RAW LATENT KPI vector ``k`` (noise excluded),
for each xApp ``i`` in the panel::

    u_i     = (mean_of_xapp_raw_kpis(k) - xapp.mean_i) / xapp.std_i
    theta_i = (threshold_i - xapp.mean_i) / xapp.std_i
    dir 0 (satisfy-above): distance_i = max(theta_i - u_i, 0) ; ok_i = u_i >= theta_i
    dir 1 (satisfy-below): distance_i = max(u_i - theta_i, 0) ; ok_i = u_i <= theta_i
    cost(k) = Σ_i w_i·distance_i·s − (Σ_i ok_i)^2      (w_i = 1, s = 10.0)
    R(k)    = −cost(k)

Higher ``R`` is better; the oracle/planner selects ``argmax_v R`` (equivalently ``argmin_v
cost`` — the same direction QACM minimizes, ``qacm.py:57-63,109-112``). This is BLOCKER 1 of the
reviewer ruling: the selector maximizes ``R``.

**H=1 selection oracle (SEMANTICS §1.1/§1.4).** For a committed pre-decision state ``s`` and a
candidate ``do(param_id = v)``, the scored latent ``k2`` is produced by the exact call list::

    apply_action(param_id, v); advance()   -> k1  (warm-up, reflects pre-decision params) UNSCORED
    advance()                  (terminal)  -> k2                                            SCORED

One ``apply_action`` + two ``advance`` calls; the terminal advance carries no action. Scoring is
latent noiseless (``obs_noise_scale = 0``). The world model that is rolled (true vs decoy) is the
caller's choice; only the selection differs, never the call list.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Locked scaling and action grid (SEMANTICS §2; GATE_CONTRACT_E2 §5).
SCALING_TERM = 10.0
# 101-point inclusive P0 grid V = {-100, -98, ..., 98, 100}.
P0_GRID = np.linspace(-100.0, 100.0, 101)


@dataclass(frozen=True)
class PanelXApp:
    """One xApp in the scored conflict panel."""

    kpi_indices: tuple[int, ...]
    mean: float
    std: float
    threshold: float
    direction: int  # 0 = satisfy-above, 1 = satisfy-below


def latent_utility(kpis: np.ndarray, xapp: PanelXApp) -> float:
    """``u_i(k) = (mean_of_the_xapp's_raw_kpis − xapp.mean)/xapp.std`` (SEMANTICS §2)."""
    m = float(np.mean([kpis[i] for i in xapp.kpi_indices]))
    return (m - xapp.mean) / xapp.std


def hinge_cost(kpis: np.ndarray, panel: list[PanelXApp], scaling: float = SCALING_TERM) -> float:
    """Locked hinge cost ``Σ_i distance_i·s − (Σ_i ok_i)^2`` (w_i = 1)."""
    total_distance = 0.0
    satisfied = 0
    for xapp in panel:
        u = latent_utility(kpis, xapp)
        theta = (xapp.threshold - xapp.mean) / xapp.std
        if int(xapp.direction) == 0:
            distance = max(theta - u, 0.0)
            ok = u >= theta
        else:
            distance = max(u - theta, 0.0)
            ok = u <= theta
        total_distance += distance
        satisfied += int(ok)
    return float(total_distance * scaling - satisfied**2)


def reward(kpis: np.ndarray, panel: list[PanelXApp], scaling: float = SCALING_TERM) -> float:
    """``R(k) = −cost(k)`` (higher is better)."""
    return -hinge_cost(kpis, panel, scaling)


def rollout_scored_latent(env, param_id: int, value: float) -> np.ndarray:
    """H=1 call list from the env's CURRENT committed state; returns the SCORED latent ``k2``.

    ``apply_action(param_id, value); advance()`` (warm-up, unscored) then ``advance()``
    (terminal, scored). The caller is responsible for restoring the committed state first.
    """
    env.apply_action(param_id, value)
    env.advance()  # warm-up k1 — reflects pre-decision params, UNSCORED
    return env.advance()  # terminal k2 — SCORED


def score_grid(
    env,
    snap,
    param_id: int,
    panel: list[PanelXApp],
    grid: np.ndarray = P0_GRID,
) -> np.ndarray:
    """``R`` for every grid value from committed state ``snap``, rolling ``env`` (the world model).

    ``env`` is restored to ``snap`` before each candidate so the rollouts are independent and
    CRN-paired by coordinate (SEMANTICS §4).
    """
    scores = np.empty(len(grid), dtype=float)
    for i, v in enumerate(grid):
        env.restore(snap)
        k2 = rollout_scored_latent(env, param_id, float(v))
        scores[i] = reward(k2, panel)
    return scores


def select_action(
    env,
    snap,
    param_id: int,
    panel: list[PanelXApp],
    grid: np.ndarray = P0_GRID,
) -> tuple[float, np.ndarray]:
    """Exhaustive H=1 single-control selection: ``argmax_v R`` (BLOCKER 1). Returns (action, scores)."""
    scores = score_grid(env, snap, param_id, panel, grid)
    idx = int(np.argmax(scores))
    return float(grid[idx]), scores
