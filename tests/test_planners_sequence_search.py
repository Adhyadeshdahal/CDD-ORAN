"""Behavioral tests for the V2 sequence search planners (CEM, MPPI) and the QACM wrapper."""
from __future__ import annotations

import numpy as np

from cdd_oran.planners.sequence import (
    _sequence_return,
    cem_sequence,
    exhaustive_fh,
    greedy_fm,
    mppi_sequence,
    qacm_v2,
)

GRID = np.linspace(0.0, 1.0, 11).tolist()


class DelayedModel:
    """k_{t+1}[0] = action applied at t; k_{t+1}[1] = action applied at t-1 (a delayed consequence)."""

    def __init__(self):
        self.a, self.prev = 0.0, 0.0

    def snapshot(self):
        return (self.a, self.prev)

    def restore(self, snap):
        self.a, self.prev = snap

    def apply_action(self, param_id, value):
        self.a = float(value)

    def advance(self):
        k = np.array([self.a, self.prev])
        self.prev = self.a
        return k


def R(k):
    # reward the immediate KPI, but penalize the delayed consequence of a large earlier action
    return float(k[0] - 2.5 * k[1] ** 2)


def test_search_planners_match_exhaustive():
    best = exhaustive_fh(DelayedModel, (0.0, 0.0), 0, GRID, 3, R)
    g_best = _sequence_return(DelayedModel, (0.0, 0.0), 0, best, R)
    for planner in (cem_sequence, mppi_sequence):
        seq = planner(DelayedModel, (0.0, 0.0), 0, GRID, 3, R)
        assert all(v in GRID for v in seq)
        assert np.isclose(_sequence_return(DelayedModel, (0.0, 0.0), 0, seq, R), g_best)


def test_qacm_is_myopic():
    # greedy maximizes the immediate term (always the largest action), which the delayed penalty punishes
    assert qacm_v2(DelayedModel, (0.0, 0.0), 0, GRID, 3, R) == greedy_fm(DelayedModel, (0.0, 0.0), 0, GRID, 3, R)
    seq = qacm_v2(DelayedModel, (0.0, 0.0), 0, GRID, 3, R)
    best = exhaustive_fh(DelayedModel, (0.0, 0.0), 0, GRID, 3, R)
    assert _sequence_return(DelayedModel, (0.0, 0.0), 0, seq, R) < _sequence_return(
        DelayedModel, (0.0, 0.0), 0, best, R
    )
