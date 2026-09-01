"""Isolation tests for the open-loop sequence planners (build-sequence-planner §5, ruling item 9).

Tested against a small deterministic FIXTURE model implementing the V2-native rollout interface
(restore/snapshot/apply_action/advance) — no env, no training, no true simulator. The fixture
reproduces the SEMANTICS §1.5 delayed-harm chain::

    new_k0 = prev_params[0]
    new_k1 = k1_from_p * prev_params[0] - chain_coeff * prev_kpis[0]

With ``chain_coeff = 2`` an early P0 gain in ``k0`` becomes a larger delayed loss in ``k1`` two
steps later (load-bearing chain); with ``chain_coeff = 0`` there is no cross-step coupling.
"""

from __future__ import annotations

import itertools

import numpy as np

from cdd_oran.analysis.v2_regret import PanelXApp, reward, select_action
from cdd_oran.planners.sequence import (
    OpenLoopSequencePlanner,
    _rollout_sequence_scored,
    exhaustive_fh,
    greedy_fm,
)


class FixtureModel:
    """Minimal deterministic V2-native rollout model (mirrors V2Env apply/advance latency)."""

    num_params = 1
    num_kpis = 2

    def __init__(self, chain_coeff: float = 0.0, k1_from_p: float = 1.0):
        self.chain_coeff = float(chain_coeff)
        self.k1_from_p = float(k1_from_p)
        self.params = np.array([0.0])
        self.prev_params = np.array([0.0])
        self.prev_kpis = np.array([0.0, 0.0])

    def apply_action(self, param_id: int, value: float) -> None:
        self.params[int(param_id)] = float(value)

    def advance(self) -> np.ndarray:
        new_k0 = self.prev_params[0]
        new_k1 = self.k1_from_p * self.prev_params[0] - self.chain_coeff * self.prev_kpis[0]
        new = np.array([new_k0, new_k1], dtype=float)
        self.prev_params = self.params.copy()
        self.prev_kpis = new
        return self.prev_kpis.copy()

    def snapshot(self):
        return (self.params.copy(), self.prev_params.copy(), self.prev_kpis.copy())

    def restore(self, snap) -> None:
        self.params = snap[0].copy()
        self.prev_params = snap[1].copy()
        self.prev_kpis = snap[2].copy()

    def latent_kpis(self) -> np.ndarray:
        return self.prev_kpis.copy()


def _start_snapshot() -> tuple:
    """Committed start (p_0 = 0, k_0 = (0,0))."""
    return FixtureModel().snapshot()


# R that maximizes k1: a single satisfy-above xApp on KPI 1 with an unreachable threshold, so it is
# never satisfied and reward = 10*k1 - const is strictly increasing in k1. Lets us reuse the LOCKED
# v2_regret.reward / select_action verbatim as the scorer.
_PANEL = [PanelXApp(kpi_indices=(1,), mean=0.0, std=1.0, threshold=1e6, direction=0)]


def _R(k: np.ndarray) -> float:
    return reward(np.asarray(k, dtype=float), _PANEL)


def _cumulative_over_sequence(factory, snap, param_id, actions) -> float:
    scored = _rollout_sequence_scored(factory, snap, param_id, actions)
    return float(sum(_R(k) for k in scored))


# --- (a) H=1 == the v2_regret H=1 selection oracle -------------------------------------------
def test_h1_matches_v2_regret_selection_oracle():
    grid = [-1.0, 0.0, 1.0, 2.0]
    snap = _start_snapshot()
    factory = lambda: FixtureModel(chain_coeff=2.0)  # noqa: E731

    fh1 = exhaustive_fh(factory, snap, 0, grid, 1, _R)
    act1 = OpenLoopSequencePlanner().act(factory, snap, 0, grid, _R)

    oracle_action, _ = select_action(FixtureModel(chain_coeff=2.0), snap, 0, _PANEL, np.array(grid))

    assert fh1 == [oracle_action]
    assert act1 == oracle_action


# --- (b) FH == independent brute-force optimum; emits H distinct-capable actions --------------
def test_fh_equals_independent_bruteforce_and_emits_distinct_actions():
    grid = [0.0, 1.0]
    snap = _start_snapshot()
    factory = lambda: FixtureModel(chain_coeff=2.0)  # noqa: E731

    fh = exhaustive_fh(factory, snap, 0, grid, 2, _R)

    # Independent brute force over all 2^2 sequences (lexicographically smallest max).
    best_seq, best_G = None, None
    for idx in itertools.product(range(len(grid)), repeat=2):
        actions = [grid[j] for j in idx]
        g = _cumulative_over_sequence(factory, snap, 0, actions)
        if best_G is None or g > best_G:
            best_G, best_seq = g, actions
    assert fh == best_seq
    # Load-bearing chain: the delayed-harm optimum trades the immediate benefit -> a1 != a2.
    assert fh == [0.0, 1.0]
    assert fh[0] != fh[1]  # the planner CAN emit distinct per-step actions


# --- (c) FM uses PREDICTED state only, never the true divergence ------------------------------
def test_fm_uses_predicted_model_only():
    grid = [0.0, 1.0]
    snap = _start_snapshot()
    # Predicted model rewards higher P0 (k1 = +p0); "true" model rewards lower P0 (k1 = -p0).
    predicted = lambda: FixtureModel(chain_coeff=0.0, k1_from_p=1.0)  # noqa: E731
    true_divergent = lambda: FixtureModel(chain_coeff=0.0, k1_from_p=-1.0)  # noqa: E731

    fm = greedy_fm(predicted, snap, 0, grid, 2, _R)

    # Independent greedy computed on the PREDICTED model only.
    greedy_pred = greedy_fm(predicted, snap, 0, grid, 2, _R)
    # Independent greedy on the TRUE (divergent) model — must NOT be what FM followed.
    greedy_true = greedy_fm(true_divergent, snap, 0, grid, 2, _R)

    assert fm == greedy_pred == [1.0, 1.0]
    assert greedy_true == [0.0, 0.0]
    assert fm != greedy_true  # FM ignored the true divergence -> used predicted state only


# --- (d) FH == FM when there is no cross-step coupling ----------------------------------------
def test_fh_equals_fm_without_coupling():
    grid = [0.0, 1.0, 2.0]
    snap = _start_snapshot()
    factory = lambda: FixtureModel(chain_coeff=0.0)  # no horizon gap  # noqa: E731

    fh = exhaustive_fh(factory, snap, 0, grid, 3, _R)
    fm = greedy_fm(factory, snap, 0, grid, 3, _R)
    assert fh == fm


# --- (e) lexicographic tie-break is stable ---------------------------------------------------
def test_lexicographic_tiebreak_stable():
    grid = [0.0, 1.0, 2.0]
    snap = _start_snapshot()
    # k1_from_p = 0 and no coupling => k1 is always 0 => every sequence ties => smallest indices.
    factory = lambda: FixtureModel(chain_coeff=0.0, k1_from_p=0.0)  # noqa: E731

    fh = exhaustive_fh(factory, snap, 0, grid, 3, _R)
    fm = greedy_fm(factory, snap, 0, grid, 3, _R)
    assert fh == [grid[0], grid[0], grid[0]]
    assert fm == [grid[0], grid[0], grid[0]]


# --- (f) exact call list: H apply-advances + 1 terminal advance, only {k2..k_{H+1}} scored ----
class _CountingModel(FixtureModel):
    def __init__(self, chain_coeff=2.0):
        super().__init__(chain_coeff=chain_coeff)
        self.n_apply = 0
        self.n_advance = 0

    def apply_action(self, param_id, value):
        self.n_apply += 1
        super().apply_action(param_id, value)

    def advance(self):
        self.n_advance += 1
        return super().advance()


def test_call_list_and_scored_window():
    snap = _start_snapshot()
    H = 3
    counter = _CountingModel()
    factory_holder = {"m": counter}

    def factory():
        # single reused counter instance so we can inspect its call counts after one rollout
        return factory_holder["m"]

    actions = [0.0, 1.0, 0.0]
    scored = _rollout_sequence_scored(factory, snap, 0, actions)

    assert counter.n_apply == H  # exactly H action-advances applied
    assert counter.n_advance == H + 1  # H + terminal
    assert len(scored) == H  # k1 warm-up dropped; window is {k2..k_{H+1}}

    # Hand-rolled expected scored window (k2, k3, k4) for the chain fixture (chain_coeff=2):
    # k1=(0,0); k2=(a1, a1); k3=(a2, a2-2a1); k4=(a3, a3-2a2)
    a1, a2, a3 = actions
    expected = [
        np.array([a1, a1]),
        np.array([a2, a2 - 2 * a1]),
        np.array([a3, a3 - 2 * a2]),
    ]
    for got, exp in zip(scored, expected, strict=True):
        assert np.allclose(got, exp), f"{got} != {exp}"
