"""E5 spine / masked-world-model regression tests (DEV env; sol review finding #i).

Protects the central equivalences of the E5 decision infrastructure:
(a) MaskedE5WorldModel(all true edges) scores identically to the true (Z-out) env;
(b) omitting the harmful edge (K_harm,P0) freezes P0 in K_harm (decoy semantics) → differs in-gate;
(c) off-gate the harmful edge is inert (masked == true regardless of the (1,0) edge);
(d) false param edges are inert; (e) binary-mask decode splits param vs chain(KPI) edges;
(f) snapshot/restore replays latent Z exactly.
"""
from __future__ import annotations

import numpy as np

from cdd_oran.analysis.v2_regret import score_grid
from cdd_oran.benchmark.masked_world_model import (
    MaskedE5WorldModel,
    edges_from_binary_mask_e5,
)
from cdd_oran.envs.v2.e5 import E5V2Env, _gate
from scripts.e5_spine import HARMFUL_EDGE, committed_state, e5_panel, is_in_gate, true_edges

SUB, CH = 0.20, 0.0  # core layer
PANEL = e5_panel()
P0 = E5V2Env.P0


def _ingate_offgate_seeds():
    ing, off = None, None
    for s in range(6000):
        snap = committed_state(s, subdom=SUB, chain_gamma=CH)
        if is_in_gate(snap) and ing is None:
            ing = (s, snap)
        if not is_in_gate(snap) and off is None:
            off = (s, snap)
        if ing and off:
            return ing, off
    raise AssertionError("could not find in-gate and off-gate seeds")


def test_all_true_edges_equals_true_env():
    """(a): the masked model with every true edge scores identically to the true (Z-out) env."""
    p_true, k_true = true_edges(E5V2Env(subdom=SUB, chain_gamma=CH))
    for s in range(30):
        snap = committed_state(s, subdom=SUB, chain_gamma=CH)
        true_env = E5V2Env(env_seed=s, theta=0.0, subdom=SUB, chain_gamma=CH)
        wm = MaskedE5WorldModel(p_true, k_true, env_seed=s, subdom=SUB, chain_gamma=CH)
        assert np.allclose(
            score_grid(true_env, snap, P0, PANEL), score_grid(wm, snap, P0, PANEL)
        ), f"seed {s}: masked all-true != true env"


def test_omit_harmful_edge_differs_in_gate_only():
    """(b)+(c): dropping (K_harm,P0) changes scores in-gate but is INERT off-gate."""
    p_true, k_true = true_edges(E5V2Env(subdom=SUB, chain_gamma=CH))
    (s_in, snap_in), (s_off, snap_off) = _ingate_offgate_seeds()
    for (s, snap, expect_diff) in [(s_in, snap_in, True), (s_off, snap_off, False)]:
        full = score_grid(
            MaskedE5WorldModel(p_true, k_true, env_seed=s, subdom=SUB, chain_gamma=CH),
            snap, P0, PANEL)
        miss = score_grid(
            MaskedE5WorldModel(p_true - {HARMFUL_EDGE}, k_true, env_seed=s, subdom=SUB, chain_gamma=CH),
            snap, P0, PANEL)
        differs = not np.allclose(full, miss)
        assert differs == expect_diff, (
            f"seed {s} in_gate={is_in_gate(snap)}: harmful-edge effect present={differs}, "
            f"expected {expect_diff}")


def test_false_edges_are_inert():
    """(d): declaring a param that is not a KPI's true parent does not change scores."""
    p_true, k_true = true_edges(E5V2Env(subdom=SUB, chain_gamma=CH))
    with_false = p_true | {(E5V2Env.K_HARM, E5V2Env.P3), (E5V2Env.K_BEN, E5V2Env.G2)}
    for s in range(20):
        snap = committed_state(s, subdom=SUB, chain_gamma=CH)
        true_env = E5V2Env(env_seed=s, theta=0.0, subdom=SUB, chain_gamma=CH)
        wm = MaskedE5WorldModel(with_false, k_true, env_seed=s, subdom=SUB, chain_gamma=CH)
        assert np.allclose(score_grid(true_env, snap, P0, PANEL), score_grid(wm, snap, P0, PANEL))


def test_binary_mask_decode_splits_param_and_chain_edges():
    """(e): candidates < num_params → param edges; >= num_params → chain (KPI-source) edges."""
    np_ = E5V2Env.num_params
    mask = np.zeros((E5V2Env.num_kpis, np_ + E5V2Env.num_kpis), dtype=int)
    mask[E5V2Env.K_HARM, E5V2Env.P0] = 1                       # param edge (1,0)
    mask[E5V2Env.K_HARM, E5V2Env.G2] = 1                       # param edge (1,2)
    mask[E5V2Env.K_HARM, np_ + E5V2Env.K_MID] = 1             # chain edge (K_harm <- K_mid)
    p_edges, k_edges = edges_from_binary_mask_e5(mask, np_)
    assert p_edges == frozenset({(1, 0), (1, 2)})
    assert k_edges == frozenset({(E5V2Env.K_HARM, E5V2Env.K_MID)})


def test_snapshot_restore_replays_latent_Z():
    """(f): restore replays committed + latent state so a re-rolled masked score is identical."""
    p_true, k_true = true_edges(E5V2Env(subdom=SUB, chain_gamma=CH))
    snap = committed_state(3, subdom=SUB, chain_gamma=CH)
    wm = MaskedE5WorldModel(p_true, k_true, env_seed=3, subdom=SUB, chain_gamma=CH)
    a = score_grid(wm, snap, P0, PANEL)
    b = score_grid(wm, snap, P0, PANEL)  # second pass after internal restores
    assert np.array_equal(a, b)
