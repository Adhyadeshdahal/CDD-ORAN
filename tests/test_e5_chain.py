"""E5 chain-layer / H>=2 rollout regression tests (DEV env; plan 015 §3, sol review #4).

Locks the mediated-path semantics:
(b) MaskedE5WorldModel(all true edges) == true env at H=2 (equivalence generalizes to the chain horizon);
(c) the chain edge (K_harm<-K_mid) is INERT at H=1 (scored terminal reads pre-action K_mid) and ACTIVE at
    H=2 (missing the 2-hop path under-corrects -> regret);
(d) missing BOTH the direct and chain edges at H=2 is strictly worse than missing only one.
"""
from __future__ import annotations

import numpy as np

from cdd_oran.benchmark.masked_world_model import MaskedE5WorldModel
from cdd_oran.envs.v2.e5 import E5V2Env
from scripts.e5_spine import (
    CHAIN_EDGE,
    HARMFUL_EDGE,
    build_ingate_bank,
    committed_state,
    e5_panel,
    score_grid_h,
    spine_regret_e5,
    true_edges,
)

SUB, CH = 0.20, 0.20
PANEL = e5_panel()
P0 = E5V2Env.P0


def test_all_true_masked_equals_true_env_h2():
    """(b): with every true edge, the masked model matches the true env at H=2."""
    p_true, k_true = true_edges(E5V2Env(subdom=SUB, chain_gamma=CH))
    for s in range(15):
        snap = committed_state(s, subdom=SUB, chain_gamma=CH)
        true_env = E5V2Env(env_seed=s, theta=0.0, subdom=SUB, chain_gamma=CH)
        wm = MaskedE5WorldModel(p_true, k_true, env_seed=s, subdom=SUB, chain_gamma=CH)
        assert np.allclose(
            score_grid_h(true_env, snap, P0, PANEL, 2), score_grid_h(wm, snap, P0, PANEL, 2)
        ), f"seed {s}"


def test_chain_edge_inert_at_h1_active_at_h2():
    """(c): omitting the chain edge is a no-op at H=1 but costs regret at H=2."""
    bank = build_ingate_bank(subdom=SUB, chain_gamma=CH, n=10)
    p_true, k_true = true_edges(E5V2Env(subdom=SUB, chain_gamma=CH))
    miss_chain = (p_true, frozenset(k_true) - {CHAIN_EDGE})
    r1 = spine_regret_e5(*miss_chain, bank, PANEL, subdom=SUB, chain_gamma=CH, H=1)
    r2 = spine_regret_e5(*miss_chain, bank, PANEL, subdom=SUB, chain_gamma=CH, H=2)
    assert r1["mean_norm"] < 1e-9, "chain edge must be inert at H=1"
    assert r2["mean_norm"] > 0.05, "chain edge must be decision-active at H=2"


def test_miss_both_worse_than_single_at_h2():
    """(d): missing direct AND chain at H=2 is strictly worse than missing only the chain."""
    bank = build_ingate_bank(subdom=SUB, chain_gamma=CH, n=10)
    p_true, k_true = true_edges(E5V2Env(subdom=SUB, chain_gamma=CH))
    miss_chain = spine_regret_e5(
        p_true, frozenset(k_true) - {CHAIN_EDGE}, bank, PANEL, subdom=SUB, chain_gamma=CH, H=2)
    miss_both = spine_regret_e5(
        p_true - {HARMFUL_EDGE}, frozenset(k_true) - {CHAIN_EDGE}, bank, PANEL,
        subdom=SUB, chain_gamma=CH, H=2)
    assert miss_both["mean_norm"] > miss_chain["mean_norm"] + 0.05
