"""E2 causal-conflict spine pytest — the productized discovery→do-propagation→decision primitive.

Proves the masked world model is a strict, bit-exact generalization of the frozen E2 env:

(a) MaskedE2WorldModel(all true edges) == the true E2V2Env (score_grid identical);
(b) MaskedE2WorldModel(true edges − P0→K5) == E2V2Env(decoy_omit_p0_k5=True) (bit-exact) — the
    productized primitive reproduces the FROZEN decoy;
(c) the oracle arm has zero decision regret;
(d) discovery FALSE edges are inert (regret unchanged) but a MISSING harmful edge triggers the trap;
(e) binary-mask decoding keeps params and drops lagged-KPI candidates.
"""

from __future__ import annotations

import numpy as np

from cdd_oran.analysis.v2_regret import score_grid
from cdd_oran.benchmark.masked_world_model import (
    TRUE_PARAM_EDGES,
    MaskedE2WorldModel,
    make_masked_factory,
    param_edges_from_binary_mask,
)
from cdd_oran.envs.v2.e2 import E2V2Env
from scripts.e2_decision_gate import (
    FULL_PANEL_IDS,
    TOL,
    build_bank,
    build_panel,
)
from scripts.e2_spine import HARMFUL_EDGE, spine_regret

SHARED = 0  # P0


def _bank_and_panel():
    bank = build_bank(E2V2Env)
    assert bank.feasible
    panel = build_panel(E2V2Env(0), FULL_PANEL_IDS)
    return bank, panel


def test_full_edges_equal_true_env():
    """(a): every true parent live ⇒ the masked model scores identically to the true env."""
    bank, panel = _bank_and_panel()
    for seed, snap in bank.positives[:8]:
        s_true = score_grid(E2V2Env(env_seed=seed), snap, SHARED, panel)
        s_mask = score_grid(
            MaskedE2WorldModel(TRUE_PARAM_EDGES, env_seed=seed), snap, SHARED, panel
        )
        assert np.array_equal(s_true, s_mask), f"seed {seed}: masked full-edges != true env"


def test_omit_p0k5_equals_frozen_decoy():
    """(b): omitting only P0→K5 reproduces the FROZEN decoy bit-exactly (the key equivalence)."""
    bank, panel = _bank_and_panel()
    edges = TRUE_PARAM_EDGES - {HARMFUL_EDGE}
    for seed, snap in bank.positives[:8]:
        s_decoy = score_grid(
            E2V2Env(env_seed=seed, decoy_omit_p0_k5=True), snap, SHARED, panel
        )
        s_mask = score_grid(MaskedE2WorldModel(edges, env_seed=seed), snap, SHARED, panel)
        assert np.array_equal(s_decoy, s_mask), f"seed {seed}: masked omit-(5,0) != frozen decoy"


def test_oracle_arm_zero_regret():
    """(c): the full-structure planner never walks into the trap — regret 0, no action mismatch."""
    bank, panel = _bank_and_panel()
    gaps, mism = spine_regret(TRUE_PARAM_EDGES, bank.positives, panel)
    assert mism == 0
    assert max(abs(g) for g in gaps) <= TOL


def test_false_edges_are_inert():
    """(d): declaring params that are not a KPI's true parents does not change the decision."""
    bank, panel = _bank_and_panel()
    # (5,3): param 3 is not a parent of K5; (0,5): param 5 is not a parent of K0. Both inert.
    with_false = TRUE_PARAM_EDGES | {(5, 3), (0, 5)}
    for seed, snap in bank.positives[:8]:
        s_true = score_grid(E2V2Env(env_seed=seed), snap, SHARED, panel)
        s_false = score_grid(MaskedE2WorldModel(with_false, env_seed=seed), snap, SHARED, panel)
        assert np.array_equal(s_true, s_false)
    gaps, mism = spine_regret(with_false, bank.positives, panel)
    assert mism == 0 and max(abs(g) for g in gaps) <= TOL


def test_missing_harmful_edge_triggers_trap():
    """(d): a mask identical to the oracle EXCEPT for the harmful edge lands at the decoy value."""
    bank, panel = _bank_and_panel()
    caught, _ = spine_regret(TRUE_PARAM_EDGES, bank.positives, panel)
    missed, _ = spine_regret(TRUE_PARAM_EDGES - {HARMFUL_EDGE}, bank.positives, panel)
    assert float(np.mean(caught)) <= TOL
    assert float(np.mean(missed)) >= 0.10  # tau_E2: the trap fires


def test_param_edges_from_binary_mask_drops_lagged_kpis():
    """(e): candidates < num_params (params) are kept; >= num_params (lagged KPIs) are dropped."""
    mask = np.zeros((E2V2Env.num_kpis, 14), dtype=int)
    mask[5, 0] = 1   # P0 -> K5   (param, kept)
    mask[5, 7] = 1   # P7 -> K5   (param, kept)
    mask[5, 12] = 1  # lagged K5 -> K5 (candidate 12 >= num_params, dropped)
    mask[0, 1] = 1   # P1 -> K0   (param, kept)
    edges = param_edges_from_binary_mask(mask)
    assert edges == frozenset({(5, 0), (5, 7), (0, 1)})


def test_make_masked_factory_binds_edges():
    """The factory mints a fresh masked env per seed with the fixed edge set."""
    factory = make_masked_factory(TRUE_PARAM_EDGES - {HARMFUL_EDGE})
    env_a, env_b = factory(3), factory(4)
    assert env_a is not env_b
    assert env_a.param_edges == env_b.param_edges == TRUE_PARAM_EDGES - {HARMFUL_EDGE}
