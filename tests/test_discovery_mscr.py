"""MSCR-v2 behavioral tests: BY selection semantics, the param-family/conditioner split, input checks."""
from __future__ import annotations

import numpy as np
import pytest

from cdd_oran.discovery import MSCRConfig, by_declare, discover_mscr


def test_by_cutoffs_m8():
    # rank-1 cutoff q/(8*H_8) = 0.0022997; floor 1/3000 passes, just above the cutoff fails
    assert by_declare([1 / 3000] + [1.0] * 7, 0.05)[0]
    assert not by_declare([0.00231] + [1.0] * 7, 0.05).any()
    # step-up: the rank-3 cutoff (0.0069) admits all three smallest p-values
    assert by_declare([0.001, 0.004, 0.0068] + [1.0] * 5, 0.05).sum() == 3


def test_lagged_columns_are_conditioners_not_tested():
    rng = np.random.default_rng(0)
    x = rng.uniform(size=(400, 5))
    y = np.column_stack([x[:, 0] + 0.1 * rng.normal(size=400), rng.normal(size=400)])
    res = discover_mscr(x, y, n_params=3, seed=1, config=MSCRConfig(n_perm=199))
    assert res.pvals.shape == res.declared.shape == (2, 3)
    assert res.declared[0, 0] and not res.declared[1].any()


def test_gated_edge_found_via_single_conditioner():
    # y depends on x0 only where x1 is high: the max over conditioners should isolate the gate
    rng = np.random.default_rng(3)
    x = rng.uniform(size=(3000, 4))
    y = 5.0 * x[:, 2] + 2.0 * x[:, 0] * (x[:, 1] > 0.83) + 0.1 * rng.normal(size=3000)
    res = discover_mscr(x, y, n_params=4, seed=0, config=MSCRConfig(n_perm=299))
    assert res.param_edges() >= {(0, 0), (0, 1), (0, 2)}


def test_rejects_bad_shapes():
    x = np.zeros((50, 4))
    with pytest.raises(ValueError):
        discover_mscr(x, np.zeros((49, 2)), n_params=2, seed=0)
    with pytest.raises(ValueError):
        discover_mscr(x, np.zeros((50, 2)), n_params=5, seed=0)
