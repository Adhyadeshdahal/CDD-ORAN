"""pmrt_nl (cdd_oran/xmethod/methods/pmrt_nl.py): nonlinear PMRT statistics (ruling R-42) on synthetic Datasets."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from cdd_oran.xmethod import api
from cdd_oran.xmethod.covariates import info_order
from cdd_oran.xmethod.methods import pmrt_nl as NL
from tests.test_xmethod_pmrt import edge, make_dataset

FAST = dict(B=299, gbm_trees=40)


def bump_dataset(n=500, seed=0, regime="R1", amp=0.15):
    """make_dataset (null) + a bump of A0 centred in its range on K0: zero linear covariance under R1."""
    ds = make_dataset(n=n, seed=seed, regime=regime)
    Y = ds.Y.copy()
    Y[:, 0] += amp * np.exp(-((ds.X_action[:, 0] - 0.5) ** 2) / (2 * 0.12 ** 2))
    return dataclasses.replace(ds, Y=Y)


def run(ds, **kw):
    return NL.PmrtNl(NL.PmrtNlConfig(**{**FAST, **kw})).run(ds)


@pytest.mark.parametrize("st,adj", [("poly", "ridge"), ("rff", "ridge"), ("gbm", "gbm"), ("poly", "gbm")])
def test_interface_and_power_on_centred_bump(st, adj):
    res = run(bump_dataset(), statistic=st, adjust=adj)
    assert res.method == "pmrt_nl" and res.version == NL.PMRT_NL_VERSION
    assert len(res.edges) == 8
    assert edge(res, "P0", "K0").p <= 0.01                       # a linear PMRT cannot see a centred bump
    kk = [e for e in res.edges if e.source in ("K0", "K1")]
    assert all(e.p is None and np.isnan(e.score) for e in kk)    # lagged-KPI sources: not applicable
    assert res.notes["statistic"] == st and res.notes["adjust"] == adj


def test_deterministic_and_chunk_invariant():
    ds = bump_dataset(seed=3)
    a = run(ds, statistic="rff")
    b = run(ds, statistic="rff")
    c = run(ds, statistic="rff", chunk=7)
    for x, y, z in zip(a.edges, b.edges, c.edges, strict=True):
        assert x.p == y.p == z.p and (x.score == y.score == z.score or np.isnan(x.score))


def test_besag_clifford_and_fixed_b_agree_on_exceedance():
    ds = make_dataset(n=300, seed=4)
    r_fix = run(ds, statistic="poly", seq_h=None, B=199)
    r_seq = run(ds, statistic="poly", seq_h=20, B=199)
    for x, y in zip(r_fix.edges, r_seq.edges, strict=True):
        if x.p is not None:
            assert (x.p <= 0.05) == (y.p <= 0.05) or abs(x.p - y.p) < 0.06


def test_null_level_exact_case():
    """R1, no KPI dynamics driven by the actions, covariates r3 (no lagged actions): W and the profile do not depend
    on the focal draws, so the CRT is exact; raw p <= .05 rate over nulls near .05."""
    ps = []
    for s in range(30):
        ds = make_dataset(n=300, seed=100 + s, regime="R1")
        for st in ("poly", "rff"):
            ps += [e.p for e in run(ds, statistic=st, covariates="r3").edges if e.p is not None]
    ps = np.array(ps)
    assert 0.01 <= np.mean(ps <= 0.05) <= 0.10
    assert 0.4 <= ps.mean() <= 0.6


def test_dither_redraws_only_random_part():
    ds = bump_dataset(regime="R2")
    order = info_order(ds)
    law = NL.design_law(ds.designs[0], ds.X_action[:, 0], order, 48)
    assert np.allclose(law.off + law.v, ds.X_action[order, 0])          # raw value = fixed part + dither
    V = law.draw(np.random.default_rng(0), 50)
    assert np.all(np.abs(V) <= 0.1 + 1e-12)                              # only the +-delta dither is redrawn
    res = run(ds, statistic="poly")
    assert np.isfinite(edge(res, "P0", "K0").score)


def test_uniform_grid_moments_exact_for_piecewise_linear():
    rng = np.random.default_rng(1)
    law = NL.Law(np.zeros(3), np.full(3, 1 / 3), None, np.zeros(3), np.zeros((1, 1)), np.ones((1, 1)), "uniform",
                 glo=np.full(3, -1.0), ghi=np.full(3, 1.0))
    H = rng.normal(size=(3, 9))
    m, v = NL._grid_moments(law, H)
    x = np.linspace(-1, 1, 400_001)
    for t in range(3):
        f = np.interp(x, np.linspace(-1, 1, 9), H[t])
        assert abs(m[t] - f.mean()) < 1e-4 and abs(v[t] - f.var()) < 1e-4


def test_categorical_and_clipped_normal_designs():
    rng = np.random.default_rng(2)
    n = 400
    vals = np.linspace(0, 1, 11)
    A0 = vals[rng.integers(0, 11, n)]
    A1 = np.clip(rng.normal(0.5, 0.4, n), 0, 1)
    Y = np.column_stack([np.sin(6 * A0) + rng.normal(0, .3, n), rng.normal(0, 1, n)])
    designs = (api.Design("iid", {"name": "categorical", "values": vals.tolist(), "p": [1 / 11] * 11}),
               api.Design("iid", {"name": "clipped_normal", "mean": 0.5, "sd": 0.4, "lo": 0.0, "hi": 1.0}))
    cands = tuple((a, k) for a in ("P0", "P1") for k in ("K0", "K1"))
    ds = api.Dataset("E1", "R1", n, 0, ("P0", "P1"), ("K0", "K1"), np.column_stack([A0, A1]), np.zeros((n, 2)),
                     Y, designs, cands, time_index=np.arange(n))
    for st in ("poly", "gbm"):
        res = run(ds, statistic=st)
        assert edge(res, "P0", "K0").p <= 0.01
        assert edge(res, "P1", "K0").p is not None


def test_none_design_not_applicable():
    ds = make_dataset(n=200, seed=5)
    ds = dataclasses.replace(ds, designs=(api.Design("none"),) + ds.designs[1:])
    res = run(ds, statistic="gbm")
    assert edge(res, "P0", "K0").p is None
    assert "P0->K0" in res.notes["not_applicable"]


def test_crt_stat_counts_and_observed_value():
    """crt_stat: T_obs = stat(v_obs); fixed-B p = (1 + #{T_b >= T_obs}) / (B + 1) on a hand-checkable statistic."""
    v = np.array([0.5, -0.2, 0.1])

    def stat(V):
        return (V ** 2).sum(1, keepdims=True)

    def draw(rng, b):
        return rng.uniform(-1, 1, (b, 3))

    r = NL.crt_stat(stat, v, draw, np.random.default_rng(0), 999, 64, None)
    Vb = draw(np.random.default_rng(0), 999)
    assert r["t"][0] == pytest.approx(0.30)
    assert r["p"][0] == pytest.approx((1 + np.sum(stat(Vb)[:, 0] >= 0.30 - 1e-9 * 1.3)) / 1000)


def test_pmrt_core_statistic_option_delegates_to_pmrt_nl():
    """pmrt_core statistic="gbm" (R-42 option, version pmrt-core-v2) = pmrt_nl gbm on the same settings; linear
    stays the default (pmrt-core-v1)."""
    from cdd_oran.xmethod.methods import pmrt_core as PC
    ds = bump_dataset(seed=6)
    core = PC.PmrtCore(PC.PmrtCoreConfig(statistic="gbm", B=299)).run(ds)
    nl = run(ds, statistic="gbm", gbm_trees=NL.PmrtNlConfig().gbm_trees)     # pmrt_core keeps pmrt_nl's defaults
    assert core.method == "pmrt_core" and core.version == PC.PMRT_CORE_V2_VERSION == "pmrt-core-v2"
    for x, y in zip(core.edges, nl.edges, strict=True):
        assert (x.source, x.target, x.p, x.sign, x.declared) == (y.source, y.target, y.p, y.sign, y.declared)
    assert core.notes["statistic"] == "gbm" and core.notes["adjust"] == "gbm"
    lin = PC.PmrtCore().run(ds)
    assert lin.version == PC.PMRT_CORE_VERSION == "pmrt-core-v1" and PC.PmrtCoreConfig().statistic == "linear"
    with pytest.raises(ValueError):
        PC.PmrtCore(PC.PmrtCoreConfig(adjust="gbm")).run(ds)
    with pytest.raises(ValueError):
        PC.PmrtCore(PC.PmrtCoreConfig(statistic="gbm", huber_c=2.5)).run(ds)
    with pytest.raises(ValueError):
        PC.PmrtCore(PC.PmrtCoreConfig(statistic="spline")).run(ds)
