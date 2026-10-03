"""PMRT core (cdd_oran/xmethod/methods/pmrt_core.py) on synthetic api.Dataset objects (no harness dependency)."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from cdd_oran.xmethod import api
from cdd_oran.xmethod.methods import pmrt_core as PC

FAST = PC.PmrtCoreConfig(B=499)


def make_dataset(n=600, seed=0, regime="R1", beta=0.0, rho=0.6, lag_effect=0.0, n_blocks=10, delta=0.1):
    """Two actions + placebo, two KPIs with AR(1) state. K0_{t+1} = rho K0_t + beta A0_t + lag_effect A0_{t-1} + noise;
    K1 independent of every action. R1: actions iid U(0, 1); R2: piecewise setpoint (n_blocks, inside [.25, .75]) +
    U(-delta, delta) dither; the setpoint also drives K1 (serial dependence common to action and KPI)."""
    rng = np.random.default_rng([seed, 99])
    p = 3
    if regime == "R1":
        A = rng.uniform(0.0, 1.0, (n, p))
        designs = tuple(api.Design("iid", {"name": "uniform", "lo": 0.0, "hi": 1.0}) for _ in range(p))
    else:
        sp = rng.uniform(0.25, 0.75, (n_blocks, p))[np.minimum(np.arange(n) * n_blocks // n, n_blocks - 1)]
        dz = rng.uniform(-delta, delta, (n, p))
        A = sp + dz
        designs = tuple(api.Design("dither", {"name": "uniform", "lo": -delta, "hi": delta}, random_part=dz[:, j],
                                   fixed_part=sp[:, j]) for j in range(p))
    K = np.zeros(2)
    lag, Y = np.zeros((n, 2)), np.zeros((n, 2))
    a_prev = np.zeros(p)
    for t in range(n):
        lag[t] = K
        drive = A[t, 0] if regime == "R1" else (A[t, 0] - 0.5)
        K = np.array([rho * K[0] + beta * drive + lag_effect * a_prev[0] + rng.normal(0, 0.1),
                      rho * K[1] + (0.0 if regime == "R1" else A[t, 1] - dz[t, 1]) + rng.normal(0, 0.1)])
        Y[t] = K
        a_prev = A[t]
    names = ("P0", "P1", "P_placebo")
    cands = tuple((a, k) for a in names for k in ("K0", "K1")) + (("K0", "K1"), ("K1", "K0"))
    return api.Dataset("E1", regime, n, seed, names, ("K0", "K1"), A, lag, Y, designs, cands,
                       time_index=np.arange(n))


def edge(res, s, t):
    return next(e for e in res.edges if e.source == s and e.target == t)


def test_interface_and_not_applicable():
    d = make_dataset(beta=0.3)
    res = PC.PmrtCore(FAST).run(d)
    assert res.method == "pmrt_core" and res.version == PC.PMRT_CORE_VERSION
    assert len(res.edges) == len(d.candidates)
    for s, t in (("K0", "K1"), ("K1", "K0")):           # lagged-KPI sources: no design -> not applicable
        e = edge(res, s, t)
        assert np.isnan(e.score) and e.p is None and not e.declared
        assert f"{s}->{t}" in res.notes[PC.NOT_APPLICABLE]
    e = edge(res, "P0", "K0")
    assert e.declared and e.sign == 1 and e.p <= 1.0 / 500 + 1e-12


def test_none_design_never_falls_back():
    d = make_dataset(beta=0.5)
    designs = (api.Design("none"),) + d.designs[1:]
    d2 = api.Dataset(**{**d.__dict__, "designs": designs})
    res = PC.PmrtCore(FAST).run(d2)
    for t in ("K0", "K1"):
        e = edge(res, "P0", t)
        assert e.p is None and np.isnan(e.score) and not e.declared
        assert "none" in res.notes[PC.NOT_APPLICABLE][f"P0->{t}"]


def test_deterministic_and_seed_stream():
    d = make_dataset(beta=0.1)
    r1, r2 = PC.PmrtCore(FAST).run(d), PC.PmrtCore(FAST).run(d)
    assert [e.p for e in r1.edges] == [e.p for e in r2.edges]


def test_chunk_invariance():
    d = make_dataset(beta=0.05)
    a = PC.PmrtCore(PC.PmrtCoreConfig(B=299, chunk=1000)).run(d)
    b = PC.PmrtCore(PC.PmrtCoreConfig(B=299, chunk=7)).run(d)
    assert [e.p for e in a.edges] == [e.p for e in b.edges]


def test_dither_redraws_only_random_part():
    d = make_dataset(regime="R2", beta=0.5)
    asg = PC.assignment(d.designs[0], d.X_action[:, 0])
    np.testing.assert_allclose(asg.v, d.designs[0].random_part)          # centred dither (mean 0)
    V = asg.draw(np.random.default_rng(1), 2000)
    assert abs(V.mean()) < 0.01 and V.min() >= -0.1 and V.max() <= 0.1
    np.testing.assert_allclose(asg.var, (0.2 ** 2) / 12)


def test_logged_categorical_propensity_centring():
    n = 400
    rng = np.random.default_rng(3)
    ctx = rng.normal(size=n)
    p1 = 1 / (1 + np.exp(-ctx))
    P = np.column_stack([1 - p1, p1])
    a = (rng.random(n) < p1).astype(float)
    des = api.Design("logged", {"name": "categorical", "values": [0.0, 1.0]}, propensity=P)
    asg = PC.assignment(des, a)
    np.testing.assert_allclose(asg.v, a - p1)
    np.testing.assert_allclose(asg.var, p1 * (1 - p1))
    V = asg.draw(np.random.default_rng(0), 4000)
    np.testing.assert_allclose(V.mean(0), 0.0, atol=0.06)


def test_power_planted_edge_r1_r2():
    for reg in ("R1", "R2"):
        res = PC.PmrtCore(FAST).run(make_dataset(n=1000, regime=reg, beta=0.3, seed=5))
        e = edge(res, "P0", "K0")
        assert e.declared and e.sign == 1, reg
        assert not edge(res, "P_placebo", "K0").declared


def test_lagged_effect_is_not_a_one_step_effect():
    # A0_{t-1} -> K0_{t+1} only: the one-step null of (P0, K0) holds; the lagged action is a predictable covariate.
    ps = [edge(PC.PmrtCore(FAST).run(make_dataset(n=800, beta=0.0, lag_effect=0.5, seed=s)), "P0", "K0").p
          for s in range(8)]
    assert np.mean(np.array(ps) <= 0.05) <= 0.25


@pytest.mark.parametrize("reg", ["R1", "R2"])
def test_null_level_small(reg):
    """Quick level screen (the full F4 run is scratchpad/xmethod/pmrt_fidelity.py): null p ~ U(0, 1)."""
    ps = []
    for s in range(12):
        res = PC.PmrtCore(PC.PmrtCoreConfig(B=199)).run(make_dataset(n=400, regime=reg, beta=0.0, seed=100 + s))
        ps += [e.p for e in res.edges if e.p is not None]
    ps = np.array(ps)
    assert 0.0 < np.mean(ps <= 0.05) <= 0.15
    assert 0.35 < ps.mean() < 0.65


def test_tune_placebo_threshold():
    """R-38 / R-29: tune's tau is the shared conformal rule score.placebo_tau over the DEV runs."""
    from cdd_oran.xmethod.score import placebo_tau
    dev = [make_dataset(n=300, beta=0.3, seed=s) for s in range(4)]
    m = PC.PmrtCore(PC.PmrtCoreConfig(B=99))
    cfg = m.tune(dev)
    res = PC.PmrtCore().run(dev[0], cfg)
    runs = [m.run(d, dataclasses.asdict(m.cfg)) for d in dev]
    pl = sorted(e.score for r in runs for e in r.edges if e.source == "P_placebo" and np.isfinite(e.score))
    assert cfg["tune_info"]["n_placebo_scores"] == len(pl) == 8
    assert cfg["tau"] == placebo_tau(runs) == pl[-1]                     # M = 8 < 19: tau = the largest score
    assert cfg["tau_rule"] == "placebo_conformal_0.05"
    total = sum(sum(1 for e in PC.PmrtCore().run(d, cfg).edges if e.source == "P_placebo" and e.score > cfg["tau"])
                for d in dev)
    assert total == 0
    assert res.notes["declared_tau"] is not None


def test_crt_engine_matches_normal_reference():
    rng = np.random.default_rng(0)
    n = 2000
    v = rng.uniform(-0.5, 0.5, n)
    W = rng.normal(size=(n, 3))
    var = np.full(n, 1 / 12)

    def draw(r, b):
        return r.random((b, n)) - 0.5
    out = PC.crt(v, var, W, draw, np.random.default_rng(1), 4999)
    from scipy.stats import norm
    np.testing.assert_allclose(out["p2"], 2 * norm.sf(np.abs(out["z"])), atol=0.03)


def test_concurrent_actions_and_by_family():
    d = make_dataset(beta=0.3)
    res = PC.PmrtCore(FAST).run(d)
    assert res.notes["adjust"]["P0"]["concurrent"] == ["P1", "P_placebo"]
    assert res.notes["by_family_m"] == 6                 # action -> KPI candidates only (ruling R-6)
    off = PC.PmrtCore(PC.PmrtCoreConfig(B=499, concurrent_actions=False)).run(d)
    assert off.notes["adjust"]["P0"]["concurrent"] == []
    # a "none" action is never a concurrent covariate and its candidates stay in the BY family (m) untested
    d2 = api.Dataset(**{**d.__dict__, "designs": (api.Design("none"),) + d.designs[1:]})
    r2 = PC.PmrtCore(FAST).run(d2)
    assert r2.notes["adjust"]["P1"]["concurrent"] == ["P_placebo"]
    assert r2.notes["by_family_m"] == 6 and r2.notes["n_tested"] == 4


def test_besag_clifford_matches_naive_and_is_chunk_invariant():
    rng = np.random.default_rng(7)
    n = 300
    v = rng.uniform(-0.5, 0.5, n)
    W = np.column_stack([rng.normal(size=n), 0.3 * v + rng.normal(size=n), v + 0.1 * rng.normal(size=n)])
    var = np.full(n, 1 / 12)

    def draw(r, b):
        return r.random((b, n)) - 0.5
    B, h = 999, 20
    out = PC.crt(v, var, W, draw, np.random.default_rng(3), B, chunk=64, seq_h=h)
    out1 = PC.crt(v, var, W, draw, np.random.default_rng(3), B, chunk=1, seq_h=h)
    np.testing.assert_array_equal(out["p2"], out1["p2"])
    np.testing.assert_array_equal(out["p_plus"], out1["p_plus"])
    # naive reference: one draw at a time from the same stream
    r = np.random.default_rng(3)
    sd = np.sqrt(var @ (W * W))
    z = (v @ W) / sd
    cnt, p = np.zeros(3, int), [None] * 3
    for k in range(1, B + 1):
        zb = (draw(r, 1) @ W)[0] / sd
        cnt += np.abs(zb) >= np.abs(z) - 1e-9 * (1 + np.abs(z))
        for j in range(3):
            if p[j] is None and cnt[j] >= h:
                p[j] = h / k
    p = [pj if pj is not None else (1 + cnt[j]) / (B + 1) for j, pj in enumerate(p)]
    np.testing.assert_allclose(out["p2"], p, rtol=0, atol=0)
    assert out["p2"][2] == 1 / (B + 1) and out["draws"][2] == B          # strong column runs to B
    assert out["draws"][0] < B                                           # null column stops early
    fixed = PC.crt(v, var, W, draw, np.random.default_rng(3), B)
    assert fixed["draws_total"] == B and np.all(fixed["p2"] >= 1 / (B + 1))


def test_clipped_normal_and_categorical_rows_designs():
    n = 200
    des = api.Design("iid", {"name": "clipped_normal", "mean": 0.5, "sd": 1.1, "lo": 0.0, "hi": 1.0})
    x = np.clip(np.random.default_rng(0).normal(0.5, 1.1, 400_000), 0, 1)
    asg = PC.assignment(des, np.full(n, 0.5))
    np.testing.assert_allclose(0.5 - asg.v, x.mean(), atol=2e-3)            # analytic mean of the clipped law
    np.testing.assert_allclose(asg.var, x.var(), rtol=1e-2)
    V = asg.draw(np.random.default_rng(1), 4000)
    assert abs(V.mean()) < 0.01 and abs(V.var() - x.var()) < 0.01
    P = np.random.default_rng(2).dirichlet(np.ones(3), n)
    rows = api.Design("logged", {"name": "categorical_rows", "values": [0.0, 0.5, 1.0]}, propensity=P)
    a = np.full(n, 1.0)
    np.testing.assert_allclose(PC.assignment(rows, a).v, a - P @ [0.0, 0.5, 1.0])
    assert isinstance(PC.assignment(api.Design("iid", {"name": "weird"}), a), str)   # not scorable, no crash


def test_categorical_bisection_equals_comparison_count():
    n, m = 50, 101
    vals = np.linspace(0, 1, m)
    rngp = np.random.default_rng(4)
    for P in (np.broadcast_to(rngp.dirichlet(np.ones(m)), (n, m)).copy(), rngp.dirichlet(np.ones(m), n)):
        _, _, draw = PC._categorical(vals, P, n)
        V = draw(np.random.default_rng(9), 300)
        U = np.random.default_rng(9).random((300, n))
        cum = np.cumsum(P, axis=1)
        cum[:, -1] = np.inf
        M = sum((U >= cum[None, :, k]).astype(np.int64) for k in range(m - 1))
        np.testing.assert_array_equal(V, vals[M] - P @ vals)


def test_r14_default_is_unclipped_plain_residual():
    assert PC.PmrtCoreConfig().huber_c is None
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(300, 3)), rng.standard_t(2, size=(300, 2))
    W0, _ = PC.predictable_weights(X, Y, PC.PmrtCoreConfig())
    Wc, _ = PC.predictable_weights(X, Y, PC.PmrtCoreConfig(huber_c=2.5))
    assert np.abs(W0).max() > np.abs(Wc).max()                  # the clip binds on heavy tails; the default does not
    assert np.any(np.abs(Wc) < np.abs(W0))
