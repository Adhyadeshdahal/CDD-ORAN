"""fdr_layer (PMRT declaration layer; formerly mscr_multi): reductions to the textbook procedures, structure, validity by simulation."""
import numpy as np
import pytest
from scipy.stats import norm

from cdd_oran.decision import fdr_layer as FL
from cdd_oran.decision.eprocess_units import e_bh as e_bh_v
from cdd_oran.discovery.mscr import by_declare


def _bh(p, q):
    m = len(p)
    o = np.argsort(p)
    ok = p[o] <= q * np.arange(1, m + 1) / m
    out = np.zeros(m, bool)
    if ok.any():
        out[o[: np.nonzero(ok)[0].max() + 1]] = True
    return out


def _holm(p, a):
    o = np.argsort(p)
    m = len(p)
    out = np.zeros(m, bool)
    for i, j in enumerate(o):
        if p[j] <= a / (m - i):
            out[j] = True
        else:
            break
    return out


@pytest.fixture
def pvals():
    rng = np.random.default_rng(1)
    return [np.concatenate([rng.uniform(size=40), rng.uniform(0, 0.004, size=20)]) for _ in range(30)]


def test_weighted_step_up_reduces_to_by_and_bh(pvals):
    for p in pvals:
        assert np.array_equal(FL.weighted_step_up(p, None, 0.05, True), by_declare(p, 0.05))
        assert np.array_equal(FL.weighted_step_up(p, None, 0.05, False), _bh(p, 0.05))


def test_weights_must_sum_to_m():
    with pytest.raises(ValueError):
        FL.weighted_step_up(np.full(4, 0.01), np.array([1, 1, 1, 2.0]))
    with pytest.raises(ValueError):
        FL.e_bh(np.ones(4), 0.05, np.array([1, 1, 1, 2.0]))


def test_dagger_flat_is_by_and_bh(pvals):
    par = [[] for _ in range(60)]
    for p in pvals:
        assert np.array_equal(FL.dagger(p, par, 0.05, True), by_declare(p, 0.05))
        assert np.array_equal(FL.dagger(p, par, 0.05, False), _bh(p, 0.05))


def test_dagger_effective_counts_and_gating():
    par = FL.gate_parents()
    ell, mm = FL.dag_effective(par)
    loads = [i for i, h in enumerate(FL.HYPOTHESES) if h[2] == "load"]
    assert len(loads) == 12 and set(ell[loads]) == {4.0} and set(mm[loads]) == {5.0}
    assert ell.sum() - ell[loads].sum() == 48
    p = np.full(60, 1e-6)
    p[loads] = 0.9                                    # gates closed: nothing below them can be rejected
    assert not FL.dagger(p, par, 0.05).any()
    p[loads[0]] = 1e-6
    rej = FL.dagger(p, par, 0.05)
    kids = [i for i, h in enumerate(FL.HYPOTHESES) if h[:2] == FL.HYPOTHESES[loads[0]][:2]]
    assert set(np.nonzero(rej)[0]) == set(kids)


def test_graphical_holm_complete_graph_is_holm(pvals):
    n = 60
    w0 = np.full(n, 1.0 / n)
    G = (np.ones((n, n)) - np.eye(n)) / (n - 1)
    for p in pvals[:10]:
        assert np.array_equal(FL.graphical_holm(p, w0, G, 0.05), _holm(p, 0.05))


def test_gate_graph_is_proper():
    w0, G = FL.gate_graph()
    assert abs(w0.sum() - 1) < 1e-12 and np.all(np.diag(G) == 0) and np.all(G.sum(1) <= 1 + 1e-12)


def test_p_filter_singletons_is_by(pvals):
    for p in pvals:
        assert np.array_equal(FL.p_filter(p, np.arange(60), None, 0.05), by_declare(p, 0.05))


def test_e_bh_matches_agent_v():
    rng = np.random.default_rng(3)
    for _ in range(20):
        e = np.exp(rng.normal(0, 3, 60))
        assert np.array_equal(FL.e_bh(e, 0.05), e_bh_v(e, 0.05))


def test_prior_weights_floor_mean_and_directions():
    rng = np.random.default_rng(0)
    z = {h: float(rng.normal(0, 3)) for h in FL.HYPOTHESES}
    pr = FL.Prior(z=z, n_prior=480)
    for n in (60, 120, 300):
        w = np.array(list(FL.prior_weights(pr, n, floor=0.2).values()))
        assert abs(w.mean() - 1) < 1e-9 and w.min() >= 0.2 - 1e-12 and w.max() <= FL.W_CAP + 1e-9
    d = FL.prior_directions(pr, 3.0)
    assert all(d[h] == (np.sign(z[h]) if abs(z[h]) >= 3 else 0) for h in FL.HYPOTHESES)
    assert np.allclose(FL.rw_weights(np.zeros(5), 0.05), 1.0)
    c = FL.cap_weights(np.array([10.0, 1, 1, 0.5, 0.5, 0, 0, 0, 0, 0]), 3.0)
    assert abs(c.sum() - 13) < 1e-9 and c.max() <= 3 + 1e-9


def _stats(p2, pp, pm, sign):
    def cal(x):                                       # p-to-e calibrator 0.5 p^-0.5 (a valid e-value)
        return 0.5 / np.sqrt(max(x, 1e-12))
    return {h: FL.HypStats(p2=p2[i], p_plus=pp[i], p_minus=pm[i], sign=int(sign[i]), e2=cal(p2[i]),
                           e_plus=cal(pp[i]), e_minus=cal(pm[i]), e_sign=int(sign[i]))
            for i, h in enumerate(FL.HYPOTHESES)}


def test_declare_one_sided_sign_and_untested():
    h0 = FL.HYPOTHESES[0]
    z = {h: 0.0 for h in FL.HYPOTHESES}
    z[h0] = -5.0
    pr = FL.Prior(z=z, n_prior=480)
    p2 = np.full(60, 0.5)
    pp, pm = np.full(60, 0.5), np.full(60, 0.5)
    pm[0], p2[0] = 1e-5, 2e-5
    sign = np.ones(60)
    st = _stats(p2, pp, pm, sign)
    d = FL.declare(st, "by1s", pr, 60)
    assert d[h0]["declared"] and d[h0]["sign"] == -1 and d[h0]["one_sided"]
    st[h0] = FL.HypStats()                             # untested
    for proc in FL.PROCEDURES:
        assert not FL.declare(st, proc, pr, 60)[h0]["declared"]
    with pytest.raises(ValueError):
        FL.declare(st, "wby", None, 60)


def test_fdr_under_dependence_simulation():
    """Arbitrarily (here: block, mixed-sign) correlated Gaussian z, 45 nulls / 15 alternatives placed at the gates and
    their children; every valid procedure keeps FDR (FWER for gholm) near <= q (MC error ~ .01)."""
    rng = np.random.default_rng(7)
    q, reps = 0.1, 300
    hyps = FL.HYPOTHESES
    loads = [i for i, h in enumerate(hyps) if h[2] == "load"]
    alt = set(loads[:3]) | {i for i, h in enumerate(hyps) if h[:2] in {hyps[j][:2] for j in loads[:2]}}
    mu = np.array([3.0 if i in alt else 0.0 for i in range(60)])
    fams = sorted({h[0] for h in hyps})
    grp = np.array([fams.index(h[0]) for h in hyps])
    zp = {h: (rng.normal(mu[i], 1.0) * 1.5) for i, h in enumerate(hyps)}      # independent prior draw
    pr = FL.Prior(z=zp, n_prior=120)
    res = {p: [] for p in FL.PROCEDURES if p != "bh_invalid"}
    for _ in range(reps):
        f = rng.normal(size=4)
        sgn = np.where(np.arange(60) % 2, 1.0, -1.0)
        z = mu + 0.8 * sgn * f[grp] + 0.6 * rng.normal(size=60)      # unit variance, mixed-sign correlation
        st = _stats(2 * norm.sf(np.abs(z)), norm.sf(z), norm.cdf(z), np.sign(z))
        for p in res:
            d = FL.declare(st, p, pr, 120, q=q)
            r = [i for i, h in enumerate(hyps) if d[h]["declared"]]
            v = sum(1 for i in r if i not in alt)
            res[p].append(v / max(len(r), 1) if p != "gholm" and p != "gholm1s" else float(v > 0))
    for p, v in res.items():
        assert np.mean(v) <= q + 0.03, (p, np.mean(v))
