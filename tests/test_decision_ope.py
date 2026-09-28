"""cdd_oran.decision.ope: IPS / SNIPS / DR on a synthetic contextual bandit with known propensities and a known
target value (unbiasedness over replications), the episode-cluster bootstrap, and cross-fitted q."""
from __future__ import annotations

import functools

import numpy as np
import pytest

from cdd_oran.decision import ope

A = 3
P0 = np.array([0.5, 0.2, 0.3])                              # context-free logging policy (as pi0)
BETA = np.array([[1.0, 0.5], [0.2, -1.0], [-0.5, 1.5]])     # mean reward mu(x, a) = 0.3 a + x . BETA[a]


def mu(X):
    return X @ BETA.T + 0.3 * np.arange(A)


def target(X):
    """Deterministic target: a = 2 if x1 > 0 else 1 if x0 < -0.5 else 0 (one-hot)."""
    a = np.where(X[:, 1] > 0, 2, np.where(X[:, 0] < -0.5, 1, 0))
    return np.eye(A)[a]


def draw(rng, n_ep=30, per_ep=40):
    """Episodes share a random effect (clustered rewards); the logged action ignores it (randomized)."""
    n = n_ep * per_ep
    ep = np.repeat(np.arange(n_ep), per_ep)
    X = rng.normal(size=(n, 2))
    a = rng.choice(A, size=n, p=P0)
    r = mu(X)[np.arange(n), a] + rng.normal(0, 1.0, n_ep)[ep] + rng.normal(0, 1.0, n)
    return X, a, r, P0[a], ep


@functools.cache
def true_value():
    X = np.random.default_rng(0).normal(size=(2_000_000, 2))
    return float((target(X) * mu(X)).sum(1).mean())


def test_ips_snips_dr_are_unbiased_on_a_synthetic_bandit():
    rng = np.random.default_rng(1)
    V = true_value()
    est = {"ips": [], "snips": [], "dr0": [], "dr_bad": [], "dr_cf": []}
    for _ in range(300):
        X, a, r, p, ep = draw(rng)
        pi = target(X)
        est["ips"].append(ope.ips(r, a, p, pi))
        est["snips"].append(ope.snips(r, a, p, pi))
        est["dr0"].append(ope.dr(r, a, p, pi, np.zeros((len(r), A))))                 # q = 0 -> IPS
        est["dr_bad"].append(ope.dr(r, a, p, pi, mu(X) + 2.0))                        # wrong model: still unbiased
        if len(est["dr_cf"]) < 60:
            q = ope.crossfit_ridge_q(X, a, r, A, ep, lam=1.0, rng=rng)
            est["dr_cf"].append(ope.dr(r, a, p, pi, q))
    for k, v in est.items():
        v = np.asarray(v)
        se = v.std(ddof=1) / np.sqrt(len(v))
        assert abs(v.mean() - V) < 4 * se + (0.02 if k == "snips" else 0.0), (k, v.mean(), V, se)
    assert np.allclose(est["dr0"], est["ips"])
    assert np.std(est["dr_cf"]) < np.std(est["ips"][:60])                              # a good q reduces variance


def test_deterministic_logging_equals_on_policy_mean():
    rng = np.random.default_rng(2)
    a = np.zeros(500, int)
    r = rng.normal(size=500)
    pi = np.eye(A)[a]
    p = np.ones(500)
    assert ope.ips(r, a, p, pi) == pytest.approx(r.mean())
    assert ope.snips(r, a, p, pi) == pytest.approx(r.mean())
    with pytest.raises(ValueError):
        ope.ips(r, a, np.zeros(500), pi)


def test_cluster_bootstrap_resamples_whole_episodes_and_covers_the_truth():
    rng = np.random.default_rng(3)
    V = true_value()
    cover, width_iid, width_cl = 0, [], []
    for _ in range(40):
        X, a, r, p, ep = draw(rng, n_ep=20, per_ep=30)
        pi = target(X)
        res = ope.evaluate(r, a, p, pi, ep, n_boot=300, rng=rng)
        cover += res["ips"]["lo"] <= V <= res["ips"]["hi"]
        width_cl.append(res["ips"]["hi"] - res["ips"]["lo"])
        iid = ope.evaluate(r, a, p, pi, np.arange(len(r)), n_boot=300, rng=rng)
        width_iid.append(iid["ips"]["hi"] - iid["ips"]["lo"])
    assert cover >= 0.8 * 40                                # nominal 90 % interval
    assert np.mean(width_cl) > np.mean(width_iid)           # clustered rewards -> wider than the iid bootstrap
    out = ope.cluster_bootstrap(lambda idx: float(len(np.unique(ep[idx]))), ep, n_boot=50, rng=rng)
    assert out["est"] == 20 and out["boot"].max() <= 20


def test_evaluate_reports_every_estimator_with_an_interval():
    rng = np.random.default_rng(4)
    X, a, r, p, ep = draw(rng, n_ep=10, per_ep=20)
    q = ope.crossfit_ridge_q(X, a, r, A, ep, rng=rng)
    res = ope.evaluate(r, a, p, target(X), ep, q=q, n_boot=200, rng=rng)
    assert set(res) == {"ips", "snips", "dr"}
    for v in res.values():
        assert v["lo"] <= v["hi"] and np.isfinite(v["est"]) and v["se"] > 0
    assert set(ope.evaluate(r, a, p, target(X), ep, n_boot=50, rng=rng)) == {"ips", "snips"}
