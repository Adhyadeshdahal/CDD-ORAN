"""Unit-level off-policy evaluation (scratchpad/e6_dev/decision/ARBITER_DESIGN.md section 3): IPS, SNIPS and doubly
robust (DR / AIPW) value estimates of a target unit policy from randomized logs, with an episode-cluster bootstrap.

Data per logged unit i: logged action a_i (index into the action set), its logging propensity p_i = pi0(a_i | x_i)
> 0, reward r_i, the target policy's action probabilities pi[i, :] (deterministic policy = one-hot), optionally a
reward model q[i, :] = q_hat(x_i, a) (cross-fitted: ``crossfit_ridge_q``), and the episode id (cluster).

  IPS   = mean_i w_i r_i,                       w_i = pi[i, a_i] / p_i
  SNIPS = sum_i w_i r_i / sum_i w_i
  DR    = mean_i ( sum_a pi[i, a] q[i, a] + w_i (r_i - q[i, a_i]) )

IPS and DR are unbiased for the per-unit value when p_i is the true logging propensity (DR for any fixed q; with
cross-fitting q never sees the unit it scores); SNIPS is consistent (O(1/n) bias). ASSUMPTION (flagged in the design):
the episode value is additive across units, i.e. no interference between units -- a unit's reward is attributed to
its own action only.

Episode-cluster bootstrap: resample whole episodes with replacement (units inside an episode share the tape and are
not independent) and recompute the estimator on the pooled resampled units.
"""
from __future__ import annotations

import numpy as np


def _w(a, p, pi):
    a = np.asarray(a, int)
    p = np.asarray(p, float)
    if np.any(p <= 0):
        raise ValueError("logging propensities must be > 0")
    return np.asarray(pi, float)[np.arange(len(a)), a] / p


def ips(r, a, p, pi) -> float:
    return float(np.mean(_w(a, p, pi) * np.asarray(r, float)))


def snips(r, a, p, pi) -> float:
    w = _w(a, p, pi)
    s = w.sum()
    return float((w * np.asarray(r, float)).sum() / s) if s > 0 else float("nan")


def dr(r, a, p, pi, q) -> float:
    a = np.asarray(a, int)
    q = np.asarray(q, float)
    pi = np.asarray(pi, float)
    n = np.arange(len(a))
    return float(np.mean((pi * q).sum(1) + _w(a, p, pi) * (np.asarray(r, float) - q[n, a])))


ESTIMATORS = ("ips", "snips", "dr")


def _est(name, r, a, p, pi, q):
    if name == "ips":
        return ips(r, a, p, pi)
    if name == "snips":
        return snips(r, a, p, pi)
    if name == "dr":
        if q is None:
            raise ValueError("dr needs q")
        return dr(r, a, p, pi, q)
    raise KeyError(name)


def cluster_bootstrap(fn, clusters, n_boot: int = 2000, rng=None, q_lo: float = 0.05, q_hi: float = 0.95):
    """``fn(idx) -> float`` on a unit index array; resample clusters with replacement. -> {est, lo, hi, se, boot}."""
    rng = rng if rng is not None else np.random.default_rng(0)
    clusters = np.asarray(clusters)
    ids = np.unique(clusters)
    members = {c: np.nonzero(clusters == c)[0] for c in ids}
    est = fn(np.arange(len(clusters)))
    boot = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        boot[b] = fn(np.concatenate([members[c] for c in pick]))
    return {"est": float(est), "lo": float(np.nanquantile(boot, q_lo)), "hi": float(np.nanquantile(boot, q_hi)),
            "se": float(np.nanstd(boot, ddof=1)), "boot": boot}


def evaluate(r, a, p, pi, clusters, q=None, n_boot: int = 2000, rng=None, estimators=None) -> dict:
    """Every estimator (DR only when ``q`` is given) with an episode-cluster bootstrap 90 % interval."""
    r, a, p, pi = np.asarray(r, float), np.asarray(a, int), np.asarray(p, float), np.asarray(pi, float)
    q = None if q is None else np.asarray(q, float)
    names = estimators or [e for e in ESTIMATORS if e != "dr" or q is not None]
    rng = rng if rng is not None else np.random.default_rng(0)
    out = {}
    for nm in names:
        res = cluster_bootstrap(lambda idx, nm=nm: _est(nm, r[idx], a[idx], p[idx], pi[idx],
                                                         None if q is None else q[idx]),
                                clusters, n_boot=n_boot, rng=rng)
        res.pop("boot")
        out[nm] = res
    return out


def ridge_fit(X, y, lam: float = 1.0):
    """Ridge on standardised X with an unpenalised intercept -> (coef, intercept, mu, sd)."""
    X = np.asarray(X, float)
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (X - mu) / sd
    ym = float(np.mean(y))
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    coef = np.linalg.solve(A, Z.T @ (np.asarray(y, float) - ym))
    return coef, ym, mu, sd


def ridge_predict(model, X):
    coef, b, mu, sd = model
    return ((np.asarray(X, float) - mu) / sd) @ coef + b


def crossfit_ridge_q(X, a, r, n_actions: int, clusters, lam: float = 1.0, n_folds: int = 5, rng=None):
    """Cross-fitted per-action ridge reward model: episodes split into ``n_folds`` folds; q[i, :] comes from the
    models fitted on the other folds (an action with < 2 training units predicts that fold's training mean)."""
    X, a, r = np.asarray(X, float), np.asarray(a, int), np.asarray(r, float)
    rng = rng if rng is not None else np.random.default_rng(0)
    ids = np.unique(clusters)
    fold_of = dict(zip(rng.permutation(ids), np.arange(len(ids)) % n_folds, strict=True))
    f = np.array([fold_of[c] for c in clusters])
    q = np.zeros((len(r), n_actions))
    for k in range(n_folds):
        te, tr = f == k, f != k
        if not te.any():
            continue
        for act in range(n_actions):
            m = tr & (a == act)
            if m.sum() < 2:
                q[te, act] = float(r[tr].mean()) if tr.any() else 0.0
                continue
            q[te, act] = ridge_predict(ridge_fit(X[m], r[m], lam), X[te])
    return q


__all__ = ["ESTIMATORS", "cluster_bootstrap", "crossfit_ridge_q", "dr", "evaluate", "ips", "ridge_fit",
           "ridge_predict", "snips"]
