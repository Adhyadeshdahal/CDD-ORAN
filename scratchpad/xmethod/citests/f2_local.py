"""F2 (own port == reference on identical inputs), local part: pdcor vs `dcor`, pcorr vs OLS coefficient t-test.

  python scratchpad/xmethod/citests/f2_local.py [--out scratchpad/xmethod/citests/F2_LOCAL.json]

Pre-stated tolerances: pdCor statistic |ours - dcor| <= 1e-9 (both U-centred, exponent 1); pcorr t-test p vs OLS
coefficient t-test p: relative difference <= 1e-8. pdcor p-values: the nulls DIFFER by construction (ours, frozen:
permute the source, then project on Z; dcor: permute the projected matrix P_xz), so only distributional agreement
is required: rejection rates at .05 on null and alternative data within 2 binomial SE of each other.
Since R-13 the PRIMARY pdcor null is proj_perm, which IS dcor's null (same statistic n <Pxz, Pyz>, permuted projected
matrix; audit null_check: T rel diff 1.2e-15, p identical 13/13); this script's p comparison uses the frozen x_perm
cross-check null.
"""
from __future__ import annotations

import argparse
import json
import math
from importlib.metadata import version

import dcor
import numpy as np
from scipy import stats

from cdd_oran.e2slice.discovery import (
    dist_matrix_1d,
    dist_matrix_euclidean,
    partial_distance_correlation,
    u_center,
    u_inner,
)
from cdd_oran.xmethod.methods import _citests_synth as synth
from cdd_oran.xmethod.methods._citests_common import SequentialP, prepare
from cdd_oran.xmethod.methods.pcorr import PCorrMethod
from cdd_oran.xmethod.methods.pdcor import _perm_null


def zs(a):
    return (a - a.mean(0)) / a.std(0)


def ours_pdcor(x, y, z, rng=None, b=0):
    cu = u_center(dist_matrix_1d(x)); cs = u_inner(cu, cu)
    tu = u_center(dist_matrix_1d(y)); ts = u_inner(tu, tu)
    zu = u_center(dist_matrix_euclidean(z)); zss = u_inner(zu, zu)
    s, _pxz, pyz, _va, vb, g = partial_distance_correlation(cu, cs, tu, ts, zu, zss, 1e-12)
    if not b:
        return s, None
    seq = SequentialP(abs(s), b, None)
    _perm_null(cu, cs, zu, zss, pyz, vb, rng, seq)
    return s, seq.p


def triple(rng, n, dz, kind):
    z = rng.normal(size=(n, dz))
    x = 0.7 * z[:, 0] + rng.normal(size=n)
    if kind == "null_lin":
        y = 0.6 * z[:, 0] - 0.4 * z[:, -1] + rng.normal(size=n)
    elif kind == "null_nonlin":     # CI holds but pdCov(x, y; z) != 0 (Szekely-Rizzo 2014: pdCov = 0 is not CI)
        y = np.sin(z[:, 0]) + 0.5 * z[:, -1] ** 2 + rng.normal(size=n)
    elif kind == "alt":
        y = np.sin(z[:, 0]) + 0.35 * x ** 2 + rng.normal(size=n)
    else:  # mixed shapes for the statistic check
        y = np.abs(x) * z[:, 0] + rng.standard_t(3, size=n)
    return zs(x), zs(y), zs(z)


def wilson(k, n):
    p = k / n; z = 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(p, 4), round(c - h, 4), round(c + h, 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scratchpad/xmethod/citests/F2_LOCAL.json")
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    rng = np.random.default_rng([7802, 2, 1])
    out = {"dcor": version("dcor"), "numpy": np.__version__}
    # 1. pdCor statistic on identical inputs
    diffs = []
    for r in range(60):
        n = int(rng.choice([20, 50, 120, 300])); dz = int(rng.choice([1, 3, 8]))
        x, y, z = triple(rng, n, dz, ["null_lin", "alt", "mix"][r % 3])
        ours, _ = ours_pdcor(x, y, z)
        ref = float(dcor.partial_distance_correlation(x, y, z))
        diffs.append(abs(ours - ref))
    out["pdcor_stat"] = {"cases": 60, "max_abs_diff": max(diffs), "tol": 1e-9, "pass": max(diffs) <= 1e-9}
    # 2. pdcor p-values, distributional (B = 999 both, n = 200, dz = 3)
    res = {}
    for kind in ("null_lin", "null_nonlin", "alt"):
        rej_o = rej_d = 0; po, pd = [], []
        for r in range(a.reps):
            x, y, z = triple(rng, 200, 3, kind)
            _, p_o = ours_pdcor(x, y, z, np.random.default_rng([7802, 3, r]), 999)
            p_d = float(dcor.independence.partial_distance_covariance_test(x, y, z, num_resamples=999,
                                                                           random_state=r).pvalue)
            po.append(p_o); pd.append(p_d); rej_o += p_o <= .05; rej_d += p_d <= .05
        se = math.sqrt((rej_o / a.reps) * (1 - rej_o / a.reps) / a.reps + (rej_d / a.reps) * (1 - rej_d / a.reps) / a.reps)
        res[kind] = {"reps": a.reps, "rej_ours": wilson(rej_o, a.reps), "rej_dcor": wilson(rej_d, a.reps),
                     "spearman_p": float(stats.spearmanr(po, pd).statistic),
                     "pass": abs(rej_o - rej_d) / a.reps <= 2 * max(se, 1e-9)}
    out["pdcor_p"] = res
    # 3. pcorr t-test vs OLS coefficient t-test on identical inputs
    rel = []
    for s in range(20):
        ds, _ = synth.make("planted" if s % 2 else "null", 400, 3_000_000 + s)
        prep = prepare(ds)
        r = PCorrMethod().run(ds, {})
        S = zs(prep.S); Y = zs(prep.Y); n, d = S.shape
        X = np.c_[np.ones(n), S]
        XtXi = np.linalg.inv(X.T @ X)
        for e, (i, j) in zip(r.edges, prep.pairs, strict=True):
            beta = XtXi @ X.T @ Y[:, j]
            res_ = Y[:, j] - X @ beta
            sig2 = res_ @ res_ / (n - d - 1)
            t = beta[1 + i] / math.sqrt(sig2 * XtXi[1 + i, 1 + i])
            p_ols = 2 * stats.t.sf(abs(t), n - d - 1)
            rel.append(abs(e.p - p_ols) / max(p_ols, 1e-300))
    out["pcorr_vs_ols"] = {"tests": len(rel), "max_rel_diff": max(rel), "tol": 1e-8, "pass": max(rel) <= 1e-8}
    print(json.dumps(out, indent=1, default=lambda v: v.item()))
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=lambda v: v.item())


if __name__ == "__main__":
    main()
