"""F3 (reproduces a known result), analytic / package part (no stored study data):

pcorr  : Gaussian data with a known precision matrix: sample partial correlation -> -W_ij / sqrt(W_ii W_jj)
         (|error| <= 4 / sqrt(n) per entry), and the t-test is exact under Gaussian CI: null size at .05 within its
         Wilson 95% CI over 2000 tests.
cmi_knn: Gaussian CMI is -0.5 log(1 - rho_partial^2) (Cover & Thomas). Known result (Kraskov et al. 2004; Frenzel &
         Pompe 2007): with SMALL k the KSG-type estimator is nearly unbiased -> tigramite CMIknn with knn = 10 at
         n = 2000 within 0.02 nats (mean error over 20 reps). First attempt (recorded, FAILED as an estimation
         check): the TEST default knn = 0.2 n gave mean error -0.055 (0.08 vs 0.22 at rho .6): large k trades bias
         for variance (Runge 2018, AISTATS, chooses it for testing, not estimation); kept as a descriptive row.
pdcor  : dcor's own doctest values for partial_distance_covariance_test reproduced with the frozen e2slice helpers
         (statistic = n * <P_xz, P_yz>): (a, a, b) -> 142.6664416..., (a, b, c) -> 0.

  python scratchpad/xmethod/citests/f3_known.py --out scratchpad/xmethod/citests/F3_KNOWN.json
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from cdd_oran.e2slice.discovery import _project_out, dist_matrix_euclidean, u_center, u_inner
from cdd_oran.xmethod.methods._citests_common import pcorr_given_z, standardize
from cdd_oran.xmethod.methods.pcorr import t_test_p


def wilson(k, n):
    p = k / n; z = 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [p, c - h, c + h]


def pcorr_check(rng):
    d = 6
    A = rng.normal(size=(d, d)) * 0.4
    W = A @ A.T + np.eye(d)                    # precision
    W[0, 5] = W[5, 0] = 0.0                    # x0 _||_ x5 | rest
    W = W + np.eye(d) * max(0, -np.linalg.eigvalsh(W).min() + 0.1)
    C = np.linalg.inv(W)
    true = -W / np.sqrt(np.outer(np.diag(W), np.diag(W)))
    n = 4000
    X = rng.multivariate_normal(np.zeros(d), C, size=n)
    S, Y = X[:, :5], X[:, 5:]                  # target = x5, sources x0..x4
    rho, _ = pcorr_given_z(standardize(S), standardize(Y), [(i, 0) for i in range(5)])
    err = np.abs(rho - true[:5, 5])
    # null size: x0 _||_ x5 | rest, 2000 reps at n = 200
    k = 0
    for _ in range(2000):
        X = rng.multivariate_normal(np.zeros(d), C, size=200)
        r, df = pcorr_given_z(standardize(X[:, :5]), standardize(X[:, 5:]), [(0, 0)])
        k += t_test_p(r, df[0])[0] <= 0.05
    w = wilson(k, 2000)
    return {"max_abs_err": float(err.max()), "tol": 4 / math.sqrt(n), "null_size": w,
            "pass": bool(err.max() <= 4 / math.sqrt(n) and w[1] <= 0.05 <= w[2])}


def cmi_check(rng, knn):
    from tigramite.independence_tests.cmiknn import CMIknn

    t = CMIknn(knn=knn, shuffle_neighbors=5, significance="fixed_thres", transform="ranks", workers=-1)
    errs, rows = [], []
    for r in range(20):
        rho_xy_z = [0.0, 0.3, 0.6][r % 3]
        n = 2000
        z = rng.normal(size=(n, 2))
        x = z @ [0.5, -0.3] + rng.normal(size=n)
        e = rng.normal(size=n)
        ex = x - z @ [0.5, -0.3]
        y = z @ [0.2, 0.4] + rho_xy_z * ex + math.sqrt(1 - rho_xy_z ** 2) * e
        true = -0.5 * math.log(1 - rho_xy_z ** 2)
        # transform="ranks" makes the estimate a copula CMI; Gaussian CMI is invariant to monotone marginals only
        # jointly with Z, so compare on the standardize transform as well
        t.transform = "standardize"
        est = float(t.get_dependence_measure(np.vstack([x, y, z.T]), np.array([0, 1, 2, 2])))
        errs.append(est - true); rows.append([rho_xy_z, true, est])
    return {"mean_err": float(np.mean(errs)), "max_abs_err": float(np.max(np.abs(errs))), "tol_mean": 0.02,
            "rows": rows, "pass": bool(abs(np.mean(errs)) <= 0.02)}


def pdcor_doctest():
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=float)
    b = np.array([[1, 0, 0, 1], [0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 1]], dtype=float)
    c = np.array([[1000, 0, 0, 1000], [0, 1000, 1000, 1000], [1000, 1000, 1000, 1000], [1000, 1000, 0, 1000]],
                 dtype=float)

    def stat(x, y, z):
        ux, uy, uz = (u_center(dist_matrix_euclidean(v)) for v in (x, y, z))
        zz = u_inner(uz, uz)
        pxz = _project_out(ux, uz, zz, u_inner(ux, uz)) if zz > 0 else ux
        pyz = _project_out(uy, uz, zz, u_inner(uy, uz)) if zz > 0 else uy
        return x.shape[0] * u_inner(pxz, pyz)

    s1, s2 = stat(a, a, b), stat(a, b, c)
    return {"aab": s1, "aab_ref": 142.6664416, "abc": s2,
            "pass": bool(abs(s1 - 142.6664416) < 1e-6 and abs(s2) < 1e-6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scratchpad/xmethod/citests/F3_KNOWN.json")
    a = ap.parse_args()
    rng = np.random.default_rng([7802, 3, 0])
    out = {"pdcor_dcor_doctest": pdcor_doctest(), "pcorr_gaussian": pcorr_check(rng), "cmi_knn_gaussian_knn10": cmi_check(rng, 10),
           "cmi_knn_gaussian_knn0.2_descriptive": cmi_check(rng, 0.2)}
    print(json.dumps(out, indent=1))
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
