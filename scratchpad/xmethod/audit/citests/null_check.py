"""audit-citests, check 1: each test's null against the authors' implementation on identical inputs.
(a) pdcor proj_perm == dcor 0.7 partial_distance_covariance_test, SAME permutation stream (Generator), fixed B.
(b) rcot2 lpd4: 1 - lpb4(w, T) vs a Monte Carlo of sum_i w_i chi2_1 (upper tail), weights of real candidates.
(c) cmi_knn native config == tigramite CMIknn.run_test_raw (fresh object, same seed), statistic and p.
(d) eq_dropped / constant columns at n 500 and 1000 (R2, every world), and the effect on pcorr / pdcor.
Run: PYTHONPATH=. uv run --group baselines --group citests python scratchpad/xmethod/audit/citests/null_check.py
"""
from __future__ import annotations

import json

import dcor
import numpy as np

from cdd_oran.e2slice.discovery import (
    dist_matrix_1d,
    dist_matrix_euclidean,
    partial_distance_correlation,
    u_center,
    u_inner,
)
from cdd_oran.e2slice.discovery_rcot import _rcot_statistic, _rff_residuals, _standardize
from cdd_oran.e2slice.discovery_rcot_v2 import frozen_config_v2
from cdd_oran.xmethod.methods import _citests_common as cc
from cdd_oran.xmethod.methods.cmi_knn import CMIKnnMethod
from cdd_oran.xmethod.methods.pdcor import _proj_perm_null
from cdd_oran.xmethod.methods.rcot2 import _momentchi2
from cdd_oran.xmethod.worlds.generate import REGIMES_OF, generate_dataset

out: dict = {}
d, _t = generate_dataset("E2", "R2", 300, 3_000_001, kappa=0.25)
p = cc.prepare(d, "eq")
xs, ys = cc.standardize(p.S), cc.standardize(p.Y)

# (a) pdcor
rows = []
for e in range(0, len(p.pairs), 7):
    i, j = p.pairs[e]
    x, y, z = xs[:, [i]], ys[:, [j]], p.z_of(xs, i)
    cu = u_center(dist_matrix_1d(x[:, 0])); tu = u_center(dist_matrix_1d(y[:, 0]))
    zu = u_center(dist_matrix_euclidean(z))
    s, pxz, pyz, *_ = partial_distance_correlation(cu, u_inner(cu, cu), tu, u_inner(tu, tu), zu, u_inner(zu, zu), 1e-12)
    B = 499
    seq = cc.SequentialP(len(x) * u_inner(pxz, pyz), B, None)
    _proj_perm_null(pxz, pyz, np.random.default_rng(11 + e), seq)
    ref = dcor.independence.partial_distance_covariance_test(x, y, z, num_resamples=B,
                                                             random_state=np.random.default_rng(11 + e))
    rows.append({"edge": "{}->{}".format(*d.candidates[e]), "T_ours": seq.obs, "T_dcor": float(ref.statistic),
                 "p_ours": seq.p, "p_dcor": ref.pvalue})
out["pdcor_vs_dcor"] = {"rows": rows,
                        "max_T_reldiff": max(abs(r["T_ours"] - r["T_dcor"]) / abs(r["T_dcor"]) for r in rows),
                        "max_p_diff": max(abs(r["p_ours"] - r["p_dcor"]) for r in rows)}

# (b) rcot2 lpb4 upper tail
hbe, lpb4 = _momentchi2()
cfg = frozen_config_v2()
cfg = type(cfg)(**{**cfg.__dict__, "dz": 100})
xs2, _, _ = _standardize(p.S)
ys2, _, _ = _standardize(p.Y)
rng = np.random.default_rng(5)
rb = []
for e in (0, 5, 11):
    i, j = p.pairs[e]
    fx, fy, rx, ry = _rff_residuals(xs2[:, i], ys2[:, j], p.z_of(xs2, i), cfg.dxy, cfg.dz, cfg.ridge,
                                    np.random.default_rng(e))
    stat, _prod, prodc = _rcot_statistic(rx, ry)
    w = np.linalg.eigvalsh(prodc.T @ prodc / prodc.shape[0]); w = w[w > 1e-12]
    mc = (rng.chisquare(1, size=(200_000, w.size)) @ w)
    for T in (stat, np.quantile(mc, .95), np.quantile(mc, .99)):
        rb.append({"edge": e, "T": float(T), "p_lpb4": 1 - float(lpb4(w, float(T))), "p_mc": float((mc >= T).mean())})
out["rcot2_lpb4_vs_mc"] = {"rows": rb, "max_abs_diff": max(abs(r["p_lpb4"] - r["p_mc"]) for r in rb)}

# (c) cmi_knn native == tigramite
from tigramite.independence_tests.cmiknn import CMIknn  # noqa: E402

m = CMIKnnMethod()
d3, _ = generate_dataset("E4", "R3", 200, 3_000_002, lam=1.0, kappa=0.25)
p3 = cc.prepare(d3, "native")
cfgn = {**m.native_config(), "sig_samples": 200, "workers": 1}
rc = []
for e, (i, j) in enumerate(p3.pairs[:3]):
    seed = 1234 + e
    t1 = m.make_test(cfgn, seed)
    v1, pv1 = t1.run_test_raw(p3.S[:, [i]], p3.Y[:, [j]], p3.z_of(p3.S, i))
    t2 = CMIknn(knn=0.2, shuffle_neighbors=5, significance="shuffle_test", transform="ranks", workers=1,
                sig_samples=200, seed=seed)
    v2, pv2 = t2.run_test_raw(p3.S[:, [i]], p3.Y[:, [j]], p3.z_of(p3.S, i))
    rc.append({"edge": e, "val": [float(v1), float(v2)], "p": [float(pv1), float(pv2)]})
out["cmi_native_vs_tigramite"] = rc

# (d) constant eq columns
cst = {}
for n in (500, 1000):
    for w, regs in REGIMES_OF.items():
        for r in regs:
            for s in range(3_000_000, 3_000_005):
                dd, _ = generate_dataset(w, r, n, s, lam=1.0, kappa=0.25)
                pp = cc.prepare(dd, "eq")
                const_any = [pp.S_names[c] for c in range(pp.S.shape[1]) if np.ptp(pp.S[:, c]) == 0]
                if pp.eq_dropped or const_any:
                    cst.setdefault(f"n{n}:{w}{r}", []).append({"seed": s, "eq_dropped": list(pp.eq_dropped),
                                                              "const_in_S": const_any})
out["eq_constant_columns"] = cst
print(json.dumps(out, indent=1, default=float))
