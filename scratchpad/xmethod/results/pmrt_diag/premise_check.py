"""H2 / exactness premises on E2 R2 (n 1000, kappa .5, DEV seeds 100-159):
(a) Y_k of every truth-null (a, k) is invariant to a's dithers: regenerate with ONLY a's dither column replaced
    (setpoints, other columns, noise fixed) -> max |dY_k| must be 0 for null k (and > 0 for some true k);
(b) the realised dither = random_part, column == fixed_part + random_part, |random_part| <= w, and the pooled
    random_part / w is U(-1, 1) (KS) and serially uncorrelated (lag-1 corr), per column;
(c) the CRT redraw of the assignment has the declared mean 0 and variance w^2 / 3 (1e6 draws)."""
import json, sys
import numpy as np
from scipy import stats
sys.path.insert(0, "scratchpad/xmethod")
from cdd_oran.xmethod.worlds import generate as G
from cdd_oran.xmethod.methods import pmrt_core as PC

out = {"invariance": {}, "law": {}}
max_null, min_true = 0.0, np.inf
U = {j: [] for j in range(9)}
lag1 = {j: [] for j in range(9)}
for s in range(3_000_100, 3_000_160):
    ds, tr = G.generate_dataset("E2", "R2", 1000, s, kappa=0.5)
    for j, d in enumerate(ds.designs):
        w = d.dist["hi"]
        rp = d.random_part
        assert np.array_equal(ds.X_action[:, j], d.fixed_part + d.random_part) or np.allclose(ds.X_action[:, j], d.fixed_part + d.random_part, rtol=0, atol=1e-12)
        assert np.all(np.abs(rp) <= w) and d.dist["lo"] == -w
        U[j].append(rp / w)
        lag1[j].append(np.corrcoef(rp[1:], rp[:-1])[0, 1])
    if s < 3_000_110:                                  # (a) on 10 seeds: replace one real column's dither
        orig = G._draw_dither
        for a in range(8):
            def patched(rng_sp, rng_d, T, n, ranges, a=a, orig=orig):
                fx, rd, bl = orig(rng_sp, rng_d, T, n, ranges)
                if len(ranges) == 8:
                    w = G.DITHER_DELTA * (ranges[a][1] - ranges[a][0])
                    rd = rd.copy(); rd[:, a] = np.random.default_rng([7801, 99, 7, s, a]).uniform(-w, w, T)
                return fx, rd, bl
            G._draw_dither = patched
            try:
                d2, _ = G.generate_dataset("E2", "R2", 1000, s, kappa=0.5)
            finally:
                G._draw_dither = orig
            for k, kn in enumerate(ds.kpi_names):
                dy = float(np.max(np.abs(d2.Y[:, k] - ds.Y[:, k])))
                if (f"P{a}", kn) in tr.null_edges:
                    max_null = max(max_null, dy)
                else:
                    min_true = min(min_true, dy)
out["invariance"] = {"max_abs_dY_null": max_null, "min_max_abs_dY_true": min_true}
for j in range(9):
    u = np.concatenate(U[j])
    out["law"][ds.action_names[j]] = {"ks_p": float(stats.kstest(u, stats.uniform(-1, 2).cdf).pvalue),
                                      "mean_lag1_corr": float(np.mean(lag1[j])), "max_abs_u": float(np.abs(u).max())}
asg = PC.assignment(ds.designs[0], ds.X_action[:, 0])
V = asg.draw(np.random.default_rng(1), 1000)
w = ds.designs[0].dist["hi"]
out["redraw"] = {"mean_over_w": float(V.mean() / w), "var_ratio": float(V.var() / (w ** 2 / 3)), "declared_var_ratio": float(asg.var[0] / (w ** 2 / 3))}
print(json.dumps(out, indent=1))
json.dump(out, open("scratchpad/xmethod/results/pmrt_diag/premise_check.json", "w"), indent=1)
