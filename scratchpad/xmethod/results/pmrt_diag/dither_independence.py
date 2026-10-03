"""Are the DEV dithers (E2 R2 n 1000, seeds 100-159) independent across columns and of the null targets?
corr(D^a_t, D^j_t) and corr(D^a_t, D^j_{t-1}) over the 9 columns, and corr(D^a_t, Y_k,t) on truth-null (a, k):
under the design each is ~N(0, 1/n); reported as z = corr * sqrt(n): mean, sd (should be 0, 1) and the largest |z|."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from cdd_oran.xmethod.worlds import generate as G
out = {}
cc, cl, cy = [], [], []
for s in range(3_000_100, 3_000_160):
    ds, tr = G.generate_dataset("E2", "R2", 1000, s, kappa=0.5)
    D = np.column_stack([d.random_part for d in ds.designs])
    n = len(D)
    C = np.corrcoef(D.T)
    cc += list(C[np.triu_indices(9, 1)] * np.sqrt(n))
    L = np.corrcoef(D[1:].T, D[:-1].T)[:9, 9:]
    cl += list(L.ravel() * np.sqrt(n - 1))
    for a, k in ds.meta["primary_candidates"]:
        if (a, k) in tr.null_edges:
            ai, ki = ds.action_names.index(a), ds.kpi_names.index(k)
            cy.append(np.corrcoef(D[:, ai], ds.Y[:, ki])[0, 1] * np.sqrt(n))
for nm, x in (("same_t_cross_column", cc), ("lag1_all_pairs", cl), ("null_D_vs_Y", cy)):
    x = np.array(x)
    out[nm] = {"count": len(x), "mean_z": round(float(x.mean()), 4), "sd_z": round(float(x.std()), 4),
               "max_abs_z": round(float(np.abs(x).max()), 3), "frac_abs_gt_1.96": round(float((np.abs(x) > 1.96).mean()), 4)}
print(json.dumps(out, indent=1))
json.dump(out, open("scratchpad/xmethod/results/pmrt_diag/dither_independence.json", "w"), indent=1)
