"""R-24 truth-free checks (DEV seeds 3_000_000-3_000_004, n 4000).

1. realised noise sd: (Y(kappa) - Y(0)) per KPI, pooled sd / (kappa sigma_k); also X_kpi_lag.
2. exact fits: residual variance share of each KPI target under the equal-information linear fit
   Y_k ~ 1 + all action columns + design_covariates(data) (Z_eq: lagged KPIs, context, setpoints, lagged actions),
   rows with row_mask; share = Var(resid) / Var(Y_k), mean over seeds. 0 = exact fit.

    PYTHONPATH=. uv run python scratchpad/xmethod/noise_check.py > scratchpad/xmethod/results/noise_check.json
"""
import json

import numpy as np

from cdd_oran.xmethod.covariates import design_covariates
from cdd_oran.xmethod.worlds import NOISE_SIGMA, REGIMES_OF, generate_dataset

SEEDS = range(3_000_000, 3_000_005)
N = 4000
KAPPAS = (0.0, 0.1, 0.3, 0.5)


def share(ds):
    _, Z, mask = design_covariates(ds)
    X = np.column_stack([np.ones(ds.n), ds.X_action, Z])[mask]
    Y = ds.Y[mask]
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    R = Y - X @ coef
    return R.var(0) / Y.var(0)


out = {"realised_sd_ratio": {}, "residual_share": {}}
for w, rs in REGIMES_OF.items():
    for r in rs:
        lam = 1.5 if w == "E4" else 1.0
        cell = f"{w} {r}" + (" lam1.5" if w == "E4" else "")
        base = [generate_dataset(w, r, N, s, lam=lam, kappa=0.0)[0] for s in SEEDS]
        sh = {}
        for k in KAPPAS:
            dss = base if k == 0 else [generate_dataset(w, r, N, s, lam=lam, kappa=k)[0] for s in SEEDS]
            sh[str(k)] = [round(float(x), 6) for x in np.mean([share(d) for d in dss], axis=0)]
            if k > 0 and r == "R1":
                dy = np.concatenate([d.Y - b.Y for d, b in zip(dss, base, strict=True)])
                dx = np.concatenate([d.X_kpi_lag - b.X_kpi_lag for d, b in zip(dss, base, strict=True)])
                tgt = k * np.array(NOISE_SIGMA[w])
                out["realised_sd_ratio"].setdefault(w, {})[str(k)] = {
                    "Y": [round(float(x), 4) for x in dy.std(0) / tgt],
                    "X_kpi_lag": [round(float(x), 4) for x in dx.std(0) / tgt]}
        out["residual_share"][cell] = sh
print(json.dumps(out, indent=1))
