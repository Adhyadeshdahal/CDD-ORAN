"""Frozen noiseless KPI spreads sigma_k for the R-24 observation-noise model (E1-E4; E5 uses E5_KPI_MOMENTS).

sigma_k = pooled sd (ddof 0) of Y[:, k] over the NOISELESS R1 datasets (randomised i.i.d. actions; E4 at lam 1.0, which
R1 ignores) of DEV seeds 3_000_000-3_000_019 at n = 24_000 (480_000 rows per world), rounded to 6 decimals and pasted
into worlds/generate.py (NOISE_SIGMA). Reference values from the env moments are printed next to them.

    uv run python scratchpad/xmethod/noise_sigma.py > scratchpad/xmethod/results/noise_sigma.json
"""
import json
import math

import numpy as np

from cdd_oran.envs.v2.e1 import E1V2Env
from cdd_oran.envs.v2.e2 import E2V2Env
from cdd_oran.envs.v2.e3 import E3V2Env
from cdd_oran.xmethod.worlds import generate_dataset

SEEDS = range(3_000_000, 3_000_020)
N = 24_000
REF = {
    "E1": {"source": "E1V2Env.sigma (analytic, P ~ U[0,1])", "sigma": [float(x) for x in E1V2Env.sigma]},
    "E2": {"source": "E2V2Env KPI_MEAN_STD (n 1e6, seed 0, iid uniform params)", "sigma": [float(x) for x in E2V2Env.std]},
    "E3": {"source": "E3V2Env.sigma (analytic steady state)", "sigma": [float(x) for x in E3V2Env.sigma]},
    "E4": {"source": "analytic Var(K_out) under R1 = Var(A on 101-grid) + 2.5^2 Var(Z) = .085 + 6.25",
           "sigma": [math.sqrt(0.085 + 6.25)]},
}
out = {}
for w in ("E1", "E2", "E3", "E4"):
    Ys = [generate_dataset(w, "R1", N, s, lam=1.0)[0].Y for s in SEEDS]
    Y = np.concatenate(Ys)
    sd = Y.std(axis=0)
    per_seed = np.array([y.std(axis=0) for y in Ys])
    out[w] = {"sigma": [round(float(x), 6) for x in sd], "rows": int(Y.shape[0]),
              "per_seed_sd_min": [round(float(x), 6) for x in per_seed.min(0)],
              "per_seed_sd_max": [round(float(x), 6) for x in per_seed.max(0)],
              "reference": REF[w], "rel_diff_vs_reference": [round(float(a / b - 1), 5)
                                                             for a, b in zip(sd, REF[w]["sigma"], strict=True)]}
print(json.dumps(out, indent=1))
