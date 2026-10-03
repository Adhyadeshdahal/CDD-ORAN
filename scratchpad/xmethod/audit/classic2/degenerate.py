"""audit-classic2: granger eq candidates with NaN score that are NOT listed not-testable (status 'degenerate'), E5 at
kappa 0 (sanity setting) and kappa .25, n 1000, seed 3_000_000; plus pc NaN-score candidates outside not_testable.
Run: PYTHONPATH=. uv run --group baselines python scratchpad/xmethod/audit/classic2/degenerate.py"""
import json

import numpy as np

from cdd_oran.xmethod.methods.classic import METHODS
from cdd_oran.xmethod.worlds.generate import generate_dataset

out = {}
for kap in (0.0, 0.25):
    for r in ("R1", "R2"):
        d, tr = generate_dataset("E5", r, 1000, 3_000_000, kappa=kap)
        res = METHODS["granger"]().run(d, {"arm": "eq"})
        nt = set(res.notes.get("not_testable_edges", []))
        silent = [f"{e.source}->{e.target}" for e in res.edges
                  if not np.isfinite(e.score) and f"{e.source}->{e.target}" not in nt]
        out[f"E5{r}k{kap}"] = {"silent_nan": silent, "true": sorted(f"{a}->{b}" for a, b in tr.edges
                                                                    if f"{a}->{b}" in silent),
                               "lag_kpi_constant": [k for j, k in enumerate(d.kpi_names)
                                                    if np.std(d.X_kpi_lag[:, j]) == 0]}
print(json.dumps(out, indent=1))
