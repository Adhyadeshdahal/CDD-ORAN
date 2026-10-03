"""audit-classic2, check 3 (R-22 / R-23): not-testable cells per world at n 1000, DEV seed 3_000_000, both arms, for
pc and granger at kappa .25 (R-27 primary), plus kappa 0 for contrast. Also the smallest residual share
RSS / TSS of any candidate's target given its eq conditioning set (exact-fit rule fires at <= 1e-10), so the margin
of the rule is visible. Run: PYTHONPATH=. uv run --group baselines python scratchpad/xmethod/audit/classic2/not_testable.py
"""
from __future__ import annotations

import json
import sys

import numpy as np

from cdd_oran.xmethod.methods import _classic_common as cc
from cdd_oran.xmethod.methods.classic import METHODS
from cdd_oran.xmethod.worlds.generate import REGIMES_OF, generate_dataset

SEED, N = 3_000_000, 1000
CELLS = [(w, r) for w in ("E1", "E2", "E3", "E4", "E5") for r in REGIMES_OF[w]]


def resid_share(y, Z):
    A = np.column_stack([np.ones(len(y)), Z])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ b
    tss = float(((y - y.mean()) ** 2).sum())
    return float(r @ r) / tss if tss > 0 else 0.0


rows = []
for kap in (0.25, 0.0):
    for w, r in CELLS:
        d, tr = generate_dataset(w, r, N, SEED, lam=1.0, kappa=kap)
        cols = cc.columns(d)
        shares = {}
        for s, t in d.candidates:
            fam, j = cc.resolve(d, cols, s)
            if j is None:
                continue
            x, src = cc.source_column(d, cols, fam, j)
            cs = cc.cond_set(d, fam, src, "eq")
            shares[(s, t)] = resid_share(cols.Y[cs.mask, d.kpi_names.index(t)], cs.Z[cs.mask])
        row = {"kappa": kap, "world": w, "regime": r, "n_cand": len(d.candidates),
               "min_resid_share_eq": min(shares.values()),
               "n_share_le_1e-10": sum(v <= 1e-10 for v in shares.values()),
               "n_share_le_1e-6": sum(v <= 1e-6 for v in shares.values())}
        for mname in ("granger", "pc"):
            if mname == "pc" and kap == 0.0:
                continue                                   # pc at kappa 0: same rule as above (exact_fit), skip cost
            for arm in ("eq", "native"):
                res = METHODS[mname]().run(d, {"arm": arm})
                nt = set(res.notes.get("not_testable_edges", []))
                tru = {f"{a}->{b}" for a, b in tr.edges}
                row[f"{mname}:{arm}"] = {"n_nt": len(nt), "n_nt_true": len(nt & tru),
                                         "n_nan_score": sum(not np.isfinite(e.score) for e in res.edges),
                                         "cpu_s": round(res.cpu_s, 2)}
        rows.append(row)
        print(json.dumps(row), file=sys.stderr, flush=True)
json.dump(rows, sys.stdout, indent=0)
