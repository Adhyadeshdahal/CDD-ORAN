"""xmethod adapter ``pcorr``: partial correlation given all other sources (E1 discovery v2 score, wrapped).

Score |rho| from ``e1slice.discovery_v2.partial_correlation_scores`` on z-standardized columns (its STOP branches
kept: a collinear source is a recorded STOP). p-value: the exact partial-correlation t-test,
t = rho sqrt(df / (1 - rho^2)), df = n - 2 - |Z| (orchestrator decision Q4; E1 v2 had no null and cut at the
per-target largest gap, kept as ``declared_native``). Sign: native (sign of rho).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from cdd_oran.analysis.auto_threshold import largest_gap
from cdd_oran.e1slice.discovery_v2 import EPS_COND, FROZEN_FLOOR, partial_correlation_scores
from cdd_oran.xmethod.methods._citests_common import CITestBase, Prepared


def _zstd(a: np.ndarray, what: str) -> np.ndarray:
    sd = a.std(axis=0)
    if (sd == 0).any():
        raise ValueError(f"pcorr: zero-variance {what} column(s) {np.nonzero(sd == 0)[0].tolist()}")
    return (a - a.mean(axis=0)) / sd


def t_test_p(rho: np.ndarray, df: float) -> np.ndarray:
    r2 = np.clip(rho ** 2, 0.0, 1.0)
    with np.errstate(divide="ignore"):
        t = np.abs(rho) * np.sqrt(df / np.maximum(1.0 - r2, 1e-300))
    return 2.0 * stats.t.sf(t, df)


class PCorrMethod(CITestBase):
    name = "pcorr"
    version = "xm-pcorr-v1"
    method_key = 2
    native_sign = True

    def _test(self, prep: Prepared, data, cfg: dict[str, Any]) -> dict[str, Any]:
        xs, ys = _zstd(prep.S, "source"), _zstd(prep.Y, "target")
        n = xs.shape[0]
        rho, df = np.zeros(len(prep.pairs)), np.zeros(len(prep.pairs))
        for cols, es in prep.column_groups():                         # Z = the other columns of the group
            xg = xs[:, cols]
            d = len(cols)
            if np.linalg.matrix_rank(np.concatenate([xg, np.ones((n, 1))], axis=1)) < d + 1:
                raise ValueError("pcorr: rank-deficient source design")
            part = partial_correlation_scores(xg, ys, EPS_COND)      # (k, d)
            for e in es:
                i, j = prep.pairs[e]
                rho[e], df[e] = part[j, cols.index(i)], n - 2 - (d - 1)
        p = t_test_p(rho, df)
        native = np.zeros(len(prep.pairs), dtype=bool)
        for j in {j for _, j in prep.pairs}:
            es = [e for e, (_, jj) in enumerate(prep.pairs) if jj == j]
            row = np.abs(rho[es])
            if row.max() > row.min():
                native[es] = row >= largest_gap(row, floor=FROZEN_FLOOR)
        return {"score": np.abs(rho), "p": p, "sign": rho, "declared_native": native,
                "notes": {"df": sorted({int(v) for v in df}), "native_rule": "per-target largest_gap(floor=0) over the target's candidates"}}
