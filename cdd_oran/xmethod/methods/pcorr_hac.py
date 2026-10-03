"""``pcorr_hac``: partial-correlation test with a Newey-West HAC standard error (ruling R-32; design-blind baseline
robust to serial dependence).

For candidate source -> target (target = KPI at t+1) with conditioning set Z (``_classic_common.cond_set``):
  OLS  Y_k(t+1) = a + g source(t) + Z c + u      on the rows in TIME order (``covariates.info_order``)
  H0: g = 0, i.e. source _||_ target | Z in the linear (partial-correlation) sense;
  SE(g) = Newey-West (1987) HAC with the Bartlett kernel, bandwidth from Andrews (1991, Econometrica 59(3) 817-858,
  automatic bandwidth with the AR(1) plug-in, univariate case): S_T = 1.1447 (alpha(1) T)^(1/3),
  alpha(1) = 4 rho^2 / ((1 - rho)^2 (1 + rho)^2), rho = the AR(1) coefficient of the tested coefficient's score
  series v_t = x~_t u_t (x~ = the source residualised on [1, Z]; by Frisch-Waugh-Lovell the HAC variance of g
  depends on v only); maxlags L = floor(S_T) (statsmodels' Bartlett weights 1 - j / (L + 1)), capped at n - 2;
  rho clipped to [-.99, .99] as a numerical guard (no prewhitening);
  small-sample factor n / (n - k) and t(n - k) reference (k = columns of [1, Z, source] after dropping exactly
  collinear Z columns), as statsmodels ``OLS.fit(cov_type='HAC', cov_kwds={'maxlags': L, 'use_correction': True},
  use_t=True)`` (Stata ``newey`` uses the same factor and t reference). F2: equal to that statsmodels call on the
  same design (tests/test_xmethod_pcorr_hac.py).
VARIANT ``config['inference'] = 'fixed_b'`` (ruling R-38): the same statistic referred to the Kiefer-Vogelsang (2005)
fixed-b limit of the Bartlett HAC t at b = (L + 1) / n (``_fixedb.py``: simulated Q(b) draws from a fixed seed, so
no data-dependent randomness); default 't'. Per-edge (t, df, b) in ``notes['t_df_b']``.
score = -log10 p, sign = sign(g) (= the sign of the partial correlation given Z: R-4 natively). No RNG.
Declaration (primary): pooled BY per family (action | kpi | diagnostic, R-2 / R-6), q = .05.

Arms (``config['arm']``): "native" (default, R-32: the R-3 set ``design_covariates(data, False, False, focal=<P>,
concurrent='all')``, "as typically applied, design-blind") and "eq_min" (R-33: R-3 set + ``sp:<P>``,
``covariates.eq_min_covariates``). A lagged-KPI source (secondary family) conditions on the R-3 KPI-source set in both
arms (it has no setpoint). No "eq" arm (any other arm raises).
Not testable (R-22 / R-23; ``notes['not_testable_edges']``, the ``score.py`` format): intercept + Z + source fit the
target EXACTLY (``_classic_common.exact_fit``; deterministic worlds at kappa 0), or the source is collinear with
[1, Z] (zero residual variance) -- score NaN, p None, never declared.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import linalg, stats

from cdd_oran.xmethod import api
from cdd_oran.xmethod.covariates import info_order

from ._classic_common import (
    ClassicBase,
    Scored,
    columns,
    cond_set,
    exact_fit,
    not_testable_notes,
    resolve,
    source_column,
    target_index,
)
from ._fixedb import fixedb_p

ANDREWS_BARTLETT = 1.1447       # Andrews (1991) automatic bandwidth constant, Bartlett kernel
RHO_CLIP = 0.99
COLLINEAR_TOL = 1e-10           # pivoted-QR |R_jj| / |R_00| below which a column is a linear combination of others
OK, EXACT, COLLINEAR = "ok", "exact", "collinear"
INFERENCE = ("t", "fixed_b")


def andrews_lags(v: np.ndarray) -> tuple[int, float, float]:
    """(maxlags L, rho, S_T) of the Andrews (1991) AR(1) plug-in bandwidth for the Bartlett kernel (module doc)."""
    v = np.asarray(v, float)
    n = len(v)
    den = float(v[:-1] @ v[:-1])
    rho = float(v[1:] @ v[:-1]) / den if den > 0 else 0.0
    rho = float(np.clip(rho, -RHO_CLIP, RHO_CLIP))
    alpha = 4.0 * rho ** 2 / ((1.0 - rho) ** 2 * (1.0 + rho) ** 2)
    s_t = ANDREWS_BARTLETT * (alpha * n) ** (1.0 / 3.0)
    return int(min(np.floor(s_t), max(n - 2, 0))), rho, float(s_t)


def independent_columns(Z: np.ndarray) -> np.ndarray:
    """Indices (ascending) of a maximal subset of Z's columns that is linearly independent together with the
    intercept: pivoted QR of the centred, unit-norm columns (constant columns dropped)."""
    if Z.shape[1] == 0:
        return np.zeros(0, np.int64)
    Zc = Z - Z.mean(0)
    nrm = np.linalg.norm(Zc, axis=0)
    live = np.nonzero(nrm > COLLINEAR_TOL * np.maximum(np.linalg.norm(Z, axis=0), 1e-300))[0]
    if not len(live):
        return np.zeros(0, np.int64)
    _, R, piv = linalg.qr(Zc[:, live] / nrm[live], mode="economic", pivoting=True)
    d = np.abs(np.diag(R))
    return np.sort(live[piv[:int(np.sum(d > COLLINEAR_TOL * d[0]))]])


def hac_design(Z: np.ndarray) -> np.ndarray:
    """[1, Z without exactly collinear or constant columns]; the source is added last in ``hac_test``."""
    n = Z.shape[0]
    return np.column_stack([np.ones(n), Z[:, independent_columns(Z)]])


def hac_test(y: np.ndarray, x: np.ndarray, Z: np.ndarray, maxlags: int | None = None,
             inference: str = "t") -> dict[str, Any]:
    """OLS t-test of x in y ~ 1 + Z + x with the HAC standard error (module docstring); rows must be in time order.
    ``inference``: "t" (t(n - k) reference) or "fixed_b" (Kiefer-Vogelsang 2005 fixed-b reference at
    b = (L + 1) / n, ``_fixedb.fixedb_p``); same statistic either way.

    Returns {status, g, se, t, p, df, maxlags, rho, s_t, k, b}; status "exact" / "collinear" = not testable (NaNs)."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    nan = float("nan")
    if inference not in INFERENCE:
        raise ValueError(f"inference must be one of {INFERENCE}, got {inference!r}")
    out = {"status": OK, "g": nan, "se": nan, "t": nan, "p": nan, "df": 0, "maxlags": None, "rho": nan,
           "s_t": nan, "k": 0, "b": nan}
    A = hac_design(np.asarray(Z, float).reshape(len(y), -1))
    if exact_fit(y, np.column_stack([A[:, 1:], x])):
        out["status"] = EXACT
        return out
    n, k = len(y), A.shape[1] + 1
    xr = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    yr = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    sxx = float(xr @ xr)
    if sxx <= COLLINEAR_TOL ** 2 * max(float(x @ x), 1e-300) or n - k <= 0:
        out["status"] = COLLINEAR
        return out
    g = float(xr @ yr) / sxx
    u = yr - g * xr                                     # = the full-model OLS residual (FWL)
    v = xr * u
    if maxlags is None:
        L, rho, s_t = andrews_lags(v)
    else:
        L, rho, s_t = int(maxlags), nan, nan
    lrv = float(v @ v)
    for j in range(1, L + 1):
        lrv += 2.0 * (1.0 - j / (L + 1.0)) * float(v[j:] @ v[:-j])
    df = n - k
    var = max(lrv, 0.0) / sxx ** 2 * n / df
    se = float(np.sqrt(var))
    t = g / se if se > 0 else (np.inf if g != 0 else 0.0)
    b = (L + 1.0) / n                                   # Bartlett M = L + 1 (weights 1 - j / M), b = M / n
    p = float(2.0 * stats.t.sf(abs(t), df)) if inference == "t" else fixedb_p(float(t), b)
    out.update(g=g, se=se, t=float(t), p=p, df=int(df), maxlags=int(L), rho=rho, s_t=s_t, k=int(k), b=float(b))
    return out


class PcorrHac(ClassicBase):
    name = "pcorr_hac"
    version = "1.1"
    uses_p = True
    method_idx = 7                      # no RNG (deterministic); index kept for the shared tag scheme
    arms = ("native", "eq_min")
    defaults: dict[str, Any] = {"q": 0.05, "arm": "native", "maxlags": None, "inference": "t"}

    def run(self, data: api.Dataset, config: dict[str, Any] | None = None) -> api.Result:
        arm = (config or {}).get("arm", self.defaults["arm"])
        if arm not in self.arms:
            raise ValueError(f"pcorr_hac arm must be one of {self.arms}, got {arm!r}")
        return super().run(data, config)

    def _score(self, data: api.Dataset, config: dict[str, Any]):
        cols = columns(data)
        arm = config["arm"]
        order = info_order(data)
        out, exact, collinear = [], [], []
        lags, zsize, rhos, n_used, tstat = {}, {}, [], set(), {}
        for s, t in data.candidates:
            fam, j = resolve(data, cols, s)
            ti = target_index(data, t)
            if j is None:
                out.append(Scored(s, t, fam, float("nan"), 0, None))
                continue
            x, src = source_column(data, cols, fam, j)
            cs = cond_set(data, fam, src, arm)
            rows = order[cs.mask[order]]
            n_used.add(len(rows))
            r = hac_test(cols.Y[rows, ti], x[rows], cs.Z[rows], config.get("maxlags"), config["inference"])
            key = f"{s}->{t}"
            zsize[key] = int(cs.Z.shape[1])
            if r["status"] != OK:
                (exact if r["status"] == EXACT else collinear).append(key)
                out.append(Scored(s, t, fam, float("nan"), 0, None))
                continue
            lags[key] = r["maxlags"]
            tstat[key] = (r["t"], r["df"], r["b"])
            rhos.append(r["rho"])
            p = r["p"]
            out.append(Scored(s, t, fam, float(-np.log10(max(p, 1e-300))), int(np.sign(r["g"])), p))
        lv = list(lags.values())
        notes = {"sign_rule": "sign(OLS coefficient) = sign of pcorr given Z", **not_testable_notes(exact, collinear),
                 "label": ("partial-correlation t-test, Newey-West HAC SE (Bartlett, Andrews 1991 AR(1) bandwidth), "
                           f"Z = {'R-3 set' if arm == 'native' else 'R-3 set + sp:<focal> (eq_min, R-33)'}"),
                 "hac": {"kernel": "bartlett",
                         "bandwidth": ("andrews1991_ar1" if config.get("maxlags") is None
                                       else f"fixed {config['maxlags']}"),
                         "use_correction": True,
                         "reference": ("t(n - k)" if config["inference"] == "t" else
                                       "fixed-b (Kiefer-Vogelsang 2005, Bartlett, b = (L + 1) / n)"),
                         "maxlags_median": float(np.median(lv)) if lv else None,
                         "maxlags_max": int(max(lv)) if lv else None},
                 "maxlags": lags, "z_size": zsize, "n_rows_used": sorted(n_used),
                 "t_df_b": tstat}
        if rhos and np.isfinite(rhos).any():
            notes["hac"]["rho_median"] = float(np.nanmedian(rhos))
        return out, notes
