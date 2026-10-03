"""``granger``: one-lag Granger F-test, two arms (xm-classic, CONTRACT sec 3 "simple"; E3 only).

Native arm label: "pairwise Granger" (F7 audit): the restricted model holds only the target's own lag, not the other
actions / lagged KPIs. The equal-information arm (below) is the conditional (VARX) form.

Rows are one-step transitions (state at t -> KPI at t+1), so one lag is what the data carry. For candidate
source -> target (target KPI ``k``):
  restricted  Y_k(t+1) = a + c K_k(t)                         (own lag; intercept only when the source IS K_k(t))
  full        Y_k(t+1) = a + c K_k(t) + g source(t)
  F = (RSS_r - RSS_f) / (RSS_f / (n - q_f)) ~ F(1, n - q_f);  score = -log10 p, sign = sign(g).
This is the bivariate (pairwise) Granger test of statsmodels ``grangercausalitytests`` ``ssr_ftest`` at maxlag 1
(F2 test: identical p on identical inputs). Declaration (primary): pooled BY per family (action | kpi), q = .05.
If the target's own lag is absent (world without lagged KPIs) the restricted model is intercept-only and the test
reduces to the marginal regression F-test (recorded in notes). That is the NATIVE arm (``config['arm'] = 'native'``,
R-18, secondary).

EQUAL-INFORMATION arm (``config['arm'] = 'eq'``, R-17, PRIMARY, default): conditional Granger / VARX with exogenous
regressors -- restricted model = intercept + the source's WHOLE eq conditioning set from the shared helper
(``_classic_common.cond_set``, R-25: lagged KPIs incl. the target's own lag, context, setpoints, actions at t-1 /
t-2, concurrent designed actions; rows with ``row_mask`` False dropped); full model adds the source; F(1, n - rank)
(duplicate helper columns only lower the rank). With one lag this is the partial-correlation F-test of (source,
target) given that set (label in notes).
Degenerate fits: the test is NaN when the source adds no rank to the restricted model (df = n - rank); such a
candidate is also NOT TESTABLE ('source collinear with Z', listed in ``notes['not_testable_edges']`` and
``notes['not_testable_collinear']``; audit-classic2 B: at kappa 0 in E5 lag K2 == P0@t-1). If the
restricted model already explains the target EXACTLY (residual share <= ``EXACT_TOL``; the deterministic worlds E1 /
E3 once the design covariates are added) the F ratio would compare two rounding errors: the candidate is NOT TESTABLE
(orchestrator Q-F1) -- score NaN, p None, never declared, ``notes['not_testable_reason']`` = 'exact fit: deterministic
world under Z' with the count and the edges (``notes['not_testable_edges']``) -- not p = 1 (which would score the method as perfect on those nulls).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import f as fdist

from cdd_oran.xmethod import api

from ._classic_common import (
    EXACT_TOL,
    ClassicBase,
    Scored,
    columns,
    cond_set,
    design_covariates,
    not_testable_notes,
    resolve,
    source_column,
    target_index,
)

EXACT = "exact"            # granger_f status of an exact restricted fit (not testable)

def _rss(A: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, int]:
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ beta
    return float(r @ r), beta, int(np.linalg.matrix_rank(A))


def granger_f(y: np.ndarray, own: np.ndarray | None, x: np.ndarray,
              Z: np.ndarray | None = None) -> tuple[float, float, float, str]:
    """(F, p, coefficient of x, status) of the one-lag Granger test; restricted model = intercept + own lag (if any)
    + the columns of ``Z`` (conditional / VARX form), full = restricted + x. status "ok"; "exact" = the restricted
    model fits y exactly (not testable, NaNs); "degenerate" = x adds no rank / no residual df (NaNs)."""
    nan = float("nan")
    n = len(y)
    base = [np.ones(n)] + ([own] if own is not None else [])
    if Z is not None and Z.shape[1]:
        base += [Z[:, i] for i in range(Z.shape[1])]
    Ar, Af = np.column_stack(base), np.column_stack(base + [x])
    rss_r, _, rank_r = _rss(Ar, y)
    rss_f, beta, rank_f = _rss(Af, y)
    tss = float(((y - y.mean()) ** 2).sum())
    if tss <= 0 or rss_r <= EXACT_TOL * tss:          # exact restricted fit: no information left for the source
        return nan, nan, nan, EXACT
    if rank_f <= rank_r or n - rank_f <= 0 or rss_f <= 0:
        return nan, nan, nan, "degenerate"
    df = n - rank_f
    F = max(rss_r - rss_f, 0.0) / (rss_f / df)
    return float(F), float(fdist.sf(F, 1, df)), float(beta[-1]), "ok"


class Granger(ClassicBase):
    name = "granger"
    version = "1.0"
    uses_p = True
    method_idx = 6
    arms = ("eq", "native")
    defaults: dict[str, Any] = {"q": 0.05, "lags": 1, "arm": "eq"}

    def _score(self, data: api.Dataset, config: dict[str, Any]):
        cols = columns(data)
        eq = config["arm"] == "eq"
        out, no_own, exact, collinear = [], set(), [], []
        for s, t in data.candidates:
            fam, j = resolve(data, cols, s)
            ti = target_index(data, t)
            tname = data.kpi_names[ti]
            own = cols.lags[:, cols.lag_names.index(tname)] if tname in cols.lag_names else None
            if own is None:
                no_own.add(tname)
            if j is None:
                out.append(Scored(s, t, fam, float("nan"), 0, None))
                continue
            if eq:
                x, src = source_column(data, cols, fam, j)
                cs = cond_set(data, fam, src, "eq")
                m = cs.mask
                F, p, g, st = granger_f(cols.Y[m, ti], None, x[m], cs.Z[m])
            else:
                x = cols.actions[:, j] if fam == "action" else cols.lags[:, j]
                if fam == "kpi" and cols.lag_names[j] == tname:
                    own = None                   # source is the target's own lag: test it against intercept only
                F, p, g, st = granger_f(cols.Y[:, ti], own, x)
            if st == EXACT:
                exact.append(f"{s}->{t}")
            elif st == "degenerate":
                collinear.append(f"{s}->{t}")
            if st != "ok":
                out.append(Scored(s, t, fam, float("nan"), 0, None))
                continue
            out.append(Scored(s, t, fam, float(-np.log10(max(p, 1e-300))), int(np.sign(g)), p))
        notes = {"sign_rule": "sign(granger coefficient)", "lags": 1, **not_testable_notes(exact, collinear),
                 "label": ("conditional Granger / VARX (one lag; restricted model = R-3 set + design covariates "
                           "Z_eq, R-17)") if eq else
                          "pairwise Granger (bivariate F-test, one lag; restricted model = target's own lag)"}
        if eq:
            base, _, mask = design_covariates(data)
            notes.update({"eq_covariates": list(base) + ["concurrent:<designed actions>"],
                          "n_rows_used": int(mask.sum())})
        if no_own:
            notes["no_own_lag_targets"] = sorted(no_own)
        return out, notes
