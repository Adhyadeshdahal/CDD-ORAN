"""Design covariates of the cross-method study: ONE shared builder (CONTRACT section 8, ruling R-17).

``design_covariates(data)`` returns exactly the predictable covariates that ``pmrt_core`` adjusts for, with one
canonical encoding, so every method of the equal-information arm (R-17) conditions on the same set Z_eq:

  block            columns (names)                                   when
  lagged KPIs      ``lag_kpi:<K>``                                   always (``include_kpi_lag``)
  context          ``ctx:<name>`` (``meta["context_names"]``,        R3 (``Dataset.context`` present)
                   else ``ctx:c<j>``)
  setpoints        ``sp:<P>`` = ``Design.fixed_part``                every dither design, focal one included
                                                                     (``include_setpoints``)
  lagged actions   ``<P>@t-<L>`` for L = 1..lags, every action       ``include_lagged_actions`` (lags = 2)
                   column incl. the placebos; ``gap@t-<L>``
  concurrent       ``concurrent:<P>``: the other action columns at t  only when ``focal`` is given

Encoding (identical to pmrt_core before this helper existed, bit for bit):
  * rows are processed in information order (``time_index`` if given, else row order; stable argsort) and returned
    in the Dataset's ROW order (``matrix[order]`` is what pmrt_core uses);
  * a block column that is all-NaN is dropped; a partly missing one is 0-filled and followed by its indicator
    ``<name>:missing``;
  * lag L of row t (information order) is the action vector of the previous L-th row if that row is a true
    previous step (``time_index`` difference == L), else 0. The first L rows are 0-filled WITHOUT an indicator;
    the indicator ``gap@t-<L>`` is added only if some later row has a time gap. At most n - 1 lags.
  * ``row_mask[t]`` = True iff every included lag of row t is a true previous step (all True without lagged
    actions). pmrt_core does not drop rows (its burn-in covers the first rows); a method that cannot handle
    0-filled lags may drop the rows with ``row_mask`` False.

Concurrent columns (``focal`` = action index): ``concurrent="designed"`` is pmrt_core's rule (R-8): every other
column whose design kind is iid / dither / logged, and none at all if the focal column has no known design;
``concurrent="all"`` is the R-3 conditioning set (every other action column, incl. the placebos).

The R-3 set (ruling R-18 native arm, R-19 pmrt ablation) is
``design_covariates(data, include_setpoints=False, include_lagged_actions=False, focal=ai)``.
The eq_min set (ruling R-33: the minimal design information) = the R-3 set (concurrent="all") + ``sp:<focal>`` only:
``eq_min_covariates(data, focal)`` = ``design_covariates(data, True, False, focal=ai, concurrent="all",
setpoints="focal")``; a focal column without a dither design (R1 / R3 / R4, the placebos there) has no setpoint, so
its eq_min set IS its R-3 set.
Reads only ``api.Dataset`` fields (never Truth); from ``meta`` only the context column names.
"""
from __future__ import annotations

import numpy as np

from cdd_oran.xmethod import api

DESIGNED_KINDS = ("iid", "dither", "logged")


def info_order(data: api.Dataset) -> np.ndarray:
    """Information order of the rows: ``time_index`` (stable argsort) if given, else row order."""
    if data.time_index is None:
        return np.arange(data.n)
    return np.argsort(np.asarray(data.time_index), kind="stable")


def _finite_block(M: np.ndarray | None, names: list[str]) -> tuple[list, list]:
    """Usable columns of M: all-NaN columns dropped; partly missing -> 0-filled + ``:missing`` indicator."""
    if M is None:
        return [], []
    M = np.asarray(M, float)
    if M.ndim == 1:
        M = M[:, None]
    cols, out = [], []
    for j in range(M.shape[1]):
        c = M[:, j]
        ok = np.isfinite(c)
        if not ok.any():
            continue
        if ok.all():
            cols.append(c)
            out.append(names[j])
        else:
            cols += [np.where(ok, c, 0.0), (~ok).astype(float)]
            out += [names[j], f"{names[j]}:missing"]
    return cols, out


def concurrent_indices(data: api.Dataset, focal: int, concurrent: str = "designed") -> list[int]:
    """Other action columns usable at t for the focal column (module docstring)."""
    if concurrent == "all":
        return [j for j in range(len(data.action_names)) if j != focal]
    if concurrent != "designed":
        raise ValueError(f"concurrent must be 'designed' or 'all', got {concurrent!r}")
    if data.designs[focal].kind not in DESIGNED_KINDS:
        return []
    return [j for j, d in enumerate(data.designs) if j != focal and d.kind in DESIGNED_KINDS]


def design_covariates(data: api.Dataset, include_setpoints: bool = True, include_lagged_actions: bool = True, *,
                      lags: int = 2, focal: int | str | None = None, concurrent: str = "designed",
                      include_kpi_lag: bool = True, include_context: bool = True, setpoints: str = "all"
                      ) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """(names, matrix [n, c] in ROW order, row_mask [n]) of the design covariates (module docstring).

    Defaults = pmrt_core's base set Z_eq without the focal-specific concurrent columns; pass ``focal`` (index or
    action name) to append them. ``include_setpoints=False, include_lagged_actions=False`` gives the R-3 set.
    ``setpoints``: "all" (every dither design) or "focal" (only the focal column's setpoint; needs ``focal``; R-33)."""
    if setpoints not in ("all", "focal"):
        raise ValueError(f"setpoints must be 'all' or 'focal', got {setpoints!r}")
    fi = None if focal is None else (data.action_names.index(focal) if isinstance(focal, str) else int(focal))
    if setpoints == "focal" and fi is None:
        raise ValueError("setpoints='focal' needs a focal action")
    n = data.n
    order = info_order(data)
    cols: list = []
    names: list[str] = []
    if include_kpi_lag and data.X_kpi_lag is not None:
        c, nm = _finite_block(np.asarray(data.X_kpi_lag)[order], [f"lag_kpi:{k}" for k in data.kpi_names])
        cols += c
        names += nm
    if include_context and data.context is not None:
        ctx = np.asarray(data.context)
        c_dim = 1 if ctx.ndim == 1 else ctx.shape[1]
        given = tuple((data.meta or {}).get("context_names") or ())
        cn = [f"ctx:{x}" for x in (given if len(given) == c_dim else [f"c{j}" for j in range(c_dim)])]
        c, nm = _finite_block(ctx[order], cn)
        cols += c
        names += nm
    if include_setpoints:
        for j, (a, d) in enumerate(zip(data.action_names, data.designs, strict=True)):
            if setpoints == "focal" and j != fi:
                continue
            if d.kind == "dither" and d.fixed_part is not None:
                c, nm = _finite_block(np.asarray(d.fixed_part, float)[order], [f"sp:{a}"])
                cols += c
                names += nm
    mask = np.ones(n, bool)
    if include_lagged_actions and lags > 0:
        A = np.asarray(data.X_action, float)[order]
        t = None if data.time_index is None else np.asarray(data.time_index)[order]
        for lag in range(1, min(lags, n - 1) + 1):
            ok = np.zeros(n, bool)
            ok[lag:] = True if t is None else (t[lag:] - t[:-lag]) == lag   # a true previous step only (no gap)
            L = np.zeros_like(A)
            L[lag:] = A[:-lag]
            L[~ok] = 0.0
            cols += [L[:, j] for j in range(A.shape[1])]
            names += [f"{a}@t-{lag}" for a in data.action_names]
            if t is not None and not ok[lag:].all():
                cols.append((~ok).astype(float))
                names.append(f"gap@t-{lag}")
            mask &= ok
    if fi is not None:
        conc = concurrent_indices(data, fi, concurrent)
        A = np.asarray(data.X_action, float)[order]
        cols += [A[:, j] for j in conc]
        names += [f"concurrent:{data.action_names[j]}" for j in conc]
    M_info = np.column_stack(cols) if cols else np.zeros((n, 0))
    inv = np.empty(n, np.int64)
    inv[order] = np.arange(n)
    return tuple(names), M_info[inv], mask[inv]


def eq_min_covariates(data: api.Dataset, focal: int | str) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Ruling R-33 eq_min set of an action ``focal``: the R-3 set (lagged KPIs, context, every other action at t)
    + ``sp:<focal>`` (its own setpoint, when it has a dither design). Same return format as ``design_covariates``."""
    return design_covariates(data, True, False, focal=focal, concurrent="all", setpoints="focal")


__all__ = ["DESIGNED_KINDS", "concurrent_indices", "design_covariates", "eq_min_covariates", "info_order"]
