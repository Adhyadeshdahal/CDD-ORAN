"""Shared plumbing of the xm-classic adapters (pc, notears, shap_dag, two_tower, corr, granger).

Contract: scratchpad/xmethod/CONTRACT.md (sections 4-6) and the orchestrator rulings recorded in
scratchpad/xmethod/status/xm-classic.md:
  - every method reports a sign; unsigned methods use the sign of the partial correlation of (source, target) given
    all other action columns + lagged KPIs (+ context in R3), ``notes['sign_rule'] = 'pcorr_given_Z'``;
  - families (R-6): PRIMARY = action -> KPI candidates (incl. ``P_placebo``), SECONDARY = lagged KPI -> KPI; each family
    is its own pooled BY family (q = .05) for p-value methods. When the harness lists them in ``Dataset.meta``
    (``primary_candidates`` / ``secondary_candidates`` / ``diagnostic_candidates``, R-10: ``P_placebo_conf`` in E4
    R3 / R4) those lists define the families, and the diagnostic candidates form a third BY family of their own (they
    are outside the primary set); otherwise the family is the source's kind;
  - score-only methods: tau = the R-29 conformal cutoff at level .05 over the ``P_placebo`` scores pooled over the
    cell's tune datasets (``score.placebo_tau``: the ceil((M+1)(1-.05))-th smallest of the M scores, the largest if that
    index exceeds M), declare iff score > tau (it replaced R-2's 2nd-largest score). The same tau is applied to both
    families (the placebo is an action, so the KPI -> KPI family has no truth-free negative control of its own;
    recorded in the hand-back).
  - arms (R-17 / R-18 / R-25): ``config['arm']`` = "eq" (PRIMARY, equal information) or "native" (SECONDARY).
    Every conditioning set comes WHOLE from the shared helper ``cdd_oran.xmethod.covariates.design_covariates``
    (``cond_set``): eq = ``design_covariates(data, focal=<action>)`` (lagged KPIs, context, setpoints, actions at
    t-1 / t-2, concurrent designed actions; rows with ``row_mask`` False dropped), native = the R-3 set
    ``design_covariates(data, False, False, focal=<action>, concurrent='all')``. Methods without a conditioning
    interface (corr, notears, shap_dag, two_tower) run native only (``notes['arm_note']``); their sign rule uses the
    native set. R-37: ``P_placebo_conf`` (the E4 R3 / R4 diagnostic) is never in another source's conditioning set
    (``drop_diagnostic``) nor a PC node of the primary graph.
Method RNG stream tag 7803 (CONTRACT sec 6). Ground truth never enters this module.
"""
from __future__ import annotations

import contextlib
import dataclasses
import re
import sys
import time
from typing import Any

import numpy as np

from cdd_oran.xmethod import api
from cdd_oran.xmethod.covariates import design_covariates, eq_min_covariates

PLACEBO = "P_placebo"
PLACEBO_CONF = "P_placebo_conf"    # R-10 diagnostic (E4 R3 / R4); R-37: never a conditioner of another source
RNG_TAG = 7803
Q_BY = 0.05
FAMILIES = ("action", "kpi", "diagnostic")
ARMS = ("eq", "native")
NO_INTERFACE = "no conditioning interface"
EXACT_TOL = 1e-10          # residual / total sum of squares at or below which a linear fit is EXACT
NOT_TESTABLE = "exact fit: deterministic world under Z"
COLLINEAR = "source collinear with Z"
META_FAMILY = (("diagnostic_candidates", "diagnostic"), ("secondary_candidates", "kpi"), ("primary_candidates", "action"))


# ------------------------------------------------------------------------------------------------ columns
@dataclasses.dataclass(frozen=True)
class Columns:
    """Resolved design matrices of one Dataset (NaN-only lag / context columns dropped)."""
    actions: np.ndarray            # [n, p]
    lags: np.ndarray               # [n, k'] lagged KPIs that are not all-NaN
    lag_names: tuple[str, ...]     # kpi names of the kept lag columns
    context: np.ndarray            # [n, c] (c = 0 when absent)
    Y: np.ndarray                  # [n, k]


def columns(data: api.Dataset) -> Columns:
    lag = np.asarray(data.X_kpi_lag, dtype=float)
    if lag.ndim == 2 and lag.shape[1]:
        keep = [j for j in range(lag.shape[1]) if np.isfinite(lag[:, j]).all()]
    else:
        keep = []
    lags = lag[:, keep] if keep else np.zeros((data.n, 0))
    ctx = np.zeros((data.n, 0)) if data.context is None else np.asarray(data.context, dtype=float).reshape(data.n, -1)
    return Columns(np.asarray(data.X_action, dtype=float), lags, tuple(data.kpi_names[j] for j in keep), ctx,
                   np.asarray(data.Y, dtype=float))


@dataclasses.dataclass(frozen=True)
class CondSet:
    """Conditioning set of one candidate source (ruling R-25), in the Dataset's ROW order."""
    names: tuple[str, ...]
    Z: np.ndarray                  # [n, c]
    mask: np.ndarray               # [n] rows usable (False: a lagged action is not a true previous step)


def cond_set(data: api.Dataset, family: str, src: int, arm: str) -> CondSet:
    """The WHOLE conditioning set of source ``src`` (action index, or KPI index for a lagged-KPI source) in ``arm``.

    Action source: eq = ``design_covariates(data, focal=src)`` (concurrent='designed', as pmrt_core); native = R-3
    set ``design_covariates(data, False, False, focal=src, concurrent='all')``.
    Lagged-KPI source (secondary family; the helper has no non-action focal): the arm's base helper set without the
    source's own lag column (``lag_kpi:<K>`` and its ``:missing`` indicator) + every action column at t
    (``concurrent:<P>``, as concurrent='all' -- the R-3 rule for a KPI source).
    Arm "eq_min" (R-33; pcorr_hac): action source = ``eq_min_covariates(data, src)`` (R-3 set + ``sp:<src>``); a
    lagged-KPI source has no focal setpoint, so its eq_min set is its native one.
    R-37: every column carrying ``P_placebo_conf`` (``is_diagnostic_column``) is removed from the set of every other
    source (actions and lagged KPIs); ``P_placebo_conf``'s own set (its diagnostic candidates) is the full helper set."""
    eq = arm == "eq"
    if family == "action":
        if arm == "eq_min":
            names, M, mask = eq_min_covariates(data, src)
        elif eq:
            names, M, mask = design_covariates(data, focal=src)
        else:
            names, M, mask = design_covariates(data, False, False, focal=src, concurrent="all")
        cs = CondSet(tuple(names), np.asarray(M, float), np.asarray(mask, bool))
        return cs if data.action_names[src] == PLACEBO_CONF else drop_diagnostic(cs, data)
    names, M, mask = design_covariates(data, eq, eq)
    k = data.kpi_names[src]
    keep = [i for i, nm in enumerate(names) if nm not in (f"lag_kpi:{k}", f"lag_kpi:{k}:missing")]
    A = np.asarray(data.X_action, float)
    return drop_diagnostic(CondSet(tuple([names[i] for i in keep] + [f"concurrent:{a}" for a in data.action_names]),
                                   np.column_stack([np.asarray(M, float)[:, keep], A]), np.asarray(mask, bool)), data)


def is_diagnostic_column(name: str) -> bool:
    """True for a helper column that carries ``P_placebo_conf`` (``concurrent:``, ``sp:``, ``@t-<L>``, and their
    ``:missing`` indicators; ruling R-37)."""
    base = name[:-len(":missing")] if name.endswith(":missing") else name
    return base in (f"concurrent:{PLACEBO_CONF}", f"sp:{PLACEBO_CONF}") or base.startswith(f"{PLACEBO_CONF}@t-")


def drop_diagnostic(cs: CondSet, data: api.Dataset) -> CondSet:
    """Ruling R-37: the diagnostic ``P_placebo_conf`` (R-10) is never a conditioner of another source."""
    if PLACEBO_CONF not in data.action_names:
        return cs
    keep = [i for i, nm in enumerate(cs.names) if not is_diagnostic_column(nm)]
    return CondSet(tuple(cs.names[i] for i in keep), cs.Z[:, keep], cs.mask)


def source_column(data: api.Dataset, cols: Columns, family: str, j: int) -> tuple[np.ndarray, int]:
    """(source column, helper source index) for ``resolve``'s (family, j): an action index, or the KPI index of a
    lagged-KPI source."""
    if family == "action":
        return cols.actions[:, j], j
    return cols.lags[:, j], data.kpi_names.index(cols.lag_names[j])


_LAG_PATTERNS = (r"^(?P<k>.+?)_lag$", r"^(?P<k>.+?)_prev$", r"^(?P<k>.+?)_t$", r"^(?P<k>.+?)\[t\]$",
                 r"^(?P<k>.+?)\(t\)$", r"^lag_(?P<k>.+)$")


def source_kind(data: api.Dataset, source: str) -> tuple[str, int]:
    """('action', column) or ('kpi', kpi index) for a candidate source name. A lagged-KPI source may be named exactly
    as the KPI or with a common lag decoration (``K0_lag``, ``K0[t]``, ...); anything else raises."""
    if source in data.action_names:
        return "action", data.action_names.index(source)
    if source in data.kpi_names:
        return "kpi", data.kpi_names.index(source)
    for pat in _LAG_PATTERNS:
        m = re.match(pat, source)
        if m and m.group("k") in data.kpi_names:
            return "kpi", data.kpi_names.index(m.group("k"))
    raise KeyError(f"candidate source {source!r} is neither an action nor a (lagged) KPI of {data.world}")


def target_index(data: api.Dataset, target: str) -> int:
    if target in data.kpi_names:
        return data.kpi_names.index(target)
    raise KeyError(f"candidate target {target!r} is not a KPI of {data.world}")


def resolve(data: api.Dataset, cols: Columns, source: str) -> tuple[str, int | None]:
    """(family, column index into cols.actions / cols.lags); index None = the lag column is absent (all-NaN)."""
    kind, i = source_kind(data, source)
    if kind == "action":
        return "action", i
    name = data.kpi_names[i]
    return "kpi", (cols.lag_names.index(name) if name in cols.lag_names else None)


# ------------------------------------------------------------------------------------------------ stats helpers
def zscore(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    sd = a.std(0)
    return (a - a.mean(0)) / np.where(sd > 0, sd, 1.0)


def _resid(v: np.ndarray, Z: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(v)), Z]) if Z.shape[1] else np.ones((len(v), 1))
    beta, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ beta


def pcorr(x: np.ndarray, y: np.ndarray, Z: np.ndarray) -> float:
    """Partial correlation of x and y given Z (OLS residualisation with intercept); NaN if degenerate."""
    rx, ry = _resid(np.asarray(x, float), Z), _resid(np.asarray(y, float), Z)
    sx, sy = rx.std(), ry.std()
    if sx <= 1e-12 * (1 + np.abs(x).max()) or sy <= 1e-12 * (1 + np.abs(y).max()):
        return float("nan")
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))


def sign_pcorr_given_Z(data: api.Dataset, cols: Columns, family: str, j: int, t: int, arm: str = "native") -> int:
    """Orchestrator sign rule (R-4): sign of pcorr(source, target | the arm's conditioning set ``cond_set``; native =
    the R-3 set: all other actions + lagged KPIs + context)."""
    x, src = source_column(data, cols, family, j)
    cs = cond_set(data, family, src, arm)
    m = cs.mask
    r = pcorr(x[m], cols.Y[m, t], cs.Z[m])
    return int(np.sign(r)) if np.isfinite(r) else 0


def exact_fit(y: np.ndarray, Z: np.ndarray) -> bool:
    """True if intercept + Z explains y exactly (residual share <= EXACT_TOL, or y constant): a conditional test of
    any further source given Z then has no information (deterministic worlds E1 / E3 under Z_eq; orchestrator Q-F1:
    such candidates are NOT TESTABLE -- score NaN, p None, never declared -- not p = 1)."""
    y = np.asarray(y, float)
    tss = float(((y - y.mean()) ** 2).sum())
    if tss <= 0:
        return True
    A = np.column_stack([np.ones(len(y)), Z]) if Z.shape[1] else np.ones((len(y), 1))
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ beta
    return float(r @ r) <= EXACT_TOL * tss


def not_testable_notes(edges: list[str], collinear: list[str] | None = None) -> dict[str, Any]:
    """Notes for candidates skipped by ``exact_fit`` (``edges``) and for sources collinear with their conditioning set
    (``collinear``; granger's "degenerate" status); empty dict if none. The edges go ONLY in ``not_testable_edges``
    (the ``score.py`` format) and the reason string in ``not_testable_reason`` (``score.py`` reads
    ``notes['not_testable']`` as an edge listing, so a str there is an error; Q1 of fix-classic2)."""
    collinear = list(collinear or [])
    allx = list(edges) + collinear
    if not allx:
        return {}
    why = "; ".join(([NOT_TESTABLE] if edges else []) + ([COLLINEAR] if collinear else []))
    out = {"not_testable_reason": why, "n_not_testable": len(allx), "not_testable_edges": allx}
    if collinear:
        out["not_testable_collinear"] = collinear
    return out


def by_declare(pvals, q: float = Q_BY) -> np.ndarray:
    """Benjamini-Yekutieli step-up (same rule as ``cdd_oran.discovery.mscr.by_declare``; NaN p never declared)."""
    p = np.asarray(pvals, dtype=float)
    out = np.zeros(len(p), dtype=bool)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    from cdd_oran.discovery.mscr import by_declare as _by
    out[np.nonzero(ok)[0]] = _by(p[ok], q)
    return out


def rng(data: api.Dataset, method_idx: int) -> np.random.Generator:
    return np.random.default_rng([RNG_TAG, int(data.seed), int(method_idx)])


def int_seed(data: api.Dataset, method_idx: int) -> int:
    return int(rng(data, method_idx).integers(0, 2**31 - 1))


# ------------------------------------------------------------------------------------------------ results
@dataclasses.dataclass
class Scored:
    """Per-candidate raw output of a method (before declaration)."""
    source: str
    target: str
    family: str                 # "action" | "kpi"
    score: float
    sign: int
    p: float | None = None
    native: bool | None = None  # the method's own declaration (e.g. PC at alpha, NOTEARS at w_threshold)


def assign_families(data: api.Dataset, scored: list[Scored]) -> list[Scored]:
    """BY families from ``Dataset.meta`` candidate lists where present (module doc), else the source's kind."""
    lists = [(set(map(tuple, data.meta[k])), f) for k, f in META_FAMILY if k in (data.meta or {})]
    if not lists:
        return scored
    out = []
    for s in scored:
        fam = next((f for cands, f in lists if (s.source, s.target) in cands), s.family)
        out.append(dataclasses.replace(s, family=fam))
    return out


def declare(scored: list[Scored], config: dict[str, Any], uses_p: bool) -> tuple[list[bool], dict[str, Any]]:
    """Primary declarations: BY per family (p-value methods) or score > tau (score-only methods; tau from
    ``config['tau']``; no tau = untuned = nothing declared). Returns (declared, notes)."""
    notes: dict[str, Any] = {}
    dec = [False] * len(scored)
    tau = config.get("tau")
    tau_dec = [bool(tau is not None and np.isfinite(s.score) and s.score > tau) for s in scored]
    if uses_p:
        for fam in FAMILIES:
            idx = [i for i, s in enumerate(scored) if s.family == fam]
            if not idx:
                continue
            d = by_declare([np.nan if scored[i].p is None else scored[i].p for i in idx], config.get("q", Q_BY))
            for i, v in zip(idx, d, strict=True):
                dec[i] = bool(v)
        notes["declare_rule"] = f"pooled BY per family (action | kpi), q = {config.get('q', Q_BY)}"
        notes["tau_declared_secondary"] = [f"{s.source}->{s.target}" for s, v in zip(scored, tau_dec, strict=True)
                                           if v] if tau is not None else None
    else:
        dec = tau_dec
        notes["declare_rule"] = ("score > tau (tau = R-29 conformal .05 cutoff of the tune placebo scores)"
                                 if tau is not None
                                 else "untuned: no tau in config, nothing declared")
    return dec, notes


def make_result(method: Any, scored: list[Scored], config: dict[str, Any], cpu_s: float, uses_p: bool,
                notes: dict[str, Any] | None = None) -> api.Result:
    dec, dnotes = declare(scored, config, uses_p)
    edges = tuple(api.EdgeResult(source=s.source, target=s.target, score=float(s.score), p=s.p, sign=int(s.sign),
                                 declared=bool(d)) for s, d in zip(scored, dec, strict=True))
    n = dict(notes or {})
    n.update(dnotes)
    n["family"] = {f"{s.source}->{s.target}": s.family for s in scored}
    if any(s.native is not None for s in scored):
        n["native_declared"] = [f"{s.source}->{s.target}" for s in scored if s.native]
    n["tau"] = config.get("tau")
    return api.Result(method=method.name, version=method.version, edges=edges, cpu_s=float(cpu_s),
                      config=dict(config), notes=n)


def placebo_tau(results: list[api.Result]) -> dict[str, Any]:
    """Ruling R-29: tau = the conformal .05 cutoff of the placebo-source scores pooled over the tune results.
    Delegates to the harness rule ``cdd_oran.xmethod.score.placebo_tau`` (+inf when no finite placebo score)."""
    from cdd_oran.xmethod.score import placebo_tau as _tau
    from cdd_oran.xmethod.score import tau_rule_name
    n = sum(1 for r in results for e in r.edges if e.source == PLACEBO and np.isfinite(e.score))
    return {"tau": float(_tau(results)), "tau_rule": tau_rule_name(), "n_placebo_scores": n}


@contextlib.contextmanager
def _threads(n: int):
    """Pin BLAS / OpenMP (threadpoolctl) and torch to ``n`` threads so cpu_s is comparable across methods."""
    from threadpoolctl import threadpool_limits
    torch = sys.modules.get("torch")
    old = torch.get_num_threads() if torch is not None else None
    try:
        if torch is not None:
            torch.set_num_threads(n)
        with threadpool_limits(n):
            yield
    finally:
        if torch is not None:
            torch.set_num_threads(old)


class ClassicBase:
    """api.Method skeleton: ``_score(data, config) -> (scored, notes)`` is the method; tune / run are shared."""
    name = "base"
    version = "0"
    uses_p = False
    method_idx = 0
    arms: tuple[str, ...] = ("native",)    # first = default; ("eq", "native") for methods with a conditioning set
    defaults: dict[str, Any] = {}

    def default_config(self) -> dict[str, Any]:
        return dict(self.defaults)

    def tune(self, dev: list[api.Dataset], truth_free: bool = True,
             config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Placebo tau on ``dev`` for ``config`` over the defaults (e.g. ``{'arm': 'native'}``: tau of that arm)."""
        if not truth_free:
            raise ValueError("xm-classic tuning is truth-free only (CONTRACT sec 5)")
        base = self.default_config()
        base.update(config or {})
        res = [self.run(d, base) for d in dev]
        cfg = dict(base)
        cfg.update(placebo_tau(res))
        cfg["n_dev"] = len(dev)
        cfg["dev_seeds"] = [int(d.seed) for d in dev]
        return cfg

    def run(self, data: api.Dataset, config: dict[str, Any] | None = None) -> api.Result:
        cfg = self.default_config()
        cfg.update(config or {})
        req = cfg.get("arm", self.arms[0])
        if req not in ARMS and req not in self.arms:
            raise ValueError(f"arm must be one of {sorted(set(ARMS) | set(self.arms))}, got {req!r}")
        cfg["arm"] = req if req in self.arms else "native"
        nthr = int(cfg.get("threads", 1))
        t0 = time.process_time()
        with _threads(nthr):
            scored, notes = self._score(data, cfg)
        scored = assign_families(data, scored)
        notes["threads"] = nthr
        notes["arm"] = cfg["arm"]
        if self.arms == ("native",):
            notes["arm_note"] = NO_INTERFACE
        if req != cfg["arm"]:
            notes["arm_requested"] = req
        return make_result(self, scored, cfg, time.process_time() - t0, self.uses_p, notes)

    def _score(self, data: api.Dataset, config: dict[str, Any]) -> tuple[list[Scored], dict[str, Any]]:
        raise NotImplementedError


# ------------------------------------------------------------------------------------------------ resources
def peak_rss_mb() -> float:
    """Peak resident set of this process in MB (Windows: PeakWorkingSetSize; POSIX: ru_maxrss)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.PeakWorkingSetSize / 2**20
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10
