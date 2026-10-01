"""Certified-safe MapGateV2 (follow-up study 2b; protocol docs/benchmark/E6P_CERTSAFE_PROTOCOL.md).

MapGateV2 (mapgate.py, unchanged) defers a request of family f and direction d when the map says accepting it harms
the highest-priority KPI it speaks about. Option (a) showed the failure mode of a map that is silent on a guard KPI:
the PMRT map has no ptx -> nbr rlf edge (GT -.061), so MapGateV2 deferred every ptx-up request without the duty
bound that the GT map's declared edge triggers, and the RLF ratio vs accept-all was 1.31.

The guard ratios are pooled KPI / pooled accept-all (noarb) KPI. Accepting is the guard reference, so the referee's
only deviation from the reference is a DEFERRAL. Deferring a request of (f, d) changes guard KPI k on relation rel by
-d * beta(f, rel, k) relative to accepting it ("dir" orientation: beta = effect of accepting a +1 knob step).

Certified-safe rule (identical for every map arm; constants fixed by the protocol):
  for every (f, d) that MapGateV2(M) can defer (decision signature not "accept"), and every guard edge
  (f, rel, k), rel in RELS_V2 = (own, nbr, far), k in GUARD_KPIS = (v, rlf) (MapGateV2's guard proxies):
    DECLARED      (f, rel, k) is a key of M: MapGateV2's own guard-conflict / duty-bound logic applies, unchanged;
    CERTIFIED     not declared, the arm's method gives a bound, and UB90(-d * beta) <= tau(f, d, k);
    UNRESOLVED    otherwise (no bound, or the bound is above tau).
  (f, d) is UNCERTIFIED iff any of its guard edges is UNRESOLVED; an uncertified (f, d) is never deferred (the
  referee abstains: the request is accepted, as the guard reference would). Everything else is MapGateV2.

Bounds (``bound_ub``): {"kind": "slope", "beta", "se"} -> UB90 = h * beta + Z90 * se (one-sided 90 %, Wald /
test inversion of the slope's statistic), {"kind": "ci", "lo", "hi"} -> UB = max(h * lo, h * hi), with h = -d.
tau (``tau_table``): tau(f, d, k) = (RHO - 1) * G_k / (max(N(f, d), 1) * len(RELS_V2)), RHO = 1.05, G_k = mean
per-episode scored guard total under accept-all (rlf: "rlf", v: "viol_ue_s"), N(f, d) = mean units per episode of
(f, d) under accept-all. With every deferred unit's harm on (f, k) at most len(RELS_V2) * tau, deferring all N(f, d)
units moves the episode guard ratio by at most RHO - 1 = .05 (half of the 1.10 guard's .10 margin).
"""
from __future__ import annotations

import math

import numpy as np

from .mapgate import (
    FAMILIES,
    K_CONF,
    RELS_V2,
    THETA,
    DirectionalUnitArbiter,
    MapGateV2,
    _sgn,
    decision_signature,
)
from .units_p import OPEN_RULES, T_UNIT

Z90 = 1.2815515655446004              # standard normal 0.90 quantile (one-sided 90 % bound)
RHO = 1.05                            # target episode guard ratio for one request class (the 1.10 guard, half margin)
GUARD_KPIS = ("v", "rlf")             # = mapgate.GUARDS_V2: the guard proxies of the map language
GUARD_FIELD = {"v": "viol_ue_s", "rlf": "rlf"}   # episode outcome field of each guard KPI (e6p_screen.outcome)
DIRS = (1, -1)
DECLARED, CERTIFIED, UNRESOLVED = "DECLARED", "CERTIFIED", "UNRESOLVED"


# ---------------------------------------------------------------------------------------------- tau / bounds
def tau_table(G: dict, N: dict, rho: float = RHO, n_rel: int = len(RELS_V2)) -> dict:
    """{(f, d, k): tau} for f in FAMILIES, d in (+1, -1), k in GUARD_KPIS. ``G`` {k: mean per-episode guard total
    under accept-all}; ``N`` {(f, d): mean units per episode under accept-all} (missing -> 0, floored at 1)."""
    out = {}
    for f in FAMILIES:
        for d in DIRS:
            n = max(float(N.get((f, d), 0.0)), 1.0)
            for k in GUARD_KPIS:
                out[(f, d, k)] = (float(rho) - 1.0) * float(G[k]) / (n * int(n_rel))
    return out


def bound_ub(b: dict | None, h: int, z: float = Z90) -> float:
    """One-sided upper bound of h * beta from a bound record (module docstring); +inf if there is none."""
    if not b:
        return math.inf
    if b["kind"] == "slope":
        beta, se = b.get("beta"), b.get("se")
        if beta is None or se is None or not (np.isfinite(beta) and np.isfinite(se)):
            return math.inf
        return float(h * float(beta) + float(z) * float(se))
    if b["kind"] == "ci":
        lo, hi = b.get("lo"), b.get("hi")
        if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
            return math.inf
        return float(max(h * float(lo), h * float(hi)))
    raise ValueError(f"unknown bound kind {b.get('kind')}")


def classify_edge(M: dict, bounds: dict | None, f: str, rel: str, k: str, d: int, tau: float) -> tuple[str, float]:
    """(status, ub) of guard edge (f, rel, k) for a DEFERRAL of direction d (harm direction h = -d)."""
    if (f, rel, k) in M:
        return DECLARED, float("nan")
    ub = bound_ub((bounds or {}).get((f, rel, k)), -int(d))
    return (CERTIFIED if ub <= float(tau) else UNRESOLVED), ub


def can_defer(M: dict, f: str, d: int, theta: float = THETA) -> bool:
    """True iff MapGateV2(M) defers (f, d) in some context (its decision-signature entry is not "accept")."""
    for f_, d_, s in decision_signature(M, theta)[1:]:
        if f_ == f and d_ == d:
            return s[0] != "accept"
    raise KeyError((f, d))


def certify(M: dict, bounds: dict | None, tau: dict, theta: float = THETA) -> dict:
    """{"classes": {"f|d": {"can_defer", "edges": [[rel, k, status, ub, tau]], "certified"}}, "uncertified":
    [[f, d], ...]} for map M with the arm's ``bounds`` ({(f, rel, k): bound record} or None = no bound)."""
    classes, unc = {}, []
    for f in FAMILIES:
        for d in DIRS:
            cd = can_defer(M, f, d, theta)
            edges, ok = [], True
            if cd:
                for rel in RELS_V2:
                    for k in GUARD_KPIS:
                        t = tau[(f, d, k)]
                        st, ub = classify_edge(M, bounds, f, rel, k, d, t)
                        edges.append([rel, k, st, ub if math.isfinite(ub) else None, t])
                        ok &= st != UNRESOLVED
            classes[f"{f}|{d}"] = {"can_defer": cd, "edges": edges, "certified": bool(ok)}
            if cd and not ok:
                unc.append([f, d])
    return {"classes": classes, "uncertified": unc}


# ---------------------------------------------------------------------------------------------- the gate
class CertSafeMapGateV2(MapGateV2):
    """MapGateV2 whose deferrals of the ``uncertified`` (family, d) classes are replaced by accept. With an empty
    set it decides exactly as MapGateV2 (tested). Extra counters (decisions unaffected): "uncert_<f>" = deferrals
    withheld, "dir_<f>_<+1|-1>" = units per opening direction."""

    def __init__(self, M: dict, uncertified=(), theta: float = THETA, k_conf: int = K_CONF):
        super().__init__(M, theta, k_conf)
        self.uncertified = frozenset((str(f), int(d)) for f, d in uncertified)
        self._withheld = False

    def decide(self, ctx) -> tuple[bool, dict]:
        defer, why = super().decide(ctx)
        why["uncertified"] = (str(ctx["knob"]), int(why["d"])) in self.uncertified
        self._withheld = bool(defer and why["uncertified"])
        return (False if self._withheld else defer), why

    def __call__(self, unit):
        out = super().__call__(unit)
        ctx = unit["ctx"]
        f = str(ctx["knob"])
        dk = f"dir_{f}_{_sgn(float(ctx['step'])):+d}"
        self.n[dk] = self.n.get(dk, 0) + 1
        if self._withheld:
            self.n["uncert_" + f] = self.n.get("uncert_" + f, 0) + 1
        return out


def certsafe_arbiter(M: dict, uncertified=(), theta: float = THETA, k_conf: int = K_CONF, T: float = T_UNIT,
                     open_rule: str = "feasible", record: bool = False) -> DirectionalUnitArbiter:
    """DirectionalUnitArbiter(CertSafeMapGateV2(M, uncertified)) active from t = 0, T = 60 s, open_rule "feasible"
    (the mapgate_v2_arbiter settings)."""
    assert open_rule in OPEN_RULES
    return DirectionalUnitArbiter(CertSafeMapGateV2(M, uncertified, theta, k_conf), T=T, warmup_s=0.0, record=record,
                                  open_rule=open_rule)


def certsafe_signature(M: dict, uncertified=(), theta: float = THETA) -> tuple:
    """mapgate.decision_signature with every uncertified (f, d) entry replaced by ("accept",): equal signatures <=>
    identical CertSafeMapGateV2 policies (an accepted unit never reads the conflict flag or the duty bound)."""
    unc = {(str(f), int(d)) for f, d in uncertified}
    sig = decision_signature(M, theta)
    return (sig[0],) + tuple((f, d, ("accept",) if (f, d) in unc else s) for f, d, s in sig[1:])


# ---------------------------------------------------------------------------------------------- estimators
def design_slope_se(ud, f: str, rel: str, kpi: str) -> dict:
    """PMRT design-based slope (the option (a) map beta, e6p_conf_analyze.design_slope) with its martingale (HC0)
    standard error: beta = sum v r / sum v^2, se = sqrt(sum v^2 (r - beta v)^2) / sum v^2, v = the design-centred
    treatment with the unit's own logged row, r = y - its predictable running centre; information order."""
    from .eprocess_units import EProcConfig, predictable_residuals, unit_order
    from .pmrt import v_design
    rows = unit_order(ud, ud.rows_of(f))
    if len(rows) < 2:
        return {"kind": "slope", "beta": 0.0, "se": None, "n": int(len(rows)), "z": None}
    v = v_design(ud.mode[rows], ud.probs[rows], ud.sgn[rows])
    y, pre = ud.y[(rel, kpi)][rows], ud.pre[(rel, kpi)][rows]
    r, _ = predictable_residuals(y, pre if np.all(np.isfinite(pre)) else None, ud.episode[rows], ud.sgn[rows],
                                 EProcConfig())
    vv = float(v @ v)
    if vv <= 0:
        return {"kind": "slope", "beta": 0.0, "se": None, "n": int(len(rows)), "z": None}
    beta = float(v @ r / vv)
    e = r - beta * v
    se = float(np.sqrt(np.sum(v * v * e * e)) / vv)
    return {"kind": "slope", "beta": beta, "se": se, "n": int(len(rows)), "z": beta / se if se > 0 else None}


def naive_slope_se(ud, f: str, rel: str, kpi: str) -> dict:
    """The associational maps' effect estimator (K-B ``naive_slope``: OLS of y on x = level * sgn over the family's
    units) with its ordinary (iid) OLS standard error, i.e. the method's own world view."""
    rows = ud.rows_of(f)
    if len(rows) < 3:
        return {"kind": "slope", "beta": 0.0, "se": None, "n": int(len(rows)), "z": None}
    x, y = ud.x[rows], ud.y[(rel, kpi)][rows]
    xc = x - x.mean()
    den = float(xc @ xc)
    if den <= 0:
        return {"kind": "slope", "beta": 0.0, "se": None, "n": int(len(rows)), "z": None}
    beta = float(xc @ (y - y.mean()) / den)
    e = (y - y.mean()) - beta * xc
    se = float(np.sqrt(float(e @ e) / (len(rows) - 2) / den))
    return {"kind": "slope", "beta": beta, "se": se, "n": int(len(rows)), "z": beta / se if se > 0 else None}


def gt_ci_bounds(cells) -> dict:
    """{(f, rel, k): {"kind": "ci", "lo", "hi"}} for the guard KPIs from a gt_p "cells" list (95 % episode-cluster
    percentile CI; its end is a one-sided 97.5 % bound, more conservative than 90 %)."""
    return {(c["family"], c["relation"], c["kpi"]): {"kind": "ci", "lo": float(c["ci"][0]), "hi": float(c["ci"][1])}
            for c in cells if c["kpi"] in GUARD_KPIS}


def bounds_to_json(b: dict | None):
    return None if b is None else [[f, r, k, v] for (f, r, k), v in sorted(b.items())]


def bounds_from_json(rows) -> dict | None:
    return None if rows is None else {(str(f), str(r), str(k)): dict(v) for f, r, k, v in rows}


def tau_to_json(tau: dict) -> list:
    return [[f, int(d), k, float(v)] for (f, d, k), v in sorted(tau.items())]


def tau_from_json(rows) -> dict:
    return {(str(f), int(d), str(k)): float(v) for f, d, k, v in rows}


__all__ = ["CERTIFIED", "DECLARED", "DIRS", "GUARD_FIELD", "GUARD_KPIS", "RHO", "UNRESOLVED", "Z90",
           "CertSafeMapGateV2", "bound_ub", "bounds_from_json", "bounds_to_json", "can_defer", "certify",
           "certsafe_arbiter", "certsafe_signature", "classify_edge", "design_slope_se", "gt_ci_bounds",
           "naive_slope_se", "tau_from_json", "tau_table", "tau_to_json"]
