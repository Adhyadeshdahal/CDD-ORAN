"""MapGate(M, theta): a causal-map-driven WG3 referee (scratchpad/e6_dev/decision/OPTION_A_PLAN.md section 3).

A ``units_p.UnitArbiter`` policy. At unit open it reads ONLY ``unit["ctx"]`` (the obs-only unit context of
``UnitArbiter.context``: the opening request's knob family / cur / prop / step and the mediator aggregates) and returns
("reject", 1.0) = DEFER or ("accept", 1.0); the mode holds for the unit's T (UnitArbiter semantics).

Map M: {(family, rel, kpi): beta | None}; family in {carrier, sleep, ptx, prot_min} (= ctx["knob"]), rel in
{own, nbr, far}, kpi in {pv, v, e, ...}. beta = the effect of ACCEPTING a +1 knob-direction step (the GT "dir"
orientation, gt_p: accept - reject per knob-value increase; sleep 1 = asleep). A key with None is DECLARED with no
effect (contributes 0); a missing key is undeclared. Only rel in ``RELS`` = (own, nbr) is read; far is ignored.

Rule for a request of family f with d = sign(step) (step = prop - cur; d = 0 -> accept):
  harm KPI  k_h = pv if any (f, rel, pv) is declared for rel in RELS, else v (fallback);
  contrib_r = d * beta(f, r, k_h) (0 if None / undeclared); harm = sum_r contrib_r > 0;
  costly    = d * sum_r beta(f, r, e) >= 0  (no declared energy edge -> 0 >= 0 -> costly: accepting saves nothing);
  pressure  = max over the HARMFUL relations (contrib_r > 0) of their protected-UE pressure:
              own -> ctx["own_prot_below_frac"], nbr -> ctx["nbr_max_prot_below_frac"] (NaN = no report yet -> 0);
  DEFER iff harm and (costly or pressure > theta); otherwise accept.
"""
from __future__ import annotations

import math

from .units_p import OPEN_RULES, T_UNIT, UnitArbiter

RELS = ("own", "nbr")
PRESSURE = {"own": "own_prot_below_frac", "nbr": "nbr_max_prot_below_frac"}
THETA = 0.05
FAMILIES = ("carrier", "sleep", "ptx", "prot_min")


def _nz(x) -> float:
    x = float(x)
    return 0.0 if math.isnan(x) else x


class MapGate:
    """UnitArbiter policy ``policy(unit) -> (mode, propensity)``; see the module docstring. ``self.n`` counts unit
    decisions per family ("defer_<f>" / "accept_<f>")."""

    def __init__(self, M: dict, theta: float = THETA):
        self.M, self.theta = dict(M), float(theta)
        self.n = {}

    def _beta(self, f, rel, kpi) -> float:
        b = self.M.get((f, rel, kpi))
        return 0.0 if b is None else float(b)

    def harm_kpi(self, f) -> str:
        return "pv" if any((f, r, "pv") in self.M for r in RELS) else "v"

    def decide(self, ctx) -> tuple[bool, dict]:
        """(defer, why) from a unit context (dict with knob, step and the two pressure fields)."""
        f = ctx["knob"]
        step = float(ctx["step"])
        d = (step > 0) - (step < 0)
        kh = self.harm_kpi(f)
        contrib = {r: d * self._beta(f, r, kh) for r in RELS}
        harm = sum(contrib.values()) > 0
        costly = d * sum(self._beta(f, r, "e") for r in RELS) >= 0
        harmful = [r for r in RELS if contrib[r] > 0]
        pressure = max((_nz(ctx.get(PRESSURE[r], float("nan"))) for r in harmful), default=0.0)
        defer = bool(d != 0 and harm and (costly or pressure > self.theta))
        return defer, {"d": d, "kpi": kh, "harm": harm, "costly": costly, "harmful": harmful, "pressure": pressure}

    def __call__(self, unit):
        ctx = unit["ctx"]
        defer, _ = self.decide(ctx)
        key = ("defer_" if defer else "accept_") + str(ctx["knob"])
        self.n[key] = self.n.get(key, 0) + 1
        return ("reject" if defer else "accept"), 1.0


def mapgate_arbiter(M: dict, theta: float = THETA, T: float = T_UNIT, open_rule: str = "feasible",
                    record: bool = False) -> UnitArbiter:
    """UnitArbiter(MapGate(M, theta)) active from t = 0 (warmup_s = 0), T = 60 s, open_rule "feasible"."""
    assert open_rule in OPEN_RULES
    return UnitArbiter(MapGate(M, theta), T=T, warmup_s=0.0, record=record, open_rule=open_rule)


# ---------------------------------------------------------------------------------------------- map builders
def map_from_gt(cells, true_only: bool = False) -> dict:
    """Map from a gt_p table's "cells": TRUE -> its mean, NULL -> None (declared no effect), INDET -> undeclared.
    ``true_only``: NULL is dropped as well."""
    M = {}
    for c in cells:
        key = (c["family"], c["relation"], c["kpi"])
        if c["status"] == "TRUE":
            M[key] = float(c["mean"])
        elif c["status"] == "NULL" and not true_only:
            M[key] = None
    return M


def own_only(M: dict) -> dict:
    """Drop every nbr / far edge."""
    return {k: v for k, v in M.items() if k[1] == "own"}


def sign_flip(M: dict, families=("carrier",)) -> dict:
    """Negate every edge of ``families`` (None stays None)."""
    fam = set(families)
    return {k: (-v if (k[0] in fam and v is not None) else v) for k, v in M.items()}


def decision_table(M: dict, theta: float = THETA) -> dict:
    """{direction: "defer" | "accept" | "defer iff <rel> pressure > theta"} per family and d in (+1, -1)."""
    g = MapGate(M, theta)
    out = {}
    for f in FAMILIES:
        for d in (1, -1):
            lo, _ = g.decide({"knob": f, "step": float(d), PRESSURE["own"]: 0.0, PRESSURE["nbr"]: 0.0})
            hi, why = g.decide({"knob": f, "step": float(d), PRESSURE["own"]: 1.0, PRESSURE["nbr"]: 1.0})
            out[(f, d)] = "defer" if lo else ("defer iff " + "/".join(why["harmful"]) + " pressure > theta"
                                              if hi else "accept")
    return out


__all__ = ["FAMILIES", "PRESSURE", "RELS", "THETA", "MapGate", "decision_table", "map_from_gt", "mapgate_arbiter",
           "own_only", "sign_flip"]
