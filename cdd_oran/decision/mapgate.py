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


# ============================================================================================== MapGate v2
# Option-(a) iteration on DEV (after K-A, 2026-09-30; NOT frozen). v1 read only the pv (fallback v) and e edges,
# counted a family with no energy edge as "costly" (every prot_min lowering was deferred), deferred costly-harmful
# requests forever, and its reject units also rejected the OPPOSITE direction (1036 collateral prot_min raises on DEV).
#
# MapGateV2(M, theta, k_conf). Same map format, same inputs: at unit open it reads unit["ctx"] and the request's knob
# cell unit["c"] (the knob identity, used by the duty bound only). For a request of family f, d = sign(step):
#   total effect    D_k = d * sum_{rel in own, nbr, far} beta(f, rel, k)  (None / undeclared -> 0). The scored KPIs
#                   (pv, e, svr, rlf) are network totals, so a request's sign is its TOTAL effect; far enters the
#                   signs only (there is no far pressure field).
#   KPI priority    P = (pv, v, rlf): pv is the SLA; v (all-UE violation, the svr / non-protected eMBB proxy) and rlf
#                   are the guardrail proxies. k* = first k in P with D_k != 0.
#   harm            D_{k*} > 0. No harm (k* is a benefit, or the map is silent on P) -> ACCEPT: a request the map says
#                   restores capacity / protects users on the highest-priority KPI it speaks about is never deferred.
#   energy class    saving D_e < 0 | costly D_e > 0 | neutral D_e = 0 (no declared energy edge -> NOT costly).
#   harmful rels    H = {r in (own, nbr): d * beta(f, r, k*) > 0}; pressure = max over H of own_prot_below_frac /
#                   nbr_max_prot_below_frac (NaN -> 0; H empty, i.e. far-only harm -> 0).
#   defer           harm and (costly or pressure > theta): a costly-harmful request (spends energy AND hurts k*) is
#                   deferred; a saving or neutral harmful request only where the map says it hurts AND pressure is high.
#   guard conflict  g* = first k in (v, rlf) with D_k != 0; conflict iff harm and D_{g*} < 0: the map says ACCEPTING
#                   helps the top guardrail it speaks about, so deferring trades that guardrail for k*.
#   duty bound      a conflicted deferral on knob (c, f) is released (accepted) when the knob's run of consecutive
#                   deferred units is already >= k_conf (K_CONF = 1: a conflicted "defer" delays by at most one unit T,
#                   then yields one unit); an accepted unit on the knob resets the run. Unconflicted deferrals: no bound.
#   direction       mapgate_v2_arbiter = DirectionalUnitArbiter: a reject unit defers its OPENING direction only;
#                   opposite-direction requests of the same (cell, xApp) inside the unit are accepted (no collateral).
# Constants (P, (v, rlf), rels, theta = .05, k_conf = 1, T = 60 s) are fixed before any v2 run; identical for every map.
RELS_V2 = ("own", "nbr", "far")
PRIORITY_V2 = ("pv", "v", "rlf")
GUARDS_V2 = ("v", "rlf")
K_CONF = 1


def _sgn(x: float) -> int:
    return (x > 0) - (x < 0)


class MapGateV2:
    """UnitArbiter policy ``policy(unit) -> (mode, propensity)``; see the v2 block comment. ``self.n`` counts unit
    decisions per family ("defer_<f>", "accept_<f>", "release_<f>" = a conflicted deferral released by the bound)."""

    def __init__(self, M: dict, theta: float = THETA, k_conf: int = K_CONF):
        self.M, self.theta, self.k_conf = dict(M), float(theta), int(k_conf)
        self.n, self.run = {}, {}

    def _beta(self, f, rel, kpi) -> float:
        b = self.M.get((f, rel, kpi))
        return 0.0 if b is None else float(b)

    def classify(self, f, d: int) -> dict:
        """The map-only part of the rule for family f and direction d (no ctx)."""
        D = {k: d * sum(self._beta(f, r, k) for r in RELS_V2) for k in PRIORITY_V2 + ("e",)}
        kstar = next((k for k in PRIORITY_V2 if _sgn(D[k]) != 0), None)
        harm = bool(d != 0 and kstar is not None and D[kstar] > 0)
        energy = {-1: "saving", 0: "neutral", 1: "costly"}[_sgn(D["e"])]
        harmful = [r for r in RELS if harm and d * self._beta(f, r, kstar) > 0]
        g = next((k for k in GUARDS_V2 if _sgn(D[k]) != 0), None)
        conflict = bool(harm and g is not None and D[g] < 0)
        return {"d": d, "D": D, "kpi": kstar, "harm": harm, "energy": energy, "harmful": harmful, "guard": g,
                "conflict": conflict}

    def decide(self, ctx) -> tuple[bool, dict]:
        """(defer BEFORE the duty bound, why) from a unit context (knob, step and the two pressure fields)."""
        why = self.classify(ctx["knob"], _sgn(float(ctx["step"])))
        pressure = max((_nz(ctx.get(PRESSURE[r], float("nan"))) for r in why["harmful"]), default=0.0)
        why["pressure"] = pressure
        return bool(why["harm"] and (why["energy"] == "costly" or pressure > self.theta)), why

    def __call__(self, unit):
        ctx = unit["ctx"]
        f = str(ctx["knob"])
        key = (int(unit["c"]), f)
        defer, why = self.decide(ctx)
        tag = "defer_" if defer else "accept_"
        if defer and why["conflict"] and self.run.get(key, 0) >= self.k_conf:
            defer, tag = False, "release_"
        self.run[key] = self.run.get(key, 0) + 1 if defer else 0
        self.n[tag + f] = self.n.get(tag + f, 0) + 1
        return ("reject" if defer else "accept"), 1.0


class DirectionalUnitArbiter(UnitArbiter):
    """UnitArbiter whose reject units defer only the direction of their opening request: a request of the unit's
    (cell, xApp) with another direction is accepted. ``self.passed`` counts those requests."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.passed = 0

    def __call__(self, obs):
        out = super().__call__(obs)
        dec = list(out["decisions"])
        for i, (r, d) in enumerate(zip(obs["requests"], dec, strict=True)):
            if d != "reject":
                continue
            u = self.active.get((int(r["knob"][1]), r["xapp"]))
            if u is not None and _sgn(float(r["prop"] - r["cur"])) != _sgn(float(u["ctx"]["step"])):
                dec[i] = "accept"
                self.passed += 1
        out["decisions"] = dec
        return out


def mapgate_v2_arbiter(M: dict, theta: float = THETA, k_conf: int = K_CONF, T: float = T_UNIT,
                       open_rule: str = "feasible", record: bool = False) -> DirectionalUnitArbiter:
    """DirectionalUnitArbiter(MapGateV2(M, theta, k_conf)) active from t = 0, T = 60 s, open_rule "feasible"."""
    assert open_rule in OPEN_RULES
    return DirectionalUnitArbiter(MapGateV2(M, theta, k_conf), T=T, warmup_s=0.0, record=record, open_rule=open_rule)


def decision_table_v2(M: dict, theta: float = THETA) -> dict:
    """{(family, d): "accept" | "defer" | "defer iff <rels> pressure > theta" | "accept (far-only harm)"}; a
    guard-conflicted deferral gets " [bounded]" (duty bound k_conf)."""
    g = MapGateV2(M, theta)
    out = {}
    for f in FAMILIES:
        for d in (1, -1):
            c = g.classify(f, d)
            if not c["harm"]:
                s = "accept"
            elif c["energy"] == "costly":
                s = "defer"
            elif c["harmful"]:
                s = "defer iff " + "/".join(c["harmful"]) + " pressure > theta"
            else:
                s = "accept (far-only harm)"
            if s.startswith("defer") and c["conflict"]:
                s += " [bounded]"
            out[(f, d)] = s
    return out


def random_sized_map(M_ref: dict, cells, tag: int = 6623, key: int = 2) -> dict:
    """Random map with |M_ref| edges: default_rng([tag, key]) draws |M_ref| distinct (family, relation, kpi) keys from
    ``cells`` (a gt_p "cells" list = the map universe) and a random sign each; |beta| = |mean| of that cell. key 2 =
    K-A2 (default, unchanged); the option-(a) study uses key 3."""
    import numpy as np

    rng = np.random.default_rng([int(tag), int(key)])
    keys = [(c["family"], c["relation"], c["kpi"]) for c in cells]
    mag = {(c["family"], c["relation"], c["kpi"]): abs(float(c["mean"])) for c in cells}
    idx = sorted(int(j) for j in rng.choice(len(keys), size=len(M_ref), replace=False))
    sg = rng.choice([-1.0, 1.0], size=len(M_ref))
    return {keys[i]: float(s * mag[keys[i]]) for i, s in zip(idx, sg, strict=True)}


# direction of the energy-SAVING request per family (+1 = the knob's +1 step): carrier off, sleep, ptx down
SAVING_DIR = {"carrier": -1, "sleep": 1, "ptx": -1}


def blanket_saving_map() -> dict:
    """No-map control: every energy-saving request (SAVING_DIR) hurts own and nbr pv (|beta| = 1) and saves own e."""
    M = {}
    for f, s in SAVING_DIR.items():
        M[(f, "own", "pv")] = float(s)
        M[(f, "nbr", "pv")] = float(s)
        M[(f, "own", "e")] = float(-s)
    return M


# ---------------------------------------------------------------------------------------------- signatures / io
# (option-(a) study, docs/benchmark/E6P_CONFOUNDED_PROTOCOL.md section 4.3; additive)
def decision_signature(M: dict, theta: float = THETA) -> tuple:
    """The map-dependent inputs of ``MapGateV2.decide`` and of its duty bound, per (family, d) in FAMILIES x (+1, -1):
    the protocol's (harm, energy == "costly", harmful rels, conflict) with the inputs ``decide`` never reads dropped,
    so equal signatures <=> identical policies:
      ("accept",)                       no harm, or harm that is neither costly nor tied to an own / nbr pressure
                                        (far-only harm: pressure 0, never > theta);
      ("defer", conflict)               harm and costly (the harmful rels are not read);
      ("defer_iff", harmful, conflict)  harm, not costly: defer iff the max pressure over ``harmful`` > theta.
    theta is part of the signature (it is a rule constant, identical for every map of the study)."""
    g = MapGateV2(M, theta)
    out = []
    for f in FAMILIES:
        for d in (1, -1):
            c = g.classify(f, d)
            if not c["harm"]:
                s = ("accept",)
            elif c["energy"] == "costly" or float(theta) < 0.0:        # pressure >= 0 > theta: always defers
                s = ("defer", bool(c["conflict"]))
            elif c["harmful"]:
                s = ("defer_iff", tuple(c["harmful"]), bool(c["conflict"]))
            else:
                s = ("accept",)
            out.append((f, d, s))
    return (("theta", float(theta)),) + tuple(out)


def signature_key(sig: tuple) -> str:
    """Stable string form of a ``decision_signature`` (JSON-able; equal keys <=> equal signatures)."""
    import json

    return json.dumps(sig, separators=(",", ":"))


def all_accept_signature(theta: float = THETA) -> tuple:
    """The signature of a map that never defers (e.g. the empty map)."""
    return decision_signature({}, theta)


def map_to_json(M: dict) -> list:
    """[[family, relation, kpi, beta | None], ...] sorted by key (exact float repr through json)."""
    return [[k[0], k[1], k[2], None if v is None else float(v)] for k, v in sorted(M.items())]


def map_from_json(rows) -> dict:
    """Inverse of ``map_to_json``."""
    return {(str(f), str(r), str(k)): (None if b is None else float(b)) for f, r, k, b in rows}


__all__ = ["FAMILIES", "GUARDS_V2", "K_CONF", "PRESSURE", "PRIORITY_V2", "RELS", "RELS_V2", "SAVING_DIR", "THETA",
           "DirectionalUnitArbiter", "MapGate", "MapGateV2", "all_accept_signature", "blanket_saving_map",
           "decision_signature", "decision_table", "decision_table_v2", "map_from_gt", "map_from_json", "map_to_json",
           "mapgate_arbiter", "mapgate_v2_arbiter", "own_only", "random_sized_map", "sign_flip", "signature_key"]
