"""WG3 plans: per region, a mode per xApp + a rollback flag; plan -> WG3 decision dict for one second's obs.

RegionPlan = {"mode": {xApp: accept|reject|half|lock}, "rb": 0|1}; Plan = {region: RegionPlan}.
  accept / reject / half = modify to cur + (prop - cur)/2 / lock = reject + lock that knob for D s;
  rb = at plan start, roll back every knob of the region changed in the last RB_WINDOW s (to last-known-good),
       except knobs whose latest change is our own rollback (never roll back a rollback).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from .adapters.e6 import knob_region

XAPPS = ("MRO", "TS", "ES", "SLICE")
MODES = ("accept", "reject", "half", "lock")
RB_WINDOW = 60.0

RegionPlan = dict   # {"mode": {xapp: mode}, "rb": int}
Plan = dict         # {region: RegionPlan}


def uniform(mode: str, rb: int = 0) -> RegionPlan:
    return {"mode": {x: mode for x in XAPPS}, "rb": rb}


def network(regions: Iterable[int], rp: RegionPlan) -> Plan:
    """Same region plan everywhere (the object is shared, as in the scratch oracle; plans are never mutated)."""
    return {int(g): rp for g in regions}


def accept_all(regions: Iterable[int]) -> Plan:
    return network(regions, uniform("accept"))


def random_region(rng: np.random.Generator) -> RegionPlan:
    """Uniform over MODES per xApp (in XAPPS order), then rb ~ U{0,1} (draw order is part of the RNG contract)."""
    return {"mode": {x: MODES[int(rng.integers(len(MODES)))] for x in XAPPS}, "rb": int(rng.integers(2))}


def is_accept_all(plan: Plan) -> bool:
    return all(not rp["rb"] and all(m == "accept" for m in rp["mode"].values()) for rp in plan.values())


def restrict(plan: Plan, allowed: set, regions: Iterable[int] | None = None) -> Plan:
    """Force "accept" for every (xapp, region) not in ``allowed``; rb survives only in regions with an allowed pair."""
    out = {}
    for g in regions if regions is not None else plan:
        rp = plan[g]
        mode = {x: (m if (x, g) in allowed else "accept") for x, m in rp["mode"].items()}
        rb = rp["rb"] if any(a[1] == g for a in allowed) else 0
        out[g] = rp if mode == rp["mode"] and rb == rp["rb"] else {"mode": mode, "rb": rb}
    return out


def decide(plan: Plan, obs: Mapping, site, D: float, last_change: Mapping, rb_at: Mapping, first: bool) -> dict:
    """WG3 decision dict for ``obs["requests"]``; rollbacks only on the plan's first second (``first``)."""
    dec = []
    for r in obs["requests"]:
        m = plan[knob_region(site, r["knob"])]["mode"][r["xapp"]]
        dec.append("accept" if m == "accept" else "reject" if m == "reject" else
                   ("lock", float(D)) if m == "lock" else ("modify", r["cur"] + 0.5 * (r["prop"] - r["cur"])))
    rb = []
    if first:
        now = obs["t"]
        rb = [k for k, t in last_change.items()
              if now - t <= RB_WINDOW and plan[knob_region(site, k)]["rb"] and rb_at.get(k) != t]
    return {"decisions": dec, "writes": [], "rollback": rb}
