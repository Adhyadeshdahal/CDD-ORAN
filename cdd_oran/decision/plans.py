"""WG3 plans: per region, a mode per xApp + a rollback flag; plan -> WG3 decision dict for one second's obs.

RegionPlan = {"mode": {xApp: accept|reject|half|lock}, "rb": 0|1}; Plan = {region: RegionPlan}.
  accept / reject / lock = reject + lock that knob for D s /
  half = HALVE THE SLEW RATE (``half_rule="slew"``, default; ``half_step``): a requested step spanning >= 2 actuator
       quanta -> ("modify", cur + ceil(n/2) quanta toward the proposal) (a real move of about half the step); a
       single-quantum (or smaller) step -> accept on alternate such requests per knob, reject otherwise
       (deterministic toggle per knob; the first such request is accepted iff the sum of the knob's cell indices
       is even, so knobs start in both phases; state = the ``half_state`` dict owned by the arbiter/collector
       and threaded through ``decide`` like ``rb_at``; rollouts copy it). Realised fraction ~ 0.5 either way.
       ``half_rule="legacy"`` = the pre-2026-09-28 rule ("modify", cur + (prop - cur) / 2), which the actuator's
       quantisation mostly rounds back to cur for one-quantum xApp steps (kept only to reproduce old v1 data);
  rb = at plan start, roll back every knob of the region changed in the last RB_WINDOW s (to last-known-good),
       except knobs whose latest change is our own rollback (never roll back a rollback).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from cdd_oran.envs.e6.config import TTT_SET_MS

from .adapters.e6 import knob_region

XAPPS = ("MRO", "TS", "ES", "SLICE")
MODES = ("accept", "reject", "half", "lock")
RB_WINDOW = 60.0
HALF_RULES = ("slew", "legacy")
QUANTUM = {"cio": 1.0, "hys": 0.5, "ll_ratio": 0.05, "sleep": 1.0, "carrier": 1.0}   # = envs.e6.ric._quantise grids

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


def half_step(knob, cur: float, prop: float, half_state: dict) -> str | tuple:
    """"half" (halve the slew rate) decision for one request; updates ``half_state`` for single-quantum steps."""
    typ = knob[0]
    if typ == "ttt":
        i0 = int(np.argmin([abs(x - cur) for x in TTT_SET_MS]))
        i1 = int(np.argmin([abs(x - prop) for x in TTT_SET_MS]))
        n = abs(i1 - i0)
        if n >= 2:
            return ("modify", float(TTT_SET_MS[i0 + int(np.sign(i1 - i0)) * int(np.ceil(n / 2))]))
    else:
        q = QUANTUM.get(typ)
        n = abs(prop - cur) / q if q else 0.0
        if n >= 2 - 1e-6:
            k = int(np.ceil(round(n / 2, 6)))
            return ("modify", float(cur + np.sign(prop - cur) * k * q))
    c = half_state.get(knob, 0)
    half_state[knob] = c + 1
    phase = sum(int(x) for x in knob[1:]) % 2          # fixed per-knob start phase: no accept-first bias overall
    return "accept" if (c + phase) % 2 == 0 else "reject"


def decide(plan: Plan, obs: Mapping, site, D: float, last_change: Mapping, rb_at: Mapping, first: bool,
           half_state: dict | None = None, half_rule: str = "slew") -> dict:
    """WG3 decision dict for ``obs["requests"]``; rollbacks only on the plan's first second (``first``).
    ``half_state`` (knob -> single-quantum "half" requests seen) is updated in place; None = a fresh dict (every
    call then accepts the first single-quantum "half" request per knob: pass the owner's dict)."""
    if half_rule not in HALF_RULES:
        raise ValueError(f"half_rule in {HALF_RULES}")
    hs = {} if half_state is None else half_state
    dec = []
    for r in obs["requests"]:
        m = plan[knob_region(site, r["knob"])]["mode"][r["xapp"]]
        if m == "half":
            dec.append(half_step(r["knob"], r["cur"], r["prop"], hs) if half_rule == "slew" else
                       ("modify", r["cur"] + 0.5 * (r["prop"] - r["cur"])))
        else:
            dec.append("accept" if m == "accept" else "reject" if m == "reject" else ("lock", float(D)))
    rb = []
    if first:
        now = obs["t"]
        rb = [k for k, t in last_change.items()
              if now - t <= RB_WINDOW and plan[knob_region(site, k)]["rb"] and rb_at.get(k) != t]
    return {"decisions": dec, "writes": [], "rollback": rb}
