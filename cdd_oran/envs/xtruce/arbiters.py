"""Arbiters for XEnv: ``decide(env, requests) -> accepted requests`` (possibly modified or arbiter-originated).

Request-level (our screen's arms): AcceptAll, RejectAll (freeze), Subset, StaticPriority, CellPriorityLock.
Paper benchmarks (Sec. V-A): Direct ("directly turns the most demanding target of each xApp into the corresponding
gNB action without checking its joint feasibility") and Clipping ("further clips the resulting actions from the
Direct scheme and proportionally rescales them to satisfy c1-c4"). Both read the proposals (targets), not the
requests, and ignore the interference and load targets (paper footnote 5). xTRUCE itself: ``xtruce_arbiter``.

Helpers for oracle / branching arbiters: ``modify``, ``half``, ``request_cells``, ``conflict_groups``.
"""
from __future__ import annotations

import numpy as np


def modify(req, value):
    r = dict(req)
    r["prop"] = float(value)
    r["modified"] = True
    return r


def half(req):
    """Half-way modification (cur + (prop - cur) / 2) for fast knobs; None for configuration knobs (not divisible:
    the caller treats None as a reject)."""
    if req["kind"] == "cfg":
        return None
    return modify(req, req["cur"] + 0.5 * (req["prop"] - req["cur"]))


def request_cells(env, req):
    cells = set(env.knob_cells(req["knob"]))
    if req["knob"][0] == "assoc":
        cells.add(int(round(req["prop"])))
    return cells


def conflict_groups(env, requests):
    """Group this epoch's requests into per-cell conflict groups (union over shared cells). A group is a conflict if
    >= 2 xApps touch it; ``direct`` if two xApps request the same knob, else indirect (same cell, different knobs).
    -> list of {"cells", "idx" (request indices), "xapps", "direct"}, only groups with >= 2 xApps."""
    n = len(requests)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_cell: dict = {}
    for i, r in enumerate(requests):
        for c in request_cells(env, r):
            if c in by_cell:
                parent[find(i)] = find(by_cell[c])
            else:
                by_cell[c] = i
    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for idx in groups.values():
        xs = {requests[i]["xapp"] for i in idx}
        if len(xs) < 2:
            continue
        per_knob: dict = {}
        for i in idx:
            per_knob.setdefault(requests[i]["knob"], set()).add(requests[i]["xapp"])
        cells = set().union(*(request_cells(env, requests[i]) for i in idx))
        out.append({"cells": sorted(cells), "idx": idx, "xapps": sorted(xs),
                    "direct": any(len(v) > 1 for v in per_knob.values())})
    return out


class AcceptAll:
    name = "accept_all"

    def decide(self, env, requests):
        return list(requests)


class RejectAll:
    name = "freeze"

    def decide(self, env, requests):
        return []


class Subset:
    """Accept only requests of the named xApps (= those xApps acting alone / together; all deployed xApps still
    run, so every arm sees the same random tape)."""

    def __init__(self, names):
        self.names = set(names)
        self.name = "subset:" + "+".join(sorted(self.names))

    def decide(self, env, requests):
        return [r for r in requests if r["xapp"] in self.names]


class StaticPriority:
    """Per knob, only the highest-priority requester in ``order`` wins (direct conflicts); other knobs pass."""

    def __init__(self, order):
        self.order = list(order)
        self.name = "prio:" + ">".join(self.order)

    def decide(self, env, requests):
        rank = {n: i for i, n in enumerate(self.order)}
        best: dict = {}
        for r in requests:
            k = r["knob"]
            if k not in best or rank.get(r["xapp"], 99) < rank.get(best[k]["xapp"], 99):
                best[k] = r
        return [r for r in requests if best[r["knob"]] is r]


class CellPriorityLock:
    """Cell-scoped priority lock (SingleCommandPerNode-like): once an xApp's request on a cell is accepted, requests
    of lower-priority xApps on any knob of that cell are rejected for ``lease`` epochs. Within an epoch, the
    highest-priority xApp touching a cell takes it."""

    def __init__(self, order, lease=60):
        self.order, self.lease = list(order), int(lease)
        self.owner: dict = {}                                   # cell -> (rank, until)
        self.name = f"cell_lock:{'>'.join(self.order)}:{self.lease}"

    def decide(self, env, requests):
        rank = {n: i for i, n in enumerate(self.order)}
        t = env.t
        top: dict = {}
        for r in requests:
            for c in request_cells(env, r):
                top[c] = min(top.get(c, 99), rank.get(r["xapp"], 99))
        acc = []
        for r in requests:
            rk = rank.get(r["xapp"], 99)
            ok = True
            for c in request_cells(env, r):
                own = self.owner.get(c)
                if (own is not None and own[1] > t and own[0] < rk) or top[c] < rk:
                    ok = False
            if ok:
                acc.append(r)
        for r in acc:
            rk = rank.get(r["xapp"], 99)
            for c in request_cells(env, r):
                own = self.owner.get(c)
                if own is None or own[1] <= t or rk <= own[0]:
                    self.owner[c] = (rk, t + self.lease)
        return acc


def _direct_action(env):
    """Paper Direct map over the valid proposals -> configured (x, p) for every UE (no joint-feasibility check)."""
    c = env.cfg
    x, p = env.x.copy(), env.p.copy()
    tg = env.valid_proposals()
    rate = {t.idx: t.theta for t in tg if t.kpi == "rate"}
    if rate:
        us = np.array(sorted(rate))
        th = np.array([rate[u] for u in us])
        r = env.psd_rate(np.full(us.size, c.p_rb_w), us)
        x[us] = th / np.maximum(r, 1.0)                           # share for the target at P_rb per RB
        p[us] = x[us] * env.K * c.p_rb_w
    for t in tg:
        if t.kpi == "energy":
            b = t.idx
            m = env.z == b
            want = max((t.theta - c.p_cir_w) / c.delta_p, 0.0)
            s = p[m].sum()
            if s > want:
                p[m] *= want / s if s > 0 else 0.0
    return x, p


def _as_requests(env, name, x, p):
    out = []
    for u in range(env.U):
        out.append({"xapp": name, "knob": ("share", u), "cur": float(env.x[u]), "prop": float(x[u]), "t": env.t,
                    "kind": "fast", "target": None})
        out.append({"xapp": name, "knob": ("power", u), "cur": float(env.p[u]), "prop": float(p[u]), "t": env.t,
                    "kind": "fast", "target": None})
    for b in np.nonzero(env.alpha)[0]:
        if env.pcap[b] < env.cfg.p_max_w:
            out.append({"xapp": name, "knob": ("pcap", int(b)), "cur": float(env.pcap[b]),
                        "prop": env.cfg.p_max_w, "t": env.t, "kind": "fast", "target": None})
    return out


class Direct:
    """Paper benchmark 1. Run with ``XConfig(phys="none")`` to execute its infeasible actions as the paper's
    violation ratios imply; with ``phys="clip"`` the plant clips them (then Direct == Clipping in outcomes)."""
    name = "direct"

    def decide(self, env, requests):
        x, p = _direct_action(env)
        return _as_requests(env, self.name, x, p)


class Clipping:
    """Paper benchmark 2: Direct, then per-UE clipping and per-cell proportional rescaling onto c1-c4."""
    name = "clipping"

    def decide(self, env, requests):
        c = env.cfg
        x, p = _direct_action(env)
        z, B = env.z, env.B
        x = np.clip(x, 0.0, 1.0)
        sx = np.bincount(z, x, minlength=B)
        x = x * np.where(sx > 1, 1.0 / np.maximum(sx, 1e-12), 1.0)[z]
        p = np.clip(p, 0.0, x * env.K * c.p_rb_w)
        bud = env.alpha * c.p_max_w
        sp = np.bincount(z, p, minlength=B)
        p = p * np.where(sp > bud, bud / np.maximum(sp, 1e-300), 1.0)[z]
        return _as_requests(env, self.name, x, p)
