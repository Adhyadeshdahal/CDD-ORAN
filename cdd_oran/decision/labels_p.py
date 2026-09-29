"""E6-P unit labels by paired lookahead rollouts (scratchpad/e6_dev/decision/ARBITER_DESIGN.md section 3).

A unit u = (c, x, t0) is labelled with probability LABEL_RATE[x] (ES / PowerES 0.3, SliceGuarantee 0.05), drawn from
``default_rng([seed, 6613, c, x_idx, int(t0)])``. For every mode m of pi0's support for x and every reseed k in
{1, 2}: ``env.copy(reseed=k)`` at t0 (between propose and apply, so the copy carries t0's requests), m held for u's
T (``units_p.HoldPolicy``), every other unit accepted (AA continuation), rolled out for H = 90 s (the apply of t0 and
then H full control seconds t0+1 .. t0+H, fewer if the episode ends). Outcomes per rollout: network (plant.sla) and
per-cell (``collect_p.CellKPITap``) protected violated UE-s (pv), energy J (e), all-UE violated UE-s (v).

Label = contrast vs accept per k (same reseed k in both arms = common random numbers): d[m][scope][kpi] = [k1, k2]
for scope "exp" (sum over N(c), the SWM's target) and "net" (whole network); ``mean`` = the mean over k. The accept
arm of a unit is exactly a direct accept-all rollout from t0 (tested), so it is cached per (t0, k) and shared by every
unit labelled at t0. Reliability (G0a) = corr over labels of the k = 1 and k = 2 contrasts.

Region probe (G0b, sampled with REGION_RATE among labelled units, tag 6614, only for regions where x acts on >= 2
cells): with the same mode m (default reject) held from t0 for T on ALL of x's cells in the unit's region (the oracle's
region = ``plant.lay.cell_site``; privileged, labeller only) and, separately, on each cell of the region alone
(``HoldPolicy`` window [t0, t0 + T): a cell's unit opens at x's first request there), network contrasts vs AA.

``Labeller(cells=True)`` (option; default off = the output above, unchanged): rollouts also snapshot the tap's extended
per-cell arrays (``collect_p.CellKPITap.arrays(ext=True)``) and every label gains ``cell_kpis`` = ``CELL_KPIS`` (pv,
e, v, rlf, load = served UE-s, prb = PRBs used) and ``cell`` = {m: (len(ks), len(CELL_KPIS), cells) array of the
per-cell contrast m - accept per reseed k} (same CRN convention as ``d``; accept rows are zero). Used by the E6-P
discovery knockout ground truth (``gt_p``).
"""
from __future__ import annotations

import time

import numpy as np

from .collect_p import PI0_HIGH, PI0_ORDER, get_tap
from .units_p import T_UNIT, X_KNOBS, HoldPolicy

LABEL_TAG, REGION_TAG = 6613, 6614
LABEL_RATE = {"ES": 0.3, "PowerES": 0.3, "Coverage": 0.3, "SliceGuarantee": 0.05}
REGION_RATE = 0.2
H_LABEL = 90
KS = (1, 2)
KPIS = ("pv", "e", "v")
CELL_KPIS = ("pv", "e", "v", "rlf", "load", "prb")


def label_modes(x: str) -> tuple[str, ...]:
    """pi0's support for x (accept first)."""
    return tuple(m for m in PI0_ORDER if m in PI0_HIGH[x])


def _u(seed, tag, unit) -> float:
    return float(np.random.default_rng([int(seed), tag, int(unit["c"]), int(unit["x_idx"]), int(unit["t0"])]).random())


def sampled(seed: int, unit, rate: float | None = None) -> bool:
    return _u(seed, LABEL_TAG, unit) < (LABEL_RATE[unit["x"]] if rate is None else rate)


def region_sampled(seed: int, unit, rate: float = REGION_RATE) -> bool:
    return _u(seed, REGION_TAG, unit) < rate


def _snapshot(sim, ext=False):
    S, tap = sim.plant.sla, get_tap(sim)
    return (float(S["prot_viol"]), float(S["energy_j"]), float(S["viol_ue_s"])), tap.arrays(ext)


def _diff(sim, s0, n_sec, ext=False):
    net1, cell1 = _snapshot(sim, ext)
    return {"net": np.array(net1) - np.array(s0[0]), "cell": np.stack([a - b for a, b in zip(cell1, s0[1],
                                                                                          strict=True)]),
            "secs": n_sec}


def rollout(env, obs, snap, policy, k: int, H: int = H_LABEL, ext: bool = False):
    """Copy ``env`` (pending second t0) with reseed k, decide t0 with ``snap.fork(policy)`` on ``obs``, run H more
    seconds under the same arbiter. -> {"net": [pv, e, v], "cell": (3, cells) array (``ext``: (6, cells) in
    ``CELL_KPIS`` order), "secs"}."""
    sim = env.copy(reseed=k)
    s0 = _snapshot(sim, ext)
    arb = snap.fork(policy)
    sim.step_apply(arb(obs))
    n = 0
    while n < H and sim.sec < sim.total_s:
        o = sim.step_propose()
        sim.step_apply(arb(o))
        n += 1
    return _diff(sim, s0, n, ext)


def aa_rollout(env, k: int, H: int = H_LABEL, ext: bool = False):
    """Direct accept-all rollout (no arbiter object at all): the reference the accept label must equal."""
    sim = env.copy(reseed=k)
    s0 = _snapshot(sim, ext)
    sim.step_apply({"decisions": ["accept"] * len(sim._pending[2]), "writes": []})
    n = 0
    while n < H and sim.sec < sim.total_s:
        sim.step(None)
        n += 1
    return _diff(sim, s0, n, ext)


def _vals(res, cells):
    idx = list(cells)
    return {"net": [float(x) for x in res["net"]], "exp": [float(res["cell"][i, idx].sum()) for i in range(3)]}


class Labeller:
    """Stateful per-episode labeller: ``AA`` rollouts cached per (t0, k); CPU accounting in ``cpu_s``. ``cells``:
    keep per-cell contrast vectors (module docstring)."""

    def __init__(self, ks=KS, H: int = H_LABEL, T: float = T_UNIT, cells: bool = False):
        self.ks, self.H, self.T = tuple(ks), int(H), float(T)
        self.cells = bool(cells)
        self.cache, self.cpu_s, self.n_roll = {}, 0.0, 0

    def aa(self, env, obs, snap, t0, k):
        key = (float(t0), int(k))
        if key not in self.cache:
            self.cache = {kk: v for kk, v in self.cache.items() if kk[0] == float(t0)}   # only t0's entries live
            self.cache[key] = rollout(env, obs, snap, HoldPolicy((), "accept", 0, 0), k, self.H, self.cells)
            self.n_roll += 1
        return self.cache[key]

    def hold(self, env, obs, snap, keys, mode, t0, k, window):
        self.n_roll += 1
        return rollout(env, obs, snap, HoldPolicy(keys, mode, t0, t0 + window), k, self.H, self.cells)

    def label(self, env, obs, snap, unit, modes=None) -> dict:
        t_cpu = time.process_time()
        c, x, t0 = unit["c"], unit["x"], unit["t0"]
        modes = tuple(modes) if modes is not None else label_modes(x)
        raw, cellres = {}, {}
        for m in modes:
            raw[m], cellres[m] = [], []
            for k in self.ks:
                res = self.aa(env, obs, snap, t0, k) if m == "accept" else \
                    self.hold(env, obs, snap, {(c, x)}, m, t0, k, 1.0)
                raw[m].append(_vals(res, unit["exp"]) | {"secs": res["secs"]})
                cellres[m].append(res["cell"])
        d, mean = {}, {}
        for m in modes:
            d[m] = {sc: {kpi: [raw[m][i][sc][j] - raw["accept"][i][sc][j] for i in range(len(self.ks))]
                         for j, kpi in enumerate(KPIS)} for sc in ("exp", "net")}
            mean[m] = {sc: {kpi: float(np.mean(v)) for kpi, v in d[m][sc].items()} for sc in d[m]}
        out = {"modes": list(modes), "H": self.H, "ks": list(self.ks), "raw": raw, "d": d, "mean": mean}
        if self.cells:
            out["cell_kpis"] = list(CELL_KPIS)
            out["cell"] = {m: np.stack([cellres[m][i] - cellres["accept"][i] for i in range(len(self.ks))])
                           for m in modes}
        cpu = time.process_time() - t_cpu
        self.cpu_s += cpu
        out["cpu_s"] = round(cpu, 3)
        return out

    def region_probe(self, env, obs, snap, unit, site, mode="reject", label=None) -> dict | None:
        """G0b probe; None when x acts on < 2 cells of the unit's region. ``label`` (the unit's label) supplies the
        own-cell rollout when it contains ``mode``."""
        c, x, t0 = unit["c"], unit["x"], unit["t0"]
        types = X_KNOBS[x]
        acts = sorted({int(k[1]) for k in obs["static"]["knobs"] if k[0] in types})
        g = int(site[c])
        cells = [q for q in acts if int(site[q]) == g]
        if len(cells) < 2:
            return None
        t_cpu = time.process_time()
        aa = [self.aa(env, obs, snap, t0, k)["net"] for k in self.ks]
        dR = [(self.hold(env, obs, snap, {(q, x) for q in cells}, mode, t0, k, self.T)["net"] - a).tolist()
              for k, a in zip(self.ks, aa, strict=True)]
        dC = {}
        for q in cells:
            if q == c and label is not None and mode in label["d"]:
                dC[q] = [[label["d"][mode]["net"][kpi][i] for kpi in KPIS] for i in range(len(self.ks))]
                continue
            dC[q] = [(self.hold(env, obs, snap, {(q, x)}, mode, t0, k, self.T)["net"] - a).tolist()
                     for k, a in zip(self.ks, aa, strict=True)]
        sum_c = [np.sum([dC[q][i] for q in cells], 0).tolist() for i in range(len(self.ks))]
        cpu = time.process_time() - t_cpu
        self.cpu_s += cpu
        return {"region": g, "cells": cells, "mode": mode, "dR": dR, "dC": {str(q): v for q, v in dC.items()},
                "sum_dC": sum_c, "kpis": list(KPIS), "cpu_s": round(cpu, 3)}


__all__ = ["CELL_KPIS", "H_LABEL", "KPIS", "KS", "LABEL_RATE", "LABEL_TAG", "REGION_RATE", "REGION_TAG", "Labeller",
           "aa_rollout", "label_modes", "region_sampled", "rollout", "sampled"]
