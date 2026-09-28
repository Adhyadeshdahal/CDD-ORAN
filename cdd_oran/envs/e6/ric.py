"""E6 near-RT RIC layer: KPM reporting (granularity, delay, drops), knob registry with hard actuator limits, request
plumbing (ACCEPT / REJECT / MODIFY / DEFER, NACK feedback), and last-writer-wins default. The WG3 LOCK / ROLLBACK
actions and the churn cap live in ``env.E6Env.step_apply``; every change still passes ``feasible`` here.

Knob keys:  ("cio", s, n) dB | ("hys", c) dB | ("ttt", c) ms | ("ll_ratio", c) fraction | ("sleep", c) 0/1 |
            ("carrier", c) active macro carriers 1..MACRO_NTRX
E6-P knobs (registered only when cfg.e6p enables them): ("ptx", c) Tx-power offset dB | ("prot_min", c) protected
            work-conserving min PRB share. Their limits come from the plant's cfg.e6p (``LIMITS`` holds the
            E6PConfig() defaults for callers that only need a reference range).
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .sim import _rng

# actuator limits: (min, max, max |step| per change, min seconds between changes)          provenance
LIMITS = {"cio": (C.CIO_RANGE[0], C.CIO_RANGE[1], 2.0, 5.0),                               # [S] range / [A] rate
          "hys": (C.HYS_RANGE[0], C.HYS_RANGE[1], 0.5, 10.0),                              # [A]
          "ttt": (min(C.TTT_SET_MS), max(C.TTT_SET_MS), None, 10.0),                       # one TTT index per change
          "ll_ratio": (0.0, 0.5, 0.10, 1.0),                                               # [A] TS 28.541 ratio
          "sleep": (0, 1, None, 120.0),                                                    # [A] min dwell 120 s
          "carrier": (1, C.MACRO_NTRX, 1.0, 120.0)}                                        # [A] min dwell 120 s
_P0 = C.E6PConfig()
LIMITS.update({"ptx": (*_P0.ptx_range_db, _P0.ptx_max_step_db, _P0.ptx_min_interval_s),          # E6-P PROVISIONAL
               "prot_min": (*_P0.prot_min_range, _P0.prot_min_max_step, _P0.prot_min_interval_s)})
P_TYPES = ("ptx", "prot_min")
GRAN_S = {"fast": 1, "thp": 5, "mob": 30, "energy": 60}                                    # E2SM-KPM granularities [A]


def knob_get(plant, k):
    typ = k[0]
    if typ == "cio":
        return float(plant.cio[k[1], k[2]])
    if typ == "hys":
        return float(plant.hys[k[1]])
    if typ == "ttt":
        return float(plant.ttt[k[1]])
    if typ == "ll_ratio":
        return float(plant.ll_ratio[k[1]])
    if typ == "sleep":
        return float(plant.asleep[k[1]])
    if typ == "carrier":
        return float(plant.n_car[k[1]])
    if typ == "ptx":
        return float(plant.ptx_off[k[1]])
    if typ == "prot_min":
        return float(plant.prot_min[k[1]])
    raise KeyError(k)


def _limits(plant, typ):
    """Actuator limits of a knob type; E6-P types read the plant's own cfg.e6p."""
    if typ == "ptx":
        P = plant.cfg.e6p
        return (*P.ptx_range_db, P.ptx_max_step_db, P.ptx_min_interval_s)
    if typ == "prot_min":
        P = plant.cfg.e6p
        return (*P.prot_min_range, P.prot_min_max_step, P.prot_min_interval_s)
    return LIMITS[typ]


def _quantise(k, v, P=None):
    typ = k[0]
    if typ in P_TYPES:
        P = P or _P0
        g = P.ptx_grid_db if typ == "ptx" else P.prot_min_grid
        return float(np.round(v / g) * g)
    if typ == "cio":
        return float(np.round(v))                     # 1 dB steps
    if typ == "hys":
        return float(np.round(v * 2) / 2)             # 0.5 dB steps
    if typ == "ttt":
        return float(min(C.TTT_SET_MS, key=lambda x: abs(x - v)))
    if typ == "ll_ratio":
        return float(np.round(v * 20) / 20)           # 5 % steps
    if typ == "sleep":
        return float(v >= 0.5)
    if typ == "carrier":
        return float(np.round(v))
    return float(v)


def feasible(plant, k, v, now, last_change):
    """Clip a requested value to the hard actuator limits (range, max step, min interval). Returns (value, ok)."""
    lo, hi, mstep, mint = LIMITS[k[0]] if k[0] not in P_TYPES else _limits(plant, k[0])
    cur = knob_get(plant, k)
    if now - last_change.get(k, -1e9) < mint:
        return cur, False
    v = min(max(_quantise(k, v, plant.cfg.e6p if k[0] in P_TYPES else None), lo), hi)
    if k[0] == "ttt":
        i0, i1 = C.TTT_SET_MS.index(int(cur)), C.TTT_SET_MS.index(int(v))
        v = float(C.TTT_SET_MS[i0 + int(np.clip(i1 - i0, -1, 1))])
    elif mstep is not None:
        v = cur + float(np.clip(v - cur, -mstep, mstep))
    return v, True


def knob_set(plant, k, v, now):
    typ = k[0]
    if typ == "cio":
        plant.cio[k[1], k[2]] = v
    elif typ == "hys":
        plant.hys[k[1]] = v
    elif typ == "ttt":
        plant.ttt[k[1]] = int(v)
    elif typ == "ll_ratio":
        plant.ll_ratio[k[1]] = v
    elif typ == "sleep":
        c = k[1]
        if v >= 0.5 and not plant.asleep[c]:
            plant.asleep[c] = True
            if plant.p_on and plant.P.pico_sleep_ho:                   # E6-P: graceful sleep (hand UEs over)
                plant.sleep_handover(c, now)
        elif v < 0.5 and plant.asleep[c]:
            plant.asleep[c] = False
            plant.waking_until[c] = now + C.PICO_WAKE_S
    elif typ == "carrier":
        c, v = k[1], int(v)
        if v > plant.n_car[c]:
            plant.car_on_at[c] = now + C.CARRIER_ON_S
        plant.n_car[c] = v
    elif typ == "ptx":
        plant.ptx_off[k[1]] = v
    elif typ == "prot_min":
        plant.prot_min[k[1]] = v


class KPM:
    """Aggregates 1 s plant counters into E2SM-KPM-like reports per granularity, delivered with delay/drops."""

    def __init__(self, cfg, plant):
        self.cfg, self.plant = cfg, plant
        self.buf = {g: [] for g in GRAN_S}
        self.in_flight = []           # (deliver_at, report)
        self.delivered = []
        if cfg.kpm == "degraded":
            self.delay, self.drop = (2.0, 5.0), 0.05
        else:
            self.delay, self.drop = tuple(cfg.kpm_delay_s), cfg.kpm_drop

    def second(self, sec):
        """Called at the end of simulated second ``sec`` (window [sec-1, sec))."""
        c = self.plant.take_counters()
        for g in self.buf:
            self.buf[g].append(c)
        salt = self.plant.rng_salt
        r = _rng(self.cfg.seed, "kpm", sec) if salt is None else _rng(self.cfg.seed, "kpm", sec, salt)
        for g, per in GRAN_S.items():
            if sec % per:
                continue
            rep = self._make(g, self.buf[g], sec - per, sec)
            self.buf[g] = []
            if r.uniform() < self.drop:
                continue
            self.in_flight.append((sec + r.uniform(*self.delay), rep))

    def deliver(self, now):
        due = [x for x in self.in_flight if x[0] <= now]
        self.in_flight = [x for x in self.in_flight if x[0] > now]
        out = [dict(rep, arrived=t) for t, rep in sorted(due, key=lambda z: z[0])]
        self.delivered.extend(out)
        return out

    def _make(self, g, cs, t0, t1):
        nc = self.plant.nc
        ticks = max(sum(c["ticks"] for c in cs), 1)
        rep = {"gran": g, "t0": t0, "t1": t1}
        if g == "fast":
            used = sum(c["prb_used"] for c in cs)
            rsv = sum(c["prb_rsv"] for c in cs)
            cap = sum(c["prb_cap"] for c in cs)                               # active PRB-ticks (carrier/sleep aware)
            with np.errstate(invalid="ignore", divide="ignore"):
                rep["prb_util"] = np.where(cap > 0, (used.sum(1) + rsv) / cap, 0.0)   # INCLUDES reserved PRBs
                rep["prb_util_slice"] = np.where(cap[:, None] > 0, used / cap[:, None], 0.0)
                rep["prb_rsv_idle"] = np.where(cap > 0, rsv / cap, 0.0)             # reserved-but-idle share
            rep["carriers"] = self.plant.n_car.copy()
            rep["act_ue"] = sum(c["act_ue"] for c in cs) / ticks
            d = [[] for _ in range(nc)]
            for c in cs:
                for i in range(nc):
                    d[i].extend(c["ll_delay"][i])
            rep["ll_delay_p95"] = np.array([np.percentile(x, 95) if x else np.nan for x in d])
            rep["ll_samples"] = np.array([len(x) for x in d])
        elif g == "thp":
            bits = sum(c["bits"] for c in cs)
            rep["thp_slice"] = bits / (len(cs) * C.CONTROL_S)
            e = [[] for _ in range(nc)]
            for c in cs:
                for i in range(nc):
                    e[i].extend(c["embb_thp"][i])
            rep["embb_thp_p5"] = np.array([np.percentile(x, 5) if x else np.nan for x in e])
        elif g == "mob":
            for key in ("ho_att", "ho_succ", "too_late", "too_early", "wrong_cell", "pingpong"):
                rep[key] = sum(c[key] for c in cs)
            rep["rlf"] = sum(c["rlf"] for c in cs)
        elif g == "energy":
            rep["energy_j"] = sum(c["energy_j"] for c in cs)
        if g == "fast" and self.plant.p_on:
            self._make_p(rep, cs, ticks)
        return rep

    def _make_p(self, rep, cs, ticks):
        """E6-P additions to the fast report (only when enabled): Tx power, edge-SINR percentile, protected slice."""
        p, nc = self.plant, self.plant.nc
        P = p.P
        rep["ptx_db"] = p.ptx_off.copy()
        rep["ptx_dbm"] = np.where(p.lay.is_macro, C.MACRO_PTX_DBM, C.PICO_PTX_DBM) + p.ptx_off
        srv = np.concatenate([x for c in cs for x in c["sinr_serv"]] or [np.zeros(0, int)])
        val = np.concatenate([x for c in cs for x in c["sinr_val"]] or [np.zeros(0)])
        edge = np.full(nc, np.nan)
        cnt = np.bincount(srv, minlength=nc)
        for i in np.nonzero(cnt)[0]:
            edge[i] = np.percentile(val[srv == i], P.edge_pct)
        rep["edge_sinr_p"] = edge
        rep["sinr_samples"] = cnt
        if not p.p_prot:
            return
        cap = sum(c["prb_cap"] for c in cs)
        with np.errstate(invalid="ignore", divide="ignore"):
            rep["prot_min"] = p.prot_min.copy()
            rep["prot_dem_share"] = sum(c["prot_dem"] for c in cs) / ticks                   # demand / active PRBs
            rep["prot_used_share"] = np.where(cap > 0, sum(c["prot_used"] for c in cs) / cap, 0.0)
            rep["prot_min_share"] = np.where(cap > 0, sum(c["prot_min_prb"] for c in cs) / cap, 0.0)
            rep["prot_thp"] = sum(c["prot_bits"] for c in cs) / (len(cs) * C.CONTROL_S)
            rep["prot_act_ue"] = sum(c["prot_act"] for c in cs) / ticks
            ev = sum(c["prot_eval"] for c in cs)
            rep["prot_eval"] = ev
            rep["prot_below_frac"] = np.where(ev > 0, sum(c["prot_below"] for c in cs) / np.maximum(ev, 1), np.nan)
