"""E6 xApps: rule-based SON controllers (TS 32.522 / 36.902 style) acting only on DELIVERED KPM reports and the
applied configuration. Each has an implementation variant drawn per seed:
  * thresholds scaled within +-25 % (and by cfg.xapp_threshold_scale for plant-family TEST draws);
  * blocked-action behaviour: 'retry' (re-propose the same value next cycle), 'escalate' (keeps integrating its own
    target, so a later acceptance jumps), 'hold' (skips one cycle). All variants are NACK-aware: they read the APPLIED
    configuration before proposing.
Requests: {"xapp", "ver", "knob", "cur", "prop", "t"}.
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .ric import knob_get
from .sim import _rng

BEHAVIOURS = ("retry", "escalate", "hold")


class XApp:
    name = "base"
    cadence = 10.0

    def __init__(self, env, idx):
        self.env, self.p = env, env.plant
        r = _rng(env.cfg.seed, "xapp", 1000 + idx)
        self.scale = r.uniform(0.75, 1.25) * env.cfg.xapp_threshold_scale
        self.behaviour = BEHAVIOURS[int(r.integers(3))]
        self.next_t = float(r.uniform(0, self.cadence))
        self.ver = 1
        self.pending_target = {}        # knob -> desired value (escalate)
        self.hold_until = -1.0
        self.reports = {}               # latest delivered report per granularity

    def observe(self, reps):
        for rep in reps:
            self.reports[rep["gran"]] = rep

    def due(self, now):
        if now + 1e-9 >= self.next_t:
            self.next_t += self.cadence
            return now >= self.hold_until
        return False

    def req(self, knob, prop, now):
        cur = knob_get(self.p, knob)
        if self.behaviour == "escalate" and knob in self.pending_target:
            prop = self.pending_target[knob] + (prop - cur)
        if abs(prop - cur) < 1e-9:
            return None
        return {"xapp": self.name, "ver": self.ver, "knob": knob, "cur": cur, "prop": float(prop), "t": now}

    def result(self, r, accepted, applied, now):
        k = r["knob"]
        if accepted:
            self.pending_target.pop(k, None)
        elif self.behaviour == "escalate":
            self.pending_target[k] = r["prop"]
        elif self.behaviour == "hold":
            self.hold_until = now + self.cadence

    def propose(self, now):
        return []

    def update_version(self):
        self.ver += 1


class MRO(XApp):
    name, cadence = "MRO", 30.0

    def __init__(self, env, idx):
        super().__init__(env, idx)
        self.tl_thr, self.te_thr, self.step = 0.02 * self.scale, 0.10 * self.scale, 1.0

    def observe(self, reps):
        super().observe(reps)
        for rep in reps:
            if rep["gran"] == "mob":
                self.mob_hist = (getattr(self, "mob_hist", []) + [rep])[-2:]      # 60 s window

    def propose(self, now):
        hist = getattr(self, "mob_hist", [])
        if not hist:
            return []
        rep = {k: sum(h[k] for h in hist) for k in ("ho_att", "too_late", "too_early", "pingpong", "wrong_cell")}
        out = []
        lay = self.p.lay
        for s in range(self.p.nc):
            hys_up = False
            for n in lay.neighbours[s]:
                att = rep["ho_att"][s, n]
                tl = rep["too_late"][s, n]
                te = rep["too_early"][s, n] + rep["pingpong"][s, n] + rep["wrong_cell"][s, n]
                den = att + tl
                if den < 3:                          # too few events to act on
                    continue
                cio = self.p.cio[s, n]
                if tl >= 2 and tl / den > self.tl_thr:
                    if cio < C.CIO_RANGE[1]:
                        out.append(self.req(("cio", s, n), cio + self.step, now))
                    else:
                        i = C.TTT_SET_MS.index(int(self.p.ttt[s]))
                        if i > 0:
                            out.append(self.req(("ttt", s), C.TTT_SET_MS[i - 1], now))
                elif te >= 3 and te / max(att, 1) > self.te_thr and tl == 0:
                    out.append(self.req(("cio", s, n), cio - self.step, now))
                    hys_up = True
            if hys_up and self.p.hys[s] < C.HYS_RANGE[1]:
                out.append(self.req(("hys", s), self.p.hys[s] + 0.5, now))
        return [r for r in out if r]

    def update_version(self):
        super().update_version()
        self.step = 2.0


class TS(XApp):
    """Traffic steering / mobility load balancing on the same CIO knob (cell range extension toward less-loaded)."""
    name, cadence = "TS", 10.0

    def __init__(self, env, idx):
        super().__init__(env, idx)
        self.u_hi, self.gap, self.u_tgt, self.cio_max = 0.6 * self.scale, 0.15 * self.scale, 0.5 * self.scale, 4.0
        self.u = None

    def propose(self, now):
        rep = self.reports.get("fast")
        if rep is None:
            return []
        u = rep["prb_util"]                    # includes reserved dedicated PRBs (implicit Slice->TS coupling)
        self.u = u if self.u is None else 0.7 * self.u + 0.3 * u
        thp = self.reports.get("thp")
        p5 = thp["embb_thp_p5"] if thp is not None else np.full(self.p.nc, np.nan)
        suffer = np.nan_to_num(p5, nan=np.inf) < C.EMBB_THP_TARGET_BPS * 1.2   # offload only where users suffer
        out = []
        for s in range(self.p.nc):
            if self.p.asleep[s]:
                continue
            for n in self.p.lay.neighbours[s]:
                if self.p.asleep[n]:
                    continue
                cio = self.p.cio[s, n]
                d = self.u[s] - self.u[n]
                if suffer[s] and self.u[s] > self.u_hi and self.u[n] < self.u_tgt and d > self.gap and cio < self.cio_max:
                    out.append(self.req(("cio", s, n), cio + 1, now))            # push s -> n
                    if self.p.cio[n, s] > C.CIO_RANGE[0]:
                        out.append(self.req(("cio", n, s), self.p.cio[n, s] - 1, now))   # and hold them in n
                elif abs(d) < 0.05 and cio != 0:
                    out.append(self.req(("cio", s, n), cio - np.sign(cio), now))
        return [r for r in out if r]

    def update_version(self):
        super().update_version()
        self.gap = 0.08 * self.scale


class ES(XApp):
    """Energy saving by macro capacity-carrier shutdown (coverage carrier always on): switch a carrier off when the
    sector's full-band load stays low and its macro neighbours have headroom; switch it back on when the remaining
    carrier gets busy. Reported utilisation is relative to ACTIVE capacity, so shutting a carrier doubles it."""
    name, cadence = "ES", 10.0

    def __init__(self, env, idx):
        super().__init__(env, idx)
        lay = self.p.lay
        self.macros = [c for c in range(lay.n_cells) if lay.is_macro[c]]
        self.nbr = {c: [n for n in lay.neighbours[c] if lay.is_macro[n]] for c in self.macros}
        self.low_since = {c: None for c in self.macros}
        self.u_off, self.u_mac, self.u_on = 0.30 * self.scale, 0.70 * self.scale, 0.80 * self.scale
        self.u = None

    def propose(self, now):
        rep = self.reports.get("fast")
        if rep is None:
            return []
        # ~20 s EWMA over the 10 s cadence: raw 1 s utilisation is too bursty to hold a low spell
        self.u = rep["prb_util"] if self.u is None else 0.6 * self.u + 0.4 * rep["prb_util"]
        full = self.u * self.p.n_car / self.p.n_trx          # load as a share of the full band
        out = []
        for c in self.macros:
            ncar, ntrx = self.p.n_car[c], self.p.n_trx[c]
            if ncar < ntrx:
                self.low_since[c] = None
                if self.u[c] > self.u_on:
                    out.append(self.req(("carrier", c), float(ntrx), now))
                continue
            nb = float(np.mean(full[self.nbr[c]])) if self.nbr[c] else 0.0
            if full[c] < self.u_off and nb < self.u_mac:
                self.low_since[c] = self.low_since[c] or now
                if now - self.low_since[c] >= 60:
                    out.append(self.req(("carrier", c), float(ncar - 1), now))
            else:
                self.low_since[c] = None
        return [r for r in out if r]

    def update_version(self):
        super().update_version()
        self.u_on = 0.90 * self.scale


class SliceSLA(XApp):
    name, cadence = "SLICE", 1.0

    def observe(self, reps):
        super().observe(reps)
        for rep in reps:
            if rep["gran"] == "fast":
                self.fast_hist = (getattr(self, "fast_hist", []) + [rep])[-5:]

    def __init__(self, env, idx):
        super().__init__(env, idx)
        self.hi, self.lo = 0.8 * self.scale, 0.4 * self.scale
        self.good_since = {}

    def propose(self, now):
        hist = getattr(self, "fast_hist", [])
        if not hist:
            return []
        out, tgt = [], C.LL_DELAY_TARGET_S
        n_s = sum(h["ll_samples"] for h in hist)
        with np.errstate(invalid="ignore"):
            d_all = np.nanmax(np.array([h["ll_delay_p95"] for h in hist]), 0)     # worst p95 over the last 5 s
        for c in range(self.p.nc):
            d = d_all[c]
            if np.isnan(d) or n_s[c] < 5:                  # not enough fresh LL samples
                continue
            r = self.p.ll_ratio[c]
            # share of the dedicated pool left idle: if LL is not PRB-starved, more reservation cannot cut its delay
            idle = float(np.mean([h["prb_rsv_idle"][c] for h in hist])) / r if r > 0 else 0.0
            if idle > 0.5 and r > 0:                        # reservation mostly wasted -> give PRBs back
                out.append(self.req(("ll_ratio", c), r - 0.05, now))
                self.good_since.pop(c, None)
            elif d > self.hi * tgt and r < 0.5 and idle < 0.2:
                out.append(self.req(("ll_ratio", c), r + 0.05, now))
                self.good_since.pop(c, None)
            elif d < self.lo * tgt and r > 0:
                self.good_since.setdefault(c, now)
                if now - self.good_since[c] >= 10:
                    out.append(self.req(("ll_ratio", c), r - 0.05, now))
                    self.good_since[c] = now
            else:
                self.good_since.pop(c, None)
        return [r for r in out if r]

    def update_version(self):
        super().update_version()
        self.hi = 0.6 * self.scale


MIXES = {"none": (), "MRO": (MRO,), "TS": (TS,), "ES": (ES,), "SLICE": (SliceSLA,), "M_TS_MRO": (MRO, TS),
         "M2": (MRO, TS, ES), "M4": (MRO, TS, ES, SliceSLA)}
