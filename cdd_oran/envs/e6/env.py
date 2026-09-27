"""E6 episode loop and the common arbiter interface.

Per control second (after the plant's 25 ticks):
  1. KPM: close the second's counters, emit reports whose granularity ends now, deliver reports whose delay elapsed;
  2. xApps observe delivered reports; due xApps propose requests (reading the APPLIED configuration);
  3. the ARBITER sees ``obs`` and returns decisions (default: no arbiter = accept everything, last writer wins);
  4. the RIC applies accepted/modified values through the hard actuator limits; xApps get ACK/NACK.

Arbiter contract: ``arbiter(obs) -> {"decisions": [...], "writes": [(knob, value), ...]}``
  decisions[i] for obs["requests"][i]: "accept" | "reject" | ("modify", value) | "defer" (re-offered next second,
  auto-reject after 10 s). ``writes`` are arbiter-originated settings (bounded joint actions / probes), limited by the
  same actuator limits and a write budget (``write_budget`` per scored hour). Arbiters get ONLY ``obs``.
obs = {"t", "new_reports", "config" (knob -> value), "requests", "static": {cells, neighbours, is_macro, knobs,
       xapps (declared knobs only)}}.
"""
from __future__ import annotations

import copy

from . import config as C
from .ric import KPM, feasible, knob_get, knob_set
from .sim import Plant, _rng
from .xapps import MIXES


class E6Env:
    def __init__(self, cfg: C.E6Config, write_budget: float = 300.0, log: bool = True):
        self.cfg = cfg
        self.plant = Plant(cfg)
        self.kpm = KPM(cfg, self.plant)
        self.xapps = [cls(self, i) for i, cls in enumerate(MIXES[cfg.mix])]
        self.last_change = {}
        self.deferred = []
        self.write_budget = write_budget
        self.writes_used = 0
        self.log_on = log
        self.log = []
        lay = self.plant.lay
        self.knobs = [("cio", s, n) for s in range(lay.n_cells) for n in lay.neighbours[s]] + \
                     [("hys", c) for c in range(lay.n_cells)] + [("ttt", c) for c in range(lay.n_cells)]
        if cfg.mix == "M4":
            self.knobs += [("ll_ratio", c) for c in range(lay.n_cells)]
        self.knobs += [("sleep", c) for c in range(lay.n_cells) if not lay.is_macro[c]]
        self.knobs += [("carrier", c) for c in range(lay.n_cells) if lay.is_macro[c]]
        r = _rng(cfg.seed, "xapp", 999)
        total_s = cfg.warmup_s + cfg.scored_s
        self.update_at = (cfg.warmup_s + r.uniform(0.15, 0.6) * cfg.scored_s) if cfg.update and self.xapps else None
        self.update_xapp = int(r.integers(max(len(self.xapps), 1)))
        self.total_s = int(total_s)
        self.stats = {"req": 0, "acc": 0, "rej": 0, "mod": 0, "def": 0, "writes": 0}
        self.sec = 0
        self._static = None

    def static(self):
        lay = self.plant.lay
        return {"cells": lay.n_cells, "neighbours": lay.neighbours, "is_macro": lay.is_macro.copy(),
                "knobs": list(self.knobs), "limits": "see ric.LIMITS",
                "xapps": {x.name: sorted({k[0] for k in self.knobs}) for x in self.xapps}}

    def config(self):
        return {k: knob_get(self.plant, k) for k in self.knobs}

    def _apply(self, k, v, now):
        v2, ok = feasible(self.plant, k, v, now, self.last_change)
        if not ok:
            return False, knob_get(self.plant, k)
        if abs(v2 - knob_get(self.plant, k)) > 1e-9:
            knob_set(self.plant, k, v2, now)
            self.last_change[k] = now
        return True, v2

    def run(self, arbiter=None):
        while self.sec < self.total_s:
            self.step(arbiter)
        return self.score()

    def copy(self):
        """Independent copy for lookahead rollouts (shares only the immutable layout and gain maps)."""
        memo = {id(self.plant.gm): self.plant.gm, id(self.plant.lay): self.plant.lay}
        return copy.deepcopy(self, memo)

    def step(self, arbiter=None):
        """Advance one control second (plant ticks, KPM, xApps, arbiter, RIC apply)."""
        p = self.plant
        if self._static is None:
            self._static = self.static()
        st = self._static
        for _ in range(C.TICKS_PER_CONTROL):
            p.tick()
        self.sec += 1
        sec = self.sec
        now = float(sec)
        self.kpm.second(sec)
        new = self.kpm.deliver(now)
        if self.update_at is not None and now >= self.update_at:
            self.xapps[self.update_xapp].update_version()
            self.update_at = None
        reqs = [d["req"] for d in self.deferred]
        for x in self.xapps:
            x.observe(new)
            if x.due(now):
                reqs.extend(x.propose(now))
        dec = {"decisions": ["accept"] * len(reqs), "writes": []}
        if arbiter is not None:
            obs = {"t": now, "new_reports": new, "config": self.config(), "requests": reqs, "static": st}
            dec = arbiter(obs)
        old_def = {id(d["req"]): d["age"] for d in self.deferred}
        self.deferred = []
        for r, d in zip(reqs, dec["decisions"], strict=True):
            self.stats["req"] += 1
            x = next(a for a in self.xapps if a.name == r["xapp"])
            if d == "defer":
                age = old_def.get(id(r), 0) + 1
                if age <= 10:
                    self.deferred.append({"req": r, "age": age})
                    self.stats["def"] += 1
                    continue
                d = "reject"
            if d == "reject":
                self.stats["rej"] += 1
                x.result(r, False, knob_get(p, r["knob"]), now)
                continue
            v = r["prop"] if d == "accept" else float(d[1])
            ok, applied = self._apply(r["knob"], v, now)
            self.stats["acc" if d == "accept" else "mod"] += int(ok)
            x.result(r, ok, applied, now)
        for k, v in dec.get("writes", []):
            if self.writes_used >= self.write_budget * max(self.cfg.scored_s, 1) / 3600 + 1e-9:
                break
            ok, _ = self._apply(k, v, now)
            if ok:
                self.writes_used += 1
                self.stats["writes"] += 1
        if self.log_on:
            self.log.append({"t": now, "config": self.config(), "reports": new, "n_req": len(reqs)})

    def score(self):
        S = self.plant.sla
        ue_h = S["ue_s"] / 3600.0
        return {"svr": S["viol_ue_s"] / max(ue_h, 1e-9),           # violated UE-seconds per UE-hour
                "viol_frac": S["viol_ue_s"] / max(S["ue_s"], 1),
                "ll_viol": S["ll_viol"], "embb_viol": S["embb_viol"], "outage_viol": S["outage_viol"],
                "severe": S["severe"], "energy_kwh": S["energy_j"] / 3.6e6, "rlf_per_ue_h": S["rlf"] / max(ue_h, 1e-9),
                "ho_per_ue_h": S["ho"] / max(ue_h, 1e-9), "pingpong": S["pingpong"], **self.stats}


def run_episode(cfg, arbiter=None, **kw):
    env = E6Env(cfg, **kw)
    return env.run(arbiter), env
