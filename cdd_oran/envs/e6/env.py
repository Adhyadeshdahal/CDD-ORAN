"""E6 episode loop and the common arbiter interface.

Per control second (after the plant's 25 ticks):
  1. KPM: close the second's counters, emit reports whose granularity ends now, deliver reports whose delay elapsed;
  2. xApps observe delivered reports; due xApps propose requests (reading the APPLIED configuration);
  3. the ARBITER sees ``obs`` and returns decisions (default: no arbiter = accept everything, last writer wins);
  4. the RIC applies accepted/modified values through the hard actuator limits; xApps get ACK/NACK.

Two-phase API: ``obs = env.step_propose()`` runs 1-2 and parks the requests on the env; ``env.step_apply(dec)`` runs 4.
``step(arbiter)`` = propose -> arbiter(obs) (None = accept all) -> apply. ``env.copy()`` taken between the phases
carries the pending requests, so ``c = env.copy(); c.step_apply(dec)`` replays exactly what ``env.step_apply(dec)`` does.

Arbiter contract: ``arbiter(obs) -> {"decisions": [...], "writes": [(knob, value), ...], "rollback": [knob, ...]}``
  decisions[i] for obs["requests"][i] (O-RAN WG3 conflict-mitigation actions):
    "accept" | "reject" | ("modify", value) | "defer" (re-offered next second, auto-reject after 10 s) |
    ("lock", secs): reject it AND lock its knob for ``secs`` s (locked while t < t_lock + secs). Requests on a
    locked knob still appear in obs (the arbiter sees the demand) but are force-rejected with NACK whatever the
    decision (counted rej + "lock_blocked").
  rollback: restore each knob to its last-known-good value = the value in force before its most recent change
    (no-op if never changed; a second rollback undoes the first). Applied BEFORE the decisions, through the same
    actuator limits (the min interval can refuse it; a same-second xApp request on that knob then NACKs).
  writes: arbiter-originated free settings, limited by the actuator limits and ``write_budget`` per scored hour.
    With ``wg3=True`` writes are not a WG3 action: a non-empty list raises ValueError.
Churn: every APPLIED knob change (accepted, modified, rolled back, written) counts in stats "changes" (whole
  episode, warm-up included). With ``churn_cap`` set, once changes >= cap every further change is refused:
  accept/modify -> NACK (counted rej + "churn_blocked"); rollback/write skipped (+ "churn_blocked").
  No-op accepts (value already in force) are not changes and are never blocked.
Trace: ``trace=True`` attaches a ``cdd_oran.decision.trace.TraceRecorder`` (per-second propose/apply/outcome rows +
  privileged per-cell labels); ``env.get_trace()`` returns it. Off by default: no recording, no extra work.
Arbiters get ONLY ``obs``.
obs = {"t", "new_reports", "config" (knob -> value), "requests", "locked" (knob -> until, active only),
       "changes", "churn_cap", "static": {cells, neighbours, is_macro, knobs, xapps (declared knobs only)}}.
"""
from __future__ import annotations

import copy

from . import config as C
from .ric import KPM, feasible, knob_get, knob_set
from .sim import Plant, _rng
from .xapps import MIXES


class E6Env:
    def __init__(self, cfg: C.E6Config, write_budget: float = 300.0, log: bool = True, wg3: bool = False,
                 churn_cap: int | None = None, trace: bool = False, trace_snapshot_s: int = 60):
        self.cfg = cfg
        self.plant = Plant(cfg)
        self.kpm = KPM(cfg, self.plant)
        self.xapps = [cls(self, i) for i, cls in enumerate(MIXES[cfg.mix])]
        self.last_change = {}
        self.prev_val = {}                # knob -> value in force before its most recent change (last-known-good)
        self.plant.oam_hook = self._on_oam  # OAM/SMO writes outside the RIC (stress scenario S2)
        self.lock_until = {}              # knob -> time the WG3 lock expires
        self.deferred = []
        self.write_budget = write_budget
        self.writes_used = 0
        self.wg3, self.churn_cap = wg3, churn_cap
        self.log_on = log
        self.log = []
        lay = self.plant.lay
        self.knobs = [("cio", s, n) for s in range(lay.n_cells) for n in lay.neighbours[s]] + \
                     [("hys", c) for c in range(lay.n_cells)] + [("ttt", c) for c in range(lay.n_cells)]
        if cfg.mix == "M4":
            self.knobs += [("ll_ratio", c) for c in range(lay.n_cells)]
        self.knobs += [("sleep", c) for c in range(lay.n_cells) if not lay.is_macro[c]]
        self.knobs += [("carrier", c) for c in range(lay.n_cells) if lay.is_macro[c]]
        fwd = {(s, n) for s in range(lay.n_cells) for n in lay.neighbours[s]}  # TS/ES also write the reverse CIO
        self.knobs += [("cio", n, s) for s in range(lay.n_cells) for n in lay.neighbours[s] if (n, s) not in fwd]
        r = _rng(cfg.seed, "xapp", 999)
        total_s = cfg.warmup_s + cfg.scored_s
        self.update_at = (cfg.warmup_s + r.uniform(0.15, 0.6) * cfg.scored_s) if cfg.update and self.xapps else None
        self.update_xapp = int(r.integers(max(len(self.xapps), 1)))
        self.total_s = int(total_s)
        self.stats = {"req": 0, "acc": 0, "rej": 0, "mod": 0, "def": 0, "writes": 0,
                      "changes": 0, "churn_blocked": 0, "lock_blocked": 0, "locks": 0, "rollbacks": 0}
        self.sec = 0
        self._static = None
        self._pending = None              # (now, new_reports, requests) between step_propose and step_apply
        self._tr = None                   # optional trace recorder (cdd_oran.decision.trace); never copied
        if trace:
            from cdd_oran.decision.trace import TraceRecorder
            self._tr = TraceRecorder(self, trace_snapshot_s)

    def static(self):
        lay = self.plant.lay
        return {"cells": lay.n_cells, "neighbours": lay.neighbours, "is_macro": lay.is_macro.copy(),
                "knobs": list(self.knobs), "limits": "see ric.LIMITS",
                "xapps": {x.name: sorted({k[0] for k in self.knobs}) for x in self.xapps}}

    def config(self):
        return {k: knob_get(self.plant, k) for k in self.knobs}

    def _apply(self, k, v, now):
        """-> (status, value): "ok" (applied or no-op) | "nack" (actuator limit) | "churn" (churn cap reached)."""
        v2, ok = feasible(self.plant, k, v, now, self.last_change)
        cur = knob_get(self.plant, k)
        if not ok:
            return "nack", cur
        if abs(v2 - cur) > 1e-9:
            if self.churn_cap is not None and self.stats["changes"] >= self.churn_cap:
                self.stats["churn_blocked"] += 1
                return "churn", cur
            knob_set(self.plant, k, v2, now)
            self.prev_val[k] = cur
            self.last_change[k] = now
            self.stats["changes"] += 1
        return "ok", v2

    def _on_oam(self, k, v, now):
        """OAM/SMO write outside the RIC (scenario S2 rollout / SMO restore): it becomes the knob's last-known-good,
        so a RIC rollback cannot undo it and the next RIC change records it as its prior. Not a RIC change: no
        dwell (``last_change``), no churn."""
        self.prev_val[k] = float(v)

    def locked(self, k, now):
        return now < self.lock_until.get(k, -1e9)

    def run(self, arbiter=None):
        while self.sec < self.total_s:
            self.step(arbiter)
        return self.score()

    def copy(self):
        """Independent copy for lookahead rollouts (shares only the immutable layout and gain maps). Taken between
        step_propose and step_apply it carries the pending requests (replayed identically by step_apply)."""
        memo = {id(self.plant.gm): self.plant.gm, id(self.plant.lay): self.plant.lay}
        if self._tr is not None:
            memo[id(self._tr)] = None     # copies are untraced (lookahead rollouts must not write the trace)
        return copy.deepcopy(self, memo)

    def get_trace(self):
        """The episode's ``decision.trace.Trace`` (requires ``trace=True``)."""
        if self._tr is None:
            raise RuntimeError("E6Env was built with trace=False")
        return self._tr.finish()

    def step(self, arbiter=None):
        """Advance one control second (plant ticks, KPM, xApps, arbiter, RIC apply)."""
        obs = self._propose(arbiter is not None)
        if arbiter is None:
            self.step_apply({"decisions": ["accept"] * len(self._pending[2]), "writes": []})
        else:
            self.step_apply(arbiter(obs))

    def step_propose(self):
        """Phase 1: plant ticks, KPM close/deliver, hidden xApp update, xApps observe/propose. Returns obs; the
        requests stay pending on the env until ``step_apply``."""
        return self._propose(True)

    def _propose(self, build_obs):
        if self._pending is not None:
            raise RuntimeError("step_propose called twice without step_apply")
        p = self.plant
        if self._static is None:
            self._static = self.static()
        for _ in range(C.TICKS_PER_CONTROL):
            p.tick()
        if self._tr is not None:
            self._tr.plant_second()
        self.sec += 1
        now = float(self.sec)
        self.kpm.second(self.sec)
        new = self.kpm.deliver(now)
        if self.update_at is not None and now >= self.update_at:
            self.xapps[self.update_xapp].update_version()
            self.update_at = None
        reqs = [d["req"] for d in self.deferred]
        for x in self.xapps:
            x.observe(new)
            if x.due(now):
                reqs.extend(x.propose(now))
        self._pending = (now, new, reqs)
        if not build_obs:
            return None
        return {"t": now, "new_reports": new, "config": self.config(), "requests": reqs, "static": self._static,
                "locked": {k: u for k, u in self.lock_until.items() if now < u},
                "changes": self.stats["changes"], "churn_cap": self.churn_cap}

    def step_apply(self, dec):
        """Phase 2: rollbacks, per-request decisions (ACK/NACK to xApps), writes, log -- for the pending second."""
        if self._pending is None:
            raise RuntimeError("step_apply called without a pending step_propose")
        if self.wg3 and dec.get("writes"):
            raise ValueError("wg3=True: free-form arbiter writes are not a WG3 action (use decisions/rollback)")
        now, new, reqs = self._pending
        self._pending = None
        p = self.plant
        tr = self._tr
        if tr is not None:
            tr.begin(now, new, reqs)
        for k in dec.get("rollback", ()):
            st = "noop"
            if k in self.prev_val and abs(self.prev_val[k] - knob_get(p, k)) > 1e-9:
                st = self._apply(k, self.prev_val[k], now)[0]
                if st == "ok":
                    self.stats["rollbacks"] += 1
            if tr is not None:
                tr.rollback(now, k, {"ok": "applied", "nack": "actuator"}.get(st, st))
        old_def = {id(d["req"]): d["age"] for d in self.deferred}
        self.deferred = []
        for r, d in zip(reqs, dec["decisions"], strict=True):
            self.stats["req"] += 1
            x = next(a for a in self.xapps if a.name == r["xapp"])
            d0, why = d, "reject"
            if self.locked(r["knob"], now):            # WG3 lock: force-reject whatever the decision
                d, why = "reject", "locked"
                self.stats["lock_blocked"] += 1
            elif isinstance(d, tuple) and d[0] == "lock":
                self.lock_until[r["knob"]] = now + float(d[1])
                self.stats["locks"] += 1
                d, why = "reject", "lock_set"
                if tr is not None:
                    tr.lock(now, r["knob"], self.lock_until[r["knob"]])
            if d == "defer":
                age = old_def.get(id(r), 0) + 1
                if age <= 10:
                    self.deferred.append({"req": r, "age": age})
                    self.stats["def"] += 1
                    if tr is not None:
                        tr.request(now, r, d0, "deferred", knob_get(p, r["knob"]), age - 1)
                    continue
                d, why = "reject", "expired"
            if d == "reject":
                self.stats["rej"] += 1
                x.result(r, False, knob_get(p, r["knob"]), now)
                if tr is not None:
                    tr.request(now, r, d0, why, knob_get(p, r["knob"]), old_def.get(id(r), 0))
                continue
            v = r["prop"] if d == "accept" else float(d[1])
            n_ch = self.stats["changes"]
            st, applied = self._apply(r["knob"], v, now)
            ok = st == "ok"
            self.stats["acc" if d == "accept" else "mod"] += int(ok)
            self.stats["rej"] += int(st == "churn")    # churn-cap NACK counts as a rejection
            x.result(r, ok, applied, now)
            if tr is not None:
                why = ("ok" if self.stats["changes"] > n_ch else "noop") if ok else \
                    ("churn" if st == "churn" else "actuator")
                tr.request(now, r, d0, why, applied, old_def.get(id(r), 0))
        for k, v in dec.get("writes", []):
            if self.writes_used >= self.write_budget * max(self.cfg.scored_s, 1) / 3600 + 1e-9:
                break
            st = self._apply(k, v, now)[0]
            if st == "ok":
                self.writes_used += 1
                self.stats["writes"] += 1
            if tr is not None:
                tr.write(now, k, v, {"nack": "actuator"}.get(st, st))
        if self.log_on:
            self.log.append({"t": now, "config": self.config(), "reports": new, "n_req": len(reqs), "requests": reqs,
                             "decisions": list(dec["decisions"]), "rollback": list(dec.get("rollback", ()))})
        if tr is not None:
            tr.end(now)

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
