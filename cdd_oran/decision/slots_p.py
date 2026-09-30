"""E6-P FIXED-SLOT decision units (MSCR+ agent V, 2026-09-30; an OPTION for fresh collection, not the frozen design;
units_p.py is unchanged).

Why. In units_p a unit opens at x's first request on c when no (c, x) unit is active, so WHICH units exist (and their
t0, request direction sgn, strata) depends on the family's own past modes: a reject leaves the request pending ->
a re-proposal unit ~60 s later (DEV, 20 eps: a same-direction successor within 70 s follows 69-76 % of carrier /
sleep rejects and < 0.5 % of applied modes; 99-100 % of such units were rejects), an accept moves the knob and starts
the dwell. The CRT re-draws modes i.i.d. on that realised unit set: not exactly valid (.tmp/mscr_plus/V).

Design. Time after the warm-up is cut into fixed slots of T = 60 s: slot k = [slot0 + k T, slot0 + (k + 1) T). For
every (cell c, xApp x) pair that owns a knob on c (static ownership, ``owner_pairs``) and every slot k there is ONE
unit u = (c, x, k) whose mode is drawn from pi0 with the key [seed, 6612, c, x_idx, t0 = slot start] (the collect_p
key, so the draw is a function of (c, x, k) only: NOT of whether / when a request arrives). The mode holds for every
request of x on c inside the slot. A slot without a request is still a unit (state "no request"; its mode is a no-op
on the plant). The unit record carries t0 = the slot start (the analysis window starts there: an intention-to-treat
window), t_req = the first request's second (None if none), and the request / ctx of that first request.

What this buys and what remains mode-dependent.
  * The unit SET {(c, x, k)} and the assignment W (i.i.d. pi0 per unit, independent of everything) are fixed by
    design: a CRT that re-draws W i.i.d. on all slots is EXACTLY the design law. It is exactly valid for any statistic
    built from W, the null-fixed outcomes Y and FIXED slot attributes (c, x, k, time, layout, side information from
    other data).
  * Still mode-dependent (functions of the family's PAST modes): whether a request is pending in slot k, its direction
    sgn_k, its size, the obs ctx at the request, n_req. Defining the unit as the slot does NOT make these fixed: a CRT
    that holds the realised sgn_k / request flag fixed while re-drawing W is again only asymptotically valid. Two exact
    options: (a) orient by a FIXED direction prior h_{c,x,k} (e.g. P(sleep request | cell, time) from OLD data) and
    use v_k = h (L(W_k) - m) with an i.i.d. CRT (exact; power loss where h mispredicts; request-free slots add noise
    only); (b) the design-based e-process (eprocess_units) with v_k = sgn_k (L(W_k) - m), sgn_k := 0 for a slot
    without request (the factor is then 1): sgn_k is F_{k-1}-measurable, so it is exact for the realised sgn on
    either unit definition. The slot grid adds nothing to (b)'s validity; it makes (a) and the i.i.d. CRT exact and
    removes the re-proposal cluster (one decision per 60 s whatever the previous mode).
  * pi0 accept .5 / reject .5 (no half; half ~ accept, applied 88-100 %): Var(L) .21 -> .25 (z x 1.09), |v| = .5 for
    every unit (symmetric; the e-process may bet up to |lambda| < 2 instead of 1.43). Costs: rejects rise .3 -> .5, so
    more pending requests / re-proposals in units_p (not in slots) and a plant regime further from the operating one.

``SlotArbiter(policy)`` has the units_p arbiter contract (obs-only; rollback modes not supported). ``slot_table(policy,
n_slots)`` lists EVERY slot unit (with or without a request) with its mode from the same key.
"""
from __future__ import annotations

from .units_p import T_UNIT, X_KNOBS, XAPPS_P, UnitArbiter, half_step_p, split_mode, x_index

KNOB_OWNER_PRIORITY = {"carrier": ("ES",), "sleep": ("ES",), "prot_min": ("SliceGuarantee",),
                       "ptx": ("PowerES", "Coverage")}


def owner_pairs(static) -> list:
    """Sorted [(c, x)] of static knob ownership: every knob (kind, c) in ``static["knobs"]`` -> the first xApp of
    KNOB_OWNER_PRIORITY[kind] present in ``static["xapps"]`` (independent of any mode)."""
    xs = set(static.get("xapps", {}) or XAPPS_P)
    out = set()
    for kind, c in static["knobs"]:
        for x in KNOB_OWNER_PRIORITY.get(kind, ()):
            if x in xs and kind in X_KNOBS[x]:
                out.add((int(c), x))
                break
    return sorted(out)


class SlotArbiter(UnitArbiter):
    """Fixed-slot units (module docstring). ``slot0`` defaults to ``warmup_s``."""

    def __init__(self, policy, T: float = T_UNIT, warmup_s: float = 0.0, record: bool = True,
                 slot0: float | None = None):
        super().__init__(policy, T=T, warmup_s=warmup_s, record=record, open_rule="first")
        self.slot0 = float(warmup_s if slot0 is None else slot0)

    def slot_of(self, now: float) -> int:
        return int((float(now) - self.slot0) // self.T)

    def slot_start(self, k: int) -> float:
        return self.slot0 + int(k) * self.T

    def __call__(self, obs):
        self._observe(obs)
        now = obs["t"]
        reqs = obs["requests"]
        self.opened = []
        self.prev_rb = ()
        if now < self.warmup_s or now < self.slot0:
            return {"decisions": ["accept"] * len(reqs), "writes": [], "rollback": []}
        k = self.slot_of(now)
        for key in [key for key, u in self.active.items() if u["slot"] != k]:
            del self.active[key]
        dec = []
        for r in reqs:
            x, kn = r["xapp"], r["knob"]
            c = int(kn[1])
            u = self.active.get((c, x))
            if u is None:
                u = {"c": c, "x": x, "x_idx": x_index(x), "t0": float(self.slot_start(k)), "slot": k,
                     "t_req": float(now), "knob": kn[0], "exp": list(self.exp[c]), "ctx": self.context(c, r, obs)}
                mode, p = self.policy(u)
                base, want_rb = split_mode(mode)
                if want_rb:
                    raise ValueError("rollback modes are not supported by SlotArbiter")
                u.update(mode=mode, p=float(p), n_req=0, n_changed=0)
                self.active[(c, x)] = u
                self.opened.append(u)
                if self.record:
                    self.units.append(u)
            base, _ = split_mode(u["mode"])
            u["n_req"] += 1
            if base == "half":
                dec.append(half_step_p(kn, r["cur"], r["prop"], self.hs))
            elif base == "lock":
                dec.append(("lock", float(max(u["t0"] + self.T - now, 1e-6))))
            else:
                dec.append(base)
        return {"decisions": dec, "writes": [], "rollback": []}

    def slot_table(self, policy, n_slots: int) -> list:
        """EVERY slot unit (c, x, k) for k < n_slots over ``owner_pairs``: {"c", "x", "slot", "t0", "mode", "p",
        "request": bool, "unit": the recorded unit or None}. Modes of request-free slots are drawn with the same key
        (policy called on a bare unit dict), so the table is the full, mode-independent unit set."""
        if self.static is None:
            raise RuntimeError("no observation seen yet")
        got = {(u["c"], u["x"], u["slot"]): u for u in self.units}
        out = []
        for c, x in owner_pairs(self.static):
            for k in range(int(n_slots)):
                u = got.get((c, x, k))
                if u is None:
                    m, p = policy({"c": c, "x": x, "x_idx": x_index(x), "t0": float(self.slot_start(k)), "slot": k})
                    out.append({"c": c, "x": x, "slot": k, "t0": self.slot_start(k), "mode": m, "p": float(p),
                                "request": False, "unit": None})
                else:
                    out.append({"c": c, "x": x, "slot": k, "t0": u["t0"], "mode": u["mode"], "p": u["p"],
                                "request": True, "unit": u})
        return out


__all__ = ["KNOB_OWNER_PRIORITY", "SlotArbiter", "owner_pairs"]
