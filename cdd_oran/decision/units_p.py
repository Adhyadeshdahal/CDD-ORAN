"""E6-P learned-arbiter decision units (scratchpad/e6_dev/decision/ARBITER_DESIGN.md sections 0-1).

Unit = (cell c, xApp x, t0). It OPENS at x's first request on c when no (c, x) unit is active, gets a mode from a
policy at that moment, and the mode holds for every request of x on c while t < t0 + T (T = 60 s); the next request
at or after t0 + T opens a new unit. Cell of a request = the knob's own cell ``knob[1]`` (every E6-P P1/P3 knob --
carrier, sleep, ptx, prot_min -- is per-cell).

Modes (base): accept | half | reject, plus a rollback flag ("+rb") for SliceGuarantee and PowerES only:
  accept  -> "accept"
  reject  -> "reject"
  half    -> the slew rule of cdd_oran.decision.plans.half_step with the E6-P quanta (ptx 1 dB, prot_min 0.05) =
             e6p_screen.half_step: a step of >= 2 quanta -> ("modify", cur + ceil(n/2) quanta toward prop); a
             single-quantum step -> alternate accept / reject per knob (per-knob phase = parity of the cell index)
  lock    -> ("lock", t0 + T - t): reject + lock the knob until the unit ends. Kept only to DECLARE lock == reject:
             in P1 / P3 every knob has exactly one writer (ES: carrier + sleep, PowerES: ptx, SliceGuarantee:
             prot_min), a WG3 lock only force-rejects requests on its own knob (env.py step_apply), so a lock held
             for the unit is plant-identical to reject (LOCK_EQUIV; tested in tests/test_e6p_units.py).
  +rb     -> at unit open, roll back x's knob on c if the arbiter SAW it change within the last RB_WINDOW = 60 s and
             that change was not our own rollback (rollback happens before the decisions, so the opening request on
             the same knob is then NACKed by the actuator's min interval; later requests of the unit get the base mode).

``UnitArbiter(policy)`` satisfies the env's arbiter contract ``arb(obs) -> {"decisions", "writes": [], "rollback"}``
and reads ONLY ``obs`` (the wg3 view): the static cell graph, the applied configuration, delivered KPM reports and the
requests. Knob change times are inferred from consecutive ``obs["config"]`` snapshots (a change applied in the
step_apply of second t-1 is visible in obs at t). ``policy(unit) -> (mode, propensity)``, where ``unit`` holds c, x,
x_idx, t0, the request and the obs-only context (``context``).

Exposure set N(c) = {c} + out-neighbours + in-neighbours of c in ``obs["static"]["neighbours"]`` (the neighbour lists
are asymmetric; the symmetrised set contains every pico's ES candidate macro, checked in the tests).
"""
from __future__ import annotations

import copy
from collections import deque

import numpy as np

from cdd_oran.envs.e6.config import E6PConfig
from cdd_oran.envs.e6.ric import LIMITS

from .plans import QUANTUM

XAPPS_P = ("ES", "PowerES", "SliceGuarantee", "Coverage")      # x_idx for RNG keys (stable across pairs)
X_KNOBS = {"ES": ("carrier", "sleep"), "PowerES": ("ptx",), "SliceGuarantee": ("prot_min",), "Coverage": ("ptx",)}
RB_XAPPS = ("SliceGuarantee", "PowerES")
BASE_MODES = ("accept", "half", "reject")
T_UNIT = 60.0
RB_WINDOW = 60.0
LOCK_EQUIV = {"P1": "reject", "P3": "reject"}                     # declared: lock == reject (one writer per knob)
_P = E6PConfig()
QUANTUM_P = dict(QUANTUM, ptx=_P.ptx_grid_db, prot_min=_P.prot_min_grid)
# "first" (design sec. 1): a unit opens at x's first request on c. "feasible" (OPTION, not the design default): a
# request the actuator must NACK because its knob is still inside the min interval (dwell) passes as accept (same
# NACK, same plant) without opening a unit, so units are not spent on dwell-blocked re-proposals.
OPEN_RULES = ("first", "feasible")
MIN_INTERVAL = {k: v[3] for k, v in LIMITS.items()} | {"ptx": _P.ptx_min_interval_s,
                                                       "prot_min": _P.prot_min_interval_s}

# obs-only mediators (design sec. 2: E6-P fast-report fields, ric.py KPM._make_p, plus the base fast report and the
# applied configuration). Report mediators are averaged over the last W_REPORTS delivered fast reports (nanmean).
REPORT_MEDIATORS = ("prb_util", "act_ue", "edge_sinr_p", "prot_dem_share", "prot_used_share", "prot_min_share",
                    "prot_thp", "prot_act_ue", "prot_below_frac")
CONFIG_MEDIATORS = ("carriers", "asleep", "ptx_db", "prot_min")
MEDIATORS = REPORT_MEDIATORS + CONFIG_MEDIATORS
W_REPORTS = 5
NBR_AGG = {"prb_util": ("mean", "max"), "act_ue": ("sum",), "edge_sinr_p": ("mean", "min"),
           "prot_dem_share": ("mean", "max"), "prot_used_share": ("mean",), "prot_min_share": ("mean",),
           "prot_thp": ("sum",), "prot_act_ue": ("sum",), "prot_below_frac": ("mean", "max"),
           "carriers": ("mean",), "asleep": ("sum",), "ptx_db": ("mean",), "prot_min": ("mean",)}


def split_mode(mode: str) -> tuple[str, bool]:
    """"half+rb" -> ("half", True); "reject" -> ("reject", False)."""
    base, _, rb = mode.partition("+")
    if base not in BASE_MODES + ("lock",) or rb not in ("", "rb"):
        raise ValueError(f"unknown mode {mode!r}")
    return base, rb == "rb"


def x_index(x: str) -> int:
    return XAPPS_P.index(x)


def half_step_p(knob, cur: float, prop: float, hs: dict):
    """Slew-rule "half" with the E6-P quanta (identical to scratchpad/e6_dev/e6p_screen.half_step)."""
    q = QUANTUM_P.get(knob[0])
    n = abs(prop - cur) / q if q else 0.0
    if n >= 2 - 1e-6:
        k = int(np.ceil(round(n / 2, 6)))
        return ("modify", float(cur + np.sign(prop - cur) * k * q))
    c = hs.get(knob, 0)
    hs[knob] = c + 1
    phase = sum(int(x) for x in knob[1:]) % 2
    return "accept" if (c + phase) % 2 == 0 else "reject"


def exposure_sets(static) -> dict:
    """cell -> sorted tuple N(c) = {c} + out- and in-neighbours (obs-only: ``obs["static"]["neighbours"]``)."""
    nb = static["neighbours"]
    n = int(static["cells"])
    out = {c: {c} | {int(v) for v in nb[c]} for c in range(n)}
    for c in range(n):
        for v in nb[c]:
            out[int(v)].add(c)
    return {c: tuple(sorted(s)) for c, s in out.items()}


def accept_all_policy(unit):
    return "accept", 1.0


class HoldPolicy:
    """Label / probe policy: ``mode`` for units of the (c, x) pairs in ``keys`` that open in [t_lo, t_hi); accept
    (propensity 1) for every other unit."""

    def __init__(self, keys, mode, t_lo, t_hi):
        self.keys, self.mode = frozenset(keys), mode
        self.t_lo, self.t_hi = float(t_lo), float(t_hi)

    def __call__(self, unit):
        if (unit["c"], unit["x"]) in self.keys and self.t_lo <= unit["t0"] < self.t_hi:
            return self.mode, 1.0
        return "accept", 1.0


class UnitArbiter:
    """Obs-only WG3 arbiter over (cell, xApp, t0) units; see the module docstring.

    ``warmup_s``: before it every request is accepted and no unit opens (the protocol's warm-up; a config constant,
    not plant state). ``record``: keep every opened unit in ``self.units`` (the collector's log). ``self.opened`` =
    units opened by the latest call (the collector hooks labels on it)."""

    def __init__(self, policy, T: float = T_UNIT, warmup_s: float = 0.0, record: bool = True,
                 open_rule: str = "first"):
        if open_rule not in OPEN_RULES:
            raise ValueError(f"open_rule in {OPEN_RULES}")
        self.policy, self.T, self.warmup_s, self.record = policy, float(T), float(warmup_s), record
        self.open_rule = open_rule
        self.active = {}                  # (c, x) -> unit dict
        self.units, self.opened = [], []
        self.hs = {}                      # half_step_p toggle state (knob -> single-quantum half requests seen)
        self.last_chg = {}                # knob -> time of its latest change seen in obs["config"] (apply second)
        self.rb_at = {}                   # knob -> change time created by our own rollback
        self.prev_cfg, self.prev_t, self.prev_rb = None, None, ()
        self.fast = deque(maxlen=W_REPORTS)
        self.exp = None
        self.static = None
        self._med_t, self._med = None, None

    # ---------------------------------------------------------------------------------------- observation state
    def _observe(self, obs):
        cfg = obs["config"]
        if self.prev_cfg is not None:
            for k, v in cfg.items():
                if abs(v - self.prev_cfg.get(k, v)) > 1e-9:
                    self.last_chg[k] = self.prev_t
                    if k in self.prev_rb:
                        self.rb_at[k] = self.prev_t
        self.prev_cfg, self.prev_t = cfg, obs["t"]
        for rep in obs["new_reports"]:
            if rep["gran"] == "fast":
                self.fast.append(rep)
        if self.static is None:
            self.static = obs["static"]
            self.exp = exposure_sets(self.static)

    def mediators(self, obs) -> np.ndarray:
        """(cells, len(MEDIATORS)) obs-only mediator matrix at this second (NaN = no data)."""
        if self._med_t == obs["t"]:
            return self._med
        n = int(self.static["cells"])
        M = np.full((n, len(MEDIATORS)), np.nan)
        if self.fast:
            import warnings
            for j, f in enumerate(REPORT_MEDIATORS):
                vals = [np.asarray(r[f], float) for r in self.fast if f in r]
                if not vals:
                    continue
                vals = [v.sum(1) if v.ndim == 2 else v for v in vals]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    M[:, j] = np.nanmean(np.stack(vals), 0)
        cfg, macro = obs["config"], np.asarray(self.static["is_macro"], bool)
        j0 = len(REPORT_MEDIATORS)
        for c in range(n):
            M[c, j0] = cfg.get(("carrier", c), np.nan) if macro[c] else np.nan
            M[c, j0 + 1] = cfg.get(("sleep", c), 0.0) if not macro[c] else 0.0
            M[c, j0 + 2] = cfg.get(("ptx", c), 0.0)
            M[c, j0 + 3] = cfg.get(("prot_min", c), 0.0)
        self._med_t, self._med = obs["t"], M
        return M

    def context(self, c, r, obs) -> dict:
        """Obs-only context of a unit opened by request ``r`` on cell ``c``: the request, c's mediators, and
        aggregates over N(c) minus c."""
        import warnings
        M = self.mediators(obs)
        k = r["knob"]
        t = obs["t"]
        ctx = {"knob": k[0], "is_macro": float(bool(self.static["is_macro"][c])), "cur": float(r["cur"]),
               "prop": float(r["prop"]), "step": float(r["prop"] - r["cur"]),
               "since_change": float(min(t - self.last_chg[k], 1e4)) if k in self.last_chg else 1e4,
               "n_exp": len(self.exp[c])}
        nb = [v for v in self.exp[c] if v != c]
        for j, f in enumerate(MEDIATORS):
            ctx["own_" + f] = float(M[c, j])
            col = M[nb, j] if nb else np.full(1, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                for a in NBR_AGG[f]:
                    fn = {"mean": np.nanmean, "max": np.nanmax, "min": np.nanmin, "sum": np.nansum}[a]
                    ctx[f"nbr_{a}_{f}"] = float(fn(col)) if np.isfinite(col).any() else float("nan")
        return ctx

    def dwell_blocked(self, k, now) -> bool:
        """The knob changed less than its actuator min interval ago (public ric limits; change time from obs)."""
        t = self.last_chg.get(k)
        return t is not None and now - t < MIN_INTERVAL[k[0]]

    # ---------------------------------------------------------------------------------------- contract
    def __call__(self, obs):
        self._observe(obs)
        now = obs["t"]
        reqs = obs["requests"]
        self.opened = []
        if now < self.warmup_s:
            self.prev_rb = ()
            return {"decisions": ["accept"] * len(reqs), "writes": [], "rollback": []}
        for key in [key for key, u in self.active.items() if now >= u["t0"] + self.T]:
            del self.active[key]
        dec, rb = [], []
        for r in reqs:
            x, k = r["xapp"], r["knob"]
            c = int(k[1])
            u = self.active.get((c, x))
            if u is None and self.open_rule == "feasible" and self.dwell_blocked(k, now):
                dec.append("accept")                        # the actuator NACKs it whatever we decide: no unit
                continue
            if u is None:
                u = {"c": c, "x": x, "x_idx": x_index(x), "t0": float(now), "knob": k[0],
                     "exp": list(self.exp[c]), "ctx": self.context(c, r, obs)}
                mode, p = self.policy(u)
                base, want_rb = split_mode(mode)
                if want_rb and x not in RB_XAPPS:
                    raise ValueError(f"rollback flag not allowed for {x}")
                u.update(mode=mode, p=float(p), n_req=0, n_changed=0)
                self.active[(c, x)] = u
                self.opened.append(u)
                if self.record:
                    self.units.append(u)
                if want_rb:                                 # x's own knob on c (SG: prot_min, PowerES: ptx)
                    t_ch = self.last_chg.get(k)
                    if t_ch is not None and now - t_ch <= RB_WINDOW and self.rb_at.get(k) != t_ch and k not in rb:
                        rb.append(k)
                        u["rb_issued"] = True
            base, _ = split_mode(u["mode"])
            u["n_req"] += 1
            if base == "accept":
                d = "accept"
            elif base == "reject":
                d = "reject"
            elif base == "half":
                d = half_step_p(k, r["cur"], r["prop"], self.hs)
            else:                                           # lock: reject + lock the knob until the unit ends
                d = ("lock", float(max(u["t0"] + self.T - now, 1e-6)))
            dec.append(d)
        self.prev_rb = tuple(rb)
        return {"decisions": dec, "writes": [], "rollback": rb}

    def fork(self, policy, record: bool = False) -> UnitArbiter:
        """Independent copy of the observation state (config / change history, report window, half toggles) with a
        new policy and NO active units (every unit re-opens under ``policy``). For lookahead rollouts: take it from a
        snapshot made before this second's call, then call it on the same obs."""
        pol, self.policy = self.policy, None
        units, opened, active = self.units, self.opened, self.active
        self.units, self.opened, self.active = [], [], {}
        try:
            c = copy.deepcopy(self)
        finally:
            self.policy, self.units, self.opened, self.active = pol, units, opened, active
        c.policy, c.record = policy, record
        return c


__all__ = ["BASE_MODES", "LOCK_EQUIV", "MEDIATORS", "OPEN_RULES", "QUANTUM_P", "RB_XAPPS", "T_UNIT", "XAPPS_P", "X_KNOBS",
           "HoldPolicy", "UnitArbiter", "accept_all_policy", "exposure_sets", "half_step_p", "split_mode", "x_index"]
