"""E6 mandatory simple baselines (arbiters that use only ``obs``).

  no arbitration  : ``arbiter=None`` in ``E6Env.run`` (accept all, last writer wins)
  freeze          : reject every request (network stays at its initial 3GPP-default configuration)
  priority        : per knob and second, keep only the request of the highest-priority xApp; accept the rest
  knob_lock       : SON-coordination style lease: the first xApp to change a knob owns it for ``lease_s``; requests
                    by other xApps on an owned knob are rejected
  subset(keep)    : reject every request from xApps not in ``keep`` (= running only ``keep``, all xApps deployed)
  cell_lock       : ``CellPriorityLock(order)``: a cell changed by a higher-priority xApp in the last 60 s rejects
                    lower-priority requests on any knob of that cell (E6-P arm 6)
  region subset   : ``region_subset(keep_by_region, site)``; ``region_hindsight`` = HINDSIGHT per-region static
                    subset by one coordinate-descent pass (E6-P arm 9, not deployable)
  static config   : ``apply_static(env, spec)`` before the run + freeze; ``tuned_static`` = HINDSIGHT coordinate
                    descent over ``tuned_static_grid()`` (upper reference for static control, not deployable)
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .ric import LIMITS, _quantise, knob_set

PRIORITY = ("SLICE", "MRO", "TS", "ES")


def freeze(obs):
    return {"decisions": ["reject"] * len(obs["requests"]), "writes": []}


def priority(obs, order=PRIORITY):
    rank = {x: i for i, x in enumerate(order)}
    best = {}
    for i, r in enumerate(obs["requests"]):
        k = r["knob"]
        if k not in best or rank.get(r["xapp"], 99) < rank.get(obs["requests"][best[k]]["xapp"], 99):
            best[k] = i
    keep = set(best.values())
    return {"decisions": ["accept" if i in keep else "reject" for i in range(len(obs["requests"]))], "writes": []}


class KnobLock:
    def __init__(self, lease_s=60.0):
        self.lease_s, self.owner = lease_s, {}

    def __call__(self, obs):
        now, dec = obs["t"], []
        for r in obs["requests"]:
            k = r["knob"]
            own = self.owner.get(k)
            if own is None or own[1] < now or own[0] == r["xapp"]:
                self.owner[k] = (r["xapp"], now + self.lease_s)
                dec.append("accept")
            else:
                dec.append("reject")
        return {"decisions": dec, "writes": []}


def knob_cells(k):
    """Cells a knob acts on: per-cell knobs (carrier / ptx / prot_min / sleep / hys / ttt / ll_ratio) -> (c,),
    CIO[s, n] -> (s, n)."""
    return (k[1], k[2]) if k[0] == "cio" else (k[1],)


class CellPriorityLock:
    """Cell-priority lock (E6P_SCREEN_PROTOCOL sec. 5 arm 6; NIST ns3-oran ``SingleCommandPerNode``-like CMM): a cell
    whose knob was CHANGED by an xApp of priority p in the last ``hold_s`` s rejects requests of every lower-priority
    xApp on any knob of that cell. Same-second requests are processed in priority order, so a higher-priority change
    in this second blocks lower-priority requests on its cells at once; same-rank requests never block each other.

    "Changed" = an accepted request whose value actually took effect: an accepted request is provisional until the
    next ``obs["config"]`` shows the knob moved away from the request's ``cur`` (an actuator NACK or a no-op leaves no
    hold). Uses only ``obs``. ``order`` = xApp names, highest priority first (unknown xApps rank last)."""

    def __init__(self, order, hold_s=60.0):
        self.rank = {x: i for i, x in enumerate(order)}
        self.hold_s = float(hold_s)
        self.hold = {}                    # cell -> (rank of the holder, time of its latest effective change)
        self.pending = []                 # (knob, cur, rank, t) accepted last second, unconfirmed

    def _mark(self, cell, rk, t):
        h = self.hold.get(cell)
        if h is None or h[1] + self.hold_s <= t or rk <= h[0]:
            self.hold[cell] = (rk, t)

    def __call__(self, obs):
        now, cfg = obs["t"], obs["config"]
        for k, cur, rk, t in self.pending:                            # confirm last second's accepted changes
            if abs(float(cfg.get(k, cur)) - cur) > 1e-9:
                for c in knob_cells(k):
                    self._mark(c, rk, t)
        self.pending = []
        reqs = obs["requests"]
        dec = [None] * len(reqs)
        now_hold = {}                                                 # cell -> best rank accepted this second
        for i in sorted(range(len(reqs)), key=lambda j: self.rank.get(reqs[j]["xapp"], 99)):
            r = reqs[i]
            rk = self.rank.get(r["xapp"], 99)
            blocked = False
            for c in knob_cells(r["knob"]):
                h = self.hold.get(c)
                if h is not None and now - h[1] < self.hold_s and h[0] < rk:
                    blocked = True
                if c in now_hold and now_hold[c] < rk:
                    blocked = True
            if blocked:
                dec[i] = "reject"
                continue
            dec[i] = "accept"
            if abs(r["prop"] - r["cur"]) > 1e-9:
                self.pending.append((r["knob"], float(r["cur"]), rk, now))
                for c in knob_cells(r["knob"]):
                    now_hold[c] = min(now_hold.get(c, 99), rk)
        return {"decisions": dec, "writes": []}


BASELINES = {"noarb": lambda: None, "freeze": lambda: freeze, "priority": lambda: priority, "lock": KnobLock}


# ------------------------------------------------------------------------ per-region subset (E6-P arm 9)
def region_subset(keep_by_region, site):
    """Arbiter that accepts a request iff its xApp is in ``keep_by_region[region(knob)]`` (region = ``site`` of the
    knob's own cell k[1], as ``decision.adapters.e6.knob_region``); rejects it otherwise. A fixed per-region xApp
    subset held for the whole episode (all xApps stay deployed and proposing)."""
    keep = {int(g): frozenset(v) for g, v in keep_by_region.items()}
    site = np.asarray(site)

    def arb(obs):
        return {"decisions": ["accept" if r["xapp"] in keep[int(site[r["knob"][1]])] else "reject"
                              for r in obs["requests"]], "writes": []}

    arb.keep = keep
    return arb


def all_subsets(xapps):
    """Every subset of ``xapps`` (tuples in ``xapps`` order), empty first, by size then lexicographic mask order."""
    import itertools
    out = [tuple(x for x, m in zip(xapps, mask, strict=True) if m)
           for mask in itertools.product((0, 1), repeat=len(xapps))]
    return sorted(out, key=len)


def region_hindsight(evaluate, regions, xapps, all_regions=None):
    """HINDSIGHT per-region static subset (E6P_SCREEN_PROTOCOL sec. 5 arm 9; privileged, not deployable).

    Start from accept-all (every region keeps every xApp), then ONE pass of coordinate descent over ``regions`` in
    index order: at region g every other subset of ``xapps`` is evaluated with the other regions held, and the region
    moves to the best ADMISSIBLE candidate if it beats the incumbent. ``evaluate(keep_by_region) -> (value,
    admissible)`` (e.g. PSVR of a full episode on the scored seed, admissible = energy-matched and guardrail-passing).
    A candidate beats the incumbent iff it is admissible and (the incumbent is not, or its value is strictly lower);
    ties keep the incumbent, ties among candidates keep the earlier one in ``all_subsets`` order. Evaluations are
    cached. ``all_regions`` (default ``regions``) = every region of the plant (regions outside ``regions`` keep every
    xApp). Returns {"best", "best_val", "best_ok", "path": [{"keep", "val", "ok"}], "n_eval"}."""
    regions = [int(g) for g in sorted(regions)]
    cache, path = {}, []

    def ev(kb):
        key = tuple(sorted((g, tuple(v)) for g, v in kb.items()))
        if key not in cache:
            val, ok = evaluate({g: tuple(v) for g, v in kb.items()})
            cache[key] = (float(val), bool(ok))
            path.append({"keep": {g: tuple(v) for g, v in kb.items()}, "val": cache[key][0], "ok": cache[key][1]})
        return cache[key]

    def better(a, b):
        return a[1] and (not b[1] or a[0] < b[0])

    full = tuple(xapps)
    best = {int(g): full for g in (all_regions if all_regions is not None else regions)}
    cur = ev(best)
    for g in regions:
        win, wv = None, cur
        for s in all_subsets(xapps):
            if s == best[g]:
                continue
            v = ev({**best, g: s})
            if better(v, wv):
                win, wv = s, v
        if win is not None:
            best, cur = {**best, g: win}, wv
    return {"best": best, "best_val": cur[0], "best_ok": cur[1], "path": path, "n_eval": len(cache)}


# ------------------------------------------------------------------------------------------ subset of the xApps
def subset(keep):
    """Arbiter that rejects every request from xApps not in ``keep`` (all xApps stay deployed and keep proposing).

    The plant only changes through accepted requests, so this reproduces running ``keep`` alone (a rejected xApp's
    internal NACK state never reaches the plant). ``keep`` = all deployed -> accept-all (no arbitration);
    ``keep`` = () -> freeze."""
    keep = frozenset(keep)

    def arb(obs):
        return {"decisions": ["accept" if r["xapp"] in keep else "reject" for r in obs["requests"]], "writes": []}

    arb.keep = keep
    return arb


# ------------------------------------------------------------------------------------ static configuration
# Network-wide static settings applied once before the run (combine with ``freeze`` for a "static configuration"
# policy). Keys (all optional; missing keys keep the plant's 3GPP-default initial value):
#   cio_to_pico : dB, cell range expansion: CIO[macro -> pico] = +x and CIO[pico -> macro] = -x for every
#                 macro/pico neighbour pair (symmetric so the bias does not induce ping-pong)
#   hys         : dB, A3 hysteresis on every cell
#   ttt         : ms, time-to-trigger on every cell (one of config.TTT_SET_MS)
#   ll_ratio    : LL dedicated PRB fraction on every cell
#   carriers    : "all" | "one" | int, active component carriers on every macro sector
STATIC_DEFAULT = {"cio_to_pico": 0.0, "hys": 2.0, "ttt": 320, "ll_ratio": 0.0, "carriers": "all"}


def _check(key, v):
    typ = {"cio_to_pico": "cio"}.get(key, key)
    lo, hi = LIMITS[typ][:2]
    if not lo <= v <= hi:
        raise ValueError(f"static {key}={v} outside actuator range [{lo}, {hi}]")
    if key == "ttt" and int(v) not in C.TTT_SET_MS:
        raise ValueError(f"static ttt={v} not in {C.TTT_SET_MS}")
    if abs(_quantise((typ,), v) - v) > 1e-9:
        raise ValueError(f"static {key}={v} not on the actuator grid")


def apply_static(env, spec):
    """Set a static network-wide configuration on ``env.plant`` via ``ric.knob_set`` BEFORE ``env.run`` (t = 0).

    Values are validated against ``ric.LIMITS`` (range + quantisation) and ``config.TTT_SET_MS``; the per-change step
    and dwell limits do not apply to the initial configuration. Returns the list of (knob, value) set."""
    if env.sec != 0:
        raise RuntimeError("apply_static must be called before the first control second")
    bad = set(spec) - set(STATIC_DEFAULT)
    if bad:
        raise KeyError(f"unknown static keys {sorted(bad)}")
    p = env.plant
    lay = p.lay
    out = []
    for key, v in spec.items():
        if key == "carriers":
            n = {"all": C.MACRO_NTRX, "one": 1}.get(v, v)
            if not isinstance(n, (int, np.integer)) or not 1 <= n <= C.MACRO_NTRX:
                raise ValueError(f"static carriers={v!r} must be 'all', 'one' or 1..{C.MACRO_NTRX}")
            out += [(("carrier", c), float(n)) for c in range(lay.n_cells) if lay.is_macro[c]]
            continue
        v = float(v)
        _check(key, v)
        if key == "cio_to_pico":
            for m in range(lay.n_cells):
                if not lay.is_macro[m]:
                    continue
                for q in lay.neighbours[m]:
                    if not lay.is_macro[q]:
                        out += [(("cio", m, q), v), (("cio", q, m), -v)]
        else:
            typ = key
            out += [((typ, c), v) for c in range(lay.n_cells)]
    for k, v in out:
        knob_set(p, k, v, 0.0)
    return out


def static_policy(env, spec):
    """``apply_static`` + the freeze arbiter: the network runs the whole episode at ``spec``."""
    apply_static(env, spec)
    return freeze


def tuned_static_grid():
    """Small coordinate-descent search space (3 values per numeric dimension, all inside ``ric.LIMITS``); the first
    value of each dimension is the 3GPP-default initial configuration (= freeze)."""
    return {"cio_to_pico": (0.0, 3.0, 6.0),
            "hys": (2.0, 1.0, 4.0),
            "ttt": (320, 160, 640),
            "ll_ratio": (0.0, 0.1, 0.2),
            "carriers": ("all", "one")}


def tuned_static(evaluate, grid=None, max_passes=2):
    """HINDSIGHT baseline: coordinate descent over ``grid`` (default ``tuned_static_grid()``) minimising
    ``evaluate(spec) -> float`` (e.g. SVR of a full static-config episode on the SAME seed/tape that is being scored).

    Chosen per seed on the evaluation tape itself, so it is generous to static control: an oracle-ish UPPER reference
    for what any fixed configuration can achieve, not a deployable method. Starts at the default configuration,
    sweeps the dimensions in order, keeps a change only if it strictly lowers the objective, and stops after
    ``max_passes`` passes or a pass without improvement. Evaluations are cached (each spec is run once).
    Returns {"best": spec, "best_val": float, "path": [{"spec", "val", "improved"}...], "n_eval": int}."""
    grid = grid or tuned_static_grid()
    cache, path = {}, []

    def ev(spec):
        key = tuple(sorted(spec.items()))
        if key not in cache:
            cache[key] = float(evaluate(dict(spec)))
            path.append({"spec": dict(spec), "val": cache[key]})
        return cache[key]

    best = {k: vals[0] for k, vals in grid.items()}
    best_val = ev(best)
    path[-1]["improved"] = True
    for _ in range(max_passes):
        improved = False
        for k, vals in grid.items():
            for v in vals:
                if v == best[k]:
                    continue
                cand = dict(best, **{k: v})
                n = len(path)
                val = ev(cand)
                if len(path) > n:
                    path[-1]["improved"] = val < best_val
                if val < best_val:
                    best, best_val, improved = cand, val, True
        if not improved:
            break
    return {"best": best, "best_val": best_val, "path": path, "n_eval": len(cache)}


# ------------------------------------------------------------------------------ SMO restore (stress scenario S2)
# Declared a priori (E6_STRESS_SCENARIOS.md sec. 4.1), not tuned: the SMO uses the MRO function's own nominal too-late
# target (xapps.MRO: 2 % of HO attempts + too-late) aggregated over the rollout cluster, sustained over MRO's 60 s window
# (2 consecutive 30 s mobility reports), with MRO's minimum-evidence rule (>= 2 too-late and >= 3 events).
SMO_TL_RATIO = 0.02
SMO_N_REPORTS = 2
SMO_MIN_TL, SMO_MIN_EVENTS = 2, 3


class SMORestore:
    """Alarm-triggered SMO restore: a reference OUTSIDE the near-RT RIC (post-rollout verification with fallback).

    Wraps an arbiter ``inner`` (None = accept all) and passes its decisions through unchanged. After the S2 rollout
    it watches the delivered mobility KPM reports whose window starts at/after the push; when the cluster too-late
    ratio tl / (ho_att + tl) exceeds ``ratio`` on ``n_reports`` consecutive reports (with the minimum evidence), it
    restores the pre-rollout Hys/TTT on the whole cluster once, as an OAM write (no RIC request, no churn; recorded
    as the knobs' last-known-good). Privileged only in what the SMO itself knows: which cells it pushed and their
    previous values. No-op outside ``scenario="mistune"``."""

    def __init__(self, env, inner=None, ratio=SMO_TL_RATIO, n_reports=SMO_N_REPORTS):
        self.env, self.inner, self.ratio, self.n_reports = env, inner, ratio, n_reports
        self.streak, self.fired_t = 0, None

    def alarm(self, rep, scn):
        """Update the streak with one mobility report; returns True when the restore should fire."""
        cl = scn.cluster
        tl = float(rep["too_late"][cl].sum())
        den = float(rep["ho_att"][cl].sum()) + tl
        bad = tl >= SMO_MIN_TL and den >= SMO_MIN_EVENTS and tl / den > self.ratio
        self.streak = self.streak + 1 if bad else 0
        return self.streak >= self.n_reports

    def __call__(self, obs):
        dec = self.inner(obs) if self.inner is not None else             {"decisions": ["accept"] * len(obs["requests"]), "writes": []}
        p = self.env.plant
        scn = p.scn
        if self.fired_t is None and getattr(scn, "push_t", None) is not None:
            for rep in obs["new_reports"]:
                if rep["gran"] == "mob" and rep["t0"] >= scn.push_t - 1e-9 and self.alarm(rep, scn):
                    scn.oam_set(p, scn.pre_hys, scn.pre_ttt, obs["t"], "restore")
                    self.fired_t = obs["t"]
                    break
        return dec
