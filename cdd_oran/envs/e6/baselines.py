"""E6 mandatory simple baselines (arbiters that use only ``obs``).

  no arbitration  : ``arbiter=None`` in ``E6Env.run`` (accept all, last writer wins)
  freeze          : reject every request (network stays at its initial 3GPP-default configuration)
  priority        : per knob and second, keep only the request of the highest-priority xApp; accept the rest
  knob_lock       : SON-coordination style lease: the first xApp to change a knob owns it for ``lease_s``; requests
                    by other xApps on an owned knob are rejected
  subset(keep)    : reject every request from xApps not in ``keep`` (= running only ``keep``, all xApps deployed)
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


BASELINES = {"noarb": lambda: None, "freeze": lambda: freeze, "priority": lambda: priority, "lock": KnobLock}


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
