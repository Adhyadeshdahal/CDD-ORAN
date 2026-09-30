"""E6-P randomized unit collection (scratchpad/e6_dev/decision/ARBITER_DESIGN.md section 3).

Logging policy pi0 (context-free), two regimes alternating by episode index j (even j = high-eps, odd j = low-eps):

  | regime   | ES / PowerES                        | SliceGuarantee                                      |
  | high-eps | accept 0.5, half 0.2, reject 0.3    | accept 0.5, half 0.15, reject 0.25, accept+rb 0.10  |
  | low-eps  | accept 0.8, the other modes scaled by 0.2 / 0.5, then every non-accept mode floored at P_MIN = 0.1
               (accept absorbs the floor): ES / PowerES accept 0.78, half 0.10, reject 0.12; SliceGuarantee accept
               0.70, half 0.10, reject 0.10, accept+rb 0.10                                                     |

Draws: ``np.random.default_rng([seed, 6612, c, x_idx, int(t0)])`` (one uniform per unit, modes in PI0_ORDER), so a
unit re-opening at the same second inside an ``env.copy`` rollout draws the same mode. Warm-up: every request is
accepted and no unit opens (``UnitArbiter.warmup_s``).

Realised KPIs (logged, never seen by the arbiter): ``CellKPITap`` wraps ``plant.take_counters`` (called once per
second by the KPM right after the second's last tick, so ``serv`` and the per-UE violation flags are exactly those of
the second's close) and accumulates per cell, over scored seconds only: protected violated UE-s (``pv``), all-UE
violated UE-s (``v``) and energy J (``e``); their network sums equal ``plant.sla`` prot_viol / viol_ue_s / energy_j.
The tap is an object attribute of the plant, so ``env.copy()`` deep-copies it with the plant (rollouts get their own).
Per unit: the sums over its exposure set N(c) (and over c alone) for the unit window = seconds t0+1 .. t0+T.
The tap also keeps (i) scored cumulative per-cell RLFs, served UE-s ("load") and PRBs used ("prb", per-second mean
PRBs summed over seconds) for ``labels_p.Labeller(cells=True)`` (``arrays(ext=True)``), and (ii) a per-second per-cell
series of EVERY second (warm-up included, flagged): ``series()`` -> fields ``SERIES_FIELDS`` = pv, v (violated UE-s),
e (J), rlf (count), prb_used / prb_cap (mean PRBs of the second), ue (UE-s served = UEs whose serving cell is c at the
second's close), prot_ue (protected UE-s served). Row i = the window (i, i+1] = env second t = i + 1 (as ``lab_*``
rows of ``decision.trace``). PRIVILEGED plant counters: outcomes / labels only, never features.

Discovery study (scratchpad/e6_dev/decision/STEP1_MSCR_PLAN.md): ``PI0_HIGH_NO_RB`` (accept 0.5 / half 0.2 / reject
0.3 for ES, PowerES and SliceGuarantee; no rollback mode) and ``PlaceboPolicy`` (draws and logs the pi0 mode exactly
like ``RandomizedUnitPolicy`` but the arbiter APPLIES accept: a sharp null; after ``run_collection`` a placebo unit
carries mode / p = the pi0 draw and ``applied_mode`` = "accept"). ``enc`` / ``dec`` = the record array codec
(base64(zlib(little-endian float32 C-order bytes)) + shape).

Option (a) confounded logging (scratchpad/e6_dev/decision/OPTION_A_PLAN.md section 1; additive, 2026-09-30):
``IncumbentPolicy`` is a CONTEXT-DEPENDENT logging policy (accept / reject only) that reads ONLY the unit's obs-only
``ctx`` (never its own past modes). Pressure s = max(ctx own_prb_util, ctx nbr_max_prb_util) (NaN ignored; none -> 0).
Request class from ctx knob / step (d = sign(step), 0 -> +1 as crt_units' sgn): saving = carrier off (d -1), sleep
(d +1), ptx down (d -1); restore = the opposite directions; SG raise / SG lower = prot_min d +1 / -1. P(accept):

  | saving  | s < .6: .85 | s >= .6: .15 |      | SG raise | own_prot_below_frac > .05: .85 | else .30 |
  | restore | s < .6: .35 | s >= .6: .85 |      | SG lower | own_prot_below_frac == 0:  .85 | else .20 |

(NaN own_prot_below_frac -> 0), clipped to [INC_FLOOR, 1 - INC_FLOOR] = [.15, .85]. Draw: one uniform
``default_rng([seed, 6622, c, x_idx, int(t0)])``, accept iff u < P(accept). The unit's FULL mode-probability row is
logged as ``unit["probs"]`` ({"accept": pa, "reject": 1 - pa}; crt_units.build_unit_data prefers it over a record's
``pi0_table``) and the class / pressure as ``unit["inc"]``. ``PlaceboIncumbent`` draws and logs exactly like it but
applies accept (sharp null; ``finalize_units`` relabels as for ``PlaceboPolicy``). For these policies the arbiter may
act from t = 0: ``run_collection(arb_warmup_s=0.0)``; ``count_all=True`` makes the tap's cumulative arrays (hence the
units' privileged ``kpi`` window sums) count EVERY second, warm-up included (default: scored seconds only, unchanged).
The per-second ``series`` always records every second.
"""
from __future__ import annotations

import base64
import time
import zlib

import numpy as np

from cdd_oran.envs.e6.env import E6Env

from .units_p import T_UNIT, UnitArbiter

PI0_TAG = 6612
P_MIN = 0.1
PI0_ORDER = ("accept", "half", "reject", "accept+rb")
PI0_HIGH = {"ES": {"accept": 0.5, "half": 0.2, "reject": 0.3},
            "PowerES": {"accept": 0.5, "half": 0.2, "reject": 0.3},
            "Coverage": {"accept": 0.5, "half": 0.2, "reject": 0.3},
            "SliceGuarantee": {"accept": 0.5, "half": 0.15, "reject": 0.25, "accept+rb": 0.10}}
PI0_HIGH_NO_RB = {"ES": {"accept": 0.5, "half": 0.2, "reject": 0.3},
                  "PowerES": {"accept": 0.5, "half": 0.2, "reject": 0.3},
                  "SliceGuarantee": {"accept": 0.5, "half": 0.2, "reject": 0.3}}
LOW_ACCEPT = 0.8
SERIES_FIELDS = ("pv", "v", "e", "rlf", "prb_used", "prb_cap", "ue", "prot_ue")
EXT_FIELDS = ("rlf", "load", "prb")          # arrays(ext=True) = (pv, e, v) + these


def low_table(high: dict, accept: float = LOW_ACCEPT, p_min: float | None = P_MIN) -> dict:
    """Low-eps table: accept -> ``accept``, other modes scaled by (1 - accept) / (1 - high accept), then floored at
    ``p_min`` with accept absorbing the difference."""
    s = (1 - accept) / (1 - high["accept"])
    out = {m: p * s for m, p in high.items() if m != "accept"}
    if p_min is not None:
        out = {m: max(p, p_min) for m, p in out.items()}
    out = {"accept": 1.0 - sum(out.values()), **out}
    if p_min is not None and out["accept"] < p_min:
        raise ValueError("floor leaves accept below p_min")
    return {m: out[m] for m in PI0_ORDER if m in out}


PI0_LOW = {x: low_table(t) for x, t in PI0_HIGH.items()}
# Discovery v4 (docs/benchmark/E6P_DISCOVERY_PROTOCOL_V4.md section 2; additive, 2026-09-30): accept .5 / reject .5,
# no "half" mode, for ES, PowerES and SliceGuarantee (same draw key / PI0_ORDER as every pi0 table).
PI0_V4 = {x: {"accept": 0.5, "reject": 0.5} for x in ("ES", "PowerES", "SliceGuarantee")}
PI0 = {"high": PI0_HIGH, "low": PI0_LOW, "high_no_rb": PI0_HIGH_NO_RB, "v4": PI0_V4}


def regime_for(j: int) -> str:
    return "high" if int(j) % 2 == 0 else "low"


class RandomizedUnitPolicy:
    """pi0: ``policy(unit) -> (mode, propensity)``. ``tables`` overrides the per-xApp mode probabilities (e.g.
    ``{x: {"accept": 1.0}}`` for the all-accept check)."""

    def __init__(self, seed: int, regime: str = "high", tables: dict | None = None, p_min: float | None = P_MIN):
        if tables is None and regime not in PI0:
            raise ValueError(f"regime in {sorted(PI0)}")
        self.seed, self.regime = int(seed), regime
        self.tables = tables if tables is not None else PI0[regime]
        for x, t in self.tables.items():
            if abs(sum(t.values()) - 1.0) > 1e-9:
                raise ValueError(f"pi0 table for {x} does not sum to 1")
            if p_min is not None and tables is None and min(t.values()) < p_min - 1e-12:
                raise ValueError(f"pi0 table for {x} has a propensity below {p_min}")

    def draw_u(self, unit) -> float:
        return float(np.random.default_rng([self.seed, PI0_TAG, int(unit["c"]), int(unit["x_idx"]),
                                            int(unit["t0"])]).random())

    def __call__(self, unit):
        t = self.tables[unit["x"]]
        u, acc = self.draw_u(unit), 0.0
        modes = [m for m in PI0_ORDER if m in t]
        for m in modes:
            acc += t[m]
            if u < acc:
                return m, float(t[m])
        return modes[-1], float(t[modes[-1]])


class PlaceboPolicy(RandomizedUnitPolicy):
    """Sharp-null placebo: draws the pi0 mode exactly like ``RandomizedUnitPolicy`` (same key, same table) and logs
    it on the unit (``pi0_mode`` / ``pi0_p``), but returns ("accept", 1.0), so the arbiter applies accept (the
    trajectory is the accept-all one). ``run_collection`` then relabels: mode / p = the pi0 draw, applied_mode =
    "accept"."""

    def __call__(self, unit):
        m, p = super().__call__(unit)
        unit["pi0_mode"], unit["pi0_p"] = m, p
        return "accept", 1.0


# -------------------------------------------------------------------------------------------- confounded incumbent
INC_TAG = 6622
INC_FLOOR = 0.15
INC_S_HI = 0.6                       # pressure threshold on s = max(own_prb_util, nbr_max_prb_util)
INC_SG_RAISE_FRAC = 0.05             # SG raise "condition met": own_prot_below_frac > .05
INC_ORDER = ("accept", "reject")
INCUMBENT_TABLE = {"saving": {"lo": 0.85, "hi": 0.15}, "restore": {"lo": 0.35, "hi": 0.85},
                   "sg_raise": {"met": 0.85, "else": 0.30}, "sg_lower": {"met": 0.85, "else": 0.20}}
INC_SAVING_DIR = {"carrier": -1, "sleep": 1, "ptx": -1}      # d of the energy-SAVING request per knob family


def _nz(x) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if not np.isfinite(x) else x


def incumbent_class(ctx) -> str:
    """"saving" | "restore" | "sg_raise" | "sg_lower" from the unit context's knob family and step."""
    f = str(ctx["knob"])
    d = 1 if float(ctx["step"]) >= 0 else -1
    if f == "prot_min":
        return "sg_raise" if d > 0 else "sg_lower"
    if f in INC_SAVING_DIR:
        return "saving" if d == INC_SAVING_DIR[f] else "restore"
    raise ValueError(f"IncumbentPolicy: no rule for knob family {f!r}")


def incumbent_pressure(ctx) -> float:
    """s = max(own_prb_util, nbr_max_prb_util) over the finite ones (none -> 0)."""
    vals = [float(ctx.get(k, np.nan)) for k in ("own_prb_util", "nbr_max_prb_util")]
    vals = [v for v in vals if np.isfinite(v)]
    return max(vals) if vals else 0.0


class IncumbentPolicy:
    """Context-dependent confounded logging policy (module docstring, "Option (a)"). ``policy(unit) -> (mode, p)``;
    reads only ``unit["ctx"]`` and the RNG key fields (c, x_idx, t0); logs ``unit["probs"]`` and ``unit["inc"]``."""

    def __init__(self, seed: int, table: dict | None = None, floor: float = INC_FLOOR, s_hi: float = INC_S_HI,
                 sg_raise_frac: float = INC_SG_RAISE_FRAC):
        self.seed, self.table = int(seed), dict(table or INCUMBENT_TABLE)
        self.floor, self.s_hi, self.sg_raise_frac = float(floor), float(s_hi), float(sg_raise_frac)
        if not 0.0 < self.floor <= 0.5:
            raise ValueError("floor in (0, .5]")

    def accept_prob(self, ctx) -> tuple[float, dict]:
        cls = incumbent_class(ctx)
        s = incumbent_pressure(ctx)
        below = _nz(ctx.get("own_prot_below_frac", np.nan))
        t = self.table[cls]
        if cls in ("saving", "restore"):
            key = "hi" if s >= self.s_hi else "lo"
        elif cls == "sg_raise":
            key = "met" if below > self.sg_raise_frac else "else"
        else:
            key = "met" if below == 0.0 else "else"
        pa = float(min(max(float(t[key]), self.floor), 1.0 - self.floor))
        return pa, {"cls": cls, "key": key, "s": s, "below": below}

    def row(self, unit) -> dict:
        pa, _ = self.accept_prob(unit["ctx"])
        return {"accept": pa, "reject": 1.0 - pa}

    def draw_u(self, unit) -> float:
        return float(np.random.default_rng([self.seed, INC_TAG, int(unit["c"]), int(unit["x_idx"]),
                                            int(unit["t0"])]).random())

    def __call__(self, unit):
        pa, info = self.accept_prob(unit["ctx"])
        row = {"accept": pa, "reject": 1.0 - pa}
        unit["probs"], unit["inc"] = row, info
        m = "accept" if self.draw_u(unit) < pa else "reject"
        return m, float(row[m])


class PlaceboIncumbent(IncumbentPolicy):
    """Sharp-null placebo of ``IncumbentPolicy``: same draw, same logged row, but the arbiter applies accept
    (``finalize_units`` relabels mode / p = the incumbent draw, applied_mode = "accept")."""

    def __call__(self, unit):
        m, p = super().__call__(unit)
        unit["pi0_mode"], unit["pi0_p"] = m, p
        return "accept", 1.0


def finalize_units(units) -> None:
    """Placebo relabel (in place): a unit with ``pi0_mode`` gets mode / p = its pi0 draw and applied_mode = the mode
    the arbiter applied. Other units are left untouched."""
    for u in units:
        if "pi0_mode" in u:
            u["applied_mode"] = u["mode"]
            u["mode"], u["p"] = u.pop("pi0_mode"), u.pop("pi0_p")


# -------------------------------------------------------------------------------------------- record codec
def enc(a) -> dict:
    """ndarray -> {"b64": base64(zlib(float32 '<f4' C-order bytes)), "shape": [...], "dtype": "float32"}."""
    a = np.ascontiguousarray(np.asarray(a, dtype="<f4"))
    return {"b64": base64.b64encode(zlib.compress(a.tobytes(), 6)).decode("ascii"), "shape": list(a.shape),
            "dtype": "float32"}


def dec(d) -> np.ndarray:
    """Inverse of ``enc`` (float32 array of the recorded shape)."""
    return np.frombuffer(zlib.decompress(base64.b64decode(d["b64"])), dtype="<f4").reshape(d["shape"]).copy()


# -------------------------------------------------------------------------------------------- realised KPI tap
class CellKPITap:
    """Per-cell cumulative scored KPIs + per-second series; installed as ``plant.take_counters`` (module docstring)."""

    def __init__(self, plant, n_sec: int | None = None, count_all: bool = False):
        self.plant = plant
        self.count_all = bool(count_all)          # True: cumulative arrays count warm-up seconds too
        n = plant.nc
        self.pv, self.v, self.e = np.zeros(n), np.zeros(n), np.zeros(n)
        self.rlf, self.load, self.prb = np.zeros(n), np.zeros(n), np.zeros(n)
        cfg = plant.cfg
        T = int(n_sec) if n_sec is not None else int(round(cfg.warmup_s + cfg.scored_s))
        self.ser = np.zeros((max(T, 1), len(SERIES_FIELDS), n), np.float32)
        self.ser_scored = np.zeros(max(T, 1), bool)
        self.n_sec = 0

    def __call__(self):
        p = self.plant
        c = type(p).take_counters(p)
        n = p.nc
        serv = p.serv
        scored = bool(getattr(p, "sec_scored", False))
        vp = getattr(p, "sec_viol_prot", None)
        pv_s = np.bincount(serv[vp], minlength=n) if vp is not None else np.zeros(n, np.int64)
        v_s = np.bincount(serv[p.sec_viol_ll | p.sec_viol_embb | p.sec_viol_be], minlength=n) \
            if hasattr(p, "sec_viol_ll") else np.zeros(n, np.int64)
        ticks = max(int(c["ticks"]), 1)
        rlf_s = np.asarray(c["rlf"], float)
        used_s = np.asarray(c["prb_used"], float).sum(1) / ticks
        ue_s = np.bincount(serv, minlength=n)
        if scored or self.count_all:
            self.pv += pv_s
            self.v += v_s
            self.e += c["energy_j"]
            self.rlf += rlf_s
            self.load += ue_s
            self.prb += used_s
        if self.n_sec >= len(self.ser):                                  # longer than declared: grow
            self.ser = np.concatenate([self.ser, np.zeros_like(self.ser)])
            self.ser_scored = np.concatenate([self.ser_scored, np.zeros_like(self.ser_scored)])
        row = self.ser[self.n_sec]
        row[0], row[1], row[2], row[3] = pv_s, v_s, c["energy_j"], rlf_s
        row[4], row[5] = used_s, np.asarray(c["prb_cap"], float) / ticks
        row[6], row[7] = ue_s, np.bincount(serv[p.prot], minlength=n) if p.prot.any() else 0.0
        self.ser_scored[self.n_sec] = scored
        self.n_sec += 1
        return c

    def sums(self, cells) -> tuple[float, float, float]:
        idx = list(cells)
        return float(self.pv[idx].sum()), float(self.e[idx].sum()), float(self.v[idx].sum())

    def arrays(self, ext: bool = False):
        """(pv, e, v) scored cumulative per-cell arrays; ``ext`` appends (rlf, load, prb) (``EXT_FIELDS``)."""
        base = (self.pv.copy(), self.e.copy(), self.v.copy())
        return base + (self.rlf.copy(), self.load.copy(), self.prb.copy()) if ext else base

    def series(self) -> dict:
        """Per-second per-cell series of every recorded second: {"fields", "t" (T,), "scored" (T,) bool,
        "data" (T, len(fields), cells) float32}; row i = env second t[i] = i + 1."""
        n = self.n_sec
        return {"fields": list(SERIES_FIELDS), "t": np.arange(1, n + 1), "scored": self.ser_scored[:n].copy(),
                "data": self.ser[:n].copy()}


def install_tap(env, count_all: bool = False) -> CellKPITap:
    """Install (or return the already installed) tap; ``count_all`` applies to a NEW tap only."""
    if isinstance(env.plant.__dict__.get("take_counters"), CellKPITap):
        return env.plant.take_counters
    tap = CellKPITap(env.plant, count_all=count_all)
    env.plant.take_counters = tap
    return tap


def get_tap(env) -> CellKPITap:
    tap = env.plant.__dict__.get("take_counters")
    if not isinstance(tap, CellKPITap):
        raise RuntimeError("no CellKPITap installed on this env (collect_p.install_tap)")
    return tap


# -------------------------------------------------------------------------------------------- episode
def _kpi(tap, u, t_end, trunc):
    pv, e, v = tap.sums(u["exp"])
    pv_c, e_c, v_c = tap.sums([u["c"]])
    o = u.pop("_open")
    u["kpi"] = {"pv": pv - o[0], "e": e - o[1], "v": v - o[2], "pv_own": pv_c - o[3], "e_own": e_c - o[4],
                "v_own": v_c - o[5], "secs": float(t_end - u["t0"]), "trunc": bool(trunc)}


def run_collection(cfg, policy, T: float = T_UNIT, labeller=None, churn_cap=None, env=None,
                   open_rule: str = "first", arb_warmup_s: float | None = None, count_all: bool = False):
    """One collection episode under ``wg3=True``: ``UnitArbiter(policy)`` + the KPI tap.

    ``labeller(env, obs, snap, opened)`` (optional) is called every second in which units opened, after the arbiter
    decided and BEFORE ``env.step_apply``: ``env`` carries the pending requests (``env.copy`` replays them) and ``snap``
    is an arbiter fork taken before this second's call. ``arb_warmup_s`` (default ``cfg.warmup_s``): the arbiter's
    all-accept / no-unit period (0.0 = units from t = 0, option (a)); ``count_all``: tap flag (``CellKPITap``).
    Returns {"env", "arb", "tap", "units", "cpu_s"}."""
    t_cpu = time.process_time()
    env = env if env is not None else E6Env(cfg, log=False, wg3=True, churn_cap=churn_cap)
    tap = install_tap(env, count_all=count_all)
    wu = float(cfg.warmup_s) if arb_warmup_s is None else float(arb_warmup_s)
    arb = UnitArbiter(policy, T=T, warmup_s=wu, open_rule=open_rule)
    live = []
    while env.sec < env.total_s:
        obs = env.step_propose()
        now = obs["t"]
        still = []
        for u in live:
            if now >= u["t0"] + T:
                _kpi(tap, u, u["t0"] + T, False)
            else:
                still.append(u)
        live = still
        snap = arb.fork(None) if labeller is not None and obs["requests"] and now >= arb.warmup_s else None
        dec = arb(obs)
        for u in arb.opened:
            u["_open"] = tap.sums(u["exp"]) + tap.sums([u["c"]])
            live.append(u)
        if labeller is not None and arb.opened:
            labeller(env, obs, snap, list(arb.opened))
        env.step_apply(dec)
    for u in live:
        _kpi(tap, u, float(env.sec), True)
    finalize_units(arb.units)
    return {"env": env, "arb": arb, "tap": tap, "units": arb.units, "cpu_s": time.process_time() - t_cpu}


__all__ = ["EXT_FIELDS", "INCUMBENT_TABLE", "INC_FLOOR", "INC_ORDER", "INC_SAVING_DIR", "INC_S_HI",
           "INC_SG_RAISE_FRAC", "INC_TAG", "P_MIN", "PI0", "PI0_HIGH", "PI0_HIGH_NO_RB", "PI0_LOW", "PI0_ORDER",
           "PI0_TAG", "PI0_V4", "SERIES_FIELDS", "CellKPITap", "IncumbentPolicy", "PlaceboIncumbent", "PlaceboPolicy",
           "RandomizedUnitPolicy", "dec", "enc", "finalize_units", "get_tap", "incumbent_class", "incumbent_pressure",
           "install_tap", "low_table", "regime_for", "run_collection"]
