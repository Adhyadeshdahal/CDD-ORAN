"""Randomized joint-policy data collection for policy-effect estimation (DESIGN.md REVISION v2, item 3).

At each decision epoch (every D s from the end of warm-up) every region r independently draws a WG3 region policy
pi_r = {mode per xApp in XAPPS: accept|reject|half|lock, rb: 0|1} (plans.RegionPlan) from the declared distribution
  with prob 1 - eps: accept-all (code 0); with prob eps: uniform over all 4^4 x 2 = 512 codes (accept-all included),
so P(code) = (1 - eps) * [code == 0] + eps / 512 exactly. code = rb * 256 + sum_i mode_idx(XAPPS[i]) * 4^i.
The policy is applied for D s with the arbiter's plan -> decision semantics (``plans.decide``: rollbacks only on the
epoch's first second, never of our own rollbacks). Before warm-up ends every request is accepted.
RNG per (cfg.seed, KEY, epoch, region): reproducible and independent of the plant tape and of other regions.
Optional observable churn allowance: ``epoch_churn_cap`` = max applied knob changes per epoch (all causes, xApp
accepts included); at each epoch start the env cap is set to (changes so far + allowance).

Output (``collect_episode``): the episode ``Trace`` with extra arrays
  pol_region_ids (R,); pol_epoch, pol_t, pol_len (E,) epoch index, start second, seconds held;
  pol_code (E, R) i2, pol_modes (E, R, 4) i1 (index into plans.MODES, XAPPS order), pol_rb (E, R) i1,
  pol_default (E, R) bool (drawn by the 1 - eps branch), pol_prop (E, R) exact P(code),
  pol_prop_eff (E, R) P of the code's class after marginalising xApps absent from the mix (their modes are inert);
  application (requests whose knob is in region r, decided at seconds [t_e, t_e + len)):
    pol_n_req (E, R), pol_out (E, R, len(OUT_CODES)) counts per request outcome, pol_rb_req / pol_rb_applied (E, R),
    pol_locks (E, R) locks set;
  observable context at the epoch start (before its decisions):
    ctx_prb_util (E, R) mean PRB utilisation of the region's cells in the latest delivered fast report,
    ctx_act_ue (E, R, 3) active UEs per slice (same report), ctx_ll_p95 (E, R) max LL p95 delay (NaN if none),
    ctx_report_age (E,) t_e - t1 of that report, ctx_nreq (E, R) requests offered at t_e, ctx_changes (E,),
    ctx_churn_left (E,) cap - changes (-1 = no cap), ctx_locked (E, R) active locks on the region's knobs;
  labels (PRIVILEGED) over the held window (t_e, t_e + len]: lab_pol_viol (E, R, 3), lab_pol_rlf (E, R),
    lab_pol_energy_j (E, R), lab_pol_outage (E, R), lab_pol_ue_s (E, R, 3).
"""
from __future__ import annotations

import numpy as np

from cdd_oran.envs.e6.env import E6Env

from . import plans as P
from .trace import OUT_CODES, RB_CODES, Trace, region_labels

KEY = 7707                              # RNG stream tag of the collector (not an env seed)
N_CODES = len(P.MODES) ** len(P.XAPPS) * 2


def decode(code: int) -> dict:
    """code -> RegionPlan."""
    rb, m = divmod(int(code), len(P.MODES) ** len(P.XAPPS))
    modes = {}
    for x in P.XAPPS:
        m, i = divmod(m, len(P.MODES))
        modes[x] = P.MODES[i]
    return {"mode": modes, "rb": rb}


def encode(rp: dict) -> int:
    return int(rp["rb"]) * len(P.MODES) ** len(P.XAPPS) + \
        sum(P.MODES.index(rp["mode"][x]) * len(P.MODES) ** i for i, x in enumerate(P.XAPPS))


def propensity(code: int, eps: float) -> float:
    return (1.0 - eps) * (int(code) == 0) + eps / N_CODES


def propensity_eff(code: int, eps: float, active) -> float:
    """P of the class of codes equal to ``code`` on the xApps in ``active`` and on rb."""
    rp = decode(code)
    n_free = sum(1 for x in P.XAPPS if x not in active)
    in_default = not rp["rb"] and all(rp["mode"][x] == "accept" for x in P.XAPPS if x in active)
    return (1.0 - eps) * in_default + eps * len(P.MODES) ** n_free / N_CODES


def draw(seed: int, epoch: int, region: int, eps: float) -> tuple[int, bool]:
    """-> (code, drawn_by_default_branch)."""
    r = np.random.default_rng([int(seed), KEY, int(epoch), int(region)])
    if r.uniform() >= eps:
        return 0, True
    return int(r.integers(N_CODES)), False


class RandomizedJointPolicy:
    """Collector arbiter. Use with the two-phase loop: dec = act(obs); env.step_apply(dec); record(dec)."""

    def __init__(self, env: E6Env, eps: float, D: int = 20, epoch_churn_cap: int | None = None,
                 start_s: float | None = None):
        if not 0.0 <= eps <= 1.0:
            raise ValueError("eps must be in [0, 1]")
        self.env, self.eps, self.D, self.epoch_churn_cap = env, float(eps), int(D), epoch_churn_cap
        self.site = np.asarray(env.plant.lay.cell_site)
        self.regions = sorted({int(x) for x in self.site})
        self.start_s = float(env.cfg.warmup_s if start_s is None else start_s)
        self.active = {x.name for x in env.xapps}
        self.plan = P.accept_all(self.regions)
        self.until, self.epoch = -1.0, -1
        self.rb_at = {}
        self.fast = None                  # latest delivered fast KPM report
        self.rows = []                    # per epoch: (t, codes, defaults, ctx dict)

    def _context(self, obs):
        env, nr = self.env, len(self.regions)
        rix = {g: i for i, g in enumerate(self.regions)}
        c = {"prb_util": np.full(nr, np.nan), "act_ue": np.full((nr, 3), np.nan), "ll_p95": np.full(nr, np.nan),
             "report_age": np.nan, "nreq": np.zeros(nr, int), "locked": np.zeros(nr, int)}
        f = self.fast
        if f is not None:
            c["report_age"] = obs["t"] - f["t1"]
            for i, g in enumerate(self.regions):
                m = self.site == g
                c["prb_util"][i] = float(np.mean(f["prb_util"][m]))
                c["act_ue"][i] = f["act_ue"][m].sum(0)
                d = f["ll_delay_p95"][m]
                c["ll_p95"][i] = float(np.nanmax(d)) if np.isfinite(d).any() else np.nan
        for r in obs["requests"]:
            c["nreq"][rix[int(self.site[r["knob"][1]])]] += 1
        for k in obs["locked"]:
            c["locked"][rix[int(self.site[k[1]])]] += 1
        c["changes"] = obs["changes"]
        cap = env.churn_cap
        c["churn_left"] = -1 if cap is None else int(cap) - int(obs["changes"])
        return c

    def act(self, obs: dict) -> dict:
        for rep in obs["new_reports"]:
            if rep["gran"] == "fast" and (self.fast is None or rep["t1"] >= self.fast["t1"]):
                self.fast = rep
        first = False
        if obs["t"] >= self.start_s and obs["t"] >= self.until:
            self.epoch += 1
            if self.epoch_churn_cap is not None:
                self.env.churn_cap = int(self.env.stats["changes"]) + int(self.epoch_churn_cap)
            ctx = self._context(obs)
            draws = [draw(self.env.cfg.seed, self.epoch, g, self.eps) for g in self.regions]
            self.plan = {g: decode(c) for g, (c, _) in zip(self.regions, draws, strict=True)}
            self.rows.append((obs["t"], [c for c, _ in draws], [b for _, b in draws], ctx))
            self.until, first = obs["t"] + self.D, True
        return P.decide(self.plan, obs, self.site, self.D, self.env.last_change, self.rb_at, first)

    def record(self, dec: dict) -> None:
        for k in dec["rollback"]:
            if k in self.env.last_change:
                self.rb_at[k] = self.env.last_change[k]

    def table(self, trace: Trace) -> dict:
        """Per (epoch, region) policy table (see module docstring), joined with the trace's outcomes."""
        a, nr = trace.arrays, len(self.regions)
        E = len(self.rows)
        end = int(a["t"][-1]) if len(a["t"]) else 0
        t = np.array([r[0] for r in self.rows], float)
        ln = np.array([min(self.D, end - int(x) + 1) for x in t], np.int32)
        codes = np.array([r[1] for r in self.rows], np.int16).reshape(E, nr)
        out = {"pol_region_ids": np.array(self.regions, np.int16), "pol_epoch": np.arange(E, dtype=np.int32),
               "pol_t": t.astype(np.int32), "pol_len": ln, "pol_code": codes,
               "pol_default": np.array([r[2] for r in self.rows], bool).reshape(E, nr)}
        dec = [decode(c) for c in codes.ravel()]
        out["pol_modes"] = np.array([[P.MODES.index(d["mode"][x]) for x in P.XAPPS] for d in dec],
                                    np.int8).reshape(E, nr, len(P.XAPPS))
        out["pol_rb"] = np.array([d["rb"] for d in dec], np.int8).reshape(E, nr)
        out["pol_prop"] = np.vectorize(lambda c: propensity(c, self.eps))(codes).astype(float).reshape(E, nr)
        out["pol_prop_eff"] = np.vectorize(lambda c: propensity_eff(c, self.eps, self.active))(codes) \
            .astype(float).reshape(E, nr)
        knobs = trace.meta["knobs"]
        kreg = np.array([int(self.site[k[1]]) for k in knobs])
        rix = np.searchsorted(self.regions, kreg)                          # knob -> region column
        ep_of = lambda tt: np.searchsorted(t, tt, side="right") - 1        # noqa: E731
        n_req = np.zeros((E, nr), np.int32)
        outc = np.zeros((E, nr, len(OUT_CODES)), np.int32)
        e = ep_of(a["rq_t"])
        ok = (e >= 0) & (a["rq_t"] < t[np.maximum(e, 0)] + ln[np.maximum(e, 0)])
        np.add.at(n_req, (e[ok], rix[a["rq_knob"][ok]]), 1)
        np.add.at(outc, (e[ok], rix[a["rq_knob"][ok]], a["rq_out"][ok]), 1)
        rb_req = np.zeros((E, nr), np.int32)
        rb_app = np.zeros((E, nr), np.int32)
        e = ep_of(a["rb_t"])
        ok = e >= 0
        np.add.at(rb_req, (e[ok], rix[a["rb_k"][ok]]), 1)
        okA = ok & (a["rb_out"] == RB_CODES.index("applied"))
        np.add.at(rb_app, (e[okA], rix[a["rb_k"][okA]]), 1)
        locks = np.zeros((E, nr), np.int32)
        e = ep_of(a["lk_t"])
        ok = e >= 0
        np.add.at(locks, (e[ok], rix[a["lk_k"][ok]]), 1)
        out.update({"pol_n_req": n_req, "pol_out": outc, "pol_rb_req": rb_req, "pol_rb_applied": rb_app,
                    "pol_locks": locks})
        ctx = [r[3] for r in self.rows]
        for k in ("prb_util", "act_ue", "ll_p95", "nreq", "locked"):
            out["ctx_" + k] = np.array([c[k] for c in ctx]).reshape(E, nr, *np.shape(ctx[0][k])[1:]) if E else \
                np.zeros((0, nr))
        for k in ("report_age", "changes", "churn_left"):
            out["ctx_" + k] = np.array([c[k] for c in ctx], float)
        lab = {k: np.zeros((E, nr) + s) for k, s in (("viol", (3,)), ("rlf", ()), ("energy_j", ()),
                                                     ("outage", ()), ("ue_s", (3,)))}
        for i in range(E):
            y = region_labels(trace, self.site, t[i], t[i] + ln[i])
            for k in lab:
                lab[k][i] = y[k]
        out.update({"lab_pol_" + k: v for k, v in lab.items()})
        return out


def collect_episode(cfg, eps: float, D: int = 20, epoch_churn_cap: int | None = None,
                    churn_cap: int | None = None, snapshot_s: int = 60) -> Trace:
    """One randomized joint-policy episode under ``wg3=True``: the trace + the (epoch, region) policy table."""
    env = E6Env(cfg, log=False, wg3=True, churn_cap=churn_cap, trace=True, trace_snapshot_s=snapshot_s)
    col = RandomizedJointPolicy(env, eps, D=D, epoch_churn_cap=epoch_churn_cap)
    while env.sec < env.total_s:
        obs = env.step_propose()
        dec = col.act(obs)
        env.step_apply(dec)
        col.record(dec)
    tr = env.get_trace()
    tr.arrays.update(col.table(tr))
    tr.meta["collector"] = {"kind": "randomized_joint_policy", "eps": col.eps, "D": col.D, "key": KEY,
                            "epoch_churn_cap": epoch_churn_cap, "churn_cap": churn_cap, "n_codes": N_CODES,
                            "xapps_order": list(P.XAPPS), "modes": list(P.MODES), "active_xapps": sorted(col.active),
                            "rb_window_s": P.RB_WINDOW, "score": env.score()}
    tr.meta["collector"]["score"] = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                                     for k, v in tr.meta["collector"]["score"].items()}
    return tr
