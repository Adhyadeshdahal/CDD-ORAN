"""Randomized joint-policy data collection for policy-effect estimation (DESIGN.md REVISION v2, item 3).

Two regimes (both kept; v1's RNG contract and output are unchanged, for reproducibility):
  v1 ``RandomizedJointPolicy`` / ``collect_episode``: eps-mixture of accept-all and uniform over all 512 codes;
  v2 ``RandomizedStepPolicy`` / ``collect_episode_v2``: the predeclared requested-step mixture of
     scratchpad/decision_stack/SOL_DATA_DESIGN.md (a), documented in ``STEP_V2_DOC`` (end of this module).
The rest of this docstring describes v1; v2 writes the same table (its own propensities) plus extra columns.

v1: at each decision epoch (every D s from the end of warm-up) every region r independently draws a WG3 region policy
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

import dataclasses
from dataclasses import dataclass

import numpy as np

from cdd_oran.envs.e6.env import E6Env

from . import plans as P
from .trace import OUT, OUT_CODES, RB_CODES, Trace, region_labels

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


def region_context(obs, fast, site, regions):
    """Observable per-region context at a decision epoch (shared by the collector and serve-time models: no
    train/serve skew). ``fast`` = latest delivered fast KPM report; churn_left uses the cap in force in ``obs``."""
    nr = len(regions)
    rix = {g: i for i, g in enumerate(regions)}
    c = {"prb_util": np.full(nr, np.nan), "act_ue": np.full((nr, 3), np.nan), "ll_p95": np.full(nr, np.nan),
         "report_age": np.nan, "nreq": np.zeros(nr, int), "locked": np.zeros(nr, int)}
    f = fast
    if f is not None:
        c["report_age"] = obs["t"] - f["t1"]
        for i, g in enumerate(regions):
            m = site == g
            c["prb_util"][i] = float(np.mean(f["prb_util"][m]))
            c["act_ue"][i] = f["act_ue"][m].sum(0)
            d = f["ll_delay_p95"][m]
            c["ll_p95"][i] = float(np.nanmax(d)) if np.isfinite(d).any() else np.nan
    for r in obs["requests"]:
        c["nreq"][rix[int(site[r["knob"][1]])]] += 1
    for k in obs["locked"]:
        c["locked"][rix[int(site[k[1]])]] += 1
    c["changes"] = obs["changes"]
    cap = obs.get("churn_cap")
    c["churn_left"] = -1 if cap is None else int(cap) - int(obs["changes"])
    return c


class RandomizedJointPolicy:
    """Collector arbiter. Use with the two-phase loop: dec = act(obs); env.step_apply(dec); record(dec)."""

    def __init__(self, env: E6Env, eps: float, D: int = 20, epoch_churn_cap: int | None = None,
                 start_s: float | None = None, half_rule: str = "slew"):
        if not 0.0 <= eps <= 1.0:
            raise ValueError("eps must be in [0, 1]")
        if half_rule not in P.HALF_RULES:
            raise ValueError(f"half_rule in {P.HALF_RULES}")
        self.half_rule, self.half_state = half_rule, {}
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

    def _draw_epoch(self, epoch: int) -> tuple[list, list]:
        """-> (codes, drawn_by_default_branch) per region (regions order) for ``epoch``."""
        draws = [draw(self.env.cfg.seed, epoch, g, self.eps) for g in self.regions]
        return [c for c, _ in draws], [b for _, b in draws]

    def _context(self, obs):
        return region_context(obs, self.fast, self.site, self.regions)

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
            codes, defaults = self._draw_epoch(self.epoch)
            self.plan = {g: decode(c) for g, c in zip(self.regions, codes, strict=True)}
            self.rows.append((obs["t"], codes, defaults, ctx))
            self.until, first = obs["t"] + self.D, True
        return P.decide(self.plan, obs, self.site, self.D, self.env.last_change, self.rb_at, first,
                        self.half_state, self.half_rule)

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
                    churn_cap: int | None = None, snapshot_s: int = 60, half_rule: str = "slew") -> Trace:
    """One randomized joint-policy episode under ``wg3=True``: the trace + the (epoch, region) policy table.
    ``half_rule="legacy"`` reproduces v1 data collected before 2026-09-28 (plans module docstring)."""
    env = E6Env(cfg, log=False, wg3=True, churn_cap=churn_cap, trace=True, trace_snapshot_s=snapshot_s)
    col = RandomizedJointPolicy(env, eps, D=D, epoch_churn_cap=epoch_churn_cap, half_rule=half_rule)
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
                            "rb_window_s": P.RB_WINDOW, "half_rule": half_rule, "score": env.score()}
    tr.meta["collector"]["score"] = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                                     for k, v in tr.meta["collector"]["score"].items()}
    return tr



# ================================================================================================= v2 (step mixture)
STEP_V2_DOC = """v2 requested-step collector (SOL_DATA_DESIGN.md (a) "Policy data"). Estimand: POLICY intention-to-treat
effects at the WG3 arbitration boundary (xApp reactions included), NOT knob do() effects: whether and what an xApp
requests still depends on load, so the assignment randomizes how requests are TREATED, not which knobs move.

Declared assignment mechanism (``StepMixture``; defaults are declared engineering choices, not tuned on outcomes).
Per region r and decision epoch e (epochs every D s from the end of warm-up), drawn from
default_rng([cfg.seed, KEY_V2, e, r]) -- a function of the keys only, so it is fixed BEFORE any outcome, independent
of the plant tape / scenario / load and of the other regions (draw order: u ~ U[0,1), then the branch's draws):
  u <  p_accept                     ACCEPT : accept-all (code 0)
  else u < p_accept + p_frac        FRAC   : every xApp i.i.d. uniform requested-step fraction f in {0, 0.5, 1}
                                             (r.integers(3, size=4), XAPPS order) -> mode reject | half | accept;
                                             f = share of the requested slew the RIC applies ("half" = halve
                                             the slew rate, plans.half_step: half of a multi-quantum step, or
                                             alternate single-quantum requests per knob accepted)
  else u < ... + p_lock             LOCK   : one xApp uniform (r.integers(4)) -> mode lock (reject + lock the
                                             requested knob for D s), the others accept
  else                              RB     : rollback block (rb = 1: at the epoch's first second roll back the
                                             region's knobs changed in the last plans.RB_WINDOW s), all accept
Defaults p_accept, p_frac, p_lock, p_rb = 0.50, 0.40, 0.05, 0.05 -> per (region, xApp) marginal
P(accept) = 0.5 + 0.4/3 + 0.05*3/4 + 0.05 = 0.7208, P(reject) = P(half) = 0.1333, P(lock) = 0.0125.
Exact code propensity P(code) = p_accept [code = 0] + p_frac [rb = 0, modes in {acc, rej, half}] / 81
  + p_lock [rb = 0, exactly one lock, others accept] / 4 + p_rb [rb = 1, all accept]     (``StepMixture.code_probs``).
Optional QUIET epochs (``quiet_epochs`` = q, default 0 = off): after an epoch whose drawn code != 0 the region's
next q epochs are forced to accept-all (branch "quiet", conditional propensity 1). Their draws are still made (keys
unchanged) and discarded. q >= ceil(H / D) - 1 (= 4 for H = 90, D = 20) keeps every perturbation's label window free
of later perturbations of the SAME region (not of its neighbours), at the price of fewer perturbed epochs.

H > D contamination (REPORT_G open issue 1): a label over (t_e, t_e + H] with H > D also covers the epochs
e+1 .. e + ceil(H/D) - 1 of the same region (and the neighbours' assignments). Those assignments come from the same
key-only mechanism (independent of outcomes; with quiet epochs they depend only on earlier draws), so they are valid
COVARIATES: the table logs them (``pol_fut_*``) for the declared ``label_H``; ``future_assignments`` rebuilds them
for any other H. Without adjusting for them the estimand is "pi for one epoch, then the logging policy", not
"pi, then accept-all".

Extra table columns (E epochs, R regions, X = 4 = len(plans.XAPPS) in XAPPS order; per-xApp request statistics count
the requests offered at seconds [t_e, t_e + pol_len) on knobs of the region, by the proposing xApp):
  pol_branch (E, R) i1 (STEP_BRANCHES index), pol_quiet (E, R) bool, pol_frac (E, R, X) f4 assigned fraction
    (accept 1, half 0.5, reject 0, lock 0), pol_prop (E, R) exact P(code | the region's earlier assignments),
    pol_prop_eff (E, R) the same for the code's class on the active xApps, pol_prop_x (E, R, X) exact
    P(logged mode of that xApp | earlier assignments);
  pol_x_elig (E, R, X) bool = the xApp offered >= 1 request in the region during the epoch;
  pol_x_n_req, pol_x_n_ack (out ok|noop), pol_x_n_nack (actuator|churn|locked), pol_x_n_rej (reject|lock_set),
    pol_x_n_changed (out ok) (E, R, X) i4;
  pol_x_req_dabs (E, R, X) sum |prop - cur| (mixed knob units; per-request detail in rq_*),
    pol_x_app_dabs (E, R, X) sum |realised delta|, pol_x_real_frac (E, R, X) mean realised / requested delta over
    requests with prop != cur (NaN if none) = first-stage compliance;
  rq_epoch (Q,) i4 epoch of each request (-1 outside the policy epochs), rq_frac (Q,) f4 assigned fraction of the
    request's (region, xApp) (NaN outside), rq_delta_req = prop - cur, rq_delta_app = applied - cur if out == ok
    else 0 (the realised change caused by the request);
  pol_region_adj (R, R) bool region adjacency from the cell neighbour lists (symmetrised, no diagonal),
    pol_nbr_n_dev (E, R) adjacent regions with code != 0, pol_nbr_n_lock, pol_nbr_n_rb (E, R),
    pol_nbr_frac (E, R, X) mean assigned fraction over the adjacent regions;
  pol_fut_code (E, R, F) i2 codes of the region's epochs e+1 .. e+F (F = ceil(label_H / D) - 1; -1 past the last
    epoch), pol_fut_t (E, F) their start seconds (-1 past the end), pol_fut_prop (E, R, F), pol_fut_frac
    (E, R, F, X), pol_fut_rb (E, R, F); pol_fut_H () = label_H.
"""

KEY_V2 = 7708                             # RNG stream tag of the v2 collector (regime tag; not an env seed)
STEP_BRANCHES = ("accept", "frac", "lock", "rb", "quiet")
FRAC_MODES = ("reject", "half", "accept")                    # index = fraction draw; fraction = FRACTIONS[index]
FRACTIONS = (0.0, 0.5, 1.0)
MODE_FRAC = {"accept": 1.0, "half": 0.5, "reject": 0.0, "lock": 0.0}


@dataclass(frozen=True)
class StepMixture:
    """Declared v2 assignment mixture (``STEP_V2_DOC``): branch probabilities + optional quiet epochs."""
    p_accept: float = 0.50
    p_frac: float = 0.40
    p_lock: float = 0.05
    p_rb: float = 0.05
    quiet_epochs: int = 0

    def __post_init__(self):
        ps = (self.p_accept, self.p_frac, self.p_lock, self.p_rb)
        if min(ps) < 0 or not np.isclose(sum(ps), 1.0) or int(self.quiet_epochs) < 0:
            raise ValueError("branch probabilities must be >= 0 and sum to 1; quiet_epochs >= 0")

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    def code_probs(self) -> np.ndarray:
        """(N_CODES,) exact P(code) of a non-quiet epoch."""
        n = len(P.XAPPS)
        pv = np.zeros(N_CODES)
        pv[0] += self.p_accept
        for c in range(N_CODES):
            rp = decode(c)
            m = [rp["mode"][x] for x in P.XAPPS]
            if not rp["rb"] and all(x in FRAC_MODES for x in m):
                pv[c] += self.p_frac / len(FRAC_MODES) ** n
            if not rp["rb"] and m.count("lock") == 1 and m.count("accept") == n - 1:
                pv[c] += self.p_lock / n
            if rp["rb"] and all(x == "accept" for x in m):
                pv[c] += self.p_rb
        return pv

    def mode_marginal(self) -> dict:
        """P(mode of one xApp) in a non-quiet epoch."""
        n = len(P.XAPPS)
        return {"accept": self.p_accept + self.p_frac / 3 + self.p_lock * (n - 1) / n + self.p_rb,
                "reject": self.p_frac / 3, "half": self.p_frac / 3, "lock": self.p_lock / n}


DEFAULT_STEP_MIXTURE = StepMixture()
ACCEPT_ALL_MIXTURE = StepMixture(1.0, 0.0, 0.0, 0.0)          # paired accept-all reference episodes


def draw_step(seed: int, epoch: int, region: int, mix: StepMixture) -> tuple[int, int]:
    """-> (code, branch index into STEP_BRANCHES) for one (seed, epoch, region), before quiet-epoch overrides."""
    r = np.random.default_rng([int(seed), KEY_V2, int(epoch), int(region)])
    u = r.uniform()
    if u < mix.p_accept:
        return 0, 0
    if u < mix.p_accept + mix.p_frac:
        f = r.integers(len(FRAC_MODES), size=len(P.XAPPS))
        return encode({"mode": {x: FRAC_MODES[int(i)] for x, i in zip(P.XAPPS, f, strict=True)}, "rb": 0}), 1
    if u < mix.p_accept + mix.p_frac + mix.p_lock:
        j = int(r.integers(len(P.XAPPS)))
        return encode({"mode": {x: ("lock" if i == j else "accept") for i, x in enumerate(P.XAPPS)}, "rb": 0}), 2
    return encode(P.uniform("accept", rb=1)), 3


def step_schedule(seed: int, n_epochs: int, regions, mix: StepMixture) -> dict:
    """Full assignment schedule, fixed before any outcome: code, branch (n_epochs, R) and prop = exact
    P(code | the region's earlier assignments) (1 in quiet epochs)."""
    pv = mix.code_probs()
    R = len(regions)
    code = np.zeros((n_epochs, R), np.int16)
    branch = np.zeros((n_epochs, R), np.int8)
    prop = np.zeros((n_epochs, R))
    quiet = STEP_BRANCHES.index("quiet")
    for j, g in enumerate(regions):
        left = 0
        for e in range(n_epochs):
            c, b = draw_step(seed, e, g, mix)
            if left > 0:
                c, b, p, left = 0, quiet, 1.0, left - 1
            else:
                p = pv[c]
                if c != 0:
                    left = mix.quiet_epochs
            code[e, j], branch[e, j], prop[e, j] = c, b, p
    return {"code": code, "branch": branch, "prop": prop}


def region_adjacency(neighbours, site) -> np.ndarray:
    """(R, R) bool: regions with any adjacent cell pair (either direction), no diagonal."""
    site = np.asarray(site)
    regions = sorted({int(x) for x in site})
    rix = np.searchsorted(regions, site)
    A = np.zeros((len(regions), len(regions)), bool)
    for c, nb in enumerate(neighbours):
        for n in nb:
            A[rix[c], rix[int(n)]] = A[rix[int(n)], rix[c]] = True
    np.fill_diagonal(A, False)
    return A


def future_assignments(trace: Trace, H: int) -> dict:
    """The region's own later assignments inside the label window (t_e, t_e + H] of each epoch, rebuilt from the
    logged table for any H: {"code" (E, R, F) (-1 past the last epoch), "t" (E, F), "F"}, F = ceil(H / D) - 1."""
    a = trace.arrays
    D = int(trace.meta["collector"]["D"])
    code, t = a["pol_code"], a["pol_t"]
    E, R = code.shape
    F = max(int(np.ceil(H / D)) - 1, 0)
    fc = np.full((E, R, F), -1, np.int16)
    ft = np.full((E, F), -1, np.int32)
    for j in range(1, F + 1):
        if E > j:
            fc[: E - j, :, j - 1] = code[j:]
            ft[: E - j, j - 1] = t[j:]
    return {"code": fc, "t": ft, "F": F}


class RandomizedStepPolicy(RandomizedJointPolicy):
    """v2 collector arbiter (``STEP_V2_DOC``). Same two-phase loop and plan semantics as v1 (``plans.decide``)."""

    def __init__(self, env: E6Env, mixture: StepMixture = DEFAULT_STEP_MIXTURE, D: int = 20, label_H: int = 90,
                 epoch_churn_cap: int | None = None, start_s: float | None = None):
        super().__init__(env, 1.0 - mixture.p_accept, D=D, epoch_churn_cap=epoch_churn_cap, start_s=start_s,
                         half_rule="slew")
        self.mix, self.label_H = mixture, int(label_H)
        n_max = int(np.ceil(env.total_s / self.D)) + 2                   # >= the epochs of any episode
        self.sched = step_schedule(env.cfg.seed, n_max, self.regions, mixture)
        self.pv = mixture.code_probs()
        self.adj = region_adjacency(env.plant.lay.neighbours, self.site)

    def _draw_epoch(self, epoch: int) -> tuple[list, list]:
        s = self.sched
        dflt = (STEP_BRANCHES.index("accept"), STEP_BRANCHES.index("quiet"))
        return [int(c) for c in s["code"][epoch]], [bool(b in dflt) for b in s["branch"][epoch]]

    def _class_key(self, code: int) -> tuple:
        rp = decode(code)
        return (rp["rb"], *(rp["mode"][x] for x in P.XAPPS if x in self.active))

    def table(self, trace: Trace) -> dict:
        out = super().table(trace)
        a, E, R, X = trace.arrays, len(self.rows), len(self.regions), len(P.XAPPS)
        s = self.sched
        codes = out["pol_code"].astype(int)
        if not np.array_equal(codes, s["code"][:E]):
            raise RuntimeError("played codes differ from the predeclared schedule")
        quiet = s["branch"][:E] == STEP_BRANCHES.index("quiet")
        out["pol_branch"], out["pol_quiet"] = s["branch"][:E].copy(), quiet
        out["pol_prop"] = s["prop"][:E].copy()
        cls = {}
        for c in range(N_CODES):
            k = self._class_key(c)
            cls[k] = cls.get(k, 0.0) + self.pv[c]
        pe = np.array([cls[self._class_key(c)] for c in codes.ravel()]).reshape(E, R)
        out["pol_prop_eff"] = np.where(quiet, 1.0, pe)
        mm = self.mix.mode_marginal()
        modes = out["pol_modes"].astype(int)                                 # (E, R, X) index into MODES
        frac_of = np.array([MODE_FRAC[m] for m in P.MODES], np.float32)
        out["pol_frac"] = frac_of[modes]
        out["pol_prop_x"] = np.where(quiet[..., None], 1.0, np.array([mm[m] for m in P.MODES])[modes])
        # ---- per-request epoch / fraction / deltas; per (epoch, region, xApp) aggregates
        xn = trace.meta["xapps"]
        kreg = np.searchsorted(self.regions, [int(self.site[k[1]]) for k in trace.meta["knobs"]])
        t, ln = out["pol_t"].astype(int), out["pol_len"].astype(int)
        e = np.searchsorted(t, a["rq_t"], side="right") - 1
        e0 = np.maximum(e, 0)
        ok = (e >= 0) & (a["rq_t"] < t[e0] + ln[e0]) if E else np.zeros(len(e), bool)
        rcol = kreg[a["rq_knob"]].astype(int)
        xcol = np.array([P.XAPPS.index(xn[i]) for i in a["rq_xapp"]], int)
        rq_frac = np.full(len(e), np.nan, np.float32)
        rq_frac[ok] = out["pol_frac"][e[ok], rcol[ok], xcol[ok]]
        changed = a["rq_out"] == OUT["ok"]
        d_req = a["rq_prop"] - a["rq_cur"]
        d_app = np.where(changed, a["rq_applied"] - a["rq_cur"], 0.0)
        out.update({"rq_epoch": np.where(ok, e, -1).astype(np.int32), "rq_frac": rq_frac, "rq_delta_req": d_req,
                    "rq_delta_app": d_app})
        idx = (e[ok], rcol[ok], xcol[ok])

        def count(mask):
            z = np.zeros((E, R, X), np.int32)
            np.add.at(z, idx, mask[ok].astype(np.int32))
            return z

        def total(v):
            z = np.zeros((E, R, X))
            np.add.at(z, idx, v[ok])
            return z

        out["pol_x_n_req"] = count(np.ones(len(e), bool))
        out["pol_x_elig"] = out["pol_x_n_req"] > 0
        out["pol_x_n_ack"] = count(np.isin(a["rq_out"], [OUT["ok"], OUT["noop"]]))
        out["pol_x_n_nack"] = count(np.isin(a["rq_out"], [OUT["actuator"], OUT["churn"], OUT["locked"]]))
        out["pol_x_n_rej"] = count(np.isin(a["rq_out"], [OUT["reject"], OUT["lock_set"]]))
        out["pol_x_n_changed"] = count(changed)
        out["pol_x_req_dabs"] = total(np.abs(d_req))
        out["pol_x_app_dabs"] = total(np.abs(d_app))
        nz = np.abs(d_req) > 1e-12
        n_nz = count(nz)
        ratio = np.divide(d_app, d_req, out=np.zeros_like(d_app), where=nz)
        out["pol_x_real_frac"] = np.divide(total(ratio), n_nz, out=np.full((E, R, X), np.nan), where=n_nz > 0)
        # ---- neighbouring regions' assignments
        A = self.adj.astype(float)
        out["pol_region_adj"] = self.adj.copy()
        out["pol_nbr_n_dev"] = ((codes != 0) @ A.T).astype(np.int16)
        out["pol_nbr_n_lock"] = ((modes == P.MODES.index("lock")).any(-1) @ A.T).astype(np.int16)
        out["pol_nbr_n_rb"] = (out["pol_rb"].astype(float) @ A.T).astype(np.int16)
        out["pol_nbr_frac"] = (np.einsum("rs,esx->erx", A, out["pol_frac"])
                               / np.maximum(A.sum(1), 1.0)[None, :, None]).astype(np.float32)
        # ---- the region's own future assignments inside the declared label horizon
        fut = future_assignments(Trace(out, {"collector": {"D": self.D}}), self.label_H)
        fc, F = fut["code"].astype(int), fut["F"]
        fp = np.full((E, R, F), np.nan)
        for j in range(1, F + 1):
            if E > j:
                fp[: E - j, :, j - 1] = out["pol_prop"][j:]
        fm = np.array([[P.MODES.index(decode(max(c, 0))["mode"][x]) for x in P.XAPPS] for c in fc.ravel()],
                      int).reshape(E, R, F, X)
        out["pol_fut_code"], out["pol_fut_t"], out["pol_fut_prop"] = fc.astype(np.int16), fut["t"], fp
        out["pol_fut_frac"] = np.where((fc >= 0)[..., None], frac_of[fm], np.nan).astype(np.float32)
        out["pol_fut_rb"] = np.where(fc >= 0, fc // len(P.MODES) ** X, -1).astype(np.int8)
        out["pol_fut_H"] = np.array(self.label_H, np.int32)
        return out


def collect_episode_v2(cfg, mixture: StepMixture = DEFAULT_STEP_MIXTURE, D: int = 20, label_H: int = 90,
                       epoch_churn_cap: int | None = None, churn_cap: int | None = None,
                       snapshot_s: int = 60) -> Trace:
    """One v2 randomized requested-step episode under ``wg3=True``: the trace + the v1 table + the v2 columns.
    ``mixture=ACCEPT_ALL_MIXTURE`` gives the paired accept-all reference on the same seed (same table layout)."""
    env = E6Env(cfg, log=False, wg3=True, churn_cap=churn_cap, trace=True, trace_snapshot_s=snapshot_s)
    col = RandomizedStepPolicy(env, mixture, D=D, label_H=label_H, epoch_churn_cap=epoch_churn_cap)
    while env.sec < env.total_s:
        obs = env.step_propose()
        dec = col.act(obs)
        env.step_apply(dec)
        col.record(dec)
    tr = env.get_trace()
    tr.arrays.update(col.table(tr))
    score = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
             for k, v in env.score().items()}
    tr.meta["collector"] = {"kind": "randomized_step_policy", "version": 2, "mixture": mixture.as_dict(),
                            "branches": list(STEP_BRANCHES), "fractions": list(FRACTIONS), "D": col.D,
                            "label_H": col.label_H, "key": KEY_V2, "eps": col.eps,
                            "epoch_churn_cap": epoch_churn_cap, "churn_cap": churn_cap, "n_codes": N_CODES,
                            "xapps_order": list(P.XAPPS), "modes": list(P.MODES), "active_xapps": sorted(col.active),
                            "rb_window_s": P.RB_WINDOW, "half_rule": col.half_rule, "score": score}
    return tr


# ================================================================================================= E6 DEV seed map
# Registered in docs/benchmark/SEED_REGISTRY.json (E6 "dev_reserved"; SOL_DATA_DESIGN.md (c)): 30 independent seeds
# per stratum (scenario x load, 6 strata), seed = base + 30 * stratum + j, stratum = 2 * SCENARIOS.index(scenario)
# + LOADS.index(load), j = 0..29; j < 20 -> "fit", j >= 20 -> "diag" (untouched diagnostics). Paired accept-all /
# no-probe references reuse the episode's own seed (same plant tape). Never TEST (>= 960000), never DEV 0-30.
DEV_SCENARIOS = ("base", "surge", "mistune")
DEV_LOADS = ("medium", "high")
DEV_PER_STRATUM, DEV_FIT = 30, 20
DEV_SEED_BASE = {"policy": 100000, "probe": 110000}


def dev_seed(kind: str, scenario: str, load: str, j: int) -> int:
    if not 0 <= int(j) < DEV_PER_STRATUM:
        raise ValueError(f"j must be in [0, {DEV_PER_STRATUM})")
    k = 2 * DEV_SCENARIOS.index(scenario) + DEV_LOADS.index(load)
    return DEV_SEED_BASE[kind] + DEV_PER_STRATUM * k + int(j)


def dev_stratum(seed: int) -> dict:
    """Inverse of ``dev_seed``: {kind, scenario, load, j, split}."""
    for kind, b in DEV_SEED_BASE.items():
        o = int(seed) - b
        if 0 <= o < DEV_PER_STRATUM * len(DEV_SCENARIOS) * len(DEV_LOADS):
            k, j = divmod(o, DEV_PER_STRATUM)
            return {"kind": kind, "scenario": DEV_SCENARIOS[k // 2], "load": DEV_LOADS[k % 2], "j": j,
                    "split": "fit" if j < DEV_FIT else "diag"}
    raise ValueError(f"seed {seed} is not an E6 DEV policy/probe seed")
