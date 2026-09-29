"""Gate B: budgeted per-region lookahead oracle restricted to O-RAN WG3 conflict-mitigation actions (DEV diagnostic).

Fixes vs oracle.py (see decision/REVIEW_method.md): copies are taken BETWEEN the propose and apply phases, so every
rollout replays the requests being decided; no relative writes (no own writes at all: WG3 actions only); freeze and
accept-all are always candidates; churn parity (applied knob changes <= the noarb run's on the same tape).

Per region (7 macro sites + 3 picos), a plan is {"mode": {xApp: accept|reject|half|lock}, "rb": 0|1}:
  accept / reject / half = modify to cur + (prop - cur)/2 / lock = reject + lock that knob for D s;
  rb = at plan start, roll back every knob of the region changed in the last RB_WINDOW s (to last-known-good).
Every D s: stage 1 = best of accept-all, freeze, incumbent, n_glob random network-wide plans; stage 2 = one pass of
coordinate descent over regions (n_loc random local alternatives each), keeping only exact improvements of
J = violated UE-s + w_ll * LL-violated UE-s + lam_e * kWh over H s on the true tape. Privileged: NOT deployable.
"""
from __future__ import annotations

import numpy as np

from cdd_oran.envs.e6 import config as C
from cdd_oran.envs.e6.env import E6Env

XAPPS = ("MRO", "TS", "ES", "SLICE")
MODES = ("accept", "reject", "half", "lock")
RB_WINDOW = 60.0


def objective(sla0, sla1, lam_e, w_ll):
    d = {k: sla1[k] - sla0[k] for k in ("viol_ue_s", "ll_viol", "energy_j")}
    return d["viol_ue_s"] + w_ll * d["ll_viol"] + lam_e * d["energy_j"] / 3.6e6


def uniform(mode, rb=0):
    return {"mode": {x: mode for x in XAPPS}, "rb": rb}


class WG3Oracle:
    def __init__(self, env, lam_e, w_ll, D=20, H=90, n_glob=6, n_loc=2):
        self.env, self.lam_e, self.w_ll, self.D, self.H = env, lam_e, w_ll, D, H
        self.n_glob, self.n_loc = n_glob, n_loc
        self.site = env.plant.lay.cell_site
        self.regions = sorted({int(x) for x in self.site})
        self.plan = {g: uniform("accept") for g in self.regions}
        self.until, self.picks, self.n_roll = -1.0, {}, 0
        self.rb_at = {}                   # knob -> time we rolled it back (never roll back our own rollback)

    # ---------------------------------------------------------------- plan -> decisions
    def _decide(self, env, plan, obs, first):
        dec = []
        for r in obs["requests"]:
            m = plan[int(self.site[r["knob"][1]])]["mode"][r["xapp"]]
            dec.append("accept" if m == "accept" else "reject" if m == "reject" else
                       ("lock", float(self.D)) if m == "lock" else ("modify", r["cur"] + 0.5 * (r["prop"] - r["cur"])))
        rb = []
        if first:
            now = obs["t"]
            rb = [k for k, t in env.last_change.items()
                  if now - t <= RB_WINDOW and plan[int(self.site[k[1]])]["rb"] and self.rb_at.get(k) != t]
        return {"decisions": dec, "writes": [], "rollback": rb}

    def _eval(self, plan, obs):
        sim = self.env.copy()
        sla0 = dict(sim.plant.sla)
        sim.step_apply(self._decide(sim, plan, obs, True))
        for _ in range(self.H - 1):
            if sim.sec >= sim.total_s:
                break
            o = sim.step_propose()
            sim.step_apply(self._decide(sim, plan, o, False))
        self.n_roll += 1
        return objective(sla0, sim.plant.sla, self.lam_e, self.w_ll)

    def _random(self, r):
        return {"mode": {x: MODES[int(r.integers(len(MODES)))] for x in XAPPS}, "rb": int(r.integers(2))}

    def choose(self, obs):
        env, t = self.env, obs["t"]
        r = np.random.default_rng([env.cfg.seed, int(t), 13])
        cands = [{g: uniform("accept") for g in self.regions}, {g: uniform("reject") for g in self.regions},
                 dict(self.plan)]
        for _ in range(self.n_glob):
            p = self._random(r)
            cands.append({g: p for g in self.regions})
        j_best, i_best = min((self._eval(c, obs), i) for i, c in enumerate(cands))
        best, changed = dict(cands[i_best]), 0
        for g in r.permutation(self.regions):
            for _ in range(self.n_loc):
                trial = dict(best)
                trial[int(g)] = self._random(r)
                j = self._eval(trial, obs)
                if j < j_best - 1e-9:
                    j_best, best, changed = j, trial, changed + 1
        key = f"stage1={['accept', 'freeze', 'incumbent'][i_best] if i_best < 3 else 'random'}|local={min(changed, 5)}"
        self.picks[key] = self.picks.get(key, 0) + 1
        return best

    def run(self):
        env = self.env
        while env.sec < env.total_s:
            obs = env.step_propose()
            first = False
            if obs["t"] >= env.cfg.warmup_s and obs["t"] >= self.until:
                self.plan, self.until, first = self.choose(obs), obs["t"] + self.D, True
            dec = self._decide(env, self.plan, obs, first)
            env.step_apply(dec)
            for k in dec["rollback"]:
                if k in env.last_change:
                    self.rb_at[k] = env.last_change[k]
        return env.score()


def run_job(cfg: C.E6Config, lam_e, w_ll, **kw):
    """noarb on the same tape sets the churn cap, then the budgeted WG3 oracle runs."""
    ref = E6Env(cfg, log=False)
    ref_score = ref.run()
    cap = ref_score["changes"]
    env = E6Env(cfg, log=False, wg3=True, churn_cap=cap)
    orc = WG3Oracle(env, lam_e, w_ll, **kw)
    s = orc.run()
    return {**s, "churn_cap": cap, "noarb_svr": ref_score["svr"], "picks": orc.picks, "n_roll": orc.n_roll}
