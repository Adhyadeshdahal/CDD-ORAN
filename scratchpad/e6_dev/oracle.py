"""E6 DEV headroom: perfect-knowledge lookahead oracle over per-xApp veto masks (accept/reject class only).

Every D seconds after warm-up the oracle copies the env (same exogenous tape), rolls each candidate mask forward H
seconds (the mask rejects every request of the masked xApps for the whole rollout), and keeps the mask with the lowest
    J = violated UE-seconds + w_ll * LL-violated UE-seconds + lam_e * energy kWh
for the next D seconds. lam_e / w_ll sweep a frontier instead of fixing an exchange rate. This is an upper-bound
diagnostic (privileged: sees the true plant and future tape), NOT a deployable arbiter.

Usage: PYTHONPATH=. python scratchpad/e6_dev/oracle.py --load medium --seed 11 --lam 0 --wll 0 --out X.jsonl
       arms: --arm oracle | noarb | freeze
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from cdd_oran.envs.e6 import config as C
from cdd_oran.envs.e6.baselines import freeze
from cdd_oran.envs.e6.env import E6Env

MASKS = [(), ("MRO",), ("TS",), ("ES",), ("SLICE",), ("TS", "ES"), ("MRO", "ES"), ("MRO", "TS", "ES", "SLICE")]


def masked(mask):
    def arb(obs):
        return {"decisions": ["reject" if r["xapp"] in mask else "accept" for r in obs["requests"]], "writes": []}
    return arb


def objective(sla0, sla1, lam_e, w_ll):
    d = {k: sla1[k] - sla0[k] for k in ("viol_ue_s", "ll_viol", "energy_j")}
    return d["viol_ue_s"] + w_ll * d["ll_viol"] + lam_e * d["energy_j"] / 3.6e6


class Oracle:
    def __init__(self, env, lam_e, w_ll, D=10, H=30):
        self.env, self.lam_e, self.w_ll, self.D, self.H = env, lam_e, w_ll, D, H
        self.mask, self.until, self.picks = (), -1, {}

    def __call__(self, obs):
        env = self.env
        if obs["t"] >= env.cfg.warmup_s and obs["t"] >= self.until:
            best = None
            for m in MASKS:
                c = env.copy()
                sla0 = dict(c.plant.sla)
                arb = masked(m)               # (the copy is taken mid-second: this second's requests are not
                #                                replayed in the rollout; they are decided with the chosen mask)
                for _ in range(self.H):
                    if c.sec >= c.total_s:
                        break
                    c.step(arb)
                j = objective(sla0, c.plant.sla, self.lam_e, self.w_ll)
                if best is None or j < best[0] - 1e-9:
                    best = (j, m)
            self.mask, self.until = best[1], obs["t"] + self.D
            self.picks[best[1]] = self.picks.get(best[1], 0) + 1
        return masked(self.mask)(obs)


XAPPS = ("MRO", "TS", "ES", "SLICE")
ACCEPT_ALL = {"mode": {x: "accept" for x in XAPPS}, "cio": 1.0, "car": 0, "ll": 1.0, "mob": 0}


def cand_writes(plant, c):
    """One-shot arbiter writes for candidate ``c`` from the CURRENT applied configuration (knob-type macros)."""
    w = []
    if c["cio"] != 1.0:
        s_, n_ = np.nonzero(plant.cio)
        w += [(("cio", int(s), int(n)), float(np.round(c["cio"] * plant.cio[s, n]))) for s, n in zip(s_, n_, strict=True)]
    if c["car"]:
        w += [(("carrier", int(i)), float(plant.n_trx[i])) for i in np.nonzero(plant.n_car < plant.n_trx)[0]]
    if c["ll"] != 1.0:
        w += [(("ll_ratio", int(i)), float(np.round(c["ll"] * plant.ll_ratio[i] * 20) / 20))
              for i in np.nonzero(plant.ll_ratio > 0)[0]]
    if c["mob"]:
        w += [(("hys", int(i)), 2.0) for i in np.nonzero(plant.hys != 2.0)[0]]
        w += [(("ttt", int(i)), 320.0) for i in np.nonzero(plant.ttt != 320)[0]]
    return w


def cand_arbiter(c, writes):
    state = {"writes": list(writes)}

    def arb(obs):
        dec = []
        for r in obs["requests"]:
            m = c["mode"][r["xapp"]]
            dec.append("accept" if m == "accept" else "reject" if m == "reject"
                       else ("modify", r["cur"] + 0.5 * (r["prop"] - r["cur"])))
        w, state["writes"] = state["writes"], []
        return {"decisions": dec, "writes": w}
    return arb


def cand_name(c):
    m = "".join({"accept": "a", "reject": "r", "half": "h"}[c["mode"][x]] for x in XAPPS)
    return f"{m}|cio{c['cio']:g}|car{c['car']}|ll{c['ll']:g}|mob{c['mob']}"


class ModOracle:
    """Joint MODIFY + own-write oracle: every D s, accept-all + incumbent + n_rand random joint candidates (per-xApp
    request mode accept/reject/half + one-shot knob-type writes), each rolled H s on the true tape; best is applied
    (writes once, request mode held D s)."""

    def __init__(self, env, lam_e, w_ll, D=15, H=60, n_rand=12):
        self.env, self.lam_e, self.w_ll, self.D, self.H, self.n_rand = env, lam_e, w_ll, D, H, n_rand
        self.cur, self.until, self.picks, self.inc = ACCEPT_ALL, -1, {}, ACCEPT_ALL
        self.arb = cand_arbiter(ACCEPT_ALL, [])

    def _random(self, r):
        return {"mode": {x: ("accept", "reject", "half")[int(r.integers(3))] for x in XAPPS},
                "cio": float(r.choice([1.0, 0.5, 0.0])), "car": int(r.integers(2)), "ll": float(r.choice([1.0, 0.5])),
                "mob": int(r.integers(2))}

    def __call__(self, obs):
        env, t = self.env, obs["t"]
        if t >= env.cfg.warmup_s and t >= self.until:
            r = np.random.default_rng([env.cfg.seed, int(t), 7])
            cands = [ACCEPT_ALL, self.inc] + [self._random(r) for _ in range(self.n_rand)]
            best = None
            for c in cands:
                sim = env.copy()
                sla0 = dict(sim.plant.sla)
                arb = cand_arbiter(c, cand_writes(sim.plant, c))
                for _ in range(self.H):
                    if sim.sec >= sim.total_s:
                        break
                    sim.step(arb)
                j = objective(sla0, sim.plant.sla, self.lam_e, self.w_ll)
                if best is None or j < best[0] - 1e-9:
                    best = (j, c)
            self.cur = self.inc = best[1]
            self.until = t + self.D
            self.arb = cand_arbiter(best[1], cand_writes(env.plant, best[1]))
            name = cand_name(best[1])
            self.picks[name] = self.picks.get(name, 0) + 1
        return self.arb(obs)



def _region_of_knob(k, site):
    return int(site[k[1]])


def region_writes(plant, acts, site):
    """Per-region one-shot writes: each region's macro touches only knobs whose first cell is in that region."""
    w = []
    for g, c in acts.items():
        for (k, v) in cand_writes(plant, c):
            if _region_of_knob(k, site) == g:
                w.append((k, v))
    return w


def region_arbiter(acts, writes, site):
    state = {"writes": list(writes)}

    def arb(obs):
        dec = []
        for r in obs["requests"]:
            m = acts[_region_of_knob(r["knob"], site)]["mode"][r["xapp"]]
            dec.append("accept" if m == "accept" else "reject" if m == "reject"
                       else ("modify", r["cur"] + 0.5 * (r["prop"] - r["cur"])))
        w, state["writes"] = state["writes"], []
        return {"decisions": dec, "writes": w}
    return arb


class SiteOracle(ModOracle):
    """Per-region joint oracle: stage 1 picks the best of accept-all / incumbent / n_glob uniform random network-wide
    candidates; stage 2 does one pass of coordinate descent over regions (sites + picos), n_loc random local
    alternatives each, keeping a change only if the network-wide rollout objective improves (same tape: exact)."""

    def __init__(self, env, lam_e, w_ll, D=20, H=90, n_glob=6, n_loc=2):
        super().__init__(env, lam_e, w_ll, D=D, H=H)
        self.site = env.plant.lay.cell_site
        self.regions = sorted(set(int(x) for x in self.site))
        self.n_glob, self.n_loc = n_glob, n_loc
        self.inc = {g: ACCEPT_ALL for g in self.regions}
        self.arb = region_arbiter(self.inc, [], self.site)
        self.n_roll = 0

    def _eval(self, acts):
        sim = self.env.copy()
        sla0 = dict(sim.plant.sla)
        arb = region_arbiter(acts, region_writes(sim.plant, acts, self.site), self.site)
        for _ in range(self.H):
            if sim.sec >= sim.total_s:
                break
            sim.step(arb)
        self.n_roll += 1
        return objective(sla0, sim.plant.sla, self.lam_e, self.w_ll)

    def __call__(self, obs):
        env, t = self.env, obs["t"]
        if t >= env.cfg.warmup_s and t >= self.until:
            r = np.random.default_rng([env.cfg.seed, int(t), 11])
            cands = [{g: ACCEPT_ALL for g in self.regions}, dict(self.inc)]
            for _ in range(self.n_glob):
                c = self._random(r)
                cands.append({g: c for g in self.regions})
            scored = [(self._eval(a), i) for i, a in enumerate(cands)]
            j_best, i_best = min(scored)
            best = dict(cands[i_best])
            changed = 0
            for g in r.permutation(self.regions):
                for _ in range(self.n_loc):
                    trial = dict(best)
                    trial[int(g)] = self._random(r)
                    j = self._eval(trial)
                    if j < j_best - 1e-9:
                        j_best, best, changed = j, trial, changed + 1
            self.inc = best
            self.until = t + self.D
            self.arb = region_arbiter(best, region_writes(env.plant, best, self.site), self.site)
            n_distinct = len({cand_name(c) for c in best.values()})
            key = f"stage1={i_best}|local_changes={min(changed, 5)}|distinct={n_distinct}"
            self.picks[key] = self.picks.get(key, 0) + 1
        return self.arb(obs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", default="medium")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--arm", default="oracle")
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--wll", type=float, default=0.0)
    ap.add_argument("--scored", type=float, default=600.0)
    ap.add_argument("--H", type=int, default=30)
    ap.add_argument("--out", default="scratchpad/e6_dev/oracle.jsonl")
    a = ap.parse_args()
    cfg = C.E6Config(seed=a.seed, load=a.load, mobility="mixed", mix="M4", warmup_s=120, scored_s=a.scored)
    env = E6Env(cfg, log=False)
    t = time.time()
    arb = {"noarb": None, "freeze": freeze}.get(a.arm)
    orc = None
    if a.arm == "oracle":
        orc = arb = Oracle(env, a.lam, a.wll, H=a.H)
    s = env.run(arb)
    rec = {"arm": a.arm, "load": a.load, "seed": a.seed, "lam": a.lam, "wll": a.wll, "H": a.H, **s,
           "picks": {"+".join(m) or "accept": n for m, n in (orc.picks.items() if orc else [])},
           "secs": round(time.time() - t, 1)}
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec[k] for k in ("arm", "load", "seed", "lam", "wll", "svr", "ll_viol", "energy_kwh", "picks",
                                          "secs")}))


if __name__ == "__main__":
    main()
