"""Diag E (tiny sim): sensitivity of J(t, t+H] to a MINIMAL perturbation (reject exactly one xApp request) vs the
network-wide single-xApp candidates. Diag seed 100020 (base-medium, a panel episode), slots t=120, 210."""
import sys, time
import numpy as np
from cdd_oran.decision import plans as P, rank_eval as RE
from cdd_oran.decision.world_model import objective
from cdd_oran.envs.e6.env import E6Env

seed, scn, load = int(sys.argv[1]), sys.argv[2], sys.argv[3]
slots = [int(x) for x in sys.argv[4].split(",")]
nper = int(sys.argv[5])
H, D = 90, 20
cfg = RE.cfg_of(scn, load, seed) if hasattr(RE, "cfg_of") else None
from cdd_oran.envs.e6 import config as C
cfg = C.E6Config(seed=seed, load=load, mobility="mixed", mix="M4", warmup_s=120.0, scored_s=600.0, scenario=scn)
env = E6Env(cfg, log=False, wg3=True)
T0 = time.time()

def roll(env, first_dec):
    sim = env.copy(); sla0 = dict(sim.plant.sla)
    sim.step_apply(first_dec)
    for i in range(1, H + 1):
        o = sim.step_propose()
        sim.step_apply({"decisions": ["accept"] * len(o["requests"]), "writes": [], "rollback": []})
    return objective(sla0, sim.plant.sla, 1e4, 5.0)

armed = set()
while env.sec < max(slots) + 60:
    obs = env.step_propose()
    t = int(obs["t"])
    if t in slots:
        armed.add(t)
    if armed and obs["requests"]:
        armed.clear()
        rq = obs["requests"]
        acc = {"decisions": ["accept"] * len(rq), "writes": [], "rollback": []}
        J0 = roll(env, acc); J0b = J0
        out = []
        for i in range(min(nper, len(rq))):
            dec = dict(acc); dec["decisions"] = ["reject" if j == i else "accept" for j in range(len(rq))]
            out.append((rq[i]["xapp"], rq[i]["knob"][0], round(rq[i]["prop"] - rq[i]["cur"], 3), round(roll(env, dec) - J0, 1)))
        print(f"t={t} nreq={len(rq)} J0={J0:.1f} repeat-diff={J0b-J0:.3g}")
        for o in out:
            print("   reject one", o)
        print("   elapsed %.0fs" % (time.time() - T0), flush=True)
    env.step_apply({"decisions": ["accept"] * len(obs["requests"]), "writes": [], "rollback": []})
