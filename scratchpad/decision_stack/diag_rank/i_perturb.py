"""Diag I (tiny sim, single machine): effect of candidates on the true state vs on a copy whose UE positions are
nudged by 1e-6 m at the slot start (an infinitesimal microstate perturbation)."""
import json, sys, time
import numpy as np
from cdd_oran.decision import collect as CO, rank_eval as RE, plans as P
from cdd_oran.decision.adapters.e6 import region_map
from cdd_oran.decision.world_model import DecisionContext, TrueSimWM
from cdd_oran.envs.e6 import config as C
from cdd_oran.envs.e6.env import E6Env

seed = int(sys.argv[1]); slots = [int(x) for x in sys.argv[2].split(",")]
st = CO.dev_stratum(seed)
cfg = C.E6Config(seed=seed, load=st["load"], mobility="mixed", mix="M4", warmup_s=120.0, scored_s=600.0, scenario=st["scenario"])
env = E6Env(cfg, log=False, wg3=True); site = region_map(env); regs = sorted({int(x) for x in site})
wm = TrueSimWM("accept_all"); T = time.time()
single = lambda x, m: P.network(regs, {"mode": {y: (m if y == x else "accept") for y in P.XAPPS}, "rb": 0})
plans = [P.accept_all(regs), single("SLICE", "reject"), single("TS", "reject")]
class Nudged:
    def __init__(self, env): self.env = env
    def copy(self):
        s = self.env.copy(); s.plant.pos = s.plant.pos + 1e-6; return s
while env.sec < max(slots) + 1:
    obs = env.step_propose(); t = int(obs["t"])
    if t in slots:
        out = []
        for e in (env, Nudged(env)):
            ctx = DecisionContext(obs, site, regs, 90, 20.0, 1e4, 5.0, dict(env.last_change), {}, e, {})
            J = [s.mean for s in wm.score(ctx, plans)]
            out.append([round(J[0], 1)] + [round(j - J[0], 1) for j in J[1:]])
        print(json.dumps({"seed": seed, "t": t, "base": out[0], "nudged": out[1], "s": round(time.time() - T)}), flush=True)
    env.step_apply({"decisions": ["accept"] * len(obs["requests"]), "writes": [], "rollback": []})
