"""Diag H (tiny sim): re-score panel candidates on THIS machine and compare with the stored panel (cloud-run)
values. Any J0 mismatch = floating-point-level microstate difference; the effect replication corr measures how
robust the oracle's candidate effects are to an infinitesimal perturbation of the microstate."""
import json, sys, time
import numpy as np
from cdd_oran.decision import collect as CO, rank_eval as RE
from cdd_oran.decision.adapters.e6 import region_map
from cdd_oran.decision.world_model import DecisionContext, TrueSimWM
from cdd_oran.envs.e6 import config as C
from cdd_oran.envs.e6.env import E6Env

seed = int(sys.argv[1]); cands = sys.argv[2].split(",")
rows = [r for r in (json.loads(l) for l in open("scratchpad/e6_dev/runs/e6-rank1/all.jsonl"))
        if r.get("type") == "slot" and r["seed"] == seed]
st = CO.dev_stratum(seed)
cfg = C.E6Config(seed=seed, load=st["load"], mobility="mixed", mix="M4", warmup_s=120.0, scored_s=600.0, scenario=st["scenario"])
env = E6Env(cfg, log=False, wg3=True); site = region_map(env); regs = sorted({int(x) for x in site})
wm = TrueSimWM("accept_all"); T = time.time()
byt = {r["t"]: r for r in rows}
while env.sec < max(byt) + 1:
    obs = env.step_propose(); t = int(obs["t"])
    if t in byt:
        r = byt[t]; srcs = [c["src"][0] for c in r["cands"]]
        ctx = DecisionContext(obs, site, regs, 90, 20.0, 1e4, 5.0, dict(env.last_change), {}, env, {})
        plans = [RE.plan_of(r["cands"][srcs.index(c)]["codes"], regs) for c in ["accept_all"] + cands]
        J = [s.mean for s in wm.score(ctx, plans)]
        Jp = [r["oracle"]["J"][srcs.index(c)] for c in ["accept_all"] + cands]
        print(json.dumps({"seed": seed, "t": t, "J0_local": round(J[0], 2), "J0_panel": round(Jp[0], 2),
                          "d_local": [round(J[i] - J[0], 1) for i in range(1, len(J))],
                          "d_panel": [round(Jp[i] - Jp[0], 1) for i in range(1, len(J))], "s": round(time.time() - T)}), flush=True)
    env.step_apply({"decisions": ["accept"] * len(obs["requests"]), "writes": [], "rollback": []})
