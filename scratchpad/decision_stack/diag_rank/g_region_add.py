"""Diag G (tiny sim): region additivity + timing sensitivity of the SLICE:reject effect at one slot.
Per region r: plan = SLICE reject in r only for D s (then accept-all), D in {20, 19}; network plan (all regions)."""
import sys, time
import numpy as np
from cdd_oran.decision import plans as P
from cdd_oran.decision.adapters.e6 import region_map
from cdd_oran.decision.world_model import DecisionContext, TrueSimWM
from cdd_oran.envs.e6 import config as C
from cdd_oran.envs.e6.env import E6Env

seed, scn, load, slot, xapp = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
cfg = C.E6Config(seed=seed, load=load, mobility="mixed", mix="M4", warmup_s=120.0, scored_s=600.0, scenario=scn)
env = E6Env(cfg, log=False, wg3=True); site = region_map(env); regs = sorted({int(x) for x in site})
T = time.time()
while True:
    obs = env.step_propose()
    if int(obs["t"]) == slot:
        break
    env.step_apply({"decisions": ["accept"] * len(obs["requests"]), "writes": [], "rollback": []})
wm = TrueSimWM("accept_all")
rp = {"mode": {y: ("reject" if y == xapp else "accept") for y in P.XAPPS}, "rb": 0}
acc = P.uniform("accept")
def J(plan, D):
    ctx = DecisionContext(obs, site, regs, 90, float(D), 1e4, 5.0, dict(env.last_change), {}, env, {})
    return wm.score(ctx, [plan])[0].mean
J0 = J(P.accept_all(regs), 20)
net = J(P.network(regs, rp), 20) - J0
per20 = [J({g: (rp if g == r else acc) for g in regs}, 20) - J0 for r in regs]
per19 = [J({g: (rp if g == r else acc) for g in regs}, 19) - J0 for r in regs]
print(f"seed {seed} t {slot} {xapp}:reject  network {net:+.1f}  sum per-region {sum(per20):+.1f}")
print("  per-region D=20:", np.round(per20, 1).tolist())
print("  per-region D=19:", np.round(per19, 1).tolist())
print("  corr(D20, D19) %.2f ; mean|D20-D19| %.1f ; mean|D20| %.1f ; elapsed %.0fs" % (
    np.corrcoef(per20, per19)[0, 1], np.mean(np.abs(np.subtract(per20, per19))), np.mean(np.abs(per20)), time.time() - T))
