"""E6 v0 DEV: SLA effect of every on/off subset of the four xApps (does arbitration have room, and is freeze best?)."""
import itertools, json, sys, time
import numpy as np
from cdd_oran.envs.e6 import config as C
from cdd_oran.envs.e6 import xapps as X
from cdd_oran.envs.e6.env import E6Env
names = ("MRO", "TS", "ES", "SLICE")
out = open(sys.argv[1], "a")
for load in ("medium", "high"):
    for mask in itertools.product((0, 1), repeat=4):
        sub = tuple(n for n, m in zip(names, mask) if m)
        key = "sub_" + "_".join(sub) if sub else "none"
        X.MIXES[key] = tuple({"MRO": X.MRO, "TS": X.TS, "ES": X.ES, "SLICE": X.SliceSLA}[n] for n in sub)
        cfg = C.E6Config(seed=11, load=load, mobility="mixed", mix=key, warmup_s=120, scored_s=600)
        t = time.time()
        env = E6Env(cfg, log=False)
        s = env.run()
        out.write(json.dumps({"load": load, "subset": sub, **s, "secs": round(time.time() - t)}) + "\n")
        out.flush()
        print(load, sub, round(s["svr"], 1), flush=True)
