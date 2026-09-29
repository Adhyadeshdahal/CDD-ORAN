"""E6 DEV headroom grid (cloud): shards of (load, seed, arm, lam, wll) jobs, resumable.

  PYTHONPATH=. python scratchpad/e6_dev/grid.py run --part i/k --out res_i.jsonl
  PYTHONPATH=. python scratchpad/e6_dev/grid.py list
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle import ModOracle, Oracle, SiteOracle  # noqa: E402

from cdd_oran.envs.e6 import config as C  # noqa: E402
from cdd_oran.envs.e6.baselines import freeze  # noqa: E402
from cdd_oran.envs.e6.env import E6Env  # noqa: E402

LOADS = ("medium", "high")
SEEDS = range(11, 14)                     # DEV (noarb/freeze for 11-15 are in headroom_all.jsonl; deterministic)
ARMS = [("site", 0.0, 0.0), ("site", 1e4, 5.0)]   # round 3 (rounds 1-2 in headroom*_all.jsonl)
SCORED = 600.0


def jobs():
    return [(ld, s, *arm) for arm, ld, s in itertools.product(ARMS, LOADS, SEEDS)]


def run_job(load, seed, arm, lam, wll):
    cfg = C.E6Config(seed=seed, load=load, mobility="mixed", mix="M4", warmup_s=120, scored_s=SCORED)
    env = E6Env(cfg, log=False, write_budget=1e9)     # diagnostic: writes unbudgeted, count reported
    t = time.time()
    orc = {"oracle": lambda: Oracle(env, lam, wll), "veto90": lambda: Oracle(env, lam, wll, H=90),
           "mod": lambda: ModOracle(env, lam, wll), "site": lambda: SiteOracle(env, lam, wll)}.get(arm, lambda: None)()
    s = env.run(orc if orc else {"noarb": None, "freeze": freeze}[arm])
    return {"arm": arm, "load": load, "seed": seed, "lam": lam, "wll": wll, **s,
            "picks": {(m if isinstance(m, str) else "+".join(m) or "accept"): n for m, n in (orc.picks.items() if orc else [])},
            "n_roll": getattr(orc, "n_roll", None), "secs": round(time.time() - t, 1)}


def run(part, out):
    i, k = map(int, part.split("/"))
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((r["load"], r["seed"], r["arm"], r["lam"], r["wll"]))
    for j, job in enumerate(jobs()):
        if j % k != i or job in done:
            continue
        rec = run_job(*job)
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({x: rec[x] for x in ("arm", "load", "seed", "lam", "wll", "svr", "energy_kwh", "secs")}),
              flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "list":
        print(len(jobs()), "jobs")
    else:
        a = dict(zip(sys.argv[2::2], sys.argv[3::2], strict=True))
        run(a["--part"], a["--out"])
