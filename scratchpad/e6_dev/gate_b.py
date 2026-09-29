"""Gate B grid (Kaggle): budgeted WG3 per-region oracle per (scenario, load, seed, objective). Resumable shards.

  PYTHONPATH=. python scratchpad/e6_dev/gate_b.py list | run --part i/k --out res.jsonl [--smoke]
Baselines (noarb, freeze, subsets, tuned static) come from gate_a.py on the same seeds and tapes.
"""
from __future__ import annotations

import dataclasses
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wg3_oracle import run_job  # noqa: E402

from cdd_oran.envs.e6 import config as C  # noqa: E402

SCENARIOS = ("surge", "mistune", "base")
LOADS = ("medium", "high")
SEEDS = range(11, 14)                     # DEV
OBJ = [(1e4, 5.0), (0.0, 0.0)]           # priced (primary) first, then SLA-only
SCORED = 600.0


def jobs():
    return [(sc, ld, s, lam, w) for (lam, w), sc, ld, s in itertools.product(OBJ, SCENARIOS, LOADS, SEEDS)]


def make_cfg(sc, ld, s, scored):
    kw = dict(seed=s, load=ld, mobility="mixed", mix="M4", warmup_s=120, scored_s=scored)
    if "scenario" in {f.name for f in dataclasses.fields(C.E6Config)}:
        kw["scenario"] = sc
    elif sc != "base":
        raise SystemExit(f"E6Config has no 'scenario' field; cannot run {sc}")
    return C.E6Config(**kw)


def run(part, out, smoke=False):
    i, k = map(int, part.split("/"))
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((r["scenario"], r["load"], r["seed"], r["lam"], r["wll"]))
    for j, (sc, ld, s, lam, w) in enumerate(jobs()):
        if j % k != i or (sc, ld, s, lam, w) in done:
            continue
        t = time.time()
        rec = {"arm": "wg3_oracle", "scenario": sc, "load": ld, "seed": s, "lam": lam, "wll": w,
               **run_job(make_cfg(sc, ld, s, 30.0 if smoke else SCORED), lam, w), "secs": round(time.time() - t, 1)}
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({x: rec[x] for x in ("scenario", "load", "seed", "lam", "svr", "noarb_svr", "changes",
                                              "churn_cap", "secs")}), flush=True)
        if smoke:
            break


if __name__ == "__main__":
    if sys.argv[1] == "list":
        print(len(jobs()), "jobs")
    else:
        a = dict(zip(sys.argv[2::2], sys.argv[3::2], strict=False))
        run(a["--part"], a["--out"], smoke="--smoke" in sys.argv)
