"""E6 gate A (controllability, scratchpad/e6_dev/decision/DECISION.md): how much SVR can ANY static configuration or
xApp subset remove, per scenario and load? Shards of (scenario, load, seed, arm) jobs, resumable.

  PYTHONPATH=. python scratchpad/e6_dev/gate_a.py list
  PYTHONPATH=. python scratchpad/e6_dev/gate_a.py run --part i/k --out res_i.jsonl [--smoke]
  PYTHONPATH=. python scratchpad/e6_dev/gate_a.py summarize res_*.jsonl

Arms (all with mix M4 deployed, so every arm sees the same xApp variant draws):
  noarb        accept all (= subset of all four)
  freeze       reject all (= empty subset; network stays at the 3GPP-default initial configuration)
  sub:A+B      subset((A, B)) for the 14 non-trivial subsets (includes each xApp alone and TS+MRO)
  tuned        HINDSIGHT tuned static: coordinate descent over baselines.tuned_static_grid() on the SAME seed/tape,
               one job running the whole descent (path + best recorded). Upper reference for static control.
"""
from __future__ import annotations

import dataclasses
import glob
import itertools
import json
import os
import sys
import time

import numpy as np

from cdd_oran.envs.e6 import config as C
from cdd_oran.envs.e6.baselines import (
    freeze,
    static_policy,
    subset,
    tuned_static,
)
from cdd_oran.envs.e6.env import E6Env

SCENARIOS = ("base", "surge", "mistune")
LOADS = ("medium", "high")
SEEDS = range(11, 16)                                    # DEV
XAPPS = ("MRO", "TS", "ES", "SLICE")
SUBSETS = [tuple(x for x, m in zip(XAPPS, mask, strict=True) if m) for mask in itertools.product((0, 1), repeat=4)]
ARMS = ["tuned", "noarb", "freeze"] + ["sub:" + "+".join(s) for s in SUBSETS if 0 < len(s) < 4]   # tuned first: longest
WARMUP, SCORED, SMOKE_SCORED = 120.0, 600.0, 30.0
HAS_SCENARIO = "scenario" in {f.name for f in dataclasses.fields(C.E6Config)}
OWN_KPI = {"MRO": ("rlf_per_ue_h", "ho_per_ue_h", "pingpong"), "TS": ("embb_viol",), "ES": ("energy_kwh",),
           "SLICE": ("ll_viol",)}
GATE_RULE = ("PASS iff in surge or mistune (at every load of that scenario): min(TS alone, TS+MRO) mean SVR ratio "
             "<= 0.85 x freeze AND controllable fraction = (SVR_freeze - SVR_best_static_or_subset) / SVR_freeze "
             ">= 0.35")


def jobs():
    return [(sc, ld, s, arm) for sc, ld, arm, s in itertools.product(SCENARIOS, LOADS, ARMS, SEEDS)]


def make_cfg(scenario, load, seed, scored):
    kw = dict(seed=seed, load=load, mobility="mixed", mix="M4", warmup_s=WARMUP, scored_s=scored)
    if HAS_SCENARIO:
        kw["scenario"] = scenario
    elif scenario != "base":
        raise RuntimeError(f"E6Config has no 'scenario' field; cannot run scenario {scenario!r}")
    return C.E6Config(**kw)


def episode(cfg, arm, spec=None):
    env = E6Env(cfg, log=False)
    if spec is not None:
        arb = static_policy(env, spec)
    elif arm == "noarb":
        arb = None
    elif arm == "freeze":
        arb = freeze
    else:
        arb = subset(arm[4:].split("+"))
    return env.run(arb)


def run_job(scenario, load, seed, arm, scored=SCORED):
    cfg = make_cfg(scenario, load, seed, scored)
    t = time.time()
    rec = {"scenario": scenario, "load": load, "seed": seed, "arm": arm, "scored_s": scored}
    if arm == "tuned":
        scores = {}

        def evaluate(spec):
            s = episode(cfg, arm, spec)
            scores[json.dumps(spec, sort_keys=True)] = s
            return s["svr"]

        res = tuned_static(evaluate)
        rec.update(scores[json.dumps(res["best"], sort_keys=True)])
        rec.update(best=res["best"], n_eval=res["n_eval"],
                   path=[dict(p, score=scores[json.dumps(p["spec"], sort_keys=True)]) for p in res["path"]])
    else:
        rec.update(episode(cfg, arm))
    rec["secs"] = round(time.time() - t, 1)
    return rec


def _key(r):
    return r["scenario"], r["load"], r["seed"], r["arm"]


def run(part, out, smoke=False):
    i, k = map(int, part.split("/"))
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                done.add(_key(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue
    for j, job in enumerate(jobs()):
        if j % k != i or job in done:
            continue
        rec = run_job(*job, scored=SMOKE_SCORED if smoke else SCORED)
        with open(out, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        print(json.dumps({x: rec[x] for x in ("scenario", "load", "seed", "arm", "svr", "energy_kwh", "secs")},
                         default=float), flush=True)
        if smoke:
            break


# ---------------------------------------------------------------------------------------------------- summary
def _ratio(res, arm, seeds, key="svr"):
    """Mean over seeds of arm/freeze (paired by seed; seeds with freeze == 0 skipped)."""
    r = [res[(s, arm)][key] / res[(s, "freeze")][key] for s in seeds
         if (s, arm) in res and res[(s, "freeze")][key] > 0]
    return float(np.mean(r)) if r else float("nan")


def summarize(paths):
    recs = {}
    for p in paths:
        for f in glob.glob(p):
            for line in open(f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                recs[_key(r)] = r                               # last record wins on duplicates
    if len({r.get("scored_s") for r in recs.values()}) > 1:
        print("WARNING: records mix scored_s values", sorted({r.get("scored_s") for r in recs.values()}))
    cells, rows = sorted({(k[0], k[1]) for k in recs}, key=lambda c: (SCENARIOS.index(c[0]), LOADS.index(c[1]))), []
    for sc, ld in cells:
        res = {(k[2], k[3]): r for k, r in recs.items() if k[:2] == (sc, ld)}
        seeds = sorted(s for s, a in res if a == "freeze")
        sub_arms = [a for a in ARMS if a.startswith("sub:")]
        full = [a for a in sub_arms + ["noarb", "freeze"] if all((s, a) in res for s in seeds)]
        if not seeds:
            continue
        mean = {a: float(np.mean([res[(s, a)]["svr"] for s in seeds])) for a in full}
        best_sub = min(full, key=mean.get) if full else None       # one subset for all seeds (chosen on the mean)
        tuned_seeds = [s for s in seeds if (s, "tuned") in res]
        per_seed_best = [min([res[(s, best_sub)]["svr"]] + ([res[(s, "tuned")]["svr"]] if (s, "tuned") in res else []))
                         for s in seeds] if best_sub else []
        f_svr = mean.get("freeze", float(np.mean([res[(s, "freeze")]["svr"] for s in seeds])))
        ctrl = (f_svr - float(np.mean(per_seed_best))) / f_svr if per_seed_best and f_svr > 0 else float("nan")
        ts, tsm = _ratio(res, "sub:TS", seeds), _ratio(res, "sub:MRO+TS", seeds)
        own = {}
        for x, kpis in OWN_KPI.items():
            own[x] = {kp: {"alone": float(np.mean([res[(s, "sub:" + x)][kp] for s in seeds if (s, "sub:" + x) in res]))
                           if any((s, "sub:" + x) in res for s in seeds) else float("nan"),
                           "freeze": float(np.mean([res[(s, "freeze")][kp] for s in seeds]))} for kp in kpis}
            for d in own[x].values():
                d["improves"] = None if np.isnan(d["alone"]) else bool(d["alone"] < d["freeze"])
        cond = bool(np.nanmin([ts, tsm]) <= 0.85 and ctrl >= 0.35) if not np.isnan([ts, tsm]).all() else False
        rows.append({"scenario": sc, "load": ld, "n_seeds": len(seeds), "n_tuned": len(tuned_seeds),
                     "svr_freeze": f_svr, "ratio_noarb": _ratio(res, "noarb", seeds), "ratio_TS": ts,
                     "ratio_TS_MRO": tsm, "best_subset": best_sub,
                     "ratio_best_subset": _ratio(res, best_sub, seeds) if best_sub else float("nan"),
                     "ratio_tuned": _ratio(res, "tuned", tuned_seeds),
                     "tuned_best": {s: res[(s, "tuned")]["best"] for s in tuned_seeds},
                     "controllable_frac": ctrl, "cell_pass": cond, "own_kpi": own})
    verdict = {}
    for sc in ("surge", "mistune"):
        rs = [r for r in rows if r["scenario"] == sc]
        verdict[sc] = bool(rs) and len(rs) == len(LOADS) and all(r["cell_pass"] for r in rs)
    out = {"rule": GATE_RULE, "cells": rows, "scenario_pass": verdict, "gate": "PASS" if any(verdict.values()) else "FAIL"}
    print(f"{'scenario':8} {'load':6} {'n':>2} {'SVR_frz':>8} {'noarb':>6} {'TS':>6} {'TS+MRO':>6} {'bestsub':>7} "
          f"{'tuned':>6} {'ctrl':>6} pass  best subset")
    for r in rows:
        print(f"{r['scenario']:8} {r['load']:6} {r['n_seeds']:>2} {r['svr_freeze']:8.1f} {r['ratio_noarb']:6.3f} "
              f"{r['ratio_TS']:6.3f} {r['ratio_TS_MRO']:6.3f} {r['ratio_best_subset']:7.3f} {r['ratio_tuned']:6.3f} "
              f"{r['controllable_frac']:6.3f} {'Y' if r['cell_pass'] else 'n':4}  {r['best_subset']}")
    print("\nown-KPI (xApp alone vs freeze, mean over seeds):")
    for r in rows:
        print(f"  {r['scenario']}/{r['load']}: " + "; ".join(
            f"{x} {kp} {d['alone']:.3g} vs {d['freeze']:.3g}{' (n/a)' if d['improves'] is None else '' if d['improves'] else ' (NOT better)'}"
            for x, kps in r["own_kpi"].items() for kp, d in kps.items()))
    print("\nrule:", GATE_RULE)
    print("per scenario:", verdict, "-> gate", out["gate"])
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "list":
        js = jobs()
        print(len(js), "jobs:", {a: sum(j[3] == a for j in js) for a in ARMS}, "| scenario field in E6Config:",
              HAS_SCENARIO)
    elif cmd == "summarize":
        o = summarize(sys.argv[2:])
        if os.environ.get("GATE_A_JSON"):
            json.dump(o, open(os.environ["GATE_A_JSON"], "w"), indent=1, default=float)
    else:
        args = sys.argv[2:]
        smoke = "--smoke" in args
        args = [a for a in args if a != "--smoke"]
        a = dict(zip(args[::2], args[1::2], strict=True))
        run(a["--part"], a["--out"], smoke=smoke)
