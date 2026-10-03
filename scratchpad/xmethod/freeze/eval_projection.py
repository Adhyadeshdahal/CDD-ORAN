"""EVAL launch projection (freeze-prep): unit counts of specs/eval/full.json for S in 40..100 and the CPU-h / session-h
they cost, from the DEV cost tables. Estimate only; the T3 decision reads the DEV costs, not this file.

  uv run python scratchpad/xmethod/freeze/eval_projection.py --dev <results/dev dir> [--cdl-records GLOB]
      [--pmrt-nl-agg results/pmrt_nl/agg_valid_gbm.json] --out scratchpad/xmethod/freeze/eval_projection.json

Cost per unit (CPU-s, mean per dataset of the (arm, world, regime, n) cell):
- 10 classic / PMRT arms: <dev>/full/agg.json (Kaggle); CI arms: <dev>/ci_c/agg.json (Kaggle / Colab / Lightning).
- cdl: the cdl DEV run's records so far (--cdl-records, any host) by n; an n without records is scaled from the
  largest measured n by the training steps min(16000, ceil(130 n / 128)) (cdl.py budget); without records the
  cdl worker's table <dev>/cdl_cost_table.json.
- pmrt_nl_eq: the R-42 gbm validity run (--pmrt-nl-agg, mean by n); an n without data: power law in n fitted on
  the measured n (log-log least squares).
- dataset generation: mean gen_cpu_s of the (world, regime, n) cell, once per dataset.
Wall (R-55): EVAL runs 1 process per vCPU (Kaggle 4 per session, Colab 2 per job), so a process's wall ~ its
CPU-s. The DEV CPU-s came from oversubscribed sessions (8 / 4 processes on 4 / 2 vCPU): an upper estimate.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import eval_analysis as E  # noqa: E402

SPEC = os.path.join(os.path.dirname(HERE), "specs", "eval", "full.json")
S_GRID = (40, 50, 60, 70, 80, 90, 100)
EVAL0 = 3_100_000


def fill(spec: dict, S: int) -> dict:
    out = json.loads(json.dumps(spec))
    for b in out["blocks"]:
        if b["seeds"] == "TBD":
            b["seeds"] = [EVAL0, EVAL0 + S - 1]
        elif b["seeds"] == "TBD_HALF":
            b["seeds"] = [EVAL0, EVAL0 + math.ceil(S / 2) - 1]
        elif isinstance(b["seeds"], str):
            raise ValueError(f"unfilled seed placeholder {b['seeds']!r}")
    return out


def steps(n: int) -> int:
    return min(16000, math.ceil(130 * n / 128))


def cost_tables(dev: str, cdl_glob: str | None, nl_agg: str | None) -> tuple[dict, dict, dict, dict]:
    cpu, wall, gen, basis = {}, {}, {}, {}
    for f in ("full/agg.json", "ci_c/agg.json"):
        agg = json.load(open(os.path.join(dev, f), encoding="utf-8"))
        for k, v in agg["cost"].items():
            cpu[k] = v["cpu_s_mean"]
            if v.get("wall_s_mean"):
                wall[k] = v["wall_s_mean"]
            if v.get("gen_cpu_s_mean"):
                gen.setdefault(k.split("|", 1)[1], []).append(v["gen_cpu_s_mean"])
            basis[k.split("|", 1)[0]] = f"DEV {f.split('/')[0]} (measured per cell)"
    # cdl
    recs = defaultdict(lambda: [[], []])
    for f in sorted(glob.glob(cdl_glob)) if cdl_glob else []:
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("status") == "ok" and r.get("arm") == "cdl":
                x = recs[int(r["job"]["n"])]
                x[0].append(float(r["method_cpu_s"]))
                x[1].append(float(r.get("child_wall_s") or r.get("wall_s")))
    cdl_n = {n: (float(np.mean(c)), float(np.mean(w)), float(np.max(c)), len(c)) for n, (c, w) in recs.items()}
    if cdl_n:
        top = max(cdl_n)
        basis["cdl"] = (f"cdl DEV run records so far ({sum(v[3] for v in cdl_n.values())} units, n "
                        f"{sorted(cdl_n)}); other n scaled from n {top} by training steps")
    else:
        tab = json.load(open(os.path.join(dev, "cdl_cost_table.json"), encoding="utf-8"))
        basis["cdl"] = "cdl worker cost table (cdl_cost_table.json)"
    for n in (500, 1000, 4000, 8000, 24000):
        if cdl_n:
            c, w = (cdl_n[n][0], cdl_n[n][1]) if n in cdl_n else (cdl_n[top][0] * steps(n) / steps(top),
                                                                   cdl_n[top][1] * steps(n) / steps(top))
        else:
            c, w = tab[f"cdl|E1|R1|n{n}"], None
        cpu[f"cdl|*|*|n{n}"] = c
        if w:
            wall[f"cdl|*|*|n{n}"] = w
    # pmrt_nl_eq
    if nl_agg:
        nl = {int(k.split("|n")[1]): v["mean"] for k, v in json.load(open(nl_agg, encoding="utf-8"))["cost"].items()
              if k.startswith("gbm|")}
        xs = sorted(nl)
        b, a = np.polyfit(np.log(xs), np.log([nl[x] for x in xs]), 1)
        for n in (500, 1000, 4000, 8000, 24000):
            cpu[f"pmrt_nl_eq|*|*|n{n}"] = nl[n] if n in nl else float(math.exp(a + b * math.log(n)))
        basis["pmrt_nl_eq"] = (f"R-42 gbm validity run (mean by n, n {xs}); n 8000 / 24000 extrapolated n^{b:.2f}")
    return cpu, wall, {k: float(np.mean(v)) for k, v in gen.items()}, basis


def t3_max(dev: str, cdl_glob: str | None, nl_agg: str | None) -> dict:
    """arm -> n -> max DEV CPU-s per unit (any host, NOT converted by a speed factor), for the T3 read."""
    out: dict = defaultdict(dict)
    for f in ("full/agg.json", "ci_c/agg.json"):
        for k, v in json.load(open(os.path.join(dev, f), encoding="utf-8"))["cost"].items():
            arm, n = k.split("|", 1)[0], k.rsplit("|n", 1)[1]
            out[arm][n] = round(max(out[arm].get(n, 0.0), v["cpu_s_max"]), 1)
    for f in sorted(glob.glob(cdl_glob)) if cdl_glob else []:
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("status") == "ok" and r.get("arm") == "cdl":
                n = str(r["job"]["n"])
                out["cdl"][n] = round(max(out["cdl"].get(n, 0.0), float(r["method_cpu_s"])), 1)
    if nl_agg:
        for k, v in json.load(open(nl_agg, encoding="utf-8"))["cost"].items():
            if k.startswith("gbm|"):
                out["pmrt_nl_eq"][k.split("|n")[1]] = v["max"]
    return {a: dict(sorted(d.items(), key=lambda x: int(x[0]))) for a, d in sorted(out.items())}


def unit_cost(tab: dict, arm: str, w: str, r: str, n: int) -> float | None:
    return tab.get(f"{arm}|{w}|{r}|n{n}", tab.get(f"{arm}|*|*|n{n}"))


def project(spec: dict, S: int, cpu: dict, gen: dict) -> dict:
    units = E.planned_units(fill(spec, S))
    by_arm, by_n, by_role = defaultdict(float), defaultdict(float), defaultdict(float)
    w_by_arm = defaultdict(float)
    missing, ds = set(), set()
    gen_s = 0.0
    for u in units.values():
        c = unit_cost(cpu, u["arm"], u["world"], u["regime"], u["n"])
        if c is None:
            missing.add(f"{u['arm']}|n{u['n']}")
            continue
        by_arm[u["arm"]] += c
        by_n[u["n"]] += c
        by_role[u["role"]] += c
        w_by_arm[u["arm"]] += c                                  # 1 process per vCPU (R-55): wall ~ CPU-s
        d = (u["world"], u["regime"], u["lam"], u["n"], u["kappa"], u["seed"])
        if d not in ds:
            ds.add(d)
            gen_s += gen.get(f"{u['world']}|{u['regime']}|n{u['n']}", 0.0)
    cpu_h = (sum(by_arm.values()) + gen_s) / 3600
    return {"S": S, "units": len(units), "datasets": len(ds),
            "tune_units": sum(u["role"] == "tune" for u in units.values()),
            "cpu_h": round(cpu_h, 1), "process_wall_h": round((sum(w_by_arm.values()) + gen_s) / 3600, 1),
            "cpu_h_generation": round(gen_s / 3600, 1),
            "cpu_h_by_arm": {k: round(v / 3600, 1) for k, v in sorted(by_arm.items(), key=lambda x: -x[1])},
            "cpu_h_by_n": {str(k): round(v / 3600, 1) for k, v in sorted(by_n.items())},
            "cpu_h_by_role": {k: round(v / 3600, 1) for k, v in by_role.items()},
            "units_without_cost": sorted(missing)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--cdl-records", default=None)
    ap.add_argument("--pmrt-nl-agg", default=None)
    ap.add_argument("--kaggle", type=int, default=4, help="concurrent Kaggle sessions")
    ap.add_argument("--colab", type=int, default=3, help="concurrent Colab CPU jobs")
    ap.add_argument("--kaggle-procs", type=int, default=4, help="processes per Kaggle session = vCPU (R-55)")
    ap.add_argument("--colab-procs", type=int, default=2, help="processes per Colab job = vCPU (R-55)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = json.load(open(SPEC, encoding="utf-8"))
    cpu, _, gen, basis = cost_tables(a.dev, a.cdl_records, a.pmrt_nl_agg)
    kp, cp = a.kaggle_procs * a.kaggle, a.colab_procs * a.colab          # concurrent processes (1 per vCPU, R-55)
    slots = kp + cp
    out = {"spec": "scratchpad/xmethod/specs/eval/full.json", "cost_basis": basis,
           "platforms": {"kaggle_sessions": a.kaggle, "kaggle_processes_per_session": a.kaggle_procs,
                         "colab_jobs": a.colab, "colab_processes_per_job": a.colab_procs, "lightning": 0},
           "t3_max_cpu_s_by_arm_n": t3_max(a.dev, a.cdl_records, a.pmrt_nl_agg), "by_S": {}}
    for S in S_GRID:
        p = project(spec, S, cpu, gen)
        p["wall_h_all_platforms"] = round(p["process_wall_h"] / slots, 1)
        p["kaggle_share_process_wall_h"] = round(p["process_wall_h"] * kp / slots, 1)
        p["colab_share_process_wall_h"] = round(p["process_wall_h"] * cp / slots, 1)
        out["by_S"][str(S)] = p
    json.dump(out, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    for S, p in out["by_S"].items():
        print(S, p["units"], p["datasets"], p["tune_units"], p["cpu_h"], p["process_wall_h"], p["wall_h_all_platforms"],
              p["units_without_cost"][:5])
    print(json.dumps(basis, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
