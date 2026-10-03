"""F2 rcot2 vs R package RCIT (Strobl et al. 2019, github ericstrobl/RCIT) on identical inputs.

  python scratchpad/xmethod/citests/f2_rcit.py gen DIR [--reps 200] [--jobs 4]   # data CSVs + our p-values
  Rscript scratchpad/xmethod/citests/f2_rcit.R DIR                                 # RCIT p-values
  python scratchpad/xmethod/citests/f2_rcit.py compare DIR                         # -> DIR/F2_RCIT.json

Scenarios (n in {500, 1000}, |Z| in {3, 13}): null_lin (y linear in Z), null_nonlin (y nonlinear in Z), alt (y depends
on x^2). Ours: rcot2 numerics (frozen v2 config: dz 25, dxy 5) with (a) the analytic HBE null and (b) the frozen
block-permutation null at native B = 299. RCIT: RCoT(x, y, z, num_f = 25, num_f2 = 5, approx = "hbe" / "lpd4" (its
default) / "perm"). RFF draws cannot be shared across R and numpy, so agreement is distributional.
v2 (R-13): also (a) EXACT: python momentchi2 0.1.8 (lpb4 / hbe / sw, used by rcot2) vs R momentchi2 on identical
(weights, x): |cdf diff| <= 1e-6 (both use Brent root finding at tol 1e-9); (b) our rcot2 PRIMARY (lpd4 null,
num_f 100) vs RCIT RCoT defaults (num_f 100, num_f2 5, lpd4): rejection rates within 2 binomial SE.
Pre-stated tolerance: per scenario, |rejection rate at .05, ours-HBE - RCIT-HBE| and |ours-block_perm - RCIT-lpd4|
within 2 binomial SE (descriptive: Spearman of p-values).
"""
from __future__ import annotations

import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SCEN = [(kind, n, dz) for kind in ("null_lin", "null_nonlin", "alt") for n in (500, 1000) for dz in (3, 13)]


def triple(rng, n, dz, kind):
    z = rng.normal(size=(n, dz))
    x = 0.7 * z[:, 0] + rng.normal(size=n)
    if kind == "null_lin":
        y = 0.6 * z[:, 0] - 0.4 * z[:, -1] + rng.normal(size=n)
    elif kind == "null_nonlin":
        y = np.sin(z[:, 0]) + 0.5 * z[:, -1] ** 2 + rng.normal(size=n)
    else:
        y = np.sin(z[:, 0]) + 0.25 * x ** 2 + rng.normal(size=n)
    return x, y, z


def ours(args):
    import dataclasses

    from cdd_oran.xmethod.methods.rcot2 import rcot_pvalue_lpd4

    from cdd_oran.e2slice.discovery_rcot import _standardize, rcot_pvalue_analytic, rcot_pvalue_block_perm
    from cdd_oran.e2slice.discovery_rcot_v2 import frozen_config_v2

    path, r = args
    a = np.loadtxt(path, delimiter=",", skiprows=1)
    a = a[a[:, 0] == r][:, 1:]
    xs, _, _ = _standardize(a)
    x, y, z = xs[:, 0], xs[:, 1], xs[:, 2:]
    cfg = frozen_config_v2()
    s_h, p_h, _ = rcot_pvalue_analytic(x, y, z, cfg, 1000 + r)
    s_b, p_b, _ = rcot_pvalue_block_perm(x, y, z, cfg, 1000 + r)
    _, p_l, _, _ = rcot_pvalue_lpd4(x, y, z, dataclasses.replace(cfg, dz=100), 1000 + r)
    return r, s_h, p_h, p_b, p_l


def gen(d, reps, jobs):
    os.makedirs(d, exist_ok=True)
    tasks = []
    for s, (kind, n, dz) in enumerate(SCEN):
        rng = np.random.default_rng([7802, 21, s])
        f = os.path.join(d, f"{kind}_n{n}_z{dz}.csv")
        rows = []
        for r in range(reps):
            x, y, z = triple(rng, n, dz, kind)
            rows.append(np.c_[np.full(n, r), x, y, z])
        hdr = ",".join(["rep", "x", "y"] + [f"z{k}" for k in range(dz)])
        np.savetxt(f, np.vstack(rows), delimiter=",", header=hdr, comments="", fmt="%.17g")
        tasks += [(f, r) for r in range(reps)]
    with ProcessPoolExecutor(jobs) as ex:
        res = list(ex.map(ours, tasks, chunksize=4))
    out = {}
    for (f, _), (r, s_h, p_h, p_b, p_l) in zip(tasks, res, strict=True):
        out.setdefault(os.path.basename(f), []).append({"rep": r, "stat_hbe": s_h, "p_hbe": p_h, "p_bperm299": p_b,
                                                         "p_lpd4_f100": p_l})
    with open(os.path.join(d, "ours.json"), "w") as fh:
        json.dump(out, fh)
    # (a) momentchi2 exact comparison inputs: weight vectors (25, like dxy^2) and evaluation points
    from cdd_oran.xmethod.methods.rcot2 import _momentchi2
    from momentchi2 import sw

    hbe, lpb4 = _momentchi2()
    rng = np.random.default_rng([7802, 22])
    rows = []
    for c in range(200):
        w = np.sort(rng.gamma(0.5, 1.0, size=int(rng.choice([4, 9, 25])))) + 1e-6
        for x in (0.25, 0.5, 1.0, 2.0, 4.0):
            xx = float(x * w.sum())
            rows.append([len(rows), xx, float(lpb4(w, xx)), float(hbe(w, xx)), float(sw(w, xx))] + w.tolist())
    with open(os.path.join(d, "m2_cases.csv"), "w") as fh:
        for row in rows:
            fh.write(",".join([repr(v) for v in row] + [""] * (30 - len(row))) + chr(10))


def wilson(k, n):
    p = k / n; z = 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(p, 4), round(c - h, 4), round(c + h, 4)]


def compare(d):
    from scipy import stats

    ours_ = json.load(open(os.path.join(d, "ours.json")))
    import csv

    rcit = {}
    for row in csv.DictReader(open(os.path.join(d, "rcit.csv"))):
        rcit.setdefault(row["file"], {})[int(row["rep"])] = row
    out = {"pairs": [["ours_hbe", "rcit_hbe"], ["ours_bperm299", "rcit_lpd4"], ["ours_lpd4_f100", "rcit_lpd4_f100"]],
           "scenarios": {}}
    all_pass = True
    for f, recs in ours_.items():
        R = rcit[f]
        cols = {"ours_hbe": [x["p_hbe"] for x in recs], "ours_bperm299": [x["p_bperm299"] for x in recs]}
        if "p_lpd4_f100" in recs[0]:
            cols["ours_lpd4_f100"] = [x["p_lpd4_f100"] for x in recs]
        for key in ("hbe", "lpd4", "perm", "lpd4_f100"):
            if f"p_{key}" not in next(iter(R.values())) or R[recs[0]["rep"]][f"p_{key}"] == "NA":
                continue                          # column absent or skipped (SKIP_PERM=1 -> perm all NA)
            cols[f"rcit_{key}"] = [float(R[x["rep"]][f"p_{key}"]) for x in recs]
        m = len(recs)
        rej = {k: int(np.sum(np.asarray(v) <= .05)) for k, v in cols.items()}
        sc = {"reps": m, "rej": {k: wilson(v, m) for k, v in rej.items()}}
        for a_, b_ in out["pairs"]:
            if a_ not in cols or b_ not in cols:
                continue
            pa, pb = rej[a_] / m, rej[b_] / m
            se = math.sqrt(pa * (1 - pa) / m + pb * (1 - pb) / m)
            ok = abs(pa - pb) <= 2 * max(se, 1.0 / m)
            sc[f"{a_}~{b_}"] = {"diff": round(pa - pb, 4), "2se": round(2 * se, 4), "pass": bool(ok),
                                 "spearman": float(stats.spearmanr(cols[a_], cols[b_]).statistic)}
            all_pass &= ok
        out["scenarios"][f] = sc
    out["pass"] = bool(all_pass)
    mp = os.path.join(d, "m2_r.csv")
    if os.path.exists(mp):
        py = {}
        for line in open(os.path.join(d, "m2_cases.csv")):
            v = line.strip().split(","); py[int(float(v[0]))] = [float(t) for t in v[2:5]]
        diffs = {"lpb4": [], "hbe": [], "sw": []}
        for row in csv.DictReader(open(mp)):
            ref = py[int(float(row["case"]))]
            for k_, i_ in (("lpb4", 0), ("hbe", 1), ("sw", 2)):
                diffs[k_].append(abs(float(row[k_]) - ref[i_]))
        out["momentchi2_py_vs_R"] = {k_: {"n": len(v), "max_abs_diff": max(v)} for k_, v in diffs.items()}
        out["momentchi2_pass"] = all(max(v) <= 1e-6 for v in diffs.values())
    json.dump(out, open(os.path.join(d, "F2_RCIT.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    cmd, d = sys.argv[1], sys.argv[2]
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 200
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else (os.cpu_count() or 1)
    gen(d, reps, jobs) if cmd == "gen" else compare(d)
