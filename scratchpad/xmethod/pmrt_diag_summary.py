"""Summarise pmrt_diag.py JSON lines: per variant, truth-null raw-p rate at .05 / .01 (real actions P0..P7) and the
P_placebo rate, seed-cluster bootstrap 95 % CI (cluster = DEV seed: replicates of a seed share its setpoints and
noise), sd / kurtosis of z, and the paired difference to a reference variant (same datasets, same CRT draws).

  uv run python scratchpad/xmethod/pmrt_diag_summary.py F.jsonl [G.jsonl ...] [--ref inv] [--reps all|dev|regen]
"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np


def load(paths, reps):
    recs = []
    for p in paths:
        with open(p) as fh:
            recs += [json.loads(line) for line in fh]
    if reps == "dev":
        recs = [r for r in recs if r["rep"] < 0]
    elif reps == "regen":
        recs = [r for r in recs if r["rep"] >= 0]
    return recs


def boot(hits, cnt, reps=4000, seed=0):
    rng = np.random.default_rng([7801, 99, seed])
    hits, cnt = np.asarray(hits, float), np.asarray(cnt, float)
    idx = rng.integers(0, len(hits), (reps, len(hits)))
    bs = hits[idx].sum(1) / np.maximum(cnt[idx].sum(1), 1)
    return float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def summarise(recs, ref=None):
    variants = [k for k in recs[0] if isinstance(recs[0][k], dict)]
    by_seed = collections.defaultdict(list)
    for r in recs:
        by_seed[r["seed"]].append(r)
    seeds = sorted(by_seed)
    out = {"n_datasets": len(recs), "n_seeds": len(seeds), "variants": {}}
    for v in variants:
        row = {}
        for lab, sel in (("null", lambda e: not e.startswith("P_placebo")), ("plac", lambda e: e.startswith("P_placebo"))):
            for a in (.05, .01):
                h = [sum(1 for r in by_seed[s] for e, (p, z) in r[v].items() if sel(e) and p <= a) for s in seeds]
                c = [sum(1 for r in by_seed[s] for e in r[v] if sel(e)) for s in seeds]
                row[f"{lab}{int(a * 100):02d}"] = [round(sum(h) / sum(c), 4), *[round(x, 4) for x in boot(h, c)]]
        Z = np.array([z for r in recs for e, (p, z) in r[v].items() if not e.startswith("P_placebo")])
        row["sd_z"] = round(float(Z.std()), 4)
        row["kurt_z"] = round(float(((Z - Z.mean()) ** 4).mean() / Z.var() ** 2), 3)
        if ref and ref in variants and v != ref:
            d = [sum(int(r[v][e][0] <= .05) - int(r[ref][e][0] <= .05) for r in by_seed[s] for e in r[v]
                     if not e.startswith("P_placebo")) for s in seeds]
            c = [sum(1 for r in by_seed[s] for e in r[v] if not e.startswith("P_placebo")) for s in seeds]
            lo, hi = boot(d, c, seed=1)
            row[f"diff05_vs_{ref}"] = [round(sum(d) / sum(c), 4), round(lo, 4), round(hi, 4)]
        out["variants"][v] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--reps", default="all", choices=("all", "dev", "regen"))
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    recs = load(a.paths, a.reps)
    s = summarise(recs, a.ref)
    print(f"datasets {s['n_datasets']}, seeds {s['n_seeds']}")
    for v, row in s["variants"].items():
        print(f"{v:9s} " + "  ".join(f"{k} {val}" for k, val in row.items()))
    if a.json:
        with open(a.json, "w", newline="\n") as fh:
            json.dump(s, fh, indent=1)


if __name__ == "__main__":
    main()
