"""Power of pmrt_diag.py --power records: per variant, recall of the true action -> KPI edges at raw p <= .05 and
under BY (q .05, fdr_layer.declare 'by' over the primary family incl. P_placebo, as pmrt_core), the null BY
declaration rate, and the paired recall difference to --ref (seed-cluster bootstrap 95 % CI).

  uv run python scratchpad/xmethod/pmrt_diag_power.py F.jsonl [...] --ref prod
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from cdd_oran.decision import fdr_layer as FL


def declared(rec, v):
    fam = [tuple(e.split("->")) for e in rec["family"]]
    st = {tuple(e.split("->")): FL.HypStats(p2=pz[0], sign=int(np.sign(pz[1]))) for e, pz in rec[v].items()}
    d = FL.declare(st, "by", q=.05, hyps=fam)
    return {f"{h[0]}->{h[1]}" for h, x in d.items() if x["declared"]}


def boot(x, reps=4000):
    x = np.asarray(x, float)
    rng = np.random.default_rng([7801, 99, 5])
    b = x[rng.integers(0, len(x), (reps, len(x)))].mean(1)
    return [round(float(np.quantile(b, .025)), 4), round(float(np.quantile(b, .975)), 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--ref", default="prod")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    recs = [json.loads(line) for p in a.paths for line in open(p)]
    vs = [k for k in recs[0] if isinstance(recs[0][k], dict)]
    out = {"n_datasets": len(recs), "seeds": sorted({r["seed"] for r in recs}).__len__(), "variants": {}}
    per = {v: {"raw": [], "by": [], "nullby": []} for v in vs}
    for r in recs:
        T = set(r["true"])
        for v in vs:
            dec = declared(r, v)
            per[v]["raw"].append(np.mean([r[v][e][0] <= .05 for e in T]))
            per[v]["by"].append(np.mean([e in dec for e in T]))
            nulls = [e for e in r[v] if e not in T]
            per[v]["nullby"].append(np.mean([e in dec for e in nulls]))
    for v in vs:
        row = {k: [round(float(np.mean(x)), 4)] + boot(x) for k, x in per[v].items()}
        if v != a.ref and a.ref in vs:
            for k in ("raw", "by"):
                d = np.array(per[v][k]) - np.array(per[a.ref][k])
                row[f"d_{k}_vs_{a.ref}"] = [round(float(d.mean()), 4)] + boot(d)
        out["variants"][v] = row
        print(f"{v:9s} " + "  ".join(f"{k} {x}" for k, x in row.items()))
    if a.json:
        json.dump(out, open(a.json, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
