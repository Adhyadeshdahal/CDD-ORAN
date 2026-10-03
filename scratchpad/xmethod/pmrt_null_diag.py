"""Null-calibration diagnostic of pmrt_core on the harness worlds (DEV seeds 3_000_000 + i, n 1000, B 999 fixed).

  uv run python scratchpad/xmethod/pmrt_null_diag.py WORLD REGIME [--lam 1.0] [--seeds 40] --out F.json

Per seed: (real) raw p of the truth's exact-null PRIMARY edges (incl. P_placebo); (swap) every primary candidate
with the KPI series (X_kpi_lag, Y) replaced by the next seed's: an exact sharp null with the same serial structure.
Variants: the production adjustment ("default") and the same without the predictable Huber clip ("no_huber").
Reports the pooled rate at .05 / .01 and a seed-cluster bootstrap 95% CI (p-values within a seed are dependent).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from cdd_oran.xmethod.methods import pmrt_core as PC  # noqa: E402
from cdd_oran.xmethod.worlds import generate_dataset  # noqa: E402

VARIANTS = {"default": {"huber_c": 2.5}, "no_huber": {"huber_c": None}}   # explicit: pmrt_core default is no clip (R-14)


def gen(w, r, s, lam):
    return generate_dataset(w, r, 1000, s, lam=lam) if w == "E4" else generate_dataset(w, r, 1000, s)


def cluster_ci(per_seed: list, thr: float, reps: int = 2000) -> list:
    rng = np.random.default_rng([7801, 77])
    hits = np.array([float(np.sum(np.asarray(p) <= thr)) for p in per_seed])
    cnt = np.array([float(len(p)) for p in per_seed])
    bs = [hits[i].sum() / max(cnt[i].sum(), 1) for i in (rng.integers(0, len(hits), len(hits)) for _ in range(reps))]
    return [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("world")
    ap.add_argument("regime")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seeds = [3_000_000 + i for i in range(a.seeds)]
    per = {(mode, v): [] for mode in ("real", "swap") for v in VARIANTS}
    for i, s in enumerate(seeds):
        ds, tr = gen(a.world, a.regime, s, a.lam)
        other, _ = gen(a.world, a.regime, seeds[(i + 1) % len(seeds)], a.lam)
        sw = dataclasses.replace(ds, X_kpi_lag=other.X_kpi_lag, Y=other.Y)
        prim = [tuple(c) for c in ds.meta["primary_candidates"]]
        for v, kw in VARIANTS.items():
            m = PC.PmrtCore(PC.PmrtCoreConfig(B=999, seq_h=None, **kw))
            r = m.run(ds)
            per[("real", v)].append([e.p for e in r.edges if (e.source, e.target) in prim
                                     and (e.source, e.target) in tr.null_edges and e.p is not None])
            r = m.run(sw)
            per[("swap", v)].append([e.p for e in r.edges if (e.source, e.target) in prim and e.p is not None])
        print(f"seed {s} done", flush=True)
    out = {"world": a.world, "regime": a.regime, "lam": a.lam, "seeds": [seeds[0], seeds[-1]], "n": 1000, "B": 999,
           "cells": {}}
    for (mode, v), ps in per.items():
        flat = np.concatenate([np.asarray(p, float) for p in ps]) if ps else np.zeros(0)
        out["cells"][f"{mode}|{v}"] = {"n_p": int(len(flat)), "rate05": float(np.mean(flat <= .05)),
                                       "ci05_cluster": cluster_ci(ps, .05), "rate01": float(np.mean(flat <= .01)),
                                       "ci01_cluster": cluster_ci(ps, .01), "mean_p": float(flat.mean())}
        print(mode, v, json.dumps(out["cells"][f"{mode}|{v}"]), flush=True)
    with open(a.out, "w", newline="\n") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
