"""Q4 mechanism diagnosis: why does pmrt_core WITH the predictable Huber clip over-reject on E2 R2 null edges?

  uv run python scratchpad/xmethod/pmrt_q4_diag.py --world E2 --regime R2 --n 1000 [--delta 0.10]
         --variants default,conc_off,... [--seeds 40] --out F.json

E2 facts used: every KPI at t+1 is a deterministic function of the actions at t only (no process noise, no KPI
memory, no true KPI -> KPI edge), so for a null edge (j, k) the target series Y_k is invariant to ALL of j's dithers.
In the production statistic, W still depends on j's PAST dithers through (i) the lagged-action covariates (j's own
A_{t-1}, A_{t-2}), (ii) the lagged KPIs that j drives, and (iii) the ridge coefficients / clip scale fitted on them.
W is predictable (martingale argument, asymptotic) but not invariant, so the fixed-W CRT is not exact.

Variants (clip on, huber_c 2.5, unless named "noclip" / "production"):
  default     the pre-R-14 production config (lag KPIs, hist 2, setpoints, concurrent other actions, clip)
  conc_off    concurrent other actions dropped (ruling Q3 / "R-8" off)
  hist_off    lagged actions dropped
  lag_off     lagged KPIs dropped
  inv         hist_off + lag_off: W = f(concurrent other actions, setpoints, past Y_k) -> invariant to j's dithers
              for every E2 null edge, so the CRT is EXACT (any statistic, clip included)
  noclip      default without the clip; noclip_inv likewise; production = the R-14 default (= noclip)
--delta overrides harness generate.DITHER_DELTA in this process only (no harness file is edited).
Records per (variant, seed, edge): p, z, target; summary per variant: pooled rate .05 / .01 with seed-cluster
bootstrap CIs, per-target rates, and per-edge mean z across seeds (a real effect mislabelled null would show a
consistent sign: t = mean z sqrt(S) / sd z).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from cdd_oran.xmethod.methods import pmrt_core as PC  # noqa: E402
from cdd_oran.xmethod.worlds import generate as G  # noqa: E402

C = {"huber_c": 2.5}                               # the clip, explicit (pmrt_core default is no clip since R-14)
VARIANTS = {
    "default": ({**C}, False), "conc_off": ({**C, "concurrent_actions": False}, False),
    "hist_off": ({**C, "hist_lags": 0}, False), "lag_off": ({**C}, True), "inv": ({**C, "hist_lags": 0}, True),
    "noclip": ({"huber_c": None}, False), "noclip_inv": ({"huber_c": None, "hist_lags": 0}, True),
    "production": ({}, False),                     # R-14 production config (= noclip)
}


def cluster_ci(hits, cnt, reps=2000):
    rng = np.random.default_rng([7801, 78])
    hits, cnt = np.asarray(hits, float), np.asarray(cnt, float)
    bs = [hits[i].sum() / max(cnt[i].sum(), 1) for i in (rng.integers(0, len(hits), len(hits)) for _ in range(reps))]
    return [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="E2")
    ap.add_argument("--regime", default="R2")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--variants", default="default,noclip")
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--B", type=int, default=999)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.delta is not None:
        G.DITHER_DELTA = float(a.delta)
    vs = a.variants.split(",")
    rec = {v: [] for v in vs}                       # per seed: {edge: (p, z)}
    t0 = time.time()
    for i in range(a.seeds):
        s = 3_000_000 + i
        ds, tr = G.generate_dataset(a.world, a.regime, a.n, s)
        nulls = [tuple(c) for c in ds.meta["primary_candidates"] if tuple(c) in tr.null_edges]
        ds_nolag = dataclasses.replace(ds, X_kpi_lag=np.full_like(ds.X_kpi_lag, np.nan))
        for v in vs:
            kw, nolag = VARIANTS[v]
            res = PC.PmrtCore(PC.PmrtCoreConfig(B=a.B, seq_h=None, **kw)).run(ds_nolag if nolag else ds)
            by = {(e.source, e.target): e for e in res.edges}
            rec[v].append({f"{j}->{k}": (by[(j, k)].p, res.notes["z"][f"{j}->{k}"]) for j, k in nulls})
        print(f"seed {s} ({time.time() - t0:.0f} s)", flush=True)
    out = {"world": a.world, "regime": a.regime, "n": a.n, "delta": G.DITHER_DELTA, "B": a.B, "seeds": a.seeds,
           "summary": {}, "records": rec}
    for v in vs:
        seeds = rec[v]
        edges = sorted(seeds[0])
        P = np.array([[sd[e][0] for e in edges] for sd in seeds])
        Z = np.array([[sd[e][1] for e in edges] for sd in seeds])
        tgt = sorted({e.split("->")[1] for e in edges})
        tz = Z.mean(0) * np.sqrt(len(seeds)) / np.maximum(Z.std(0, ddof=1), 1e-12)
        out["summary"][v] = {
            "n_p": int(P.size), "rate05": float((P <= .05).mean()), "ci05": cluster_ci((P <= .05).sum(1), [P.shape[1]] * len(P)),
            "rate01": float((P <= .01).mean()), "ci01": cluster_ci((P <= .01).sum(1), [P.shape[1]] * len(P)),
            "sd_z": float(Z.std()), "kurt_z": float(((Z - Z.mean()) ** 4).mean() / Z.var() ** 2),
            "per_target05": {k: float((P[:, [i for i, e in enumerate(edges) if e.endswith("->" + k)]] <= .05).mean())
                             for k in tgt},
            "edge_t_max": float(np.max(np.abs(tz))), "edge_t_top": sorted(
                [(edges[i], round(float(tz[i]), 2)) for i in range(len(edges))], key=lambda x: -abs(x[1]))[:5],
            "n_edges_abs_t_gt_3": int(np.sum(np.abs(tz) > 3)), "n_edges": len(edges)}
        print(v, json.dumps({k: out["summary"][v][k] for k in ("rate05", "ci05", "rate01", "ci01", "sd_z", "kurt_z",
                                                                "edge_t_max", "n_edges_abs_t_gt_3")}), flush=True)
    with open(a.out, "w", newline="\n") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
