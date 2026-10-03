"""R-42 steps 2-3: run the pmrt_nl candidates (and linear pmrt_core) on DEV datasets and synthetic worlds; aggregate.

  uv run python scratchpad/xmethod/pmrt_nl_runs.py list --plan PLAN [--arms a,b]
  uv run python scratchpad/xmethod/pmrt_nl_runs.py run  --plan PLAN --arms a,b --part i/P --out res_i.jsonl
  uv run python scratchpad/xmethod/pmrt_nl_runs.py agg  --inputs 'runs/x/**/*.jsonl' --out agg.json

Arms: linear = pmrt_core (pmrt_eq, the DEV primary); poly / rff = pmrt_nl with the pmrt_core ridge residual; gbm =
pmrt_nl learned matched filter with its own gbm adjustment; poly_g / rff_g = poly / rff with the gbm adjustment (orchestrator,
Q2: statistic and adjustment effects separable). All production settings (B 9999, Besag-Clifford h 20, no clip).

Plans (units; every seed in the DEV block 3_000_100-3_000_159, never 3_000_160-189, never EVAL):
  recall  DEV E2 / E5, R1 (seeds 100-119) and R2 (100-159), n 1000, kappa .25; SYN-NL R1 / R2 n 1000, reps 100-159.
  f4      synthetic null level: FID (scratchpad/xmethod/pmrt_fidelity.world, beta 0: every edge null; iid and the R2
          setpoint + dither design with a slow latent that drives setpoints and KPIs) and SYN-NL (its null edges:
          nonlinear state, nonlinear nuisance, a two-step chain), R1 / R2, n 500 / 1000 / 4000, reps 100-159.
  valid   DEV truth-null and placebo rates: E1-E5 R1 (seeds 100-119), R2 (100-159); E4 R1 / R2 at lam 1, R3 at lam
          {0, .5, 1, 1.5} (100-159), R4 at the same lams (100-119); n 500 / 1000 / 4000; kappa .25.
  cost    DEV E2 / E5 R1 n 4000 seed 3_000_100 (cost at n 4000 vs the 2 CPU-h budget).

SYN-NL (pre-specified 2026-10-03 before any candidate result on it; effect sizes fixed a priori: every planted
mechanism f is scaled to sd .25 under the R1 design, noise sd 1). Actions P0..P4, P_placebo; R1 i.i.d. U(0, 1);
R2 setpoint .5 + .3 tanh(load_j Z_block) (20 blocks, Z a slow AR(.995) latent) + U(-.1, .1) dither (only the dither
is redrawn). KPIs (state at t -> t + 1, e ~ N(0, 1)):
  K0' = .5 K0 + f_even(P0) + f_gate(P1, P4) + e        f_even = bump at .5 (sd .12); f_gate = (P1 - .5) 1[P4 < .25]
  K1' = .5 K1 + f_int(P2, P3) + .8 tanh(K0) [+ .5 Z] + e   f_int = bump of P2 + P3 at 1 (sd .08): pure interaction
  K2' = .5 K2 + 3 sin(2 pi P3) + f_lin(P2) + e          dominant nonlinear nuisance + a small linear edge
  K3' = .5 K3 + 2 cos(1.5 K2) [+ .5 Z] + e             no action edge (nonlinear state)
True action edges: P0->K0 (even), P1->K0 (gated), P4->K0 (gate switch), P2->K1, P3->K1 (interaction), P2->K2
(linear), P3->K2 (nuisance). Every other action -> KPI pair is a one-step null (P0 reaches K1 two steps later).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from cdd_oran.xmethod import api  # noqa: E402
from cdd_oran.xmethod.methods import pmrt_core as PC  # noqa: E402
from cdd_oran.xmethod.methods import pmrt_nl as NL  # noqa: E402
from cdd_oran.xmethod.worlds.generate import generate_dataset, truth_for  # noqa: E402

KAPPA = 0.25
SEEDS_R1 = range(3_000_100, 3_000_120)
SEEDS_R2 = range(3_000_100, 3_000_160)
RESERVED = range(3_000_160, 3_000_190)
SYN_TAG = 910
ARMS = {
    "linear": lambda: PC.PmrtCore(),
    "poly": lambda: NL.PmrtNl(NL.PmrtNlConfig(statistic="poly", adjust="ridge")),
    "rff": lambda: NL.PmrtNl(NL.PmrtNlConfig(statistic="rff", adjust="ridge")),
    "gbm": lambda: NL.PmrtNl(NL.PmrtNlConfig(statistic="gbm", adjust="gbm")),
    "poly_g": lambda: NL.PmrtNl(NL.PmrtNlConfig(statistic="poly", adjust="gbm")),
    "rff_g": lambda: NL.PmrtNl(NL.PmrtNlConfig(statistic="rff", adjust="gbm")),
}


# ============================================================================================ SYN-NL
def _bump(x, c, s):
    return np.exp(-((x - c) ** 2) / (2 * s * s))


def _mech():
    """The planted mechanisms, each scaled to sd .25 under U(0, 1) actions (fixed MC, rng 0)."""
    u = np.random.default_rng(0).uniform(0, 1, (400_000, 5))
    raw = {"even": lambda a: _bump(a[..., 0], .5, .12),
           "gate": lambda a: (a[..., 1] - .5) * (a[..., 4] < .25),
           "int": lambda a: _bump(a[..., 2] + a[..., 3], 1.0, .08),
           "lin": lambda a: a[..., 2] - .5}
    out = {}
    for k, f in raw.items():
        v = f(u)
        m, s = float(v.mean()), float(v.std())
        out[k] = (lambda a, f=f, m=m, s=s: 0.25 * (f(a) - m) / s)
    return out


MECH = _mech()
SYN_TRUE = frozenset({("P0", "K0"), ("P1", "K0"), ("P4", "K0"), ("P2", "K1"), ("P3", "K1"), ("P2", "K2"),
                      ("P3", "K2")})


def syn_nl(n: int, seed: int, regime: str) -> api.Dataset:
    rng = np.random.default_rng([seed, PC.RNG_TAG, SYN_TAG, 1 if regime == "R1" else 2])
    p = 6
    names = ("P0", "P1", "P2", "P3", "P4", "P_placebo")
    kn = ("K0", "K1", "K2", "K3")
    if regime == "R1":
        A = rng.uniform(0, 1, (n, p))
        designs = tuple(api.Design("iid", {"name": "uniform", "lo": 0.0, "hi": 1.0}) for _ in range(p))
        Z = np.zeros(n)
    else:
        z = np.zeros(n)
        e = rng.normal(0, 0.1, n)
        for t in range(1, n):
            z[t] = 0.995 * z[t - 1] + e[t]
        Z = z / max(z.std(), 1e-9)
        blk = np.minimum(np.arange(n) * 20 // n, 19)
        zb = np.array([Z[blk == b].mean() for b in range(20)])
        load = rng.uniform(0.5, 1.5, p) * rng.choice([-1.0, 1.0], p)
        sp = (0.5 + 0.3 * np.tanh(np.outer(zb, load)))[blk]
        dz = rng.uniform(-0.1, 0.1, (n, p))
        A = sp + dz
        designs = tuple(api.Design("dither", {"name": "uniform", "lo": -0.1, "hi": 0.1}, random_part=dz[:, j],
                                   fixed_part=sp[:, j]) for j in range(p))
    eps = rng.normal(0, 1, (n, 4))
    K = np.zeros(4)
    lag, Y = np.zeros((n, 4)), np.zeros((n, 4))
    zc = 0.5 if regime == "R2" else 0.0
    for t in range(n):
        lag[t] = K
        a = A[t]
        K = np.array([0.5 * K[0] + MECH["even"](a) + MECH["gate"](a),
                      0.5 * K[1] + MECH["int"](a) + 0.8 * np.tanh(K[0]) + zc * Z[t],
                      0.5 * K[2] + 3 * np.sin(2 * np.pi * a[3]) + MECH["lin"](a),
                      0.5 * K[3] + 2 * np.cos(1.5 * K[2]) + zc * Z[t]]) + eps[t]
        Y[t] = K
    cands = tuple((x, k) for x in names for k in kn) + tuple((x, k) for x in kn for k in kn)
    return api.Dataset("SYN-NL", regime, n, seed, names, kn, A, lag, Y, designs, cands, time_index=np.arange(n),
                       meta={"primary_candidates": tuple((x, k) for x in names for k in kn)})


def syn_fid(n: int, seed: int, regime: str) -> api.Dataset:
    import pmrt_fidelity as F
    return F.world(n, seed - F.DEV0, regime, beta=0.0)


# ============================================================================================ plans
def plan(name: str) -> list[tuple]:
    u = []
    if name == "recall":
        for w in ("E2", "E5"):
            u += [("dev", w, "R1", None, 1000, s) for s in SEEDS_R1]
            u += [("dev", w, "R2", None, 1000, s) for s in SEEDS_R2]
        u += [("syn", "nl", r, None, 1000, s) for r in ("R1", "R2") for s in SEEDS_R2]
    elif name == "f4":
        u += [("syn", sc, r, None, n, s) for sc in ("fid", "nl") for r in ("R1", "R2") for n in (500, 1000, 4000)
              for s in SEEDS_R2]
    elif name == "valid":
        for n in (500, 1000, 4000):
            for w in ("E1", "E2", "E3", "E4", "E5"):
                lam = 1.0 if w == "E4" else None
                u += [("dev", w, "R1", lam, n, s) for s in SEEDS_R1]
                u += [("dev", w, "R2", lam, n, s) for s in SEEDS_R2]
            for lam in (0.0, 0.5, 1.0, 1.5):
                u += [("dev", "E4", "R3", lam, n, s) for s in SEEDS_R2]
                u += [("dev", "E4", "R4", lam, n, s) for s in SEEDS_R1]
    elif name == "cost":
        u += [("dev", w, "R1", None, 4000, 3_000_100) for w in ("E2", "E5")]
    else:
        raise ValueError(name)
    assert all(not (3_000_160 <= x[-1] < 3_000_190) for x in u)
    return u


def ukey(unit) -> str:
    kind, w, r, lam, n, s = unit
    return f"{kind}|{w}|{r}|{'' if lam is None else f'lam{lam}|'}n{n}|s{s}"


def make_data(unit):
    kind, w, r, lam, n, s = unit
    if kind == "dev":
        kw = {"lam": lam} if lam is not None else {}
        ds, tr = generate_dataset(w, r, n, s, kappa=KAPPA, **kw)
        return ds, sorted(tr.edges)
    ds = syn_nl(n, s, r) if w == "nl" else syn_fid(n, s, r)
    return ds, sorted(SYN_TRUE) if w == "nl" else []


def run(a) -> None:
    arms = a.arms.split(",")
    units = plan(a.plan)
    units.sort(key=lambda x: (x[4], ukey(x)))                     # round-robin over cost-sorted units
    i, P = (int(x) for x in a.part.split("/"))
    mine = units[i::P][::-1] if a.reverse else units[i::P]   # reverse: a 2nd platform from the n 4000 end (agg dedups keys)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error") is None:
                done.add(r["key"])
    with open(a.out, "a", encoding="utf-8") as fo:
        for unit in mine:
            todo = [arm for arm in arms if f"{arm}|{ukey(unit)}" not in done]
            if not todo:
                continue
            ds, true = make_data(unit)
            for arm in todo:
                rec = {"key": f"{arm}|{ukey(unit)}", "arm": arm, "unit": list(unit), "true": true, "error": None}
                try:
                    t = time.process_time()
                    res = ARMS[arm]().run(ds)
                    rec["cpu_s"] = time.process_time() - t
                    rec["edges"] = [[e.source, e.target, e.p, e.score if math.isfinite(e.score) else None, e.sign,
                                     bool(e.declared)] for e in res.edges if e.source not in ds.kpi_names]
                    rec["version"] = res.version
                except Exception:
                    rec["error"] = traceback.format_exc()
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
                print(rec["key"], rec.get("cpu_s"), "ERR" if rec["error"] else "", flush=True)


# ============================================================================================ aggregate
def _cluster_ci(per_seed: list[tuple[int, int]], B: int = 2000) -> tuple[float, float, float]:
    """Rate (pooled) and seed-cluster bootstrap 95 % CI from per-seed (hits, count)."""
    h = np.array([x for x, _ in per_seed], float)
    c = np.array([y for _, y in per_seed], float)
    if c.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(h), (B, len(h)))
    bs = h[idx].sum(1) / np.maximum(c[idx].sum(1), 1)
    return float(h.sum() / c.sum()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def r30(lo: float, hi: float) -> str:
    if not math.isfinite(lo):
        return "NO_READ"
    return "INVALID" if lo > .05 else ("VALID" if hi <= .075 else "INCONCLUSIVE")


def agg(a) -> None:
    recs = {}
    for f in sorted({f for pat in a.inputs for f in glob.glob(pat, recursive=True)}):
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error") is None or r["key"] not in recs:
                recs[r["key"]] = r
    errors = sorted(k for k, r in recs.items() if r.get("error"))
    null, plac, rec_e, cost = {}, {}, {}, {}
    for r in recs.values():
        if r.get("error"):
            continue
        kind, w, rg, lam, n, s = r["unit"]
        cell = f"{kind}|{w}|{rg}|{'' if lam is None else f'lam{lam}|'}n{n}"
        true = {tuple(e) for e in r["true"]}
        hn = cn = hp = cp = 0
        for src, tgt, p, _, _, _ in r["edges"]:
            if p is None or src == "P_placebo_conf":
                continue
            if (src, tgt) in true:
                rec_e.setdefault((r["arm"], cell, f"{src}->{tgt}"), []).append(int(p <= .05))
            elif src == "P_placebo":
                hp, cp = hp + int(p <= .05), cp + 1
            else:
                hn, cn = hn + int(p <= .05), cn + 1
        null.setdefault((r["arm"], cell), []).append((hn, cn))
        plac.setdefault((r["arm"], cell), []).append((hp, cp))
        cost.setdefault((r["arm"], n), []).append(r["cpu_s"])
    out = {"errors": errors, "n_records": len(recs), "validity": {}, "recall_edges": {}, "recall": {}, "cost": {}}
    for (arm, cell), v in sorted(null.items()):
        rn = _cluster_ci(v)
        rp = _cluster_ci(plac[(arm, cell)])
        out["validity"][f"{arm}|{cell}"] = {"seeds": len(v), "null": [round(x, 4) for x in rn], "null_class": r30(*rn[1:]),
                                            "placebo": [round(x, 4) for x in rp], "placebo_class": r30(*rp[1:])}
    by_arm: dict = {}
    for (arm, cell, e), v in sorted(rec_e.items()):
        out["recall_edges"][f"{arm}|{cell}|{e}"] = [round(float(np.mean(v)), 3), len(v)]
        by_arm.setdefault(arm, {}).setdefault(cell, []).append(float(np.mean(v)))
    for arm, cells in by_arm.items():
        allv = [x for v in cells.values() for x in v]
        out["recall"][arm] = {"mean_per_edge": round(float(np.mean(allv)), 4), "n_edges": len(allv),
                              "cells": {c: round(float(np.mean(v)), 4) for c, v in cells.items()}}
    for (arm, n), v in sorted(cost.items()):
        out["cost"][f"{arm}|n{n}"] = {"mean": round(float(np.mean(v)), 1), "max": round(float(np.max(v)), 1),
                                      "units": len(v)}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({"n_records": len(recs), "errors": len(errors), "recall": {k: v["mean_per_edge"] for k, v in
                                                                                out["recall"].items()}}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("list", "run", "agg"):
        p = sub.add_parser(c)
        if c in ("list", "run"):
            p.add_argument("--plan", required=True)
            p.add_argument("--arms", default=",".join(ARMS))
        if c == "run":
            p.add_argument("--part", default="0/1")
            p.add_argument("--out", required=True)
            p.add_argument("--reverse", action="store_true")
        if c == "agg":
            p.add_argument("--inputs", nargs="+", required=True)
            p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "list":
        u = plan(a.plan)
        print(json.dumps({"units": len(u), "jobs": len(u) * len(a.arms.split(","))}))
    elif a.cmd == "run":
        run(a)
    else:
        agg(a)


if __name__ == "__main__":
    main()
