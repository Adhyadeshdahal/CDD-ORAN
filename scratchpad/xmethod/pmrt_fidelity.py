"""F4 fidelity of pmrt_core on synthetic api.Dataset corpora (no harness dependency).

  uv run python scratchpad/xmethod/pmrt_fidelity.py null  --reps 200 --out F.json [--n 500,1000,4000] [--native --B 999]
  uv run python scratchpad/xmethod/pmrt_fidelity.py power --reps 50  --out F.json [--native --B 999]
  uv run python scratchpad/xmethod/pmrt_fidelity.py scale --out F.json [--n 500,1000,4000,8000,24000] [--native-n 4000]
Default = the production config (ruling R-9: max B 9999, Besag-Clifford h = 20); --native = fixed B (gate check).
  uv run python scratchpad/xmethod/pmrt_fidelity.py table --null F --power F [--scale F]

World (one time series, E2-like shapes): actions P0, P1, P2, P_placebo; KPIs K0..K3, K_{t+1} = R K_t + effects +
noise (sd 0.1; R: AR 0.7 on the diagonal and the lagged KPI edge K0 -> K2 0.5). Planted one-step edges P0 -> K0 (+beta)
and P1 -> K1 (-beta); every other (action, KPI) pair is an exact one-step null (P0 reaches K2 only two steps later).
  R1  actions i.i.d. U(0, 1) (design iid).
  R2  action = setpoint + dither: setpoint = blockwise (20 blocks) 0.5 + 0.2 tanh(Z) per action, Z a slow latent
      AR(0.995) that ALSO drives K1 and K3 (+0.3 Z): the setpoints of the null actions P2 / P_placebo are correlated
      with K1 / K3 (confounded serial dependence: a row permutation / association test fails here); dither
      U(-0.1, 0.1) (design dither, only the dither is redrawn).
  R1t R1 with Student-t(3) noise (scaled to sd 0.1): heavy tails.
Seeds: corpus seeds from the DEV block 3_000_000 + r (r < 200), streams default_rng([seed, 7801, 900, scenario]).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tracemalloc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from cdd_oran.xmethod import api  # noqa: E402
from cdd_oran.xmethod.methods import pmrt_core as PC  # noqa: E402

DEV0 = 3_000_000
SCEN = {"R1": 1, "R2": 2, "R1t": 3}
ACTIONS = ("P0", "P1", "P2", "P_placebo")
KPIS = ("K0", "K1", "K2", "K3")
TRUE = {("P0", "K0"): 1, ("P1", "K1"): -1}


def world(n: int, r: int, regime: str, beta: float, n_actions: int = 4, n_kpis: int = 4) -> api.Dataset:
    seed = DEV0 + r
    rng = np.random.default_rng([seed, PC.RNG_TAG, 900, SCEN[regime]])
    p, k = n_actions, n_kpis
    anames = ACTIONS if p == 4 else tuple(f"P{j}" for j in range(p - 1)) + ("P_placebo",)
    knames = KPIS if k == 4 else tuple(f"K{j}" for j in range(k))
    if regime in ("R1", "R1t"):
        A = rng.uniform(0.0, 1.0, (n, p))
        designs = tuple(api.Design("iid", {"name": "uniform", "lo": 0.0, "hi": 1.0}) for _ in range(p))
        Z = np.zeros(n)
    else:
        z = np.zeros(n)
        e = rng.normal(0.0, 0.1, n)
        for t in range(1, n):
            z[t] = 0.995 * z[t - 1] + e[t]
        Z = z / max(z.std(), 1e-9)
        nb = 20
        blk = np.minimum(np.arange(n) * nb // n, nb - 1)
        zb = np.array([Z[blk == b].mean() for b in range(nb)])
        load = rng.uniform(0.5, 1.5, p) * rng.choice([-1.0, 1.0], p)
        sp = 0.5 + 0.2 * np.tanh(np.outer(zb, load))[blk]
        dz = rng.uniform(-0.1, 0.1, (n, p))
        A = sp + dz
        designs = tuple(api.Design("dither", {"name": "uniform", "lo": -0.1, "hi": 0.1}, random_part=dz[:, j],
                                   fixed_part=sp[:, j]) for j in range(p))
    Rm = 0.7 * np.eye(k)
    Rm[2 % k, 0] = 0.5
    if regime == "R1t":
        noise = rng.standard_t(3, (n, k)) * 0.1 / np.sqrt(3.0)
    else:
        noise = rng.normal(0.0, 0.1, (n, k))
    K = np.zeros(k)
    lag, Y = np.zeros((n, k)), np.zeros((n, k))
    for t in range(n):
        lag[t] = K
        K = Rm @ K + noise[t]
        K[0] += beta * (A[t, 0] - 0.5)
        K[1 % k] -= beta * (A[t, 1] - 0.5)
        if regime == "R2":
            K[1 % k] += 0.3 * Z[t]
            K[3 % k] += 0.3 * Z[t]
        Y[t] = K
    cands = tuple((a, kk) for a in anames for kk in knames) + tuple((a, b) for a in knames for b in knames if a != b)
    return api.Dataset("SYN", regime, n, seed, anames, knames, A, lag, Y, designs, cands, time_index=np.arange(n),
                       meta={"beta": beta})


def wilson(x: int, n: int, z: float = 1.96) -> list:
    if n == 0:
        return [float("nan")] * 2
    ph = x / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def cfg_of(a) -> PC.PmrtCoreConfig:
    """Production config (ruling R-9: max B 9999, Besag-Clifford h 20); --native = fixed B (fidelity-gate check)."""
    return PC.PmrtCoreConfig(B=a.B, seq_h=None if a.native else 20)


def run_null(a):
    out = {}
    m = PC.PmrtCore(cfg_of(a))
    for reg in a.regimes.split(","):
        for n in map(int, a.n.split(",")):
            t = time.time()
            ps, dec_null, n_by_fp, pl = [], 0, 0, []
            for r in range(a.reps):
                res = m.run(world(n, r, reg, beta=a.beta))
                fp = 0
                for e in res.edges:
                    if e.p is None or (e.source, e.target) in TRUE:
                        continue
                    ps.append(e.p)
                    fp += int(e.declared)
                    if e.source == "P_placebo":
                        pl.append(e.p)
                dec_null += fp
                n_by_fp += int(fp > 0)
            ps = np.array(ps)
            x05 = int((ps <= 0.05).sum())
            out[f"{reg}|{n}"] = {"reps": a.reps, "n_null_p": len(ps), "rate05": x05 / len(ps),
                                 "ci05": wilson(x05, len(ps)), "rate01": float(np.mean(ps <= 0.01)),
                                 "rate10": float(np.mean(ps <= 0.10)), "mean_p": float(ps.mean()),
                                 "placebo_rate05": float(np.mean(np.array(pl) <= 0.05)),
                                 "by_null_declarations": dec_null, "by_any_null_declared_frac": n_by_fp / a.reps,
                                 "beta_true_edges": a.beta, "wall_s": round(time.time() - t, 1)}
            print(reg, n, json.dumps(out[f"{reg}|{n}"]), flush=True)
    return out


def run_power(a):
    out = {}
    m = PC.PmrtCore(cfg_of(a))
    for reg in a.regimes.split(","):
        for n in map(int, a.n.split(",")):
            for beta in map(float, a.betas.split(",")):
                t = time.time()
                hit = {h: 0 for h in TRUE}
                sgn = {h: 0 for h in TRUE}
                fdp = []
                for r in range(a.reps):
                    res = m.run(world(n, 100 + r, reg, beta=beta))
                    dec = [(e.source, e.target) for e in res.edges if e.declared]
                    for h, s in TRUE.items():
                        e = next(x for x in res.edges if (x.source, x.target) == h)
                        hit[h] += int(e.declared)
                        sgn[h] += int(e.declared and e.sign == s)
                    fdp.append(sum(d not in TRUE for d in dec) / max(len(dec), 1))
                out[f"{reg}|{n}|{beta}"] = {"reps": a.reps, "power": {"|".join(h): hit[h] / a.reps for h in TRUE},
                                            "sign_ok": {"|".join(h): sgn[h] / max(hit[h], 1) for h in TRUE},
                                            "mean_fdp": float(np.mean(fdp)), "wall_s": round(time.time() - t, 1)}
                print(reg, n, beta, json.dumps(out[f"{reg}|{n}|{beta}"]), flush=True)
    return out


def run_scale(a):
    out = {}
    m = PC.PmrtCore(cfg_of(a))
    for n in map(int, a.n.split(",")):
        for reg in a.regimes.split(","):
            d = world(n, 199, reg, beta=0.1, n_actions=9, n_kpis=6)        # E2-like: 8 actions + placebo, 6 KPIs
            for mode, mm in (("seq", m), ("native", PC.PmrtCore(PC.PmrtCoreConfig(B=a.B, seq_h=None)))):
                if mode == "native" and n not in a.native_n:
                    continue
                tracemalloc.start()
                res = mm.run(d)
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                out[f"{reg}|{n}|{mode}"] = {"cpu_s": round(res.cpu_s, 2), "peak_mb_traced": round(peak / 2 ** 20, 1),
                                            "B": a.B, "seq_h": mm.cfg.seq_h, "p_actions": 9, "k_kpis": 6,
                                            "n_decl": sum(e.declared for e in res.edges)}
                print(reg, n, mode, json.dumps(out[f"{reg}|{n}|{mode}"]), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("null", "power", "scale", "table"))
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--B", type=int, default=9999)
    ap.add_argument("--native", action="store_true", help="fixed B (no sequential stopping)")
    ap.add_argument("--n", default="500,1000,4000")
    ap.add_argument("--regimes", default="R1,R2,R1t")
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--betas", default="0.05,0.1,0.2")
    ap.add_argument("--native-n", dest="native_n", type=lambda x: [int(v) for v in x.split(",")], default=[4000])
    ap.add_argument("--out", default=None)
    ap.add_argument("--null", default=None)
    ap.add_argument("--power", default=None)
    ap.add_argument("--scale", default=None)
    a = ap.parse_args()
    if a.mode == "table":
        return table(a)
    t = time.process_time()
    res = {"null": run_null, "power": run_power, "scale": run_scale}[a.mode](a)
    res = {"mode": a.mode, "args": vars(a), "version": PC.PMRT_CORE_VERSION, "cpu_s_total": time.process_time() - t,
           "cells": res}
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", newline="\n") as fh:
            json.dump(res, fh, indent=1)


def table(a):
    if a.null:
        d = json.load(open(a.null))
        print(f"NULL (B {d['args']['B']}, reps {d['args']['reps']}): cell | #p | rate .05 [95% CI] | .01 | .10 | mean p |"
              " placebo .05 | BY null decl (any)")
        for c, v in d["cells"].items():
            print(f"  {c:10s} {v['n_null_p']:5d}  {v['rate05']:.3f} [{v['ci05'][0]:.3f}, {v['ci05'][1]:.3f}]  "
                  f"{v['rate01']:.3f}  {v['rate10']:.3f}  {v['mean_p']:.3f}  {v['placebo_rate05']:.3f}  "
                  f"{v['by_null_declarations']} ({v['by_any_null_declared_frac']:.2f})")
    if a.power:
        d = json.load(open(a.power))
        print(f"POWER (BY q .05; B {d['args']['B']}, reps {d['args']['reps']}): cell | P0->K0 | P1->K1 | sign ok | mean FDP")
        for c, v in d["cells"].items():
            pw = v["power"]
            print(f"  {c:16s} {pw['P0|K0']:.2f}  {pw['P1|K1']:.2f}  {min(v['sign_ok'].values()):.2f}  "
                  f"{v['mean_fdp']:.3f}")
    if a.scale:
        d = json.load(open(a.scale))
        for c, v in d["cells"].items():
            print(f"  SCALE {c:18s} cpu {v['cpu_s']:.1f} s  peak {v['peak_mb_traced']:.0f} MB (traced)")


if __name__ == "__main__":
    main()
