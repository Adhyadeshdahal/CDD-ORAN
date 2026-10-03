"""R-42 step 1: effect shape of every true action -> KPI edge of E2 / E5 under the R1 design (Monte Carlo on the frozen
SCM update functions; no data, no seeds of the study: RNG default_rng(42_000)).

Per edge (a, k), with Var = Var(Y_k) + observation-noise variance (kappa .25):
  lin    corr(a, Y_k)^2 * Var(Y_k) / Var            share a LINEAR statistic in a can see
  main   Var(E[Y_k | a]) / Var                        share of a's main effect (any shape; 50 bins)
  total  E[(f(A) - f(A with a redrawn))^2] / 2 / Var  Sobol total-effect share (main + all interactions)
  resid  Var(Y_k - linear fit on every other action) / Var   noise left after a LINEAR adjustment
  even   main share of the even part in a (symmetric bump centred in the range)

    uv run python scratchpad/xmethod/pmrt_nl_shapes.py > scratchpad/xmethod/results/pmrt_nl/shapes.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

from cdd_oran.envs.v2.e2 import E2V2Env
from cdd_oran.envs.v2.e5 import E5V2Env
from cdd_oran.xmethod.worlds.generate import NOISE_SIGMA, truth_for

N = 20_000
KAPPA = 0.25


def _f(world: str, env, A: np.ndarray, kmid_prev: np.ndarray, z: np.ndarray) -> np.ndarray:
    out = np.empty((len(A), env.num_kpis))
    for i in range(len(A)):
        prev_k = np.zeros(env.num_kpis)
        if world == "E5":
            prev_k[E5V2Env.K_MID] = kmid_prev[i]
            env.prev_Z = z[i]
        out[i] = env._update_kpis(A[i], prev_k)
    return out


def main() -> None:
    rng = np.random.default_rng(42_000)
    res = {}
    for world, cls in (("E2", E2V2Env), ("E5", E5V2Env)):
        env = cls(env_seed=0)
        env.reset(episode=0)
        lo, hi = np.array(cls.id_ranges, float).T
        A = rng.uniform(lo, hi, size=(N, len(lo)))
        A2 = rng.uniform(lo, hi, size=(N, len(lo)))
        km = rng.uniform(-100, 100, N)                       # K_mid_prev = P0 at t-1 (R1: iid)
        z = rng.standard_normal(N)
        Y = _f(world, env, A, km, z)
        noise_var = (KAPPA * np.array(NOISE_SIGMA[world])) ** 2
        tr = truth_for(world, "R1")
        for src, tgt in sorted(tr.edges):
            if not src.startswith("P"):
                continue
            a, k = int(src[1:]), int(tgt[1:])
            y = Y[:, k]
            var = y.var() + noise_var[k]
            Ab = A.copy()
            Ab[:, a] = A2[:, a]
            yb = _f(world, env, Ab, km, z)[:, k]
            total = float(np.mean((y - yb) ** 2) / 2 / var)
            lin = float(np.corrcoef(A[:, a], y)[0, 1] ** 2 * y.var() / var)
            bins = np.clip(((A[:, a] - lo[a]) / (hi[a] - lo[a]) * 50).astype(int), 0, 49)
            cm = np.array([y[bins == b].mean() for b in range(50)])
            main = float(cm[bins].var() / var)
            ev = (cm + cm[::-1]) / 2                         # even part around the range midpoint
            even = float(ev[bins].var() / var)
            others = np.delete(A, a, axis=1)
            X = np.column_stack([np.ones(N), others])
            r = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
            resid = float((r.var() + noise_var[k]) / var)
            res[f"{world}|{src}->{tgt}"] = {"lin": round(lin, 4), "main": round(main, 4), "even": round(even, 4),
                                            "total": round(total, 4), "resid": round(resid, 3)}
    json.dump(res, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
