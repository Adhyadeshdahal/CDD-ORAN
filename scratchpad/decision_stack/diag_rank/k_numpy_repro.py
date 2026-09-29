"""Diag K (numpy reproducibility): fingerprint every numeric primitive the E6 sim uses, then a short accept-all
E6 trajectory, so two environments (numpy/scipy version, OS, CPU) can be diffed layer by layer.

  PYTHONPATH=. python scratchpad/decision_stack/diag_rank/k_numpy_repro.py [SECONDS=200] > fp.json
Layers: rng (Generator methods the sim calls, same SeedSequence shape as sim._rng), ufunc (transcendentals on
bit-stable inputs), scipy (gaussian_filter, norm.cdf), gain (GainMaps construction), traj (per-second sla hash).
"""
import hashlib
import json
import platform
import sys

import numpy as np
import scipy


def h(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()[:16]


def inputs(n=200_000):
    raw = np.random.PCG64(12345).random_raw(n)             # raw bit stream: version/platform stable by contract
    return (raw >> np.uint64(11)).astype(np.float64) * (1.0 / 2**53)


def main(seconds=200):
    out = {"numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(),
           "machine": platform.machine(), "python": platform.python_version()}
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as F
        out["cpu_avx512"] = bool(F.get("AVX512F")) and bool(F.get("AVX512_SKX"))
        out["cpu_avx2"] = bool(F.get("AVX2"))
    except Exception as e:  # pragma: no cover
        out["cpu"] = repr(e)
    rng = {}
    for m in ["uniform", "integers", "normal", "poisson_small", "poisson_big", "choice"]:
        r = np.random.default_rng([7, 6600, 3, 11])
        v = {"uniform": lambda: r.uniform(0, 1, 5000), "integers": lambda: r.integers(0, 37, 5000),
             "normal": lambda: r.normal(0, 1, 5000), "poisson_small": lambda: r.poisson(np.linspace(0, 9, 5000)),
             "poisson_big": lambda: r.poisson(np.linspace(10, 400, 5000)),
             "choice": lambda: r.choice(500, 40, replace=False)}[m]()
        rng[m] = h(v)
    out["rng"] = rng
    x = inputs()
    y = x * 200 - 100
    uf = {"log10": np.log10(x + 1e-3), "log": np.log(x + 1e-3), "log2": np.log2(1 + 10 ** (y / 10)),
          "pow10": 10 ** (y / 10), "exp": np.exp(-y / 7), "sin": np.sin(y), "cos": np.cos(y),
          "arctan2": np.arctan2(y, x - 0.5), "sqrt": np.sqrt(x), "sum": [np.sum(y), np.mean(y)],
          "bincount": np.bincount((x * 97).astype(int), weights=y)}
    out["ufunc"] = {k: h(v) for k, v in uf.items()}
    from scipy.ndimage import gaussian_filter
    from scipy.stats import norm
    z = np.random.default_rng(1).normal(size=(128, 128))
    out["scipy_fp"] = {"gaussian_filter": h(gaussian_filter(z, 4.0, mode="wrap")), "norm_cdf": h(norm.cdf(y / 30))}

    from cdd_oran.envs.e6 import config as C
    from cdd_oran.envs.e6.env import E6Env
    cfg = C.E6Config(seed=100020, load="medium", mobility="mixed", mix="M4", warmup_s=120.0, scored_s=600.0,
                     scenario="base")
    env = E6Env(cfg, log=False, wg3=True)
    gm = env.plant.gm
    out["gain"] = h(np.concatenate([np.ravel(np.asarray(v, dtype=float)) for k, v in sorted(vars(gm).items())
                                    if isinstance(v, np.ndarray) and v.dtype.kind == "f"]))
    traj = []
    while env.sec < seconds:
        obs = env.step_propose()
        env.step_apply({"decisions": ["accept"] * len(obs["requests"]), "writes": [], "rollback": []})
        sla = env.plant.sla
        traj.append(h([float(sla[k]) for k in sorted(sla) if np.isscalar(sla[k])]))
    out["traj"] = traj
    out["sla_final"] = {k: float(v) for k, v in sorted(env.plant.sla.items()) if np.isscalar(v)}
    print(json.dumps(out))


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
