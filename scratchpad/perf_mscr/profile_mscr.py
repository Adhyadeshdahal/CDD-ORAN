"""Profile MSCR-v2 at small n: where time and memory go.

    PYTHONPATH=. .venv/Scripts/python.exe scratchpad/perf_mscr/profile_mscr.py [n] [n_perm]

Uses a real E2 corpus (14 columns = 8 params + 6 lagged KPIs, 6 targets), n rows.
"""
from __future__ import annotations

import cProfile
import pstats
import sys
import time
import tracemalloc

import numpy as np

from cdd_oran.discovery import mscr
from cdd_oran.discovery.mscr import MSCRConfig, discover_mscr
from cdd_oran.e2slice.dataset import E2DatasetConfig, generate_rows


def corpus(n: int, seed: int = 0):
    rows = generate_rows(E2DatasetConfig(n_rows_per_seed=n, seed=seed))
    return np.concatenate([rows.x_params, rows.x_kpis], axis=1), rows.y_kpis


def phase_times(x, y, cfg, seed=0):
    """Time bank vs observed S* vs p-values, per target, with the module's own functions."""
    t_bank = t_obs = t_p = 0.0
    for j in range(y.shape[1]):
        t0 = time.perf_counter()
        denom, null, strata = mscr._build_bank(x, y[:, j], cfg, np.random.default_rng([seed, j, cfg.n_perm]))
        t1 = time.perf_counter()
        s = [mscr._s_star(x[:, i], i, denom, strata, cfg) for i in range(8)]
        t2 = time.perf_counter()
        [mscr._pval(s[i], i, null) for i in range(8)]
        t3 = time.perf_counter()
        t_bank += t1 - t0
        t_obs += t2 - t1
        t_p += t3 - t2
    return t_bank, t_obs, t_p


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 299
    cfg = MSCRConfig(n_perm=b)
    x, y = corpus(n)
    print(f"E2 corpus n={n}, cols={x.shape[1]}, targets={y.shape[1]}, n_perm={b}")
    tb, to, tp = phase_times(x, y, cfg)
    print(f"phases: bank {tb:.2f}s  observed S* {to:.3f}s  pvals {tp:.4f}s")

    tracemalloc.start()
    discover_mscr(x, y, n_params=8, seed=0, config=cfg)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"tracemalloc peak during discover_mscr: {peak / 2**20:.1f} MiB")

    pr = cProfile.Profile()
    pr.enable()
    discover_mscr(x, y, n_params=8, seed=0, config=cfg)
    pr.disable()
    pstats.Stats(pr).sort_stats("tottime").print_stats(12)


if __name__ == "__main__":
    main()
