"""Interleaved A/B of the serial bank (v2 reference vs optimized, n_jobs=1, cpu), small n: is the serial
path slower, or is it machine noise? Alternates A and B so load drift hits both.

    PYTHONPATH=. .venv/Scripts/python.exe scratchpad/perf_mscr/ab_serial.py [n] [n_perm] [reps]
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

import numpy as np

from cdd_oran.discovery import mscr
from cdd_oran.discovery.mscr import MSCRConfig
from cdd_oran.e2slice.dataset import E2DatasetConfig, generate_rows

_HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 299
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    spec = importlib.util.spec_from_file_location("mscr_ref", os.path.join(_HERE, "mscr_v2_reference.py"))
    assert spec is not None and spec.loader is not None
    ref = importlib.util.module_from_spec(spec)
    sys.modules["mscr_ref"] = ref
    spec.loader.exec_module(ref)
    r = generate_rows(E2DatasetConfig(n_rows_per_seed=n, seed=0))
    x, y = np.concatenate([r.x_params, r.x_kpis], 1), r.y_kpis
    cfg = MSCRConfig(n_perm=b)
    ta, tb = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        for j in range(y.shape[1]):
            ref._build_bank(x, y[:, j], cfg, np.random.default_rng([0, j, b]))
        ta.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        mscr._build_banks(x, y, cfg, [np.random.default_rng([0, j, b]) for j in range(y.shape[1])], n_jobs=1,
                          device="cpu")
        tb.append(time.perf_counter() - t0)
    print(f"bank only, E2 n={n} B={b}: v2 min {min(ta):.3f}s med {np.median(ta):.3f}s | "
          f"optimized serial min {min(tb):.3f}s med {np.median(tb):.3f}s")


if __name__ == "__main__":
    main()
