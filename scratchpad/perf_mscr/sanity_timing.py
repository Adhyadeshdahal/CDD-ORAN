"""Light timing SANITY at small n (not the benchmark; bench_mscr.py is). Confirms each path runs, returns
identical output, and moves in the expected direction. Timings on a loaded machine are indicative only.

    PYTHONPATH=. [SANITY_DEVICES=cpu,cuda] .venv/Scripts/python.exe scratchpad/perf_mscr/sanity_timing.py [n] [n_perm] [n_jobs ...]
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

import numpy as np

from cdd_oran.discovery.mscr import MSCRConfig, discover_mscr
from cdd_oran.e2slice.dataset import E2DatasetConfig, generate_rows

_HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 299
    jobs = [int(a) for a in sys.argv[3:]] or [1, 2]
    spec = importlib.util.spec_from_file_location("mscr_ref", os.path.join(_HERE, "mscr_v2_reference.py"))
    assert spec is not None and spec.loader is not None
    ref = importlib.util.module_from_spec(spec)
    sys.modules["mscr_ref"] = ref
    spec.loader.exec_module(ref)
    r = generate_rows(E2DatasetConfig(n_rows_per_seed=n, seed=0))
    x, y = np.concatenate([r.x_params, r.x_kpis], 1), r.y_kpis
    cfg = MSCRConfig(n_perm=b)
    t0 = time.perf_counter()
    r0 = ref.discover_mscr(x, y, n_params=8, seed=0, config=cfg)
    t_ref = time.perf_counter() - t0
    print(f"E2 n={n} B={b}: v2-reference {t_ref:.2f}s")
    for dev in os.environ.get("SANITY_DEVICES", "cpu,cuda").split(","):
        for j in jobs:
            try:
                t0 = time.perf_counter()
                r1 = discover_mscr(x, y, n_params=8, seed=0, config=cfg, n_jobs=j, device=dev)
                dt = time.perf_counter() - t0
            except RuntimeError as e:
                print(f"  {dev} n_jobs={j}: skipped ({e})")
                continue
            same = all(np.array_equal(a, c) for a, c in ((r0.pvals, r1.pvals), (r0.s_star, r1.s_star),
                                                         (r0.declared, r1.declared)))
            print(f"  {dev:4s} n_jobs={j}: {dt:.2f}s (x{t_ref / dt:.2f})  identical={same}")


if __name__ == "__main__":
    main()
