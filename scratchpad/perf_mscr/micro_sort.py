"""Micro-benchmark of the bank's per-stratum kernel pieces.

    PYTHONPATH=. .venv/Scripts/python.exe scratchpad/perf_mscr/micro_sort.py
"""
from __future__ import annotations

import time

import numpy as np


def bench(fn, reps=5):
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    for ns in (667, 2000, 4000):
        one(ns)


def one(ns: int, b: int = 299):
    rng = np.random.default_rng(0)
    u = rng.random((b, ns))
    y = rng.normal(size=ns)
    nb = 8
    sizes = np.bincount((np.arange(ns) * nb) // ns, minlength=nb)
    offsets = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    perm = np.argsort(u, axis=1)
    shift = int(ns - 1).bit_length()
    t = {
        "random": bench(lambda: np.random.default_rng(1).random((b, ns))),
        "argsort": bench(lambda: np.argsort(u, axis=1)),
        "argsort_stable": bench(lambda: np.argsort(u, axis=1, kind="stable")),
        "sort_f64": bench(lambda: np.sort(u, axis=1)),
        "packed_sort": bench(lambda: np.sort(
            ((u * 9007199254740992.0).astype(np.uint64) >> np.uint64(53 + shift - 64 if 53 + shift > 64 else 0))
            << np.uint64(shift) | np.arange(ns, dtype=np.uint64), axis=1)),
        "gather": bench(lambda: y[perm]),
        "reduceat": bench(lambda: np.add.reduceat(y[perm], offsets, axis=1)),
        "permuted_fy": bench(lambda: np.random.default_rng(1).permuted(np.tile(np.arange(ns), (b, 1)), axis=1)),
    }
    el = b * ns
    print(f"ns={ns}: " + "  ".join(f"{k}={v / el * 1e9:.1f}ns/el" for k, v in t.items()))


if __name__ == "__main__":
    main()
