"""Packed-float-key sort vs np.argsort for the bank's permutation rows (identity + speed, small)."""
from __future__ import annotations

import time

import numpy as np


def perm_packed(u: np.ndarray) -> np.ndarray:
    ns = u.shape[1]
    s = max(int(ns - 1).bit_length(), 1)
    hi = np.floor(u * float(2 ** (53 - s)))          # top (53-s) bits of the 53-bit uniform, exact
    key = hi * float(2 ** s) + np.arange(ns, dtype=np.float64)  # exact integer < 2^53
    key.sort(axis=1)
    perm = key.astype(np.int64) & ((1 << s) - 1)
    return perm


def bench(fn, reps=7):
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    for ns in (333, 667, 2000, 4000):
        for cb in (64, 299):
            one(ns, cb)


def one(ns: int, cb: int):
    u = np.random.default_rng(ns).random((cb, ns))
    assert np.array_equal(perm_packed(u), np.argsort(u, axis=1))
    ta = bench(lambda: np.argsort(u, axis=1))
    tp = bench(lambda: perm_packed(u))
    print(f"ns={ns} rows={cb}: argsort {ta / u.size * 1e9:.1f} ns/el, packed {tp / u.size * 1e9:.1f} ns/el")


if __name__ == "__main__":
    main()
