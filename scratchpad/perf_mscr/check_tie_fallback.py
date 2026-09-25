"""ONE-OFF: the torch kernel's exact-tie fallback. Real PCG64 uniforms almost never tie, so feed both
kernels a stub generator whose uniforms are rounded to 2 decimals (many ties per row): torch must reproduce
np.argsort's tie order through the host fallback, bit for bit.

    PYTHONPATH=. .venv/Scripts/python.exe scratchpad/perf_mscr/check_tie_fallback.py [device]
"""
from __future__ import annotations

import sys

import numpy as np

from cdd_oran.discovery.mscr import MSCRConfig, _rowsums_cpu, _strata_plan
from cdd_oran.discovery.mscr_torch import rowsums_torch


class TieRng:
    """Quantized uniforms (ties everywhere); not jumpable, so both kernels draw sequentially."""

    def __init__(self, seed: int):
        self._g = np.random.default_rng(seed)
        self.bit_generator = np.random.MT19937(0)

    def random(self, shape):
        return np.round(self._g.random(shape), 2)


def main() -> int:
    device = sys.argv[1] if len(sys.argv) > 1 else "torch:cpu"
    rng = np.random.default_rng(1)
    cfg = MSCRConfig(nc=3, nb=8, min_stratum=10, n_perm=50)
    x = rng.uniform(size=(600, 3))
    y = rng.normal(size=(600, 2))
    plan = _strata_plan(x, cfg)
    ys = [[[y[rows, j] for rows, _, _ in kept] for kept in plan] for j in range(2)]
    a = _rowsums_cpu(ys, plan, cfg, [TieRng(j) for j in range(2)], 1, 1 << 12)
    b = rowsums_torch(ys, plan, cfg, [TieRng(j) for j in range(2)], device, 1, 1 << 12)
    ok = all(np.array_equal(a[j][g][k], b[j][g][k]) for j in range(2) for g in range(3) for k in range(len(plan[g])))
    print(f"tie fallback on {device}: {'IDENTICAL' if ok else 'MISMATCH'}")

    # device OOM -> row-halving retry: fake an OOM for any block with more than 3 rows
    import torch

    from cdd_oran.discovery import mscr_torch
    real = mscr_torch._groupsums

    def flaky(u, y_dev, offsets, dev):
        if u.shape[1] > 3:
            raise torch.OutOfMemoryError("fake")
        return real(u, y_dev, offsets, dev)

    setattr(mscr_torch, "_groupsums", flaky)  # noqa: B010 - test-only monkeypatch
    try:
        c = rowsums_torch(ys, plan, cfg, [np.random.default_rng([3, j]) for j in range(2)], device, 1, 1 << 12)
    finally:
        mscr_torch._groupsums = real
    d = _rowsums_cpu(ys, plan, cfg, [np.random.default_rng([3, j]) for j in range(2)], 1, 1 << 12)
    ok2 = all(np.array_equal(c[j][g][k], d[j][g][k]) for j in range(2) for g in range(3) for k in range(len(plan[g])))
    print(f"OOM row-halving retry on {device}: {'IDENTICAL' if ok2 else 'MISMATCH'}")
    return int(not (ok and ok2))


if __name__ == "__main__":
    sys.exit(main())
