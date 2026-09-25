"""ONE-OFF: torch ``_reduceat_last`` vs ``np.add.reduceat`` on every segment-length regime, incl. signed
zeros: length 1 (singleton, returned unchanged by numpy), 2..8 (rest < 8: numpy's scalar loop), 9..129
(8-accumulator block), > 129 (recursive split), with all-(-0.0) segments and mixed +/-0.0.

    PYTHONPATH=. .venv/Scripts/python.exe scratchpad/perf_mscr/check_segments.py [device ...]
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from cdd_oran.discovery.mscr_torch import _reduceat_last

LENGTHS = [1, 2, 3, 7, 8, 9, 10, 16, 17, 100, 128, 129, 130, 131, 257, 500, 501]


def rows_for(length: int, rng) -> np.ndarray:
    r = rng.normal(size=(6, length)) * np.exp(rng.normal(size=(6, length)) * 4)
    r[0] = -0.0                                        # all negative zero
    r[1] = 0.0                                         # all positive zero
    r[2, ::2] = -0.0                                   # mixed signed zeros
    r[2, 1::2] = 0.0
    r[3, 0] = -0.0                                     # singleton / first element -0.0, rest data
    return r


def main() -> int:
    devices = sys.argv[1:] or ["cpu"]
    rng = np.random.default_rng(0)
    total = 0
    for dev in devices:
        bad = 0
        # segments of mixed lengths in one row, as reduceat sees them (every pair of lengths adjacent)
        for a in LENGTHS:
            for b in (1, 2, 9, 130):
                blocks = [rows_for(a, rng), rows_for(b, rng), rows_for(a, rng)]
                arr = np.concatenate(blocks, axis=1)
                offsets = np.array([0, a, a + b])
                ref = np.add.reduceat(arr, offsets, axis=1)
                got = _reduceat_last(torch.from_numpy(arr).to(dev), offsets).cpu().numpy()
                same = np.array_equal(ref, got) and np.array_equal(np.signbit(ref), np.signbit(got))
                if not same:
                    bad += 1
                    print(f"MISMATCH device={dev} lengths=({a},{b},{a})")
        total += bad
        print(f"{dev}: {len(LENGTHS) * 4} length combos incl. signed zeros -> "
              f"{'IDENTICAL (values and zero signs)' if bad == 0 else 'MISMATCHES'}")
    # what numpy does with signed zeros (documents the semantics being replicated)
    z = np.array([[-0.0, -0.0, -0.0, -0.0]])
    print("numpy reduceat([-0]) =", np.add.reduceat(z[:, :1], [0], axis=1)[0, 0],
          " reduceat([-0,-0,-0,-0]) =", np.add.reduceat(z, [0], axis=1)[0, 0])
    return int(total > 0)


if __name__ == "__main__":
    sys.exit(main())
