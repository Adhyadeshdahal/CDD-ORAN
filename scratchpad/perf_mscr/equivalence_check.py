"""ONE-OFF Tier-A equivalence check: optimized cdd_oran.discovery.mscr (+ mscr_torch) vs the frozen v2 code
(scratchpad/perf_mscr/mscr_v2_reference.py = verbatim copy of feat/v2 2918ecb). Not a suite test.

Bit-for-bit equality of pvals, s_star, declared and, stricter, of the raw bank (denom and every null
draw) and of the generator's end state, over random small cases incl. ties, dropped tiny strata,
constant columns/targets, discrete y, lagged-conditioner columns, real E2/E5 corpora, and every
execution path: serial numpy, threaded numpy with PCG64 jump-ahead, threaded numpy with a non-jumpable
generator (MT19937, sequential draws), several block sizes, and the torch kernel on 'torch:cpu'.
Pass a device (e.g. `cuda`) to check only that device's paths.

    PYTHONPATH=. .venv/Scripts/python.exe scratchpad/perf_mscr/equivalence_check.py [device]
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

from cdd_oran.discovery import mscr as new
from cdd_oran.discovery.mscr import MSCRConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("mscr_ref", os.path.join(_HERE, "mscr_v2_reference.py"))
assert _spec is not None and _spec.loader is not None
ref = importlib.util.module_from_spec(_spec)
sys.modules["mscr_ref"] = ref
_spec.loader.exec_module(ref)

# (n_jobs, device, block_elems)
PATHS = [(1, "cpu", None), (1, "cpu", 37), (3, "cpu", None), (3, "cpu", 1000), (4, "cpu", 1),
         (1, "torch:cpu", None), (3, "torch:cpu", 5000)]


def cases():
    rng = np.random.default_rng(20260925)
    out = []
    for k in range(8):  # plain continuous, gated signal
        n, c, nt = int(rng.integers(60, 700)), int(rng.integers(2, 7)), int(rng.integers(1, 4))
        npar = int(rng.integers(1, c + 1))
        x = rng.uniform(size=(n, c))
        y = np.column_stack([x[:, 0] * (x[:, -1] > 0.6) + 0.3 * rng.normal(size=n) for _ in range(nt)])
        out.append((f"cont{k}", x, y, npar, MSCRConfig(n_perm=int(rng.integers(19, 120)))))
    n = 300
    x = rng.integers(0, 3, size=(n, 4)).astype(float)  # heavy ties in every column
    y = rng.integers(0, 4, size=(n, 2)).astype(float) * 0.1  # discrete targets
    out.append(("ties", x, y, 3, MSCRConfig(n_perm=99)))
    x = rng.uniform(size=(250, 4))
    x[:, 1] = 7.0  # constant column (as candidate and as conditioner)
    y = np.column_stack([np.full(250, 1.1), x[:, 0] + rng.normal(size=250)])  # constant target (audit A2)
    out.append(("constant", x, y, 3, MSCRConfig(n_perm=59)))
    x = rng.uniform(size=(150, 3))  # strata of 25 < min_stratum -> all dropped
    out.append(("all_dropped", x, x[:, :1] + rng.normal(size=(150, 1)), 2, MSCRConfig(n_perm=39)))
    x = rng.uniform(size=(260, 3))  # strata of 43/44: min_stratum 44 drops some
    out.append(("some_dropped", x, rng.normal(size=(260, 2)), 3, MSCRConfig(n_perm=49, min_stratum=44)))
    x = rng.uniform(size=(200, 5))  # odd nc/nb, bins < 8 rows (n<8 pairwise branch)
    out.append(("odd_cfg", x, rng.normal(size=(200, 1)), 5, MSCRConfig(nc=5, nb=7, min_stratum=10, n_perm=77)))
    x = rng.uniform(size=(640, 3))  # segments > 128 rows: recursive pairwise path
    out.append(("long_seg", x, (x[:, 0] + rng.normal(size=640))[:, None], 3,
                MSCRConfig(nc=2, nb=2, min_stratum=10, n_perm=31)))
    out.append(("frozen_B", rng.uniform(size=(300, 3)), rng.normal(size=(300, 1)), 2, MSCRConfig()))  # B=2999
    from cdd_oran.e2slice.dataset import E2DatasetConfig, generate_rows
    from scripts.e5_baselines import corpus as e5_corpus
    r = generate_rows(E2DatasetConfig(n_rows_per_seed=1500, seed=3))
    out.append(("E2_real", np.concatenate([r.x_params, r.x_kpis], 1), r.y_kpis, 8, MSCRConfig(n_perm=199)))
    xe, ye = e5_corpus(n=1500, seed=3)
    out.append(("E5_real", xe, ye, xe.shape[1], MSCRConfig(n_perm=199)))
    return out


def same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True)


def same_state(r0: np.random.Generator, r1: np.random.Generator) -> bool:
    return str(r0.bit_generator.state) == str(r1.bit_generator.state)


def main() -> int:
    global PATHS
    if len(sys.argv) > 1:  # e.g. `cuda`: only that device's paths
        PATHS = [(1, sys.argv[1], None), (3, sys.argv[1], None), (3, sys.argv[1], 5000)]
    bad = 0
    for name, x, y, npar, cfg in cases():
        seed = 11
        r0 = ref.discover_mscr(x, y, n_params=npar, seed=seed, config=cfg)
        bank_seed = [seed, 0, cfg.n_perm]
        g0 = np.random.default_rng(bank_seed)
        d0, n0, _ = ref._build_bank(x, y[:, 0], cfg, g0)
        m0 = np.random.Generator(np.random.MT19937(5))
        dm0, nm0, _ = ref._build_bank(x, y[:, 0], cfg, m0)
        for n_jobs, device, block in PATHS:
            if block is None:
                r1 = new.discover_mscr(x, y, n_params=npar, seed=seed, config=cfg, n_jobs=n_jobs, device=device)
                ok_out = same(r0.pvals, r1.pvals) and same(r0.s_star, r1.s_star) and same(r0.declared, r1.declared)
            else:
                ok_out = True  # block size is covered by the bank comparison below
            g1 = np.random.default_rng(bank_seed)
            d1, n1, _ = new._build_bank(x, y[:, 0], cfg, g1, n_jobs=n_jobs, device=device, block_elems=block)
            ok_bank = same(d0, d1) and same(n0, n1) and same_state(g0, g1)
            m1 = np.random.Generator(np.random.MT19937(5))
            dm1, nm1, _ = new._build_bank(x, y[:, 0], cfg, m1, n_jobs=n_jobs, device=device, block_elems=block)
            ok_mt = same(dm0, dm1) and same(nm0, nm1) and same_state(m0, m1)
            if not (ok_out and ok_bank and ok_mt):
                bad += 1
                print(f"  MISMATCH {name} n_jobs={n_jobs} device={device} block={block}: "
                      f"outputs={ok_out} bank+state={ok_bank} mt19937={ok_mt}")
        print(f"{name:13s} n={x.shape[0]:4d} cols={x.shape[1]:2d} targets={y.shape[1]} B={cfg.n_perm:4d} "
              f"declared={int(r0.declared.sum()):2d}  checked {len(PATHS)} execution paths")
    print("ALL IDENTICAL" if bad == 0 else f"{bad} MISMATCHES")
    return int(bad > 0)


if __name__ == "__main__":
    sys.exit(main())
