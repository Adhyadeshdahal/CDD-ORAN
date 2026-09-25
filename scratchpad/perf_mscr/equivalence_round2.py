"""ONE-OFF round-2 equivalence (sol merge conditions), optimized bank vs frozen v2 reference:

  R1 caller generator with a POPULATED 32-bit buffer (``rng.integers(..., dtype=int32)`` beforehand):
     exact final state incl. has_uint32 / uinteger must match v2's sequential consumption.
  R2 block boundaries: last block exactly one row; block_elems < ns (the max(1, ...) one-row path).
  R3 full output (pvals, s_star, declared) across n_jobs in {1, 2, 3, 5}, several targets, buffered RNGs.
  R4 device-string validation fails early.

    PYTHONPATH=. .venv/Scripts/python.exe scratchpad/perf_mscr/equivalence_round2.py [device ...]
Devices default to cpu and torch:cpu; pass e.g. `cpu torch:cpu cuda`.
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


def buffered_rng(seed, kind="pcg"):
    """Generator whose 32-bit buffer is populated (asserted)."""
    bg = np.random.PCG64(seed) if kind == "pcg" else np.random.MT19937(seed)
    g = np.random.Generator(bg)
    g.integers(0, 1000, size=3, dtype=np.int32)  # odd number of 32-bit draws -> one buffered half
    if kind == "pcg":
        assert g.bit_generator.state["has_uint32"] == 1, g.bit_generator.state
    return g


def state(g) -> str:
    return str(g.bit_generator.state)


def outputs(x, y, npar, cfg, banks):
    """discover_mscr's post-bank pipeline on given banks (same functions, same order)."""
    n_t = y.shape[1]
    s = np.zeros((n_t, npar))
    p = np.zeros((n_t, npar))
    d = np.zeros((n_t, npar), dtype=bool)
    for j in range(n_t):
        denom, null, strata = banks[j]
        for i in range(npar):
            s[j, i] = ref._s_star(x[:, i], i, denom, strata, cfg)
            p[j, i] = ref._pval(s[j, i], i, null)
        d[j] = ref.by_declare(p[j], cfg.q)
    return p, s, d


def eq(a, b) -> bool:
    return all(np.asarray(u).dtype == np.asarray(v).dtype and np.array_equal(u, v) for u, v in zip(a, b, strict=True))


def main() -> int:
    devices = sys.argv[1:] or ["cpu", "torch:cpu"]
    rng = np.random.default_rng(7)
    bad = 0
    # n=300, nc=6 -> every stratum has ns=50 rows: block_elems=200 -> 4 rows/block, n_perm=41 -> last block = 1 row
    x = rng.uniform(size=(300, 4))
    y = np.column_stack([x[:, 0] * (x[:, 3] > 0.5) + 0.2 * rng.normal(size=300), rng.normal(size=300),
                         x[:, 1] + rng.normal(size=300)])
    cfg = MSCRConfig(n_perm=41, min_stratum=10)
    plan = new._strata_plan(x, cfg)
    assert {len(r) for kept in plan for r, _, _ in kept} == {50}
    tasks, _ = new._stream_tasks(plan, cfg.n_perm, 200)
    assert tasks[10][3] - tasks[10][2] == 1 and tasks[10][2] == 40, tasks[:11]  # 11th task: rows 40..41
    t1, _ = new._stream_tasks(plan, cfg.n_perm, 10)
    assert all(hi - lo == 1 for _, _, lo, hi, _ in t1)  # block_elems < ns -> one row per task

    for dev in devices:
        for kind in ("pcg", "mt"):
            for block in (200, 10, 50, None):
                for n_jobs in (1, 2, 3, 5):
                    r_ref = [buffered_rng([3, j], kind) for j in range(3)]
                    r_new = [buffered_rng([3, j], kind) for j in range(3)]
                    b_ref = [ref._build_bank(x, y[:, j], cfg, r_ref[j]) for j in range(3)]
                    b_new = new._build_banks(x, y, cfg, r_new, n_jobs=n_jobs, device=dev, block_elems=block)
                    ok_bank = all(np.array_equal(b_ref[j][0], b_new[j][0]) and np.array_equal(b_ref[j][1], b_new[j][1])
                                  for j in range(3))
                    ok_state = all(state(a) == state(b) for a, b in zip(r_ref, r_new, strict=True))
                    ok_out = eq(outputs(x, y, 4, cfg, b_ref), outputs(x, y, 4, cfg, b_new))
                    if not (ok_bank and ok_state and ok_out):
                        bad += 1
                        print(f"  MISMATCH dev={dev} rng={kind} block={block} n_jobs={n_jobs}: "
                              f"bank={ok_bank} state={ok_state} outputs={ok_out}")
        print(f"{dev}: R1 buffered uint32 state + R2 block boundaries (last block 1 row; block<ns) + "
              f"R3 3 targets x n_jobs(1,2,3,5) x PCG64/MT19937 -> {'IDENTICAL' if bad == 0 else 'MISMATCH'}")
        ex = buffered_rng([3, 0])
        print(f"   example final state after bank: has_uint32={ex.bit_generator.state['has_uint32']} (before), "
              f"after={_after(x, y, cfg, dev)}")

    for badname in ("gpu", "cuda:99", "torch:nonsense", "CPU"):
        try:
            new.resolve_device(badname)
        except ValueError as e:
            print(f"R4 resolve_device({badname!r}) -> ValueError: {e}")
        else:
            bad += 1
            print(f"R4 resolve_device({badname!r}) accepted (MISMATCH)")
    print("ALL IDENTICAL" if bad == 0 else f"{bad} FAILURES")
    return int(bad > 0)


def _after(x, y, cfg, dev) -> str:
    a, b = buffered_rng([3, 0]), buffered_rng([3, 0])
    ref._build_bank(x, y[:, 0], cfg, a)
    new._build_bank(x, y[:, 0], cfg, b, n_jobs=3, device=dev)
    sa, sb = a.bit_generator.state, b.bit_generator.state
    return (f"ref(has_uint32={sa['has_uint32']}, uinteger={sa['uinteger']}) "
            f"new(has_uint32={sb['has_uint32']}, uinteger={sb['uinteger']}) equal={str(sa) == str(sb)}")


if __name__ == "__main__":
    sys.exit(main())
