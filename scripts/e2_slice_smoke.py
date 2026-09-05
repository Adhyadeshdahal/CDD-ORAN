"""TRUTH-FREE memory/runtime SMOKE TEST for the E2 discovery pipeline (Plan 007 §15 op note).

Purpose: confirm the pipeline runs end-to-end and MEASURE cost, to project the full frozen
``4000 x 999 x 84 x 10-seed`` job's wall-clock + peak memory. It is TRUTH-FREE (never reads
``true_adj_matrix``; never runs recovery) and uses clearly-labelled THROWAWAY constants -- NEVER the
frozen §14 constants. Nothing is persisted to the canonical discovery artifact path; the tiny
dataset is generated in-memory and discarded.

    uv run python -m scripts.e2_slice_smoke

The frozen §14 constants remain untouched; this harness must not be used to justify reducing them.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import replace

import numpy as np

from cdd_oran.e2slice.dataset import E2DatasetConfig, generate_rows
from cdd_oran.e2slice.discovery import (
    FROZEN_B_PERM,
    E2DiscoveryConfig,
    discover_graph,
    dist_matrix_1d,
    dist_matrix_euclidean,
    frozen_config,
    u_center,
    u_inner,
)

# --- THROWAWAY smoke constants (NOT the frozen §14 values) -------------------
SMOKE_N = 200          # rows (frozen real: 4000)
SMOKE_B_PERM = 49      # permutations (frozen real: 999)
SMOKE_SEED = 0
MICRO_SIZES = (150, 300)  # two n values to empirically fit the O(n^2) permutation-op scaling

# --- frozen real-job scale (for extrapolation only; NEVER used to run) -------
REAL_N = 4000
REAL_B_PERM = FROZEN_B_PERM  # 999
REAL_CANDIDATES = 84
REAL_SEEDS = 10


def _peak_working_set_bytes() -> int | None:
    """Peak process working set via the Windows API (ctypes); None off Windows / on failure."""
    try:
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.GetCurrentProcess.restype = wintypes.HANDLE
        k.K32GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD
        ]
        k.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = _PMC()
        counters.cb = ctypes.sizeof(_PMC)
        if not k.K32GetProcessMemoryInfo(
            k.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return None
        return int(counters.PeakWorkingSetSize)
    except Exception:
        return None


def _time_single_permutation_op(n: int, repeats: int = 30) -> float:
    """Mean wall-clock of ONE permutation inner op at row count ``n`` (the perm-loop hot path).

    Mirrors the discovery permutation body: permute the candidate column, recompute its U-centered
    distance matrix, project out the fixed 13-dim conditioning, and recompute |pdCor|. The fixed
    conditioning / target matrices are precomputed once (as in ``discover_graph``) and reused.
    """
    rng = np.random.default_rng(0)
    xi = rng.standard_normal(n)
    z = rng.standard_normal((n, 13))
    yj = rng.standard_normal(n)
    cond_u = u_center(dist_matrix_euclidean(z))
    cond_self = u_inner(cond_u, cond_u)
    tgt_u = u_center(dist_matrix_1d(yj))
    pyz = tgt_u - (u_inner(tgt_u, cond_u) / cond_self) * cond_u
    vb_proj = u_inner(pyz, pyz)

    # Warm up (JIT-free numpy, but touch caches once).
    _ = u_center(dist_matrix_1d(xi[rng.permutation(n)]))

    start = time.perf_counter()
    for _ in range(repeats):
        cand_u_p = u_center(dist_matrix_1d(xi[rng.permutation(n)]))
        pxz_p = cand_u_p - (u_inner(cand_u_p, cond_u) / cond_self) * cond_u
        va_proj_p = u_inner(pxz_p, pxz_p)
        denom = (va_proj_p * vb_proj) ** 0.5
        _ = abs(u_inner(pxz_p, pyz) / denom) if denom > 0 else 0.0
    return (time.perf_counter() - start) / repeats


def _fmt_bytes(b: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(b) < 1024 or unit == "GiB":
            return f"{b:.2f} {unit}"
        b /= 1024
    return f"{b:.2f} GiB"


def _fmt_secs(s: float) -> str:
    if s < 90:
        return f"{s:.2f} s"
    if s < 5400:
        return f"{s / 60:.2f} min"
    return f"{s / 3600:.2f} h"


def main() -> int:
    print("=== E2 discovery TRUTH-FREE smoke test (throwaway constants) ===")
    print(f"smoke: N={SMOKE_N}, B_perm={SMOKE_B_PERM}, seeds=1  (frozen real: "
          f"N={REAL_N}, B_perm={REAL_B_PERM}, candidates={REAL_CANDIDATES}, seeds={REAL_SEEDS})")
    assert SMOKE_N != REAL_N and SMOKE_B_PERM != REAL_B_PERM, "smoke must not use frozen constants"

    # (1) End-to-end run at smoke scale (in-memory dataset, no truth, no persisted mask).
    ds_cfg = E2DatasetConfig(n_rows_per_seed=SMOKE_N, seed=SMOKE_SEED, sampling_seed=0)
    rows = generate_rows(ds_cfg)
    smoke_cfg = replace(frozen_config(), b_perm=SMOKE_B_PERM)
    assert isinstance(smoke_cfg, E2DiscoveryConfig)

    tracemalloc.start()
    base_peak_ws = _peak_working_set_bytes()
    t0 = time.perf_counter()
    result = discover_graph(rows, smoke_cfg)
    smoke_wall = time.perf_counter() - t0
    _tm_cur, tm_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_peak_ws = _peak_working_set_bytes()

    n_selected = int(result.binary_mask.sum())
    n_guarded = int(result.guarded_mask.sum())
    print(f"\n[1] end-to-end discover_graph: {_fmt_secs(smoke_wall)}  "
          f"(selected {n_selected}, guarded {n_guarded} of 84)")
    print(f"    tracemalloc peak (Python-tracked): {_fmt_bytes(tm_peak)}")
    if end_peak_ws is not None:
        print(f"    process PeakWorkingSet: {_fmt_bytes(end_peak_ws)} "
              f"(baseline at start: {_fmt_bytes(base_peak_ws or 0)})")

    # (2) Micro-benchmark the permutation hot path at two sizes to fit O(n^2) scaling.
    micro = {n: _time_single_permutation_op(n) for n in MICRO_SIZES}
    n1, n2 = MICRO_SIZES
    exponent = np.log(micro[n2] / micro[n1]) / np.log(n2 / n1)
    # Fit t(n) = c * n^2 from the larger micro size (closest to the hot path shape).
    c = micro[n2] / (n2 ** 2)
    per_perm_real = c * (REAL_N ** 2)
    print(f"\n[2] single-permutation op: t({n1})={micro[n1] * 1e3:.3f} ms, "
          f"t({n2})={micro[n2] * 1e3:.3f} ms  -> empirical exponent {exponent:.2f} (expect ~2.0)")
    print(f"    fitted t(n) = {c:.3e} * n^2  ->  t({REAL_N}) = {per_perm_real * 1e3:.2f} ms/perm")

    # (3) Extrapolate the full frozen job (permutation-dominated: the perm loop is the hot path).
    total_perms = REAL_CANDIDATES * REAL_B_PERM * REAL_SEEDS  # 84 * 999 * 10
    est_perm_wall = per_perm_real * total_perms
    # Non-perm overhead (per-candidate distance matrices etc.) scales the same way; estimate it from
    # the smoke end-to-end minus its perm share, then rescale by n^2 and count.
    smoke_perm_share = _time_single_permutation_op(SMOKE_N) * (REAL_CANDIDATES * SMOKE_B_PERM)
    smoke_overhead = max(smoke_wall - smoke_perm_share, 0.0)
    overhead_real = smoke_overhead * ((REAL_N / SMOKE_N) ** 2) * REAL_SEEDS
    est_total_wall = est_perm_wall + overhead_real
    print(f"\n[3] EXTRAPOLATION to the full frozen {REAL_N}x{REAL_B_PERM}x{REAL_CANDIDATES}"
          f"x{REAL_SEEDS}-seed job:")
    print(f"    total permutations = {REAL_CANDIDATES} x {REAL_B_PERM} x {REAL_SEEDS} = {total_perms:,}")
    print(f"    perm-loop wall-clock  ~ {_fmt_secs(est_perm_wall)}")
    print(f"    + non-perm overhead   ~ {_fmt_secs(overhead_real)}")
    print(f"    TOTAL est. wall-clock ~ {_fmt_secs(est_total_wall)}  "
          f"(~{_fmt_secs(est_total_wall / REAL_SEEDS)} per seed)")

    # (4) MEASURED peak-memory scaling. Tiny-N matrices are negligible, so measure the process peak
    # working set while running ONE full per-(i,j) + permutation hot path at a MODERATE n (matrices
    # large enough to dominate the working set), then project to N=4000 by the N^2 array footprint.
    n_mem = 2000
    before = _peak_working_set_bytes()
    rng = np.random.default_rng(0)
    xi = rng.standard_normal(n_mem)
    z = rng.standard_normal((n_mem, 13))
    yj = rng.standard_normal(n_mem)
    cond_u = u_center(dist_matrix_euclidean(z))
    cond_self = u_inner(cond_u, cond_u)
    tgt_u = u_center(dist_matrix_1d(yj))
    pyz = tgt_u - (u_inner(tgt_u, cond_u) / cond_self) * cond_u
    cand_u = u_center(dist_matrix_1d(xi))
    pxz = cand_u - (u_inner(cand_u, cond_u) / cond_self) * cond_u
    _ = u_inner(pxz, pyz)  # keep the live set referenced through the measurement
    after = _peak_working_set_bytes()
    nn_matrix = REAL_N * REAL_N * 8  # one (N x N) float64 matrix at the real scale
    print(f"\n[4] MEASURED peak-memory at n={n_mem} (one hot-path evaluation): ", end="")
    if before is not None and after is not None:
        measured_footprint = after - before
        # Project the per-op footprint to N=4000 by the N^2 array-size ratio.
        projected = after + measured_footprint * ((REAL_N / n_mem) ** 2 - 1)
        print(f"PeakWorkingSet {_fmt_bytes(after)} (+{_fmt_bytes(measured_footprint)} vs before)")
        print(f"    array footprint scales as N^2 -> projected peak at N={REAL_N}: "
              f"~{_fmt_bytes(projected)}")
    else:
        print("(process peak-WS probe unavailable on this platform)")
    # Theoretical dominant term: the live set of (N x N) float64 matrices held simultaneously in the
    # hot path (candidate A~, conditioning C~, target B~, Pyz, Pxz, permuted A~, Pxz_p, plus the
    # euclidean-distance d2 + Gram temporaries during C~ construction) -- ~9 as a conservative bound.
    est_live_matrices = 9
    print(f"    theoretical: one (NxN) float64 matrix = {_fmt_bytes(nn_matrix)}; "
          f"~{est_live_matrices} live -> ~{_fmt_bytes(nn_matrix * est_live_matrices)} peak at N={REAL_N}")
    print("\nsmoke complete: no truth read, no recovery run, no canonical mask persisted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
