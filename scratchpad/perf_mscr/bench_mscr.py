"""MSCR-v2 speed/memory benchmark at the REAL operating points. Run later, on an idle machine.

    PYTHONPATH=. .venv/Scripts/python.exe scratchpad/perf_mscr/bench_mscr.py            # full matrix
    PYTHONPATH=. .venv/Scripts/python.exe scratchpad/perf_mscr/bench_mscr.py --quick    # E2 n=4000 only

Every configuration runs in a FRESH subprocess (clean peak-RSS / CUDA-memory numbers, no warm caches
shared between configs). Matrix:

    corpus   : E2 n=4000 (14 cols, 6 targets) | E2 n=24000 | E5 n=24000 (4 cols, 4 targets); B=2999
    impl     : v2-reference (frozen code, scratchpad/perf_mscr/mscr_v2_reference.py)
               cpu n_jobs=1 | cpu n_jobs=2,4,...,cpu_count | cuda (if available) with n_jobs=1 and =cpu_count

Each run records wall time, peak RSS of the process, CUDA peak allocation, and a SHA-256 of
(pvals, s_star, declared). Every non-reference run must hash-match the reference for the same corpus;
a mismatch is printed as FAIL and the run's timing must not be reported. Output: runs/perf-mscr/bench.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(_REPO, "runs", "perf-mscr", "bench.json")
SEED = 0


def _peak_rss_mb() -> float:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),  # type: ignore[attr-defined]
                                                 ctypes.byref(pmc), pmc.cb)
        return pmc.PeakWorkingSetSize / 2**20
    import resource
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1024 if sys.platform != "darwin" else kb / 2**20


def _corpus(name: str):
    import numpy as np

    if name.startswith("E2"):
        from cdd_oran.e2slice.dataset import E2DatasetConfig, generate_rows
        r = generate_rows(E2DatasetConfig(n_rows_per_seed=int(name.split("_")[1]), seed=SEED))
        return np.concatenate([r.x_params, r.x_kpis], 1), r.y_kpis, 8
    from scripts.e5_baselines import NP, corpus
    x, y = corpus(n=int(name.split("_")[1]), seed=SEED)
    return x, y, NP


def child(corpus: str, impl: str, n_jobs: int, device: str) -> dict:
    """One configuration, in this (fresh) process."""
    import numpy as np

    sys.path.insert(0, _REPO)
    x, y, npar = _corpus(corpus)
    base_rss = _peak_rss_mb()
    t0 = time.perf_counter()
    if impl == "v2-reference":
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mscr_ref", os.path.join(_REPO, "scratchpad", "perf_mscr", "mscr_v2_reference.py"))
        assert spec is not None and spec.loader is not None
        ref = importlib.util.module_from_spec(spec)
        sys.modules["mscr_ref"] = ref
        spec.loader.exec_module(ref)
        res = ref.discover_mscr(x, y, n_params=npar, seed=SEED)
    else:
        from cdd_oran.discovery.mscr import discover_mscr
        res = discover_mscr(x, y, n_params=npar, seed=SEED, n_jobs=n_jobs, device=device)
    wall = time.perf_counter() - t0
    cuda_peak = None
    if device.startswith("cuda"):
        import torch
        cuda_peak = torch.cuda.max_memory_allocated() / 2**20
    h = hashlib.sha256()
    for a in (res.pvals, res.s_star, res.declared):
        h.update(np.ascontiguousarray(a).tobytes())
    return {"corpus": corpus, "impl": impl, "n_jobs": n_jobs, "device": device, "wall_s": round(wall, 2),
            "peak_rss_mb": round(_peak_rss_mb(), 1), "rss_before_mb": round(base_rss, 1),
            "cuda_peak_mb": cuda_peak, "hash": h.hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--child", nargs=4, metavar=("CORPUS", "IMPL", "NJOBS", "DEVICE"))
    a = ap.parse_args()
    if a.child:
        print(json.dumps(child(a.child[0], a.child[1], int(a.child[2]), a.child[3])))
        return
    corpora = ["E2_4000"] if a.quick else ["E2_4000", "E2_24000", "E5_24000"]
    ncpu = os.cpu_count() or 1
    jobs = sorted({1, *[2 ** k for k in range(1, 8) if 2 ** k < ncpu], ncpu})
    cuda = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
                          capture_output=True, text=True).stdout.strip() == "True"
    configs = [("v2-reference", 1, "cpu")] + [("optimized", j, "cpu") for j in jobs]
    if cuda:
        configs += [("optimized", 1, "cuda"), ("optimized", ncpu, "cuda")]
    env = {**os.environ, "PYTHONPATH": _REPO}
    records = []
    for corpus in corpora:
        ref_hash = None
        for impl, j, dev in configs:
            out = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", corpus, impl, str(j), dev],
                                 capture_output=True, text=True, env=env, cwd=_REPO)
            if out.returncode != 0:
                print(f"{corpus} {impl} n_jobs={j} {dev}: ERROR\n{out.stderr[-2000:]}")
                continue
            rec = json.loads(out.stdout.strip().splitlines()[-1])
            if impl == "v2-reference":
                ref_hash = rec["hash"]
            rec["identical_to_reference"] = rec["hash"] == ref_hash
            base = next((r["wall_s"] for r in records if r["corpus"] == corpus and r["impl"] == "v2-reference"), None)
            rec["speedup_vs_reference"] = round(base / rec["wall_s"], 2) if base else None
            records.append(rec)
            flag = "" if rec["identical_to_reference"] else "   FAIL: output differs from v2, do not report"
            print(f"{corpus:9s} {impl:13s} n_jobs={j:3d} {dev:5s} wall={rec['wall_s']:8.1f}s "
                  f"x{rec['speedup_vs_reference']}  peakRSS={rec['peak_rss_mb']}MB cuda={rec['cuda_peak_mb']}{flag}",
                  flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"cpu_count": ncpu, "cuda": cuda, "records": records}, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
