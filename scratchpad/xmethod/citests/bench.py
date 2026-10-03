"""CPU-s / wall / peak-RAM benchmark of the citests adapters (one (method, n, mode) per process).

  python scratchpad/xmethod/citests/bench.py METHOD N [--mode full|probe] [--out FILE.jsonl] [--native]

full : run the adapter on one synthetic planted dataset (_citests_synth, 36 candidates, 8 conditioners, DEV seed
       3_000_000), default config (B 9999 Besag-Clifford h 20) unless --native.
probe: run on two candidates only, P0->K0 (true, strong: runs to B) and P_placebo->K0 (null: stops early), to get
       setup cost + cost per true / null edge for extrapolation of cells too heavy to run in full.
Peak RAM = process peak RSS (Linux ru_maxrss; Windows PeakWorkingSetSize).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import sys
import time


def peak_rss_mb() -> float:
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
        k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.PeakWorkingSetSize / 2**20
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("method")
    ap.add_argument("n", type=int)
    ap.add_argument("--mode", default="full", choices=["full", "probe"])
    ap.add_argument("--native", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from cdd_oran.xmethod.methods import _citests_synth as synth
    from cdd_oran.xmethod.methods.cmi_knn import CMIKnnMethod
    from cdd_oran.xmethod.methods.mscr import MSCRMethod
    from cdd_oran.xmethod.methods.pcorr import PCorrMethod
    from cdd_oran.xmethod.methods.pdcor import PDCorMethod
    from cdd_oran.xmethod.methods.rcot2 import RCoT2Method

    cls = {"mscr": MSCRMethod, "pcorr": PCorrMethod, "pdcor": PDCorMethod, "rcot2": RCoT2Method,
           "cmi_knn": CMIKnnMethod}[a.method]
    m = cls()
    ds, truth = synth.make("planted", a.n, 3_000_000)
    if a.mode == "probe":
        ds = dataclasses.replace(ds, candidates=(("P0", "K0"), ("P_placebo", "K0")))
    cfg = m.native_config() if a.native and hasattr(m, "native_config") else m.default_config()
    base_rss = peak_rss_mb()
    t0, c0 = time.time(), time.process_time()
    r = m.run(ds, cfg)
    rec = {"method": a.method, "n": a.n, "mode": a.mode, "native": a.native, "cpu_s": time.process_time() - c0,
           "wall_s": time.time() - t0, "peak_rss_mb": peak_rss_mb(), "rss_before_mb": base_rss,
           "n_candidates": len(ds.candidates), "n_true": sum(c in truth.edges for c in ds.candidates),
           "mc_draws": r.notes.get("mc_draws"), "stop": r.notes.get("stop"),
           "declared": [f"{e.source}->{e.target}" for e in r.edges if e.declared],
           "host": platform.node(), "cpus": os.cpu_count(), "platform": platform.platform(),
           "config": {k: v for k, v in r.config.items()}}
    line = json.dumps(rec)
    print(line, flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
