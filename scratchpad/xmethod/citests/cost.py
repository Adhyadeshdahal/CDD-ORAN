"""R-13 cost campaign: CPU-s and peak RAM per (method, dataset) at n in {500, 1000, 4000, 8000, 24000} on DEV data.

  python scratchpad/xmethod/citests/cost.py all OUT.jsonl [--methods ...] [--worlds E1,...] [--ns ...]
                 [--budget 7200] [--probe 1200] [--jobs 4] [--arm eq|native]
  python scratchpad/xmethod/citests/cost.py task OUT.jsonl METHOD WORLD N CAP [ARM]

--arm (R-17 / R-18): conditioning arm (default eq, the primary). Run xm-citests-cost-1 predates the arms = native.

Each (method, world) is a chain over n ascending, single-threaded (BLAS / numba / MSCR / tigramite threads = 1), one
DEV harness dataset per cell (regime R1, seed 3_000_001), default (primary) config. A task runs in its own process
with a hard CPU cap (Linux RLIMIT_CPU): CAP = the budget (proposal 2 CPU-h) until the chain first exceeds it; after
that each larger n runs as a PROBE capped at --probe CPU-s. Status per task: "ok" (finished; cpu_s measured),
"over_budget" (killed at the budget: cost > budget, measured), "probe" (killed at the probe cap: cost >= cap
measured, plus an extrapolation = cpu at last finished edge x total / done, a LOWER-BIASED guess because true
edges cost more), "oom" / "error". Dataset generation time is excluded and recorded separately.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

SEED = 3_000_001
HERE = os.path.dirname(os.path.abspath(__file__))
THREAD_ENV = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                               "NUMBA_NUM_THREADS", "MSCR_NUM_THREADS")}


def task(out, method, world, n, cap, arm="eq"):
    try:
        import resource
    except ImportError:          # Windows smoke tests only: no CPU cap
        resource = None
    sys.path.insert(0, HERE)
    from bench import peak_rss_mb

    from cdd_oran.xmethod.methods.cmi_knn import CMIKnnMethod
    from cdd_oran.xmethod.methods.mscr import MSCRMethod
    from cdd_oran.xmethod.methods.pcorr import PCorrMethod
    from cdd_oran.xmethod.methods.pdcor import PDCorMethod
    from cdd_oran.xmethod.methods.rcot2 import RCoT2Method
    from cdd_oran.xmethod.worlds import generate_dataset

    t0 = time.process_time()
    ds, truth = generate_dataset(world, "R1", n, SEED)
    gen_cpu = time.process_time() - t0
    m = {"mscr": MSCRMethod, "pcorr": PCorrMethod, "pdcor": PDCorMethod, "rcot2": RCoT2Method,
         "cmi_knn": CMIKnnMethod}[method]()
    cfg = m.default_config()
    cfg.update({"n_jobs": 1} if method == "mscr" else {"workers": 1} if method == "cmi_knn" else {})
    cfg["arm"] = arm
    cfg["progress_file"] = f"{out}.progress.{method}.{world}.{n}"
    used = time.process_time()
    if resource is not None:
        resource.setrlimit(resource.RLIMIT_CPU, (int(used + cap), int(used + cap) + 30))
    r = m.run(ds, cfg)
    rec = {"method": method, "world": world, "regime": "R1", "n": n, "seed": SEED, "status": "ok", "cap_s": cap,
           "arm": arm, "n_cond": len(r.notes.get("conditioning", [])),
           "cpu_s": r.cpu_s, "gen_cpu_s": gen_cpu, "peak_rss_mb": peak_rss_mb(), "n_candidates": len(ds.candidates),
           "n_true": len(truth.edges & set(ds.candidates)), "max_mc_draws": max([v for v in r.notes.get("mc_draws") or [] if v is not None] or [0]),
           "stop": r.notes.get("stop"), "version": r.version}
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")


def chain(out, method, world, ns, budget, probe, arm="eq"):
    over = False
    for n in ns:
        cap = probe if over else budget
        prog = f"{out}.progress.{method}.{world}.{n}"
        cmd = [sys.executable, os.path.abspath(__file__), "task", out, method, world, str(n), str(cap), arm]
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **THREAD_ENV},
                           timeout=cap * 3 + 600)
        if p.returncode == 0:
            continue
        last = open(prog).read().split("\n")[-2].split() if os.path.exists(prog) and os.path.getsize(prog) else None
        killed = p.returncode in (-24, -9, 152, 137) or "CPU time limit" in p.stderr
        status = ("probe" if over else "over_budget") if killed else ("oom" if "MemoryError" in p.stderr else "error")
        rec = {"method": method, "world": world, "regime": "R1", "n": n, "seed": SEED, "status": status, "cap_s": cap,
               "arm": arm,
               "returncode": p.returncode, "wall_s": time.time() - t0, "stderr": p.stderr[-1500:]}
        if last:
            done, total, cpu = int(last[0]), int(last[1]), float(last[2])
            rec.update(progress_done=done, progress_total=total, cpu_at_last_edge=cpu,
                       extrapolated_cpu_s=cpu * total / max(done, 1))
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(method, world, n, status, flush=True)
        if status in ("over_budget", "probe", "oom"):
            over = True


def main():
    a = sys.argv[1:]
    if a[0] == "task":
        _, out, method, world, n, cap = a[:6]
        task(out, method, world, int(n), float(cap), a[6] if len(a) > 6 else "eq")
        return
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d  # noqa: E731
    out = a[1]
    methods = opt("--methods", "pcorr,rcot2,mscr,pdcor,cmi_knn").split(",")
    worlds = opt("--worlds", "E1,E2,E3,E4,E5").split(",")
    ns = [int(v) for v in opt("--ns", "500,1000,4000,8000,24000").split(",")]
    budget, probe, jobs = float(opt("--budget", 7200)), float(opt("--probe", 1200)), int(opt("--jobs", 4))
    chains = [(m, w) for m in methods for w in worlds]
    with ThreadPoolExecutor(jobs) as ex:
        list(ex.map(lambda mw: chain(out, mw[0], mw[1], ns, budget, probe, opt("--arm", "eq")), chains))


if __name__ == "__main__":
    main()
