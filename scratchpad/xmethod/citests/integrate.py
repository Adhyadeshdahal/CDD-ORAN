"""End-to-end integration of the citests adapters on real harness datasets (orchestrator step after the harness merge).

  python scratchpad/xmethod/citests/integrate.py all OUT.jsonl [--n 1000] [--methods a,b] [--jobs 4]
                     [--timeout SEC] [--native METHODS] [--arm eq|native] --kappa K
  python scratchpad/xmethod/citests/integrate.py task OUT.jsonl WORLD REGIME METHOD --kappa K [--n 1000] [--native]
                     [--arm A]

--kappa (REQUIRED; R-24 / R-27 observation noise: .25 primary, {.125, .5} sweep, 0 = noiseless sanity only).

--arm (R-17 / R-18) sets the conditioning arm of the default config (default: the adapters' primary, eq);
--native METHODS uses those adapters' native_config() (native B AND native arm, the fidelity-gate config).
Rows written before the arms existed (run xm-citests-int-3) are arm native.

For every world / regime (worlds.REGIMES_OF; E4 at lambda 1.0, plus lambda 1.5 for R3 / R4), one DEV seed
(3_000_000), n = 1000: generate_dataset -> Method.run (default config = R-9 B 9999 Besag-Clifford, or native B for
the methods in --native) -> score.score; one JSON line per (cell, method) with scores, cpu_s, wall, peak RSS.
Each task runs in its own process (timeout -> a "timeout" record).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

SEED = 3_000_000
HERE = os.path.dirname(os.path.abspath(__file__))


def cells():
    from cdd_oran.xmethod.worlds import REGIMES_OF

    out = []
    for w, regs in REGIMES_OF.items():
        for r in regs:
            out.append((w, r, 1.0))
            if w == "E4" and r in ("R3", "R4"):
                out.append((w, r, 1.5))
    return out


def task(out, world, regime, method, n, native, lam, arm=None, kappa=None):
    sys.path.insert(0, HERE)
    from bench import peak_rss_mb

    from cdd_oran.xmethod import score as sc
    from cdd_oran.xmethod.methods.cmi_knn import CMIKnnMethod
    from cdd_oran.xmethod.methods.mscr import MSCRMethod
    from cdd_oran.xmethod.methods.pcorr import PCorrMethod
    from cdd_oran.xmethod.methods.pdcor import PDCorMethod
    from cdd_oran.xmethod.methods.rcot2 import RCoT2Method
    from cdd_oran.xmethod.worlds import generate_dataset

    m = {"mscr": MSCRMethod, "pcorr": PCorrMethod, "pdcor": PDCorMethod, "rcot2": RCoT2Method,
         "cmi_knn": CMIKnnMethod}[method]()
    ds, truth = generate_dataset(world, regime, n, SEED, lam=lam, kappa=kappa)
    cfg = m.native_config() if native else m.default_config()
    if arm and not native:
        cfg["arm"] = arm
    if method == "mscr":
        cfg["n_jobs"] = 1
    if method == "cmi_knn":
        cfg["workers"] = 1
    t0 = time.time()
    r = m.run(ds, cfg)
    s = sc.score(r, truth)
    rec = {"world": world, "regime": regime, "lam": lam, "kappa": kappa, "n": n, "seed": SEED, "method": method,
           "native": native,
           "arm": r.notes.get("arm"), "n_rows_used": r.notes.get("n_rows_used"),
           "covariates_helper": r.notes.get("covariates_helper"),
           "n_not_testable_true": sum(tuple(c) in truth.edges for c in r.notes.get("not_testable_edges", [])),
           "n_not_testable_null": sum(tuple(c) not in truth.edges for c in r.notes.get("not_testable_edges", [])),
           "cpu_s": r.cpu_s, "wall_s": time.time() - t0, "peak_rss_mb": peak_rss_mb(), "stop": r.notes.get("stop"),
           "n_candidates": len(ds.candidates), "score": s, "sign_rule": r.notes.get("sign_rule"),
           "max_mc_draws": max([v for v in r.notes.get("mc_draws") or [] if v is not None] or [0]), "version": r.version,
           "edges": [[e.source, e.target, e.score, e.p, e.sign, e.declared] for e in r.edges]}
    with open(out, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    print(method, world, regime, lam, f"cpu {r.cpu_s:.1f}", "primary f1", round(s["f1"], 3), "declared",
          s["declared_edges"], flush=True)


def main():
    a = sys.argv[1:]
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d  # noqa: E731
    n = int(opt("--n", 1000))
    if "--kappa" not in a:
        raise SystemExit("--kappa is required (R-24 / R-27: .25 primary; 0 = noiseless sanity only)")
    kappa = float(opt("--kappa", None))
    if a[0] == "task":
        _, out, w, r, meth = a[:5]
        task(out, w, r, meth, n, "--native" in a, float(opt("--lam", 1.0)), opt("--arm", None), kappa)
        return
    out = a[1]
    methods = opt("--methods", "pcorr,mscr,rcot2,pdcor,cmi_knn").split(",")
    native = set(opt("--native", "").split(","))
    jobs, timeout = int(opt("--jobs", 4)), float(opt("--timeout", 36000))
    todo = [(w, r, lam, m) for m in methods for (w, r, lam) in cells()]

    def run(t):
        w, r, lam, m = t
        cmd = [sys.executable, os.path.abspath(__file__), "task", out, w, r, m, "--n", str(n), "--lam", str(lam),
               "--kappa", str(kappa)]
        if m in native:
            cmd.append("--native")
        if opt("--arm", None):
            cmd += ["--arm", opt("--arm", None)]
        t0 = time.time()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            print(p.stdout.strip()[-400:], p.stderr.strip()[-800:] if p.returncode else "", flush=True)
            if p.returncode:
                rec = {"world": w, "regime": r, "lam": lam, "n": n, "method": m, "error": p.stderr[-2000:]}
            else:
                return
        except subprocess.TimeoutExpired:
            rec = {"world": w, "regime": r, "lam": lam, "n": n, "method": m, "timeout_s": time.time() - t0}
            print("TIMEOUT", t, flush=True)
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")

    with ThreadPoolExecutor(jobs) as ex:
        list(ex.map(run, todo))


if __name__ == "__main__":
    main()
