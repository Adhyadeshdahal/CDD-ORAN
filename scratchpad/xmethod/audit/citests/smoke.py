"""audit-citests, check 5: validity smoke (descriptive). kappa .25, n 1000, DEV seeds 3_000_000-004; cells E2 R2, E1 R2,
E4 R1, E4 R3 (lambda 1); five tests x arms eq / native. Truth is used only by the auditor, after the run, to label
edges (and, for pdcor / cmi_knn, to skip TRUE edges to save compute: candidates = primary nulls + placebos).
pcorr / rcot2 / mscr: full candidate set, default config. pdcor: default config (proj_perm) with B capped at 999 (BC h 20; same smoke deviation as cmi_knn).
cmi_knn: torch backend on the local GPU (== CPU, gpu_check.json), sig_samples capped at 999 (BC h 20) for cost
(smoke deviation: a raw p <= .05 call needs no finer resolution).
Run: PYTHONPATH=. uv run --group baselines --group citests python scratchpad/xmethod/audit/citests/smoke.py
     cpu METHODS(comma) JOBS | gpu | e4n500      (env SMOKE_OUT = output jsonl)
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import os

OUT = os.environ.get("SMOKE_OUT", "scratchpad/xmethod/audit/citests/smoke_runs.jsonl")
SEEDS = range(3_000_000, 3_000_005)
CELLS = (("E2", "R2"), ("E1", "R2"), ("E4", "R1"), ("E4", "R3"))


def run(task):
    w, r, seed, method, arm, n, extra = task
    from cdd_oran.xmethod.methods.cmi_knn import CMIKnnMethod
    from cdd_oran.xmethod.methods.mscr import MSCRMethod
    from cdd_oran.xmethod.methods.pcorr import PCorrMethod
    from cdd_oran.xmethod.methods.pdcor import PDCorMethod
    from cdd_oran.xmethod.methods.rcot2 import RCoT2Method
    from cdd_oran.xmethod.worlds.generate import generate_dataset

    m = {"mscr": MSCRMethod, "pcorr": PCorrMethod, "pdcor": PDCorMethod, "rcot2": RCoT2Method,
         "cmi_knn": CMIKnnMethod}[method]()
    d, t = generate_dataset(w, r, n, seed, lam=1.0, kappa=0.25)
    if method in ("pdcor", "cmi_knn"):
        d = dataclasses.replace(d, candidates=tuple(c for c in d.candidates
                                                    if c[0] in d.action_names and c not in t.edges))
    cfg = {**m.default_config(), "arm": arm, **extra}
    if method == "mscr":
        cfg["n_jobs"] = 1
    t0 = time.time()
    res = m.run(d, cfg)

    def role(s, tg):
        if s == "P_placebo":
            return "placebo"
        if s == "P_placebo_conf":
            return "placebo_conf"
        if (s, tg) in t.edges:
            return "true"
        if s in d.action_names and (s, tg) in t.null_edges:
            return "null"
        return "kpi_null" if (s, tg) in t.null_edges else "other"

    mc = res.notes.get("mc_draws") or [None] * len(res.edges)
    rec = {"cell": [w, r], "n": n, "seed": seed, "method": method, "arm": arm, "extra": extra, "cpu_s": res.cpu_s,
           "wall_s": time.time() - t0, "stop": res.notes.get("stop"), "eq_dropped": res.notes.get("eq_dropped"),
           "not_testable": res.notes.get("not_testable_edges"),
           "edges": [[e.source, e.target, role(e.source, e.target), e.score, e.p, e.sign, e.declared, k]
                     for e, k in zip(res.edges, mc, strict=True)]}
    with open(OUT, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    return f"{w}{r} {seed} {method} {arm} {time.time() - t0:.0f}s"


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "cpu":
        cap = {"pdcor": {"b_perm": 999}}
        tasks = [(w, r, s, m, a, 1000, cap.get(m, {})) for m in sys.argv[2].split(",") for (w, r) in CELLS
                 for s in SEEDS for a in ("eq", "native")]
        jobs = int(sys.argv[3])
        if jobs == 1:
            for t in tasks:
                print(run(t), flush=True)
        else:
            with ProcessPoolExecutor(jobs) as ex:
                for msg in ex.map(run, tasks):
                    print(msg, flush=True)
    elif mode == "gpu":
        g = {"backend": "torch", "device": "cuda", "sig_samples": 999, "workers": 1}
        for (w, r) in (("E4", "R1"), ("E4", "R3"), ("E1", "R2"), ("E2", "R2")):
            for s in SEEDS:
                for a in ("eq", "native"):
                    print(run((w, r, s, "cmi_knn", a, 1000, g)), flush=True)
    elif mode == "e4more":   # follow-up: E4 R1 placebo rate of cmi_knn on 40 more DEV seeds, n 500 and 1000
        g = {"backend": "torch", "device": "cuda", "workers": 1}
        for n in (1000, 500):
            for s in range(3_000_005, 3_000_045):
                for a in ("eq", "native"):
                    print(run(("E4", "R1", s, "cmi_knn", a, n, g)), flush=True)
    elif mode == "e4n500":   # the author's E4 R1 n 500 observation, full R-9 resolution (B 9999, h 20)
        g = {"backend": "torch", "device": "cuda", "workers": 1}
        for s in SEEDS:
            for a in ("eq", "native"):
                print(run(("E4", "R1", s, "cmi_knn", a, 500, g)), flush=True)
