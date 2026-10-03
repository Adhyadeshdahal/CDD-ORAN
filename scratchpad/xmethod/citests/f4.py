"""F4 synthetic gate: null false-positive rate at nominal on null data, and recovery of planted edges.

  python scratchpad/xmethod/citests/f4.py run OUT.jsonl METHOD KIND N REPS [--jobs J] [--primary-only] [--cfg JSON]
  python scratchpad/xmethod/citests/f4.py summary OUT.jsonl [--out F4.json]

Resumable (R-26: one rep = one shard): ``run`` skips the reps already in OUT.jsonl for the same (method, kind, n,
primary_only, arm), so a reclaimed session is relaunched with the same command.

Data: ``_citests_synth`` (R1-like: i.i.d. uniform actions P0..P3 + P_placebo, AR(1) KPIs). KIND null: no action
effect (true edges = K_j -> K_j self lags only); planted: P0->K0 (+), P1->K1 (-), P2,P3->K2 (gated), K3->K3.
Seeds: DEV block 3_000_000 + rep (synthetic generator stream tagged 7802/99, distinct from the harness).
Per dataset: default (primary) config of the adapter. Recorded: raw p and BY declarations per edge.
Summary per (method, kind, n): per-test rejection rate of null edges at raw p <= .05 (primary family and KPI->KPI
separately) with Wilson 95% CI and a dataset-cluster bootstrap 95% CI; declared-null count (BY families) and
FDP; recall per planted edge; sign accuracy on declared signed edges.
"""
from __future__ import annotations

import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SEED0 = 3_000_000


def one(args):
    method, kind, n, rep, primary_only, cfg = args
    import dataclasses

    from cdd_oran.xmethod.methods import _citests_synth as synth
    from cdd_oran.xmethod.methods.cmi_knn import CMIKnnMethod
    from cdd_oran.xmethod.methods.mscr import MSCRMethod
    from cdd_oran.xmethod.methods.pcorr import PCorrMethod
    from cdd_oran.xmethod.methods.pdcor import PDCorMethod
    from cdd_oran.xmethod.methods.rcot2 import RCoT2Method

    m = {"mscr": MSCRMethod, "pcorr": PCorrMethod, "pdcor": PDCorMethod, "rcot2": RCoT2Method,
         "cmi_knn": CMIKnnMethod}[method]()
    ds, truth = synth.make(kind, n, SEED0 + rep)
    if primary_only:
        ds = dataclasses.replace(ds, candidates=tuple(c for c in ds.candidates if c[0] in ds.action_names))
    c = {**m.default_config(), **cfg}
    if method == "mscr":
        c["n_jobs"] = 1
    if method == "cmi_knn":
        c["workers"] = 1
    r = m.run(ds, c)
    return {"method": method, "kind": kind, "n": n, "rep": rep, "cpu_s": r.cpu_s, "arm": r.notes.get("arm"),
            "primary_only": primary_only,
            "config": {k: v for k, v in c.items()},
            "stop": r.notes.get("stop"),
            "edges": [[e.source, e.target, e.p, e.sign, e.declared, (e.source, e.target) in truth.edges,
                       truth.signs.get((e.source, e.target), 0)] for e in r.edges]}


def wilson(k, n):
    if n == 0:
        return [None, None, None]
    p = k / n; z = 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(p, 4), round(c - h, 4), round(c + h, 4)]


def summary(path):
    recs = [json.loads(line) for line in open(path)]
    out = {}
    for key in sorted({(r["method"], r["kind"], r["n"]) for r in recs}):
        rs = [r for r in recs if (r["method"], r["kind"], r["n"]) == key]
        res = {"reps": len(rs), "cpu_s_mean": float(np.mean([r["cpu_s"] for r in rs]))}
        for fam, sel in (("primary", lambda s: s.startswith("P")), ("kpi_kpi", lambda s: s.startswith("K"))):
            per = [[e for e in r["edges"] if sel(e[0]) and not e[5] and e[2] is not None] for r in rs]
            k = sum(sum(e[2] <= .05 for e in es) for es in per)
            tot = sum(len(es) for es in per)
            if not tot:
                continue
            rate = np.array([np.mean([e[2] <= .05 for e in es]) for es in per if es])
            boot = np.random.default_rng(0).choice(rate, size=(2000, len(rate))).mean(1)
            decl = [sum(e[4] for e in es) for es in per]
            tru = [[e for e in r["edges"] if sel(e[0]) and e[5]] for r in rs]
            fdp = [d / max(d + sum(e[4] for e in t), 1) for d, t in zip(decl, tru, strict=True)]
            res[fam] = {"null_tests": tot, "rej_rate_raw05": wilson(k, tot),
                        "cluster_boot95": [round(float(np.quantile(boot, .025)), 4), round(float(np.quantile(boot, .975)), 4)],
                        "null_declared_total": int(sum(decl)), "mean_fdp": round(float(np.mean(fdp)), 4)}
        rec = {}
        for r in rs:
            for e in r["edges"]:
                if e[5]:
                    d = rec.setdefault(f"{e[0]}->{e[1]}", [0, 0, 0, 0])
                    d[0] += int(e[4]); d[1] += 1
                    if e[4] and e[6]:
                        d[2] += int(np.sign(e[3]) == e[6]); d[3] += 1
        res["recall"] = {k: {"declared": v[0], "of": v[1], "sign_correct": v[2], "sign_n": v[3]} for k, v in rec.items()}
        res["placebo_declared"] = int(sum(e[4] for r in rs for e in r["edges"] if e[0] == "P_placebo"))
        out["|".join(map(str, key))] = res
    return out


def main():
    a = sys.argv[1:]
    if a[0] == "summary":
        res = summary(a[1])
        print(json.dumps(res, indent=1))
        if "--out" in a:
            json.dump(res, open(a[a.index("--out") + 1], "w"), indent=1)
        return
    _, out, method, kind, n, reps = a[:6]
    jobs = int(a[a.index("--jobs") + 1]) if "--jobs" in a else 4
    cfg = json.loads(a[a.index("--cfg") + 1]) if "--cfg" in a else {}
    po = "--primary-only" in a
    arm = cfg.get("arm", "eq")
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            if (r["method"], r["kind"], r["n"], r.get("primary_only"), r.get("arm")) == (method, kind, int(n), po, arm):
                done.add(r["rep"])
    tasks = [(method, kind, int(n), rep, po, cfg) for rep in range(int(reps)) if rep not in done]
    print(f"{method} {kind} n {n}: {len(done)} reps done, {len(tasks)} to run", flush=True)
    with ProcessPoolExecutor(jobs) as ex:
        for rec in ex.map(one, tasks):
            with open(out, "a") as f:
                f.write(json.dumps(rec, default=float) + "\n")
            print(method, kind, n, rec["rep"], f"cpu {rec['cpu_s']:.1f}", flush=True)


if __name__ == "__main__":
    main()
