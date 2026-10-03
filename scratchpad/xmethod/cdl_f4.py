"""cdl F4 (synthetic planted edges / null level), sharded and resumable (one cdl fit per unit, any platform).

Same design as ``scripts/xm_classic_fidelity.py f4`` (FIDELITY_CLASSIC F4): generator ``_classic_synth`` R1, planted
A0 -> Y0 +, A1 -> Y1 -, K0 -> Y0 +, K1 -> Y2 + at b (alt), global null b = 0; tau = R-29 conformal placebo cutoff from
10 SEPARATE tune datasets (seeds 3_000_000..09) of the same kind, applied to ``reps`` test datasets (3_000_100..).
Units = (n, kind alt | null, role tune | test, seed); ``run --part i/k`` does units[i::k], skipping units already in
``--out``; ``summarize`` merges any number of jsonl files (duplicates / missing units tolerated, reported).

  python scratchpad/xmethod/cdl_f4.py run --ns 500,4000 --reps 20 --part 0/4 --out out/f4_0.jsonl
  python scratchpad/xmethod/cdl_f4.py summarize out/f4_*.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cdd_oran.xmethod.methods import _classic_synth as SYN  # noqa: E402

N_TUNE = 10
B_ALT = 1.0


def units(ns, reps):
    out = []
    for n in ns:
        for kind in ("alt", "null"):
            out += [(n, kind, "tune", 3_000_000 + i) for i in range(N_TUNE)]
            out += [(n, kind, "test", 3_000_100 + i) for i in range(reps)]
    return out


def key(u):
    return f"{u[0]}|{u[1]}|{u[2]}|{u[3]}"


def run(ns, reps, part, out, config):
    from cdd_oran.xmethod.methods.classic import METHODS
    i, k = (int(x) for x in part.split("/"))
    done = set()
    if os.path.exists(out):
        done = {json.loads(line)["key"] for line in open(out) if line.startswith("{")}
    m = METHODS["cdl"]()
    for u in units(ns, reps)[i::k]:
        if key(u) in done:
            continue
        n, kind, role, seed = u
        d, _ = SYN.make(n, seed, b=B_ALT if kind == "alt" else 0.0)
        t0 = time.time()
        r = m.run(d, dict(config))
        import torch
        rec = {"key": key(u), "n": n, "kind": kind, "role": role, "seed": seed, "cpu_s": round(r.cpu_s, 1),
               "wall_s": round(time.time() - t0, 1), "torch": torch.__version__, "numpy": np.__version__,
               "config": config, "edges": [[e.source, e.target, e.score, e.sign] for e in r.edges],
               "final_loss": r.notes["fit"]["final_loss"]}
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({k_: rec[k_] for k_ in ("key", "cpu_s", "wall_s")}), flush=True)


def _wilson(x, n, z=1.96):
    if n == 0:
        return [float("nan")] * 2
    p = x / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 4), round(c + h, 4)]


def conformal_tau(scores, alpha=0.05):
    s = np.sort(np.asarray([x for x in scores if np.isfinite(x)]))
    if not len(s):
        return float("inf")
    idx = math.ceil((len(s) + 1) * (1 - alpha))
    return float(s[min(idx, len(s)) - 1])


def summarize(paths):
    recs = {}
    dup = 0
    for p in paths:
        for line in open(p):
            if line.startswith("{"):
                r = json.loads(line)
                dup += r["key"] in recs
                recs[r["key"]] = r
    out = []
    for n in sorted({r["n"] for r in recs.values()}):
        row = {"n": n, "duplicates": dup}
        taus = {}
        for kind in ("alt", "null"):
            tune = [r for r in recs.values() if r["n"] == n and r["kind"] == kind and r["role"] == "tune"]
            taus[kind] = conformal_tau([e[2] for r in tune for e in r["edges"] if e[0] == "P_placebo"])
            row[f"n_tune_{kind}"] = len(tune)
        row["tau_alt"], row["tau_null"] = taus["alt"], taus["null"]
        alt = [r for r in recs.values() if r["n"] == n and r["kind"] == "alt" and r["role"] == "test"]
        nul = [r for r in recs.values() if r["n"] == n and r["kind"] == "null" and r["role"] == "test"]
        row["reps_alt"], row["reps_null"] = len(alt), len(nul)
        for thr_name, thr_alt, thr_null, ge in (("tau", taus["alt"], taus["null"], False), ("fixed_.16", .16, .16, True)):
            def dec(s, t, ge=ge):
                return bool(np.isfinite(s) and (s >= t if ge else s > t))
            pw, sg = {}, {}
            fp = nn = 0
            for (a, t), sign in SYN.PLANTED.items():
                hits = [e for r in alt for e in r["edges"] if (e[0], e[1]) == (a, t) and dec(e[2], thr_alt)]
                pw[f"{a}->{t}"] = round(len(hits) / max(len(alt), 1), 3)
                sg[f"{a}->{t}"] = round(np.mean([e[3] == sign for e in hits]), 3) if hits else None
            for r in alt:
                for e in r["edges"]:
                    if (e[0], e[1]) not in SYN.PLANTED:
                        nn += 1
                        fp += dec(e[2], thr_alt)
            fp0 = sum(dec(e[2], thr_null) for r in nul for e in r["edges"])
            n0 = sum(len(r["edges"]) for r in nul)
            fw = sum(any(dec(e[2], thr_null) for e in r["edges"]) for r in nul)
            row[thr_name] = {"power": pw, "sign_correct": sg, "null_edge_fpr_alt": round(fp / max(nn, 1), 4),
                             "null_edge_fpr_alt_ci": _wilson(fp, nn), "global_null_fpr": round(fp0 / max(n0, 1), 4),
                             "global_null_fpr_ci": _wilson(fp0, n0), "global_null_fwer": round(fw / max(len(nul), 1), 3)}
        # overfitting diagnostic: mean CMI of null candidates by source kind, global-null test data
        for fam, pred in (("action", lambda s: s.startswith(("A", "P"))), ("kpi", lambda s: s.startswith("Y"))):
            v = [e[2] for r in nul for e in r["edges"] if pred(e[0])]
            row[f"null_cmi_mean_{fam}"] = round(float(np.mean(v)), 4) if v else None
        pl = [e[2] for r in alt for e in r["edges"] if (e[0], e[1]) in SYN.PLANTED]
        row["planted_cmi_mean"] = round(float(np.mean(pl)), 4) if pl else None
        row["cpu_s_per_fit_median"] = float(np.median([r["cpu_s"] for r in recs.values() if r["n"] == n]))
        out.append(row)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "summarize"))
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--ns", default="500,4000")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--out", default="f4.jsonl")
    ap.add_argument("--config", default="{}")
    a = ap.parse_args(argv)
    if a.what == "run":
        run([int(x) for x in a.ns.split(",")], a.reps, a.part, a.out, json.loads(a.config))
    else:
        paths = [p for g in a.paths for p in glob.glob(g)]
        for row in summarize(paths):
            print(json.dumps(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
