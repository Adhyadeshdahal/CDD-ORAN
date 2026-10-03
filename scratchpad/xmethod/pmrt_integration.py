"""End-to-end integration of pmrt_core with the merged harness (feat/v2 138576e): generate_dataset -> PmrtCore.run
(production config: max B 9999, Besag-Clifford h 20) -> score.score, one DEV seed per (world, regime), n 1000.

  uv run python scratchpad/xmethod/pmrt_integration.py [--n 1000] [--seed 3000000] [--out F.json]
  uv run python scratchpad/xmethod/pmrt_integration.py --null-seeds 10 --out F.json   (null-edge p sanity, DEV seeds)

The method is loaded through the runner's registry path ("module:Class") to check that it plugs in as-is.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cdd_oran.xmethod import runner, score  # noqa: E402
from cdd_oran.xmethod.methods import pmrt_core as PC  # noqa: E402
from cdd_oran.xmethod.worlds import generate_dataset  # noqa: E402

CELLS = ([(w, r, 1.0) for w in ("E1", "E2", "E3", "E5") for r in ("R1", "R2")]
         + [("E4", r, 1.0) for r in ("R1", "R2")] + [("E4", r, lam) for r in ("R3", "R4") for lam in (1.0, 1.5)])
KEEP = ("n_declared", "tp", "fp", "precision", "recall", "f1", "fdp", "null_fpr", "sign_acc", "sign_n", "n_true",
        "n_null", "placebo_declared", "placebo_conf_declared")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=3_000_000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--null-seeds", type=int, default=0)
    a = ap.parse_args()
    if a.null_seeds:
        return null_sanity(a)
    method = runner.load_method("cdd_oran.xmethod.methods.pmrt_core:PmrtCore")
    assert isinstance(method, PC.PmrtCore)
    out = {"n": a.n, "seed": a.seed, "version": PC.PMRT_CORE_VERSION, "cells": {}}
    for w, r, lam in CELLS:
        t = time.time()
        ds, truth = generate_dataset(w, r, a.n, a.seed, lam=lam) if w == "E4" else generate_dataset(w, r, a.n, a.seed)
        res = method.run(ds, None)
        sc = score.score(res, truth, ds.candidates)
        na = res.notes[PC.NOT_APPLICABLE]
        prim = set(map(tuple, ds.meta["primary_candidates"]))
        rec = {k: sc.get(k) for k in KEEP}
        rec.update(n_candidates=len(ds.candidates), n_primary=len(prim),
                   primary_tested=sum(1 for e in res.edges if (e.source, e.target) in prim and e.p is not None),
                   not_applicable=len(na), na_reasons=sorted(set(na.values())),
                   declared=sc["declared_edges"], cpu_s=round(res.cpu_s, 2), wall_s=round(time.time() - t, 2),
                   z_true={f"{s}->{tt}": round(res.notes["z"].get(f"{s}->{tt}", float("nan")), 2)
                           for s, tt in sorted(truth.edges) if (s, tt) in prim})
        key = f"{w}|{r}" + (f"|lam{lam}" if w == "E4" else "")
        out["cells"][key] = rec
        print(key, json.dumps({k: rec[k] for k in ("n_true", "tp", "fp", "recall", "sign_acc", "placebo_declared",
                                                     "placebo_conf_declared", "primary_tested", "n_primary",
                                                     "cpu_s")}), flush=True)
    if a.out:
        with open(a.out, "w", newline="\n") as fh:
            json.dump(out, fh, indent=1, default=str)


def null_sanity(a):
    """Pooled raw-p rate at .05 / .01 over the truth's exact-null PRIMARY edges (incl. P_placebo), DEV seeds
    seed .. seed + null_seeds - 1, per (world, regime). Truth read only for scoring."""
    import numpy as np
    method = PC.PmrtCore()
    out = {"n": a.n, "seeds": [a.seed, a.seed + a.null_seeds - 1], "cells": {}}
    for w, r, lam in CELLS:
        ps, pl, decl_null, t = [], [], 0, time.time()
        for s in range(a.seed, a.seed + a.null_seeds):
            ds, truth = generate_dataset(w, r, a.n, s, lam=lam) if w == "E4" else generate_dataset(w, r, a.n, s)
            res = method.run(ds, None)
            prim = set(map(tuple, ds.meta["primary_candidates"]))
            for e in res.edges:
                if (e.source, e.target) in prim and (e.source, e.target) in truth.null_edges and e.p is not None:
                    ps.append(e.p)
                    decl_null += int(e.declared)
                    if e.source == "P_placebo":
                        pl.append(e.p)
        ps = np.array(ps)
        key = f"{w}|{r}" + (f"|lam{lam}" if w == "E4" else "")
        out["cells"][key] = {"n_null_p": len(ps), "rate05": float(np.mean(ps <= .05)) if len(ps) else None,
                             "rate01": float(np.mean(ps <= .01)) if len(ps) else None,
                             "placebo_rate05": float(np.mean(np.array(pl) <= .05)) if pl else None,
                             "null_declared": decl_null, "wall_s": round(time.time() - t, 1)}
        print(key, json.dumps(out["cells"][key]), flush=True)
    if a.out:
        with open(a.out, "w", newline="\n") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
