"""audit-classic2, check 6: quick validity smoke at kappa .25 (R-27), n 1000, DEV seeds 3_000_000-004, E2 R2 and E1 R2
(+ E3 R2 for granger, its study world). Descriptive, not a gate. Per (method, arm, cell): truth-NULL primary edges
declared and P_placebo declared, summed over the 5 seeds; recall for context. Score-only methods: tau = 2nd-largest
P_placebo score pooled over the 5 datasets (R-2), so the placebo count is <= 1 by construction and the null count is the
informative one. granger: pooled BY per family as returned (q .05) + raw p <= .05 counts. pc: native (alpha .05) too.
Raw per-run records -> smoke_runs.jsonl. Run: PYTHONPATH=. uv run --group baselines python scratchpad/xmethod/audit/classic2/smoke.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from cdd_oran.xmethod.methods.classic import METHODS
from cdd_oran.xmethod.score import apply_threshold, placebo_tau
from cdd_oran.xmethod.worlds.generate import generate_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = range(3_000_000, 3_000_005)
KAP, N = 0.25, 1000
JOBS = [("pc", "eq"), ("pc", "native"), ("granger", "eq"), ("granger", "native"), ("shap_dag", "native"),
        ("two_tower", "native")]
CELLS = [("E2", "R2"), ("E1", "R2"), ("E3", "R2")]
PL = "P_placebo"

data = {(c, s): generate_dataset(*c, N, s, kappa=KAP) for c in CELLS for s in SEEDS}
summary = []
with open(os.path.join(HERE, "smoke_runs.jsonl"), "w") as fh:
    for cell in CELLS:
        for mname, arm in JOBS:
            if cell == ("E3", "R2") and mname != "granger":
                continue
            res = []
            for s in SEEDS:
                d, tr = data[(cell, s)]
                r = METHODS[mname]().run(d, {"arm": arm})
                res.append((r, d, tr))
                fh.write(json.dumps({"cell": cell, "seed": s, "method": mname, "arm": arm, "cpu_s": r.cpu_s,
                                     "notes": {k: r.notes.get(k) for k in ("n_not_testable", "fisherz_pinv_fallback_rate",
                                                                           "arm", "tau")},
                                     "edges": [[e.source, e.target, e.score, e.p, e.sign, e.declared]
                                               for e in r.edges]}, default=float) + "\n")
            uses_p = mname == "granger"
            tau = None if uses_p else placebo_tau([r for r, _, _ in res])
            agg = {"null_decl": 0, "n_null": 0, "pl_decl": 0, "n_pl": 0, "tp": 0, "n_true": 0,
                   "null_rawp05": 0, "pl_rawp05": 0, "null_native": 0, "pl_native": 0, "nt": 0, "sign_ok": 0,
                   "sign_n": 0}
            for r, d, tr in res:
                rr = r if uses_p else apply_threshold(r, tau)
                nat = set(r.notes.get("native_declared") or [])
                nt = set(r.notes.get("not_testable_edges") or [])
                prim = set(map(tuple, d.meta["primary_candidates"]))
                for e in rr.edges:
                    key = (e.source, e.target)
                    if key not in prim:
                        continue
                    dec = e.declared and f"{e.source}->{e.target}" not in nt
                    agg["nt"] += f"{e.source}->{e.target}" in nt
                    if e.source == PL:
                        agg["n_pl"] += 1
                        agg["pl_decl"] += dec
                        agg["pl_rawp05"] += e.p is not None and e.p <= 0.05
                        agg["pl_native"] += f"{e.source}->{e.target}" in nat
                    elif key in tr.null_edges:
                        agg["n_null"] += 1
                        agg["null_decl"] += dec
                        agg["null_rawp05"] += e.p is not None and e.p <= 0.05
                        agg["null_native"] += f"{e.source}->{e.target}" in nat
                    elif key in tr.edges:
                        agg["n_true"] += 1
                        agg["tp"] += dec
                        if dec and key in tr.signs and e.sign != 0:
                            agg["sign_n"] += 1
                            agg["sign_ok"] += int(np.sign(e.sign) == tr.signs[key])
            row = {"cell": "".join(cell), "method": mname, "arm": arm, "tau": tau,
                   "cpu_s_mean": float(np.mean([r.cpu_s for r, _, _ in res])), **agg}
            summary.append(row)
            print(json.dumps(row, default=float), file=sys.stderr, flush=True)
json.dump(summary, sys.stdout, indent=0, default=float)
