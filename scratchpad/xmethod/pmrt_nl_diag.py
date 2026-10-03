"""R-42 step 1: per-true-edge recall of pmrt_eq vs shap_dag / two_tower / pc_eq on DEV E2 / E5 (R1, R2, n 1000,
kappa .25, measurement seeds only). Reads the DEV merged records read-only (branch xm/dev-runs).

    uv run python scratchpad/xmethod/pmrt_nl_diag.py [MERGED] > scratchpad/xmethod/results/pmrt_nl/diag_table.json
"""
from __future__ import annotations

import collections
import gzip
import json
import math
import sys

import numpy as np

from cdd_oran.xmethod.worlds.generate import truth_for

MERGED = (sys.argv[1] if len(sys.argv) > 1 else
          "D:/academia/major-project/CDD-ORAN-wt/xm-pmrt/scratchpad/xmethod/results/dev/full/merged.jsonl.gz")
ARMS = ("pmrt_eq", "pmrt_r3", "shap_dag", "two_tower", "pc_eq", "corr")


def main() -> None:
    recs = collections.defaultdict(lambda: {"tune": [], "measure": []})
    for line in gzip.open(MERGED, "rt"):
        r = json.loads(line)
        j = r["job"]
        if (j["world"] in ("E2", "E5") and j["n"] == 1000 and j["kappa"] == 0.25 and r["arm"] in ARMS
                and r["status"] == "ok"):
            recs[(r["arm"], j["world"], j["regime"])][r["role"]].append(r)
    tau = {}
    for key, g in recs.items():       # R-29 conformal placebo cutoff from the cell's tune seeds (score.placebo_tau)
        s = sorted(e["score"] for r in g["tune"] for e in r["edges"]
                   if e["source"] == "P_placebo" and e["score"] is not None and np.isfinite(e["score"]))
        tau[key] = s[min(math.ceil((len(s) + 1) * 0.95 - 1e-9), len(s)) - 1] if s else math.inf
    hit = collections.defaultdict(list)       # (world, regime, edge, arm, rule) -> [0/1 per seed]
    pz = collections.defaultdict(list)        # (world, regime, edge) -> pmrt_eq z per seed
    nullp = collections.defaultdict(list)     # (world, regime) -> pmrt_eq truth-null raw p <= .05
    for r in (r for g in recs.values() for r in g["measure"]):
        j = r["job"]
        w, g = j["world"], j["regime"]
        t = tau[(r["arm"], w, g)]
        tr = truth_for(w, g)
        for e in r["edges"]:
            ed = (e["source"], e["target"])
            if ed in tr.edges and e["source"].startswith("P"):
                key = (w, g, f"{ed[0]}->{ed[1]}", r["arm"])
                hit[key + ("decl",)].append(int(bool(e["declared"])))
                hit[key + ("tau",)].append(int(e["score"] is not None and np.isfinite(e["score"]) and e["score"] > t))
                if e["p"] is not None:
                    hit[key + ("raw05",)].append(int(e["p"] <= 0.05))
                if r["arm"] == "pmrt_eq":
                    pz[(w, g, f"{ed[0]}->{ed[1]}")].append(r["notes"]["z"].get(f"{ed[0]}->{ed[1]}", np.nan))
            elif r["arm"] == "pmrt_eq" and ed in tr.null_edges and e["source"].startswith("P") and e["p"] is not None:
                nullp[(w, g)].append(int(e["p"] <= 0.05))
    out = {}
    for (w, g, ed, arm, rule), v in sorted(hit.items()):
        out.setdefault(f"{w}|{g}|{ed}", {})[f"{arm}:{rule}"] = [round(float(np.mean(v)), 3), len(v)]
    for (w, g, ed), z in pz.items():
        z = np.asarray(z, float)
        out[f"{w}|{g}|{ed}"]["pmrt_eq:z_med_absmed_signfrac+"] = [round(float(np.median(z)), 2),
                                                                  round(float(np.median(np.abs(z))), 2),
                                                                  round(float(np.mean(z > 0)), 2)]
    out["_null_raw05_pmrt_eq"] = {f"{w}|{g}": [round(float(np.mean(v)), 3), len(v)] for (w, g), v in nullp.items()}
    out["_tau"] = {"|".join(k): v for k, v in sorted(tau.items())}
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
