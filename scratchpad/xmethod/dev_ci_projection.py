"""Full CI-test DEV projection from the two CI pilots (CPU arms on Kaggle CPU, cmi_knn on Kaggle T4).

    uv run python scratchpad/xmethod/dev_ci_projection.py

Inputs: results/dev/pilot_ci_{cpu,gpu}/agg.json, specs/dev/full_ci.json. Cost per (arm, world, regime, n) =
campaign._fit_cost (measured pilot mean, else a power law in n over the arm's measured n in that cell, else over
the arm). pcorr_{eq,native,eq_min} errored in the pilot (bundle lacked configs/), so pcorr_hac's costs stand in.
Two readings: 'capped' = campaign.project (every unit run, cost capped at the budget); 'registry' = the DEV
driver's shared registry (T3): an arm whose projected cell-mean cost exceeds its budget at n in any (world,
regime) is infeasible at n' >= n and those units are not run. Output: results/dev/pilot_ci_projection.json.
``--table SPEC OUT``: instead write the flat (arm|world|regime|n) -> estimated cost table of SPEC's cells (same
estimates) for ``campaign --cost-table`` (balanced parts).
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from cdd_oran.xmethod import campaign as C  # noqa: E402

X = os.path.join(ROOT, "scratchpad", "xmethod")
PROXY = {"pcorr_eq": "pcorr_hac", "pcorr_native": "pcorr_hac", "pcorr_eq_min": "pcorr_hac_eq_min"}


def cost_table(full: dict) -> dict:
    gpu = {a for a, d in full["arms"].items() if d.get("budget_wall_s")}
    agg = {"cost": {}}
    for p in ("pilot_ci_cpu", "pilot_ci_gpu"):
        agg["cost"].update(json.load(open(os.path.join(X, "results", "dev", p, "agg.json"), encoding="utf-8"))["cost"])
    T = C.cost_table_from_agg(agg, gpu)
    for a, src in PROXY.items():
        T.update({a + k[len(src):]: v for k, v in T.items() if k.startswith(src + "|") and a + k[len(src):] not in T})
    return T


def main() -> int:
    if sys.argv[1:2] == ["--table"]:
        spec = json.load(open(sys.argv[2], encoding="utf-8"))
        T = cost_table(spec)
        out = {f"{u.arm}|{u.world}|{u.regime}|n{u.n}": C._fit_cost(T, u.arm, u.world, u.regime, u.n)[0]
               for u in C.expand(spec)}
        json.dump(out, open(sys.argv[3], "w", encoding="utf-8", newline="\n"), indent=0)
        print(f"{len(out)} cells -> {sys.argv[3]}")
        return 0
    full = json.load(open(os.path.join(X, "specs", "dev", "full_ci.json"), encoding="utf-8"))
    gpu = {a for a, d in full["arms"].items() if d.get("budget_wall_s")}
    T = cost_table(full)
    units = C.expand(full)
    cell = defaultdict(lambda: [0, 0.0, set()])          # (arm, n, world|regime) -> [units, cost each, bases]
    for u in units:
        c, how = C._fit_cost(T, u.arm, u.world, u.regime, u.n)
        e = cell[(u.arm, u.n, f"{u.world}|{u.regime}")]
        e[0] += 1
        e[1] = c
        e[2].add(how.split(" ")[0])
    out = {"arms": {}, "budget_s": 7200}
    for arm in sorted(full["arms"]):
        b = C.arm_budgets(full, arm, full.get("budget_cpu_s"))
        cap = b[1] if arm in gpu else b[0]
        ns = sorted({n for (a, n, _) in cell if a == arm})
        over = {n: sorted(w for (a, nn, w), e in cell.items() if a == arm and nn == n and e[1] > cap) for n in ns}
        n_inf = next((n for n in ns if over[n]), None)
        r = {"kind": "GPU wall-h (T4)" if arm in gpu else "CPU-h (Kaggle core)", "budget_s": cap,
             "infeasible_from_n": n_inf, "over_budget_cells": {str(n): v for n, v in over.items() if v},
             "by_n": {}}
        for n in ns:
            es = [e for (a, nn, _), e in cell.items() if a == arm and nn == n]
            r["by_n"][str(n)] = {"units": sum(e[0] for e in es),
                                 "capped_h": sum(e[0] * min(e[1], cap) for e in es) / 3600,
                                 "registry_h": (0.0 if n_inf is not None and n >= n_inf
                                                else sum(e[0] * e[1] for e in es) / 3600),
                                 "max_cell_s": max(e[1] for e in es),
                                 "basis": sorted(set().union(*(e[2] for e in es)))}
        r["capped_h"] = sum(v["capped_h"] for v in r["by_n"].values())
        r["registry_h"] = sum(v["registry_h"] for v in r["by_n"].values())
        out["arms"][arm] = r
    for kind in ("CPU", "GPU"):
        arms = [a for a in out["arms"] if (a in gpu) == (kind == "GPU")]
        out[f"{kind}_total"] = {"capped_h": sum(out["arms"][a]["capped_h"] for a in arms),
                                "registry_h": sum(out["arms"][a]["registry_h"] for a in arms),
                                "registry_units_run": sum(v["units"] for a in arms for n, v in
                                                          out["arms"][a]["by_n"].items() if v["registry_h"] > 0)}
    out["n_units"] = len(units)
    out["proxy_costs"] = PROXY
    json.dump(out, open(os.path.join(X, "results", "dev", "pilot_ci_projection.json"), "w", encoding="utf-8",
                        newline="\n"), indent=1)
    for a, r in out["arms"].items():
        print(f"{a:20s} inf_from {str(r['infeasible_from_n']):6s} capped {r['capped_h']:8.1f} registry "
              f"{r['registry_h']:8.1f}  " + " ".join(f"n{n}:{v['registry_h']:.1f}" for n, v in r["by_n"].items()))
    print({k: v for k, v in out.items() if k.endswith("_total")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
