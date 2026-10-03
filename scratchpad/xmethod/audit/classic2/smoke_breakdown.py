"""audit-classic2: breakdown of the smoke (smoke_runs.jsonl). For score-only methods: leave-one-seed-out tau (2nd-largest
placebo score of the other 4 DEV datasets) applied to the held-out one -> held-out placebo vs truth-null declaration
rates (exchangeability check), and which null edges are declared (in-sample tau). Truth used for REPORTING only.
Run: PYTHONPATH=. uv run python scratchpad/xmethod/audit/classic2/smoke_breakdown.py"""
import collections
import json
import math
import os

from cdd_oran.xmethod.worlds.generate import truth_for

HERE = os.path.dirname(os.path.abspath(__file__))
runs = [json.loads(x) for x in open(os.path.join(HERE, "smoke_runs.jsonl"))]
groups = collections.defaultdict(list)
for r in runs:
    groups[("".join(r["cell"]), r["method"], r["arm"])].append(r)


def tau_of(rs):
    s = sorted((e[2] for r in rs for e in r["edges"] if e[0] == "P_placebo" and e[2] is not None
                and not math.isnan(e[2])), reverse=True)
    return s[1] if len(s) > 1 else -math.inf


out = {}
for (cell, m, arm), rs in groups.items():
    if m == "granger":
        continue
    tr = truth_for(cell[:2], cell[2:])
    nulls = {f"{a}->{b}" for a, b in tr.null_edges if a.startswith("P") and a != "P_placebo"}
    loo = {"pl": 0, "n_pl": 0, "null": 0, "n_null": 0}
    for i, r in enumerate(rs):
        t = tau_of(rs[:i] + rs[i + 1:])
        for e in r["edges"]:
            k = f"{e[0]}->{e[1]}"
            dec = e[2] is not None and not math.isnan(e[2]) and e[2] > t
            if e[0] == "P_placebo":
                loo["n_pl"] += 1
                loo["pl"] += dec
            elif k in nulls:
                loo["n_null"] += 1
                loo["null"] += dec
    t_in = tau_of(rs)
    cnt = collections.Counter(f"{e[0]}->{e[1]}" for r in rs for e in r["edges"]
                              if f"{e[0]}->{e[1]}" in nulls and e[2] is not None and not math.isnan(e[2])
                              and e[2] > t_in)
    out[f"{cell}:{m}:{arm}"] = {"loo": loo, "declared_nulls_insample": dict(cnt.most_common(8))}
print(json.dumps(out, indent=1))
