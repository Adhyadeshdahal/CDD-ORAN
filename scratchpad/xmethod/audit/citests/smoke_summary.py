"""Summarize smoke_runs*.jsonl: per (cell, method, arm) raw p <= .05 counts on truth-null primary edges (null) and
P_placebo (Pl), BY-declared counts, P_placebo_conf, true-edge recall (full-candidate methods only), cost.
Run: PYTHONPATH=. uv run --group baselines --group citests python scratchpad/xmethod/audit/citests/smoke_summary.py FILES...
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict


def wilson(k, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


agg = defaultdict(lambda: defaultdict(int))
seen = set()
for fn in sys.argv[1:]:
    for line in open(fn):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (tuple(r["cell"]), r["n"], r["method"], r["arm"])
        if (key, r["seed"]) in seen:
            continue
        seen.add((key, r["seed"]))
        a = agg[key]
        a["seeds"] += 1
        a["cpu"] = max(a["cpu"], r["cpu_s"])
        a["stop"] += r["stop"] is not None
        for s, t, role, sc, p, sg, dec, k in r["edges"]:
            if role not in ("null", "placebo", "placebo_conf", "true"):
                continue
            a[f"{role}_n"] += 1
            a[f"{role}_raw"] += p is not None and p <= 0.05
            a[f"{role}_by"] += bool(dec)
            a[f"{role}_cap"] += k is not None and k >= 999 and p is not None and p <= 0.05
out = []
for key in sorted(agg):
    a = agg[key]
    row = {"cell": "".join(key[0]), "n": key[1], "method": key[2], "arm": key[3], "seeds": a["seeds"],
           "max_cpu_s": round(a["cpu"], 1), "stops": a["stop"]}
    for role in ("null", "placebo", "placebo_conf", "true"):
        if a[f"{role}_n"]:
            row[role] = f"raw {a[f'{role}_raw']}/{a[f'{role}_n']} {wilson(a[f'{role}_raw'], a[f'{role}_n'])} BY {a[f'{role}_by']}"
    out.append(row)
    print(json.dumps(row))
