"""DEV E2 R2 pmrt truth-null raw-p rates per (arm, kappa, n), seed-cluster bootstrap CI; per-source rates."""
import json, sys, collections
import numpy as np
from cdd_oran.xmethod.worlds.generate import truth_for
T = truth_for("E2", "R2")
recs = json.load(open(sys.argv[1]))
cells = collections.defaultdict(dict)
for r in recs:
    arm, w, reg, k, n, s = r["key"].split("|")
    if r["role"] != "measure":
        continue
    cells[(arm, k, n)][s] = r["edges"]
rng = np.random.default_rng(0)
for key in sorted(cells):
    seeds = sorted(cells[key])
    rej = np.array([[e[2] is not None and e[2] <= .05 for e in cells[key][s] if (e[0], e[1]) in T.null_edges and e[0].startswith("P") and e[0] != "P_placebo"] for s in seeds], float)
    pl = np.array([[e[2] <= .05 for e in cells[key][s] if e[0] == "P_placebo"] for s in seeds], float)
    per = rej.mean(1)
    bs = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(2000)]
    src = collections.defaultdict(list)
    names = [(e[0], e[1]) for e in cells[key][seeds[0]] if (e[0], e[1]) in T.null_edges and e[0].startswith("P") and e[0] != "P_placebo"]
    for j, (a, _) in enumerate(names):
        src[a].append(rej[:, j].mean())
    print(key, "S", len(seeds), f"{seeds[0]}-{seeds[-1]}", "m", rej.shape[1], f"null {per.mean():.3f} [{np.quantile(bs,.025):.3f},{np.quantile(bs,.975):.3f}]",
          f"plac {pl.mean():.3f}", " ".join(f"{a}:{np.mean(v):.3f}" for a, v in sorted(src.items())))
