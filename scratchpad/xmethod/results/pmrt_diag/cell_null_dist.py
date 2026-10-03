"""Null distribution of a DEV cell's truth-null rate (E2 R2 n 1000 kappa .5) from the dither-regenerated replicates:
a simulated cell = one random replicate per seed (seeds 100-119: 20-seed cell; 100-159: 60-seed cell); P(rate >= the
observed DEV rate) and P(seed-cluster bootstrap lower CI > .05), for prod and inv."""
import json, sys
import numpy as np
R = "scratchpad/xmethod/results/pmrt_diag/raw/"
reg = [json.loads(l) for l in open(R + "e2r2_n1000_k05_regen.jsonl")]
dev = {d["seed"]: d for d in map(json.loads, open(R + "e2r2_n1000_k05_dev.jsonl"))}
rng = np.random.default_rng([7801, 99, 11])
out = {}
for v in ("prod", "inv"):
    H = {}
    for d in reg:
        H.setdefault(d["seed"], []).append(sum(p <= .05 for e, (p, z) in d[v].items() if not e.startswith("P_pl")))
    m = 32
    for lo, hi in ((3000100, 3000119), (3000100, 3000159)):
        seeds = list(range(lo, hi + 1))
        obs = np.mean([sum(p <= .05 for e, (p, z) in dev[s][v].items() if not e.startswith("P_pl")) for s in seeds]) / m
        sims, inval = [], 0
        for _ in range(4000):
            h = np.array([H[s][rng.integers(len(H[s]))] for s in seeds], float)
            sims.append(h.mean() / m)
            bs = h[rng.integers(0, len(h), (400, len(h)))].mean(1) / m
            inval += np.quantile(bs, .025) > .05
        sims = np.array(sims)
        out[f"{v}|{lo}-{hi}"] = {"dev_rate": round(float(obs), 4), "sim_mean": round(float(sims.mean()), 4),
                                 "sim_sd": round(float(sims.std()), 4), "P_sim_ge_dev": round(float((sims >= obs - 1e-12).mean()), 4),
                                 "P_INVALID": round(inval / 4000, 4)}
        print(v, lo, hi, out[f"{v}|{lo}-{hi}"])
json.dump(out, open("scratchpad/xmethod/results/pmrt_diag/cell_null_dist.json", "w"), indent=1)
