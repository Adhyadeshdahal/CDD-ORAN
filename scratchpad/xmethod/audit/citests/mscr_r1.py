"""audit-citests: MSCR on the iid regime R1 (E1, E2; seeds 3_000_000-001; kappa .25, n 1000) vs R2 smoke, both arms:
separates a serial-dependence (R2) failure from an adapter bug. Raw p <= .05 counts on truth-null primary edges and
P_placebo. Run: PYTHONPATH=. uv run --group baselines --group citests python scratchpad/xmethod/audit/citests/mscr_r1.py"""
import json
from cdd_oran.xmethod.methods.mscr import MSCRMethod
from cdd_oran.xmethod.worlds.generate import generate_dataset
m = MSCRMethod()
for w in ("E1", "E2"):
    for seed in (3_000_000, 3_000_001):
        d, t = generate_dataset(w, "R1", 1000, seed, kappa=0.25)
        for arm in ("eq", "native"):
            r = m.run(d, {**m.default_config(), "arm": arm, "n_jobs": 1})
            nul = [e.p for e in r.edges if e.source in d.action_names and e.source != "P_placebo"
                   and (e.source, e.target) in t.null_edges]
            pl = [e.p for e in r.edges if e.source == "P_placebo"]
            print(json.dumps({"cell": f"{w}R1", "seed": seed, "arm": arm,
                              "null_raw": f"{sum(p <= .05 for p in nul)}/{len(nul)}",
                              "placebo_raw": f"{sum(p <= .05 for p in pl)}/{len(pl)}",
                              "placebo_p": [round(p, 4) for p in pl]}), flush=True)
