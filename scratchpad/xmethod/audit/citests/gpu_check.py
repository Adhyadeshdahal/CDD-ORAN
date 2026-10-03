"""audit-citests, check 4 (R-26): cmi_knn torch backend (local CUDA, RTX 2050) == CPU tigramite backend, statistic and
p, same seed, on real harness datasets in both arms; plus per-surrogate time. Fixed small B so the CPU side is cheap.
Run: PYTHONPATH=. uv run --group baselines --group citests python scratchpad/xmethod/audit/citests/gpu_check.py
"""
from __future__ import annotations

import json
import time

import numpy as np

from cdd_oran.xmethod.methods import _citests_common as cc
from cdd_oran.xmethod.methods.cmi_knn import CMIKnnMethod
from cdd_oran.xmethod.worlds.generate import generate_dataset

m = CMIKnnMethod()
rows = []
for w, r, n, B in (("E4", "R1", 1000, 40), ("E4", "R3", 600, 40), ("E1", "R2", 500, 20), ("E2", "R2", 400, 10)):
    d, _ = generate_dataset(w, r, n, 3_000_003, lam=1.0, kappa=0.25)
    for arm in ("eq", "native"):
        p = cc.prepare(d, arm)
        for e in list(range(len(p.pairs)))[:: max(1, len(p.pairs) // 3)][:3]:
            i, j = p.pairs[e]
            res = {}
            for be in ("cpu", "torch"):
                t = m.make_test({**m.default_config(), "sig_samples": B, "bc_h": None, "workers": 1, "backend": be,
                                 "device": "cuda"}, 777 + e)
                t0 = time.time()
                v, pv = t.run_test_raw(p.S[:, [i]], p.Y[:, [j]], p.z_of(p.S, i))
                res[be] = (float(v), float(pv), (time.time() - t0) / (B + 1))
            rows.append({"cell": f"{w}{r}", "n": n, "arm": arm, "edge": "{}->{}".format(*d.candidates[e]),
                         "dimZ": len(p.z_cols(i)), "val_cpu": res["cpu"][0], "val_gpu": res["torch"][0],
                         "p_cpu": res["cpu"][1], "p_gpu": res["torch"][1],
                         "s_per_surr_cpu": res["cpu"][2], "s_per_surr_gpu": res["torch"][2]})
out = {"rows": rows, "max_val_absdiff": max(abs(x["val_cpu"] - x["val_gpu"]) for x in rows),
       "max_p_absdiff": max(abs(x["p_cpu"] - x["p_gpu"]) for x in rows)}
print(json.dumps(out, indent=1))
