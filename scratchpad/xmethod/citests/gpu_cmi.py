"""cmi_knn torch (GPU) backend: F2 reproduction of CPU tigramite CMIknn and R-9 cost per dataset (orchestrator task).

  python scratchpad/xmethod/citests/gpu_cmi.py f2 OUT_DIR [--device cuda]
  python scratchpad/xmethod/citests/gpu_cmi.py cost OUT_DIR WORLD REGIME N [--cap-min 40] [--device cuda]

f2: identical inputs (harness DEV datasets, kappa .25, arm eq, seed 3_000_010) -> CPU tigramite 5.2.10.1 vs the torch
backend: (a) raw neighbour counts (k_xz, k_yz, k_z) of one ``_get_nearest_neighbors`` call (the same noise draws:
both from a fresh random_state with the same seed) at n 500 / 1000 / 4000; (b) run_test_raw (statistic + p of a
10-surrogate restricted-shuffle null, same seed) for 2 candidates per world at n 500 and 1 at n 1000, worlds E1 R1, E2 R1,
E2 R2, E3 R2, E4 R3, E5 R1. Pre-stated tolerance: EXACT (integer counts identical; statistic and p bit-identical).
-> OUT_DIR/F2_GPU.json.

cost: one DEV dataset (seed 3_000_011, kappa .25, arm eq), PRODUCTION config (backend torch, B 9999 Besag-Clifford
h 20, R-9), candidates in order, one JSON line per candidate (wall s, draws, p, capped) -> OUT_DIR/cost_<w>_<r>_<n>.jsonl
(resumable: done candidates skipped). A wall cap (--cap-min) stops the run mid-candidate (that candidate is
"capped", its p invalid): the per-surrogate time t_s and the draws give the extrapolation done in the summary.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

SEED_F2, SEED_COST = 3_000_010, 3_000_011
KAPPA = 0.25


def _setup():
    from cdd_oran.xmethod.methods._citests_common import method_seed, prepare
    from cdd_oran.xmethod.methods.cmi_knn import CMIKnnMethod
    from cdd_oran.xmethod.worlds import generate_dataset

    return CMIKnnMethod(), prepare, generate_dataset, method_seed


def f2(out_dir, device):
    import torch

    m, prepare, gen, _ = _setup()
    res = {"device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
           "torch": torch.__version__, "counts": [], "tests": []}
    ok = True
    for w, r, n in (("E2", "R2", 500), ("E5", "R1", 1000), ("E2", "R1", 4000), ("E4", "R3", 4000)):
        ds, _ = gen(w, r, n, SEED_F2, kappa=KAPPA)
        p = prepare(ds, "eq")
        i, j = p.pairs[0]
        x, y, z = p.S[:, [i]], p.Y[:, [j]], p.z_of(p.S, i)
        arr = np.vstack([x.T, y.T, z.T])
        xyz = np.array([0, 1] + [2] * z.shape[1])
        knn = max(1, int(0.2 * n))
        cnt, sec = [], []
        for backend in ("cpu", "torch"):
            t = m.make_test({**m.default_config(), "backend": backend, "device": device, "workers": -1}, 77)
            t0 = time.time()
            cnt.append(t._get_nearest_neighbors(arr.copy(), xyz, knn))
            sec.append(time.time() - t0)
        same = all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(cnt[0], cnt[1], strict=True))
        ok &= same
        res["counts"].append({"world": w, "regime": r, "n": n, "dim_z": int(z.shape[1]), "knn": knn,
                              "identical": bool(same), "cpu_s": sec[0], "torch_s": sec[1]})
        print(res["counts"][-1], flush=True)
    for w, r in (("E1", "R1"), ("E2", "R1"), ("E2", "R2"), ("E3", "R2"), ("E4", "R3"), ("E5", "R1")):
        for n in (500, 1000):
            ds, _ = gen(w, r, n, SEED_F2, kappa=KAPPA)
            p = prepare(ds, "eq")
            for i, j in p.pairs[:2 if n == 500 else 1]:
                vals = []
                for backend in ("cpu", "torch"):
                    t = m.make_test({**m.default_config(), "backend": backend, "device": device, "workers": -1,
                                     "sig_samples": 10, "bc_h": None}, 1234)
                    t0 = time.time()
                    v, pv = t.run_test_raw(p.S[:, [i]], p.Y[:, [j]], p.z_of(p.S, i))
                    vals.append((float(v), float(pv), time.time() - t0))
                same = vals[0][:2] == vals[1][:2]
                ok &= same
                res["tests"].append({"world": w, "regime": r, "n": n, "candidate": list(ds.candidates[p.pairs.index((i, j))]),
                                     "stat": vals[0][0], "p": vals[0][1], "identical": bool(same),
                                     "abs_stat_diff": abs(vals[0][0] - vals[1][0]), "cpu_s": vals[0][2],
                                     "torch_s": vals[1][2]})
                print(res["tests"][-1], flush=True)
    res["tolerance"] = "exact (integer counts identical; statistic and p bit-identical)"
    res["pass"] = bool(ok)
    json.dump(res, open(os.path.join(out_dir, "F2_GPU.json"), "w"), indent=1)
    print("F2_GPU pass", ok, flush=True)


def cost(out_dir, world, regime, n, cap_min, device):
    import torch

    m, prepare, gen, method_seed = _setup()
    out = os.path.join(out_dir, f"cost_{world}_{regime}_{n}.jsonl")
    done = set()
    if os.path.exists(out):
        done = {tuple(json.loads(line)["candidate"]) for line in open(out) if '"capped": false' in line}
    deadline = time.time() + 60 * cap_min
    t0 = time.time()
    ds, truth = gen(world, regime, n, SEED_COST, kappa=KAPPA)
    p = prepare(ds, "eq")
    prep_s = time.time() - t0
    cfg = {**m.default_config(), "backend": "torch", "device": device}
    test = m.make_test(cfg, method_seed(ds.seed, m.method_key))
    test.deadline = deadline
    for e, (i, j) in enumerate(p.pairs):
        cand = tuple(ds.candidates[e])
        if cand in done:
            continue
        if time.time() > deadline:
            break
        t1 = time.time()
        v, pv = test.run_test_raw(p.S[:, [i]], p.Y[:, [j]], p.z_of(p.S, i))
        if device == "cuda":
            torch.cuda.synchronize()
        rec = {"world": world, "regime": regime, "n": n, "kappa": KAPPA, "arm": "eq", "e": e, "candidate": list(cand),
               "true": cand in truth.edges, "dim_z": len(p.z_cols(i)), "wall_s": time.time() - t1,
               "draws": test.last_draws, "p": float(pv), "stat": float(v), "capped": bool(test.last_capped),
               "prep_s": prep_s, "n_candidates": len(p.pairs), "device": device,
               "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
               "cache_hits": test.knn_counts.hits if test.knn_counts else None,
               "cache_misses": test.knn_counts.misses if test.knn_counts else None}
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(rec, flush=True)
        if rec["capped"]:
            break


if __name__ == "__main__":
    a = sys.argv[1:]
    dev = a[a.index("--device") + 1] if "--device" in a else "cuda"
    os.makedirs(a[1], exist_ok=True)
    if a[0] == "f2":
        f2(a[1], dev)
    else:
        cap = float(a[a.index("--cap-min") + 1]) if "--cap-min" in a else 40.0
        cost(a[1], a[2], a[3], int(a[4]), cap, dev)
