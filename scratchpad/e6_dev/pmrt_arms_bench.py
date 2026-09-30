"""PMRT arms bench job (agent S; cdd_oran/decision/pmrt.py; formerly mscr_plus_bench.py, method label "MSCR+").
Kaggle only for the big pool (1680 eps).

  python scratchpad/e6_dev/pmrt_arms_bench.py all --cache DIR_OR_FILES --small DIR --out DIR
         [--train sub=v2,stage=dev] [--eval sub=v3] [--sizes 60,120,300] [--R 60:10,120:10] [--B 9999]
         [--cv-reps 3] [--no-gb] [--params P.pkl (skip fit)] [--baselines v2,eproc] [--max-subsets 0]

1. FIT (training = episodes matching --train: ev2 sub "v2" + DEV stage "dev"): pmrt.fit_pmrt -> out/params.pkl
   (+ params.json summary: per hypothesis the CV z of every variant, the chosen "best", in-sample z).
2. BENCH on --eval episodes only (ev3 sub "v3"; never seen by the fit): n = 60 x R, 120 x R disjoint random subsets
   (seeded default_rng([6690, n, R, seed]), split 20 + i), n = 300 = the four v3 folds (split 1 + k). Per subset one
   CRT per family gives every PMRT arm / combo; each arm is scored after BY (disc_bench.subset_metrics vs gtref);
   baselines on the same subsets: mscr-crt-units-v2 (B) and V's e-process (eprocess_units, e-BH).
3. PLACEBO (placebo1, plxc2 from --small): per arm #BY declarations, rate p <= .05.
Writes out/bench_plus.json (+ progress in stdout).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from cdd_oran.decision import pmrt as PM  # noqa: E402
from cdd_oran.decision import crt_units_v2 as V2  # noqa: E402
from cdd_oran.decision import disc_bench as DB  # noqa: E402
from cdd_oran.decision import eprocess_units as EP  # noqa: E402


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def expand(spec):
    out = []
    for p in [x for x in spec.split(",") if x]:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "**", "*.npz"), recursive=True))
        elif any(ch in p for ch in "*?["):
            out += sorted(glob.glob(p, recursive=True))
        else:
            out.append(p)
    return out


def select(pool, spec):
    """episodes matching any 'key=value' of spec (keys: sub, stage, src, file)."""
    m = np.zeros(pool.n_eps, bool)
    for kv in [x for x in spec.split(",") if x]:
        k, v = kv.split("=", 1)
        m |= np.array([str(x) == v for x in pool.eps[k]])
    return np.nonzero(m)[0]


def subsets(pool, ids, n, R, seed=0):
    E = pool.eps
    if n == 300 and all(((E["fold"][ids] == k).sum() == 300) for k in range(4)):
        return [{"name": f"v3fold{k}", "episodes": ids[E["fold"][ids] == k], "split": 1 + k} for k in range(4)]
    rng = np.random.default_rng([DB.BENCH_TAG, int(n), int(R), int(seed)])
    perm = rng.permutation(ids)
    out = []
    for i in range(R):
        if (i + 1) * n > len(perm):
            break
        out.append({"name": f"rand{i}", "episodes": np.sort(perm[i * n:(i + 1) * n]), "split": 20 + i})
    return out


def eproc_edges(ud):
    run = EP.run_eprocess_units(ud)
    out = {}
    for r in run["results"]:
        out[(r["family"], r["relation"], r["kpi"])] = {"declared": r["status"] == "declared", "sign": int(r["sign"]),
                                                       "p": r.get("p_anytime"), "z_approx": r.get("z_equiv")}
    return out


def agg(rows):
    out = {}
    for m in DB.METRICS:
        v = np.array([r[m] for r in rows], float)
        f = v[np.isfinite(v)]
        out[m] = round(float(f.mean()), 4) if len(f) else None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("op", choices=["all", "fit", "bench"])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--small", required=True, help="dir with dev1.npz, placebo1.npz, plxc2.npz, gtref_gt1_gtx3.json")
    ap.add_argument("--out", default=os.environ.get("JOB_OUT", "."))
    ap.add_argument("--train", default="sub=v2,stage=dev")
    ap.add_argument("--eval", default="sub=v3")
    ap.add_argument("--sizes", default="60,120,300")
    ap.add_argument("--R", default="60:10,120:10")
    ap.add_argument("--B", type=int, default=9999)
    ap.add_argument("--cv-reps", type=int, default=3)
    ap.add_argument("--h-pow", type=float, default=1.0)
    ap.add_argument("--no-gb", action="store_true")
    ap.add_argument("--params", default="")
    ap.add_argument("--baselines", default="v2,eproc")
    ap.add_argument("--max-subsets", type=int, default=0)
    ap.add_argument("--no-dev", action="store_true", help="do not add small/dev1.npz to the pool")
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    os.makedirs(a.out, exist_ok=True)
    files = expand(a.cache) + ([] if a.no_dev else [os.path.join(a.small, "dev1.npz")])
    t = time.time()
    pool = PM.load_pmrt_pool(files, stages=None)
    log(f"pool {len(files)} files, {pool.n_eps} eps, {len(pool.u['gep'])} units, subs "
        f"{ {s: int((pool.eps['sub'] == s).sum()) for s in sorted(set(pool.eps['sub'].tolist()))} } "
        f"[{time.time() - t:.0f} s]")
    tr, ev = select(pool, a.train), select(pool, a.eval)
    assert not set(tr.tolist()) & set(ev.tolist()), "train / eval overlap"
    log(f"train eps {len(tr)}, eval eps {len(ev)}")
    cfg = PM.PmrtConfig(B=a.B, cv_reps=a.cv_reps, gb=not a.no_gb, h_pow=a.h_pow)
    if a.params:
        params = PM.load_params(a.params)
    else:
        t = time.time()
        pdt = PM.pmrt_data(pool, tr)
        log(f"train PmrtData {pdt.n} units")
        params = PM.fit_pmrt(pdt, cfg, log=log)
        params["train_eps"] = len(tr)
        del pdt
        PM.save_params(params, os.path.join(a.out, "params.pkl"))
        log(f"fit done {time.time() - t:.0f} s")
    if a.op == "fit":
        return
    g = DB.load_ref(os.path.join(a.small, "gtref_gt1_gtx3.json"))
    ref = g["ref"]
    res = {"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(a).items()},
           "pmrt_config": PM.dataclasses.asdict(cfg), "sizes": {}, "placebo": {},
           "best": {h: p["best"] for h, p in params["hyp"].items()}}
    Rm = {int(k): int(v) for k, v in (x.split(":") for x in a.R.split(","))}
    bl = [b for b in a.baselines.split(",") if b]

    def dump():
        json.dump(json.loads(json.dumps(res, default=DB._js)), open(os.path.join(a.out, "bench_plus.json"), "w"),
                  indent=1)

    # placebo first (fast)
    for nm in ("placebo1", "plxc2"):
        f = os.path.join(a.small, f"{nm}.npz")
        pdp = PM.pmrt_data(PM.load_pmrt_pool([f]))
        run = PM.run_pmrt(pdp, params, cfg, split=9)
        pr = {}
        for arm in run["arms"]:
            e = PM.edges_pmrt(run, arm)
            ps = [v["p"] for v in e.values() if np.isfinite(v["p"])]
            pr[arm] = {"n_declared": int(sum(v["declared"] for v in e.values())),
                       "declared": sorted("|".join(h) for h, v in e.items() if v["declared"]),
                       "rate": round(float(np.mean(np.array(ps) <= .05)), 4), "min_p": float(min(ps))}
        if "v2" in bl:
            e = V2.edges_from_crt_v2(V2.run_crt_units_v2(pdp.ud, V2.UnitCRTConfigV2(B=a.B), 9))
            ps = [v["p"] for v in e.values() if np.isfinite(v["p"])]
            pr["v2"] = {"n_declared": int(sum(v["declared"] for v in e.values())),
                        "rate": round(float(np.mean(np.array(ps) <= .05)), 4), "min_p": float(min(ps))}
        if "eproc" in bl:
            e = eproc_edges(pdp.ud)
            pr["eproc"] = {"n_declared": int(sum(v["declared"] for v in e.values()))}
        res["placebo"][nm] = pr
        log(f"placebo {nm}: " + ", ".join(f"{k} {v['n_declared']}/{v.get('rate')}" for k, v in pr.items()))
        dump()
    for n in [int(x) for x in a.sizes.split(",")]:
        subs = subsets(pool, ev, n, Rm.get(n, 10))
        if a.max_subsets:
            subs = subs[: a.max_subsets]
        rows = {}
        for s in subs:
            t = time.time()
            pd = PM.pmrt_data(pool, s["episodes"])
            run = PM.run_pmrt(pd, params, cfg, split=s["split"])
            decls = {arm: PM.edges_pmrt(run, arm, cfg.q) for arm in run["arms"]}
            if "v2" in bl:
                decls["v2"] = V2.edges_from_crt_v2(V2.run_crt_units_v2(pd.ud, V2.UnitCRTConfigV2(B=a.B), s["split"]))
            if "eproc" in bl:
                decls["eproc"] = eproc_edges(pd.ud)
            for arm, d in decls.items():
                m = DB.subset_metrics(d, ref)
                m.update(name=s["name"], units=int(pd.n),
                         declared=sorted(("|".join(h), int(v.get("sign", 0))) for h, v in d.items()
                                         if v and v.get("declared")))
                rows.setdefault(arm, []).append(m)
            zs = {arm: round(float(decls[arm][DB.PREMISE].get("z_approx") or np.nan), 2)
                  for arm in ("v2", "pred", "plain", "best", "max_pred_best", "loadsp", "rank1") if arm in decls}
            log(f"n={n} {s['name']}: units {pd.n} ({time.time() - t:.0f} s) premise z {zs} | " + " ".join(
                f"{arm}:{rows[arm][-1]['chain_hits']}/{rows[arm][-1]['ind_f1']:.2f}/{rows[arm][-1]['n_declared']}"
                for arm in ("v2", "eproc", "pred", "plain", "best", "best_gb", "max_pred_best") if arm in rows))
            del pd
            res["sizes"][str(n)] = {"R": len(rows[next(iter(rows))]), "summary": {arm: agg(r) for arm, r in rows.items()},
                                    "subsets": rows}
            dump()
        summ = res["sizes"][str(n)]["summary"]
        for arm in sorted(summ, key=lambda k: -(summ[k]["ind_f1"] or 0)):
            m = summ[arm]
            log(f"  n={n} {arm:>18}: chain {m['chain_hits']:.2f} prem {m['premise_hit']:.2f} indR {m['ind_recall']:.2f} "
                f"indP {m['ind_precision']} indF1 {m['ind_f1']:.3f} ovP {m['ov_precision']} sign {m['sign_acc']} "
                f"#dec {m['n_declared']:.1f} far {m['far_declared']:.1f}")
    dump()
    log("done")


if __name__ == "__main__":
    main()
