"""Runner of the data-efficiency BENCH (cdd_oran/decision/disc_bench.py; PMRT research).

  python scratchpad/e6_dev/disc_bench_run.py cache --inputs PATHS --stage eval --out DIR [--workers 4]
        PATHS: comma list of JSONL files / directories (recursive res_*.jsonl, /bundle/ skipped) / globs. One npz per
        source file: DIR/<source label>__<file stem>.npz (label = the path component starting "e6p-disc-", else the
        parent dir). Streams one episode at a time.
  python scratchpad/e6_dev/disc_bench_run.py gtref --gt F1,F2 --out ref.json          # pooled GT reference
  python scratchpad/e6_dev/disc_bench_run.py bench --cache PATHS --gtref ref.json --dev dev.npz
        [--placebo name=npz,name=npz] [--methods mscr_v2,corr,granger,granger_by] [--B 9999] [--H 90] [--H-pre 90]
        [--sizes 60,120,300] [--R 60:10,120:10,300:5] [--seed 0] [--stages eval] [--check-v3] [--out DIR]
        Writes DIR/bench_<method>.json (+ bench_check_v3.json). --cache: comma list of npz files / dirs (recursive
        *.npz) / globs; only --stages (default eval) episodes enter the pool. On Kaggle attach the cache kernel
        (--sources bishalpanta/disc-bench-1) and pass --cache "$JOB_SRC/disc-bench-1/out/cache" (its
        out/cache_small/ holds dev1 / placebo1 / plxc2 npz + the pooled GT reference gtref_gt1_gtx3.json).
Env: JOB_OUT (kaggle_job.py) is the default --out.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from cdd_oran.decision import disc_bench as DB  # noqa: E402


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def expand(spec, pattern):
    out = []
    for p in [x for x in spec.split(",") if x]:
        if os.path.isdir(p):
            out += sorted(f for f in glob.glob(os.path.join(p, "**", pattern), recursive=True)
                          if "/bundle/" not in f.replace("\\", "/"))
        elif any(ch in p for ch in "*?["):
            out += sorted(glob.glob(p, recursive=True))
        else:
            out.append(p)
    seen, uniq = set(), []
    for f in out:
        if os.path.abspath(f) not in seen:
            seen.add(os.path.abspath(f))
            uniq.append(f)
    return uniq


def label_of(path):
    parts = os.path.abspath(path).replace("\\", "/").split("/")
    lab = next((c for c in reversed(parts[:-1]) if c.startswith("e6p-disc-")), parts[-2])
    return lab


def _cache_one(args):
    path, stage, out, Hs = args
    stem = os.path.splitext(os.path.basename(path))[0]
    lab = label_of(path)
    dst = os.path.join(out, f"{lab}__{stem}.npz")
    return DB.build_cache([path], dst, stages={stage}, Hs=Hs, src=f"{lab}/{os.path.basename(path)}")


def cmd_cache(a):
    files = expand(a.inputs, "res_*.jsonl")
    if not files:
        sys.exit(f"no inputs in {a.inputs}")
    Hs = tuple(int(h) for h in a.H.split(","))
    os.makedirs(a.out, exist_ok=True)
    log(f"cache: {len(files)} files, stage {a.stage}, H {Hs}, workers {a.workers}")
    jobs = [(f, a.stage, a.out, Hs) for f in files]
    t = time.time()
    if a.workers > 1:
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            res = []
            for s in pool.imap_unordered(_cache_one, jobs):
                log(f"  {s}")
                res.append(s)
    else:
        res = []
        for j in jobs:
            s = _cache_one(j)
            log(f"  {s}")
            res.append(s)
    summ = {"files": len(res), "episodes": sum(s["episodes"] for s in res), "units": sum(s["units"] for s in res),
            "mb": round(sum(s["mb"] for s in res), 1), "wall_s": round(time.time() - t, 1), "per_file": res}
    json.dump(summ, open(os.path.join(a.out, "cache_summary.json"), "w"), indent=1)
    log(f"cache done: {len(res)} files, {summ['episodes']} episodes, {summ['units']} units, {summ['mb']} MB, "
        f"{summ['wall_s']} s")


def cmd_gtref(a):
    g = DB.gt_reference_files(a.gt.split(","))
    DB.save_ref(g, a.out)
    log(f"GT ref: {g['n_episodes']} episodes, {g['n_labels']} labels {g['labels_per_family']}; counts {g['counts']}; "
        f"nbr TRUE {g['nbr_true']} -> {a.out}")


def _jsonable(d):
    return json.loads(json.dumps(d, default=DB._js))


def cmd_bench(a):
    out = a.out or os.environ.get("JOB_OUT") or "."
    os.makedirs(out, exist_ok=True)
    files = expand(a.cache, "*.npz")
    if not files:
        sys.exit(f"no cache files in {a.cache}")
    t = time.time()
    pool = DB.load_pool(files, H=a.H, H_pre=a.H_pre, stages=set(a.stages.split(",")) if a.stages else None)
    log(f"pool: {len(files)} files, {pool.n_eps} episodes, {len(pool.u['gep'])} units, H {pool.H}/{pool.H_pre}, "
        f"subs { {s: int((pool.eps['sub'] == s).sum()) for s in sorted(set(pool.eps['sub'].tolist()))} } "
        f"[{time.time() - t:.1f} s]")
    g = DB.load_ref(a.gtref)
    ref = g["ref"]
    log(f"GT ref: {g['n_episodes']} episodes, counts {g['counts']}, nbr TRUE {g['nbr_true']}")
    dev = DB.load_pool(a.dev.split(","), H=a.H, H_pre=a.H_pre).unit_data() if a.dev else None
    placebos = {}
    for spec in [x for x in (a.placebo or "").split(",") if x]:
        nm, f = spec.split("=", 1)
        placebos[nm] = DB.load_pool([f], H=a.H, H_pre=a.H_pre).unit_data()
    R = {int(k): int(v) for k, v in (x.split(":") for x in a.R.split(","))}
    sizes = tuple(int(s) for s in a.sizes.split(","))
    for name in a.methods.split(","):
        t = time.time()
        if name == "mscr_v2":
            m = DB.method_mscr_v2(B=a.B)
        else:
            if dev is None and name != "granger_by":
                sys.exit(f"{name} needs --dev (DEV-tuned tau)")
            m = DB.method_baseline(name, dev)
        log(f"== method {m.label} (tau {getattr(m, 'tau', None)})")
        res = DB.bench(pool, m, ref, sizes=sizes, R=R, seed=a.seed, placebos=placebos, log=log, keep_decl=True)
        res.update(method=m.label, tau=getattr(m, "tau", None), B=a.B if name == "mscr_v2" else None,
                   gt={k: g[k] for k in ("n_episodes", "n_labels", "counts", "nbr_true", "seeds")},
                   cache_files=[os.path.basename(f) for f in files], wall_s=round(time.time() - t, 1))
        json.dump(_jsonable(res), open(os.path.join(out, f"bench_{name}.json"), "w"), indent=1)
        log(f"== {m.label} done in {res['wall_s']} s -> bench_{name}.json")
    if a.check_v3:
        from cdd_oran.decision import crt_units_v2 as V2
        from cdd_oran.decision.crt_units import SPLIT_POOLED
        v3 = pool.episodes((pool.eps["sub"] == "v3") & (pool.eps["stage"] == "eval"))
        chk = {"n_eps": int(len(v3))}
        if len(v3):
            cfg = V2.UnitCRTConfigV2(B=9999)
            d = pool.unit_data(v3)
            chk["units"] = int(d.n)
            run = V2.run_crt_units_v2(d, cfg, SPLIT_POOLED)
            chk["pooled"] = {"n_declared": run["n_declared"], "wall_s": run["wall_s"],
                             "declared": sorted((r["family"], r["relation"], r["kpi"], r["sign"],
                                                 round(r["z_approx"], 2)) for r in run["results"]
                                                if r["status"] == "declared")}
            del d
            e = V2.edges_from_crt_v2(run)
            chk["pooled"]["score"] = DB.subset_metrics(e, ref)
            chk["folds"] = {}
            for k in range(4):
                d = pool.unit_data(v3[pool.eps["fold"][v3] == k])
                r = V2.run_crt_units_v2(d, cfg, 1 + k)
                chk["folds"][k] = r["n_declared"]
            chk["expected (STEP1_V3_RESULT.md)"] = {"pooled_declared": 28, "folds": [20, 21, 20, 23],
                                                    "sleep_nbr_pv_z": 6.11, "sleep_nbr_load_z": 29.53}
        json.dump(_jsonable(chk), open(os.path.join(out, "bench_check_v3.json"), "w"), indent=1)
        log(f"check v3: {json.dumps(_jsonable({k: v for k, v in chk.items() if k != 'pooled'}))}; pooled declared "
            f"{chk.get('pooled', {}).get('n_declared')}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="op", required=True)
    c = sp.add_parser("cache")
    c.add_argument("--inputs", required=True)
    c.add_argument("--stage", default="eval")
    c.add_argument("--out", required=True)
    c.add_argument("--workers", type=int, default=1)
    c.add_argument("--H", default=",".join(map(str, DB.HS)))
    g = sp.add_parser("gtref")
    g.add_argument("--gt", required=True)
    g.add_argument("--out", required=True)
    b = sp.add_parser("bench")
    b.add_argument("--cache", required=True)
    b.add_argument("--gtref", required=True)
    b.add_argument("--dev", default=None)
    b.add_argument("--placebo", default="")
    b.add_argument("--methods", default="mscr_v2,corr,granger,granger_by")
    b.add_argument("--B", type=int, default=9999)
    b.add_argument("--H", type=int, default=90)
    b.add_argument("--H-pre", dest="H_pre", type=int, default=None)
    b.add_argument("--sizes", default="60,120,300")
    b.add_argument("--R", default="60:10,120:10,300:5")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--stages", default="eval", help="pool = episodes of these stages only ('' = all)")
    b.add_argument("--check-v3", action="store_true")
    b.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    np.seterr(all="ignore")
    {"cache": cmd_cache, "gtref": cmd_gtref, "bench": cmd_bench}[a.op](a)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    main()
