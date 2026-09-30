"""Agent M (PMRT research): bench of the DECLARATION layer (cdd_oran/decision/fdr_layer.py; formerly
mscr_multi.py / mscr_multi_bench.py) on MSCR-CRT v2 statistics.

  python scratchpad/e6_dev/fdr_layer_bench.py stats --cache DIR --small DIR [--B 9999] [--out DIR]     (Kaggle)
        1. PRIOR: MSCR-CRT v2 on the 480 ev2 episodes (sub "v2", stage eval) -> per-hypothesis z_approx / sign / p.
        2. EVAL subsets drawn from the 1200 ev3 episodes ONLY (sub "v3"): n = 60 x 10 and 120 x 10 (disjoint random,
           default_rng([6690, n, R, 0, 71])), n = 300 = the four v3 EVAL folds, n = 1200 = all of v3.
        3. per subset / placebo: the v2 statistic with TWO-SIDED and both ONE-SIDED CRT p-values from the SAME draws
           (p2 identical to crt_units_v2.run_crt_units_v2: asserted), beta / sign / z; and agent V's e-process e-values
           (eprocess_units, default config): two-sided grapa and one-sided (+ / -) grapa.
        -> DIR/mscr_multi_stats.json (small; everything the declaration layer needs).
  python scratchpad/e6_dev/fdr_layer_bench.py eval --stats F --gtref F [--out DIR]                        (local)
        applies every procedure of fdr_layer.PROCEDURES and scores it with disc_bench.subset_metrics.
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

from cdd_oran.decision import crt_units_v2 as V2  # noqa: E402
from cdd_oran.decision import disc_bench as DB  # noqa: E402
from cdd_oran.decision import eprocess_units as EP  # noqa: E402
from cdd_oran.decision.crt_units import (  # noqa: E402
    FAMILIES,
    SPLIT_PLACEBO,
    SPLIT_POOLED,
    PiAssignment,
)

TAG_M = 71


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def sided_family(data, family, cfg, split):
    """crt_units_v2.family_tests_v2 (lag 0) + one-sided counts from the SAME draws. Returns {(rel, kpi): dict}."""
    targets = [(f"{r}_{k}", (r, k)) for r in data.relations for k in data.kpis]
    base = V2.family_tests_v2(data, family, targets, cfg, 0, split)       # reference p2 (identical stream)
    fi = FAMILIES.index(family)
    rows = data.rows_of(family)
    out = {}
    tested = [(t, b) for t, b in enumerate(base) if b["status"] == "tested"]
    for b in base:
        out[tuple(b["key"])] = {"status": b["status"], "reason": b.get("reason", ""), "p2": b["p"],
                                "p_plus": float("nan"), "p_minus": float("nan"), "beta": b["beta"],
                                "sign": int(b["sign"]), "z": b["z_approx"], "n": b["n"]}
    if not tested:
        return out
    strata = V2.strata_of(data, rows)
    cols = []
    for _t, b in tested:
        key = tuple(b["key"])
        r, _ = V2.residualise(data.y[key][rows], data.pre[key][rows], strata)
        cols.append(r)
    R = np.column_stack(cols)
    src = rows
    v = V2.design_regressor(data, src)
    s_obs = v @ R
    tol = 1e-9 * (1.0 + np.abs(R).sum(0))
    uniq, inv = np.unique(src, return_inverse=True)
    sg, mu = data.sgn[src], V2.design_mean(data, src)
    pa = PiAssignment(data)
    rng = V2._stream(cfg, fi, 0, split)
    ch = V2._chunk(cfg, len(uniq))
    c2 = np.zeros(len(cols), np.int64)
    cp = np.zeros(len(cols), np.int64)
    cm = np.zeros(len(cols), np.int64)
    done = 0
    while done < cfg.B:
        bb = min(ch, cfg.B - done)
        M = pa.conditional_draws(rng, family, bb, rows=uniq, what="mode")
        V = V2.LEVEL_V2_ARR[M]
        del M
        if len(uniq) != len(src) or np.any(uniq != src):
            V = V[:, inv]
        V -= mu[None, :]
        V *= sg[None, :]
        S = V @ R
        del V
        c2 += np.count_nonzero(np.abs(S) >= (np.abs(s_obs) - tol)[None, :], axis=0)
        cp += np.count_nonzero(S >= (s_obs - tol)[None, :], axis=0)
        cm += np.count_nonzero(S <= (s_obs + tol)[None, :], axis=0)
        done += bb
    for j, (_t, b) in enumerate(tested):
        o = out[tuple(b["key"])]
        p2 = float((1.0 + c2[j]) / (cfg.B + 1.0))
        if abs(p2 - b["p"]) > 1e-12:
            raise AssertionError(f"p2 mismatch {family} {b['key']}: {p2} vs {b['p']}")
        o["p_plus"] = float((1.0 + cp[j]) / (cfg.B + 1.0))
        o["p_minus"] = float((1.0 + cm[j]) / (cfg.B + 1.0))
    return out


def subset_stats(data, cfg, split, eproc=True):
    t = time.time()
    res = {}
    for f in FAMILIES:
        for (r, k), o in sided_family(data, f, cfg, split).items():
            res["|".join((f, r, k))] = o
    if eproc:
        ecfg = EP.EProcConfig()
        runs = {"e2": EP.run_eprocess_units(data, ecfg)}
        for nm, s in (("e_plus", 1), ("e_minus", -1)):
            sg = {f: {f"{r}_{k}": s for r in data.relations for k in data.kpis} for f in FAMILIES}
            runs[nm] = EP.run_eprocess_units(data, ecfg, signs=sg)
        for nm, run in runs.items():
            for r in run["results"]:
                h = "|".join((r["family"], r["relation"], r["kpi"]))
                res[h][nm] = float(r["e"]) if r["status"] != "undetermined" else float("nan")
                if nm == "e2":
                    res[h]["e_sign"] = int(r["sign"])
                    res[h]["e_status"] = r["status"] if r["status"] == "undetermined" else "tested"
    return {"hyp": res, "units": int(data.n), "wall_s": round(time.time() - t, 1)}


def v3_subsets(pool, seed=0):
    E = pool.eps
    v3 = np.nonzero((E["sub"] == "v3") & (E["stage"] == "eval"))[0]
    out = []
    for n, R in ((60, 10), (120, 10)):
        rng = np.random.default_rng([DB.BENCH_TAG, n, R, seed, TAG_M])
        perm = rng.permutation(v3)
        for i in range(R):
            out.append({"n": n, "name": f"v3r{n}_{i}", "episodes": np.sort(perm[i * n:(i + 1) * n]),
                        "split": 40 + i})
    for k in range(4):
        f = v3[E["fold"][v3] == k]
        out.append({"n": 300, "name": f"v3fold{k}", "episodes": f, "split": 1 + k})
    out.append({"n": 1200, "name": "v3all", "episodes": v3, "split": SPLIT_POOLED})
    return out


def cmd_stats(a):
    out = a.out or os.environ.get("JOB_OUT") or "."
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.cache, "**", "*.npz"), recursive=True))
    pool = DB.load_pool(files, H=90, H_pre=90, stages={"eval"})
    subs = {s: int((pool.eps["sub"] == s).sum()) for s in sorted(set(pool.eps["sub"].tolist()))}
    log(f"pool {pool.n_eps} eps, subs {subs}")
    cfg = V2.UnitCRTConfigV2(B=a.B)
    res = {"B": a.B, "H": 90, "pool_subs": subs, "subsets": [], "placebo": {}}
    # 1. prior (ev2)
    ev2 = pool.episodes((pool.eps["sub"] == "v2") & (pool.eps["stage"] == "eval"))
    d = pool.unit_data(ev2)
    st = subset_stats(d, cfg, SPLIT_POOLED, eproc=False)
    res["prior"] = dict(st, name="ev2", n_eps=int(len(ev2)))
    del d
    log(f"prior ev2 {len(ev2)} eps, {st['units']} units, {st['wall_s']} s")
    # 2. placebos
    for nm in ("placebo1", "plxc2"):
        d = DB.load_pool([os.path.join(a.small, f"{nm}.npz")], H=90, H_pre=90).unit_data()
        st = subset_stats(d, cfg, SPLIT_PLACEBO)
        res["placebo"][nm] = st
        log(f"placebo {nm}: {st['units']} units {st['wall_s']} s")
        del d
    # 3. eval subsets
    for s in v3_subsets(pool):
        d = pool.unit_data(s["episodes"])
        st = subset_stats(d, cfg, s["split"], eproc=s["n"] <= 1200)
        res["subsets"].append(dict(st, n=s["n"], name=s["name"], split=s["split"], n_eps=int(len(s["episodes"])),
                                   seeds_head=[int(x) for x in pool.eps["seed"][s["episodes"][:3]]]))
        log(f"{s['name']}: {st['units']} units {st['wall_s']} s")
        del d
        json.dump(res, open(os.path.join(out, "mscr_multi_stats.json"), "w"), default=DB._js)
    json.dump(res, open(os.path.join(out, "mscr_multi_stats.json"), "w"), default=DB._js)
    log("done")


def cmd_local(a):
    """Local smoke test of the stats pipeline on a small npz (subsets = the whole file)."""
    d = DB.load_pool([a.npz], H=90, H_pre=90).unit_data()
    cfg = V2.UnitCRTConfigV2(B=a.B)
    st = subset_stats(d, cfg, SPLIT_PLACEBO)
    json.dump(st, open(a.out, "w"), default=DB._js)
    log(f"{a.npz}: {st['units']} units {st['wall_s']} s -> {a.out}")


def cmd_eval(a):
    from cdd_oran.decision import fdr_layer as FL
    S = json.load(open(a.stats))
    ref = DB.load_ref(a.gtref)["ref"]
    prior = FL.Prior.from_stats(S["prior"]["hyp"], n_prior=S["prior"]["n_eps"], source="ev2 MSCR-CRT v2 (480 eps)")
    procs = FL.PROCEDURES if not a.procs else a.procs.split(",")
    out = {"prior": {"n": prior.n_prior, "z": {"|".join(h): v for h, v in prior.z.items()}},
           "procedures": {}, "config": dict(q=a.q, floor=a.floor, thr=a.thr)}
    for n in (20, 60, 120, 300, 1200):
        w = FL.prior_weights(prior, n, a.q, a.floor)
        top = sorted(w.items(), key=lambda kv: -kv[1])[:12]
        out.setdefault("weights", {})[str(n)] = {"|".join(h): round(v, 3) for h, v in w.items()}
        log(f"weights n={n}: top " + ", ".join(f"{'|'.join(h)} {v:.2f}" for h, v in top)
            + f"; premise {w[DB.PREMISE]:.2f}; #floor {sum(v < a.floor + 1e-6 for v in w.values())}")
    dirs = FL.prior_directions(prior, a.thr)
    out["directions"] = {"|".join(h): d for h, d in dirs.items() if d}
    log(f"directions (|z_ev2| >= {a.thr}): {len(out['directions'])}: {out['directions']}")
    table = []
    for pr in procs:
        rows_by_n = {}
        for s in S["subsets"]:
            stats = FL.stats_from_json(s["hyp"])
            decl = FL.declare(stats, pr, prior, n_target=s["n"], q=a.q, floor=a.floor, thr=a.thr)
            m = DB.subset_metrics(decl, ref)
            m.update(name=s["name"])
            rows_by_n.setdefault(s["n"], []).append(m)
        plc = {}
        for nm, s in S["placebo"].items():
            stats = FL.stats_from_json(s["hyp"])
            decl = FL.declare(stats, pr, prior, n_target=20, q=a.q, floor=a.floor, thr=a.thr)
            plc[nm] = sorted("|".join(h) for h, v in decl.items() if v["declared"])
            key = "e" if "e" in next(iter(decl.values())) else "p"
            if key == "p":
                ps = [v["p"] for v in decl.values() if np.isfinite(v["p"])]
                plc[nm + "_rate05"] = round(float(np.mean([x <= 0.05 for x in ps])), 4) if ps else None
            else:
                es = [v["e"] for v in decl.values() if np.isfinite(v["e"])]
                plc[nm + "_e_ge20"] = int(sum(x >= 20 for x in es))
        summ = {str(n): DB._agg(rows) for n, rows in rows_by_n.items()}
        out["procedures"][pr] = {"summary": summ, "placebo": plc,
                                 "subsets": {str(n): rows for n, rows in rows_by_n.items()}}
        for n, a_ in summ.items():
            table.append((pr, int(n), a_["chain_hits"]["mean"], a_["premise_hit"]["mean"], a_["ind_recall"]["mean"],
                          a_["ind_precision"]["mean"], a_["ind_f1"]["mean"], a_["ov_precision"]["mean"],
                          a_["sign_acc"]["mean"], a_["n_declared"]["mean"], sum(len(v) for v in plc.values() if isinstance(v, list))))
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "mscr_multi_eval.json"), "w"), indent=1, default=DB._js)
    hdr = f"{'proc':<14}{'n':>5}{'chain':>7}{'prem':>6}{'indR':>6}{'indP':>6}{'indF1':>7}{'ovP':>6}{'sign':>6}" \
          f"{'#dec':>6}{'plc':>5}"
    lines = [hdr]
    for n in sorted({t[1] for t in table}):
        for t in [t for t in table if t[1] == n]:
            lines.append(f"{t[0]:<14}{t[1]:>5}{t[2]:>7.2f}{t[3]:>6.2f}{t[4]:>6.2f}{t[5]:>6.2f}{t[6]:>7.3f}"
                         f"{t[7]:>6.2f}{t[8]:>6.2f}{t[9]:>6.1f}{t[10]:>5}")
        lines.append("")
    txt = "\n".join(lines)
    open(os.path.join(a.out, "mscr_multi_eval.txt"), "w").write(txt)
    print(txt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="op", required=True)
    s = sp.add_parser("stats")
    s.add_argument("--cache", required=True)
    s.add_argument("--small", required=True)
    s.add_argument("--B", type=int, default=9999)
    s.add_argument("--out", default=None)
    lo = sp.add_parser("local")
    lo.add_argument("--npz", required=True)
    lo.add_argument("--B", type=int, default=999)
    lo.add_argument("--out", required=True)
    e = sp.add_parser("eval")
    e.add_argument("--stats", required=True)
    e.add_argument("--gtref", required=True)
    e.add_argument("--procs", default="")
    e.add_argument("--q", type=float, default=0.05)
    e.add_argument("--floor", type=float, default=0.2)
    e.add_argument("--thr", type=float, default=3.0)
    e.add_argument("--out", default=".tmp/mscr_plus/M")
    a = ap.parse_args(argv)
    np.seterr(all="ignore")
    {"stats": cmd_stats, "local": cmd_local, "eval": cmd_eval}[a.op](a)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    main()
