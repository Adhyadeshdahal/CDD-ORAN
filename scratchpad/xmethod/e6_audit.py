"""Experiment D (ruling R-15): E6 validity audit of PMRT on the EXISTING v4 data. Descriptive, post-hoc, disclosed;
the frozen v4 verdict is unchanged. Runs on Kaggle with the v4 collection kernels as --sources.

  python scratchpad/xmethod/e6_audit.py --eval SPEC --placebo SPEC --gt SPEC --ref-analysis analysis_v4.json
         --out DIR [--B 9999] [--workers 4]

Data and loaders exactly as the v4 analyzer (scratchpad/e6_dev/e6p_disc_analyze_v4.py): expand / build_caches /
load_pmrt_pool (unit window H 90, H_pre 90), slice_plan (pooled split 0; slice60 i split 20 + i; slice120 k split
40 + k), placebo logs split 9, GT = disc_bench.gt_reference_files over the gt_v4 episodes (frozen rule). Weights =
the artifact docs/benchmark/artifacts/E6P_PMRT_V4.json (read only), cfg = pmrt_artifacts.pmrt_config(params, B).
Statistic = pmrt_bench.integrated_family's arithmetic (lean_columns weights, the SAME RNG stream
default_rng([0, 6616, 3, family_idx, split]) and the same exceedance counting) for the arms
  loadsp_c   the frozen primary arm (predictable Huber clip)
  loadsp     the same kernel and running centre WITHOUT the clip (w = r)
  plain      only for the kernel-arm sign fallback, as in the frozen code.
Per hypothesis: two-sided p2, one-sided p_plus / p_minus (effect direction o x z), and p_used = the p the frozen
primary layer wby1s uses (one-sided in the frozen ev2 prior direction where |z_prior| >= 3, else p2; the loadsp_c prior
directions are used for loadsp too: the artifact holds no loadsp prior).
Check: the recomputed loadsp_c p2 / p_plus / p_minus / used p must equal the stored v4 hyp tables (--ref-analysis).
Summary: rejection rates at .05 / .01 over the hypotheses the v4 GT labels NULL (status "NULL"; INDET and TRUE
excluded), per slice kind (60, 120, pooled), with a cluster bootstrap over the slices of that kind (cluster =
episode slice; pooled = one cluster: Wilson interval over hypotheses given for reference only); placebo logs: every
tested hypothesis (all null by construction), one cluster.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (ROOT, os.path.join(ROOT, "scratchpad", "e6_dev")):
    if p not in sys.path:
        sys.path.insert(0, p)

import e6p_disc_analyze_v4 as AZ  # noqa: E402
import numpy as np  # noqa: E402
import pmrt_artifacts as PAR  # noqa: E402
import pmrt_bench as PB  # noqa: E402

from cdd_oran.decision import disc_bench as DB  # noqa: E402
from cdd_oran.decision import fdr_layer as FL  # noqa: E402
from cdd_oran.decision import pmrt as PM  # noqa: E402
from cdd_oran.decision.crt_units import FAMILIES, LEVEL_ARR, PiAssignment  # noqa: E402
from cdd_oran.decision.crt_units_v2 import LEVEL_V2_ARR  # noqa: E402

ART = os.path.join(ROOT, "docs", "benchmark", "artifacts", "E6P_PMRT_V4.json")
ARMS = ("plain", "loadsp_c", "loadsp")
REPORT = ("loadsp_c", "loadsp")
THR = 3.0
_G: dict = {}


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def audit_family(pd, f, params, cfg, split) -> dict:
    """{(rel, kpi): {"status", arm: {z, p2, p_plus, p_minus, sign, o}}}: integrated_family's arithmetic per arm."""
    ud = pd.ud
    fi = FAMILIES.index(f)
    rows0 = ud.rows_of(f)
    lv = LEVEL_ARR[ud.mode[rows0]]
    targets = [(r, k) for r in ud.relations for k in ud.kpis]
    if (len(rows0) < cfg.min_units or (lv == 1.0).sum() < cfg.min_accept or (lv == 0.0).sum() < cfg.min_reject
            or len(np.unique(ud.episode[rows0])) < cfg.min_episodes):
        return {t: {"status": "undetermined", "reason": "support"} for t in targets}
    rows, W, targets, meta = PB.lean_columns(pd, f, params, cfg, arms=ARMS)
    na, nt = len(ARMS), len(targets)
    v = PM.v_design(ud.mode[rows], ud.probs[rows], ud.sgn[rows])
    var = PM.v_var(ud.probs[rows])
    sd = np.sqrt(var @ (W * W))
    sd_safe = np.where(sd > 0, sd, 1.0)
    z_obs = (v @ W) / sd_safe
    ORI = np.array([[(meta[t]["o"][a] if meta[t] else 0.0) for a in ARMS] for t in range(nt)]).ravel()
    zo_obs = ORI * z_obs
    tol = 1e-9 * (1.0 + np.abs(z_obs))
    pa = PiAssignment(ud)
    rng = PM._stream(cfg, fi, split)
    ch = int(max(1, min(cfg.chunk, cfg.max_chunk_bytes // (8 * max(len(rows), 1)))))
    c2, cp, cm = (np.zeros(W.shape[1], np.int64) for _ in range(3))
    mu = ud.probs[rows] @ LEVEL_V2_ARR
    sgv = ud.sgn[rows]
    done = 0
    while done < cfg.B:
        b = min(ch, cfg.B - done)
        M = pa.conditional_draws(rng, f, b, rows=rows, what="mode")
        V = LEVEL_V2_ARR[M]
        del M
        V -= mu[None, :]
        V *= sgv[None, :]
        Zb = (V @ W) / sd_safe[None, :]
        del V
        c2 += np.count_nonzero(np.abs(Zb) >= (np.abs(z_obs) - tol)[None, :], axis=0)
        Zo = Zb * ORI[None, :]
        cp += np.count_nonzero(Zo >= (zo_obs - tol)[None, :], axis=0)
        cm += np.count_nonzero(Zo <= (zo_obs + tol)[None, :], axis=0)
        done += b
    B1 = cfg.B + 1.0
    ai = {a: i for i, a in enumerate(ARMS)}
    out = {}
    for t, (rel, kpi) in enumerate(targets):
        if meta[t] is None:
            out[(rel, kpi)] = {"status": "undetermined", "reason": "degenerate_outcome"}
            continue
        o = {"status": "tested"}
        plain_sign = float(np.sign(z_obs[t * na + ai["plain"]]))
        for a in REPORT:
            j = t * na + ai[a]
            oa = ORI[j]
            sgn_ = float(np.sign(z_obs[j])) * oa if oa else plain_sign
            o[a] = {"p2": (1 + c2[j]) / B1, "p_plus": (1 + cp[j]) / B1 if oa else float("nan"),
                    "p_minus": (1 + cm[j]) / B1 if oa else float("nan"), "z": float(abs(z_obs[j]) * (sgn_ or 1.0)),
                    "z_raw": float(z_obs[j]), "sign": int(sgn_), "o": float(oa)}
        out[(rel, kpi)] = o
    return out


def p_used(arm: dict, d: int) -> float:
    """fdr_layer._vectors: one-sided in the prior direction d when available, else two-sided."""
    v = {1: arm["p_plus"], -1: arm["p_minus"]}.get(d, arm["p2"])
    return float(arm["p2"]) if (d and not np.isfinite(v)) else float(v)


def run_task(task):
    name, kind, split, episodes = task
    t = time.time()
    params, cfg, dirs = _G["params"], _G["cfg"], _G["dirs"]
    pool = _G["ppool"] if kind == "placebo" else _G["pool"]
    pd = PM.pmrt_data(pool) if kind == "placebo" else PM.pmrt_data(pool, episodes)
    res = {}
    for f in FAMILIES:
        for (rel, kpi), o in audit_family(pd, f, params, cfg, split).items():
            h = (f, rel, kpi)
            if o["status"] == "tested":
                for a in REPORT:
                    o[a]["p_used"] = p_used(o[a], dirs.get(h, 0))
            res["|".join(h)] = o
    log(f"{name}: {pd.n} units, {time.time() - t:.0f} s")
    return name, kind, split, int(len(episodes)) if episodes is not None else int(pool.n_eps), res


def _num(x) -> float:
    return float("nan") if x is None else float(x)


def wilson(x, n, z=1.96):
    if n == 0:
        return [float("nan")] * 2
    ph, d = x / n, 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def rates(slices: list, nulls: set, arm: str, pkey: str, thr: float, reps: int = 4000) -> dict:
    hits, cnt = [], []
    for s in slices:
        ps = [s["res"][h][arm][pkey] for h in s["res"] if s["res"][h]["status"] == "tested"
              and (nulls is None or h in nulls)]
        ps = [p for p in ps if np.isfinite(p)]
        hits.append(sum(p <= thr for p in ps))
        cnt.append(len(ps))
    hits, cnt = np.array(hits, float), np.array(cnt, float)
    rate = float(hits.sum() / max(cnt.sum(), 1))
    out = {"rate": rate, "x": int(hits.sum()), "m": int(cnt.sum()), "n_clusters": len(slices)}
    if len(slices) > 1:
        rng = np.random.default_rng([7801, 15, int(thr * 1000)])
        bs = [hits[i].sum() / max(cnt[i].sum(), 1) for i in (rng.integers(0, len(hits), len(hits)) for _ in range(reps))]
        out["ci_cluster"] = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
    else:
        out["ci_cluster"] = None
    out["ci_wilson_hyp"] = wilson(int(hits.sum()), int(cnt.sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--placebo", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--ref-analysis", dest="ref", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=9999)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--kinds", default="placebo,60,120,pooled")
    ap.add_argument("--stages", default="eval,placebo", help="cache stages (dry run on other data only)")
    a = ap.parse_args()
    t_all = time.time()
    os.makedirs(a.out, exist_ok=True)
    params, priors = PAR.load_artifact(ART)
    cfg = PAR.pmrt_config(params, a.B)
    dirs = FL.prior_directions(priors["loadsp_c"], THR)
    cache_dir = os.path.join(a.out, "cache")
    files = {k: AZ.expand(getattr(a, k)) for k in ("eval", "placebo")}
    st_e, st_p = a.stages.split(",")
    caches = {k: AZ.build_caches(files[k], st, os.path.join(cache_dir, k), k, a.workers)
              for k, st in (("eval", st_e), ("placebo", st_p))}
    pool = PM.load_pmrt_pool(caches["eval"], stages={st_e})
    ppool = PM.load_pmrt_pool(caches["placebo"], stages={st_p})
    g = DB.load_ref(a.gt) if a.gt.endswith(".json") else DB.gt_reference_files(AZ.expand(a.gt))
    status = {(c["family"], c["relation"], c["kpi"]): c["status"] for c in g["cells"]}
    nulls = {"|".join(h) for h, s in status.items() if s == "NULL"}
    log(f"GT counts {g['counts']} (v4: TRUE 29, NULL 21, INDET 10); eval eps {pool.n_eps}, placebo eps {ppool.n_eps}")
    ev_seeds = sorted(int(s) for s in pool.eps["seed"])
    dry = st_e != "eval"
    plan = [sl for sl in AZ.slice_plan(ev_seeds, pseudo=dry, sizes=(5, 10, 300) if dry else (60, 120, 300))
            if sl["kind"] in ("pooled", "60", "120")]
    kinds = set(a.kinds.split(","))
    tasks = [("placebo", "placebo", AZ.SPLIT["placebo"], None)] if "placebo" in kinds else []
    tasks += [(sl["name"], sl["kind"], sl["split"], AZ.episodes_of(pool, sl["seeds"])) for sl in plan
              if sl["kind"] in kinds]
    tasks.sort(key=lambda x: x[1] != "pooled")              # the big pooled task first
    _G.update(params=params, cfg=cfg, dirs=dirs, pool=pool, ppool=ppool)
    done = []
    if a.workers > 1:
        from multiprocessing import get_context
        with get_context("fork").Pool(a.workers) as mp:
            done = list(mp.imap_unordered(run_task, tasks))
    else:
        done = [run_task(t) for t in tasks]
    slices = [{"name": n, "kind": k, "split": s, "n_eps": ne, "res": r} for n, k, s, ne, r in done]
    # ---- reproduction check against the stored frozen v4 tables
    check = None
    if a.ref and os.path.isfile(a.ref):
        ref = json.load(open(a.ref))
        tabs = {nm: v["hyp"] for nm, v in ref["slices_result"].items()}
        tabs["placebo"] = ref["placebo_hyp"]
        n_cmp, bad = 0, []
        for s in slices:
            if s["name"] not in tabs:
                continue
            for h, o in s["res"].items():
                r = tabs[s["name"]].get(h, {})
                if o["status"] != "tested" or r.get("status") != "tested":
                    if o["status"] != r.get("status"):
                        bad.append((s["name"], h, "status", o["status"], r.get("status")))
                    continue
                n_cmp += 1
                mine, theirs = o["loadsp_c"], r["loadsp_c"]
                for k in ("p2", "p_plus", "p_minus"):
                    x, y = _num(mine[k]), _num(theirs[k])
                    if not (x == y or (math.isnan(x) and math.isnan(y))):
                        bad.append((s["name"], h, k, x, y))
                pu = r["primary"].get("p")
                if pu is not None and np.isfinite(pu) and mine["p_used"] != pu:
                    bad.append((s["name"], h, "p_used", mine["p_used"], pu))
        check = {"n_hyp_compared": n_cmp, "n_mismatch": len(bad), "mismatches": bad[:20]}
        log(f"reproduction of the stored loadsp_c p-values: {n_cmp} hyp, {len(bad)} mismatches")
    # ---- summary
    summ = {}
    for kind in ("60", "120", "pooled", "placebo"):
        ss = [s for s in slices if s["kind"] == kind]
        if not ss:
            continue
        nl = None if kind == "placebo" else nulls
        summ[kind] = {arm: {pk: {f"{thr}": rates(ss, nl, arm, pk, thr) for thr in (.05, .01)}
                            for pk in ("p2", "p_used")} for arm in REPORT}
        for arm in REPORT:
            r5, r1 = summ[kind][arm]["p2"]["0.05"], summ[kind][arm]["p2"]["0.01"]
            log(f"{kind:>7} {arm:9s} p2: .05 {r5['rate']:.3f} ({r5['x']}/{r5['m']}) CI {r5['ci_cluster']} | "
                f".01 {r1['rate']:.3f} ({r1['x']}/{r1['m']}) CI {r1['ci_cluster']}")
    out = {"schema": "xm-e6-audit/1", "ruling": "R-15", "artifact": os.path.relpath(ART, ROOT).replace("\\", "/"),
           "B": a.B, "thr_prior": THR, "gt_counts": g["counts"], "null_hyps": sorted(nulls),
           "gt_status": {"|".join(h): s for h, s in status.items()}, "reproduction_check": check, "summary": summ,
           "slices": [{k: s[k] for k in ("name", "kind", "split", "n_eps")} for s in slices],
           "per_slice": {s["name"]: s["res"] for s in slices}, "wall_s": round(time.time() - t_all, 1)}
    with open(os.path.join(a.out, "e6_audit.json"), "w", newline="\n") as fh:
        json.dump(out, fh, indent=1, default=float)
    log(f"done [{out['wall_s']} s]")


if __name__ == "__main__":
    main()
