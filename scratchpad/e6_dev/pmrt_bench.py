"""INTEGRATED PMRT bench (agent I, 2026-09-30): statistic arms {plain_c, loadsp_c, max(plain_c, loadsp_c)}
(cdd_oran/decision/pmrt.py, params fitted by agent S on ev2 + DEV) x declaration layers {by, wby1s, dagger1s}
(cdd_oran/decision/fdr_layer.py) with the prior (weights / directions) recomputed from the SAME statistic on the 480 ev2
episodes. Pre-specification: .tmp/mscr_plus/I/PRESPEC.md (written before the first run). Formerly
``mscr_integrate_bench.py`` (method label "MSCR+"; the rename is label-only, docs/benchmark/METHOD_NAMES.md).

  python scratchpad/e6_dev/pmrt_bench.py bench --cache DIR --small DIR --params P.pkl --out DIR [--B 9999]
         [--sizes 60,120,300,1200] [--R 60:10,120:10] [--baselines v2,corr,granger,granger_by] [--max-subsets 0]
  python scratchpad/e6_dev/pmrt_bench.py null --pi0 F[,F] --params P.pkl --prior prior_ev2.json --out DIR
         [--group 60] [--shifts 4] [--B 999]
  python scratchpad/e6_dev/pmrt_bench.py smoke --small DIR --params P.pkl [--B 199]          (local, small)
  python scratchpad/e6_dev/pmrt_bench.py table --bench bench_int.json [--null null_int.json]   (local)

Statistic (``integrated_family``): the predictable weighted residual columns of the arms plain / plain_c / loadsp_c are
built by ``lean_columns`` (only the needed kernels; bit-identical to pmrt.family_columns, checked by
``smoke``); ONE set of B conditional re-draws of the family's modes (pmrt._stream: the same stream as
family_tests_pmrt, so the two-sided p equal agent S's) gives, per target and arm, the two-sided p (|z|), and the
one-sided p_plus / p_minus of the EFFECT direction: the arm's effect sign is sign(z) x o with o = +1 for plain-type
kernels and o = the training sign of the kernel's plain mean for loadsp (0 if |z_plain_train| < 2: no one-sided p,
the layer falls back to the two-sided p - valid). The max arm: two-sided max |z|, one-sided max of o z over the two
arms (NaN if an o is 0); its sign = the sign of the arm attaining the max |z|.
Prior: the same statistic on ev2 (480 eps, split 0): signed effect z per hypothesis -> fdr_layer.Prior(n_prior 480).
Subsets: EXACTLY agent S's (pmrt_arms_bench.subsets on the ev3 episodes: n 60 x 10, 120 x 10 random disjoint,
300 = the 4 v3 folds) + n 1200 = all of ev3 (split 0). Baselines on the same subsets: MSCR-CRT v2 (+ BY), corr /
granger (DEV-tuned tau) and granger_by (disc_bench.method_baseline). Placebo: placebo1 / plxc2 (split 9, n_target 20).
Null-outcome check (``null``): agent S's recipe (validity_check.variants: groups of G ev3 episodes, cyclic shifts of
other episodes' series = exact sharp null for every family), per variant every combination's declarations
(n_target = G) and the rate of the p-value each layer uses.
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
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from cdd_oran.decision import pmrt as PM  # noqa: E402
from cdd_oran.decision import crt_units_v2 as V2  # noqa: E402
from cdd_oran.decision import disc_bench as DB  # noqa: E402
from cdd_oran.decision import fdr_layer as FL  # noqa: E402
from cdd_oran.decision.crt_units import FAMILIES, LEVEL_ARR, PiAssignment  # noqa: E402
from cdd_oran.decision.crt_units_v2 import LEVEL_V2_ARR  # noqa: E402
from cdd_oran.decision.eprocess_units import unit_order  # noqa: E402

COL_ARMS = ("plain", "plain_c", "loadsp_c")          # "plain" only for the sign fallback of kernel arms
STATS = ("plain_c", "loadsp_c", "max")
MAX_OF = ("plain_c", "loadsp_c")
LAYERS = ("by", "dagger1s", "wby1s")
COMBOS = [(s, l) for s in STATS for l in LAYERS]
ORDER = [("plain_c", "by"), ("plain_c", "dagger1s"), ("loadsp_c", "by"), ("plain_c", "wby1s"),
         ("loadsp_c", "dagger1s"), ("loadsp_c", "wby1s"), ("max", "by"), ("max", "dagger1s"), ("max", "wby1s")]
N_PRIOR = 480
Q, FLOOR, THR = 0.05, 0.2, 3.0


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


# ================================================================================================ params
class _Stub:
    """Placeholder for unpicklable GBDT objects (sklearn / its Cython _loss module): the integrated arms never use the
    GBDT g-hat (arm best_gb), so the params load without sklearn version coupling."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        pass

    def __call__(self, *a, **k):
        return _Stub()


def load_params_nogb(path: str) -> dict:
    """pmrt.load_params without the GBDT objects (hp["gb"] = None)."""
    import pickle

    class U(pickle.Unpickler):
        def find_class(self, module, name):
            if module.split(".")[0] in ("sklearn", "_loss") or module.startswith("sklearn"):
                return _Stub
            return super().find_class(module, name)
    with open(path, "rb") as fh:
        params = U(fh).load()
    for hp in params["hyp"].values():
        hp["gb"] = None
    return params


# ================================================================================================ statistic
def arm_sdir(a, core) -> float:
    base = PM.split_arm(a)[0]
    if base in ("pred",) + PM.SIGN_FROM_S:
        return 1.0
    if abs(core["z_plain"]) < 2.0:
        return 0.0
    return float(core["sdir"][base] or 0.0)


def lean_columns(pd, f, params, cfg, arms=COL_ARMS):
    """(rows, W (n, n_targets * n_arms), targets, meta) like pmrt.family_columns restricted to ``arms``
    (no GBDT, only the needed kernels; params may be a frozen artifact holding only those)."""
    ud = pd.ud
    rows = unit_order(ud, ud.rows_of(f))
    rho = {g: np.array(params["rho"][g]) for g in params["rho"]}
    ep, sg = ud.episode[rows], ud.sgn[rows]
    targets = [(r, k) for r in ud.relations for k in ud.kpis]
    need = sorted({PM.split_arm(a)[0] for a in arms})
    cols, meta = [], []
    for rel, kpi in targets:
        hp = params["hyp"].get(f"{f}|{rel}|{kpi}")
        y = ud.y[(rel, kpi)][rows]
        if hp is None or not np.all(np.isfinite(y)):
            cols.append(np.zeros((len(rows), len(arms))))
            meta.append(None)
            continue
        core = hp["core"]
        P, X, okp = PM.hyp_design(pd, rows, f, rel, kpi, rho, params["ctx_keys"])
        E = np.nan_to_num(P) - PM._xs(core, X) @ core["B"]
        has3 = okp[:, 3]
        k = {nm: np.where(has3, E @ core["K"][nm][0], E @ core["K"][nm][1]) for nm in need}
        r = {nm: PM.running_center(k[nm], ep, sg, cfg.n0, cfg.n1,
                                   {1: [core["prior"][nm][1]], -1: [core["prior"][nm][-1]]}) for nm in need}
        W = {}
        for a in arms:
            base, mod = PM.split_arm(a)
            W[a] = PM.modified(r[base], mod, hp["h"][base], core, X, cfg)
        cols.append(np.column_stack([W[a] for a in arms]))
        meta.append({"o": {a: arm_sdir(a, core) for a in arms}, "z_plain_train": float(core["z_plain"])})
    return rows, np.column_stack(cols), targets, meta


def integrated_family(pd, f, params, cfg, split):
    """{(rel, kpi): {"status", stat: {"p2", "p_plus", "p_minus", "z", "sign", "via"?}}} for STATS (module docstring)."""
    ud = pd.ud
    fi = FAMILIES.index(f)
    rows0 = ud.rows_of(f)
    lv = LEVEL_ARR[ud.mode[rows0]]
    n_acc, n_rej = int((lv == 1.0).sum()), int((lv == 0.0).sum())
    n_ep = len(np.unique(ud.episode[rows0]))
    targets = [(r, k) for r in ud.relations for k in ud.kpis]
    if (len(rows0) < cfg.min_units or n_acc < cfg.min_accept or n_rej < cfg.min_reject
            or n_ep < cfg.min_episodes):
        return {t: {"status": "undetermined", "reason": "support"} for t in targets}
    rows, W, targets, meta = lean_columns(pd, f, params, cfg)
    na = len(COL_ARMS)
    nt = len(targets)
    v = PM.v_design(ud.mode[rows], ud.probs[rows], ud.sgn[rows])
    var = PM.v_var(ud.probs[rows])
    sd = np.sqrt(var @ (W * W))
    sd_safe = np.where(sd > 0, sd, 1.0)
    z_obs = (v @ W) / sd_safe
    O = np.array([[(meta[t]["o"][a] if meta[t] else 0.0) for a in COL_ARMS] for t in range(nt)]).ravel()
    zo_obs = O * z_obs                                               # oriented (effect-direction) z; 0 where o = 0
    ai = {a: i for i, a in enumerate(COL_ARMS)}
    mx = [ai[a] for a in MAX_OF]
    Z3 = z_obs.reshape(nt, na)
    O3 = O.reshape(nt, na)
    tmax_obs = np.abs(Z3[:, mx]).max(1)
    omax_ok = (O3[:, mx] != 0).all(1)
    pmax_obs = (O3[:, mx] * Z3[:, mx]).max(1)                        # one-sided "+" statistic of max
    mmax_obs = (-O3[:, mx] * Z3[:, mx]).max(1)                       # one-sided "-" statistic of max
    tol = 1e-9 * (1.0 + np.abs(z_obs))
    tolm = 1e-9 * (1.0 + tmax_obs)
    pa = PiAssignment(ud)
    rng = PM._stream(cfg, fi, split)
    ch = int(max(1, min(cfg.chunk, cfg.max_chunk_bytes // (8 * max(len(rows), 1)))))
    c2 = np.zeros(W.shape[1], np.int64)
    cp = np.zeros(W.shape[1], np.int64)
    cm = np.zeros(W.shape[1], np.int64)
    c2m, cpm, cmm = (np.zeros(nt, np.int64) for _ in range(3))
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
        Zo = Zb * O[None, :]
        cp += np.count_nonzero(Zo >= (zo_obs - tol)[None, :], axis=0)
        cm += np.count_nonzero(Zo <= (zo_obs + tol)[None, :], axis=0)
        Z3b = Zb.reshape(b, nt, na)[:, :, mx]
        c2m += np.count_nonzero(np.abs(Z3b).max(2) >= (tmax_obs - tolm)[None, :], axis=0)
        Ob = O3[None, :, :][:, :, mx]
        cpm += np.count_nonzero((Ob * Z3b).max(2) >= (pmax_obs - tolm)[None, :], axis=0)
        cmm += np.count_nonzero((-Ob * Z3b).max(2) >= (mmax_obs - tolm)[None, :], axis=0)
        done += b
    B1 = cfg.B + 1.0
    out = {}
    for t, (rel, kpi) in enumerate(targets):
        if meta[t] is None:
            out[(rel, kpi)] = {"status": "undetermined", "reason": "degenerate_outcome"}
            continue
        o = {"status": "tested", "z_plain_train": meta[t]["z_plain_train"]}
        plain_sign = float(np.sign(z_obs[t * na + ai["plain"]]))
        for a in MAX_OF:
            j = t * na + ai[a]
            oa = O[j]
            sgn_ = float(np.sign(z_obs[j])) * oa if oa else plain_sign
            o[a] = {"p2": (1 + c2[j]) / B1, "p_plus": (1 + cp[j]) / B1 if oa else float("nan"),
                    "p_minus": (1 + cm[j]) / B1 if oa else float("nan"), "z": float(abs(z_obs[j]) * (sgn_ or 1.0)),
                    "z_raw": float(z_obs[j]), "sign": int(sgn_), "o": oa}
        jb = MAX_OF[int(np.argmax(np.abs(Z3[t, mx])))]
        o["max"] = {"p2": (1 + c2m[t]) / B1, "p_plus": (1 + cpm[t]) / B1 if omax_ok[t] else float("nan"),
                    "p_minus": (1 + cmm[t]) / B1 if omax_ok[t] else float("nan"), "z": o[jb]["z"],
                    "sign": o[jb]["sign"], "via": jb}
        out[(rel, kpi)] = o
    return out


def run_integrated(pd, params, cfg, split) -> dict:
    """{"f|rel|kpi": per-target dict} over every family."""
    res = {}
    for f in FAMILIES:
        for (rel, kpi), o in integrated_family(pd, f, params, cfg, split).items():
            res["|".join((f, rel, kpi))] = o
    return res


def hypstats(run: dict, stat: str) -> dict:
    out = {}
    for h, o in run.items():
        key = tuple(h.split("|"))
        if o.get("status") != "tested":
            out[key] = FL.HypStats()
            continue
        s = o[stat]
        out[key] = FL.HypStats(p2=float(s["p2"]), sign=int(s["sign"]), p_plus=float(s["p_plus"]),
                               p_minus=float(s["p_minus"]))
    return out


def prior_from_run(run: dict, stat: str, n_prior: int = N_PRIOR) -> FL.Prior:
    d = {}
    for h, o in run.items():
        if o.get("status") != "tested":
            d[h] = {"status": "undetermined"}
        else:
            d[h] = {"z": float(o[stat]["z"]), "status": "tested"}
    return FL.Prior.from_stats(d, n_prior=n_prior, source=f"ev2 {stat}")


def declare_all(run: dict, priors: dict, n_target: int) -> dict:
    """{(stat, layer): decl dict in edge_score format}."""
    out = {}
    for s, lay in COMBOS:
        dec = FL.declare(hypstats(run, s), lay, priors[s], n_target=n_target, q=Q, floor=FLOOR, thr=THR)
        for h, v in dec.items():
            o = run.get("|".join(h), {})
            v["z_approx"] = o.get(s, {}).get("z") if o.get("status") == "tested" else None
        out[(s, lay)] = dec
    return out


def used_p(decl: dict) -> list:
    return [v["p"] for v in decl.values() if v.get("status") != "undetermined" and np.isfinite(v.get("p", np.nan))]


def cname(s, lay):
    return f"{s}+{lay}"


# ================================================================================================ helpers
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


def load_priors(path):
    d = json.load(open(path))
    return {s: FL.Prior.from_stats(d["z"][s], n_prior=d["n_prior"], source=f"ev2 {s}") for s in STATS}


def prior_json(run_ev2, n_eps):
    return {"n_prior": int(n_eps), "source": "integrated statistic on ev2 (sub v2), split 0",
            "z": {s: {h: ({"z": o[s]["z"], "status": "tested"} if o.get("status") == "tested"
                          else {"status": "undetermined"}) for h, o in run_ev2.items()} for s in STATS},
            "run": {h: o for h, o in run_ev2.items()}}


def agg(rows):
    out = {}
    for m in DB.METRICS:
        v = np.array([r[m] for r in rows], float)
        f = v[np.isfinite(v)]
        out[m] = round(float(f.mean()), 4) if len(f) else None
    return out


# ================================================================================================ bench
def cmd_bench(a):
    import pmrt_arms_bench as SB                                   # agent S's subset draw (identical subsets)
    os.makedirs(a.out, exist_ok=True)
    cfg = PM.PmrtConfig(B=a.B)
    files = expand(a.cache) + [os.path.join(a.small, "dev1.npz")]
    t = time.time()
    pool = PM.load_pmrt_pool(files, stages=None)
    log(f"pool {len(files)} files, {pool.n_eps} eps [{time.time() - t:.0f} s]")
    params = load_params_nogb(a.params)
    ev2, ev3 = SB.select(pool, "sub=v2"), SB.select(pool, "sub=v3")
    tr = SB.select(pool, "sub=v2,stage=dev")
    assert not set(tr.tolist()) & set(ev3.tolist())
    log(f"ev2 {len(ev2)} eps, ev3 {len(ev3)} eps")
    res = {"config": vars(a), "prespec": ".tmp/mscr_plus/I/PRESPEC.md", "sizes": {}, "placebo": {}}

    def dump():
        json.dump(json.loads(json.dumps(res, default=DB._js)), open(os.path.join(a.out, "bench_int.json"), "w"))

    # 1. prior on ev2 (the same statistic; only z / sign are used)
    t = time.time()
    if a.prior:
        pj = json.load(open(a.prior))
    else:
        pd2 = PM.pmrt_data(pool, ev2)
        run2 = run_integrated(pd2, params, PM.PmrtConfig(B=a.B_prior), 0)
        del pd2
        pj = json.loads(json.dumps(prior_json(run2, len(ev2)), default=DB._js))
        json.dump(pj, open(os.path.join(a.out, "prior_ev2.json"), "w"))
    if a.op == "prior":
        log(f"prior written [{time.time() - t:.0f} s]")
        return
    priors = {s: FL.Prior.from_stats(pj["z"][s], n_prior=pj["n_prior"]) for s in STATS}
    res["prior"] = {s: {"directions": {"|".join(h): d for h, d in FL.prior_directions(priors[s], THR).items() if d},
                        "weights": {str(n): {"|".join(h): round(w, 3) for h, w in
                                             FL.prior_weights(priors[s], n, Q, FLOOR).items()}
                                    for n in (20, 60, 120, 300, 600, 1200)}} for s in STATS}
    log(f"prior ev2 done [{time.time() - t:.0f} s]; directions: "
        + ", ".join(f"{s} {len(res['prior'][s]['directions'])}" for s in STATS)
        + f"; premise z " + ", ".join(f"{s} {priors[s].z.get(DB.PREMISE, 0):.2f}" for s in STATS))
    dump()
    bl = [b for b in a.baselines.split(",") if b]
    dev = DB.load_pool([os.path.join(a.small, "dev1.npz")], H=90, H_pre=90).unit_data()
    bmeth = {b: DB.method_baseline(b, dev) for b in bl if b != "v2"}
    g = DB.load_ref(os.path.join(a.small, "gtref_gt1_gtx3.json"))
    ref = g["ref"]
    # 2. placebo
    for nm in ("placebo1", "plxc2"):
        pdp = PM.pmrt_data(PM.load_pmrt_pool([os.path.join(a.small, f"{nm}.npz")]))
        run = run_integrated(pdp, params, cfg, 9)
        decl = declare_all(run, priors, 20)
        pr = {}
        for (s, lay), d in decl.items():
            ps = used_p(d)
            pr[cname(s, lay)] = {"n_declared": int(sum(v["declared"] for v in d.values())),
                                 "declared": sorted("|".join(h) for h, v in d.items() if v["declared"]),
                                 "rate05": round(float(np.mean(np.array(ps) <= .05)), 4), "min_p": float(min(ps))}
        if "v2" in bl:
            e = V2.edges_from_crt_v2(V2.run_crt_units_v2(pdp.ud, V2.UnitCRTConfigV2(B=a.B), 9))
            ps = [v["p"] for v in e.values() if np.isfinite(v["p"])]
            pr["v2+by"] = {"n_declared": int(sum(v["declared"] for v in e.values())),
                           "rate05": round(float(np.mean(np.array(ps) <= .05)), 4)}
        for b, m in bmeth.items():
            c = DB.placebo_check(m, pdp.ud)
            pr[b] = {"n_declared": c["n_declared"], "rate05": c["rate"]}
        res["placebo"][nm] = pr
        log(f"placebo {nm}: " + ", ".join(f"{k} {v['n_declared']}/{v.get('rate05')}" for k, v in pr.items()))
        dump()
    # 3. subsets
    Rm = {int(k): int(v) for k, v in (x.split(":") for x in a.R.split(","))}
    for n in [int(x) for x in a.sizes.split(",")]:
        if n >= len(ev3):
            subs = [{"name": "v3all", "episodes": ev3, "split": 0}]
        else:
            subs = SB.subsets(pool, ev3, n, Rm.get(n, 10))
        if a.max_subsets:
            subs = subs[: a.max_subsets]
        rows = {}
        for s in subs:
            t = time.time()
            pd = PM.pmrt_data(pool, s["episodes"])
            run = run_integrated(pd, params, cfg, s["split"])
            decls = {cname(*k): v for k, v in declare_all(run, priors, n).items()}
            if "v2" in bl:
                decls["v2+by"] = V2.edges_from_crt_v2(V2.run_crt_units_v2(pd.ud, V2.UnitCRTConfigV2(B=a.B),
                                                                          s["split"]))
            for b, m in bmeth.items():
                decls[b] = m(pd.ud, s["split"])
            for arm, d in decls.items():
                m = DB.subset_metrics(d, ref)
                m.update(name=s["name"], units=int(pd.n),
                         declared=sorted(("|".join(h), int(v.get("sign", 0))) for h, v in d.items()
                                         if v and v.get("declared")))
                rows.setdefault(arm, []).append(m)
            pz = {st: round(float(run["|".join(DB.PREMISE)][st]["z"]), 2) for st in STATS
                  if run["|".join(DB.PREMISE)].get("status") == "tested"}
            log(f"n={n} {s['name']}: units {pd.n} ({time.time() - t:.0f} s) premise z {pz} | " + " ".join(
                f"{k}:{rows[k][-1]['chain_hits']}/{rows[k][-1]['ind_f1']:.2f}/{rows[k][-1]['n_declared']}"
                for k in rows))
            del pd
            res["sizes"][str(n)] = {"R": len(rows[next(iter(rows))]),
                                    "summary": {k: agg(r) for k, r in rows.items()}, "subsets": rows}
            dump()
        summ = res["sizes"][str(n)]["summary"]
        for k in sorted(summ, key=lambda k: -(summ[k]["ind_f1"] or 0)):
            m = summ[k]
            log(f"  n={n} {k:>18}: chain {m['chain_hits']:.2f} prem {m['premise_hit']:.2f} indR {m['ind_recall']:.2f} "
                f"indP {m['ind_precision']} indF1 {m['ind_f1']:.3f} ovP {m['ov_precision']} sign {m['sign_acc']} "
                f"#dec {m['n_declared']:.1f}")
    dump()
    log("bench done")


# ================================================================================================ null check
def cmd_null(a):
    sys.path.insert(0, os.path.join(ROOT, ".tmp", "mscr_plus", "V"))
    import tempfile

    import validity_check as VC
    os.makedirs(a.out, exist_ok=True)
    params = load_params_nogb(a.params)
    priors = load_priors(a.prior)
    cfg = PM.PmrtConfig(B=a.B)
    recs = VC.load(a.pi0, set(a.stages.split(",")))
    G = a.group
    n_g = len(recs) // G
    if a.max_groups:
        n_g = min(n_g, a.max_groups)
    log(f"[null] {len(recs)} pi0 episodes, {n_g} groups of {G}, {a.shifts} shifts, B {a.B}")
    tmp = tempfile.mkdtemp(prefix="nullint_")
    per = {cname(s, lay): {"n_var_decl": 0, "n_decl": 0, "p": [], "fam": []} for s, lay in COMBOS}
    per["v2+by"] = {"n_var_decl": 0, "n_decl": 0, "p": [], "fam": []}
    nv = 0
    for g in range(n_g):
        sub = recs[g * G:(g + 1) * G]
        for name, rr in VC.variants(sub, [], a.shifts, 0):
            t = time.time()
            pd = PM.pmrt_data_from_records(rr, tmp)
            run = run_integrated(pd, params, cfg, 0)
            for (s, lay), d in declare_all(run, priors, G).items():
                o = per[cname(s, lay)]
                k = int(sum(v["declared"] for v in d.values()))
                o["n_decl"] += k
                o["n_var_decl"] += int(k > 0)
                for h, v in d.items():
                    if v.get("status") != "undetermined" and np.isfinite(v.get("p", np.nan)):
                        o["p"].append(float(v["p"]))
                        o["fam"].append(h[0])
            v2 = V2.run_crt_units_v2(pd.ud, V2.UnitCRTConfigV2(B=a.B))
            o = per["v2+by"]
            o["n_decl"] += int(v2["n_declared"])
            o["n_var_decl"] += int(v2["n_declared"] > 0)
            for r in v2["results"]:
                if r["status"] != "undetermined":
                    o["p"].append(float(r["p"]))
                    o["fam"].append(r["family"])
            nv += 1
            log(f"[g{g} {name}] units {pd.n} ({time.time() - t:.0f} s) decl " + " ".join(
                f"{k}:{v['n_decl']}" for k, v in per.items()))
            del pd, rr
    out = {"n_variants": nv, "G": G, "shifts": a.shifts, "B": a.B, "combos": {}}
    for k, o in per.items():
        P, F = np.array(o["p"]), np.array(o["fam"])
        out["combos"][k] = {"n_variants_with_decl": o["n_var_decl"], "n_decl": o["n_decl"], "n_p": int(len(P)),
                            "rate05": round(float(np.mean(P <= .05)), 4) if len(P) else None,
                            "rate01": round(float(np.mean(P <= .01)), 4) if len(P) else None,
                            "rate05_family": {f: round(float(np.mean(P[F == f] <= .05)), 4) for f in sorted(set(F))}}
    json.dump(out, open(os.path.join(a.out, "null_int.json"), "w"), indent=1)
    log(json.dumps(out, indent=1))
    log("null done")


# ================================================================================================ smoke (local)
def cmd_smoke(a):
    params = load_params_nogb(a.params)
    cfg = PM.PmrtConfig(B=a.B)
    pdd = PM.pmrt_data(PM.load_pmrt_pool([os.path.join(a.small, "dev1.npz")]))
    run_d = run_integrated(pdd, params, cfg, 0)
    priors = {s: prior_from_run(run_d, s, 20) for s in STATS}
    for nm in ("placebo1", "plxc2"):
        pdp = PM.pmrt_data(PM.load_pmrt_pool([os.path.join(a.small, f"{nm}.npz")]))
        t = time.time()
        run = run_integrated(pdp, params, cfg, 9)
        log(f"{nm}: integrated {time.time() - t:.1f} s")
        ref = PM.run_pmrt(pdp, params, cfg, 9)
        for r in ref["results"]:
            if r["status"] != "tested":
                continue
            o = run["|".join((r["family"], r["relation"], r["kpi"]))]
            for s in MAX_OF:
                assert abs(o[s]["p2"] - r["arms"][s]["p"]) < 1e-12, (nm, r["family"], r["relation"], r["kpi"], s)
                assert o[s]["sign"] == r["arms"][s]["sign"], ("sign", r["family"], r["relation"], r["kpi"], s)
        log(f"{nm}: two-sided p / sign of plain_c, loadsp_c identical to pmrt.family_tests_pmrt")
        for (s, lay), d in declare_all(run, priors, 20).items():
            ps = used_p(d)
            log(f"  {cname(s, lay):>18}: declared {sum(v['declared'] for v in d.values())} rate05 "
                f"{np.mean(np.array(ps) <= .05):.3f}")
    # the one-sided p's are valid only if p_plus + p_minus ~ 1 + 1/(B+1) (no ties): sanity
    o = run["|".join(DB.PREMISE)]
    log(f"premise plxc2: {o}")


# ================================================================================================ table (local)
def cmd_table(a):
    R = json.load(open(a.bench))
    N = json.load(open(a.null)) if a.null else None
    lines = [f"{'method':<20}{'n':>5}{'chain':>7}{'prem':>6}{'indR':>6}{'indP':>6}{'indF1':>7}{'ovP':>6}{'sign':>6}"
             f"{'#dec':>6}{'plc':>5}"]
    for n, blk in R["sizes"].items():
        for k in [cname(s, lay) for s, lay in ORDER] + ["v2+by", "corr", "granger", "granger_by"]:
            if k not in blk["summary"]:
                continue
            m = blk["summary"][k]
            plc = sum(R["placebo"][p].get(k, {}).get("n_declared", 0) for p in R["placebo"])
            f = lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else "  - "  # noqa: E731
            lines.append(f"{k:<20}{n:>5}{m['chain_hits']:>7.2f}{m['premise_hit']:>6.2f}{m['ind_recall']:>6.2f}"
                         f"{f(m['ind_precision']):>6}{m['ind_f1']:>7.3f}{f(m['ov_precision']):>6}"
                         f"{f(m['sign_acc']):>6}{m['n_declared']:>6.1f}{plc:>5}")
        lines.append("")
    # pre-specified selection
    sizes = [s for s in ("60", "120", "300") if s in R["sizes"]]
    f1 = {}
    for s, lay in COMBOS:
        k = cname(s, lay)
        f1[k] = {n: np.array([r["ind_f1"] for r in R["sizes"][n]["subsets"][k]], float) for n in sizes}
    elig = {}
    for s, lay in COMBOS:
        k = cname(s, lay)
        plc = sum(R["placebo"][p][k]["n_declared"] for p in R["placebo"])
        ok = plc <= 1
        if N:
            c = N["combos"][k]
            ok = ok and c["rate05"] <= .075 and max(c["rate05_family"].values()) <= .10 and c["n_variants_with_decl"] <= 5
        elig[k] = ok
    score = {k: float(np.mean([np.nanmean(f1[k][n]) for n in sizes])) for k in f1}
    best = max((k for k in score if elig[k]), key=score.get, default=None)
    lines.append("selection (PRESPEC): score = mean over n of mean ind F1; eligible / within-noise")
    rec = None
    for s, lay in ORDER:
        k = cname(s, lay)
        if best is None:
            break
        d = [f1[best][n] - f1[k][n] for n in sizes]
        D = float(np.mean([np.nanmean(x) for x in d]))
        se = float(np.sqrt(sum((1 / len(sizes)) ** 2 * np.nanvar(x, ddof=1) / len(x) for x in d if len(x) > 1)))
        p120 = lambda kk: float(np.mean([r["premise_hit"] for r in R["sizes"]["120"]["subsets"][kk]]))  # noqa: E731
        within = D <= max(.02, se) and p120(k) >= p120(best) - .2
        lines.append(f"  {k:<20} score {score[k]:.3f} eligible {elig[k]} D {D:+.3f} SE {se:.3f} prem120 "
                     f"{p120(k):.2f} within {within}")
        if rec is None and elig[k] and within:
            rec = k
    lines.append(f"best eligible {best}; RECOMMENDED (first in simplicity order within noise): {rec}")
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="op", required=True)
    for op in ("bench", "prior"):
        b = sp.add_parser(op)
        b.add_argument("--cache", required=True)
        b.add_argument("--small", required=True)
        b.add_argument("--params", required=True)
        b.add_argument("--out", default=os.environ.get("JOB_OUT", "."))
        b.add_argument("--prior", default="", help="prior_ev2.json (skip the ev2 run)")
        b.add_argument("--B", type=int, default=9999)
        b.add_argument("--B-prior", type=int, default=999)
        b.add_argument("--sizes", default="60,120,300,1200")
        b.add_argument("--R", default="60:10,120:10")
        b.add_argument("--baselines", default="v2,corr,granger,granger_by")
        b.add_argument("--max-subsets", type=int, default=0)
    n = sp.add_parser("null")
    n.add_argument("--pi0", required=True)
    n.add_argument("--params", required=True)
    n.add_argument("--prior", required=True)
    n.add_argument("--out", default=os.environ.get("JOB_OUT", "."))
    n.add_argument("--stages", default="eval")
    n.add_argument("--group", type=int, default=60)
    n.add_argument("--shifts", type=int, default=4)
    n.add_argument("--max-groups", type=int, default=0)
    n.add_argument("--B", type=int, default=999)
    s = sp.add_parser("smoke")
    s.add_argument("--small", required=True)
    s.add_argument("--params", required=True)
    s.add_argument("--B", type=int, default=199)
    t = sp.add_parser("table")
    t.add_argument("--bench", required=True)
    t.add_argument("--null", default="")
    t.add_argument("--out", default="")
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    {"bench": cmd_bench, "prior": cmd_bench, "null": cmd_null, "smoke": cmd_smoke, "table": cmd_table}[a.op](a)


if __name__ == "__main__":
    main()
