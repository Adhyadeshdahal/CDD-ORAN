"""R-60 post hoc diagnostics X5 / X6 (scratchpad/xmethod/EXTRAS_PROTOCOL.md, Amendments 2026-10-05): analysis.

    uv run python -m cdd_oran.xmethod.diag x5-step1 --eval-merged EVAL.jsonl.gz --out DIR
    uv run python -m cdd_oran.xmethod.diag x5 --fail FAIL.jsonl.gz --adj ADJ.jsonl.gz --out DIR
    uv run python -m cdd_oran.xmethod.diag x6 --merged X6.jsonl.gz --out DIR
    uv run python -m cdd_oran.xmethod.diag tables --out DIR          # x5_step1 / x5 / x6 json -> diag_tables.json + md

EXPLORATORY. Every rate, CI and label comes from the frozen eval_analysis (screen, cells, cluster bootstrap, R-30
class); the extra statistics are the ones declared in the X5 / X6 entries. EVAL records are read only (X5 step 1).
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy import stats

from cdd_oran.xmethod import extras as X
from cdd_oran.xmethod.worlds import generate as G

ALPHA = 0.05
VALID_UB = 0.075             # R-30 VALID bound (eval_analysis.VALID_UB)
EVAL_SPEC_REL = "scratchpad/xmethod/specs/eval/full.json"
E4R3_LAMS = (0.0, 0.5, 1.0, 1.5)
E4R3_NS = (500, 1000, 4000, 8000, 24000)
QQ_LEVELS = (0.01, 0.025, 0.05, 0.10, 0.20, 0.50)
TAIL_LEVELS = (0.01, 0.025, 0.05, 0.10)
KIND_EDGE = {"plac": "P_placebo->K0", "conf": "P_placebo_conf->K0"}
LABEL_NSIM, LABEL_SEED = 10_000, 20261005
TOP = ("x5_step1", "x5", "x6")


def _E():
    return X._eval_analysis()


def _f(x, d=3):
    return "NA" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def cp_ci(k: int, n: int) -> list[float]:
    """Two-sided 95 % Clopper-Pearson interval."""
    lo = 0.0 if k == 0 else float(stats.beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        out[i] = run
    return out


# ================================================================================================ record access
def screened(merged: str, spec: dict, keep=None) -> tuple[list[dict], dict, dict, dict]:
    """(used records, screen info, cells, seed rows) under the frozen eval_analysis rules; ``keep`` filters the
    records before screening (then only the kept arms / cells are expected to be present)."""
    E = _E()
    if keep is None:
        recs = E.load_records(merged)
    else:                                       # streamed (the EVAL file does not fit in memory as records)
        recs = []
        with (gzip.open if merged.endswith(".gz") else open)(merged, "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line) if line.strip() else None
                except json.JSONDecodeError:
                    continue
                if r is not None and keep(r):
                    recs.append(r)
        recs.sort(key=lambda r: (r.get("key", ""), json.dumps(r, sort_keys=True)))     # = eval_analysis order
    use, scr = E.screen(recs, spec)
    cells, rows = E.build_cells(use, spec)
    return use, scr, cells, rows


def e4r3_pvals(use: list[dict], arms) -> dict:
    """{(arm, lam, n): {seed: {kind: p}}} of the measurement-role ok records (raw edge p, as eval_analysis)."""
    out: dict = defaultdict(dict)
    for r in use:
        j = r["job"]
        if (r.get("status") or "ok") != "ok" or (r.get("role") or "measure") != "measure":
            continue
        arm = r.get("arm") or r["key"].split("|")[0]
        if arm not in arms or j["world"] != "E4" or j["regime"] != "R3":
            continue
        ps = {}
        for e in r.get("edges", []):
            name = f"{e['source']}->{e['target']}"
            for kind, en in KIND_EDGE.items():
                if name == en and e.get("p") is not None:
                    ps[kind] = float(e["p"])
        out[(arm, float(j["lam"]), int(j["n"]))][int(j["seed"])] = ps
    return dict(out)


def _cell(cells: dict, arm: str, lam: float, n: int) -> dict | None:
    for e in cells.values():
        if e["arm"] == arm and e["world"] == "E4" and e["regime"] == "R3" and e["lam"] == lam and e["n"] == n \
                and e["kappa"] == 0.25:
            return e
    return None


# ================================================================================================ X5 step 1
def label_prob(S: int, nsim: int = LABEL_NSIM, seed: int = LABEL_SEED) -> dict:
    """P(INVALID label) for one rate at exactly .05 over S independent datasets (one candidate each), under the frozen
    seed-cluster bootstrap (same draws as eval_analysis.cluster_ci); also P(VALID)."""
    E = _E()
    m = E._boot_mult(S)                                                 # reps x S, the frozen bootstrap draws
    rng = np.random.default_rng(seed)
    h = (rng.random((nsim, S)) < ALPHA).astype(float)
    bs = (h @ m.T) / S
    lo, hi = np.quantile(bs, 0.025, axis=1), np.quantile(bs, 0.975, axis=1)
    inv = lo > E.ALPHA
    val = (hi <= E.VALID_UB) & ~inv
    return {"S": S, "nsim": nsim, "seed": seed, "p_invalid": float(inv.mean()), "p_valid": float(val.mean()),
            "min_hits_invalid": int(h.sum(1)[inv].min()) if inv.any() else None}


def x5_step1(eval_merged: str) -> dict:
    spec = json.load(open(X._path(EVAL_SPEC_REL), encoding="utf-8"))
    arms = X.X5_ARMS

    def keep(r):
        j = r.get("job") or {}
        return (r.get("arm") or r.get("key", "").split("|")[0]) in arms and j.get("world") == "E4" \
            and j.get("regime") == "R3"
    use, scr, cells, _ = screened(eval_merged, spec, keep)
    pv = e4r3_pvals(use, arms)
    out = {"source": os.path.basename(eval_merged), "records_used": len(use), "cells": [], "overlap_lambda": [],
           "label_count": {}}
    S_all = set()
    for arm in arms:
        for n in E4R3_NS:
            for lam in E4R3_LAMS:
                d = pv.get((arm, lam, n), {})
                e = _cell(cells, arm, lam, n) or {}
                prim = e.get("primary") or {}
                row = {"arm": arm, "lam": lam, "n": n, "seeds": len(d)}
                S_all.add(len(d))
                rej = {}
                for kind in KIND_EDGE:
                    ps = np.array(sorted(x[kind] for x in d.values() if kind in x))
                    fr = prim.get(f"{kind}_raw") or {}
                    ks = stats.kstest(ps, "uniform") if len(ps) else None
                    rej[kind] = {s for s, x in d.items() if x.get(kind, 1.0) <= ALPHA}
                    row[kind] = {
                        "n": int(len(ps)), "frozen_rate": fr.get("rate"), "frozen_ci": fr.get("ci"),
                        "frozen_label": fr.get("validity"),
                        "ks_d": None if ks is None else float(ks.statistic),
                        "ks_p": None if ks is None else float(ks.pvalue),
                        "qq": {str(q): float(np.quantile(ps, q)) for q in QQ_LEVELS} if len(ps) else None,
                        "tail": {str(a): {"rate": float(np.mean(ps <= a)), "obs_exp": float(np.mean(ps <= a) / a)}
                                 for a in TAIL_LEVELS} if len(ps) else None}
                both = len(rej["plac"] & rej["conf"])
                N = len(d)
                exp = len(rej["plac"]) * len(rej["conf"]) / N if N else None
                row["overlap_plac_conf"] = {
                    "both": both, "plac": len(rej["plac"]), "conf": len(rej["conf"]), "expected": exp,
                    "p_upper": float(stats.hypergeom.sf(both - 1, N, len(rej["plac"]), len(rej["conf"])))
                    if N else None}
                row["failing"] = any((lam, n, f"{k}_raw") in X.X5_PRIMARY for k in KIND_EDGE) and arm == "pmrt_eq"
                out["cells"].append(row)
            # (c) across lambdas at the same n: seeds rejecting in >= 2 of the 4 lambdas, vs independence
            for kind in KIND_EDGE:
                sets = {lam: {s for s, x in pv.get((arm, lam, n), {}).items() if x.get(kind, 1.0) <= ALPHA}
                        for lam in E4R3_LAMS}
                seeds = set().union(*(set(pv.get((arm, lam, n), {})) for lam in E4R3_LAMS))
                N = len(seeds)
                if not N:
                    continue
                cnt = np.array([sum(s in sets[lam] for lam in E4R3_LAMS) for s in sorted(seeds)])
                ps_l = [len(sets[lam]) / N for lam in E4R3_LAMS]
                # P(>= 2 of 4) under independence with the observed per-lambda rates (Poisson-binomial)
                dist = np.array([1.0])
                for p in ps_l:
                    dist = np.convolve(dist, [1 - p, p])
                pairs = []
                for i, a in enumerate(E4R3_LAMS):
                    for b in E4R3_LAMS[i + 1:]:
                        o = len(sets[a] & sets[b])
                        pairs.append({"lams": [a, b], "both": o, "expected": len(sets[a]) * len(sets[b]) / N})
                out["overlap_lambda"].append({
                    "arm": arm, "n": n, "kind": kind, "seeds": N, "per_lambda_hits": [len(sets[lam]) for lam in E4R3_LAMS],
                    "seeds_ge2": int((cnt >= 2).sum()), "expected_ge2": float(N * dist[2:].sum()),
                    "seeds_ge3": int((cnt >= 3).sum()), "expected_ge3": float(N * dist[3:].sum()),
                    "lambda_corr_note": "EVAL lambdas share each seed's streams (positive dependence expected)",
                    "pairs": pairs})
    S = max(S_all) if S_all else 0
    lp = label_prob(S)
    out["label_prob"] = lp
    for arm in arms:
        labs = [c[k]["frozen_label"] for c in out["cells"] if c["arm"] == arm for k in KIND_EDGE]
        k_obs = sum(1 for x in labs if x == "INVALID")
        m = len(labs)
        out["label_count"][arm] = {
            "rates": m, "invalid_observed": k_obs, "expected": m * lp["p_invalid"],
            "p_ge_observed_indep": float(stats.binom.sf(k_obs - 1, m, lp["p_invalid"])) if m else None,
            "note": "rates treated as independent (P_placebo / P_placebo_conf share a dataset; lambdas and n share "
                    "seeds): an approximation"}
    # (d') added after the declaration, descriptive: the frozen dependence-aware null of the INVALID count (EVAL's own
    # F_max simulation: shared seeds, lambda correlation, nested n, same-dataset candidates; dependence.json)
    E = _E()
    fm = E.fmax_simulate([("E4", "R3", lam, n) for lam in E4R3_LAMS for n in E4R3_NS], S, ["plac_raw", "conf_raw"],
                         E.load_dependence())
    tot = sum(fm["count_dist"].values())
    out["fmax_dependence"] = {
        "f_max": fm["f_max"], "nsim": fm["nsim"], "n_cells": fm["n_cells"], "p_cell_invalid": fm["p_cell_invalid"],
        "count_dist": fm["count_dist"],
        "p_ge": {arm: sum(v for k, v in fm["count_dist"].items()
                          if int(k) >= sum(1 for c in out["cells"] if c["arm"] == arm
                                           and any(c[kk]["frozen_label"] == "INVALID" for kk in KIND_EDGE))) / tot
                 for arm in arms},
        "note": "counts INVALID cells (a cell is INVALID if either rate is), as the frozen C3 rule"}
    out["screen"] = {"duplicates": len(scr["duplicates"]), "unexpected": len(scr["unexpected"]),
                     "role_mismatch": len(scr["role_mismatch"])}
    return out


# ================================================================================================ X5 step 2 / 3
def _provenance(use: list[dict], per_cell: int = 10) -> dict:
    """The first ``per_cell`` datasets of each cell regenerated by the frozen generator: hash must match."""
    seen, ok, bad = defaultdict(set), 0, []
    for r in sorted(use, key=lambda r: r["key"]):
        j = r["job"]
        ck = (j["lam"], j["n"])
        if len(seen[ck]) >= per_cell or j["seed"] in seen[ck] or (r.get("status") or "ok") != "ok":
            continue
        seen[ck].add(j["seed"])
        ds, _ = G.generate_dataset(j["world"], j["regime"], j["n"], j["seed"], lam=j["lam"], kappa=j["kappa"])
        if G.dataset_hash(ds) == r.get("dataset_sha256"):
            ok += 1
        else:
            bad.append(r["key"])
    return {"checked": ok + len(bad), "equal": ok, "differ": bad[:5], "ok": not bad}


def x5(fail_merged: str, adj_merged: str | None) -> dict:
    out = {"cells": [], "primary": [], "checks": {}}
    pvs: dict = {}
    for name, merged in (("x5_fail", fail_merged), ("x5_adj", adj_merged)):
        if not merged:
            continue
        spec = X.load_spec(name)
        use, scr, cells, _ = screened(merged, spec)
        out["checks"][name] = {"records": len(use), "duplicates": len(scr["duplicates"]),
                               "unexpected": len(scr["unexpected"]), "role_mismatch": len(scr["role_mismatch"]),
                               "provenance": _provenance(use)}
        pv = e4r3_pvals(use, X.X5_ARMS)
        pvs.update(pv)
        for (arm, lam, n), d in sorted(pv.items()):
            e = _cell(cells, arm, lam, n) or {}
            prim = e.get("primary") or {}
            row = {"spec": name, "arm": arm, "lam": lam, "n": n, "seeds": len(d), "status": e.get("status")}
            for kind in KIND_EDGE:
                k = sum(1 for x in d.values() if x.get(kind, 1.0) <= ALPHA)
                m = sum(1 for x in d.values() if kind in x)
                fr = prim.get(f"{kind}_raw") or {}
                row[kind] = {"hits": k, "n": m, "rate": k / m if m else None, "cp_ci": cp_ci(k, m) if m else None,
                             "boot_ci": fr.get("ci"), "r30": fr.get("validity"),
                             "p_binom_gt05": float(stats.binomtest(k, m, ALPHA, alternative="greater").pvalue)
                             if m else None}
            out["cells"].append(row)
    prim_rows = []
    for lam, n, key in X.X5_PRIMARY:
        kind = key.split("_")[0]
        c = next((r for r in out["cells"] if r["arm"] == "pmrt_eq" and r["lam"] == lam and r["n"] == n), None)
        prim_rows.append({"lam": lam, "n": n, "rate_key": key,
                          **({} if c is None else {("n_" if k == "n" else k): v for k, v in c[kind].items()})})
    have = [r for r in prim_rows if r.get("p_binom_gt05") is not None]
    adj = holm([r["p_binom_gt05"] for r in have]) if len(have) == len(prim_rows) else None
    for i, r in enumerate(prim_rows):
        if adj is None:
            r["outcome"] = None
            continue
        r["p_holm"] = adj[i]
        r["outcome"] = "EXCESS" if adj[i] < ALPHA else "CHANCE" if r["cp_ci"][1] <= VALID_UB else "UNRESOLVED"
    out["primary"] = prim_rows
    oc = [r["outcome"] for r in prim_rows]
    out["answer"] = (None if None in oc else "real excess" if "EXCESS" in oc
                     else "chance" if all(o == "CHANCE" for o in oc) else "unresolved")
    # step 3: exact McNemar eq vs r3 in each EXCESS pair (paired over datasets)
    out["step3"] = []
    for r in prim_rows:
        kind = r["rate_key"].split("_")[0]
        a, b = pvs.get(("pmrt_eq", r["lam"], r["n"]), {}), pvs.get(("pmrt_r3", r["lam"], r["n"]), {})
        seeds = sorted(s for s in set(a) & set(b) if kind in a[s] and kind in b[s])
        eq_only = sum(1 for s in seeds if a[s][kind] <= ALPHA < b[s][kind])
        r3_only = sum(1 for s in seeds if b[s][kind] <= ALPHA < a[s][kind])
        nd = eq_only + r3_only
        out["step3"].append({"lam": r["lam"], "n": r["n"], "rate_key": r["rate_key"], "paired": len(seeds),
                             "eq_only": eq_only, "r3_only": r3_only,
                             "p_mcnemar_exact": float(stats.binomtest(eq_only, nd, 0.5).pvalue) if nd else 1.0,
                             "triggered": r.get("outcome") == "EXCESS"})
    return out


# ================================================================================================ X6
def _arm_rows(cells: dict) -> dict:
    return {e["arm"]: e for e in cells.values()}


def x6(merged: str) -> dict:
    E = _E()
    spec = X.load_spec("x6_gbm")
    use, scr, cells, seed_rows = screened(merged, spec)
    kinds = dict(E._kinds(X.X6_WORLD, "R2"))                         # "S->K" -> null / plac / conf
    out = {"checks": {"records": len(use), "duplicates": len(scr["duplicates"]), "unexpected": len(scr["unexpected"]),
                      "role_mismatch": len(scr["role_mismatch"]), "provenance": X.repro_check(use, _x3_like(spec))},
           "arms": []}
    by_arm: dict[str, list] = defaultdict(list)
    for r in use:
        if (r.get("status") or "ok") == "ok" and (r.get("role") or "measure") == "measure":
            by_arm[r.get("arm") or r["key"].split("|")[0]].append(r)
    mult_cache: dict[int, np.ndarray] = {}
    for _ck, e in sorted(cells.items()):
        arm = e["arm"]
        d = spec["arms"][arm]
        told = d.get("told") or {}
        prim = e.get("primary") or {}
        row = {"arm": arm, "base": d.get("base", arm), "width": float(told.get("width", 1.0)),
               "centre": told.get("centre", "told" if told else "exact"),
               "redraw": told.get("redraw", "told" if told else "exact"),
               "seeds": e["n_measure"], "null_raw": X._rate(prim.get("null_raw")),
               "plac_raw": X._rate(prim.get("plac_raw"))}
        recs = sorted(by_arm.get(arm, []), key=lambda r: r["job"]["seed"])
        if recs and "x6" in recs[0]:
            row["bias"] = _bias(recs, kinds, mult_cache)
        out["arms"].append(row)
    out["predictions"] = _predictions(out["arms"])
    out["curve"] = [{"base": a["base"], "width": a["width"], "null_raw": a["null_raw"], "plac_raw": a["plac_raw"],
                     "z_bias_null": ((a.get("bias") or {}).get("null") or {}).get("mean")}
                    for a in sorted(out["arms"], key=lambda a: (a["base"], a["width"]))
                    if a["centre"] in ("told", "exact") and a["redraw"] in ("told", "exact")]
    return out


def _x3_like(spec: dict) -> dict:
    """X6 data is the frozen generator's (told law only at the method): provenance check as X3 (hash equal)."""
    return {**spec, "extras": {**spec["extras"], "experiment": "X3"}}


def _bias(recs: list[dict], kinds: dict, mult_cache: dict) -> dict:
    """Per kind (null / plac): mean z_bias over (seed, candidate) with seed-cluster CI; the rejection rate by z_bias
    tercile (pooled cut points) and top - bottom with a seed-bootstrap CI (frozen bootstrap draws); mean c, corr."""
    E = _E()
    out = {}
    for kind in ("null", "plac"):
        items = []                                                         # (seed, z, rejected, mean_c, corr)
        for r in recs:
            tg = r["x6"]["targets"]
            pmap = {f"{e['source']}->{e['target']}": e.get("p") for e in r.get("edges", [])}
            for a, b in r["x6"]["bias"].items():
                for ti, t in enumerate(tg):
                    name = f"{a}->{t}"
                    if kinds.get(name) != kind or pmap.get(name) is None or b["z_bias"][ti] is None:
                        continue
                    items.append((r["job"]["seed"], b["z_bias"][ti], float(pmap[name]) <= ALPHA,
                                  b["mean_c"][ti], b["corr_cw"][ti]))
        if not items:
            continue
        seeds = sorted({s for s, *_ in items})
        si = {s: i for i, s in enumerate(seeds)}
        S = len(seeds)
        z = np.array([x[1] for x in items])
        zs, zc = np.zeros(S), np.zeros(S)
        for (s, zz, *_rest) in items:
            zs[si[s]] += zz
            zc[si[s]] += 1
        q1, q2 = np.quantile(z, [1 / 3, 2 / 3])
        hb, cb, ht, ctp = np.zeros(S), np.zeros(S), np.zeros(S), np.zeros(S)
        for (s, zz, rej, *_rest) in items:
            if zz <= q1:
                hb[si[s]] += rej
                cb[si[s]] += 1
            elif zz > q2:
                ht[si[s]] += rej
                ctp[si[s]] += 1
        m = mult_cache.setdefault(S, E._boot_mult(S))
        with np.errstate(invalid="ignore", divide="ignore"):
            diff_bs = (m @ ht) / (m @ ctp) - (m @ hb) / (m @ cb)
        diff_bs = diff_bs[np.isfinite(diff_bs)]
        mc = np.array([x[3] for x in items if x[3] is not None], float)
        cr = np.array([x[4] for x in items if x[4] is not None], float)
        out[kind] = {"candidates": len(items), "seeds": S, "mean": float(z.mean()), "ci": E.cluster_ci(zs, zc),
                     "sd": float(z.std(ddof=1)) if len(z) > 1 else None,
                     "tercile_cuts": [float(q1), float(q2)],
                     "rej_bottom": float(hb.sum() / cb.sum()) if cb.sum() else None,
                     "rej_top": float(ht.sum() / ctp.sum()) if ctp.sum() else None,
                     "top_minus_bottom": (float(ht.sum() / ctp.sum() - hb.sum() / cb.sum())
                                          if cb.sum() and ctp.sum() else None),
                     "top_minus_bottom_ci": ([float(np.quantile(diff_bs, 0.025)), float(np.quantile(diff_bs, 0.975))]
                                             if len(diff_bs) else [None, None]),
                     "mean_c": float(np.nanmean(mc)) if len(mc) else None,
                     "mean_corr_cw": float(np.nanmean(cr)) if np.isfinite(cr).any() else None}
    return out


def _predictions(arms: list[dict]) -> dict:
    g = {(a["centre"], a["redraw"]): a for a in arms if a["base"] == "pmrt_nl_eq" and a["width"] == 2.0}
    TT, Tt, tT, tt = g.get(("told", "told")), g.get(("told", "true")), g.get(("true", "told")), g.get(("true", "true"))

    def lab(a):
        return ((a or {}).get("null_raw") or {}).get("validity")

    def ci(a):
        return ((a or {}).get("null_raw") or {}).get("ci") or [None, None]

    def zb(a):
        return ((a or {}).get("bias") or {}).get("null") or {}
    p1 = None if Tt is None or tt is None else (lab(Tt) != "INVALID" and lab(tt) != "INVALID")
    p2 = None if TT is None else lab(TT) == "INVALID"
    p3 = None if tT is None or ci(tT)[1] is None else ci(tT)[1] <= VALID_UB
    b_TT, b_tT = zb(TT), zb(tT)
    p4a = None if not b_TT else (b_TT["mean"] > 0 and b_TT["ci"][0] is not None and b_TT["ci"][0] > 0)
    p4b = None if not b_TT or b_TT["top_minus_bottom_ci"][0] is None else b_TT["top_minus_bottom_ci"][0] > 0
    p4c = None if not b_TT or not b_tT else abs(b_tT["mean"]) < abs(b_TT["mean"]) / 4
    p4 = None if None in (p4a, p4b, p4c) else (p4a and p4b and p4c)
    preds = {"P1": p1, "P2": p2, "P3": p3, "P4": p4, "P4a": p4a, "P4b": p4b, "P4c": p4c}
    if None in (p1, p2, p3, p4):
        outcome = None
    elif p1 and p2 and p3 and p4:
        outcome = "SUPPORTED"
    elif p2 and lab(tT) == "INVALID":
        outcome = "REFUTED"
    else:
        outcome = "PARTIAL (failed: " + ", ".join(k for k in ("P1", "P2", "P3", "P4") if not preds[k]) + ")"
    return {**preds, "outcome": outcome}


# ================================================================================================ tables
def tables_md(t: dict) -> str:
    L = ["# R-60 diagnostics X5 / X6: tables", "",
         "EXPLORATORY, post hoc (EXTRAS_PROTOCOL.md Amendments 2026-10-05). Rates are raw p <= .05 over testable "
         "candidates; boot CI = frozen seed-cluster bootstrap, CP = Clopper-Pearson. R-30 labels are descriptive.", ""]
    s1 = t.get("x5_step1")
    if s1:
        L += ["## X5 step 1 (EVAL records, read only)", "",
              f"Label probability at exactly .05 (S = {s1['label_prob']['S']}, {s1['label_prob']['nsim']} sims): "
              f"P(INVALID) = {_f(s1['label_prob']['p_invalid'], 4)}, P(VALID) = {_f(s1['label_prob']['p_valid'], 4)}.", ""]
        for arm, c in s1["label_count"].items():
            L.append(f"- {arm}: {c['invalid_observed']} INVALID of {c['rates']} E4 R3 C3 rates; expected "
                     f"{_f(c['expected'], 2)}; P(>= observed | independent) = {_f(c['p_ge_observed_indep'], 3)}")
        fd = s1.get("fmax_dependence")
        if fd:
            L += ["", f"Dependence-aware (frozen `fmax_simulate`, added after the declaration): F_max {fd['f_max']}, "
                  f"P(cell INVALID) {_f(fd['p_cell_invalid'], 4)}, count distribution {fd['count_dist']} "
                  f"(nsim {fd['nsim']}); P(count >= observed): "
                  + ", ".join(f"{a} {_f(p, 4)}" for a, p in fd["p_ge"].items())]
        L += ["", "| arm | lam | n | kind | rate | label | KS D | KS p | q.01 | q.05 | rej .01 o/e | rej .025 o/e | "
              "rej .05 o/e | rej .10 o/e |", "|" + "---|" * 14]
        for c in s1["cells"]:
            for k in KIND_EDGE:
                x = c[k]
                if not x["n"]:
                    continue
                tl = x["tail"]
                L.append(f"| {c['arm']} | {c['lam']:g} | {c['n']} | {k} | {_f(x['frozen_rate'])} | "
                         f"{x['frozen_label'] or ''}{' *' if c['failing'] and (c['lam'], c['n'], k + '_raw') in X.X5_PRIMARY else ''} | "
                         f"{_f(x['ks_d'])} | {_f(x['ks_p'])} | {_f(x['qq']['0.01'])} | {_f(x['qq']['0.05'])} | "
                         + " | ".join(_f(tl[str(a)]['obs_exp'], 2) for a in TAIL_LEVELS) + " |")
        L += ["", "Overlap of rejecting seeds (p <= .05), P_placebo and P_placebo_conf in a cell:", "",
              "| arm | lam | n | plac | conf | both | expected | hypergeom p |", "|" + "---|" * 8]
        for c in s1["cells"]:
            o = c["overlap_plac_conf"]
            L.append(f"| {c['arm']} | {c['lam']:g} | {c['n']} | {o['plac']} | {o['conf']} | {o['both']} | "
                     f"{_f(o['expected'], 1)} | {_f(o['p_upper'])} |")
        L += ["", "Across the 4 lambdas at one n (EVAL lambdas share each seed's streams):", "",
              "| arm | n | kind | hits per lambda | seeds >= 2 (exp. indep.) | seeds >= 3 (exp.) |", "|" + "---|" * 6]
        for o in s1["overlap_lambda"]:
            L.append(f"| {o['arm']} | {o['n']} | {o['kind']} | {o['per_lambda_hits']} | {o['seeds_ge2']} "
                     f"({_f(o['expected_ge2'], 1)}) | {o['seeds_ge3']} ({_f(o['expected_ge3'], 1)}) |")
    x5t = t.get("x5")
    if x5t:
        L += ["", "## X5 step 2 (fresh datasets)", "", f"Checks: `{json.dumps(x5t['checks'], sort_keys=True)}`", "",
              "Primary (pmrt_eq, one-sided exact binomial vs .05, Holm over 3):", "",
              "| lam | n | rate | hits/n | CP 95 % | p one-sided | p Holm | outcome |", "|" + "---|" * 8]
        for r in x5t["primary"]:
            L.append(f"| {r['lam']:g} | {r['n']} | {r['rate_key']} | {r.get('hits')}/{r.get('n_', r.get('n'))} | "
                     f"[{_f((r.get('cp_ci') or [None])[0])}, {_f((r.get('cp_ci') or [None, None])[1])}] | "
                     f"{_f(r.get('p_binom_gt05'), 4)} | {_f(r.get('p_holm'), 4)} | {r.get('outcome')} |")
        L += ["", f"Answer to B: **{x5t['answer']}**", "", "| spec | arm | lam | n | kind | hits/n | rate | CP 95 % | "
              "boot 95 % | R-30 | p one-sided |", "|" + "---|" * 11]
        for c in x5t["cells"]:
            for k in KIND_EDGE:
                x = c[k]
                L.append(f"| {c['spec']} | {c['arm']} | {c['lam']:g} | {c['n']} | {k} | {x['hits']}/{x['n']} | "
                         f"{_f(x['rate'])} | [{_f(x['cp_ci'][0])}, {_f(x['cp_ci'][1])}] | "
                         f"[{_f((x['boot_ci'] or [None])[0])}, {_f((x['boot_ci'] or [None, None])[1])}] | "
                         f"{x['r30'] or ''} | {_f(x['p_binom_gt05'], 4)} |")
        L += ["", "Step 3 (exact McNemar pmrt_eq vs pmrt_r3, read only if EXCESS):", ""]
        for s in x5t["step3"]:
            L.append(f"- lam {s['lam']:g} n {s['n']} {s['rate_key']}: paired {s['paired']}, eq only {s['eq_only']}, "
                     f"r3 only {s['r3_only']}, p {_f(s['p_mcnemar_exact'], 4)}, "
                     f"{'triggered' if s['triggered'] else 'not triggered'}")
    x6t = t.get("x6")
    if x6t:
        L += ["", "## X6 (E1 R2 n 1000, 200 seeds)", "", f"Checks: `{json.dumps(x6t['checks'], sort_keys=True)}`", "",
              "| arm | width | centring | redraw | truth-null raw [boot CI] | label | P_placebo raw [CI] | "
              "mean z_bias null [CI] | rej bottom / top tercile | top - bottom [CI] | mean c |", "|" + "---|" * 11]
        for a in x6t["arms"]:
            b = (a.get("bias") or {}).get("null") or {}
            L.append(f"| {a['arm']} | {a['width']:g} | {a['centre']} | {a['redraw']} | {X._txt(a['null_raw'])} | "
                     f"{(a['null_raw'] or {}).get('validity') or ''} | {X._txt(a['plac_raw'])} | "
                     + (f"{_f(b.get('mean'))} [{_f(b['ci'][0])}, {_f(b['ci'][1])}] | "
                        f"{_f(b.get('rej_bottom'))} / {_f(b.get('rej_top'))} | {_f(b.get('top_minus_bottom'))} "
                        f"[{_f(b['top_minus_bottom_ci'][0])}, {_f(b['top_minus_bottom_ci'][1])}] | "
                        f"{_f(b.get('mean_c'), 4)} |" if b else "| | | |"))
        L += ["", f"Predictions: `{json.dumps(x6t['predictions'], sort_keys=True)}`"]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cdd_oran.xmethod.diag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("x5-step1")
    p.add_argument("--eval-merged", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("x5")
    p.add_argument("--fail", required=True)
    p.add_argument("--adj", default=None)
    p.add_argument("--out", required=True)
    p = sub.add_parser("x6")
    p.add_argument("--merged", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("tables")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.cmd == "tables":
        t = {"protocol_sha256": X.protocol_sha256()}
        for k in TOP:
            f = os.path.join(a.out, f"diag_{k}.json")
            if os.path.exists(f):
                t[k] = json.load(open(f, encoding="utf-8"))
        X._write(os.path.join(a.out, "diag_tables.json"), json.dumps(t, indent=1, sort_keys=True) + "\n")
        X._write(os.path.join(a.out, "DIAG_TABLES.md"), tables_md(t))
        print(tables_md(t))
        return 0
    res, key = ((x5_step1(a.eval_merged), "x5_step1") if a.cmd == "x5-step1" else
                (x5(a.fail, a.adj), "x5") if a.cmd == "x5" else (x6(a.merged), "x6"))
    X._write(os.path.join(a.out, f"diag_{key}.json"), json.dumps(res, indent=1, sort_keys=True, default=float) + "\n")
    print(tables_md({key: res}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
