"""Figures + numbers for the 2026-09-30 archive report: step-1 causal discovery (MSCR-CRT, three attempts v1 / v2 / v3)
and the step-2 referee DEV check on E6-P P3 surge-L40.

Reads result files LIVE (json + numpy + matplotlib + stdlib only; no import of the repo's analysis code) and
recomputes, independently of the analyzers:
  step 1  * GT tables (dir): TRUE / NULL / INDET counts, the 8 GT-TRUE neighbour (nbr) edges, sleep -> far statuses,
            the G premise, for gt-1 (v1) and gt_ext (v2 / v3);
          * every method's scores (precision / recall / F1 / sign accuracy, overall and indirect = nbr) from its
            declared-edge lists and the GT cells, re-implementing cdd_oran/decision/edge_score.confusion; asserted equal
            to the analyzers' stored scores;
          * P1 (v1), P1v2 (v2, v3) and P2 / P2v2 (DEV-tuned baseline set only), asserted equal to the stored verdict parts;
          * the BY cut-off (q .05, m = 60) from the v2 / v3 per-hypothesis p-values, asserted to give the stored number of
            declarations; per-edge beta / z_approx / p for the neighbour edges, pooled and per fold;
          * K0 / K1 and the placebo (sharp-null) declaration counts of every method;
          * descriptive: the v1-EVAL replay of a design-centred statistic by diagnosis agent B
            (.tmp/diag/B/eval_lite_B1999.json) and the v2 statistic's DEV dry run (.tmp/diag/build/dry_dev_v2.json);
          * the LF-normalised sha256 of the three frozen protocol docs, asserted equal to the hashes in the result JSONs;
          * a run ledger from the raw discovery run files (headers + per-episode cpu_s; files are git-ignored, so the
            ledger is skipped for any file that is not present).
  step 2  * K-L0 constants from step2_kl0.json;
          * the 40-seed DEV arms from the RAW records runs/e6p-s2dev-1/all.jsonl: pooled V, R, energy retention, guard
            ratios, eligibility, paired-seed bootstrap 90 % CIs (N 10 000, default_rng([6618, n_seeds]), V_ref re-minimised
            per resample, exactly the driver's definition), mean policy counts; asserted equal to step2_dev_summary.json.
Emits fig1..fig7 (PNG) + figures_data.json next to this file.

Run (repo root):  .venv/Scripts/python.exe reports/2026-09-30-step1-discovery-and-step2-dev/make_figs.py
"""
import hashlib
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEC = os.path.join(ROOT, "scratchpad", "e6_dev", "decision")
RUNS = os.path.join(ROOT, "scratchpad", "e6_dev", "runs")
TMP = os.path.join(ROOT, ".tmp")
OUT = HERE

FAMILIES = ("carrier", "sleep", "ptx", "prot_min")
RELATIONS = ("own", "nbr", "far")
KPIS = ("pv", "v", "e", "rlf", "load")
HYP = tuple((f, r, k) for f in FAMILIES for r in RELATIONS for k in KPIS)
CHAIN = (("sleep", "nbr", "load", 1), ("sleep", "nbr", "pv", 1), ("sleep", "nbr", "v", 1), ("ptx", "nbr", "load", -1))
PREMISE = ("sleep", "nbr", "pv")
TUNED = ("shap_gbdt", "corr", "granger", "two_tower", "int", "qacm")          # P2 set (granger_by, rowperm not in P2)
Q_BY, M_HYP = 0.05, 60
PROTOCOLS = {"v1": "E6P_DISCOVERY_PROTOCOL.md", "v2": "E6P_DISCOVERY_PROTOCOL_V2.md",
             "v3": "E6P_DISCOVERY_PROTOCOL_V3.md"}
N_EPS = {"v1": 60, "v2": 480, "v3": 1200}
# step 2 (scratchpad/e6_dev/e6p_step2_dev.py constants)
S2_SINGLES = ("sub:ES", "sub:PowerES", "sub:SliceGuarantee")
S2_REF = ("freeze", "sub:ES+PowerES", "noarb") + S2_SINGLES
S2_POLICY = ("B1", "B2", "M1", "M2")
S2_FIELDS = ("prot_viol", "prot_ue_s", "energy_j", "viol_ue_s", "ue_s", "nonprot_embb_viol", "ll_viol", "rlf")
GUARD_KEYS = ("svr", "nonprot_embb_viol", "ll_viol", "rlf")
X_MATCH, GUARD, N_BOOT, S2_TAG = 0.90, 1.10, 10_000, 6618
S2_BAR = 0.35                                                                  # K-L0 kill bar R_pred < .35

INK = "#1d1f23"; SOFT = "#5f6368"; GRID = "#e4e2dc"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                         "#e87ba4", "#008300", "#4a3aa7", "#e34948")
GREY = "#8d8a83"; LIGHT = "#d9d6cf"
ATT_COL = {"v1": GREY, "v2": BLUE, "v3": VIOLET}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": "#c9c5bb",
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": SOFT, "ytick.color": SOFT,
                     "axes.titlesize": 11.5, "figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.spines.top": False, "axes.spines.right": False})


def J(path):
    return json.load(open(path, encoding="utf-8"))


def close(a, b, tol=1e-9):
    if a is None or b is None:
        return a is b
    a, b = float(a), float(b)
    if a != a or b != b:
        return (a != a) and (b != b)
    return abs(a - b) <= tol * max(1.0, abs(b))


# ======================================================================================== step 1: scoring
def gt_ref(cells):
    return {(c["family"], c["relation"], c["kpi"]): {"status": c["status"], "sign": int(c["sign"]), "mean": c["mean"],
                                                     "ci": c["ci"], "n": c["n"]} for c in cells}


def decl_from_list(lst):
    """analyzer 'declared' list [["f|r|k", sign], ...] -> {h: {declared, sign}} (every other hypothesis not declared)."""
    d = {h: {"declared": False, "sign": 0} for h in HYP}
    for key, s in lst:
        d[tuple(key.split("|"))] = {"declared": True, "sign": int(s)}
    return d


def decl_from_crt(run):
    return {tuple(r["h"]): {"declared": r["status"] == "declared", "sign": int(r["sign"]), "p": r["p"],
                            "beta": r["beta"], "z": r["z_approx"], "n": r["n"], "status": r["status"]}
            for r in run["per_hypothesis"]}


def _safe(a, b):
    return a / b if b else float("nan")


def confusion(decl, ref, relation=None):
    tp = fp = fn = tn = sok = 0
    for h in HYP:
        if relation is not None and h[1] != relation:
            continue
        g = ref[h]
        if g["status"] == "INDET":
            continue
        d = decl.get(h) or {}
        if d.get("declared"):
            if g["status"] == "TRUE":
                tp += 1; sok += int(d["sign"] == g["sign"])
            else:
                fp += 1
        elif g["status"] == "TRUE":
            fn += 1
        else:
            tn += 1
    p, r = _safe(tp, tp + fp), _safe(tp, tp + fn)
    f1 = (0.0 if tp + fp + fn else float("nan")) if tp == 0 else 2 * p * r / (p + r)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": p, "recall": r, "f1": f1, "sign_acc": _safe(sok, tp)}


def score(decl, ref):
    far = sum(1 for h in HYP if h[1] == "far" and (decl.get(h) or {}).get("declared"))
    return {"overall": confusion(decl, ref), "indirect": confusion(decl, ref, "nbr"), "far_declared": far,
            "n_declared": sum(1 for d in decl.values() if d.get("declared"))}


def check_scores(mine, stored, tag):
    for part in ("overall", "indirect"):
        for k in ("tp", "fp", "fn", "precision", "recall", "f1", "sign_acc"):
            assert close(mine[part][k], stored[part][k]), (tag, part, k, mine[part][k], stored[part][k])
    assert mine["far_declared"] == stored["far_declared"] and mine["n_declared"] == stored["n_declared"], tag


def by_cutoff(ps, q=Q_BY):
    """BY step-up over the tested p-values: (number declared, p threshold of the last rank)."""
    ps = np.sort(np.asarray(ps, float)); m = ps.size
    cm = np.sum(1.0 / np.arange(1, m + 1))
    thr = np.arange(1, m + 1) * q / (m * cm)
    ok = np.nonzero(ps <= thr)[0]
    k = int(ok[-1]) + 1 if ok.size else 0
    return k, float(thr[k - 1]) if k else 0.0, float(thr[k]) if k < m else float("nan")


def p2(mscr, base, splits_f, min_folds):
    per = {}
    for s in ("pooled", *splits_f):
        vals = {m: (0.0 if not np.isfinite(v.get(s, np.nan)) else v[s]) for m, v in base.items()}
        best_m = max(vals, key=lambda m: (vals[m], m))
        mm = mscr[s]
        per[s] = {"mscr": mm, "best_baseline": vals[best_m], "best_method": best_m,
                  "ok": bool(np.isfinite(mm) and mm >= vals[best_m] - 1e-12)}
    nf = sum(per[s]["ok"] for s in splits_f)
    return {"pass": per["pooled"]["ok"] and nf >= min_folds, "folds_ok": nf, "per_split": per}


def proto_sha(name):
    raw = open(os.path.join(ROOT, "docs", "benchmark", name), "rb").read().replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def gt_summary(ref):
    cnt = {s: sum(1 for h in HYP if ref[h]["status"] == s) for s in ("TRUE", "NULL", "INDET")}
    nbr_true = [h for h in HYP if h[1] == "nbr" and ref[h]["status"] == "TRUE"]
    return {"counts": cnt, "nbr_true": [("|".join(h), ref[h]["sign"]) for h in nbr_true],
            "sleep_far": {k: ref[("sleep", "far", k)]["status"] + f"({ref[('sleep', 'far', k)]['sign']:+d})"
                          for k in KPIS},
            "premise": {"mean": ref[PREMISE]["mean"], "ci": ref[PREMISE]["ci"], "status": ref[PREMISE]["status"],
                        "sign": ref[PREMISE]["sign"], "n": ref[PREMISE]["n"]},
            "nbr_status": {"|".join(h): ref[h]["status"] for h in HYP if h[1] == "nbr"}}


def step1():
    v1 = J(os.path.join(DEC, "step1_v1_full.json"))
    v1gt = J(os.path.join(DEC, "step1_v1_gt_tables.json"))
    R = {"v2": J(os.path.join(DEC, "step1_v2_full.json")), "v3": J(os.path.join(DEC, "step1_v3_full.json"))}
    P1 = {"v2": J(os.path.join(DEC, "step1_v2_pass1.json")), "v3": J(os.path.join(DEC, "step1_v3_pass1.json"))}
    out = {"protocol_sha256": {}}
    # --- frozen protocol hashes
    for v, doc in PROTOCOLS.items():
        out["protocol_sha256"][v] = proto_sha(doc)
    # the analyzers record the doc they check: v1 -> protocol v1; v2 AND v3 -> protocol v2 (e6p_disc_analyze_v2.py only
    # knows PROTOCOL_DOC_V2; protocol v3 changes only the EVAL sample and is not recorded in step1_v3_*.json)
    assert v1["protocol"]["sha256"] == out["protocol_sha256"]["v1"]
    assert R["v2"]["protocol"]["sha256"] == out["protocol_sha256"]["v2"]
    assert R["v3"]["protocol"]["sha256"] == out["protocol_sha256"]["v2"]
    out["protocol_recorded_in_v3_json"] = R["v3"]["protocol"]["doc"]
    # --- GT
    ref1 = gt_ref(v1gt["dir"]["cells"])
    ref1b = gt_ref(v1["gt"]["cells"])
    assert all(ref1[h]["status"] == ref1b[h]["status"] and close(ref1[h]["mean"], ref1b[h]["mean"]) for h in HYP)
    ref2 = gt_ref(R["v2"]["gt"]["cells"])
    ref3 = gt_ref(R["v3"]["gt"]["cells"])
    assert all(ref2[h]["status"] == ref3[h]["status"] and close(ref2[h]["mean"], ref3[h]["mean"]) for h in HYP)
    out["gt"] = {"gt1": gt_summary(ref1), "gt_ext": gt_summary(ref2),
                 "gt1_labels": v1["gt"]["n_labels"], "gt_ext_labels": R["v2"]["gt"]["n_labels"],
                 "gt1_sleep_labels": v1["gt"]["labels_per_family"]["sleep"],
                 "gt_ext_sleep_labels": R["v2"]["gt"]["labels_per_family"]["sleep"],
                 "gt1_act_counts": v1["gt"]["act_counts"], "gt_ext_act_counts": R["v2"]["gt"]["act_counts"],
                 "gt1_delta": v1["gt"]["delta"], "gt_ext_delta": R["v2"]["gt"]["delta"],
                 "gt1_receiving": v1["gt"]["receiving"], "gt_ext_receiving": R["v2"]["gt"]["receiving"]}
    assert out["gt"]["gt1"]["counts"] == v1["gt"]["counts"] and out["gt"]["gt_ext"]["counts"] == R["v2"]["gt"]["counts"]
    nbr_edges = [tuple(k.split("|")) for k, _ in out["gt"]["gt_ext"]["nbr_true"]]
    assert [k for k, _ in out["gt"]["gt1"]["nbr_true"]] == [k for k, _ in out["gt"]["gt_ext"]["nbr_true"]]
    # edge-level GT means (winner's curse)
    out["gt_edges"] = {"|".join(h): {"gt1": {k: ref1[h][k] for k in ("mean", "ci", "status", "sign", "n")},
                                     "gt_ext": {k: ref2[h][k] for k in ("mean", "ci", "status", "sign", "n")}}
                       for h in nbr_edges}

    # --- v1: scores for every method / split, P1, P2
    sc1 = {}
    for m, dd in v1["declared"].items():
        for s, lst in dd.items():
            if m == "mscr_rowperm":
                continue                       # rowperm is scored over its 24 mapped hypotheses only (not recomputed)
            mine = score(decl_from_list(lst), ref1)
            check_scores(mine, v1["scores"][m][s], ("v1", m, s))
            sc1.setdefault(m, {})[s] = mine
    m1 = sc1["mscr_crt"]["pooled"]
    p1_v1 = {"indirect_recall": m1["indirect"]["recall"] >= 2 / 3 - 1e-12,
             "indirect_precision": m1["indirect"]["precision"] >= 0.80,
             "overall_f1": m1["overall"]["f1"] >= 0.60, "sign_accuracy": m1["overall"]["sign_acc"] >= 0.90,
             "far_declarations": m1["far_declared"] <= 1}
    assert p1_v1 == v1["P1"]["parts"], (p1_v1, v1["P1"]["parts"])
    f1_1 = {s: sc1["mscr_crt"][s]["indirect"]["f1"] for s in ("pooled", "fold0", "fold1", "fold2")}
    b1 = {m: {s: sc1[m][s]["indirect"]["f1"] for s in ("pooled", "fold0", "fold1", "fold2")} for m in TUNED}
    p2_v1 = p2(f1_1, b1, ("fold0", "fold1", "fold2"), 2)
    assert p2_v1["pass"] == v1["P2"]["pass"] and p2_v1["folds_ok"] == v1["P2"]["folds_ok"]
    for s, v in p2_v1["per_split"].items():
        assert close(v["best_baseline"], v1["P2"]["per_split"][s]["best_baseline"]), s
    out["v1"] = {"episodes": v1["episodes"], "K0": {k: v1["K0"][k] for k in ("rate_mscr", "n_reject_mscr",
                                                                          "n_by_declared", "k_max", "pass")},
                 "K1": {k: v1["K1"][k] for k in ("sleep_units", "sleep_rejects", "pass")},
                 "units_eval": v1["units"]["by_family_eval"], "dropped_eval": v1["units"]["dropped_eval"],
                 "G": v1["gt"]["G"]["pass"], "crt_declared": v1["crt"]["pooled"]["declared"],
                 "n_declared": {s: v1["crt"][s]["n_declared"] for s in ("pooled", "fold0", "fold1", "fold2")},
                 "audit": {k: v1["crt"]["audit"][k] for k in ("n_declared", "n_tested", "rate_p_le_alpha")},
                 "top1_receiving": v1["top1_receiving"]["mscr_crt"], "P1_parts": p1_v1,
                 "P1_values": {"indirect_recall": m1["indirect"]["recall"],
                               "indirect_precision": m1["indirect"]["precision"],
                               "overall_precision": m1["overall"]["precision"], "overall_f1": m1["overall"]["f1"],
                               "sign_accuracy": m1["overall"]["sign_acc"], "far_declared": m1["far_declared"]},
                 "P2": p2_v1, "verdict": v1["verdict"]["verdict"], "scores": sc1,
                 "tau": {m: (v1["baselines"][m]["tau"] or {}).get("tau") for m in TUNED},
                 "timing_s": v1["timing_s"], "power_floor": v1["crt"]["pooled"]["power_floor"]}
    # which nbr GT-TRUE edges v1 declared
    d1 = decl_from_list(v1["declared"]["mscr_crt"]["pooled"])
    out["v1"]["nbr_declared"] = {"|".join(h): d1[h]["declared"] for h in nbr_edges}
    # prot_min own pv sign error
    out["v1"]["prot_min_own_pv"] = {"declared_sign": d1[("prot_min", "own", "pv")]["sign"],
                                    "gt_sign": ref1[("prot_min", "own", "pv")]["sign"]}

    # --- v2 / v3
    for v in ("v2", "v3"):
        r, p1j = R[v], P1[v]
        folds = ("fold0", "fold1", "fold2", "fold3")
        runs = {s: decl_from_crt(r["crt_v2"][s]) for s in ("pooled", *folds)}
        # pass-1 and full runs must agree on the CRT (the baseline pass cannot move MSCR)
        for s in ("pooled", *folds):
            a = {tuple(x["h"]): x for x in p1j["crt_v2"][s]["per_hypothesis"]}
            b = {tuple(x["h"]): x for x in r["crt_v2"][s]["per_hypothesis"]}
            for h in a:     # status and p identical; beta / z agree to float round-off (<= 5e-11 seen)
                assert a[h]["status"] == b[h]["status"] and a[h]["p"] == b[h]["p"], (v, s, h)
                assert close(a[h]["z_approx"], b[h]["z_approx"], 1e-9) and close(a[h]["beta"], b[h]["beta"], 1e-9)
        # BY recomputation
        by = {}
        for s in ("pooled", *folds):
            ps = [x["p"] for x in runs[s].values() if x["status"] != "undetermined"]
            k, thr, nxt = by_cutoff(ps)
            assert k == r["crt_v2"][s]["n_declared"], (v, s, k, r["crt_v2"][s]["n_declared"])
            by[s] = {"n_tested": len(ps), "n_declared": k, "p_threshold_last_rank": thr, "p_threshold_next_rank": nxt}
        # scores
        sc = {}
        for m, dd in r["declared"].items():
            for s, lst in dd.items():
                mine = score(decl_from_list(lst), ref2)
                check_scores(mine, r["scores"][m][s], (v, m, s))
                sc.setdefault(m, {})[s] = mine
        for s in ("pooled", *folds):
            assert score(runs[s], ref2)["indirect"]["f1"] == sc["mscr_crt_v2"][s]["indirect"]["f1"]
        # P1v2
        mp = runs["pooled"]
        members = []
        for f, rel, k, sgn in CHAIN:
            h = (f, rel, k)
            members.append({"h": "|".join(h), "expected": sgn, "gt": ref2[h]["status"], "gt_sign": ref2[h]["sign"],
                            "declared": mp[h]["declared"], "sign": mp[h]["sign"], "p": mp[h]["p"],
                            "beta": mp[h]["beta"], "z": mp[h]["z"],
                            "hit": bool(mp[h]["declared"] and mp[h]["sign"] == sgn)})
        c_true = [x for x in members if x["gt"] == "TRUE" and x["gt_sign"] == x["expected"]]
        need = min(3, len(c_true))
        ov = sc["mscr_crt_v2"]["pooled"]["overall"]
        parts = {"premise_edge_declared_plus": bool(mp[PREMISE]["declared"] and mp[PREMISE]["sign"] == 1),
                 "chain_hits": sum(x["hit"] for x in c_true) >= need,
                 "overall_precision": ov["precision"] >= 0.80 - 1e-12, "sign_accuracy": ov["sign_acc"] >= 0.90 - 1e-12}
        assert parts == r["P1v2"]["parts"], (v, parts, r["P1v2"]["parts"])
        # P2v2 (4 folds, >= 3)
        f1m = {s: sc["mscr_crt_v2"][s]["indirect"]["f1"] for s in ("pooled", *folds)}
        bf = {m: {s: sc[m][s]["indirect"]["f1"] for s in ("pooled", *folds)} for m in TUNED}
        p2v = p2(f1m, bf, folds, 3)
        assert p2v["pass"] == r["P2v2"]["pass"] and p2v["folds_ok"] == r["P2v2"]["folds_ok"], v
        for s, x in p2v["per_split"].items():
            assert close(x["best_baseline"], r["P2v2"]["per_split"][s]["best_baseline"]), (v, s)
            assert x["best_method"] == r["P2v2"]["per_split"][s]["best_method"], (v, s)
        # which method (incl. untuned granger_by) has the highest indirect F1 per split: descriptive
        top_any = {s: max(((m, sc[m][s]["indirect"]["f1"]) for m in sc if m != "mscr_crt_v2"),
                          key=lambda t: (np.nan_to_num(t[1]), t[0])) for s in ("pooled", *folds)}
        # per-edge nbr numbers pooled and per fold
        edges = {}
        for h in nbr_edges:
            edges["|".join(h)] = {s: {k: runs[s][h][k] for k in ("declared", "sign", "p", "beta", "z", "n")}
                                  for s in ("pooled", *folds)}
        # all nbr hypotheses (pooled) for the table
        nbr_all = {"|".join(h): dict({k: mp[h][k] for k in ("declared", "sign", "p", "beta", "z", "status")},
                                     gt=ref2[h]["status"], gt_sign=ref2[h]["sign"]) for h in HYP if h[1] == "nbr"}
        zdec = [abs(x["z"]) for x in mp.values() if x["declared"]]
        znot = [abs(x["z"]) for x in mp.values() if x["status"] == "not_detected"]
        out[v] = {"episodes": r["episodes"], "fold_units": r["fold_units"], "units_eval": r["units"]["by_family_eval"],
                  "K0": {k: r["K0"][k] for k in ("rate", "n_reject", "n_by_declared", "k_max", "pass")},
                  "K0_placebo_dev": ({k: r["K0_placebo_dev_descriptive"][k] for k in ("rate", "n_reject", "n_by_declared")}
                                     if "K0_placebo_dev_descriptive" in r else None),
                  "K1": {k: r["K1"][k] for k in ("sleep_units", "sleep_rejects", "pass")},
                  "G": r["gt"]["G"]["pass"], "BY": by,
                  "declared_pooled": [["|".join(h), mp[h]["sign"], mp[h]["p"], mp[h]["beta"], mp[h]["z"]]
                                      for h in HYP if mp[h]["declared"]],
                  "z_boundary": {"min_declared": min(zdec), "max_not_detected": max(znot)},
                  "chain": members, "chain_true": len(c_true), "chain_need": need,
                  "P1v2_parts": parts, "P1v2_pass": all(parts.values()),
                  "P1v2_values": {"overall_precision": ov["precision"], "sign_accuracy": ov["sign_acc"],
                                  "indirect_recall": sc["mscr_crt_v2"]["pooled"]["indirect"]["recall"],
                                  "indirect_precision": sc["mscr_crt_v2"]["pooled"]["indirect"]["precision"],
                                  "far_declared": sc["mscr_crt_v2"]["pooled"]["far_declared"],
                                  "chain_hits": sum(x["hit"] for x in c_true)},
                  "P2v2": p2v, "top_indirect_f1_any_method": top_any,
                  "placebo_declarations": r["validity_placebo_declarations"],
                  "nbr_edges": edges, "nbr_all": nbr_all, "scores": sc, "verdict": r["verdict"]["verdict"],
                  "timing_s": r["timing_s"], "peak_rss_mb": r["peak_rss_mb"],
                  "tau": {m: (r["baselines"][m]["tau"] or {}).get("tau") for m in TUNED}}
    # --- descriptive diagnostics
    diagB = J(os.path.join(TMP, "diag", "B", "eval_lite_B1999.json"))
    out["diagB_v1eval"] = {f"{x['family']}|{x['relation']}|{x['kpi']}": {"z_adj": x["z_adj"], "p_adj": x["p_adj"],
                                                                       "p_mscr_B1999": x["p_mscr"], "n": x["n"]}
                           for x in diagB if x["relation"] == "nbr"}
    dry = J(os.path.join(TMP, "diag", "build", "dry_dev_v2.json"))
    out["dry_dev_v2"] = {"|".join(x["h"]): {"z": x["z_approx"], "p": x["p"], "n": x["n"], "status": x["status"]}
                         for x in dry["crt_v2"]["pooled"]["per_hypothesis"] if x["h"][1] == "nbr"}
    out["dry_dev_v2_meta"] = {"mode": dry["mode"], "episodes": dry["episodes"]["eval"]}
    snr = J(os.path.join(TMP, "diag", "D", "snr.json"))
    out["diagD_snr"] = {"|".join(e["edge"]): {"snr_unit": e["snr_unit"], "snr_label": e["snr_label"],
                                              "n_units_z4": e["n_units_z4"], "dev_units_per_ep": e["dev_units_per_ep"]}
                        for e in snr["edges"]}
    return out


# ======================================================================================== run ledger (raw, optional)
CPU_RE = re.compile(r'"cpu_s": ([0-9.eE+-]+)')


def ledger():
    runs = ["e6p-disc-dev-1", "e6p-disc-placebo-1", "e6p-disc-eval-1", "e6p-disc-gt-1", "e6p-disc-gtx-3",
            "e6p-disc-ev2-2", "e6p-disc-plx-c2", "e6p-disc-ev3-1"]
    out = {}
    for d in runs:
        f = os.path.join(RUNS, d, "all.jsonl")
        if not os.path.exists(f):
            out[d] = None
            continue
        hs, n, cpu = [], 0, 0.0
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                head = line[:300]
                if '"kind": "header"' in head:
                    hs.append(json.loads(line))
                elif '"kind": "episode"' in head:
                    n += 1
                    m = CPU_RE.findall(line[-600:])
                    cpu += float(m[-1]) if m else 0.0   # the top-level cpu_s is the last one in the line
        out[d] = {"episodes": n, "cpu_h": cpu / 3600.0, "parts": len(hs),
                  "first_header_utc": min(h.get("utc", "~") for h in hs),
                  "git_head": sorted({(h.get("code") or {}).get("git_head", "?")[:7] for h in hs}),
                  "e6_dirty": sorted({str((h.get("code") or {}).get("e6_dirty")) for h in hs}),
                  "platform": sorted({(h.get("numeric_env") or {}).get("platform", "?") for h in hs}),
                  "mb": os.path.getsize(f) / 2 ** 20}
    return out


# ======================================================================================== step 2
def step2():
    kl0 = J(os.path.join(DEC, "step2_kl0.json"))
    recs = [json.loads(l) for l in open(os.path.join(RUNS, "e6p-s2dev-1", "all.jsonl"), encoding="utf-8") if l.strip()]
    heads = [r for r in recs if r.get("kind") == "header"]
    Rr = {}
    for r in recs:
        if r.get("kind") == "job" and r.get("schema") == "e6p-s2dev-rec/1" and not r.get("smoke"):
            Rr.setdefault(int(r["seed"]), {})[r["arm"]] = r
    arms_all = S2_REF + S2_POLICY
    seeds = sorted(s for s in Rr if set(arms_all) <= set(Rr[s]))
    n = len(seeds)
    A = {a: {f: np.array([float(Rr[s][a][f]) for s in seeds]) for f in S2_FIELDS} for a in arms_all}

    def pooled(Aa, idx=None):
        s = {f: (v.sum() if idx is None else v[idx].sum(1)) for f, v in Aa.items()}
        return {"V": 3600.0 * s["prot_viol"] / np.maximum(s["prot_ue_s"], 1e-9), "E": s["energy_j"],
                "svr": 3600.0 * s["viol_ue_s"] / np.maximum(s["ue_s"], 1e-9),
                "nonprot_embb_viol": s["nonprot_embb_viol"], "ll_viol": s["ll_viol"], "rlf": s["rlf"]}

    P = {a: pooled(A[a]) for a in arms_all}
    V_AA = P["noarb"]["V"]; ref = min(S2_SINGLES, key=lambda a: P[a]["V"]); V_ref = P[ref]["V"]
    den = V_AA - V_ref; Ef, EA = P["freeze"]["E"], P["sub:ES+PowerES"]["E"]
    rng = np.random.default_rng([S2_TAG, n])
    idx = rng.integers(0, n, (N_BOOT, n))
    Pb = {a: pooled(A[a], idx) for a in arms_all}
    den_b = Pb["noarb"]["V"] - np.min([Pb[a]["V"] for a in S2_SINGLES], 0)

    def ci(x):
        x = np.asarray(x, float); x = x[np.isfinite(x)]
        return [float(np.quantile(x, 0.05)), float(np.quantile(x, 0.95))]

    def ratio(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(b > 0, a / np.where(b > 0, b, 1.0), np.where(a > 0, np.inf, 1.0))

    arms = {}
    stored = J(os.path.join(DEC, "step2_dev_summary.json"))["arms"]
    for a in arms_all:
        p = P[a]
        g = {k: float(ratio(np.float64(p[k]), np.float64(P["noarb"][k]))) for k in GUARD_KEYS}
        matched = bool(Ef - p["E"] >= X_MATCH * (Ef - EA) - 1e-9)
        gok = bool(all(p[k] <= GUARD * P["noarb"][k] + 1e-9 for k in GUARD_KEYS))
        with np.errstate(divide="ignore", invalid="ignore"):
            Rb = (Pb["noarb"]["V"] - Pb[a]["V"]) / den_b
            retb = (Pb["freeze"]["E"] - Pb[a]["E"]) / (Pb["freeze"]["E"] - Pb["sub:ES+PowerES"]["E"])
            dRb = (Pb["B1"]["V"] - Pb[a]["V"]) / den_b
        pc = {}
        for s in seeds:
            for k, v in (Rr[s][a].get("policy_counts") or {}).items():
                pc[k] = pc.get(k, 0) + v
        arms[a] = {"V": float(p["V"]), "R": float((V_AA - p["V"]) / den), "R_ci90": ci(Rb),
                   "retention": float((Ef - p["E"]) / (Ef - EA)), "retention_ci90": ci(retb), "matched": matched,
                   "guard_ratio": g, "guard_ratio_ci90": {k: ci(ratio(Pb[a][k], Pb["noarb"][k])) for k in GUARD_KEYS},
                   "guard_ok": gok, "eligible": matched and gok,
                   "dR_vs_B1": float((P["B1"]["V"] - p["V"]) / den), "dR_vs_B1_ci90": ci(dRb),
                   "policy_counts_per_ep": {k: v / n for k, v in pc.items()},
                   "cpu_s_mean": float(np.mean([Rr[s][a]["cpu_s"] for s in seeds])),
                   "E_kj_per_ep": float(p["E"] / n / 1e3)}
        st = stored[a]
        for k in ("V", "R", "retention"):
            assert close(arms[a][k], st[k], 1e-9), (a, k)
        for k in GUARD_KEYS:
            assert close(arms[a]["guard_ratio"][k], st["guard_ratio"][k], 1e-9), (a, k)
        assert arms[a]["eligible"] == st["eligible"], a
        assert all(close(x, y, 1e-9) for x, y in zip(arms[a]["R_ci90"], st["R_ci90"])), (a, arms[a]["R_ci90"], st["R_ci90"])
    cpu_all = [Rr[s][a]["cpu_s"] for s in Rr for a in Rr[s]]
    return {"n_seeds": n, "seeds": [seeds[0], seeds[-1]], "V_AA": V_AA, "V_ref": V_ref, "ref_arm": ref, "den": den,
            "arms": arms, "cpu_h": sum(cpu_all) / 3600.0, "n_jobs": len(cpu_all),
            "run": {"parts": len(heads), "first_header_utc": min(h["utc"] for h in heads),
                    "git_head": sorted({h["code"]["git_head"][:7] for h in heads}),
                    "e6_dirty": sorted({h["code"]["e6_dirty"] for h in heads}),
                    "platform": sorted({h["numeric_env"]["platform"] for h in heads})},
            "kl0": {"constants": kl0["constants"], "counts_pi0_per_ep": kl0["counts_pi0"]["per_episode"],
                    "episodes_pi0": kl0["counts_pi0"]["episodes"],
                    "verdict": {b: {"R_ref": v["R_ref"], "R_global": v["R_global"], "kill": v["kill"]}
                                for b, v in kl0["verdict"].items()},
                    "sign_agreement_pv": kl0["sign_agreement_pv"]["agree"],
                    "sign_agreement_all": kl0["sign_agreement_all"]["agree"],
                    "gt_direction_effects": kl0["gt_direction_effects"], "K_L0": kl0["K_L0"]}}


# ======================================================================================== figures
def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=140, bbox_inches="tight"); plt.close(fig)


EDGE_LBL = {"sleep|nbr|pv": "pico sleep → nbr protected viol. (premise)", "sleep|nbr|v": "pico sleep → nbr violations",
            "sleep|nbr|load": "pico sleep → nbr load", "sleep|nbr|e": "pico sleep → nbr energy",
            "ptx|nbr|load": "PowerES ptx → nbr load", "ptx|nbr|v": "PowerES ptx → nbr violations",
            "ptx|nbr|e": "PowerES ptx → nbr energy", "carrier|nbr|pv": "carrier → nbr protected viol."}


def fig1(s1):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    mets = [("indirect recall\n(8 GT-TRUE nbr)\nbar: v1 P1 ≥ 2/3", "rec"),
            ("indirect\nprecision\nbar: v1 P1 ≥ .80", "iprec"),
            ("overall\nprecision\nbar: P1v2 ≥ .80", "prec"),
            ("overall sign\naccuracy\nbar: ≥ .90", "sign"),
            ("chain members\nhit (of 4)\nbar: P1v2 ≥ 3", "chain")]
    vals = {"v1": {"rec": s1["v1"]["P1_values"]["indirect_recall"],
                   "iprec": s1["v1"]["P1_values"]["indirect_precision"],
                   "prec": s1["v1"]["P1_values"]["overall_precision"], "sign": s1["v1"]["P1_values"]["sign_accuracy"],
                   "chain": sum(s1["v1"]["nbr_declared"][h] for h in ("sleep|nbr|load", "sleep|nbr|pv",
                                                                      "sleep|nbr|v", "ptx|nbr|load")) / 4}}
    for v in ("v2", "v3"):
        pv = s1[v]["P1v2_values"]
        vals[v] = {"rec": pv["indirect_recall"], "iprec": pv["indirect_precision"], "prec": pv["overall_precision"],
                   "sign": pv["sign_accuracy"], "chain": pv["chain_hits"] / 4}
    x = np.arange(len(mets)); w = 0.27
    for i, v in enumerate(("v1", "v2", "v3")):
        ys = [vals[v][k] for _, k in mets]
        ax.bar(x + (i - 1) * w, ys, width=w * 0.92, color=ATT_COL[v],
               label=f"{v}: {N_EPS[v]} EVAL eps" + (" (MSCR-CRT v1)" if v == "v1" else " (MSCR-CRT v2)"))
        for xi, yv in zip(x + (i - 1) * w, ys):
            ax.text(xi, yv + 0.02, f"{yv:.2f}", ha="center", va="bottom", fontsize=7.2, rotation=90)
    thr = {"rec": 2 / 3, "prec": 0.80, "sign": 0.90, "iprec": 0.80, "chain": 0.75}
    for xi, (_, k) in zip(x, mets):
        ax.plot([xi - 1.5 * w, xi + 1.5 * w], [thr[k], thr[k]], color=RED, lw=1.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels([m for m, _ in mets], fontsize=8.0)
    ax.set_ylim(0, 1.42); ax.set_yticks(np.arange(0, 1.01, 0.2)); ax.set_ylabel("share")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.legend(fontsize=7.8, frameon=False, loc="upper left", ncol=3)
    ax.set_title("(a) edge recovery on pooled EVAL vs that attempt's GT (red dashes = bar)", loc="left", fontsize=10)
    ax = axes[1]
    zb = s1["diagB_v1eval"]["sleep|nbr|pv"]["z_adj"]
    e2, e3 = s1["v2"]["nbr_edges"]["sleep|nbr|pv"]["pooled"], s1["v3"]["nbr_edges"]["sleep|nbr|pv"]["pooled"]
    pts = [("v1", "v1\n60 eps", zb, False, f"z {zb:.2f}\nagent-B replay\n(v1: not declared)"),
           ("v2", "v2\n480 eps", e2["z"], e2["declared"], f"z {e2['z']:.2f}, p {e2['p']:.3f}\nnot declared"),
           ("v3", "v3\n1200 eps", e3["z"], e3["declared"],
            f"z {e3['z']:.2f}, p {e3['p']:.4f}\nβ {e3['beta']:+.2f}, DECLARED")]
    ax.axhspan(3.0, 3.3, color=RED, alpha=0.13, lw=0)
    ax.text(-0.45, 3.42, "declaration threshold ≈ z 3.0-3.3", color=RED, fontsize=7.8, va="bottom")
    for i, (v, lab, z, dec, txt) in enumerate(pts):
        ax.scatter([i], [z], s=120, color=ATT_COL[v] if dec else "white", edgecolor=ATT_COL[v], lw=2.2, zorder=3)
        ax.text(i, z - 0.38, txt, fontsize=7.6, va="top", ha="center")
    for v, i in (("v2", 1), ("v3", 2)):
        zbd = s1[v]["z_boundary"]
        ax.plot([i - 0.3] * 2, [zbd["max_not_detected"], zbd["min_declared"]], color=SOFT, lw=4, alpha=0.45,
                solid_capstyle="butt")
    ax.set_xticks(range(3)); ax.set_xticklabels([p[1] for p in pts], fontsize=8.6)
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(0, 7.0)
    ax.set_ylabel("z_approx of sleep → nbr pv"); ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_title("(b) the premise edge, which P1v2 must declare", loc="left", fontsize=10)
    fig.suptitle("Fig 1 — Step 1, three attempts: v1 KILL, v2 KILL (near miss on the premise edge), v3 P1v2 PASS "
                 "→ PARTIAL", x=0.01, ha="left", fontweight="bold", fontsize=11, y=1.02)
    fig.text(0.01, -0.07, "Each attempt: frozen protocol, fresh EVAL seeds; v1 scored against gt-1, v2 and v3 against "
             "gt_ext. v1's result file has no per-edge z: the hollow v1 point is diagnosis agent B's design-centred "
             "replay on v1 EVAL\n(descriptive, not the v1 statistic). Grey bars in (b): that run's boundary, from the "
             "largest |z| not declared to the smallest |z| declared (BY at q .05 over 60 hypotheses).",
             fontsize=8.1, color=SOFT)
    fig.tight_layout(); save(fig, "fig1_attempts_ladder.png")


def fig2(s1):
    edges = [k for k, _ in s1["gt"]["gt_ext"]["nbr_true"]]
    order = ["sleep|nbr|pv", "sleep|nbr|v", "sleep|nbr|load", "sleep|nbr|e", "ptx|nbr|load", "ptx|nbr|v",
             "ptx|nbr|e", "carrier|nbr|pv"]
    assert sorted(order) == sorted(edges)
    fig, ax = plt.subplots(figsize=(11.6, 5.0))
    y = np.arange(len(order))[::-1]
    mk = {"v1": "^", "v2": "o", "v3": "s"}
    off = {"v1": 0.22, "v2": 0.0, "v3": -0.22}
    for yi, e in zip(y, order):
        gts = s1["gt"]["gt_ext"]
        for v in ("v1", "v2", "v3"):
            if v == "v1":
                z = s1["diagB_v1eval"][e]["z_adj"]; dec = s1["v1"]["nbr_declared"][e]
            else:
                z = s1[v]["nbr_edges"][e]["pooled"]["z"]; dec = s1[v]["nbr_edges"][e]["pooled"]["declared"]
            az = max(abs(z), 0.12)
            ax.scatter([az], [yi + off[v]], marker=mk[v], s=70, color=ATT_COL[v] if dec else "white",
                       edgecolor=ATT_COL[v], lw=1.8, zorder=3)
            ax.text(az * 1.12, yi + off[v], f"{z:+.1f}", fontsize=7.2, va="center", color=ATT_COL[v])
    ax.axvspan(3.0, 3.3, color=RED, alpha=0.12, lw=0)
    ax.set_xscale("log"); ax.set_xlim(0.1, 70)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{EDGE_LBL[e]}\nGT gt_ext {s1['gt_edges'][e]['gt_ext']['mean']:+.4g} "
                        f"(gt-1 {s1['gt_edges'][e]['gt1']['mean']:+.4g})" for e in order], fontsize=8.0)
    ax.set_xlabel("|z| (log scale; the signed value is printed next to each point)")
    ax.grid(axis="x", color=GRID, lw=0.7, which="both")
    h = [Line2D([], [], marker=mk[v], ls="", color=ATT_COL[v], mfc=ATT_COL[v], ms=8,
                label={"v1": "v1 EVAL 60 eps (agent B replay; filled = declared by v1)",
                       "v2": "v2 EVAL 480 eps", "v3": "v3 EVAL 1200 eps"}[v]) for v in ("v1", "v2", "v3")]
    h += [Line2D([], [], marker="o", ls="", color=SOFT, mfc="white", ms=8, label="hollow = not declared"),
          Patch(color=RED, alpha=0.2, label="≈ declaration threshold z 3.0-3.3")]
    ax.legend(handles=h, fontsize=7.8, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=3)
    ax.set_xticks([0.1, 0.3, 1, 3, 10, 30]); ax.set_xticklabels(["0.1", "0.3", "1", "3", "10", "30"])
    ax.set_title("Fig 2 — The 8 GT-TRUE neighbour edges: v1 declared 3, v2 4, v3 7; none of the three declared a "
                 "neighbour edge the GT calls null", loc="left", fontweight="bold", fontsize=10.8)
    fig.text(0.01, -0.08, "z = z_approx of MSCR-CRT v2 (S / sd of S under the logged-π0 redraws). Sign convention "
             "\"dir\": effect of a knob-value increase (sleep 1 = asleep). The one v3 miss is PowerES ptx → nbr energy "
             f"(z {s1['v3']['nbr_edges']['ptx|nbr|e']['pooled']['z']:+.2f}, p {s1['v3']['nbr_edges']['ptx|nbr|e']['pooled']['p']:.4f}; "
             "BY cut-off at that run ≈ p .005).", fontsize=8.1, color=SOFT)
    fig.tight_layout(); save(fig, "fig2_nbr_edges_z.png")


MLBL = {"mscr_crt_v2": "MSCR-CRT v2", "granger": "Granger", "granger_by": "Granger + BY", "corr": "|corr|",
        "shap_gbdt": "SHAP-GBDT", "qacm": "QACM", "two_tower": "two-tower GNN", "int": "PACIFISTA INT"}


def fig3(s1):
    ms = ["mscr_crt_v2", "granger", "granger_by", "corr", "shap_gbdt", "qacm", "two_tower", "int"]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.5), gridspec_kw={"width_ratios": [1, 1, 1]})
    y = np.arange(len(ms))[::-1]; h = 0.36
    for ax, key, title in ((axes[0], "overall", "(a) overall F1 (60 hypotheses)"),
                           (axes[1], "indirect", "(b) indirect (nbr) F1 — the P2v2 score")):
        for j, v in enumerate(("v2", "v3")):
            vals = [np.nan_to_num(s1[v]["scores"][m]["pooled"][key]["f1"]) for m in ms]
            ax.barh(y + (0.5 - j) * h, vals, height=h * 0.92, color=ATT_COL[v],
                    label=f"{v} pooled ({N_EPS[v]} eps)")
            for yi, val in zip(y + (0.5 - j) * h, vals):
                ax.text(val + 0.01, yi, f"{val:.2f}", va="center", fontsize=7.2)
        ax.set_xlim(0, 1.05); ax.set_yticks(y); ax.set_yticklabels([MLBL[m] for m in ms], fontsize=8.6)
        ax.grid(axis="x", color=GRID, lw=0.7); ax.set_title(title, loc="left", fontsize=10)
    axes[0].legend(fontsize=7.8, frameon=False, loc="lower right")
    ax = axes[2]
    plc = [s1["v3"]["placebo_declarations"][m]["placebo"] for m in ms]
    plc_dev = [s1["v2"]["placebo_declarations"][m].get("placebo_dev") for m in ms]
    assert plc == [s1["v2"]["placebo_declarations"][m]["placebo"] for m in ms]
    ax.barh(y + h / 2, plc, height=h * 0.92, color=[GREEN if p == 0 else ORANGE for p in plc],
            label="placebo_ext (K0 data of v2 and v3)")
    ax.barh(y - h / 2, plc_dev, height=h * 0.92, color="white", edgecolor=[GREEN if p == 0 else ORANGE for p in plc_dev],
            hatch="////", label="placebo-1 (v1 placebo), descriptive")
    for yi, a, b in zip(y, plc, plc_dev):
        ax.text(a + 0.2, yi + h / 2, str(a), va="center", fontsize=7.6, fontweight="bold")
        ax.text(b + 0.2, yi - h / 2, str(b), va="center", fontsize=7.2, color=SOFT)
    ax.axvline(1, color=RED, ls="--", lw=1.2)
    ax.text(1.3, y[0] + 0.62, "K0 allows ≤ 1 BY declaration", color=RED, fontsize=7.4)
    ax.set_yticks(y); ax.set_yticklabels([MLBL[m] for m in ms], fontsize=8.6)
    ax.set_xlim(0, 21); ax.grid(axis="x", color=GRID, lw=0.7)
    ax.legend(handles=[Patch(color=ORANGE, label="placebo_ext (K0 data of v2 and v3)"),
                       Patch(facecolor="white", edgecolor=ORANGE, hatch="////", label="placebo-1 (v1's placebo), descr."),
                       Patch(color=GREEN, label="zero declarations")],
              fontsize=7.4, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=2)
    ax.set_title("(c) edges declared on placebo (sharp null:\n     every declaration is false)", loc="left", fontsize=10)
    fig.suptitle("Fig 3 — MSCR-CRT v2 vs the baselines on the same logs: best overall F1 in v2 and v3, best pooled "
                 "indirect F1 only in v3,\nand the only method with zero placebo declarations", x=0.01, ha="left",
                 fontweight="bold", fontsize=10.8, y=1.06)
    fig.text(0.01, -0.1, "Baselines keep the v1 code and the DEV far-FPR τ rule (tuned on 20 DEV episodes, not re-tuned). "
             "Granger + BY is untuned and is reported but not in the P2 set. The placebo counts use the same τ as "
             "the scores.", fontsize=8.1, color=SOFT)
    fig.tight_layout(); save(fig, "fig3_methods_f1_placebo.png")


def fig4(s1):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.3), sharey=True)
    splits = ["pooled", "fold0", "fold1", "fold2", "fold3"]
    for ax, v in zip(axes, ("v2", "v3")):
        per = s1[v]["P2v2"]["per_split"]
        x = np.arange(len(splits))
        ms_ = [per[s]["mscr"] for s in splits]
        bb = [per[s]["best_baseline"] for s in splits]
        gr = [np.nan_to_num(s1[v]["scores"]["granger"][s]["indirect"]["f1"]) for s in splits]
        ax.bar(x - 0.2, ms_, width=0.38, color=ATT_COL[v], label="MSCR-CRT v2")
        ax.bar(x + 0.2, bb, width=0.38, color=LIGHT, edgecolor=SOFT, label="best DEV-tuned baseline (P2 set)")
        for xi, s, a, b in zip(x, splits, ms_, bb):
            ax.text(xi - 0.2, a + 0.015, f"{a:.2f}", ha="center", fontsize=7.8, fontweight="bold")
            ax.text(xi + 0.2, b + 0.015, f"{b:.2f}\n{MLBL[per[s]['best_method']]}", ha="center", fontsize=7.0)
            ok = per[s]["ok"]
            ax.text(xi, -0.1, "✓" if ok else "✗", ha="center", fontsize=12, color=GREEN if ok else RED,
                    fontweight="bold")
        ax.scatter(x + 0.2, gr, marker="_", s=260, color=ORANGE, zorder=4, label="Granger (for reference)")
        per_fold = N_EPS[v] // 4
        ax.set_xticks(x); ax.set_xticklabels(["pooled\n" + str(N_EPS[v]) + " eps"] +
                                             [f"fold {i}\n{per_fold} eps" for i in range(4)], fontsize=8.4)
        ax.set_ylim(-0.17, 1.12); ax.axhline(0, color="#bdb8ad", lw=0.8); ax.grid(axis="y", color=GRID, lw=0.7)
        ax.set_title(f"({'a' if v == 'v2' else 'b'}) {v}: pooled {'✓' if per['pooled']['ok'] else '✗'}, folds "
                     f"{s1[v]['P2v2']['folds_ok']}/4 (needs ≥ 3) → P2v2 {'PASS' if s1[v]['P2v2']['pass'] else 'FAIL'}",
                     loc="left", fontsize=10)
    axes[0].set_ylabel("indirect (nbr) F1 vs gt_ext")
    axes[0].legend(fontsize=7.8, frameon=False, loc="upper right", ncol=1)
    fig.suptitle("Fig 4 — P2v2: MSCR wins pooled at 1200 episodes, but not on 3 of 4 smaller folds, so step 1 is "
                 "PARTIAL, not PASS", x=0.01, ha="left", fontweight="bold", fontsize=10.8, y=1.03)
    g3 = s1["v3"]["scores"]["granger"]
    fig.text(0.01, -0.05, "v3: Granger wins folds 0 and 1 at indirect precision "
             f"{g3['fold0']['indirect']['precision']:.2f} / {g3['fold1']['indirect']['precision']:.2f} (it declares "
             f"{s1['v3']['placebo_declarations']['granger']['placebo']} edges on the placebo); |corr| wins fold 2 at "
             f"precision 1.00 ({s1['v3']['placebo_declarations']['corr']['placebo']} placebo declarations). "
             "MSCR's indirect precision is 1.00 in every split of v2 and v3.", fontsize=8.1, color=SOFT)
    fig.tight_layout(); save(fig, "fig4_p2_folds.png")


def fig5(s1):
    order = ["sleep|nbr|pv", "sleep|nbr|v", "sleep|nbr|load", "ptx|nbr|load", "sleep|nbr|e", "ptx|nbr|v",
             "ptx|nbr|e", "carrier|nbr|pv"]
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.4))
    for ax, e in zip(axes.flat, order):
        g1, g2 = s1["gt_edges"][e]["gt1"], s1["gt_edges"][e]["gt_ext"]
        rows = [("gt-1\n(v1 GT)", g1["mean"], g1["ci"], GREY, "o"), ("gt_ext\n(v2/v3 GT)", g2["mean"], g2["ci"], INK, "o")]
        for v in ("v2", "v3"):
            x = s1[v]["nbr_edges"][e]["pooled"]
            se = abs(x["beta"] / x["z"]) if x["z"] else np.nan
            rows.append((f"MSCR {v}\nβ", x["beta"], [x["beta"] - 1.645 * se, x["beta"] + 1.645 * se], ATT_COL[v],
                         "s" if x["declared"] else "D"))
        for i, (lab, m, ci_, col, mk) in enumerate(rows):
            ax.errorbar([i], [m], yerr=[[m - ci_[0]], [ci_[1] - m]], fmt=mk, color=col, ms=7, capsize=4, lw=1.4,
                        mfc=col if mk != "D" else "white")
        ax.axhline(0, color="#bdb8ad", lw=0.9)
        ax.set_xticks(range(4)); ax.set_xticklabels([r[0] for r in rows], fontsize=7.2)
        ax.set_xlim(-0.5, 3.5); ax.grid(axis="y", color=GRID, lw=0.6)
        t = EDGE_LBL[e].replace(" (premise)", "")
        ax.set_title(t + (" ★" if e == "sleep|nbr|pv" else ""), fontsize=8.8, loc="left",
                     fontweight="bold" if e == "sleep|nbr|pv" else "normal")
        if e == "sleep|nbr|pv":
            ax.text(0.98, 0.96, f"GT {g1['mean']:+.1f} → {g2['mean']:+.2f}", transform=ax.transAxes, fontsize=8.4,
                    color=RED, va="top", ha="right", fontweight="bold")
    fig.suptitle("Fig 5 — The planning input was optimistic: the premise effect was +15.4 in gt-1 and +9.07 in the "
                 "fresh gt_ext; MSCR v3 estimates +10.3", x=0.01, ha="left", fontweight="bold", fontsize=10.8, y=1.01)
    fig.text(0.01, -0.035, "GT points: pooled paired-knockout mean with its 95 % episode-cluster bootstrap CI (20 "
             "episodes each, frozen gt_p rule). MSCR points: design-centred slope β with an approximate 90 % interval "
             "β ± 1.645·|β/z| (descriptive).\nSquare = declared, hollow diamond = not declared. Units per 90 s window "
             "summed over the neighbour cells: pv / v in violated UE-s, e in J, load in served UE-s. The estimands "
             "differ (π0 vs accept-all continuation; declared in the protocol).", fontsize=8.0, color=SOFT)
    fig.tight_layout(); save(fig, "fig5_winners_curse.png")


def fig6(s2):
    A = s2["arms"]
    rows = [("B1: reject every PowerES ptx-up\n(blanket, after warm-up)", "B1"),
            ("B2: B1 + reject pico sleep\n+ macro carrier-off", "B2"),
            ("M1: reject ptx-up only in\nsaturated cells (prb_util ≥ .999)", "M1"),
            ("M2: M1 + release while the\nregion's LL delay is rising", "M2")]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, a) in zip(y, rows):
        r, (lo, hi) = A[a]["R"], A[a]["R_ci90"]
        col = GREEN if A[a]["eligible"] else ORANGE
        ax.barh(yi, r, color=col, height=0.55)
        ax.errorbar(r, yi, xerr=[[r - lo], [hi - r]], fmt="none", ecolor=INK, lw=1.1, capsize=4)
        why = "eligible" if A[a]["eligible"] else ("INELIGIBLE: RLF " + f"{A[a]['guard_ratio']['rlf']:.2f}× AA"
                                                   if not A[a]["guard_ok"] else "INELIGIBLE: energy")
        ax.text(max(hi, 0) + 0.03, yi + 0.12, f"R = {r:.3f}", fontsize=9, fontweight="bold", va="center")
        ax.text(max(hi, 0) + 0.03, yi - 0.16, f"[{lo:.2f}, {hi:.2f}] · {why}", fontsize=7.8, va="center",
                color=GREEN if A[a]["eligible"] else RED)
    ax.axvline(S2_BAR, color=RED, ls="--", lw=1.2)
    ax.text(S2_BAR + 0.01, len(rows) - 0.42, "K-L0 kill bar: R < 0.35", color=RED, fontsize=8)
    ax.axvline(0, color=SOFT, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8.4)
    ax.set_xlim(-0.2, 1.05); ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.set_xlabel("recovery R = (V_AA − V_arm) / (V_AA − V_ref)"); ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_title("(a) R with 90 % paired-seed bootstrap CI", loc="left", fontsize=10)
    ax = axes[1]
    names = {"svr": "all-UE SVR", "nonprot_embb_viol": "non-prot. eMBB", "ll_viol": "LL viol.", "rlf": "RLF"}
    cols = [BLUE, AQUA, VIOLET, ORANGE]
    w = 0.17
    for i, (_, a) in enumerate(rows):
        for j, k in enumerate(GUARD_KEYS):
            g = A[a]["guard_ratio"][k]; lo, hi = A[a]["guard_ratio_ci90"][k]
            xx = i + (j - 1.5) * w
            ax.bar(xx, g, width=w * 0.9, color=cols[j], label=names[k] if i == 0 else None)
            ax.errorbar(xx, g, yerr=[[g - lo], [hi - g]], fmt="none", ecolor=INK, lw=0.8, capsize=2)
            if g > GUARD:
                ax.text(xx, hi + 0.02, f"{g:.2f}", ha="center", fontsize=7.6, color=RED, fontweight="bold")
    ax.axhline(GUARD, color=RED, ls="--", lw=1.3, label="limit 1.10× AA")
    ax.axhline(1.0, color="#bdb8ad", lw=0.9)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{a}\nenergy ret. {A[a]['retention']:.2f}" for _, a in rows], fontsize=8.4)
    ax.set_ylim(0.6, 1.72)
    ax.set_ylabel("arm ÷ accept-all (pooled, 40 seeds)"); ax.grid(axis="y", color=GRID, lw=0.7)
    ax.legend(fontsize=7.6, frameon=False, ncol=3, loc="upper left")
    ax.set_title("(b) guardrail ratios (90 % CI) and energy retention", loc="left", fontsize=10)
    fig.suptitle(f"Fig 7 — Step-2 DEV (P3 surge-L40, {s2['n_seeds']} DEV seeds): the rules that recover ~30 % break "
                 "the RLF guardrail; the safe rules recover ~10 %", x=0.01, ha="left", fontweight="bold", fontsize=10.8,
                 y=1.03)
    fig.text(0.01, -0.05, f"V_AA {s2['V_AA']:.2f}, V_ref {s2['V_ref']:.2f} (SliceGuarantee alone), denominator "
             f"{s2['den']:.2f} psvr. Eligible = energy retention ≥ 0.90 and every guardrail ≤ 1.10× AA. Bootstrap "
             f"default_rng([6618, 40]), 10 000 resamples. DEV only, not frozen, not an evaluation.",
             fontsize=8.1, color=SOFT)
    fig.tight_layout(); save(fig, "fig7_step2_dev_arms.png")


def fig7(s1):
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    e = "sleep|nbr|pv"
    n = np.logspace(np.log10(12), np.log10(2600), 200)
    ax.fill_between(n, 1.4 * np.sqrt(n / 60), 1.7 * np.sqrt(n / 60), color=GREY, alpha=0.18, lw=0,
                    label="v2 sizing: z 1.4-1.7 per 60 eps (DEV power analysis, anchored on gt-1 +15.4)")
    z2 = s1["v2"]["nbr_edges"][e]["pooled"]["z"]
    ax.plot(n, z2 * np.sqrt(n / 480), color=VIOLET, lw=1.4, ls="--",
            label=f"v3 sizing: z {z2:.2f}·√(n/480) (from v2's observed z)")
    ax.axhspan(3.0, 3.3, color=RED, alpha=0.12, lw=0)
    ax.text(14, 3.15, "declaration threshold ≈ z 3.0-3.3", color=RED, fontsize=8, va="center")
    ax.axhline(4.1, color=VIOLET, lw=0.8, ls=":")
    ax.text(14, 4.2, "≈ 80 % power target z 4.1 (protocol v3 §1)", color=VIOLET, fontsize=7.6)
    # points
    d = s1["dry_dev_v2"][e]
    ax.scatter([20], [d["z"]], marker="P", s=80, color=SOFT, zorder=4, label=f"DEV dry run, 20 eps (development data): z {d['z']:.2f}")
    zb = s1["diagB_v1eval"][e]["z_adj"]
    ax.scatter([60], [zb], marker="^", s=80, color="white", edgecolor=GREY, lw=1.8, zorder=4,
               label=f"v1 EVAL 60 eps, agent B replay (descriptive): z {zb:.2f}")
    for v, mk in (("v2", "o"), ("v3", "s")):
        fz = [s1[v]["nbr_edges"][e][f"fold{i}"]["z"] for i in range(4)]
        ax.scatter([N_EPS[v] // 4] * 4, fz, marker=mk, s=38, color="white", edgecolor=ATT_COL[v], lw=1.4, zorder=4,
                   label=f"{v} folds ({N_EPS[v] // 4} eps each): z " + ", ".join(f"{z:.2f}" for z in fz))
        zp = s1[v]["nbr_edges"][e]["pooled"]["z"]
        ax.scatter([N_EPS[v]], [zp], marker=mk, s=110, color=ATT_COL[v], zorder=5,
                   label=f"{v} pooled ({N_EPS[v]} eps): z {zp:.2f}" + (" DECLARED" if s1[v]["nbr_edges"][e]["pooled"]["declared"] else " not declared"))
    ax.set_xscale("log"); ax.set_xlim(12, 2600); ax.set_ylim(0, 8.6)
    ax.set_xticks([20, 60, 120, 300, 480, 1200]); ax.set_xticklabels(["20", "60", "120", "300", "480", "1200"])
    ax.set_xlabel("EVAL episodes (log scale)"); ax.set_ylabel("z_approx, sleep → nbr pv")
    ax.grid(color=GRID, lw=0.6, which="both")
    ax.legend(fontsize=7.3, frameon=False, loc="upper left")
    ax.set_title("Fig 6 — Sample size and the premise edge: v2 was sized on an inflated effect; v3 was sized on v2's "
                 "observed z", loc="left", fontweight="bold", fontsize=10.6)
    fig.text(0.01, -0.03, "Only the pooled v2 and v3 points are protocol results. The 20-episode DEV point and the "
             "60-episode replay use development data and are shown for scale only.", fontsize=8.1, color=SOFT)
    fig.tight_layout(); save(fig, "fig6_sample_size_z.png")


# ======================================================================================== output
def _r(x, n=5):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if x != x else float(f"{float(x):.{n + 2}g}")
    if isinstance(x, dict):
        return {str(k): _r(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, n) for v in x]
    return x


def main():
    s1 = step1()
    s2 = step2()
    led = ledger()
    fig1(s1); fig2(s1); fig3(s1); fig4(s1); fig5(s1); fig6(s2); fig7(s1)
    data = {"step1": s1, "step2": s2, "run_ledger": led}
    json.dump(_r(data), open(os.path.join(OUT, "figures_data.json"), "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)
    # ---- console summary (the numbers quoted in index.html)
    print("protocol sha256:", {k: v[:8] for k, v in s1["protocol_sha256"].items()})
    g1, g2 = s1["gt"]["gt1"], s1["gt"]["gt_ext"]
    print("GT gt-1", g1["counts"], "premise", _r(g1["premise"]), "sleep far", g1["sleep_far"])
    print("GT gt_ext", g2["counts"], "premise", _r(g2["premise"]), "sleep far", g2["sleep_far"])
    print("nbr TRUE", g2["nbr_true"], "| nbr status gt-1", g1["nbr_status"])
    print("v1:", s1["v1"]["verdict"], "K0", s1["v1"]["K0"], "K1", s1["v1"]["K1"], "P1", _r(s1["v1"]["P1_values"]),
          "nbr declared", s1["v1"]["nbr_declared"], "P2", _r(s1["v1"]["P2"]), "sign", s1["v1"]["prot_min_own_pv"],
          "top1", s1["v1"]["top1_receiving"], "audit", s1["v1"]["audit"], "ndecl", s1["v1"]["n_declared"])
    for v in ("v2", "v3"):
        x = s1[v]
        print(f"{v}: {x['verdict']} K0 {x['K0']} K0dev {x['K0_placebo_dev']} K1 {x['K1']} BY {_r(x['BY'])}")
        print("   P1v2", x["P1v2_parts"], _r(x["P1v2_values"]), "zbound", _r(x["z_boundary"]))
        for m in x["chain"]:
            print("   chain", _r(m))
        print("   P2v2", _r(x["P2v2"]), "\n   top any", _r(x["top_indirect_f1_any_method"]))
        print("   placebo", x["placebo_declarations"])
        for e, d in x["nbr_all"].items():
            print(f"   nbr {e:16s} {d['status']:12s} p {d['p']:.4g} beta {d['beta']:+.4g} z {d['z']:+.2f} GT {d['gt']}({d['gt_sign']:+d})")
        for m, ss in x["scores"].items():
            for s, sc in ss.items():
                print(f"   {m:12s} {s:6s} indP {sc['indirect']['precision']:.2f} indR {sc['indirect']['recall']:.2f} "
                      f"indF1 {sc['indirect']['f1']:.2f} ovP {sc['overall']['precision']:.2f} F1 {sc['overall']['f1']:.2f} "
                      f"sign {sc['overall']['sign_acc']:.2f} far {sc['far_declared']} n {sc['n_declared']}")
        print("   fold z premise", [round(x["nbr_edges"]["sleep|nbr|pv"][f"fold{i}"]["z"], 2) for i in range(4)])
    print("winner's curse:", {e: (round(d["gt1"]["mean"], 3), round(d["gt_ext"]["mean"], 3)) for e, d in s1["gt_edges"].items()})
    print("diagB z_adj:", {e: round(d["z_adj"], 2) for e, d in s1["diagB_v1eval"].items()})
    print("step2:", s2["n_seeds"], "V_AA", round(s2["V_AA"], 2), "V_ref", round(s2["V_ref"], 2), s2["ref_arm"],
          "den", round(s2["den"], 2), "cpu_h", round(s2["cpu_h"], 2), "jobs", s2["n_jobs"], s2["run"])
    for a, st in s2["arms"].items():
        print(f"   {a:20s} V {st['V']:.2f} R {st['R']:+.3f} {np.round(st['R_ci90'], 3)} ret {st['retention']:.3f} "
              f"g {_r(st['guard_ratio'], 3)} elig {st['eligible']} dB1 {st['dR_vs_B1']:+.3f} "
              f"{np.round(st['dR_vs_B1_ci90'], 3)} pc {_r(st['policy_counts_per_ep'], 3)}")
    print("K-L0:", s2["kl0"]["K_L0"], _r(s2["kl0"]["verdict"]), "counts", _r(s2["kl0"]["counts_pi0_per_ep"]),
          "agree", s2["kl0"]["sign_agreement_pv"], s2["kl0"]["sign_agreement_all"])
    print("ledger:", json.dumps(_r(led)))
    print("wrote fig1..fig7 + figures_data.json to", OUT)


if __name__ == "__main__":
    main()
