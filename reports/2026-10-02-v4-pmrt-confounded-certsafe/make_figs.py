"""Figures + numbers for the 2026-10-02 archive report: discovery protocol v4 (PMRT on fresh small slices), the
label-only rename "MSCR+" -> PMRT and its full-data equivalence check, option (a) (confounded incumbent logs -> causal
maps -> MapGateV2 referee), the reviewer synthesis, and follow-up 2b (certified-safe referee).

Reads result files LIVE (json + numpy + matplotlib + stdlib only; no import of the repo's analysis code) and
recomputes, independently of the analyzers:
  v4      * every method's scores on every slice (pooled 600, ten 60-slices, five 120-slices, two 300-slices) from its
            declared-edge list and the gt_v4 cells (re-implementing edge_score.confusion and disc_bench.subset_metrics);
            asserted equal to the stored metrics;
          * K0 (binomial tail, exact, from the placebo per-hypothesis p-values), K1, P1, P2 (wins per baseline), P3, S
            and the verdict precedence; asserted equal to v4_verdict.json;
          * the LF sha256 of the frozen v4 protocol, of the frozen MSCR+ artifact and of the PMRT artifact;
  rename  * a leaf-by-leaf diff of the frozen-code and renamed-code analysis_v4.json / verdict_v4.json trees with the
            pmrt_equivalence.py "reports" semantics (VOLATILE keys skipped, "MSCR+" -> "PMRT" on strings, NaN == NaN,
            exact comparison otherwise); asserted equal to the stored equivalence records; leaves counted per category;
  opt (a) * DISC-PASS inputs (K0, X2 precision / sign accuracy from the scored edge list);
          * map quality per arm (edges, TRUE right sign, TRUE wrong sign, NULL, INDET; placebo-map edges) from the maps
            and the option (a) GT cells; asserted equal to edge_quality in conf_disc_conf.json and conf_map_quality.txt;
          * the 160-seed EVAL from the RAW records runs/e6p-conf-eval-1/{a,b,c}/res_*.jsonl: pooled V, R, R*, energy
            retention, guard ratios, eligibility, paired-seed bootstrap 90 % CIs (N 10 000, default_rng([6624, 20, n]),
            V_ref re-minimised per resample), D1, D2 (Holm), never_sleep contrast, deferral counts; asserted equal to
            conf_eval.json / conf_verdict.json;
  2b      * tau = .05 G_k / (max(N(f, d), 1) * 3) from cs_calib.json, and G_k / N(f, d) from the RAW DEV records
            runs/e6p-cs-dev-1/all.jsonl; asserted equal to the artifact;
          * the certification table of every CS arm (DECLARED / CERTIFIED / UNRESOLVED per guard edge, uncertified
            classes) from the frozen maps, the bound tables and tau; asserted equal to E6P_CERTSAFE.json;
          * the 160-seed EVAL from the RAW records runs/e6p-cs-eval-1/{a..e}/res_*.jsonl (20 shards): as option (a) plus
            D3 and S1; asserted equal to cs_eval.json / cs_verdict.json; per-seed identity CS:PMRT vs never_sleep.
Emits fig1..fig7 (PNG) + figures_data.json next to this file.

Run (repo root):  .venv/Scripts/python.exe reports/2026-10-02-v4-pmrt-confounded-certsafe/make_figs.py
"""
import glob
import hashlib
import json
import math
import os

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
BENCH = os.path.join(ROOT, "docs", "benchmark")
ART = os.path.join(BENCH, "artifacts")
OUT = HERE

FAMILIES = ("carrier", "sleep", "ptx", "prot_min")
RELATIONS = ("own", "nbr", "far")
KPIS = ("pv", "v", "e", "rlf", "load")
HYP = tuple((f, r, k) for f in FAMILIES for r in RELATIONS for k in KPIS)
CHAIN = (("sleep", "nbr", "load", 1), ("sleep", "nbr", "pv", 1), ("sleep", "nbr", "v", 1), ("ptx", "nbr", "load", -1))
PREMISE = ("sleep", "nbr", "pv")
PRIMARY = "loadsp_c+wby1s"
COMBOS = tuple(f"{s}+{l}" for s in ("plain_c", "loadsp_c", "max") for l in ("by", "dagger1s", "wby1s"))
BASELINES = ("corr", "granger", "granger_by")
DESC = ("shap_gbdt", "int", "qacm", "two_tower")
VOLATILE = {"argv", "artifact", "protocol", "timing", "timing_s", "wall_s", "peak_rss_mb", "rss_mb", "out", "cache_dir",
            "generated", "host", "elapsed_s"}
# Gate A (e6p_step2_dev / e6p_conf_analyze definitions)
SINGLES = ("sub:ES", "sub:PowerES", "sub:SliceGuarantee")
A_ARM = "sub:ES+PowerES"
ANCHORS = ("freeze", A_ARM, "noarb") + SINGLES
SUM_FIELDS = ("prot_viol", "prot_ue_s", "energy_j", "viol_ue_s", "ue_s", "nonprot_embb_viol", "ll_viol", "rlf")
GUARD_KEYS = ("svr", "nonprot_embb_viol", "ll_viol", "rlf")
X_MATCH, GUARD, N_BOOT, BOOT_KEY = 0.90, 1.10, 10_000, (6624, 20)
D1_MARGIN, ALPHA_D2, RLF_CAP = 0.10, 0.05, 1.20
ASSOC = ("corr@dev", "corr@plc", "granger@dev", "granger@plc", "shap_gbdt@dev", "shap_gbdt@plc", "int@dev",
         "int@plc", "qacm@dev", "qacm@plc", "two_tower@dev", "two_tower@plc", "granger_by")
CONF_ARMS = ANCHORS + ("incumbent", "never_sleep", "B2") + ("MG:PMRT", "MG:GT", "MG:rand", "blanket2") + tuple(
    f"MG:{b}" for b in ASSOC)
CS_ASSOC = tuple(f"CS:{b}" for b in ASSOC)
CS_ARMS = ("CS:PMRT", "CS:GT", "CS:rand", "CS:blanket2") + CS_ASSOC
CS_ALL = ANCHORS + ("incumbent", "never_sleep", "MG:PMRT") + CS_ARMS
OUTCOME_SKIP = {"key", "arm", "policy_counts", "signature", "aliases", "cpu_s", "secs", "rss_mb"}
RHO, Z90 = 1.05, 1.2815515655446004
GUARD_KPIS = ("v", "rlf")

INK = "#1d1f23"; SOFT = "#5f6368"; GRID = "#e4e2dc"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                         "#e87ba4", "#008300", "#4a3aa7", "#e34948")
GREY = "#8d8a83"; LIGHT = "#d9d6cf"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": "#c9c5bb",
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": SOFT, "ytick.color": SOFT,
                     "axes.titlesize": 11.5, "figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.spines.top": False, "axes.spines.right": False})

CHECKS = []          # (name, passed) of every assertion block, printed at the end


def check(name, cond, detail=""):
    assert cond, f"{name}: {detail}"
    CHECKS.append(name)


def J(path):
    return json.load(open(path, encoding="utf-8"))


def sha_lf(path):
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def close(a, b, tol=1e-9):
    if a is None or b is None:
        return a is b
    a, b = float(a), float(b)
    if a != a or b != b:
        return (a != a) and (b != b)
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= tol * max(1.0, abs(b))


def jsonl(paths):
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


# ======================================================================================== scoring (edge_score)
def gt_ref(cells):
    return {(c["family"], c["relation"], c["kpi"]): {"status": c["status"], "sign": int(c["sign"]), "mean": c["mean"],
                                                     "ci": c["ci"]} for c in cells}


def decl_from_list(lst):
    d = {h: {"declared": False, "sign": 0} for h in HYP}
    for key, s in lst:
        d[tuple(key.split("|"))] = {"declared": True, "sign": int(s)}
    return d


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
        d = decl[h]
        if d["declared"]:
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
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1, "sign_acc": _safe(sok, tp)}


def subset_metrics(decl, ref):
    ov, ind = confusion(decl, ref), confusion(decl, ref, "nbr")
    c_true = hits = 0
    for f, r, k, s in CHAIN:
        g = ref[(f, r, k)]
        ok = g["status"] == "TRUE" and g["sign"] == s
        c_true += ok
        hits += ok and decl[(f, r, k)]["declared"] and decl[(f, r, k)]["sign"] == s
    pr = decl[PREMISE]
    return {"chain_hits": int(hits), "chain_true": int(c_true),
            "premise_hit": float(pr["declared"] and pr["sign"] == 1),
            "ind_recall": ind["recall"], "ind_precision": ind["precision"], "ind_f1": ind["f1"], "ind_tp": ind["tp"],
            "ind_fp": ind["fp"], "ov_precision": ov["precision"], "ov_recall": ov["recall"], "ov_f1": ov["f1"],
            "sign_acc": ov["sign_acc"], "n_declared": sum(d["declared"] for d in decl.values()),
            "far_declared": sum(1 for h in HYP if h[1] == "far" and decl[h]["declared"])}


def binom_sf_ge(k, m, p):
    """P(Binom(m, p) >= k), exact."""
    return float(sum(math.comb(m, i) * p ** i * (1 - p) ** (m - i) for i in range(k, m + 1)))


# ======================================================================================== section 2: v4
def v4():
    A = J(os.path.join(DEC, "v4_analysis.json"))
    V = J(os.path.join(DEC, "v4_verdict.json"))
    check("v4: decision copies identical to the run outputs",
          sha_lf(os.path.join(DEC, "v4_analysis.json")) == sha_lf(os.path.join(RUNS, "e6p-disc-v4an-1", "out", "analysis_v4.json"))
          and sha_lf(os.path.join(DEC, "v4_verdict.json")) == sha_lf(os.path.join(RUNS, "e6p-disc-v4an-1", "out", "verdict_v4.json")))
    out = {"protocol_sha256": sha_lf(os.path.join(BENCH, "E6P_DISCOVERY_PROTOCOL_V4.md")),
           "artifact_sha256": {"frozen_mscrplus": sha_lf(os.path.join(ART, "E6P_MSCRPLUS_V4_FROZEN.json")),
                               "pmrt": sha_lf(os.path.join(ART, "E6P_PMRT_V4.json"))}}
    check("v4: protocol LF sha256 = recorded", out["protocol_sha256"] == V["protocol"]["sha256"] == A["protocol"]["sha256"])
    check("v4: frozen artifact sha256 = recorded", out["artifact_sha256"]["frozen_mscrplus"] == V["artifact"]["sha256"])
    check("v4: primary combination", A["primary"] == PRIMARY)
    ref = gt_ref(A["gt"]["cells"])
    cnt = {s: sum(1 for h in HYP if ref[h]["status"] == s) for s in ("TRUE", "NULL", "INDET")}
    nbr_true = sorted("|".join(h) for h in HYP if h[1] == "nbr" and ref[h]["status"] == "TRUE")
    check("v4: GT counts", cnt == A["gt"]["counts"], (cnt, A["gt"]["counts"]))
    out["gt"] = {"counts": cnt, "nbr_true": nbr_true, "n_episodes": A["gt"]["n_episodes"], "n_labels": A["gt"]["n_labels"],
                 "premise": {k: A["G"]["cell"][k] for k in ("mean", "ci", "n", "n_eps", "status", "sign")}}
    # ---- every method x slice
    keys = ("chain_hits", "chain_true", "premise_hit", "ind_recall", "ind_precision", "ind_f1", "ind_tp", "ind_fp",
            "ov_precision", "ov_recall", "ov_f1", "sign_acc", "n_declared", "far_declared")
    sc, n_checked = {}, 0
    for sname, sl in A["slices_result"].items():
        for m, st in sl["metrics"].items():
            mine = subset_metrics(decl_from_list(st["declared"]), ref)
            for k in keys:
                assert close(mine[k], st[k]), (sname, m, k, mine[k], st[k])
            sc.setdefault(m, {})[sname] = mine
            n_checked += 1
    check(f"v4: scores of {n_checked} (method, slice) pairs recomputed from declared lists", n_checked > 0)
    names = list(A["slices_result"])
    s60 = [s for s in names if s.startswith("s60_")]
    s120 = [s for s in names if s.startswith("s120_")]
    s300 = [s for s in names if s.startswith("s300_")]
    # ---- K0 from the placebo per-hypothesis table (exact binomial tail)
    ps = [v["primary"]["p"] for v in A["placebo_hyp"].values()
          if v["primary"].get("status") != "undetermined" and v["primary"].get("p") is not None
          and np.isfinite(v["primary"]["p"])]
    m, k = len(ps), int(sum(p <= 0.05 for p in ps))
    tail = binom_sf_ge(k, m, 0.05)
    nd = sum(bool(v["primary"].get("declared")) for v in A["placebo_hyp"].values())
    K0 = {"m": m, "n_reject": k, "binom_tail": tail, "n_declared": nd, "pass": tail >= 0.01 and nd <= 1}
    check("v4: K0 recomputed (m, rejects, tail, declarations)", m == V["K0"]["m"] and k == V["K0"]["n_reject"]
          and close(tail, V["K0"]["binom_tail"], 1e-9) and nd == V["K0"]["n_declared"] and K0["pass"] == V["K0"]["pass"])
    # ---- K1
    K1 = {s: {"sleep_units": A["slices_result"][s]["units"]["sleep_units"],
              "sleep_rejects": A["slices_result"][s]["units"]["sleep_rejects"]} for s in s120}
    k1_pass = all(v["sleep_units"] >= 60 and v["sleep_rejects"] >= 15 for v in K1.values())
    check("v4: K1", k1_pass == V["K1"]["pass"] and all(K1[s]["sleep_units"] == V["K1"]["slices"][s]["sleep_units"] for s in s120))
    # ---- P1
    P = sc[PRIMARY]
    p1 = [{"slice": s, "premise": P[s]["premise_hit"] == 1.0, "chain_hits": P[s]["chain_hits"],
           "ok": P[s]["premise_hit"] == 1.0 and P[s]["chain_hits"] >= min(3, P[s]["chain_true"])} for s in s120]
    p1_pass = sum(x["ok"] for x in p1) >= 3
    check("v4: P1", p1_pass == V["P1"]["pass"] and sum(x["ok"] for x in p1) == V["P1"]["n_ok"])

    def win(mm, b):
        b = 0.0 if not np.isfinite(b) else b
        return bool(np.isfinite(mm) and mm >= b - 1e-12)
    P2 = {}
    for b in BASELINES:
        w60 = [win(P[s]["ind_f1"], sc[b][s]["ind_f1"]) for s in s60]
        w120 = [win(P[s]["ind_f1"], sc[b][s]["ind_f1"]) for s in s120]
        P2[b] = {"wins60": sum(w60), "wins120": sum(w120), "win60": w60, "win120": w120,
                 "ok": sum(w60) >= 6 and sum(w120) >= 3}
        st = V["P2"]["per_baseline"][b]
        check(f"v4: P2 vs {b}", P2[b]["wins60"] == st["wins60"] and P2[b]["wins120"] == st["wins120"]
              and P2[b]["win60"] == st["win60"] and P2[b]["win120"] == st["win120"] and P2[b]["ok"] == st["ok"])
    p2_pass = all(v["ok"] for v in P2.values())
    # ---- P3 (pooled), S
    pp = P["pooled"]
    p3_parts = {"premise_declared_plus": pp["premise_hit"] == 1.0, "chain_hits": pp["chain_hits"] >= min(3, pp["chain_true"]),
                "overall_precision": pp["ov_precision"] >= 0.80 - 1e-12, "sign_accuracy": pp["sign_acc"] >= 0.90 - 1e-12}
    check("v4: P3", p3_parts == V["P3"]["parts"] and close(pp["ov_precision"], V["P3"]["values"]["overall_precision"])
          and close(pp["ind_recall"], V["P3"]["values"]["ind_recall"]) and pp["far_declared"] == V["P3"]["values"]["far_declared"])
    p3_pass = all(p3_parts.values())
    s_vals = [P[s]["sign_acc"] for s in s120 if np.isfinite(P[s]["sign_acc"])]
    S = {"pooled": pp["sign_acc"], "mean120": float(np.mean(s_vals))}
    s_pass = S["pooled"] >= 0.9 - 1e-12 and S["mean120"] >= 0.9 - 1e-12
    check("v4: S", close(S["pooled"], V["S"]["pooled"]) and close(S["mean120"], V["S"]["mean120"]) and s_pass == V["S"]["pass"])
    g_pass = A["G"]["cell"]["status"] == "TRUE" and A["G"]["cell"]["sign"] == 1
    if not (K0["pass"] and V["K0n"]["pass"]):
        verdict = "INVALID"
    elif not g_pass:
        verdict = "NO-CHAIN"
    elif not k1_pass:
        verdict = "UNDERPOWERED"
    elif p1_pass and p2_pass and p3_pass and s_pass:
        verdict = "PASS"
    elif p3_pass and s_pass:
        verdict = "PARTIAL"
    else:
        verdict = "KILL"
    check("v4: verdict precedence", verdict == V["verdict"]["verdict"] == "PARTIAL", verdict)
    # ---- placebo declarations of every method (K0 data)
    plc = {m: A["placebo_descriptive"][m]["n_declared"] for m in A["placebo_descriptive"]}
    for m in plc:
        if "declared" in A["placebo_descriptive"][m]:
            assert plc[m] == len(A["placebo_descriptive"][m]["declared"]), m
    check("v4: placebo declaration counts consistent with the declared lists", True)

    def rng(m, key, sl=names):
        v = [sc[m][s][key] for s in sl if s in sc[m] and np.isfinite(sc[m][s][key])]
        return [min(v), max(v)] if v else None
    out.update({
        "slices": {"names": names, "n_episodes": {s: A["slices_result"][s]["n_episodes"] for s in names}},
        "K0": K0, "K0n": {k: V["K0n"][k] for k in ("pass", "n_variants", "rate05", "rate05_family", "n_variants_with_decl")},
        "K1": K1, "G": g_pass, "P1": p1, "P1_pass": p1_pass, "P2": P2, "P2_pass": p2_pass, "P3_parts": p3_parts,
        "P3_values": {k: pp[k] for k in ("ov_precision", "sign_acc", "ind_f1", "ind_precision", "ind_recall",
                                          "chain_hits", "n_declared", "far_declared")},
        "S": S, "verdict": verdict,
        "chain_z_pooled": {h: V["P3"]["chain"][h]["z"] for h in V["P3"]["chain"]},
        "placebo_declarations": plc,
        "pooled": {m: {k: sc[m]["pooled"][k] for k in ("ind_f1", "ind_precision", "ind_recall", "ov_precision",
                                                        "ov_f1", "sign_acc", "n_declared", "chain_hits")}
                   for m in sc if "pooled" in sc[m]},
        "ranges": {m: {"ov_precision_all18": rng(m, "ov_precision"), "ind_f1_60": rng(m, "ind_f1", s60),
                       "ind_f1_120": rng(m, "ind_f1", s120), "ind_f1_300": rng(m, "ind_f1", s300),
                       "n_declared_all18": rng(m, "n_declared"),
                       "ov_precision_pooled_120_300": rng(m, "ov_precision", ["pooled"] + s120 + s300)}
                   for m in (PRIMARY,) + BASELINES},
        "per_slice": {m: {s: {k: sc[m][s][k] for k in ("ind_f1", "ov_precision", "n_declared", "sign_acc")}
                          for s in names} for m in (PRIMARY,) + BASELINES},
        "desc_tau": A["desc_tau"], "baseline_tau": A["baseline_tau"],
        "timing_total_s": A["timing_s"]["total"],
        "pooled_units": A["slices_result"]["pooled"]["units"]["units"],
    })
    return out


# ======================================================================================== section 3: rename diff
def _label(x):
    return x.replace("MSCR+", "PMRT") if isinstance(x, str) else x


def tree_diff(x, y, path=(), out=None):
    """pmrt_equivalence.diff semantics (reports mode): VOLATILE keys skipped, labels mapped, NaN == NaN, exact."""
    out = [] if out is None else out
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if k in VOLATILE:
                continue
            if k not in x or k not in y:
                out.append((path + (k,), x.get(k, "<missing>"), y.get(k, "<missing>")))
            else:
                tree_diff(x[k], y[k], path + (k,), out)
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            out.append((path + ("#len",), len(x), len(y)))
        for i, (u, v) in enumerate(zip(x, y)):
            tree_diff(u, v, path + (i,), out)
    elif isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
        pass
    elif isinstance(x, str) and isinstance(y, str) and _label(x) == _label(y):
        pass
    elif type(x) is not type(y) or x != y:
        out.append((path, x, y))
    return out


def leaves(x, path=()):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, path + (k,))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, path + (i,))
    else:
        yield path


def pstr(p):
    s = ""
    for k in p:
        s += f"[{k}]" if isinstance(k, int) else ("#len" if k == "#len" else f"/{k}")
    return s


CAT_LABEL = {"criterion": "PMRT statistic x layer + criteria (K0, K0n, G, K1, P1-P3, S)",
             "assoc": "corr / granger / granger_by / v2+by baselines",
             "desc_nn": "descriptive baselines (SHAP, INT, QACM, two-tower)",
             "volatile": "volatile (provenance, timing; ignored by rule)",
             "other": "other (GT table, data checks, slice plan, unit counts, labels)"}


def category(p):
    if any(isinstance(k, str) and k in VOLATILE for k in p):
        return "volatile"
    top = p[0]
    if top in ("K0", "K0n", "G", "K1", "P1", "P2", "P3", "S", "placebo_hyp"):
        return "criterion"
    if top == "K0n_descriptive":
        return "criterion" if p[1] == "combos" else "other"
    if top == "placebo_descriptive":
        return "criterion" if p[1] in COMBOS else ("desc_nn" if p[1] in DESC else "assoc")
    if top == "slices_result" and len(p) > 2:
        if p[2] == "hyp":
            return "criterion"
        if p[2] == "metrics":
            return "criterion" if p[3] in COMBOS else ("desc_nn" if p[3] in DESC else "assoc")
    if top in ("desc_tau", "desc_errors"):
        return "desc_nn"
    if top == "baseline_tau":
        return "assoc"
    return "other"


def rename():
    fa = os.path.join(RUNS, "e6p-disc-v4an-1", "out")
    fb = os.path.join(RUNS, "e6p-disc-v4an-pmrt-1", "out")
    out = {}
    for name in ("analysis_v4.json", "verdict_v4.json"):
        a, b = J(os.path.join(fa, name)), J(os.path.join(fb, name))
        d = tree_diff(a, b)
        n_leaf = sum(1 for _ in leaves(a))
        out[name] = {"n_leaves": n_leaf, "n_diff": len(d),
                     "diffs": [[pstr(p), repr(x)[:60], repr(y)[:60]] for p, x, y in d]}
        if name == "analysis_v4.json":
            cats = {}
            for p in leaves(a):
                cats.setdefault(category(p), [0, 0])[0] += 1
            for p, _, _ in d:
                c = category(p if p[-1] != "#len" else p[:-1])
                cats.setdefault(c, [0, 0])[1] += 1
            out[name]["by_category"] = {c: {"leaves": v[0], "diffs": v[1]} for c, v in cats.items()}
            out[name]["diff_paths_top"] = sorted({"/".join(str(k) for k in p[:4] if not isinstance(k, int) or True)
                                                  for p, _, _ in d})
            out[name]["verdict_equal"] = a["verdict"] == b["verdict"]
            out[name]["v_a"], out[name]["v_b"] = a["verdict"]["verdict"], b["verdict"]["verdict"]
        st = J(os.path.join(fb, "equivalence_" + name.split("_")[0] + ".json"))
        check(f"rename: {name} leaf count / diff count / diff paths = stored equivalence record",
              n_leaf == st["n_leaves"] and len(d) == st["n_diff"]
              and [pstr(p) for p, _, _ in d[:20]] == [x[0] for x in st["first_diffs"]],
              (n_leaf, len(d), st["n_leaves"], st["n_diff"]))
    cats = out["analysis_v4.json"]["by_category"]
    check("rename: category leaf counts sum to the tree size",
          sum(v["leaves"] for v in cats.values()) == out["analysis_v4.json"]["n_leaves"])
    check("rename: zero differences in PMRT / criterion leaves", cats["criterion"]["diffs"] == 0)
    out["criterion_leaves"] = cats["criterion"]["leaves"]
    out["addendum_claim"] = {"n_leaves": 44663, "n_diff": 14, "criterion_leaves": 36667}
    out["addendum_agrees"] = (out["analysis_v4.json"]["n_leaves"] == 44663 and out["analysis_v4.json"]["n_diff"] == 14
                              and cats["criterion"]["leaves"] == 36667)
    out["artifact_sha256"] = {"E6P_MSCRPLUS_V4_FROZEN.json": sha_lf(os.path.join(ART, "E6P_MSCRPLUS_V4_FROZEN.json")),
                              "E6P_PMRT_V4.json": sha_lf(os.path.join(ART, "E6P_PMRT_V4.json"))}
    check("rename: artifact sha256 as in the addendum",
          out["artifact_sha256"]["E6P_MSCRPLUS_V4_FROZEN.json"].startswith("4735a85a")
          and out["artifact_sha256"]["E6P_PMRT_V4.json"].startswith("9991c390"))
    pm = J(os.path.join(ART, "E6P_PMRT_V4.json"))
    out["pmrt_supersedes"] = pm.get("supersedes")
    return out


# ======================================================================================== Gate A (shared)
def load_jobs(paths, sub, stage_key, stage):
    R, heads = {}, []
    for r in jsonl(paths):
        if r.get("kind") == "header" and r.get(stage_key) == stage:
            heads.append(r)
        if (r.get("kind") != "job" or r.get("schema") != "e6p-optaka2-rec/1" or r.get("sub") != sub
                or r.get(stage_key) != stage or r.get("smoke")):
            continue
        R.setdefault(int(r["seed"]), {})[r["arm"]] = r
    return R, heads


def alias_fill(R, alias_of):
    for s in R:
        for arm, tgt in alias_of.items():
            if tgt != arm and tgt in R[s] and arm not in R[s]:
                R[s][arm] = dict(R[s][tgt], arm=arm, alias_of=tgt)


def pooled(Aa, idx=None):
    s = {f: (v.sum() if idx is None else v[idx].sum(1)) for f, v in Aa.items()}
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"V": 3600.0 * s["prot_viol"] / np.maximum(s["prot_ue_s"], 1e-9), "E": s["energy_j"],
                "svr": 3600.0 * s["viol_ue_s"] / np.maximum(s["ue_s"], 1e-9),
                "nonprot_embb_viol": s["nonprot_embb_viol"], "ll_viol": s["ll_viol"], "rlf": s["rlf"]}


def ratio(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1.0), np.where(a > 0, np.inf, 1.0))


def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return [float(np.quantile(x, 0.05)), float(np.quantile(x, 0.95))] if x.size else [None, None]


def gate_a(R, arms):
    need = list(dict.fromkeys(list(ANCHORS) + list(arms)))
    seeds = sorted(s for s in R if set(need) <= set(R[s]))
    n = len(seeds)
    A = {a: {f: np.array([float(R[s][a][f]) for s in seeds]) for f in SUM_FIELDS} for a in need}
    P = {a: pooled(A[a]) for a in need}
    V_AA = P["noarb"]["V"]; ref = min(SINGLES, key=lambda x: P[x]["V"]); den = V_AA - P[ref]["V"]
    Ef, EA = P["freeze"]["E"], P[A_ARM]["E"]
    rng = np.random.default_rng([*BOOT_KEY, n])
    idx = rng.integers(0, n, (N_BOOT, n))
    Pb = {a: pooled(A[a], idx) for a in need}
    den_b = Pb["noarb"]["V"] - np.min([Pb[x]["V"] for x in SINGLES], 0)
    point, boot = {}, {}
    for a in need:
        p = P[a]
        Rv = float((V_AA - p["V"]) / den)
        matched = bool(Ef - p["E"] >= X_MATCH * (Ef - EA) - 1e-9)
        gok = bool(all(p[k] <= GUARD * P["noarb"][k] + 1e-9 for k in GUARD_KEYS))
        el = matched and gok
        point[a] = {"V": float(p["V"]), "R": Rv, "Rstar": Rv if el else min(Rv, 0.0),
                    "retention": float((Ef - p["E"]) / (Ef - EA)), "matched": matched, "guard_ok": gok, "eligible": el,
                    "guard_ratio": {k: float(ratio(np.float64(p[k]), np.float64(P["noarb"][k]))) for k in GUARD_KEYS}}
        with np.errstate(divide="ignore", invalid="ignore"):
            Rb = (Pb["noarb"]["V"] - Pb[a]["V"]) / den_b
            boot[a] = {"R": Rb, "Rstar": Rb if el else np.minimum(Rb, 0.0),
                       "retention": (Pb["freeze"]["E"] - Pb[a]["E"]) / (Pb["freeze"]["E"] - Pb[A_ARM]["E"]),
                       "guard": {k: ratio(Pb[a][k], Pb["noarb"][k]) for k in GUARD_KEYS}}
        point[a].update(R_ci90=ci(boot[a]["R"]), retention_ci90=ci(boot[a]["retention"]),
                        guard_ratio_ci90={k: ci(boot[a]["guard"][k]) for k in GUARD_KEYS})
    return {"seeds": seeds, "n": n, "V_AA": float(V_AA), "V_ref": float(P[ref]["V"]), "ref_arm": ref, "den": float(den),
            "point": point, "boot": boot}


def d1(point, boot, primary, assoc):
    best = max(assoc, key=lambda b: point[b]["Rstar"])
    dd = float(point[primary]["Rstar"] - point[best]["Rstar"])
    diff = boot[primary]["Rstar"] - np.max(np.stack([boot[b]["Rstar"] for b in assoc]), 0)
    fin = diff[np.isfinite(diff)]
    lb = float(np.quantile(fin, 0.05))
    return {"pass": bool(dd >= D1_MARGIN - 1e-12 and lb > 0), "delta1": dd, "best_point": best, "lb90": lb,
            "ci90": ci(diff)}


def holm(pv, alpha=ALPHA_D2):
    order = sorted(pv, key=lambda k: (pv[k], k))
    m, rej, stop = len(order), {}, False
    for i, k in enumerate(order):
        if not stop and pv[k] <= alpha / (m - i) + 1e-15:
            rej[k] = True
        else:
            stop = True; rej[k] = False
    return rej


def d2(boot, primary, assoc, sig):
    groups = {}
    for b in assoc:
        groups.setdefault(sig[b], []).append(b)
    pv, rows = {}, {}
    for _, bs in groups.items():
        b = bs[0]
        diff = boot[primary]["Rstar"] - boot[b]["Rstar"]
        p = float((1 + np.count_nonzero(~(diff > 0))) / (len(diff) + 1))
        pv[b] = p; rows[b] = {"members": bs, "p": p}
    rej = holm(pv)
    for b in rows:
        rows[b]["rejected"] = rej[b]
    return {"pass": all(rej.values()), "n_hypotheses": len(rows), "hypotheses": rows}


def check_arms(tag, st, stored_arms):
    for a, p in st["point"].items():
        s = stored_arms[a]
        for k in ("V", "R", "Rstar", "retention"):
            assert close(p[k], s[k]), (tag, a, k, p[k], s[k])
        for k in GUARD_KEYS:
            assert close(p["guard_ratio"][k], s["guard_ratio"][k]), (tag, a, k)
            assert all(close(x, y) for x, y in zip(p["guard_ratio_ci90"][k], s["guard_ratio_ci90"][k])), (tag, a, k)
        assert p["eligible"] == s["eligible"] and p["matched"] == s["matched"], (tag, a)
        assert all(close(x, y) for x, y in zip(p["R_ci90"], s["R_ci90"])), (tag, a, p["R_ci90"], s["R_ci90"])
        assert all(close(x, y) for x, y in zip(p["retention_ci90"], s["retention_ci90"])), (tag, a)


def sum_counts(R, seeds, arm, grp):
    tot = {}
    for s in seeds:
        for k, v in ((R[s][arm].get("policy_counts") or {}).get(grp) or {}).items():
            if isinstance(v, (int, float)):
                tot[k] = tot.get(k, 0) + v
    return dict(sorted(tot.items()))


def run_ledger(heads, R):
    jobs = [r for s in R for r in R[s].values() if not r.get("alias_of")]
    return {"parts": len(heads), "first_header_utc": min(h["utc"] for h in heads),
            "git_head": sorted({h["code"]["git_head"][:7] for h in heads}),
            "e6_dirty": sorted({str(h["code"]["e6_dirty"]) for h in heads}),
            "platform": sorted({h["platform"]["platform"] for h in heads}),
            "cpu_h": sum(r["cpu_s"] for r in jobs) / 3600.0, "n_jobs": len(jobs)}


# ======================================================================================== section 4: option (a)
def opt_a():
    disc = J(os.path.join(DEC, "conf_disc_conf.json"))
    maps_art = J(os.path.join(ART, "E6P_CONF_MAPS.json"))
    out = {"protocol_sha256": sha_lf(os.path.join(BENCH, "E6P_CONFOUNDED_PROTOCOL.md")),
           "maps_sha256": sha_lf(os.path.join(ART, "E6P_CONF_MAPS.json")),
           "disc_json_sha256": sha_lf(os.path.join(DEC, "conf_disc_conf.json"))}
    check("opt(a): protocol sha256 = disc record", out["protocol_sha256"] == disc["protocol"]["sha256"])
    check("opt(a): maps artifact sha256 = .sha256 file",
          out["maps_sha256"] == open(os.path.join(ART, "E6P_CONF_MAPS.json.sha256")).read().split()[0])
    check("opt(a): disc_conf.json = the copy the maps artifact pins", out["disc_json_sha256"] == maps_art["sha256"]["disc_conf.json"])
    ref = gt_ref(disc["gt"]["cells"])
    # ---- discovery criteria
    X2 = disc["X"]["X2"]
    scored = [e for e in X2["edges"] if e[2] != "INDET"]
    tr = [e for e in scored if e[2] == "TRUE"]
    prec = len(tr) / len(scored)
    sign = sum(e[1] == e[3] for e in tr) / len(tr)
    check("opt(a): X2 precision and sign accuracy from the scored edge list",
          close(prec, X2["precision"]) and close(sign, X2["sign_accuracy"]) and len(scored) == X2["n_scored"])
    out["disc"] = {"label": disc["label"]["label"], "K0": {k: disc["K0"][k] for k in ("pass", "n_reject", "m", "n_declared", "min_p")},
                   "K0n": {k: disc["K0n"][k] for k in ("pass", "rate05", "n_variants_with_decl")},
                   "K1": {k: disc["K1"][k] for k in ("pass", "sleep_units", "sleep_rejects")},
                   "G": {k: disc["G"]["cell"][k] for k in ("mean", "ci", "status", "sign")},
                   "X1": disc["X"]["X1"], "X2": {"precision": prec, "sign_accuracy": sign, "n_scored": len(scored),
                                                 "n_true": len(tr)},
                   "X3": {k: disc["X"]["X3"][k] for k in ("chain_hits", "chain_true")},
                   "gt_counts": disc["gt"]["counts"], "units": disc["units"]["disc"]["units"],
                   "n_episodes": disc["data"]}
    # ---- map quality
    mq = {}
    for arm, edges in disc["maps"].items():
        c = {"edges": 0, "true_right": 0, "true_wrong": 0, "null": 0, "indet": 0, "unknown": 0}
        for f, r, k, beta in edges:
            c["edges"] += 1
            g = ref.get((f, r, k))
            if g is None:
                c["unknown"] += 1
            elif g["status"] == "TRUE":
                c["true_right" if int(np.sign(beta)) == g["sign"] else "true_wrong"] += 1
            elif g["status"] == "NULL":
                c["null"] += 1
            else:
                c["indet"] += 1
        eq = disc["edge_quality"][arm]
        assert (c["edges"], c["true_right"], c["true_wrong"], c["null"], c["indet"]) == (
            eq["n_edges"], eq["n_true_right"], eq["n_true_wrong"], eq["n_null"], eq["n_indet"]), (arm, c, eq)
        pl = disc["placebo_maps"].get(arm)
        c["placebo_edges"] = None if pl is None else len(pl)
        mq[arm] = c
        assert [list(e) for e in maps_art["arms"][arm]["map"]] == [list(e) for e in edges] if "map" in maps_art["arms"].get(arm, {}) else True
    check("opt(a): map quality of all 17 maps recomputed from maps + GT cells", len(mq) == 17)
    # cross-check the text table
    txt = {}
    for line in open(os.path.join(DEC, "conf_map_quality.txt"), encoding="utf-8").read().splitlines()[1:]:
        left, right = line.split("|")
        t = left.split()
        txt[t[0]] = (int(t[1]), int(t[2]), int(t[3]), int(t[4]), int(t[5]), None if right.strip() == "-" else int(right))
    check("opt(a): conf_map_quality.txt agrees",
          all(txt[a] == (v["edges"], v["true_right"], v["true_wrong"], v["null"], v["indet"], v["placebo_edges"])
              for a, v in mq.items()))
    out["map_quality"] = mq
    gt_rlf = {"|".join(h): {"status": ref[h]["status"], "mean": ref[h]["mean"], "ci": ref[h]["ci"]}
              for h in HYP if h[2] == "rlf" and h[1] in ("nbr", "far") and h[0] in ("sleep", "ptx")}
    out["gt_rlf"] = gt_rlf
    hy = disc["disc_hyp"]["ptx|nbr|rlf"]
    out["ptx_nbr_rlf"] = {"plain_c_z": hy["plain_c"]["z"], "plain_c_p2": hy["plain_c"]["p2"], "loadsp_c_z": hy["loadsp_c"]["z"],
                          "loadsp_c_p2": hy["loadsp_c"]["p2"], "primary_p": hy["primary"]["p"],
                          "primary_status": hy["primary"]["status"], "gt": gt_rlf["ptx|nbr|rlf"]}
    out["pmrt_has_ptx_nbr_rlf"] = any((e[0], e[1], e[2]) == ("ptx", "nbr", "rlf") for e in disc["maps"]["MG:PMRT"])
    out["gt_has_ptx_nbr_rlf"] = any((e[0], e[1], e[2]) == ("ptx", "nbr", "rlf") for e in disc["maps"]["MG:GT"])
    out["decision_tables"] = {a: disc["decision_tables"][a] for a in ("MG:PMRT", "MG:GT")}
    # ---- EVAL from RAW
    files = sorted(glob.glob(os.path.join(RUNS, "e6p-conf-eval-1", "*", "res_*.jsonl")))
    check("opt(a): 12 raw EVAL shards present", len(files) == 12, len(files))
    R, heads = load_jobs(files, "conf", "conf_stage", "eval")
    led = run_ledger(heads, R)
    alias_fill(R, maps_art["alias_of"])
    arms = [a for a in CONF_ARMS if any(a in R[s] for s in R)]
    st = gate_a(R, arms)
    E = J(os.path.join(DEC, "conf_eval.json"))
    Vd = J(os.path.join(DEC, "conf_verdict.json"))
    check("opt(a): 160 complete seeds, every arm", st["n"] == E["n_seeds"] == 160 and len(arms) == 26)
    check("opt(a): Gate A constants", close(st["V_AA"], E["gate_a"]["V_AA"]) and close(st["den"], E["gate_a"]["den"])
          and st["ref_arm"] == E["gate_a"]["ref_arm"])
    check_arms("opt(a)", st, E["arms"])
    check("opt(a): every arm's V, R, R*, retention, guard ratios, eligibility and 90% CIs = conf_eval.json", True)
    P, B = st["point"], st["boot"]
    assoc = [f"MG:{b}" for b in ASSOC]
    sig = {a: v["signature"] for a, v in maps_art["arms"].items()}
    D1 = d1(P, B, "MG:PMRT", assoc)
    D2 = d2(B, "MG:PMRT", assoc, sig)
    check("opt(a): D1", D1["pass"] == Vd["D1"]["pass"] and close(D1["delta1"], Vd["D1"]["delta1"])
          and close(D1["lb90"], Vd["D1"]["lb90"]) and D1["best_point"] == Vd["D1"]["best_point"])
    check("opt(a): D2 (Holm)", D2["pass"] == Vd["D2"]["pass"] and D2["n_hypotheses"] == Vd["D2"]["n_hypotheses"]
          and all(close(D2["hypotheses"][b]["p"], Vd["D2"]["hypotheses"][b]["p"]) and
                  D2["hypotheses"][b]["rejected"] == Vd["D2"]["hypotheses"][b]["rejected"] for b in D2["hypotheses"]))
    ns = {"dR": P["MG:PMRT"]["R"] - P["never_sleep"]["R"], "ci90": ci(B["MG:PMRT"]["R"] - B["never_sleep"]["R"])}
    check("opt(a): PMRT - never_sleep", close(ns["dR"], Vd["never_sleep"]["dR"])
          and all(close(x, y) for x, y in zip(ns["ci90"], Vd["never_sleep"]["ci90"])))
    E_pass = P["MG:PMRT"]["eligible"]
    verdict = "NOT ELIGIBLE" if not E_pass else ("PASS" if D1["pass"] and D2["pass"] else
                                                ("PARTIAL" if D1["pass"] or D2["pass"] else "FAIL"))
    check("opt(a): verdict", verdict == Vd["verdict"]["verdict"] == "NOT ELIGIBLE", verdict)
    counts = {a: {g: sum_counts(R, st["seeds"], a, g) for g in ("defer", "defer_units", "units", "policy")} for a in arms}
    for a in arms:
        for g in ("defer", "defer_units", "units", "policy"):
            assert counts[a][g] == E["arms"][a]["counts"][g], (a, g)
    check("opt(a): deferral / unit / policy counts = conf_eval.json", True)
    ptx_up = {a: counts[a]["defer_units"].get("ptx_post", 0) + counts[a]["defer_units"].get("ptx_pre", 0)
              for a in ("MG:PMRT", "MG:GT")}
    sleep_units = {a: counts[a]["defer_units"].get("sleep_post", 0) + counts[a]["defer_units"].get("sleep_pre", 0)
                   for a in ("MG:PMRT", "MG:GT")}
    out["eval"] = {"n": st["n"], "seeds": [st["seeds"][0], st["seeds"][-1]], "V_AA": st["V_AA"], "V_ref": st["V_ref"],
                   "ref_arm": st["ref_arm"], "den": st["den"], "arms": P, "D1": D1, "D2": D2, "never_sleep": ns,
                   "E": E_pass, "verdict": verdict, "ptx_up_defer_units": ptx_up, "sleep_defer_units": sleep_units,
                   "counts": {a: counts[a] for a in ("MG:PMRT", "MG:GT", "never_sleep", "MG:granger_by", "MG:corr@dev")},
                   "secondary": {c: {"dR": P["MG:PMRT"]["R"] - P[c]["R"], "ci90": ci(B["MG:PMRT"]["R"] - B[c]["R"])}
                                 for c in ("noarb", "incumbent", "blanket2", "never_sleep", "B2", "MG:GT", "MG:rand")},
                   "run": led}
    return out


# ======================================================================================== section 6: 2b certsafe
def tau_table(G, N):
    return {(f, d, k): (RHO - 1.0) * G[k] / (max(N.get((f, d), 0.0), 1.0) * 3)
            for f in FAMILIES for d in (1, -1) for k in GUARD_KPIS}


def bound_ub(b, h):
    if not b:
        return math.inf
    if b["kind"] == "slope":
        if b.get("beta") is None or b.get("se") is None:
            return math.inf
        return h * b["beta"] + Z90 * b["se"]
    return max(h * b["lo"], h * b["hi"])


def certsafe():
    art = J(os.path.join(ART, "E6P_CERTSAFE.json"))
    cal = J(os.path.join(DEC, "cs_calib.json"))
    out = {"protocol_sha256": sha_lf(os.path.join(BENCH, "E6P_CERTSAFE_PROTOCOL.md")),
           "artifact_sha256": sha_lf(os.path.join(ART, "E6P_CERTSAFE.json"))}
    check("2b: artifact sha256 = .sha256 file",
          out["artifact_sha256"] == open(os.path.join(ART, "E6P_CERTSAFE.json.sha256")).read().split()[0])
    check("2b: protocol sha256 = artifact record", out["protocol_sha256"] == art["protocol"]["sha256"])
    # ---- calibration from the RAW DEV records
    Rd, _ = load_jobs([os.path.join(RUNS, "e6p-cs-dev-1", "all.jsonl")], "certsafe", "cs_stage", "dev")
    seeds = sorted(s for s in Rd if "CAL:allaccept" in Rd[s])
    calr = [Rd[s]["CAL:allaccept"] for s in seeds]
    G = {"v": float(np.mean([r["viol_ue_s"] for r in calr])), "rlf": float(np.mean([r["rlf"] for r in calr]))}
    N = {(f, d): float(np.mean([float(((r.get("policy_counts") or {}).get("policy") or {}).get(f"dir_{f}_{d:+d}", 0))
                                for r in calr])) for f in FAMILIES for d in (1, -1)}
    rep = {s: max(abs(float(Rd[s]["CAL:allaccept"][f]) - float(Rd[s]["noarb"][f])) for f in SUM_FIELDS)
           for s in (191000, 191001)}
    check("2b: G_k, N(f, d) recomputed from the raw DEV records = cs_calib.json",
          len(seeds) == 40 == cal["n"] and all(close(G[k], cal["G"][k]) for k in GUARD_KPIS)
          and all(close(N[(f, int(d))], v) for f, d, v in cal["N"]) and all(v == 0 for v in rep.values()))
    tau = tau_table(G, N)
    check("2b: tau = .05 G_k / (max(N, 1) * 3) = artifact tau",
          all(close(tau[(f, int(d), k)], v) for f, d, k, v in art["tau"]))
    # ---- certification table of every CS arm
    bset = {m: {(f, r, k): v for f, r, k, v in rows} for m, rows in art["bounds"].items()}
    cert = {}
    for arm, a in art["arms"].items():
        mkeys = {(e[0], e[1], e[2]) for e in a["map"]}
        b = bset.get(a["bound_method"]) if a["bound_method"] else None
        unc = []
        for cls, c in a["certification"].items():
            f, d = cls.split("|")[0], int(cls.split("|")[1])
            if not c["can_defer"]:
                continue
            ok = True
            for i, rel in enumerate(RELATIONS):
                for j, k in enumerate(GUARD_KPIS):
                    t = tau[(f, d, k)]
                    if (f, rel, k) in mkeys:
                        stt = "DECLARED"
                    else:
                        ub = bound_ub((b or {}).get((f, rel, k)), -d)
                        stt = "CERTIFIED" if ub <= t else "UNRESOLVED"
                        stored = c["edges"][i * 2 + j]
                        assert stored[3] is None and not math.isfinite(ub) or close(stored[3], ub), (arm, cls, rel, k)
                    assert c["edges"][i * 2 + j][:3] == [rel, k, stt], (arm, cls, c["edges"][i * 2 + j], stt)
                    ok &= stt != "UNRESOLVED"
            assert ok == c["certified"], (arm, cls)
            if not ok:
                unc.append([f, d])
        assert sorted(unc) == sorted(a["uncertified"]), (arm, unc, a["uncertified"])
        cert[arm] = {"uncertified": unc, "bound_method": a["bound_method"], "alias_of": art["alias_of"][arm],
                     "decision_table": a["decision_table"],
                     "classes": {cls: c for cls, c in a["certification"].items() if c["can_defer"]}}
    check("2b: certification (edge statuses, upper bounds, uncertified classes) of all 17 CS arms = artifact", len(cert) == 17)
    out["calib"] = {"G": G, "N": {f"{f}|{d:+d}": v for (f, d), v in N.items()}, "n_dev": len(seeds), "repro": rep}
    out["tau"] = {f"{f}|{d:+d}|{k}": v for (f, d, k), v in tau.items()}
    out["cert"] = cert
    out["jobs"] = art["jobs"]
    bp = bset["pmrt"]
    out["pmrt_bound_ptx_nbr_rlf"] = {"beta": bp[("ptx", "nbr", "rlf")]["beta"], "se": bp[("ptx", "nbr", "rlf")]["se"],
                                     "z": bp[("ptx", "nbr", "rlf")]["z"], "ub90_defer_up": bound_ub(bp[("ptx", "nbr", "rlf")], -1),
                                     "tau": tau[("ptx", 1, "rlf")]}
    # ---- EVAL from RAW
    files = sorted(glob.glob(os.path.join(RUNS, "e6p-cs-eval-1", "*", "res_*.jsonl")))
    check("2b: all 20 raw EVAL shards present", len(files) == 20, len(files))
    closes = 0
    for f in files:
        lines = open(f, encoding="utf-8").read().splitlines()
        closes += json.loads(lines[-1]).get("kind") == "close"
    check("2b: every shard ends with a close record", closes == 20, closes)
    R, heads = load_jobs(files, "certsafe", "cs_stage", "eval")
    led = run_ledger(heads, R)
    check("2b: eval headers carry the frozen certsafe.py sha256",
          {h.get("certsafe_sha256") for h in heads} == {art["sha256"]["cdd_oran/decision/certsafe.py"]})
    alias_fill(R, art["alias_of"])
    arms = [a for a in CS_ALL if any(a in R[s] for s in R)]
    st = gate_a(R, arms)
    E = J(os.path.join(DEC, "cs_eval.json"))
    Vd = J(os.path.join(DEC, "cs_verdict.json"))
    check("2b: 160 complete seeds 191100-191259, every arm", st["n"] == 160 == E["n_seeds"] and len(arms) == 26
          and st["seeds"] == list(range(191100, 191260)))
    check("2b: Gate A constants", close(st["V_AA"], E["gate_a"]["V_AA"]) and close(st["den"], E["gate_a"]["den"]))
    check_arms("2b", st, E["arms"])
    check("2b: every arm's V, R, R*, retention, guard ratios, eligibility and 90% CIs = cs_eval.json", True)
    P, B = st["point"], st["boot"]
    sig = {a: v["signature"] for a, v in art["arms"].items()}
    D1 = d1(P, B, "CS:PMRT", list(CS_ASSOC))
    D2 = d2(B, "CS:PMRT", list(CS_ASSOC), sig)
    dd = float(P["CS:PMRT"]["Rstar"] - P["never_sleep"]["Rstar"])
    diff3 = B["CS:PMRT"]["Rstar"] - B["never_sleep"]["Rstar"]
    lb3 = float(np.quantile(diff3[np.isfinite(diff3)], 0.05))
    D3 = {"pass": bool(dd > 0 and lb3 > 0), "delta3": dd, "lb90": lb3, "ci90": ci(diff3)}
    rr = B["CS:PMRT"]["guard"]["rlf"]
    ub = float(np.quantile(rr[np.isfinite(rr)], 0.95))
    S1 = {"pass": ub <= RLF_CAP, "rlf_ratio_ub90": ub}
    check("2b: D1", D1["pass"] == Vd["D1"]["pass"] and close(D1["delta1"], Vd["D1"]["delta1"])
          and close(D1["lb90"], Vd["D1"]["lb90"]) and D1["best_point"] == Vd["D1"]["best_point"])
    check("2b: D2 (Holm, 7 signature groups)", D2["pass"] == Vd["D2"]["pass"] and D2["n_hypotheses"] == Vd["D2"]["n_hypotheses"]
          and all(close(D2["hypotheses"][b]["p"], Vd["D2"]["hypotheses"][b]["p"])
                  and D2["hypotheses"][b]["members"] == Vd["D2"]["hypotheses"][b]["members"] for b in D2["hypotheses"]))
    check("2b: D3", D3["pass"] == Vd["D3"]["pass"] and close(D3["delta3"], Vd["D3"]["delta3"]) and close(D3["lb90"], Vd["D3"]["lb90"]))
    check("2b: S1", S1["pass"] == Vd["S1"]["pass"] and close(S1["rlf_ratio_ub90"], Vd["S1"]["rlf_ratio_ub90"]))
    E_pass = P["CS:PMRT"]["eligible"]
    n_ok = D1["pass"] + D2["pass"] + D3["pass"]
    verdict = "NOT ELIGIBLE" if not E_pass else ("PASS" if n_ok == 3 else ("PARTIAL" if n_ok else "FAIL"))
    check("2b: verdict", verdict == Vd["verdict"]["verdict"] == "PARTIAL", verdict)
    counts = {a: {g: sum_counts(R, st["seeds"], a, g) for g in ("defer", "defer_units", "units", "policy")} for a in arms}
    for a in arms:
        for g in ("defer", "defer_units", "units", "policy"):
            assert counts[a][g] == E["arms"][a]["counts"][g], (a, g)
    check("2b: deferral / unit / policy counts = cs_eval.json", True)
    # ---- per-seed identity CS:PMRT vs never_sleep
    same, same_defer, fields = 0, 0, None
    for s in st["seeds"]:
        a, b = R[s]["CS:PMRT"], R[s]["never_sleep"]
        ks = sorted(k for k in a if k not in OUTCOME_SKIP)
        fields = ks
        same += all(a[k] == b[k] for k in ks) and set(ks) == {k for k in b if k not in OUTCOME_SKIP}
        same_defer += a["policy_counts"]["defer"] == b["policy_counts"]["defer"]
    num_fields = [k for k in fields if isinstance(R[st["seeds"][0]]["CS:PMRT"][k], (int, float))
                  and not isinstance(R[st["seeds"][0]]["CS:PMRT"][k], bool)]
    check("2b: CS:PMRT record identical to never_sleep on every seed (all outcome fields, deferred requests)",
          same == 160 and same_defer == 160, (same, same_defer))
    pc = counts["CS:PMRT"]["policy"]
    check("2b: CS:PMRT deferred only sleep-up units", {k for k in pc if k.startswith("defer_")} == {"defer_sleep"}
          and pc["defer_sleep"] == pc["dir_sleep_+1"])
    comp = {}
    for a in ("MG:PMRT", "CS:PMRT", "CS:GT", "CS:corr@dev", "CS:granger_by", "CS:granger@dev", "CS:two_tower@dev",
              "CS:shap_gbdt@dev", "CS:qacm@dev"):
        p = counts[a]["policy"]
        comp[a] = {"defer": {f: p.get(f"defer_{f}", 0) for f in FAMILIES},
                   "withheld": {f: p.get(f"uncert_{f}", 0) for f in FAMILIES}}
    comp["never_sleep"] = {"defer_requests": counts["never_sleep"]["defer"]}
    comp["CS:PMRT"]["defer_requests"] = counts["CS:PMRT"]["defer"]
    out["eval"] = {"n": st["n"], "seeds": [st["seeds"][0], st["seeds"][-1]], "V_AA": st["V_AA"], "V_ref": st["V_ref"],
                   "ref_arm": st["ref_arm"], "den": st["den"], "arms": P, "D1": D1, "D2": D2, "D3": D3, "S1": S1,
                   "E": E_pass, "verdict": verdict, "identical_seeds": same, "identical_fields": num_fields,
                   "n_identical_fields": len(fields), "composition": comp,
                   "secondary": {c: {"dR": P["CS:PMRT"]["R"] - P[c]["R"], "ci90": ci(B["CS:PMRT"]["R"] - B[c]["R"])}
                                 for c in ("noarb", "incumbent", "never_sleep", "MG:PMRT", "CS:GT")},
                   "run": led, "cpu_h_eval_stored": E["cpu_h_eval"]}
    return out


# ======================================================================================== figures
def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=140, bbox_inches="tight"); plt.close(fig)


MLBL = {PRIMARY: "PMRT (primary)", "granger": "Granger", "granger_by": "Granger + BY", "corr": "|corr|",
        "shap_gbdt": "SHAP-GBDT", "qacm": "QACM", "two_tower": "two-tower", "int": "PACIFISTA INT",
        "v2+by": "MSCR-CRT v2 (v3 method)"}


def fig1(v):
    ms = [PRIMARY, "v2+by", "granger_by", "granger", "corr", "int", "qacm", "two_tower", "shap_gbdt"]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), gridspec_kw={"width_ratios": [1.15, 1.25, 0.9]})
    ax = axes[0]
    y = np.arange(len(ms))[::-1]; h = 0.38
    f1 = [np.nan_to_num(v["pooled"][m]["ind_f1"]) for m in ms]
    pr = [np.nan_to_num(v["pooled"][m]["ov_precision"]) for m in ms]
    ax.barh(y + h / 2, f1, height=h * 0.92, color=BLUE, label="indirect (nbr) F1")
    ax.barh(y - h / 2, pr, height=h * 0.92, color=LIGHT, edgecolor=SOFT, label="overall precision")
    for yi, a, b in zip(y, f1, pr):
        ax.text(a + 0.01, yi + h / 2, f"{a:.2f}", va="center", fontsize=7.4)
        ax.text(b + 0.01, yi - h / 2, f"{b:.2f}", va="center", fontsize=7.0, color=SOFT)
    ax.set_yticks(y); ax.set_yticklabels([MLBL[m] for m in ms], fontsize=8.4)
    ax.set_xlim(0, 1.12); ax.grid(axis="x", color=GRID, lw=0.7)
    ax.axvline(0.80, color=RED, ls="--", lw=1.0)
    ax.legend(fontsize=7.6, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.08), ncol=2)
    ax.set_title("(a) pooled EVAL (600 episodes) vs gt_v4", loc="left", fontsize=10)
    ax = axes[1]
    names = v["slices"]["names"]
    groups = [("n 60", [s for s in names if s.startswith("s60_")]), ("n 120", [s for s in names if s.startswith("s120_")]),
              ("n 300", [s for s in names if s.startswith("s300_")]), ("pooled", ["pooled"])]
    cols = {PRIMARY: BLUE, "granger_by": ORANGE, "granger": VIOLET, "corr": GREY}
    offs = {PRIMARY: -0.27, "granger_by": -0.09, "granger": 0.09, "corr": 0.27}
    for gi, (gl, ss) in enumerate(groups):
        for m in (PRIMARY, "granger_by", "granger", "corr"):
            vals = [v["per_slice"][m][s]["ov_precision"] for s in ss]
            ax.scatter([gi + offs[m]] * len(vals), vals, s=22, color=cols[m], alpha=0.85, zorder=3,
                       label=MLBL[m] if gi == 0 else None)
    ax.axhline(0.80, color=RED, ls="--", lw=1.0)
    ax.text(3.45, 0.805, "P3 bar .80", color=RED, fontsize=7.4, ha="right", va="bottom")
    ax.set_xticks(range(4)); ax.set_xticklabels([g for g, _ in groups], fontsize=8.6)
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("overall precision"); ax.grid(axis="y", color=GRID, lw=0.7)
    ax.legend(fontsize=7.4, frameon=False, loc="lower left", ncol=2)
    p2 = v["P2"]["granger_by"]
    ax.set_title(f"(b) precision per slice; P2 vs Granger + BY: {p2['wins60']}/10 at n 60, {p2['wins120']}/5 at n 120",
                 loc="left", fontsize=9.6)
    ax = axes[2]
    pl = [v["placebo_declarations"][m] if m != PRIMARY else v["placebo_declarations"][PRIMARY] for m in ms]
    ax.barh(y, pl, height=0.6, color=[GREEN if p == 0 else ORANGE for p in pl])
    for yi, p in zip(y, pl):
        ax.text(p + 0.2, yi, str(p), va="center", fontsize=8, fontweight="bold")
    ax.axvline(1, color=RED, ls="--", lw=1.1)
    ax.text(1.3, y[-1] - 0.45, "K0 allows ≤ 1", color=RED, fontsize=7.4)
    ax.set_yticks(y); ax.set_yticklabels([MLBL[m] for m in ms], fontsize=8.4)
    ax.set_xlim(0, 15); ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_title("(c) declarations on placebo_v4\n     (40 eps, sharp null)", loc="left", fontsize=10)
    fig.suptitle("Fig 1: Discovery v4: PMRT is precise on every slice and declares nothing on placebo; Granger + BY wins "
                 "the small-slice F1 race with 13 placebo declarations", x=0.01, ha="left", fontweight="bold",
                 fontsize=10.8, y=1.04)
    fig.text(0.01, -0.06, "corr and Granger use the DEV-v4 far-FPR τ; Granger + BY is untuned; SHAP-GBDT, INT, QACM and "
             "two-tower are descriptive (DEV τ, scored on pooled, 300 and 120 slices). MSCR-CRT v2 is the v3 method, run "
             "as a descriptive comparator. Placebo for PMRT: the primary combination (all 9 combinations declare 0).",
             fontsize=8.0, color=SOFT)
    fig.tight_layout(); save(fig, "fig1_v4_methods.png")


def fig2(rn):
    cats = rn["analysis_v4.json"]["by_category"]
    order = ["criterion", "assoc", "desc_nn", "other", "volatile"]
    fig, ax = plt.subplots(figsize=(11.6, 3.8))
    y = np.arange(len(order))[::-1]
    lv = [cats[c]["leaves"] for c in order]
    df = [cats[c]["diffs"] for c in order]
    ax.barh(y, lv, color=LIGHT, edgecolor=SOFT, height=0.6)
    for yi, c, l, d in zip(y, order, lv, df):
        col = RED if d and c != "volatile" else (SOFT if c == "volatile" else GREEN)
        txt = f"{l:,} leaves · " + ("not compared" if c == "volatile" else f"{d} differ")
        ax.text(l * 1.15 + 3, yi, txt, va="center", fontsize=8.6, color=col, fontweight="bold")
    ax.set_xscale("log"); ax.set_xlim(5, 4e5)
    ax.set_yticks(y); ax.set_yticklabels([CAT_LABEL[c] for c in order], fontsize=8.4)
    ax.set_xlabel("leaves of analysis_v4.json (log scale)"); ax.grid(axis="x", color=GRID, lw=0.7, which="both")
    a = rn["analysis_v4.json"]; vv = rn["verdict_v4.json"]
    ax.set_title(f"Fig 2: Rename check on the full v4 data: verdict tree {vv['n_diff']} of {vv['n_leaves']} leaves differ; "
                 f"analysis tree {a['n_diff']} of {a['n_leaves']:,}, all in descriptive baselines",
                 loc="left", fontweight="bold", fontsize=10.4)
    fig.text(0.01, -0.1, "Frozen code (e6p-disc-v4an-1) vs renamed code + E6P_PMRT_V4.json (e6p-disc-v4an-pmrt-1), same "
             "inputs, two Kaggle machines. Comparison as pmrt_equivalence.py reports mode: exact, NaN = NaN, \"MSCR+\" read "
             "as \"PMRT\". The 14 differences: two_tower τ and QACM's declared list on slice s300_1.", fontsize=8.0, color=SOFT)
    fig.tight_layout(); save(fig, "fig2_rename_diff.png")


ALBL = {"MG:PMRT": "PMRT map", "MG:GT": "GT map (oracle)", "MG:rand": "random map", "blanket2": "blanket2 (static)",
        "MG:granger_by": "Granger + BY", "MG:granger@dev": "Granger (DEV τ)", "MG:corr@dev": "|corr| (DEV τ)",
        "MG:two_tower@dev": "two-tower (DEV τ)", "MG:shap_gbdt@dev": "SHAP-GBDT (DEV τ)", "MG:qacm@dev": "QACM (DEV τ)",
        "MG:int@dev": "INT (DEV τ)"}


def fig3(a):
    arms = ["MG:PMRT", "MG:GT", "MG:granger_by", "MG:granger@dev", "MG:corr@dev", "MG:two_tower@dev", "MG:shap_gbdt@dev",
            "MG:qacm@dev", "MG:int@dev", "MG:rand"]
    mq = a["map_quality"]
    fig, ax = plt.subplots(figsize=(12.2, 4.8))
    y = np.arange(len(arms))[::-1]
    parts = [("true_right", GREEN, "GT TRUE, right sign"), ("true_wrong", RED, "GT TRUE, WRONG sign"),
             ("null", ORANGE, "GT NULL"), ("indet", LIGHT, "GT INDET")]
    left = np.zeros(len(arms))
    for key, col, lab in parts:
        vals = np.array([mq[x][key] for x in arms], float)
        ax.barh(y, vals, left=left, color=col, height=0.6, label=lab, edgecolor="white", lw=0.5)
        left += vals
    for yi, x in zip(y, arms):
        q = mq[x]
        pl = q["placebo_edges"]
        ax.text(q["edges"] + 0.6, yi, f"{q['edges']} edges · {q['true_right']} right · {q['true_wrong']} wrong-sign"
                + ("" if pl is None else f" · {pl} on placebo"), va="center", fontsize=8.0,
                color=RED if (pl or 0) > 1 or q["true_wrong"] else INK)
    ax.set_yticks(y); ax.set_yticklabels([ALBL[x] for x in arms], fontsize=8.6)
    ax.set_xlim(0, 80); ax.set_xlabel("map edges (60 possible: 4 families x 3 relations x 5 KPIs)")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.legend(fontsize=7.8, frameon=False, loc="lower right", ncol=2)
    ax.set_title("Fig 3: Option (a) maps from confounded incumbent logs: PMRT has no wrong-sign and no placebo edges; "
                 "the associational maps have both", loc="left", fontweight="bold", fontsize=10.4)
    fig.text(0.01, -0.05, "Scored against the option (a) GT (40 knock-out episodes; 29 TRUE / 24 NULL / 7 INDET). \"on "
             "placebo\" = edges the same method declares on 200 placebo episodes (modes logged, accept applied). The "
             "GT and random maps have no placebo map.", fontsize=8.0, color=SOFT)
    fig.tight_layout(); save(fig, "fig3_conf_maps.png")


def _rplot(ax, P, rows, title, cut=-1.5):
    y = np.arange(len(rows))[::-1]
    for yi, (lab, a) in zip(y, rows):
        r, (lo, hi) = P[a]["R"], P[a]["R_ci90"]
        col = GREEN if P[a]["eligible"] else ORANGE
        if r < cut:
            ax.barh(yi, cut, color=col, height=0.55, alpha=0.25)
            ax.text(cut + 0.05, yi, f"R = {r:+.2f} [{lo:+.1f}, {hi:+.1f}] (off scale)", va="center", fontsize=7.8,
                    color=INK, fontweight="bold")
            continue
        ax.barh(yi, r, color=col, height=0.55)
        ax.errorbar(r, yi, xerr=[[r - lo], [hi - r]], fmt="none", ecolor=INK, lw=1.0, capsize=3)
        ax.text(max(hi, 0) + 0.03, yi, f"{r:+.3f} [{lo:+.2f}, {hi:+.2f}]", va="center", fontsize=7.6)
    ax.axvline(0, color=SOFT, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8.2)
    ax.set_xlim(cut, 1.0); ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_xlabel("R = (V_AA − V_arm) / (V_AA − V_ref), 90 % CI")
    ax.set_title(title, loc="left", fontsize=10)


def fig4(a):
    P = a["eval"]["arms"]
    rows = [("MG:PMRT (PMRT map)", "MG:PMRT"), ("MG:GT (oracle map)", "MG:GT"), ("never_sleep (static)", "never_sleep"),
            ("blanket2 (static)", "blanket2"), ("B2 (static)", "B2"), ("MG:corr@dev", "MG:corr@dev"),
            ("MG:qacm@dev", "MG:qacm@dev"), ("incumbent", "incumbent"), ("MG:rand", "MG:rand"),
            ("MG:granger_by", "MG:granger_by"), ("MG:shap_gbdt@dev", "MG:shap_gbdt@dev"),
            ("MG:two_tower@dev", "MG:two_tower@dev"), ("MG:granger@dev", "MG:granger@dev")]
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6), gridspec_kw={"width_ratios": [1.45, 1]})
    _rplot(axes[0], P, rows, "(a) recovery R (green = eligible, orange = not)")
    ax = axes[1]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, x) in zip(y, rows):
        g = P[x]["guard_ratio"]["rlf"]; lo, hi = P[x]["guard_ratio_ci90"]["rlf"]
        ax.errorbar(g, yi, xerr=[[g - lo], [hi - g]], fmt="o", color=RED if g > GUARD else INK, ms=5, capsize=3, lw=1.0)
        ax.text(hi + 0.02, yi, f"{g:.2f}", va="center", fontsize=7.6, color=RED if g > GUARD else SOFT)
    ax.axvline(GUARD, color=RED, ls="--", lw=1.2); ax.axvline(1.0, color="#bdb8ad", lw=0.9)
    ax.text(GUARD + 0.01, y[0] + 0.5, "guard 1.10× accept-all", color=RED, fontsize=7.6)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8.2)
    ax.set_xlim(0.85, 1.7); ax.set_ylim(-0.6, len(rows) - 0.1); ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_xlabel("RLF ratio vs accept-all (pooled, 90 % CI)")
    ax.set_title("(b) radio-link failure guard", loc="left", fontsize=10)
    e = a["eval"]
    fig.suptitle(f"Fig 4: Option (a) EVAL ({e['n']} seeds): the PMRT-map referee has the highest R of all arms but "
                 f"breaks the RLF guard ({P['MG:PMRT']['guard_ratio']['rlf']:.2f}×), so the verdict is NOT ELIGIBLE",
                 x=0.01, ha="left", fontweight="bold", fontsize=10.6, y=1.02)
    fig.text(0.01, -0.04, f"V_AA {e['V_AA']:.2f}, V_ref {e['V_ref']:.2f} (SliceGuarantee alone), denominator {e['den']:.2f}. "
             "Eligible = energy retention ≥ 0.90 and all four guard ratios (SVR, non-protected eMBB, LL, RLF) ≤ 1.10. "
             "Paired-seed bootstrap, default_rng([6624, 20, 160]), 10 000 resamples. Arms aliased to accept-all "
             "(corr@plc, granger@plc, int, two_tower@plc) not shown (R 0).", fontsize=8.0, color=SOFT)
    fig.tight_layout(); save(fig, "fig4_conf_eval.png")


def fig5(a, c):
    fams = ["sleep", "ptx", "carrier", "prot_min"]
    arms = [("option (a)\nMG:PMRT", a["eval"]["counts"]["MG:PMRT"]["policy"], None),
            ("option (a)\nMG:GT", a["eval"]["counts"]["MG:GT"]["policy"], None),
            ("2b seeds\nMG:PMRT", None, "MG:PMRT"), ("2b\nCS:PMRT", None, "CS:PMRT"), ("2b\nCS:GT", None, "CS:GT")]
    fig, ax = plt.subplots(figsize=(11.8, 4.6))
    x = np.arange(len(arms)); w = 0.18
    fc = {"sleep": BLUE, "ptx": RED, "carrier": AQUA, "prot_min": YELLOW}
    for j, f in enumerate(fams):
        vals, wh = [], []
        for lab, pol, key in arms:
            if pol is not None:
                vals.append(pol.get(f"defer_{f}", 0)); wh.append(0)
            else:
                cc = c["eval"]["composition"][key]
                vals.append(cc["defer"][f]); wh.append(cc["withheld"][f])
        xx = x + (j - 1.5) * w
        ax.bar(xx, vals, width=w * 0.92, color=fc[f], label=f"deferred {f} units")
        ax.bar(xx, wh, width=w * 0.92, bottom=vals, color="white", edgecolor=fc[f], hatch="////",
               label="withheld (uncertified, accepted)" if j == 1 else None)
        for xi, v_, h_ in zip(xx, vals, wh):
            if v_ + h_:
                lab = f"{h_:,} withheld" if (h_ and not v_) else (f"{v_:,}" + (f" +{h_:,} withheld" if h_ else ""))
                ax.text(xi, v_ + h_ + 120, lab, ha="center", fontsize=6.9)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in arms], fontsize=8.6)
    ax.set_ylim(0, 10500); ax.set_ylabel("units deferred, summed over 160 seeds")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.legend(fontsize=7.6, frameon=False, loc="upper right", ncol=2)
    pu = a["eval"]["ptx_up_defer_units"]
    ax.set_title(f"Fig 5: The RLF mechanism: without a ptx → nbr RLF edge the PMRT-map referee defers {pu['MG:PMRT']:,} "
                 f"power-up units (GT map: {pu['MG:GT']:,}); the 2b gate withholds them", loc="left", fontweight="bold",
                 fontsize=10.4)
    p = a["ptx_nbr_rlf"]
    fig.text(0.01, -0.06, f"GT ptx → nbr RLF {p['gt']['mean']:+.3f} (95 % CI {p['gt']['ci'][0]:+.3f}, {p['gt']['ci'][1]:+.3f}): "
             "power-up lowers neighbour RLF, so deferring power-ups raises it. PMRT on the confounded DISC logs: plain_c "
             f"z {p['plain_c_z']:+.2f}, primary loadsp_c z {p['loadsp_c_z']:+.2f} (p {p['loadsp_c_p2']:.3f}), not declared. "
             "Option (a) and 2b use different EVAL seeds (187000-159, 191100-259).", fontsize=8.0, color=SOFT)
    fig.tight_layout(); save(fig, "fig5_ptx_deferrals.png")


def fig6(c):
    P = c["eval"]["arms"]
    rows = [("CS:PMRT (gated PMRT map)", "CS:PMRT"), ("never_sleep (static)", "never_sleep"),
            ("CS:GT (gated oracle map)", "CS:GT"), ("MG:PMRT (ungated, as option (a))", "MG:PMRT"),
            ("CS:corr@dev", "CS:corr@dev"), ("CS:qacm@dev", "CS:qacm@dev"), ("incumbent", "incumbent"),
            ("CS:granger_by", "CS:granger_by"), ("CS:shap_gbdt@dev", "CS:shap_gbdt@dev"),
            ("CS:two_tower@dev", "CS:two_tower@dev"), ("CS:granger@dev", "CS:granger@dev")]
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2), gridspec_kw={"width_ratios": [1.45, 1]})
    _rplot(axes[0], P, rows, "(a) recovery R (green = eligible, orange = not)")
    axes[0].annotate("", xy=(P["CS:PMRT"]["R"] + 0.33, len(rows) - 1.02), xytext=(P["CS:PMRT"]["R"] + 0.33, len(rows) - 1.98),
                     arrowprops=dict(arrowstyle="<->", color=RED, lw=1.2))
    axes[0].text(P["CS:PMRT"]["R"] + 0.36, len(rows) - 1.5, f"identical on all\n{c['eval']['identical_seeds']} seeds (Δ = 0)",
                 fontsize=7.6, color=RED, va="center")
    ax = axes[1]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, x) in zip(y, rows):
        g = P[x]["guard_ratio"]["rlf"]; lo, hi = P[x]["guard_ratio_ci90"]["rlf"]
        ax.errorbar(g, yi, xerr=[[g - lo], [hi - g]], fmt="o", color=RED if g > GUARD else INK, ms=5, capsize=3, lw=1.0)
        ax.text(hi + 0.02, yi, f"{g:.2f} (UB {hi:.2f})", va="center", fontsize=7.4, color=SOFT)
    ax.axvline(GUARD, color=RED, ls="--", lw=1.2); ax.axvline(RLF_CAP, color=VIOLET, ls=":", lw=1.3)
    ax.axvline(1.0, color="#bdb8ad", lw=0.9)
    ax.text(GUARD + 0.01, y[-1] - 0.5, "point guard 1.10", color=RED, fontsize=7.4)
    ax.text(RLF_CAP + 0.01, y[0] + 0.3, "S1 cap on UB90: 1.20", color=VIOLET, fontsize=7.4)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8.2)
    ax.set_xlim(0.8, 1.45); ax.set_ylim(-0.9, len(rows) - 0.1); ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_xlabel("RLF ratio vs accept-all (pooled, 90 % CI)")
    ax.set_title("(b) radio-link failure guard", loc="left", fontsize=10)
    e = c["eval"]
    fig.suptitle(f"Fig 6: 2b EVAL ({e['n']} fresh seeds): the gated PMRT referee is eligible and safe, beats every gated "
                 "associational map (D1, D2), but equals never_sleep exactly (D3 fails) → PARTIAL", x=0.01, ha="left",
                 fontweight="bold", fontsize=10.4, y=1.02)
    fig.text(0.01, -0.04, f"V_AA {e['V_AA']:.2f}, V_ref {e['V_ref']:.2f}, denominator {e['den']:.2f}. Same Gate A and "
             "bootstrap as option (a). Arms aliased to accept-all (CS:rand, CS:blanket2, corr@plc, granger@plc, "
             "shap_gbdt@plc, int, two_tower@plc) have R 0 and are not shown; CS:qacm@plc = CS:qacm@dev.",
             fontsize=8.0, color=SOFT)
    fig.tight_layout(); save(fig, "fig6_certsafe_eval.png")


def fig7(c):
    cert = c["cert"]
    arms = ["CS:PMRT", "CS:GT", "CS:corr@dev", "CS:granger@dev", "CS:granger_by", "CS:two_tower@dev", "CS:shap_gbdt@dev",
            "CS:qacm@dev"]
    classes = ["carrier|1", "carrier|-1", "sleep|1", "sleep|-1", "ptx|1", "ptx|-1", "prot_min|1", "prot_min|-1"]
    fig, ax = plt.subplots(figsize=(12.0, 4.4))
    for i, a in enumerate(arms):
        for j, cl in enumerate(classes):
            cc = cert[a]["classes"].get(cl)
            if cc is None:
                col, txt = "#f2f0ea", "never\ndeferred"
            else:
                st = [e[2] for e in cc["edges"]]
                if cc["certified"]:
                    col = "#cfe9d9"; txt = f"certified\n{st.count('DECLARED')}D {st.count('CERTIFIED')}C"
                else:
                    col = "#f6d2c8"; txt = f"UNCERTIFIED\n{st.count('UNRESOLVED')} unresolved"
            ax.add_patch(plt.Rectangle((j, len(arms) - 1 - i), 0.96, 0.92, color=col))
            ax.text(j + 0.48, len(arms) - 1 - i + 0.46, txt, ha="center", va="center", fontsize=6.9)
    ax.set_xlim(0, len(classes)); ax.set_ylim(0, len(arms))
    ax.set_xticks(np.arange(len(classes)) + 0.48)
    ax.set_xticklabels([cl.replace("|1", " +").replace("|-1", " −") for cl in classes], fontsize=8.4)
    ax.set_yticks(np.arange(len(arms)) + 0.46); ax.set_yticklabels(arms[::-1], fontsize=8.4)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Fig 7: Certification of each map's deferrable (family, direction) classes on the guard edges "
                 "(v, RLF) x (own, nbr, far): uncertified classes are accepted", loc="left", fontweight="bold",
                 fontsize=10.4)
    t = c["pmrt_bound_ptx_nbr_rlf"]
    fig.text(0.01, -0.05, "D = edge declared in the map (MapGateV2 logic applies), C = certified: one-sided UB90 of the "
             "deferral's harm ≤ τ = .05·G_k / (N(f, d)·3). Bounds: PMRT design slope + martingale SE (CS:PMRT), GT 95 % CI "
             f"(CS:GT), naive OLS (associational). CS:PMRT ptx + on nbr RLF: UB90 {t['ub90_defer_up']:.3f} vs τ {t['tau']:.3f}.",
             fontsize=8.0, color=SOFT)
    fig.tight_layout(); save(fig, "fig7_certification.png")


# ======================================================================================== output
def _r(x, n=5):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if x != x else (str(x) if math.isinf(x) else float(f"{float(x):.{n + 2}g}"))
    if isinstance(x, dict):
        return {str(k): _r(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, n) for v in x]
    return x


def main():
    v = v4()
    rn = rename()
    a = opt_a()
    c = certsafe()
    fig1(v); fig2(rn); fig3(a); fig4(a); fig5(a, c); fig6(c); fig7(c)
    data = {"v4": v, "rename": rn, "option_a": a, "certsafe": c, "assertions": CHECKS}
    json.dump(_r(data), open(os.path.join(OUT, "figures_data.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    # ---- console summary
    print("v4:", v["verdict"], "K0", _r(v["K0"]), "P2", {b: (x["wins60"], x["wins120"]) for b, x in v["P2"].items()})
    print("   P3", _r(v["P3_values"]), "S", _r(v["S"]), "placebo", v["placebo_declarations"])
    print("   ranges", json.dumps(_r(v["ranges"])))
    print("   pooled", json.dumps(_r(v["pooled"])))
    print("rename:", {k: (x["n_leaves"], x["n_diff"]) for k, x in rn.items() if k.endswith(".json")},
          rn["analysis_v4.json"]["by_category"], "addendum agrees", rn["addendum_agrees"])
    for d in rn["analysis_v4.json"]["diffs"]:
        print("   diff", d)
    print("opt(a) disc:", json.dumps(_r(a["disc"]))[:900])
    print("   maps:", {k: (q["edges"], q["true_right"], q["true_wrong"], q["null"], q["indet"], q["placebo_edges"])
                       for k, q in a["map_quality"].items()})
    print("   ptx nbr rlf:", _r(a["ptx_nbr_rlf"]), "gt rlf", _r(a["gt_rlf"]))
    e = a["eval"]
    print("   eval", e["n"], "den", round(e["den"], 3), "verdict", e["verdict"], "D1", _r(e["D1"]), "D2 pass", e["D2"]["pass"],
          "ns", _r(e["never_sleep"]), "ptx-up", e["ptx_up_defer_units"], "sleep", e["sleep_defer_units"], "run", _r(e["run"]))
    for k, p in e["arms"].items():
        print(f"     {k:20s} R {p['R']:+.3f} {np.round(p['R_ci90'], 3)} ret {p['retention']:.3f} "
              f"g {_r(p['guard_ratio'], 3)} rlfci {np.round(p['guard_ratio_ci90']['rlf'], 3)} el {p['eligible']}")
    print("2b: tau", _r(c["tau"], 3), "calib", _r(c["calib"]))
    print("   uncertified", {k: x["uncertified"] for k, x in c["cert"].items()}, "ptxbound", _r(c["pmrt_bound_ptx_nbr_rlf"]))
    e = c["eval"]
    print("   eval", e["n"], "den", round(e["den"], 3), "verdict", e["verdict"], "D1", _r(e["D1"]), "D2",
          {b: (h["p"], h["rejected"]) for b, h in e["D2"]["hypotheses"].items()}, "D3", _r(e["D3"]), "S1", _r(e["S1"]),
          "identical", e["identical_seeds"], e["n_identical_fields"], "run", _r(e["run"]))
    print("   comp", json.dumps(_r(e["composition"])))
    print("   secondary", _r(e["secondary"]))
    for k, p in e["arms"].items():
        print(f"     {k:20s} R {p['R']:+.3f} {np.round(p['R_ci90'], 3)} ret {p['retention']:.3f} "
              f"g {_r(p['guard_ratio'], 3)} rlfci {np.round(p['guard_ratio_ci90']['rlf'], 3)} el {p['eligible']}")
    print(f"ALL {len(CHECKS)} ASSERTION BLOCKS PASSED")
    for n_ in CHECKS:
        print("  ok:", n_)
    print("wrote fig1..fig7 + figures_data.json to", OUT)


if __name__ == "__main__":
    main()
