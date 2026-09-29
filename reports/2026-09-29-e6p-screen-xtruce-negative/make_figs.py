"""Figures for the 2026-09-29 report: E6-P Gate A v2 screen (DEAD) and xTRUCE XTS-v1 stage 0 (NOT SCREENABLE).

Reads the raw run files LIVE (json + numpy + matplotlib only; no import of the screen drivers) and recomputes,
independently of `e6p_screen.py summary` / `xtruce_screen.py summary`:
  * E6-P stage 0a load calibration, stage 0b M4 (8a amended rule), stage 0c oracle timing;
  * E6-P stage 1 arm tables per (pair, stratum): pooled PSVR V, energy retention, energy-matched, guardrails,
    recovery R, Lambda, V_AA - V_ref, paired-seed bootstrap LB90 (RNG [6611, pair_idx, stratum], 10 000 resamples);
  * E6-P stage 2 oracle: raw R, retention, guardrail ratios vs accept-all, rho_sign, n deviations;
  * POST-HOC diagnostics from the per-deviation records (true-tape vs redrawn advantage);
  * xTRUCE stage 0 M4 own-KPI per xApp x stratum, M6 sleep, 0c timing.
Emits fig1..fig6 (PNG) + figures_data.json next to this file.

Run (repo root):  PYTHONPATH=. .venv/Scripts/python.exe reports/2026-09-29-e6p-screen-xtruce-negative/make_figs.py
Frozen constants copied from docs/benchmark/E6P_SCREEN_PROTOCOL.md (sec. 7, 8a) and XTRUCE_SCREEN_PROTOCOL.md (sec. 6).
"""
import json
import os
from collections import defaultdict
from statistics import NormalDist

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUNS = os.path.join(ROOT, "scratchpad", "e6_dev", "runs")
OUT = HERE

# ---- frozen E6-P constants (E6P_SCREEN_PROTOCOL.md sec. 3, 7, 8a; E6P_SPEC.md sec. 5)
X_MATCH, SCREEN_FLOOR, GUARD = 0.90, 0.01, 1.10
LAMBDA_MIN, MATERIAL, R_OR_MIN, HEADROOM, RHO_MIN = 0.15, 36.0, 0.50, 0.10, 0.70
N_BOOT, BOOT_TAG, LB_Q = 10000, 6611, 0.10
N_SCREEN = 8
GUARD_KEYS = ("svr", "nonprot_embb_viol", "ll_viol", "rlf")
PAIRS = {"P1": dict(idx=1, xapps=("ES", "SliceGuarantee"), A="sub:ES"),
         "P2": dict(idx=2, xapps=("PowerES", "Coverage"), A="sub:PowerES"),
         "P3": dict(idx=3, xapps=("ES", "PowerES", "SliceGuarantee"), A="sub:ES+PowerES")}
STRATA = {0: "base-L10", 1: "base-L40", 2: "surge-L10", 3: "surge-L40"}
M4_OWN = {"ES": "energy", "PowerES": "energy", "SliceGuarantee": "psvr", "Coverage": "lowsinr"}
M4_MAT = {"psvr": 36.0, "lowsinr": 0.01}
# ---- frozen xTRUCE constants (XTRUCE_SCREEN_PROTOCOL.md sec. 3, 6)
X_STRATA = {0: "L6-h0", 1: "L6-h50", 2: "L1-h0", 3: "L1-h50"}
X_OWN = {"QoS": "V", "ES": "E", "IC": "intf", "LB": "maxload"}
POST_HOC_BOOT_SEED = 20260929        # post-hoc seed-cluster bootstrap of the summed redrawn advantage (not protocol)

# ---- palette (dataviz reference categorical slots, light mode) + ink
INK = "#1d1f23"; SOFT = "#5f6368"; GRID = "#e4e2dc"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                         "#e87ba4", "#008300", "#4a3aa7", "#e34948")
GREY = "#8d8a83"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": "#c9c5bb",
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": SOFT, "ytick.color": SOFT,
                     "axes.titlesize": 11.5, "figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.spines.top": False, "axes.spines.right": False})


def load(path):
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    jobs = {}
    for r in recs:
        if r.get("kind") == "job":
            jobs[tuple(r["key"])] = r          # key = (stage, pair, stratum, seed, arm); last write wins
    return recs, jobs


def pool(rs):
    g = lambda k: sum(r[k] for r in rs)                     # noqa: E731
    return {"V": 3600.0 * g("prot_viol") / max(g("prot_ue_s"), 1e-9), "E": g("energy_j"),
            "svr": 3600.0 * g("viol_ue_s") / max(g("ue_s"), 1e-9), "nonprot_embb_viol": g("nonprot_embb_viol"),
            "ll_viol": g("ll_viol"), "rlf": g("rlf"), "lowsinr": g("lowsinr_ue_s") / max(g("ue_s"), 1e-9),
            "changes": float(np.mean([r["st_changes"] for r in rs]))}


def arm_group(a):
    if a == "freeze": return 1
    if a == "noarb": return 3
    if a.startswith("prio:"): return 5
    if a.startswith("cell:"): return 6
    if a == "lock": return 7
    if a == "qacm": return 8
    if a == "oracle": return 10
    if a.startswith("sub:"): return 2 if "+" not in a else 4
    return 0


# ======================================================================================== E6-P
def e6p_stage0():
    recs, J = load(os.path.join(RUNS, "e6p-s0a", "all.jsonl"))
    s0a = {r["target"]: {"load_factor": r["load_factor"], "fit_L_mean": r["iters"][-1]["L_mean"],
                         "n_iter": len(r["iters"]), "check_L_mean": r["check_L_mean"],
                         "check_L_min": r["check_L_min"], "check_L_max": r["check_L_max"]}
           for r in recs if r.get("kind") == "0a_result"}
    _, J = load(os.path.join(RUNS, "e6p-s0b", "all.jsonl"))
    unit = {k[4]: r for k, r in J.items() if k[4] in ("M1", "M2", "M3", "M5b")}
    m1 = [r["M1"] for k, r in J.items() if k[4] == "M1"]
    m4 = {}
    for pair, P in PAIRS.items():
        for x in P["xapps"]:
            kpi, per = M4_OWN[x], {}
            for s in STRATA:
                seeds = [150020 + 10 * s + j for j in range(4)]
                al = [J[("0b", pair, s, sd, "sub:" + x)] for sd in seeds]
                fr = [J[("0b", "*", s, sd, "freeze")] for sd in seeds]
                val = lambda rs: {"energy": pool(rs)["E"], "psvr": pool(rs)["V"], "lowsinr": pool(rs)["lowsinr"]}[kpi]  # noqa: E731
                acts = float(np.mean([a["st_changes"] for a in al])) >= 1.0
                imp = val(al) < val(fr)
                n_imp = int(sum(val([a]) < val([f]) for a, f in zip(al, fr)))
                mat = kpi == "energy" or val(fr) >= M4_MAT[kpi]
                per[s] = {"acts": acts, "material": bool(mat), "own_alone": val(al), "own_freeze": val(fr),
                          "pooled_improves": bool(imp), "n_seeds_improve": n_imp,
                          "ok_amended": bool((not acts) or (not mat) or (imp and n_imp >= 3)),
                          "ok_literal": bool((not acts) or (imp and n_imp >= 3))}
            n_act = sum(v["acts"] and v["material"] for v in per.values())
            m4[f"{pair}:{x}"] = {"kpi": kpi, "strata": per, "inert": n_act < 2,
                                 "pass_amended": all(v["ok_amended"] for v in per.values()) and n_act >= 2,
                                 "pass_literal": all(v["ok_literal"] for v in per.values())}
    m6 = [r["M6"] for k, r in J.items() if k[1] == "P1" and k[2] == 0 and "M6" in r]
    recs, J = load(os.path.join(RUNS, "e6p-s0c", "all.jsonl"))
    oc = [r for r in J.values()][0]
    s0c = {k: oc.get(k) for k in ("secs", "ref_secs", "oracle_secs", "n_roll", "n_epochs", "n_deviations",
                                  "pair", "stratum", "seed")}
    return {"stage0a": s0a,
            "stage0b": {"M1_pass": all(x["pass"] for x in m1),
                        "M1_b_frac_all_macros": [x["b_frac"] for x in m1],
                        "M1_b_frac_loaded_macros": [x["b_frac_loaded_macros_diag"] for x in m1],
                        "M2_pass": unit["M2"]["M2"]["pass"], "M3_pass": unit["M3"]["M3"]["pass"],
                        "M5b_pass": unit["M5b"]["M5b"]["pass"],
                        "M6_rlf_within_2s": sum(x["rlf_within_2s"] for x in m6),
                        "M6_n_eligible": sum(x["n_eligible"] for x in m6), "M4": m4},
            "stage0c": s0c}


def e6p_cells():
    _, J1 = load(os.path.join(RUNS, "e6p-s1", "all.jsonl"))
    _, J2 = load(os.path.join(RUNS, "e6p-s2", "all.jsonl"))
    J = {**J1, **J2}
    arms_by_pair = defaultdict(set)
    for k in J1:
        if k[1] != "*":
            arms_by_pair[k[1]].add(k[4])
    cells = {}
    for pair, P in PAIRS.items():
        if pair not in arms_by_pair:
            continue                      # P2: not screenable, never run in stage 1
        for s in STRATA:
            seeds = [150100 + 10 * s + j for j in range(N_SCREEN)]
            per = {"freeze": [J[("1", "*", s, sd, "freeze")] for sd in seeds]}
            for a in sorted(arms_by_pair[pair]):
                per[a] = [J[("1", pair, s, sd, a)] for sd in seeds]
            orc = [J.get(("2", pair, s, sd, "oracle")) for sd in seeds]
            if all(orc):
                per["oracle"] = orc
            Pl = {a: pool(rs) for a, rs in per.items()}
            Ef, EA, AA = Pl["freeze"]["E"], Pl[P["A"]]["E"], Pl["noarb"]
            singles = ["sub:" + x for x in P["xapps"]]
            ref = min(singles, key=lambda a: Pl[a]["V"])
            V_AA, V_ref = AA["V"], Pl[ref]["V"]
            den = V_AA - V_ref
            table = {}
            for a, p in Pl.items():
                matched = Ef - p["E"] >= X_MATCH * (Ef - EA) - 1e-9
                ratios = {k: (p[k] / AA[k] if AA[k] else float("nan")) for k in GUARD_KEYS}
                guard = all(p[k] <= GUARD * AA[k] + 1e-9 for k in GUARD_KEYS)
                table[a] = {"group": arm_group(a), "V": p["V"], "R": (V_AA - p["V"]) / den,
                            "retention": (Ef - p["E"]) / (Ef - EA), "matched": bool(matched), "guard": bool(guard),
                            "eligible": bool(matched and guard), "churn": p["changes"], "guard_ratio": ratios}
            static = [a for a in table if 4 <= table[a]["group"] <= 9 and table[a]["eligible"]]
            R_static = max(table[a]["R"] for a in static) if static else None
            rng = np.random.default_rng([BOOT_TAG, P["idx"], s])
            idx = rng.integers(0, N_SCREEN, (N_BOOT, N_SCREEN))
            vb = lambda rs: 3600.0 * np.array([r["prot_viol"] for r in rs])[idx].sum(1) / \
                np.array([r["prot_ue_s"] for r in rs])[idx].sum(1)          # noqa: E731
            d_b = vb(per["noarb"]) - np.min([vb(per[a]) for a in singles], 0)
            lb = float(np.quantile(d_b, LB_Q))
            screenable = Ef - EA >= SCREEN_FLOOR * Ef
            c1 = bool(screenable and table["noarb"]["matched"] and den / V_ref >= LAMBDA_MIN and den >= MATERIAL
                      and lb > 0)
            cell = {"pair": pair, "stratum": s, "stratum_name": STRATA[s], "A_saving_frac": (Ef - EA) / Ef,
                    "V_AA": V_AA, "V_ref": V_ref, "ref_arm": ref, "Lambda": den / V_ref, "V_AA_minus_V_ref": den,
                    "lb90": lb, "AA_retention": table["noarb"]["retention"], "R_static": R_static,
                    "R_static_arm": max(static, key=lambda a: table[a]["R"]) if static else None, "c1": c1,
                    "arms": table}
            if "oracle" in per:
                o = table["oracle"]
                devs = [d for r in per["oracle"] for d in r["deviations"]]
                kept = [d["sign_true"] == d["sign_redraw"] for d in devs]
                cell.update(R_or=(o["R"] if o["eligible"] else 0.0), R_or_raw=o["R"], oracle_eligible=o["eligible"],
                            c2=bool(c1 and o["eligible"] and o["R"] >= R_OR_MIN), rho_sign=float(np.mean(kept)),
                            n_dev=len(devs), c4=bool(devs) and float(np.mean(kept)) >= RHO_MIN,
                            oracle_secs_mean=float(np.mean([r["secs"] for r in per["oracle"]])),
                            oracle_nroll_mean=float(np.mean([r["n_roll"] for r in per["oracle"]])),
                            posthoc=posthoc(per["oracle"], o["R"], float(np.mean(kept))))
            cells[f"{pair}|{s}"] = cell
    return cells


def posthoc(orecs, R_raw, rho):
    """POST HOC (not a protocol quantity). Advantage = AA - plan in protected violated UE-s over the H=90 s rollout."""
    t, rd, kept, seed_ix = [], [], [], []
    for i, r in enumerate(orecs):
        for d in r["deviations"]:
            t.append(d["aa_true"][0] - d["plan_true"][0]); rd.append(d["aa_redraw"][0] - d["plan_redraw"][0])
            kept.append(d["sign_true"] == d["sign_redraw"]); seed_ix.append(i)
    t, rd, kept, seed_ix = map(np.asarray, (t, rd, kept, seed_ix))
    share = rd.sum() / t.sum()
    rng = np.random.default_rng(POST_HOC_BOOT_SEED)
    per_seed = np.array([rd[seed_ix == i].sum() for i in range(len(orecs))])
    boot = per_seed[rng.integers(0, len(orecs), (N_BOOT, len(orecs)))].sum(1)
    w = np.abs(t)
    p = (1 + np.sqrt(max(2 * rho - 1, 0.0))) / 2                     # rho = p^2 + (1-p)^2, p = Phi(mu/sigma)
    return {"corr_true_redraw": float(np.corrcoef(t, rd)[0, 1]), "sum_true": float(t.sum()),
            "sum_redraw": float(rd.sum()), "kept_share": float(share),
            "sum_redraw_boot_p05": float(np.quantile(boot, 0.05)),
            "wtd_sign_retention_abs_true": float((w * kept).sum() / w.sum()),
            "frac_redraw_prot_adv_pos": float(np.mean(rd > 0)), "frac_true_prot_adv_zero": float(np.mean(t == 0)),
            "R_denoised": float(R_raw * share), "implied_mu_over_sigma": float(NormalDist().inv_cdf(p)),
            "true": t.tolist(), "redraw": rd.tolist(), "kept": kept.astype(int).tolist()}


# ======================================================================================== xTRUCE
def xts_stage0():
    _, J = load(os.path.join(RUNS, "xts-s0", "all.jsonl"))
    unit = [r for k, r in J.items() if k[4] == "unit"][0]
    m4 = {}
    for x, kpi in X_OWN.items():
        per = {}
        for s in X_STRATA:
            seeds = [190000 + 50 * s + j for j in range(44, 49)]
            al = [J[("0", "X4", s, sd, "sub:" + x)] for sd in seeds]
            fr = [J[("0", "*", s, sd, "freeze")] for sd in seeds]

            def val(rs):
                if kpi == "V":
                    return 3600.0 * sum(r["pv"] for r in rs) / sum(r["pu"] for r in rs)
                if kpi == "E":
                    return sum(r["E"] for r in rs)
                return sum(r[kpi] for r in rs) / sum(r["n_ep"] for r in rs)
            acts = float(np.mean([a["acc_by"].get(x, 0) for a in al])) >= 1.0
            mat = kpi in ("E", "intf") or (val(fr) >= 36.0 if kpi == "V" else val(fr) > 0.80)
            imp = val(al) < val(fr)
            n_imp = int(sum(val([a]) < val([f]) for a, f in zip(al, fr)))
            per[s] = {"acts": acts, "material": bool(mat), "own_alone": val(al), "own_freeze": val(fr),
                      "pooled_improves": bool(imp), "n_seeds_improve": n_imp,
                      "ok": bool((not acts) or (not mat) or (imp and n_imp >= 4))}
        n_act = sum(v["acts"] and v["material"] for v in per.values())
        m4[x] = {"kpi": kpi, "strata": per, "inert": n_act < 2,
                 "pass": all(v["ok"] for v in per.values()) and n_act >= 2}
    m6 = {s: [J[("0", "X4", s, 190000 + 50 * s + j, "sub:ES")]["sleep_cs"] for j in range(44, 49)] for s in X_STRATA}
    m7 = {}
    for p in ("X1", "X2", "X3", "X4"):
        for s in X_STRATA:
            rs = [J[("0", p, s, 190000 + 50 * s + j, "aa")] for j in range(44, 49)]
            m7[f"{p}|{s}"] = float(np.mean([r["n_conf"] / r["n_ep"] for r in rs]))
    _, J0c = load(os.path.join(RUNS, "xts-s0c", "all.jsonl"))
    oc = list(J0c.values())[0]
    bad = sorted({p for p, xs in {"X1": ("QoS", "ES"), "X2": ("QoS", "ES", "IC"), "X3": ("QoS", "ES", "LB"),
                                  "X4": ("QoS", "ES", "IC", "LB")}.items() if any(not m4[x]["pass"] for x in xs)})
    return {"unit": {m: unit[m]["pass"] for m in ("M1", "M2", "M3", "M5", "M8")}, "M4": m4,
            "M6_sleep_cs": m6, "M7_conflict_frac_mean": m7, "not_screenable_pairs": bad,
            "stage0c": {k: oc.get(k) for k in ("secs", "ref_secs", "oracle_secs", "n_roll", "n_decisions",
                                               "n_deviations")}}


# ======================================================================================== figures
C1_CELLS = ["P1|3", "P3|1", "P3|3"]
CELL_LBL = {"P1|3": "P1 ES×SG · surge-L40", "P3|1": "P3 ES×PowerES×SG · base-L40", "P3|3": "P3 ES×PowerES×SG · surge-L40"}


def short(a):
    if a in ("lock", "qacm"):
        return {"lock": "knob-lock", "qacm": "QACM"}[a]
    return (a.replace("SliceGuarantee", "SG").replace("PowerES", "PwrES").replace("noarb", "accept-all (AA)")
            .replace("prio:", "priority ").replace("cell:", "cell-lock ").replace("sub:", "only "))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=140, bbox_inches="tight"); plt.close(fig)


def fig1(cells, x0):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    n_all = 12
    n_scr = sum(1 for k in cells)                                     # P1, P3 x 4 strata
    n_c1 = sum(c["c1"] for c in cells.values())
    n_c2 = sum(c.get("c2", False) for c in cells.values())
    n_c4 = sum(c.get("c4", False) for c in cells.values())
    st = ["pair×stratum\ncells", "stage 0\nscreenable", "stage 1\nc1 loss", "stage 2\nc2 recover",
          "stage 2\nc4 ρ≥0.70", "stage 3\nc3 headroom"]
    vals = [n_all, n_scr, n_c1, n_c2, n_c4, 0]
    cols = [GREY, BLUE, BLUE, RED, RED, "#d9d6cf"]
    ax.bar(range(len(st)), vals, color=cols, width=0.62)
    notes = ["3 pairs × 4", "P2 out (M4:\nCoverage\ninert)", "P1 s3,\nP3 s1,\nP3 s3", "oracle\nbreaks a\nguardrail",
             "ρ = 0.55,\n0.61, 0.66", "not run"]
    for i, (v, n) in enumerate(zip(vals, notes)):
        ax.text(i, v + 0.25, str(v), ha="center", fontsize=11, fontweight="bold", color=INK)
        ax.text(i, -3.1, n, ha="center", va="top", fontsize=7.8, color=SOFT)
    ax.set_xticks(range(len(st))); ax.set_xticklabels(st, fontsize=8.6)
    ax.set_ylim(0, 13.5); ax.set_ylabel("cells still in play"); ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_title("E6-P (Gate A v2, E6P-v1)  →  P1 DEAD, P3 DEAD, P2 not screenable", loc="left", fontsize=10.5)
    ax.tick_params(axis="x", pad=2)
    ax = axes[1]
    n_x = 16
    n_xs = 16 - 4 * len(x0["not_screenable_pairs"])
    st2 = ["pair×stratum\ncells", "stage 0\nscreenable", "stage 1-3"]
    ax.bar(range(3), [n_x, n_xs, 0], color=[GREY, RED, "#d9d6cf"], width=0.55)
    for i, (v, n) in enumerate(zip([n_x, n_xs, 0], ["4 pairs × 4", "M4 fails: QoS\n(in every pair), LB",
                                                   "not run"])):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=11, fontweight="bold")
        ax.text(i, -4.1, n, ha="center", va="top", fontsize=8, color=SOFT)
    ax.set_xticks(range(3)); ax.set_xticklabels(st2, fontsize=8.6); ax.set_ylim(0, 18)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_title("xTRUCE plant (XTS-v1)  →  all 4 pairs NOT SCREENABLE", loc="left", fontsize=10.5)
    fig.suptitle("Fig 1 — Gate A v2 funnel: no cell of either plant reaches a PASS", x=0.01, ha="left",
                 fontweight="bold", y=1.02)
    fig.tight_layout(); save(fig, "fig1_gate_funnel.png")


def arm_color(a, t):
    g = t["group"]
    if a == "noarb": return INK
    if g in (5, 7, 8): return BLUE
    if g == 6: return VIOLET
    if g in (2, 4): return AQUA
    if g == 1: return GREY
    if g == 10: return ORANGE
    return GREY


def fig2(cells):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 6.6), gridspec_kw={"width_ratios": [0.62, 1, 1]})
    CAP = 200
    for ax, k in zip(axes, C1_CELLS):
        c = cells[k]; T = c["arms"]
        order = sorted([a for a in T if a != "oracle"], key=lambda a: (T[a]["group"], a)) + ["oracle"]
        y = np.arange(len(order))[::-1]
        for yi, a in zip(y, order):
            t = T[a]; v = min(t["V"], CAP)
            ax.barh(yi, v, color=arm_color(a, t), height=0.66, alpha=1.0 if t["eligible"] else 0.38,
                    hatch=None if t["eligible"] else "///", edgecolor="white", lw=0.6)
            lab = f"{t['V']:.1f}" + ("" if t["V"] <= CAP else " ▸")
            ax.text(v + 3, yi, lab, va="center", fontsize=7.4, color=SOFT)
        ax.axvline(c["V_AA"], color=INK, lw=1.1, ls="-"); ax.axvline(c["V_ref"], color=AQUA, lw=1.3, ls="--")
        ax.text(c["V_AA"], -1.0, " V_AA", fontsize=7.8, color=INK, va="center")
        ax.text(c["V_ref"], -1.0, "V_ref ", fontsize=7.8, color=AQUA, va="center", ha="right")
        ax.set_ylim(-1.5, len(order) - 0.4)
        ax.set_yticks(y); ax.set_yticklabels([short(a) for a in order], fontsize=7.8)
        ax.set_xlim(0, CAP + 38); ax.set_xlabel("PSVR  V  (violated UE-s per protected UE-h; lower better)", fontsize=8.4)
        ax.set_title(f"{CELL_LBL[k]}\nΛ={c['Lambda']:.2f}  R_static={c['R_static']:.3f}", loc="left", fontsize=9.4)
        ax.grid(axis="x", color=GRID, lw=0.7)
    from matplotlib.patches import Patch
    hs = [Patch(color=INK, label="accept-all"), Patch(color=BLUE, label="per-knob priority / knob-lock / QACM"),
          Patch(color=VIOLET, label="cell-priority lock"), Patch(color=AQUA, label="single xApp / subset"),
          Patch(color=GREY, label="freeze"), Patch(color=ORANGE, label="oracle (stage 2)"),
          Patch(facecolor="white", edgecolor=SOFT, hatch="///", label="ineligible (energy or guardrail)")]
    fig.legend(handles=hs, loc="lower center", ncol=4, fontsize=8.2, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Fig 2 — Stage 1: every priority order, the knob-lock and QACM land on accept-all "
                 "(one writer per knob); only non-energy-matched subsets approach V_ref",
                 x=0.01, ha="left", fontweight="bold", fontsize=11, y=1.01)
    fig.text(0.01, -0.12, f"Bars clipped at V={CAP} (▸ = off-scale, true value printed). Oracle bar added from "
             "stage 2 for reference. Pooled over 8 screen seeds per cell.", fontsize=8.6, color=SOFT)
    fig.tight_layout(); save(fig, "fig2_stage1_arms.png")


def fig3(cells):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), sharey=True)
    LO = -1.0
    for ax, k in zip(axes, C1_CELLS):
        c = cells[k]; T = c["arms"]
        ax.axvspan(0.90, 1.6, color=AQUA, alpha=0.07); ax.axhline(0, color="#bdb8ad", lw=0.9)
        ax.axhline(R_OR_MIN, color=SOFT, ls=":", lw=1); ax.axvline(0.90, color=AQUA, ls="--", lw=1.1)
        ax.text(0.91, -0.86, "energy-matched →", fontsize=7.6, color=AQUA)
        ax.text(0.02, R_OR_MIN + 0.02, "c2 bar R=0.50", fontsize=7.4, color=SOFT)
        for a, t in T.items():
            if a == "freeze":
                continue
            r = max(t["R"], LO); col = arm_color(a, t)
            mk = "*" if a == "oracle" else ("v" if t["R"] < LO else "o")
            ax.scatter(t["retention"], r, s=150 if a == "oracle" else 34, marker=mk,
                       facecolor=col if t["eligible"] else "white", edgecolor=col, lw=1.4, zorder=3)
        o = T["oracle"]
        bad = [g for g in GUARD_KEYS if o["guard_ratio"][g] > GUARD + 1e-9]
        ax.annotate(f"oracle R={o['R']:.2f}, ret {o['retention']:.2f}\nINELIGIBLE: {', '.join(bad)} "
                    f"{max(o['guard_ratio'][g] for g in bad):.3f}× AA", (o["retention"], o["R"]),
                    xytext=(0.22, 0.90), textcoords="axes fraction", fontsize=7.6, color=ORANGE, ha="left",
                    arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))
        ax.set_xlim(-0.05, 1.55); ax.set_ylim(LO - 0.08, 1.25)
        ax.set_xlabel("energy retention  (1 = A-alone saving)"); ax.set_title(CELL_LBL[k], loc="left", fontsize=9.6)
        ax.grid(color=GRID, lw=0.6)
    axes[0].set_ylabel("recovery  R  (1 = back to V_ref)")
    fig.suptitle("Fig 3 — Recovery vs energy retention: nothing eligible recovers; the oracle recovers only by "
                 "over-sleeping and breaking a guardrail", x=0.01, ha="left", fontweight="bold", fontsize=11, y=1.03)
    fig.text(0.01, -0.06, "Filled = eligible (energy-matched and all 4 guardrails ≤ 1.10× AA); hollow = ineligible. "
             "▼ = R below −1 (clipped). Colours as Fig 2. Freeze omitted (retention 0).", fontsize=8.6, color=SOFT)
    fig.tight_layout(); save(fig, "fig3_recovery_vs_energy.png")


def fig4(cells):
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    names = {"svr": "all-UE SVR", "nonprot_embb_viol": "non-prot. eMBB viol.", "ll_viol": "LL viol.", "rlf": "RLF"}
    cols = [BLUE, AQUA, VIOLET, ORANGE]
    w = 0.19
    for i, k in enumerate(C1_CELLS):
        gr = cells[k]["arms"]["oracle"]["guard_ratio"]
        for j, g in enumerate(GUARD_KEYS):
            x = i + (j - 1.5) * w
            ax.bar(x, gr[g], width=w * 0.92, color=cols[j], label=names[g] if i == 0 else None)
            ax.text(x, gr[g] + 0.012, f"{gr[g]:.3f}", ha="center", fontsize=7.4, rotation=90, va="bottom",
                    color=INK if gr[g] > GUARD else SOFT, fontweight="bold" if gr[g] > GUARD else "normal")
    ax.axhline(GUARD, color=RED, ls="--", lw=1.4); ax.text(1.5, GUARD + 0.01, "limit 1.10× AA", color=RED,
                                                         fontsize=8.4, ha="center", va="bottom")
    ax.axhline(1.0, color="#bdb8ad", lw=0.9)
    ax.set_xticks(range(3)); ax.set_xticklabels([CELL_LBL[k] for k in C1_CELLS], fontsize=8.6)
    ax.set_ylim(0.6, 1.55); ax.set_ylabel("oracle ÷ accept-all  (pooled, 8 seeds)")
    ax.legend(fontsize=8.2, ncol=4, frameon=False, loc="upper left"); ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_title("Fig 4 — Oracle guardrails vs accept-all: one breach per cell makes it ineligible (R_or := 0)",
                 loc="left", fontweight="bold", fontsize=11)
    fig.tight_layout(); save(fig, "fig4_oracle_guardrails.png")


def fig5(cells):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3))
    for ax, k in zip(axes, C1_CELLS):
        ph = cells[k]["posthoc"]
        t, r, kp = np.array(ph["true"]), np.array(ph["redraw"]), np.array(ph["kept"]).astype(bool)
        lim = max(np.abs(t).max(), np.abs(r).max()) * 1.08
        ax.fill_between([0, lim], -lim, 0, color=RED, alpha=0.06, lw=0)
        ax.axhline(0, color="#bdb8ad", lw=0.9); ax.axvline(0, color="#bdb8ad", lw=0.9)
        ax.plot([-lim, lim], [-lim, lim], color=SOFT, lw=0.9, ls=":")
        ax.scatter(t[kp], r[kp], s=14, color=BLUE, alpha=0.75, lw=0, label="sign kept")
        ax.scatter(t[~kp], r[~kp], s=16, facecolor="white", edgecolor=RED, lw=1.0, label="sign flipped")
        ax.set_xlim(-lim * 0.25, lim); ax.set_ylim(-lim, lim)
        c = cells[k]
        ax.text(0.03, 0.97, f"ρ_sign = {c['rho_sign']:.3f}  (n = {c['n_dev']})\ncorr = {ph['corr_true_redraw']:.2f}"
                f"\nredrawn / true sum = {ph['kept_share']:.2f}", transform=ax.transAxes, va="top", fontsize=8.4,
                bbox=dict(facecolor="white", edgecolor=GRID, boxstyle="round,pad=0.3"))
        ax.set_title(CELL_LBL[k], loc="left", fontsize=9.6)
        ax.set_xlabel("advantage on the TRUE tape\n(AA − plan, protected violated UE-s / 90 s)", fontsize=8.6)
        ax.grid(color=GRID, lw=0.5)
    axes[0].set_ylabel("advantage on a REDRAWN future")
    axes[0].legend(fontsize=8, loc="lower right", frameon=False)
    fig.suptitle("Fig 5 — Per-deviation oracle advantage: selected on the true tape, much of it does not survive "
                 "a redrawn future", x=0.01, ha="left", fontweight="bold", fontsize=11, y=1.03)
    fig.text(0.01, -0.08, "Each point = one oracle decision that deviated from accept-all (8 seeds pooled). Sign is "
             "lexicographic (protected, then all-UE violations), so a point on x=0 can still count as a positive "
             "true sign. Shaded = redraw says the plan is worse. corr / sums are POST HOC.", fontsize=8.4, color=SOFT)
    fig.tight_layout(); save(fig, "fig5_true_vs_redraw.png")


def fig6(x0):
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.9))
    units = {"V": "protected V (per prot. UE-h)", "E": "scored energy, 5 seeds (MJ)", "intf": "mean caused interference (W)",
             "maxload": "mean max demand load"}
    for ax, x in zip(axes, ("QoS", "ES", "IC", "LB")):
        m = x0["M4"][x]; kpi = m["kpi"]
        sc = 1e-6 if kpi == "E" else 1.0
        for s in X_STRATA:
            d = m["strata"][s]
            f, a = d["own_freeze"] * sc, d["own_alone"] * sc
            ax.bar(s - 0.19, f, width=0.36, color=GREY, label="freeze" if s == 0 else None)
            ax.bar(s + 0.19, a, width=0.36, color=BLUE if d["ok"] else RED,
                   alpha=1.0 if d["material"] else 0.4, label=None)
            tag = f"{d['n_seeds_improve']}/5" + ("" if d["material"] else "\nn/m")
            top = max(f, a)
            ax.text(s, top * (1.35 if kpi == "intf" else 1.04), tag, ha="center", fontsize=8,
                    color=RED if not d["ok"] else SOFT, fontweight="bold" if not d["ok"] else "normal")
        if kpi == "intf":
            ax.set_yscale("log")
        ax.set_xticks(list(X_STRATA)); ax.set_xticklabels(list(X_STRATA.values()), fontsize=8)
        ax.set_title(f"{x}  ({'PASS' if m['pass'] else 'FAIL'})", loc="left", fontsize=10.2,
                     color=INK if m["pass"] else RED, fontweight="bold")
        ax.set_ylabel(units[kpi], fontsize=8.4); ax.grid(axis="y", color=GRID, lw=0.6)
        ax.margins(y=0.18)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=GREY, label="freeze"), Patch(color=BLUE, label="xApp alone: stratum OK"),
                        Patch(color=RED, label="xApp alone: stratum FAILS M4"),
                        Patch(color=BLUE, alpha=0.4, label="not material (n/m): not evaluated")],
               loc="lower center", ncol=4, fontsize=8.4, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Fig 6 — xTRUCE stage 0, M4: each xApp alone vs freeze on its own KPI (lower is better; "
                 "k/5 = mechanism seeds that improve; rule needs ≥4/5)", x=0.01, ha="left", fontweight="bold",
                 fontsize=11, y=1.03)
    fig.tight_layout(); save(fig, "fig6_xtruce_m4.png")


def _r(x, n=4):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if x != x else float(f"{float(x):.{n + 2}g}")      # 6 significant digits
    if isinstance(x, dict):
        return {str(k): _r(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, n) for v in x]
    return x


def main():
    e0 = e6p_stage0()
    cells = e6p_cells()
    x0 = xts_stage0()
    assert sorted(k for k, c in cells.items() if c["c1"]) == sorted(C1_CELLS)
    fig1(cells, x0); fig2(cells); fig3(cells); fig4(cells); fig5(cells); fig6(x0)
    verdict = {}
    for p in ("P1", "P3"):
        cs = [c for c in cells.values() if c["pair"] == p]
        c1 = [c for c in cs if c["c1"]]
        verdict[p] = ("DEAD (no loss)" if not c1 else "DEAD (not recoverable)" if not any(c["c2"] for c in c1)
                      else "see stage 3")
    verdict["P2"] = "NOT SCREENABLE (inert xApp, M4)"
    data = {"e6p": {"verdict": verdict, **e0,
                    "cells": {k: {kk: vv for kk, vv in c.items() if kk != "posthoc"} for k, c in cells.items()},
                    "posthoc_POST_HOC": {k: {kk: vv for kk, vv in c["posthoc"].items()
                                             if kk not in ("true", "redraw", "kept")} for k, c in cells.items()
                                         if "posthoc" in c},
                    "fig5_points": {k: {kk: cells[k]["posthoc"][kk] for kk in ("true", "redraw", "kept")}
                                    for k in C1_CELLS}},
            "xtruce": {"verdict": {p: "NOT SCREENABLE" for p in x0["not_screenable_pairs"]}, **x0}}
    json.dump(_r(data), open(os.path.join(OUT, "figures_data.json"), "w"), indent=1)
    print("verdicts:", verdict, "| xTRUCE not screenable:", x0["not_screenable_pairs"])
    for k in C1_CELLS:
        c = cells[k]; ph = c["posthoc"]; o = c["arms"]["oracle"]
        print(k, f"V_AA {c['V_AA']:.1f} V_ref {c['V_ref']:.1f} Lam {c['Lambda']:.3f} LB90 {c['lb90']:.1f} "
                 f"Rst {c['R_static']:.4f} Rraw {c['R_or_raw']:.3f} ret {o['retention']:.3f} "
                 f"guard {_r(o['guard_ratio'], 3)} rho {c['rho_sign']:.3f} n {c['n_dev']} | corr "
                 f"{ph['corr_true_redraw']:.3f} share {ph['kept_share']:.3f} sumrd {ph['sum_redraw']:.0f} "
                 f"p05 {ph['sum_redraw_boot_p05']:.0f} wtd {ph['wtd_sign_retention_abs_true']:.3f} "
                 f"Rdn {ph['R_denoised']:.3f} mu/sd {ph['implied_mu_over_sigma']:.3f}")
    print("wrote fig1..fig6 + figures_data.json to", OUT)


if __name__ == "__main__":
    main()
