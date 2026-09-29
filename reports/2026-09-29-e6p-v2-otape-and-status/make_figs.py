"""Figures + numbers for the 2026-09-29 sequel report: E6-P v2 O_tape falsifier (P3 NOISE STOP) and project status.

Reads raw run files LIVE (json + numpy + matplotlib only; no import of the screen drivers) and recomputes,
independently of `e6p_v2.py summary`:
  * v2 (fresh seeds 155000 + 10*stratum + j) and, with the same code, v1 (150100 + 10*stratum + j) for the three
    v1 criterion-1 cells: pooled PSVR V per arm, energy retention, eligibility, recovery R, Lambda (+ 5-95 %),
    V_AA - V_ref, paired-seed bootstrap LB90 (RNG [6611, pair_idx, stratum], 10 000 resamples; driver's stream),
    R_or with 5-95 % interval, R_static (deployable arms 4-8, and with arm 9 hindsight static), headroom (+ 5-95 %),
    rho_sign, criteria c1-c4 and the verdict per pair;
  * v2 descriptive quantities (protocol sec. 4): summed re-drawn / true-tape advantage, per-seed sums, seed-bootstrap
    LB90 (RNG [6612, pair_idx, stratum]), oracle guardrail ratios vs AA, trial-rejection counts;
  * POST HOC (never a criterion): corr(true, redrawn), kept share, de-noised R = raw R x kept share, and 5-95 %
    intervals of R for the non-oracle ladder arms from the same bootstrap draws as the driver's R_or interval;
  * operational facts (Colab probe / self-test / timing) and the status of every earlier screen, parsed from the
    result files named in SOURCES.
Emits fig1..fig6 (PNG) + figures_data.json next to this file.

Run (repo root):  PYTHONPATH=. .venv/Scripts/python.exe reports/2026-09-29-e6p-v2-otape-and-status/make_figs.py
Optional cross-check against the driver:  ... make_figs.py --check DRIVER_SUMMARY.json
  where DRIVER_SUMMARY.json = output of `e6p_v2.py summary --in <4 v2 files> --json DRIVER_SUMMARY.json`
  (never with --write-state).
Frozen constants copied from docs/benchmark/E6P_SCREEN_PROTOCOL.md (sec. 7) and E6P_V2_OTAPE_PROTOCOL.md (sec. 1-4).
"""
import ast
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SP = os.path.join(ROOT, "scratchpad")
RUNS = os.path.join(SP, "e6_dev", "runs")
OUT = HERE
PREV = os.path.join(ROOT, "reports", "2026-09-29-e6p-screen-xtruce-negative", "figures_data.json")

# ---- frozen constants (E6P_SCREEN_PROTOCOL.md sec. 7; E6P_V2_OTAPE_PROTOCOL.md sec. 1-4)
X_MATCH, SCREEN_FLOOR, GUARD = 0.90, 0.01, 1.10
LAMBDA_MIN, MATERIAL, R_OR_MIN, HEADROOM, RHO_MIN = 0.15, 36.0, 0.50, 0.10, 0.70
N_BOOT, BOOT_TAG, ADV_TAG, LB_Q = 10000, 6611, 6612, 0.10
N_SCREEN = 8
GUARD_KEYS = ("svr", "nonprot_embb_viol", "ll_viol", "rlf")
PAIRS = {"P1": dict(idx=1, xapps=("ES", "SliceGuarantee"), A="sub:ES"),
         "P3": dict(idx=3, xapps=("ES", "PowerES", "SliceGuarantee"), A="sub:ES+PowerES")}
STRATA = {1: "base-L40", 3: "surge-L40"}
CELLS = ["P1|3", "P3|1", "P3|3"]
CELL_LBL = {"P1|3": "P1 ES×SG · surge-L40", "P3|1": "P3 ES×PwrES×SG · base-L40",
            "P3|3": "P3 ES×PwrES×SG · surge-L40"}
V1 = dict(runs={"1": ["e6p-s1"], "2": ["e6p-s2"]}, base=150100)
V2 = dict(runs={"1": ["e6p-v2-s1"], "2": ["e6p-v2-s2k", "e6p-v2-s2r"], "3": ["e6p-v2-s3"]}, base=155000)

# ---- palette (same slots as the sibling report) + ink
INK = "#1d1f23"; SOFT = "#5f6368"; GRID = "#e4e2dc"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                         "#e87ba4", "#008300", "#4a3aa7", "#e34948")
GREY = "#8d8a83"; LIGHT = "#d9d6cf"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": "#c9c5bb",
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": SOFT, "ytick.color": SOFT,
                     "axes.titlesize": 11.5, "figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.spines.top": False, "axes.spines.right": False})


# ======================================================================================== loading
def read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load(version):
    J, heads, per_dir = {}, [], {}
    for st, dirs in version["runs"].items():
        for d in dirs:
            recs = [r for r in read(os.path.join(RUNS, d, "all.jsonl")) if not r.get("smoke")]
            jobs = [r for r in recs if r.get("kind") == "job"]
            for r in jobs:
                J[tuple(r["key"])] = r                                   # last write wins
            hs = [r for r in recs if r.get("kind") == "header"]
            heads += hs
            per_dir[d] = {"n_jobs": len(jobs), "n_parts": len(hs), "cpu_h": sum(r["secs"] for r in jobs) / 3600.0,
                          "first_header_utc": min((h["utc"] for h in hs), default=None),
                          "git_head": sorted({(h.get("code") or {}).get("git_head") for h in hs} - {None}),
                          "e6_dirty": sorted({(h.get("code") or {}).get("e6_dirty") for h in hs} - {None}),
                          "platform": sorted({(h.get("numeric_env") or {}).get("platform") for h in hs} - {None}),
                          "numpy": sorted({(h.get("numeric_env") or {}).get("numpy") for h in hs} - {None})}
    return J, per_dir


def pool(rs):
    g = lambda k: sum(r[k] for r in rs)                                   # noqa: E731
    return {"V": 3600.0 * g("prot_viol") / max(g("prot_ue_s"), 1e-9), "E": g("energy_j"),
            "svr": 3600.0 * g("viol_ue_s") / max(g("ue_s"), 1e-9), "nonprot_embb_viol": g("nonprot_embb_viol"),
            "ll_viol": g("ll_viol"), "rlf": g("rlf"), "changes": float(np.mean([r["st_changes"] for r in rs]))}


def vboot(rs, idx):
    pv = np.array([r["prot_viol"] for r in rs]); pu = np.array([r["prot_ue_s"] for r in rs])
    return 3600.0 * pv[idx].sum(1) / np.maximum(pu[idx].sum(1), 1e-9)


def arm_group(a):
    if a == "freeze": return 1
    if a == "noarb": return 3
    if a.startswith("prio:"): return 5
    if a.startswith("cell:"): return 6
    if a == "lock": return 7
    if a == "qacm": return 8
    if a == "hind": return 9
    if a == "oracle": return 10
    if a.startswith("sub:"): return 2 if "+" not in a else 4
    return 0


def q(x, p):
    return float(np.nanquantile(x, p))


# ======================================================================================== one cell
def analyse(J, base, pair, s):
    P = PAIRS[pair]
    seeds = [base + 10 * s + j for j in range(N_SCREEN)]
    per = {"freeze": [J[("1", "*", s, sd, "freeze")] for sd in seeds]}
    arms1 = sorted({k[4] for k in J if k[0] == "1" and k[1] == pair and k[2] == s})
    for a in arms1:
        per[a] = [J[("1", pair, s, sd, a)] for sd in seeds]
    for st, a in (("2", "oracle"), ("3", "hind")):
        rs = [J.get((st, pair, s, sd, a)) for sd in seeds]
        if all(rs):
            per[a] = rs
    Pl = {a: pool(rs) for a, rs in per.items()}
    Ef, EA, AA = Pl["freeze"]["E"], Pl[P["A"]]["E"], Pl["noarb"]
    singles = ["sub:" + x for x in P["xapps"]]
    ref = min(singles, key=lambda a: Pl[a]["V"])
    V_AA, V_ref = AA["V"], Pl[ref]["V"]
    den = V_AA - V_ref
    T = {}
    for a, p in Pl.items():
        matched = Ef - p["E"] >= X_MATCH * (Ef - EA) - 1e-9
        guard = all(p[k] <= GUARD * AA[k] + 1e-9 for k in GUARD_KEYS)
        T[a] = {"group": arm_group(a), "V": p["V"], "R": (V_AA - p["V"]) / den, "retention": (Ef - p["E"]) / (Ef - EA),
                "matched": bool(matched), "guard": bool(guard), "eligible": bool(matched and guard),
                "churn": p["changes"], "guard_ratio": {k: (p[k] / AA[k] if AA[k] else None) for k in GUARD_KEYS}}
    dep = [a for a in T if 4 <= T[a]["group"] <= 8 and T[a]["eligible"]]
    stat = [a for a in T if 4 <= T[a]["group"] <= 9 and T[a]["eligible"]]
    best = lambda L: max(L, key=lambda a: T[a]["R"]) if L else None      # noqa: E731
    # paired-seed bootstrap: same stream and draw order as e6p_screen.analyse_pair_stratum
    rng = np.random.default_rng([BOOT_TAG, P["idx"], s])
    idx = rng.integers(0, N_SCREEN, (N_BOOT, N_SCREEN))
    Vb = {a: vboot(rs, idx) for a, rs in per.items()}
    Vref_b = np.min([Vb[a] for a in singles], 0)
    d_b = Vb["noarb"] - Vref_b
    with np.errstate(divide="ignore", invalid="ignore"):
        lam_b = d_b / Vref_b
        Rb = {a: (Vb["noarb"] - Vb[a]) / d_b for a in Vb}
    lb = q(d_b, LB_Q)
    screenable = Ef - EA >= SCREEN_FLOOR * Ef
    c1_parts = {"screenable": bool(screenable), "AA_matched": T["noarb"]["matched"],
                "Lambda>=0.15": bool(den / V_ref >= LAMBDA_MIN), "diff>=36": bool(den >= MATERIAL), "LB90>0": bool(lb > 0)}
    c = {"pair": pair, "stratum": s, "stratum_name": STRATA[s], "seeds": seeds, "A_saving_frac": (Ef - EA) / Ef,
         "V_freeze": Pl["freeze"]["V"], "V_AA": V_AA, "V_ref": V_ref, "ref_arm": ref, "Lambda": den / V_ref,
         "Lambda_ci90": [q(lam_b, .05), q(lam_b, .95)], "V_AA_minus_V_ref": den, "lb90": lb,
         "c1_parts": c1_parts, "c1": all(c1_parts.values()),
         "R_static_deployable": T[best(dep)]["R"] if dep else None, "R_static_deployable_arm": best(dep),
         "R_static": T[best(stat)]["R"] if stat else None, "R_static_arm": best(stat),
         "R_ci90_report_side": {a: [q(Rb[a], .05), q(Rb[a], .95)] for a in Rb if a != "freeze"},
         "arms": T}
    if "oracle" in per:
        o = T["oracle"]
        R_or = o["R"] if o["eligible"] else 0.0
        Ror_b = Rb["oracle"] if o["eligible"] else np.zeros(N_BOOT)
        devs = [d for r in per["oracle"] for d in r["deviations"]]
        kept = [d["sign_true"] == d["sign_redraw"] for d in devs]
        rho = float(np.mean(kept))
        c.update(R_or=R_or, R_or_raw=o["R"], oracle_eligible=o["eligible"], R_or_ci90=[q(Ror_b, .05), q(Ror_b, .95)],
                 c2=bool(c["c1"] and R_or >= R_OR_MIN), rho_sign=rho, n_dev=len(devs), c4=rho >= RHO_MIN,
                 oracle_retention=o["retention"], oracle_guard_ratio=o["guard_ratio"],
                 oracle_secs_mean=float(np.mean([r["secs"] for r in per["oracle"]])),
                 oracle_nroll_total=int(sum(r["n_roll"] for r in per["oracle"])))
        if "hind" in per:
            Rs_b = np.max([Rb[a] for a in stat], 0)
            c.update(headroom=R_or - c["R_static"], headroom_ci90=[q(Ror_b - Rs_b, .05), q(Ror_b - Rs_b, .95)],
                     c3=bool(c["c2"] and R_or - c["R_static"] >= HEADROOM),
                     hind_regions_changed_per_seed=[len(r.get("best_keep") or {}) for r in per["hind"]])
        # (d) descriptive + POST HOC
        t = np.array([d["aa_true"][0] - d["plan_true"][0] for d in devs], float)
        rd = np.array([d["aa_redraw"][0] - d["plan_redraw"][0] for d in devs], float)
        per_r = np.array([sum(d["aa_redraw"][0] - d["plan_redraw"][0] for d in r["deviations"]) for r in per["oracle"]], float)
        per_t = np.array([sum(d["aa_true"][0] - d["plan_true"][0] for d in r["deviations"]) for r in per["oracle"]], float)
        rng2 = np.random.default_rng([ADV_TAG, P["idx"], s])
        b = per_r[rng2.integers(0, N_SCREEN, (N_BOOT, N_SCREEN))].sum(1)
        rej, rej_k = {}, {}
        for r in per["oracle"]:
            g = r.get("v2_guard") or {}
            for k, v in (g.get("rejections") or {}).items():
                rej[k] = rej.get(k, 0) + v
            for k, v in (g.get("rejections_by_key") or {}).items():
                rej_k[k] = rej_k.get(k, 0) + v
        c["adv"] = {"sum_true": float(t.sum()), "sum_redraw": float(rd.sum()), "sum_redraw_lb90_tag6612": q(b, LB_Q),
                    "per_seed_redraw": per_r.tolist(), "per_seed_true": per_t.tolist(),
                    "n_seeds_redraw_pos": int((per_r > 0).sum()), "kept_share_POST_HOC": float(rd.sum() / t.sum()),
                    "corr_true_redraw_POST_HOC": float(np.corrcoef(t, rd)[0, 1]),
                    "R_denoised_POST_HOC": float(o["R"] * rd.sum() / t.sum()),
                    "true": t.tolist(), "redraw": rd.tolist(), "kept": [int(k) for k in kept]}
        if rej:
            c["trial_rejections"] = rej
            c["trial_rejections_by_key"] = rej_k
    return c


def verdict(cs):
    S1 = [c for c in cs if c["c1"]]
    if not S1:
        return "DEAD (no loss)"
    S12 = [c for c in S1 if c.get("c2")]
    if not S12:
        return "DEAD (not recoverable)"
    if any("c3" not in c for c in S12):
        return "PENDING stage 3"
    S123 = [c for c in S12 if c["c3"]]
    if any(c["c4"] for c in S123):
        return "PASS"
    return "NOISE STOP" if S123 else "NO-EDGE STOP"


# ======================================================================================== ops + status sources
def ops_facts():
    sd = os.path.join(RUNS, "e6colab-selftest")
    e2e = json.load(open(os.path.join(sd, "e2e_report.json")))
    ticks = [l.split(" ", 2)[:2] for l in open(os.path.join(sd, "tick.log"), encoding="utf-8") if l.strip()]
    probe = json.load(open(os.path.join(RUNS, "e6v2-colab-probe", "colab.json")))
    tim = read(os.path.join(RUNS, "e6v2-colab-smoke", "timing.json"))[0]
    kag = next(r for r in read(os.path.join(RUNS, "e6p-s1", "all.jsonl"))
               if r.get("kind") == "job" and r["key"] == tim["key"])
    s2c = json.load(open(os.path.join(RUNS, "e6p-v2-s2c", "colab.json")))
    lost_has_results = os.path.exists(os.path.join(RUNS, "e6p-v2-s2c", "all.jsonl"))
    return {"selftest": {"passed": e2e["passed"], "stopped_after_min": e2e["stopped_after_min"],
                         "resume": e2e["resume"], "verify": e2e["verify"], "tick_events": ticks},
            "colab_probe": {"cgroup_cpu_max": probe["probe"]["cgroup_cpu_max"], "nproc": probe["probe"]["nproc"],
                            "throughput_x_single": {b["n"]: b["throughput_x_single"] for b in probe["probe"]["batches"]},
                            "fp_traj_match_kaggle": probe["startup"]["fp_traj_match_kaggle"],
                            "ufunc_diff_vs_kaggle": probe["startup"]["ufunc_diff_vs_kaggle"],
                            "env": probe["startup"]["env"]},
            "speed_same_job": {"key": tim["key"], "colab_secs": tim["secs"], "kaggle_secs": kag["secs"],
                               "ratio": kag["secs"] / tim["secs"],
                               "identical_outcome": all(tim[k] == kag[k] for k in ("prot_viol", "energy_j", "viol_ue_s",
                                                                                  "rlf", "st_changes"))},
            "lost_colab_half": {"session": s2c["session"], "script": s2c["script"], "parts": s2c["parts"],
                                "launched_utc": s2c.get("launched_utc"), "git_head": s2c["git_head"],
                                "results_on_disk": lost_has_results, "cpu_quota": s2c["startup"]["cpu_quota"]}}


def status_sources():
    ga = [l.split() for l in open(os.path.join(SP, "e6_dev", "GATE_A_RESULT.txt"))
          if re.match(r"^(base|surge|mistune)\s", l)]
    ts = [float(r[5]) for r in ga]; ctrl = [float(r[9]) for r in ga]
    rk = {}
    for l in open(os.path.join(SP, "decision_stack", "RANK1_SUMMARY.txt")):
        name, rest = l.split(" ", 1)
        rk[name] = ast.literal_eval(rest.strip())
    pil = open(os.path.join(SP, "decision_stack", "req_pilot", "PILOT_RESULT.md"), encoding="utf-8").read()
    auc = [float(m) for m in re.findall(r"^\| (?:10|30|90) \|[^|]*\|[^|]*\| ([0-9.]+)", pil, re.M)]
    nz = float(re.search(r"Nonzero-vs-zero AUC ([0-9.]+)", pil).group(1))
    diag = open(os.path.join(SP, "decision_stack", "RANK_DIAG_RESULT.md"), encoding="utf-8").read()
    rel = float(re.search(r"pooled\s+reliability ([0-9.]+)", diag).group(1))
    prev = json.load(open(PREV))
    xm4 = {x: v["pass"] for x, v in prev["xtruce"]["M4"].items()}
    return {"e6_gateA": {"TS_over_freeze": [min(ts), max(ts)], "controllable": [min(ctrl), max(ctrl)],
                         "rule": "TS <= 0.85 x freeze AND controllable >= 0.35 (surge or mistune)"},
            "e6_rank": {m: {"rho_raw": rk[m]["rho_raw"][0], "improvement_gated": rk[m]["improvement_gated"],
                            "regret_accept_all": rk[m]["regret_accept_all"][0]} for m in ("all", "topology", "none")},
            "e6_rank_diag": {"oracle_reliability_cross_platform": rel},
            "req_pilot": {"best_sign_auc_H10_30_90": auc, "nonzero_vs_zero_auc_H10": nz},
            "xtruce_M4_pass": xm4, "xtruce_not_screenable": prev["xtruce"]["not_screenable_pairs"]}


# ======================================================================================== figures
def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=140, bbox_inches="tight"); plt.close(fig)


def fig1(v1, v2):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3))
    x = np.arange(len(CELLS)); w = 0.36
    ax = axes[0]
    for off, cs, col, lab in ((-w / 2, v1, GREY, "v1 screen seeds 150100+ (selected here)"),
                              (w / 2, v2, BLUE, "v2 fresh seeds 155000+")):
        vals = [cs[k]["Lambda"] for k in CELLS]
        lo = [cs[k]["Lambda"] - cs[k]["Lambda_ci90"][0] for k in CELLS]
        hi = [cs[k]["Lambda_ci90"][1] - cs[k]["Lambda"] for k in CELLS]
        ax.bar(x + off, vals, width=w * 0.92, color=col, label=lab)
        ax.errorbar(x + off, vals, yerr=[lo, hi], fmt="none", ecolor=INK, lw=1, capsize=3)
        for xi, v in zip(x + off, vals):
            ax.text(xi + w * 0.08, v + 0.03, f"{v:.2f}", ha="left", va="bottom", fontsize=8.6, fontweight="bold")
    ax.axhline(LAMBDA_MIN, color=RED, ls="--", lw=1.2, label="c1 threshold Λ ≥ 0.15")
    ax.set_xticks(x); ax.set_xticklabels([CELL_LBL[k] for k in CELLS], fontsize=8.4)
    ax.set_ylabel("Λ = (V_AA − V_ref) / V_ref"); ax.grid(axis="y", color=GRID, lw=0.7)
    ax.legend(fontsize=8.2, frameon=False, loc="upper left")
    ax.set_title("(a) relative loss Λ, with 5-95 % paired-seed bootstrap", loc="left", fontsize=10)
    ax = axes[1]
    for off, cs, col in ((-w / 2, v1, GREY), (w / 2, v2, BLUE)):
        vals = [cs[k]["V_AA_minus_V_ref"] for k in CELLS]
        ax.bar(x + off, vals, width=w * 0.92, color=col)
        for xi, k in zip(x + off, CELLS):
            c = cs[k]
            ax.plot([xi - w * 0.4, xi + w * 0.4], [c["lb90"]] * 2, color=INK, lw=2)
            ax.text(xi, c["V_AA_minus_V_ref"] + 1.5, f"{c['V_AA_minus_V_ref']:.1f}", ha="center", fontsize=8.4)
            ax.text(xi, -7, "c1 ✓" if c["c1"] else "c1 ✗", ha="center", fontsize=8.4,
                    color=GREEN if c["c1"] else RED, fontweight="bold")
    ax.axhline(MATERIAL, color=RED, ls="--", lw=1.2, label="c1 materiality floor 36")
    ax.legend(fontsize=8.2, frameon=False, loc="upper left")
    ax.set_ylim(-12, 95); ax.axhline(0, color="#bdb8ad", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([CELL_LBL[k] for k in CELLS], fontsize=8.4)
    ax.set_ylabel("V_AA − V_ref  (violated UE-s per protected UE-h)"); ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_title("(b) absolute loss; black tick = one-sided LB90 (must be > 0)", loc="left", fontsize=10)
    fig.suptitle("Fig 1 — Winner's curse: the three cells picked by v1's criterion 1 lose much less on fresh seeds",
                 x=0.01, ha="left", fontweight="bold", y=1.02)
    fig.text(0.01, -0.04, "Grey = v1 (the seeds on which these cells were selected); blue = v2 fresh seeds, same "
             "protocol and thresholds. P1 s3 and P3 s1 fail c1 on the 36-unit materiality floor only.",
             fontsize=8.6, color=SOFT)
    fig.tight_layout(); save(fig, "fig1_winners_curse.png")


def fig2(c):
    T, CI = c["arms"], c["R_ci90_report_side"]
    dep = c["R_static_deployable_arm"]
    rows = [("accept-all (AA)", "noarb", INK, "reference, R = 0 by definition"),
            ("QACM (published arbiter)", "qacm", BLUE, "deployable static"),
            ("best deployable static rule\n(" + dep.replace("cell:", "cell-lock ").replace("SliceGuarantee", "SG")
             .replace("PowerES", "PwrES") + ")", dep, VIOLET, "deployable static"),
            ("per-region hindsight static\n(arm 9, stage 3)", "hind", AQUA, "privileged: picked per seed on the true tape"),
            ("O_tape oracle\n(arm 10, guardrail-admissible)", "oracle", ORANGE, "privileged: true-tape lookahead")]
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, a, col, note) in zip(y, rows):
        r = T[a]["R"]
        ax.barh(yi, r, color=col, height=0.56)
        if a != "noarb":
            lo, hi = CI[a]
            ax.errorbar(r, yi, xerr=[[r - lo], [hi - r]], fmt="none", ecolor=INK, lw=1.1, capsize=4)
            ax.text(hi + 0.04, yi + 0.13, f"R = {r:.3f}", va="center", fontsize=9, fontweight="bold")
            ax.text(hi + 0.04, yi - 0.17, f"[{lo:.2f}, {hi:.2f}] · {note}", va="center", fontsize=7.8, color=SOFT)
        else:
            ax.text(0.04, yi, f"R = 0 · {note}", va="center", fontsize=8.4, color=SOFT,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    wb = dict(facecolor="white", edgecolor="none", pad=1.5)
    ax.axvline(R_OR_MIN, color=RED, ls="--", lw=1.2)
    ax.text(R_OR_MIN - 0.02, len(rows) - 0.45, "c2: R_or ≥ 0.50 ", color=RED, fontsize=8.4, va="bottom", ha="right")
    hs = c["R_static"]
    ax.axvspan(hs, hs + HEADROOM, color=RED, alpha=0.10, lw=0)
    ax.axvline(hs + HEADROOM, color=RED, ls=":", lw=1.2)
    ax.text(hs + HEADROOM + 0.02, len(rows) - 0.45, "c3: R_or ≥ R_static + 0.10", color=RED, fontsize=8.4,
            va="bottom")
    ax.axvline(1.0, color=SOFT, ls="-", lw=0.8)
    ax.text(1.02, -0.72, "R = 1: back to V_ref (SG alone)", fontsize=7.8, color=SOFT, ha="left", bbox=wb)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8.8)
    ax.set_xlim(-0.35, 2.75); ax.set_ylim(-0.95, len(rows) - 0.1)
    ax.set_xlabel("recovery R = (V_AA − V_arm) / (V_AA − V_ref)"); ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_title(f"Fig 2 — P3 surge-L40 (v2 fresh seeds): recovery ladder.  c1 ✓  c2 ✓  c3 ✓  "
                 f"c4 ✗ (ρ_sign {c['rho_sign']:.3f} < 0.70)  →  NOISE STOP",
                 loc="left", fontweight="bold", fontsize=10.6)
    fig.text(0.01, -0.05, "Intervals: 5-95 % of the paired-seed bootstrap (driver stream [6611, 3, 3], 10 000 draws). "
             "The oracle's interval is the driver's R_or CI; the others use the same draws, computed report-side.\n"
             "All arms shown are eligible (energy-matched, 4 guardrails ≤ 1.10× AA). Shaded band = the +0.10 "
             "headroom that c3 demands above the best static arm (here the hindsight static).",
             fontsize=8.3, color=SOFT)
    fig.tight_layout(); save(fig, "fig2_recovery_ladder.png")


def fig3(v1, v2):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.3), gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    names = {"svr": "all-UE SVR", "nonprot_embb_viol": "non-prot. eMBB", "ll_viol": "LL viol.", "rlf": "RLF"}
    cols = [BLUE, AQUA, VIOLET, ORANGE]
    w = 0.2
    for i, k in enumerate(CELLS):
        for j, g in enumerate(GUARD_KEYS):
            x = i + (j - 1.5) * w
            a, b = v1[k]["oracle_guard_ratio"][g], v2[k]["oracle_guard_ratio"][g]
            ax.bar(x - w * 0.22, a, width=w * 0.42, color="white", edgecolor=cols[j], hatch="////", lw=1.0)
            ax.bar(x + w * 0.22, b, width=w * 0.42, color=cols[j], label=names[g] if i == 0 else None)
            if a > GUARD:
                ax.text(x - w * 0.22, a + 0.01, f"{a:.2f}", ha="center", va="bottom", fontsize=7.4, color=RED,
                        fontweight="bold", rotation=90)
    ax.axhline(GUARD, color=RED, ls="--", lw=1.4, label="limit 1.10× AA")
    ax.axhline(1.0, color="#bdb8ad", lw=0.9)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{CELL_LBL[k]}\nenergy retention v1 {v1[k]['oracle_retention']:.2f} → v2 "
                        f"{v2[k]['oracle_retention']:.2f}" for k in CELLS], fontsize=8.2)
    ax.set_ylim(0.6, 1.5); ax.set_ylabel("oracle ÷ accept-all  (pooled, 8 seeds)")
    h, l = ax.get_legend_handles_labels()
    h += [Patch(facecolor="white", edgecolor=SOFT, hatch="////", label="v1 oracle (hatched)"),
          Patch(color=SOFT, label="v2 O_tape (solid)")]
    ax.legend(handles=h, fontsize=7.8, ncol=3, frameon=False, loc="upper left"); ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_title("(a) guardrail ratios: v1 breaks one per cell, v2 none (all < 1, max " + f"{max(max(v2[k]['oracle_guard_ratio'].values()) for k in CELLS):.3f})", loc="left", fontsize=10)
    ax = axes[1]
    kinds = [("guard_only", "guardrail only", ORANGE), ("energy_only", "energy only", BLUE), ("both", "both", VIOLET)]
    bottom = np.zeros(3)
    for key, lab, col in kinds:
        vals = np.array([v2[k]["trial_rejections"].get(key, 0) for k in CELLS], float)
        ax.bar(range(3), vals, bottom=bottom, color=col, width=0.55, label=lab, edgecolor="white", lw=1)
        bottom += vals
    for i, k in enumerate(CELLS):
        n = v2[k]["oracle_nroll_total"]
        ax.text(i, bottom[i] + 25, f"{int(bottom[i])} of {n}\ntrue-tape trials\n({bottom[i] / n:.1%})",
                ha="center", fontsize=7.8)
    ax.set_xticks(range(3)); ax.set_xticklabels([CELL_LBL[k].replace(" · ", "\n") for k in CELLS], fontsize=8.2)
    ax.set_ylim(0, max(bottom) * 1.35); ax.set_ylabel("rejected trial plans (8 episodes)")
    ax.legend(fontsize=8, frameon=False, loc="upper right"); ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_title("(b) v2: why O_tape rejected trial plans", loc="left", fontsize=10)
    fig.suptitle("Fig 3 — Guardrail-admissible oracle: the v1 breaches are gone; energy over-saving stays "
                 "(the energy bound is one-sided by design)", x=0.01, ha="left", fontweight="bold", fontsize=11, y=1.03)
    fig.tight_layout(); save(fig, "fig3_oracle_guardrails.png")


def fig4(v1, v2):
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 8.0))
    for row, (cs, tag) in enumerate(((v1, "v1 oracle (seeds 150100+)"), (v2, "v2 O_tape (seeds 155000+)"))):
        for col, k in enumerate(CELLS):
            ax = axes[row, col]; a = cs[k]["adv"]
            t, r, kp = np.array(a["true"]), np.array(a["redraw"]), np.array(a["kept"]).astype(bool)
            lim = 290
            ax.fill_between([-lim, lim], -lim, 0, color=RED, alpha=0.05, lw=0)
            ax.axhline(0, color="#bdb8ad", lw=0.9); ax.axvline(0, color="#bdb8ad", lw=0.9)
            ax.plot([-lim, lim], [-lim, lim], color=SOFT, lw=0.9, ls=":")
            ax.scatter(np.clip(t[kp], -lim, lim), np.clip(r[kp], -lim, lim), s=13, color=BLUE, alpha=0.75, lw=0,
                       label="sign kept")
            ax.scatter(np.clip(t[~kp], -lim, lim), np.clip(r[~kp], -lim, lim), s=15, facecolor="white",
                       edgecolor=RED, lw=1.0, label="sign flipped")
            ax.set_xlim(-lim * 0.3, lim); ax.set_ylim(-lim, lim)
            c = cs[k]
            ok = c["rho_sign"] >= RHO_MIN
            ax.text(0.03, 0.97, f"ρ_sign = {c['rho_sign']:.3f}  (n = {c['n_dev']})\n"
                    f"Σ redrawn / Σ true = {a['sum_redraw']:.0f} / {a['sum_true']:.0f}",
                    transform=ax.transAxes, va="top", fontsize=8.3,
                    bbox=dict(facecolor="white", edgecolor=GRID, boxstyle="round,pad=0.3"))
            ax.set_title(f"{tag}\n{CELL_LBL[k]}", loc="left", fontsize=9.2, color=INK if not ok else GREEN)
            if row == 1:
                ax.set_xlabel("advantage on the TRUE tape (AA − plan, prot. viol. UE-s / 90 s)", fontsize=8.4)
            if col == 0:
                ax.set_ylabel("advantage on a REDRAWN future", fontsize=8.8)
            ax.grid(color=GRID, lw=0.5)
    axes[0, 0].legend(fontsize=8, loc="lower right", frameon=False)
    fig.suptitle("Fig 4 — Per-deviation sign reliability: no cell reaches ρ_sign ≥ 0.70 in either version "
                 "(best: v2 P3 surge-L40, 0.676)", x=0.01, ha="left", fontweight="bold", fontsize=11, y=1.0)
    fig.text(0.01, -0.02, "Each point = one oracle decision that deviated from accept-all (8 seeds pooled). The sign "
             "used by ρ_sign is lexicographic (protected, then all-UE violations), so points on x = 0 can still count. "
             "Axes clipped at ±290 (a few points sit on the edge).", fontsize=8.4, color=SOFT)
    fig.tight_layout(); save(fig, "fig4_true_vs_redraw.png")


def fig5(v2):
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    for ax, k in zip(axes, ["P3|3", "P1|3", "P3|1"]):
        a = v2[k]["adv"]; seeds = v2[k]["seeds"]
        x = np.arange(len(seeds))
        ax.bar(x - 0.2, a["per_seed_true"], width=0.38, color=LIGHT, label="true tape (selected on)")
        cols = [BLUE if v > 0 else RED for v in a["per_seed_redraw"]]
        ax.bar(x + 0.2, a["per_seed_redraw"], width=0.38, color=cols, label="redrawn future")
        ax.axhline(0, color="#bdb8ad", lw=0.9)
        ax.set_xticks(x); ax.set_xticklabels([str(s)[-2:] for s in seeds], fontsize=8)
        ax.set_xlabel(f"seed {str(seeds[0])[:4]}xx", fontsize=8.4)
        ax.grid(axis="y", color=GRID, lw=0.6)
        main = k == "P3|3"
        ax.set_title(f"{CELL_LBL[k]}{'' if main else '  (c1 fails; descriptive)'}\n"
                     f"{a['n_seeds_redraw_pos']}/8 seeds positive · Σ redrawn {a['sum_redraw']:+.0f} "
                     f"(LB90 {a['sum_redraw_lb90_tag6612']:+.0f}) · Σ true {a['sum_true']:.0f}",
                     loc="left", fontsize=9.4 if main else 8.4, fontweight="bold" if main else "normal")
    axes[0].set_ylabel("summed oracle advantage over AA\n(protected violated UE-s, over its deviations)")
    axes[0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("Fig 5 — Per-seed re-drawn advantage (protocol §4, descriptive only): positive on 8/8 seeds only "
                 "in P3 surge-L40", x=0.01, ha="left", fontweight="bold", fontsize=11, y=1.04)
    fig.text(0.01, -0.05, "LB90 = 10th percentile of a seed bootstrap of the 8 per-seed sums (RNG [6612, pair, "
             "stratum], 10 000 draws). Blue = redrawn sum > 0, red = < 0. This quantity can never change a verdict.",
             fontsize=8.4, color=SOFT)
    fig.tight_layout(); save(fig, "fig5_per_seed_redraw.png")


def fig6(st, v1, v2):
    g, rk, pl = st["e6_gateA"], st["e6_rank"], st["req_pilot"]
    P, F, N, H = "pass", "fail", "none", "info"
    rows = [
        ("E6 v1 Gate A", "09-28 02:46", "KILL", RED, [
            ("TS ≤ 0.85× freeze", F, f"{g['TS_over_freeze'][0]:.2f}-{g['TS_over_freeze'][1]:.2f}"),
            ("controllable ≥ 0.35", F, f"{g['controllable'][0]:.2f}-{g['controllable'][1]:.2f}")]),
        ("E6 learned ranking", "09-28 05:58", "FAIL (ρ≈0)", RED, [
            ("oracle headroom", P, f"regret {rk['all']['regret_accept_all']:.0f}/slot"),
            ("learned ρ vs oracle", F, f"{rk['all']['rho_raw']:.2f}/{rk['topology']['rho_raw']:.2f}/{rk['none']['rho_raw']:.2f}"),
            ("gated gain LB > 0", F, f"{rk['all']['improvement_gated'][0]:+.1f} [{rk['all']['improvement_gated'][1]:.1f}, "
                                     f"{rk['all']['improvement_gated'][2]:.1f}]")]),
        ("Request-level pilot", "09-28 09:28", "DEAD", RED, [
            ("reject matters?", P, f"AUC {pl['nonzero_vs_zero_auc_H10']:.2f}"),
            ("sign AUC ≥ 0.70", F, "/".join(f"{a:.2f}" for a in pl["best_sign_auc_H10_30_90"]))]),
        ("E6-P v1 (Gate A v2)", "09-29 06:30", "DEAD", RED, [
            ("stage 0", P, "P2 out"),
            ("c1 loss", P, f"{sum(v1[k]['c1'] for k in CELLS)}/8 cells"),
            ("c2 recover", F, "oracle ineligible"),
            ("c3 headroom", N, "not run"),
            ("c4 ρ ≥ 0.70", H, f"{min(v1[k]['rho_sign'] for k in CELLS):.2f}-{max(v1[k]['rho_sign'] for k in CELLS):.2f}")]),
        ("xTRUCE XTS-v1", "09-29 06:47", "NOT SCREENABLE", YELLOW, [
            ("stage 0 M4", F, "QoS, LB fail"),
            ("c1 loss", N, ""), ("c2", N, ""), ("c3", N, ""), ("c4", N, "")]),
        ("E6-P v2 O_tape", "09-29 18:54", "NOISE STOP", YELLOW, [
            ("c1 loss (fresh)", P, f"{sum(v2[k]['c1'] for k in CELLS)}/3 cells"),
            ("c2 recover", P, f"R_or {v2['P3|3']['R_or']:.2f}"),
            ("c3 headroom", P, f"+{v2['P3|3']['headroom']:.2f}"),
            ("c4 ρ ≥ 0.70", F, f"{v2['P3|3']['rho_sign']:.3f}")]),
    ]
    fig, ax = plt.subplots(figsize=(13.0, 5.6))
    fill = {P: ("#e3f3ea", GREEN), F: ("#fbe3e1", RED), N: ("#f1efea", "#b8b3a8"), H: ("#fbe3e1", RED)}
    bw, bh, x0 = 1.55, 0.72, 2.9
    for i, (name, when, verd, vcol, checks) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.text(0.0, y + 0.1, name, fontsize=10, fontweight="bold", va="center")
        ax.text(0.0, y - 0.22, when + " (NST)", fontsize=8, color=SOFT, va="center")
        for j, (lab, stt, num) in enumerate(checks):
            fc, ec = fill[stt]
            bx = x0 + j * (bw + 0.12)
            box = FancyBboxPatch((bx, y - bh / 2), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.08",
                                 facecolor=fc if stt != H else "white", edgecolor=ec, lw=1.5,
                                 ls=(0, (2, 2)) if stt == H else "-")
            ax.add_patch(box)
            mark = {P: "✓ ", F: "✗ ", N: "", H: "✗ "}[stt]
            ax.text(bx + bw / 2, y + 0.14, mark + lab, ha="center", va="center", fontsize=7.7,
                    color=INK if stt != N else "#9a958a", fontweight="bold" if stt in (P, F) else "normal")
            ax.text(bx + bw / 2, y - 0.17, num, ha="center", va="center", fontsize=7.6, color=SOFT)
        vx = x0 + 5 * (bw + 0.12) + 0.1
        ax.add_patch(FancyBboxPatch((vx, y - 0.24), 1.95, 0.48, boxstyle="round,pad=0.02,rounding_size=0.2",
                                    facecolor=vcol, edgecolor="none", alpha=0.9))
        ax.text(vx + 0.975, y, verd, ha="center", va="center", fontsize=8.6, fontweight="bold",
                color="white" if vcol == RED else INK)
        if i < len(rows) - 1:
            ax.plot([0, vx + 1.95], [y - 0.5, y - 0.5], color=GRID, lw=0.7)
    ax.set_xlim(-0.1, x0 + 5 * (bw + 0.12) + 2.2); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.axis("off")
    ax.legend(handles=[Patch(facecolor="#e3f3ea", edgecolor=GREEN, label="check passed"),
                       Patch(facecolor="#fbe3e1", edgecolor=RED, label="check failed"),
                       Patch(facecolor="white", edgecolor=RED, ls=(0, (2, 2)),
                             label="failed, computed but not gating (c2 already failed)"),
                       Patch(facecolor="#f1efea", edgecolor="#b8b3a8", label="not reached")],
              loc="lower center", ncol=4, fontsize=8.2, frameon=False, bbox_to_anchor=(0.5, -0.1))
    ax.set_title("Fig 6 — Every screen so far, in order, and how far each got through its own preregistered checks",
                 loc="left", fontweight="bold", fontsize=11)
    fig.tight_layout(); save(fig, "fig6_screen_funnel.png")


# ======================================================================================== output
def _r(x, n=4):
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


def check_against_driver(v2, path):
    d = json.load(open(path))
    bad = []
    for pair in ("P1", "P3"):
        for s, r in d["pairs"][pair]["rows"].items():
            c = v2[f"{pair}|{s}"]
            pairs = [("V_AA", c["V_AA"]), ("V_ref", c["V_ref"]), ("Lambda", c["Lambda"]), ("lb90_diff", c["lb90"]),
                     ("R_or", c["R_or"]), ("rho_sign", c["rho_sign"]), ("n_deviations", c["n_dev"]),
                     ("R_static", c["R_static"])]
            pairs += [("R_or_ci90", c["R_or_ci90"]), ("Lambda_ci90", c["Lambda_ci90"])]
            if "headroom" in r:
                pairs += [("headroom", c["headroom"]), ("headroom_ci90", c["headroom_ci90"])]
            for k, mine in pairs:
                if not np.allclose(np.asarray(r[k], float), np.asarray(mine, float), rtol=1e-9, atol=1e-9):
                    bad.append((pair, s, k, r[k], mine))
            for k in ("c1", "c2", "c4") + (("c3",) if "c3" in r else ()):
                if bool(r[k]) != bool(c[k]):
                    bad.append((pair, s, k, r[k], c[k]))
            adv = r["v2_redraw_advantage"]
            for k, mine in (("sum_redraw", c["adv"]["sum_redraw"]), ("sum_true", c["adv"]["sum_true"]),
                            ("sum_redraw_lb90", c["adv"]["sum_redraw_lb90_tag6612"])):
                if abs(adv[k] - mine) > 1e-9:
                    bad.append((pair, s, k, adv[k], mine))
            if r["v2_trial_rejections"] != c["trial_rejections"]:
                bad.append((pair, s, "rejections", r["v2_trial_rejections"], c["trial_rejections"]))
        if d["pairs"][pair]["verdict"] != verdict([v2[k] for k in CELLS if k.startswith(pair)]):
            bad.append((pair, "verdict"))
    print("driver cross-check:", "ALL AGREE" if not bad else bad)
    return not bad


def main():
    J1, dirs1 = load(V1)
    J2, dirs2 = load(V2)
    v1 = {k: analyse(J1, V1["base"], k[:2], int(k[3])) for k in CELLS}
    v2 = {k: analyse(J2, V2["base"], k[:2], int(k[3])) for k in CELLS}
    # v1 must reproduce the sibling report
    prev = json.load(open(PREV))["e6p"]["cells"]
    for k in CELLS:
        for f, g in (("Lambda", "Lambda"), ("V_AA", "V_AA"), ("lb90", "lb90"), ("rho_sign", "rho_sign")):
            assert abs(v1[k][f] - prev[k][g]) <= 1e-4 * max(1, abs(prev[k][g])), (k, f, v1[k][f], prev[k][g])
    ver = {p: verdict([v2[k] for k in CELLS if k.startswith(p)]) for p in ("P1", "P3")}
    ver1 = {p: verdict([v1[k] for k in CELLS if k.startswith(p)]) for p in ("P1", "P3")}
    ops, st = ops_facts(), status_sources()
    fig1(v1, v2); fig2(v2["P3|3"]); fig3(v1, v2); fig4(v1, v2); fig5(v2); fig6(st, v1, v2)
    strip = lambda c: {kk: (vv if kk != "adv" else {a: b for a, b in vv.items() if a not in ("true", "redraw", "kept")})
                       for kk, vv in c.items()}                                  # noqa: E731
    data = {"v2": {"verdict": ver, "cells": {k: strip(c) for k, c in v2.items()}, "runs": dirs2},
            "v1_same_code": {"verdict_3cells": ver1, "cells": {k: strip(c) for k, c in v1.items()}, "runs": dirs1},
            "fig4_points": {v: {k: {kk: c["adv"][kk] for kk in ("true", "redraw", "kept")} for k, c in cs.items()}
                            for v, cs in (("v1", v1), ("v2", v2))},
            "ops": ops, "status_sources": st,
            "cpu_h_v2_total": sum(d["cpu_h"] for d in dirs2.values())}
    json.dump(_r(data), open(os.path.join(OUT, "figures_data.json"), "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)
    print("v2 verdicts:", ver, "| v1 (3 cells, same code):", ver1)
    for k in CELLS:
        c, o = v2[k], v1[k]
        print(f"{k} v1 Lam {o['Lambda']:.3f} -> v2 Lam {c['Lambda']:.3f} CI {np.round(c['Lambda_ci90'], 3)} "
              f"VAA {c['V_AA']:.1f} Vref {c['V_ref']:.1f} diff {c['V_AA_minus_V_ref']:.1f} LB {c['lb90']:.1f} "
              f"c1 {c['c1']} {c['c1_parts']}")
        print(f"   R_or {c['R_or']:.3f} CI {np.round(c['R_or_ci90'], 3)} ret {c['oracle_retention']:.3f} "
              f"guard {_r(c['oracle_guard_ratio'], 3)} rej {c['trial_rejections']} bykey {c['trial_rejections_by_key']} "
              f"nroll {c['oracle_nroll_total']} rho {c['rho_sign']:.4f} n {c['n_dev']} "
              f"Rdep {c['R_static_deployable']:.3f} ({c['R_static_deployable_arm']}) Rst {c['R_static']:.3f} "
              f"({c['R_static_arm']}) QACM {c['arms']['qacm']['R']:.3f}"
              + (f" head {c['headroom']:.3f} CI {np.round(c['headroom_ci90'], 3)} c3 {c['c3']} "
                 f"hind regions/seed {c['hind_regions_changed_per_seed']}" if "headroom" in c else ""))
        a = c["adv"]
        print(f"   adv redraw {a['sum_redraw']:.0f} LB90 {a['sum_redraw_lb90_tag6612']:.1f} true {a['sum_true']:.0f} "
              f"pos {a['n_seeds_redraw_pos']}/8 | POST HOC corr {a['corr_true_redraw_POST_HOC']:.3f} "
              f"share {a['kept_share_POST_HOC']:.3f} Rdn {a['R_denoised_POST_HOC']:.3f} | v1 share "
              f"{o['adv']['kept_share_POST_HOC']:.3f} v1 Rdn {o['adv']['R_denoised_POST_HOC']:.3f}")
        print("   ladder CI", {a_: np.round(v, 3).tolist() for a_, v in c["R_ci90_report_side"].items()
                                if a_ in ("qacm", "hind", "oracle", c["R_static_deployable_arm"])})
    print("runs v2:", {d: (x["n_jobs"], round(x["cpu_h"], 2), x["first_header_utc"], x["git_head"], x["platform"])
                       for d, x in dirs2.items()}, "total CPU-h", round(data["cpu_h_v2_total"], 1))
    print("ops:", json.dumps(_r({k: v for k, v in ops.items() if k != "selftest"}))[:900])
    print("selftest:", ops["selftest"]["passed"], ops["selftest"]["verify"], ops["selftest"]["resume"])
    print("status:", json.dumps(_r(st)))
    if "--check" in sys.argv:
        assert check_against_driver(v2, sys.argv[sys.argv.index("--check") + 1])
    print("wrote fig1..fig6 + figures_data.json to", OUT)


if __name__ == "__main__":
    main()
