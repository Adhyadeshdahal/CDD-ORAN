"""Figures for the 2026-09-09 design-adaptive M3 calibration stress test.

Numbers read live from the truth-free stress-test JSONs (no hand transcription):
  scratchpad/design_adaptive_m3/results_axisA.json      (design-diversity continuum, m_active 1..6)
  scratchpad/design_adaptive_m3/results_axisB.json      (cause<->non-parent-proxy co-actuation, rho)
  scratchpad/design_adaptive_m3/results_thinprobe.json  (trajectory-thinning serial-dependence probe)
Records are raw p_naive/p_adaptive + abstain flags; BH selection is applied here (inline, so the
report is self-contained) with the mechanism-induced true-edge set. All truth-free DEV; nothing frozen.

Run from repo root (headless, single-core):
  MPLBACKEND=Agg .venv/Scripts/python.exe \
    reports/2026-09-09-design-adaptive-m3-calibration-stress/make_figs.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SP = ROOT / "scratchpad" / "design_adaptive_m3"
B = 299
FLOOR = 1.0 / (B + 1.0)
Q = 0.05
TRUE = {(0, 0), (0, 1), (1, 2), (1, 3), (1, 4), (2, 5), (3, 1), (3, 6)}
GATED, KK, WIDTH, WRONG = (1, 2), (3, 6), (1, 3), (1, 0)

INK, MUT, FAINT, RULE = "#1a1d22", "#4c545e", "#79818c", "#d8dce2"
DISC, ORAC, WARN, ACC = "#0E7A73", "#A8412F", "#C8792B", "#0E5F73"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10.5, "axes.edgecolor": RULE, "axes.linewidth": 0.9,
    "axes.labelcolor": MUT, "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
    "axes.titlecolor": INK, "figure.dpi": 150, "savefig.dpi": 150,
})


def _finish(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def bh(pvals, q=Q):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    thr = (np.arange(1, m + 1) / m) * q
    passed = p[order] <= thr
    d = np.zeros(m, bool)
    if passed.any():
        d[order[: np.nonzero(passed)[0].max() + 1]] = True
    return d


def load_units(path):
    recs = json.load(open(path))["records"]
    U = defaultdict(lambda: {"pn": [None] * 10, "pa": [None] * 10, "abst": None, "ffa": None})
    for r in recs:
        k = (r["m_active"], r["rho"], r["seed"], r["target"])
        U[k]["pn"][r["candidate"]] = r["p_naive"]
        U[k]["pa"][r["candidate"]] = r["p_adaptive"]
        U[k]["abst"] = r["abstain"]
        U[k]["ffa"] = r["floor_frac_adaptive"]
    return U


def per_point(path):
    """grid_key -> dict of scored metrics."""
    U = load_units(path)
    G = defaultdict(list)
    for (m, rho, s, t), u in U.items():
        G[(m, rho)].append((s, t, u))
    out = {}
    for gk, rows in G.items():
        fn = tn = fa = ta = nab = nu = 0
        rec = defaultdict(lambda: [0, 0])  # edge -> [hits_adapt, denom_nonabstain]
        wp = 0
        wp_den = 0
        for (s, t, u) in rows:
            nu += 1
            ab = u["abst"]
            nab += ab
            dn = bh(u["pn"]); da = np.zeros(10, bool) if ab else bh(u["pa"])
            for c in range(10):
                it = (t, c) in TRUE
                if dn[c]:
                    tn += 1; fn += (not it)
                if da[c]:
                    ta += 1; fa += (not it)
            for e in TRUE:
                if e[0] == t:
                    if not ab:
                        rec[e][1] += 1
                        rec[e][0] += bool(da[e[1]])
            if t == WRONG[0]:
                wp_den += 1
                wp += bool((np.zeros(10, bool) if ab else bh(u["pa"]))[WRONG[1]])
        out[gk] = {
            "naive_fdr": fn / tn if tn else float("nan"),
            "adapt_fdr": fa / ta if ta else float("nan"),
            "abstain": nab / nu,
            "rec": {e: (rec[e][0], rec[e][1]) for e in TRUE},
            "wrong_parent": (wp, wp_den),
        }
    return out


def fig1_axisA_calibration():
    P = per_point(SP / "results_axisA.json")
    ms = [1, 2, 3, 4, 6]
    naive = [P[(m, 0.0)]["naive_fdr"] for m in ms]
    adapt = [P[(m, 0.0)]["adapt_fdr"] for m in ms]
    abst = [P[(m, 0.0)]["abstain"] for m in ms]
    x = np.arange(len(ms))

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.bar(x, abst, 0.62, color=RULE, edgecolor="white", zorder=1,
           label="abstention rate (adaptive)")
    ax.plot(x, naive, "-o", color=ORAC, lw=2.0, ms=7, zorder=4, label="naive M3 realized FDR")
    adapt_plot = [a if a == a else np.nan for a in adapt]
    ax.plot(x, adapt_plot, "--s", color=ACC, lw=2.0, ms=6, zorder=4,
            label="adaptive M3 realized FDR (non-abstaining)")
    for xi, (n, a, ab) in enumerate(zip(naive, adapt, abst)):
        ax.text(xi, n + 0.03, f"{n:.2f}", ha="center", fontsize=8.0, color=ORAC, fontweight="bold")
        if a == a:
            ax.text(xi, a - 0.055, f"{a:.2f}", ha="center", fontsize=8.0, color=ACC, fontweight="bold")
        ax.text(xi, ab + 0.02, f"{ab:.0%}", ha="center", fontsize=7.6, color=MUT)
    ax.axhline(Q, color=INK, lw=1.1, ls=":", zorder=3)
    ax.text(len(ms) - 0.5, Q + 0.02, "q = 0.05", fontsize=8.0, color=INK, ha="right")
    # danger zone: uncontrolled FDR that abstention does NOT catch (m=3,4)
    ax.axvspan(1.5, 3.5, color=WARN, alpha=0.10, zorder=0)
    ax.text(2.5, 0.93, "uncontrolled FDR,\nabstention does NOT fire", ha="center",
            fontsize=8.4, color=WARN, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"m={m}\n{'single-param (E1-like)' if m==1 else 'i.i.d. (E2/E3-like)' if m==6 else ''}"
                        for m in ms], fontsize=8.6)
    ax.set_xlabel("actuation refresh  m_active  (params redrawn per step; carry the rest)")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.02)
    ax.set_title("Adaptive components do NOT restore FDR control, and abstention misses the middle\n"
                 "adaptive FDR tracks naive everywhere; abstention fires only at the extreme (m=1,2)",
                 fontsize=9.8)
    ax.legend(frameon=False, fontsize=8.2, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=RULE, lw=0.6, alpha=0.5)
    fig.tight_layout()
    _finish(fig, "fig1_axisA_calibration.png")


def fig2_axisA_recovery():
    P = per_point(SP / "results_axisA.json")
    ms = [1, 2, 3, 4, 6]

    def rec(e):
        out = []
        for m in ms:
            h, d = P[(m, 0.0)]["rec"][e]
            out.append(h / d if d else np.nan)  # fraction over non-abstaining seeds
        return out

    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    x = np.arange(len(ms))
    ax.plot(x, rec(GATED), "-o", color=ORAC, lw=2.2, ms=7, label="gated  K1<-P2  (buried cause)")
    ax.plot(x, rec(KK), "-s", color=DISC, lw=2.2, ms=6, label="KPI->KPI  K3<-K0lag")
    ax.plot(x, rec(WIDTH), "--^", color=FAINT, lw=1.6, ms=6,
            label="width-gate  K1<-P3  (known M3 weak spot)")
    ax.axhline(0.8, color=INK, lw=1.0, ls=":", zorder=2)
    ax.text(len(ms) - 0.5, 0.82, "8/10 bar", fontsize=8.0, color=INK, ha="right")
    ax.axvspan(-0.4, 0.5, color=ORAC, alpha=0.07, zorder=0)
    ax.text(0.05, 0.5, "m=1:\nall units abstain\n(no adaptive decls)", fontsize=7.8, color=ORAC, ha="left")
    ax.set_xticks(x); ax.set_xticklabels([f"m={m}" for m in ms])
    ax.set_xlabel("actuation refresh  m_active")
    ax.set_ylabel("recovery fraction over non-abstaining seeds")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Recovery is strong wherever the design is non-degenerate\n"
                 "gated + KPI->KPI edges recovered ~10/10; only the variance-modulating width parent lags",
                 fontsize=9.8)
    ax.legend(frameon=False, fontsize=8.2, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=RULE, lw=0.6, alpha=0.5)
    fig.tight_layout()
    _finish(fig, "fig2_axisA_recovery.png")


def fig3_axisB_identifiability():
    P = per_point(SP / "results_axisB.json")
    rhos = [0.0, 0.5, 0.9, 0.99, 1.0]
    wp = [P[(6, r)]["wrong_parent"][0] / P[(6, r)]["wrong_parent"][1] for r in rhos]
    abst = [P[(6, r)]["abstain"] for r in rhos]
    gated = [P[(6, r)]["rec"][GATED][0] / max(1, P[(6, r)]["rec"][GATED][1]) for r in rhos]
    x = np.arange(len(rhos))

    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    ax.plot(x, wp, "-o", color=ORAC, lw=2.2, ms=7,
            label="wrong-parent FP: declares P0->K1 (P0 a NON-parent proxy)")
    ax.plot(x, gated, "-s", color=DISC, lw=1.8, ms=6, label="true cause K1<-P2 still recovered")
    ax.plot(x, abst, "--D", color=ACC, lw=2.0, ms=6, label="abstention rate (fail-closed)")
    for xi, (w, ab) in enumerate(zip(wp, abst)):
        ax.text(xi, w + 0.03, f"{w:.0%}", ha="center", fontsize=8.0, color=ORAC, fontweight="bold")
        ax.text(xi, ab + 0.03, f"{ab:.0%}", ha="center", fontsize=8.0, color=ACC, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"{r:g}" for r in rhos])
    ax.set_xlabel("cause<->proxy co-actuation  rho  (m_active = 6, full-rank)")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Identifiability edge: as co-actuation -> 1 the method confidently declares the WRONG parent,\n"
                 "and fail-closed abstention never fires (the test is confident, just wrong)",
                 fontsize=9.6)
    ax.legend(frameon=False, fontsize=8.0, loc="center right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=RULE, lw=0.6, alpha=0.5)
    fig.tight_layout()
    _finish(fig, "fig3_axisB_identifiability.png")


def fig4_diagnostics():
    # (a) serial-dependence probe: full n=4000 vs thinned 250, naive type-I at m=1 and m=6
    tp = json.load(open(SP / "results_thinprobe.json"))["records"]
    thin = defaultdict(lambda: [0, 0])
    for r in tp:
        if (r["target"], r["candidate"]) in TRUE:
            continue
        thin[r["m_active"]][1] += 1
        thin[r["m_active"]][0] += (r["p_naive"] <= Q)
    # full-n type-I from axis A raw records
    full = defaultdict(lambda: [0, 0])
    for r in json.load(open(SP / "results_axisA.json"))["records"]:
        if r["m_active"] in (1, 6) and (r["target"], r["candidate"]) not in TRUE:
            full[r["m_active"]][1] += 1
            full[r["m_active"]][0] += (r["p_naive"] <= Q)

    # (b) FDR vs floor_frac bins (axis A, RAW adaptive decls no abstention)
    edges = [0.1, 0.3, 0.5, 0.7, 1.01]
    binf = [0] * (len(edges) - 1); bint = [0] * (len(edges) - 1)
    U = load_units(SP / "results_axisA.json")
    for (m, rho, s, t), u in U.items():
        da = bh(u["pa"]); ff = u["ffa"]
        bi = next((b for b in range(len(edges) - 1) if edges[b] <= ff < edges[b + 1]), None) \
            if ff >= edges[0] else None
        # first bin catches [0,0.3): fold <0.1 into first
        if ff < edges[0]:
            bi = 0
        for c in range(10):
            if da[c]:
                bint[bi] += 1
                binf[bi] += ((t, c) not in TRUE)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    # left
    x = np.arange(2); w = 0.36
    fullv = [full[1][0] / full[1][1], full[6][0] / full[6][1]]
    thinv = [thin[1][0] / thin[1][1], thin[6][0] / thin[6][1]]
    axL.bar(x - w / 2, fullv, w, color=ORAC, edgecolor="white", label="full n=4000")
    axL.bar(x + w / 2, thinv, w, color=ACC, edgecolor="white", label="thinned 1 row/traj (n=250)")
    for xi, (f, t) in enumerate(zip(fullv, thinv)):
        axL.text(xi - w / 2, f + 0.02, f"{f:.2f}", ha="center", fontsize=8.0, color=ORAC, fontweight="bold")
        axL.text(xi + w / 2, t + 0.02, f"{t:.2f}", ha="center", fontsize=8.0, color=ACC, fontweight="bold")
    axL.axhline(Q, color=INK, lw=1.0, ls=":")
    axL.set_xticks(x); axL.set_xticklabels(["m=1 (E1-like)", "m=6 (i.i.d.)"])
    axL.set_ylabel("naive per-test type-I")
    axL.set_ylim(0, 1.05)
    axL.set_title("Serial-dependence probe: thinning\ncollapses m=1 inflation 1.00 -> 0.23", fontsize=9.2)
    axL.legend(frameon=False, fontsize=7.8, loc="upper right")
    # right
    labels = ["[0,0.3)", "[0.3,0.5)", "[0.5,0.7)", "[0.7,1.0]"]
    rate = [binf[b] / bint[b] if bint[b] else np.nan for b in range(4)]
    colors = [DISC, WARN, ORAC, ORAC]
    axR.bar(np.arange(4), rate, 0.64, color=colors, edgecolor="white")
    for xi, (rt, tt) in enumerate(zip(rate, bint)):
        if rt == rt:
            axR.text(xi, rt + 0.02, f"{rt:.2f}\n(n={tt})", ha="center", fontsize=7.6, color=INK)
    axR.axvline(1.5, color=INK, lw=1.1, ls="--")
    axR.text(1.55, 0.9, "tau=0.5\nabstain ->", fontsize=7.8, color=INK, ha="left")
    axR.set_xticks(np.arange(4)); axR.set_xticklabels(labels, fontsize=8.2)
    axR.set_xlabel("floor_frac_adaptive bin")
    axR.set_ylabel("realized FP-rate of declared edges")
    axR.set_ylim(0, 1.05)
    axR.set_title("The diagnostic does not separate: the\n[0.3,0.5) bin (kept) is already 55% false",
                  fontsize=9.2)
    for ax in (axL, axR):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=RULE, lw=0.6, alpha=0.5)
    fig.tight_layout()
    _finish(fig, "fig4_diagnostics.png")


def fig5_certificate_concordance():
    """Sol's design-based admission certificate vs the known param->KPI FDR across the grid."""
    v = json.load(open(SP / "certificate_validation.json"))
    rows = v["rows"]
    labels = [r["point"].replace("A:", "").replace("B:rho", "rho=") for r in rows]
    fdr = [r["param_fdr"] for r in rows]
    cells = [r["cell"] for r in rows]
    cmap = {"correct-admit": DISC, "correct-ABSTAIN": FAINT, "DANGEROUS-false-admit": ORAC}
    colors = [cmap[c] for c in cells]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    ax.bar(x, fdr, 0.66, color=colors, edgecolor="white", zorder=3)
    ax.axhline(v["controlled_tau"], color=INK, lw=1.1, ls=":", zorder=2)
    ax.text(len(rows) - 0.5, v["controlled_tau"] + 0.02, "controlled bar (FDR<=0.10)",
            fontsize=8.0, color=INK, ha="right")
    ax.axvline(4.5, color=RULE, lw=1.0)
    ax.text(2.0, 0.82, "axis A: design diversity\n(C1 randomization gate)", ha="center",
            fontsize=8.4, color=MUT)
    ax.text(7.0, 0.82, "axis B: co-actuation\n(C2 collinearity gate)", ha="center",
            fontsize=8.4, color=MUT)
    for xi, r in enumerate(rows):
        tag = "ADMIT" if r["admit"] else "abstain"
        ax.text(xi, fdr[xi] + 0.02, tag, ha="center", fontsize=7.0,
                color=(DISC if r["admit"] and r["controlled"] else ORAC if r["admit"] else MUT),
                fontweight="bold" if r["admit"] else "normal", rotation=90, va="bottom")
    # flag the dangerous cell
    for xi, r in enumerate(rows):
        if r["cell"] == "DANGEROUS-false-admit":
            ax.annotate("false-admit\n(|pcorr|=0.48<0.80,\nweak-ID, FDR 0.22)",
                        (xi, fdr[xi]), (xi - 1.6, 0.55), fontsize=7.8, color=ORAC, ha="center",
                        arrowprops=dict(arrowstyle="->", color=ORAC, lw=0.9))
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.0, rotation=0)
    ax.set_ylabel("param->KPI realized M3 FDR")
    ax.set_ylim(0, 1.0)
    ax.set_title("Design-based admission certificate vs realized FDR: green=correct-admit, grey=correct-abstain,\n"
                 "red=dangerous false-admit. Gates the whole design-diversity axis; one gap at moderate co-actuation.",
                 fontsize=9.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=RULE, lw=0.6, alpha=0.5)
    fig.tight_layout()
    _finish(fig, "fig5_certificate_concordance.png")


def fig6_crt_validity_power():
    """Phase 2: assignment-aware CRT vs naive M3 across co-actuation -- validity (left) and power (right)."""
    d = json.load(open(SP / "crt_validation.json"))
    rows = d["rows"]
    rho = [r["rho"] for r in rows]
    x = np.arange(len(rho))
    nf = [r["naive"]["FDR"] for r in rows]; cf = [r["crt"]["FDR"] for r in rows]
    nwp = [r["naive"]["wrong_parent_x10"] / 10 for r in rows]
    cwp = [r["crt"]["wrong_parent_x10"] / 10 for r in rows]
    ccause = [r["crt"]["cause_recovery_x10"] / 10 for r in rows]
    r2 = [r["residual_R2_cause_P2"] for r in rows]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.8, 4.2))
    # LEFT: validity
    axL.plot(x, nf, "-o", color=ORAC, lw=2.0, ms=6, label="naive FDR")
    axL.plot(x, cf, "-s", color=ACC, lw=2.2, ms=6, label="CRT FDR")
    axL.plot(x, nwp, "--o", color=ORAC, lw=1.4, ms=5, alpha=0.7, label="naive wrong-parent rate")
    axL.plot(x, cwp, "--s", color=ACC, lw=1.6, ms=5, alpha=0.9, label="CRT wrong-parent rate")
    axL.axhline(Q, color=INK, lw=1.0, ls=":")
    axL.text(len(rho) - 0.5, Q + 0.03, "q=0.05", fontsize=7.8, color=INK, ha="right")
    axL.set_xticks(x); axL.set_xticklabels([f"{r:g}" for r in rho])
    axL.set_xlabel("cause<->proxy co-actuation  rho")
    axL.set_ylabel("rate"); axL.set_ylim(-0.03, 1.02)
    axL.set_title("VALIDITY: CRT holds FDR + kills the wrong-parent FP\nat every rho; naive fails from rho=0.5",
                  fontsize=9.2)
    axL.legend(frameon=False, fontsize=7.4, loc="center left")
    # RIGHT: power + residual info
    axR.plot(x, ccause, "-s", color=DISC, lw=2.2, ms=7, label="CRT cause recovery (K1<-P2)")
    axR.plot(x, r2, "--^", color=FAINT, lw=1.6, ms=6, label="residual R2 of cause (=rho^2)")
    axR.axvline(4.0, color=ORAC, lw=1.0, ls="--")
    axR.text(3.95, 0.5, "exact co-actuation\n-> ineligible (both parents)", fontsize=7.6,
             color=ORAC, ha="right")
    axR.set_xticks(x); axR.set_xticklabels([f"{r:g}" for r in rho])
    axR.set_xlabel("cause<->proxy co-actuation  rho")
    axR.set_ylabel("rate"); axR.set_ylim(-0.03, 1.05)
    axR.set_title("POWER: cause recovery holds to rho=0.99,\nthen cliffs at exact non-identification",
                  fontsize=9.2)
    axR.legend(frameon=False, fontsize=7.6, loc="center left")
    for ax in (axL, axR):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=RULE, lw=0.6, alpha=0.5)
    fig.tight_layout()
    _finish(fig, "fig6_crt_validity_power.png")


def fig7_noise_robustness():
    d = json.load(open(SP / "noise_validation.json"))
    fracs = d["fracs"]
    def series(rho, key, method="crt"):
        return [next(r[method][key] for r in d["rows"] if r["frac"] == f and r["rho"] == rho) for f in fracs]
    x = np.arange(len(fracs))
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    # CRT FDR (validity) vs noise
    ax.plot(x, series(0.0, "FDR"), "-s", color=ACC, lw=2.0, ms=6, label="CRT error rate, no co-actuation")
    ax.plot(x, series(0.5, "FDR"), "-D", color=DISC, lw=2.0, ms=6, label="CRT error rate, rho=0.5")
    # CRT cause recovery (power) vs noise
    ax.plot(x, [c / 10 for c in series(0.0, "cause_recovery_x10")], "--o", color=ACC, lw=1.6, ms=6,
            alpha=0.8, label="CRT buried-cause recovery, no co-actuation")
    ax.plot(x, [c / 10 for c in series(0.5, "cause_recovery_x10")], "--^", color=DISC, lw=1.6, ms=6,
            alpha=0.8, label="CRT buried-cause recovery, rho=0.5")
    # naive FDR at corners (contrast)
    corners = [(r["frac"], r["rho"], r["naive"]["FDR"]) for r in d["rows"] if r["has_naive"] and r["rho"] == 0.5]
    for f, rho, fdr in corners:
        ax.plot(fracs.index(f), fdr, "x", color=ORAC, ms=9, mew=2)
    ax.plot([], [], "x", color=ORAC, ms=9, mew=2, label="naive error rate, rho=0.5 (stays broken)")
    ax.axhline(Q, color=INK, lw=1.0, ls=":"); ax.text(len(fracs) - 0.5, Q + 0.03, "q=0.05", fontsize=7.8, color=INK, ha="right")
    ax.set_xticks(x); ax.set_xticklabels([f"{f:g}" for f in fracs])
    ax.set_xlabel("observation noise (fraction of each KPI's std)")
    ax.set_ylabel("rate"); ax.set_ylim(-0.03, 1.05)
    ax.set_title("Noise robustness: CRT stays valid (error rate ~q) AND keeps finding the buried cause\n"
                 "(10/10) up to 0.50 sigma, with and without co-actuation; naive stays broken under co-actuation",
                 fontsize=9.0)
    ax.legend(frameon=False, fontsize=7.6, loc="center right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=RULE, lw=0.6, alpha=0.5)
    fig.tight_layout()
    _finish(fig, "fig7_noise_robustness.png")


def fig8_break_fix():
    d = json.load(open(SP / "break_fix_validation.json"))
    rows = d["rows"]; g = [r["gamma"] for r in rows]; x = np.arange(len(g))
    iid = [r["iid"]["A_falsepos_rate"] for r in rows]
    blk = [r["block"]["A_falsepos_rate"] for r in rows]
    sta = [r["state"]["A_falsepos_rate"] for r in rows]
    rec = [r["iid"]["true_cause_recovery_rate"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    ax.plot(x, iid, "-o", color=ORAC, lw=2.4, ms=7, label="i.i.d. CRT: false alarm on confounded knob")
    ax.plot(x, blk, "-s", color=ACC, lw=2.2, ms=6, label="block-CRT (log batch IDs): fixed")
    ax.plot(x, sta, "--D", color=DISC, lw=1.8, ms=6, label="state-CRT (log hidden state): fixed")
    ax.plot(x, rec, ":^", color=FAINT, lw=1.6, ms=6, label="true cause still recovered (all)")
    for xi, v in enumerate(iid):
        ax.text(xi, v + 0.03 if v < 0.5 else v - 0.06, f"{v:.2f}", ha="center", fontsize=8.0,
                color=ORAC, fontweight="bold")
    ax.axhline(Q, color=INK, lw=1.0, ls=":"); ax.text(len(g) - 0.5, Q + 0.03, "q=0.05", fontsize=7.8, color=INK, ha="right")
    ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" for v in g])
    ax.set_xlabel("hidden-state confounding strength (gamma)")
    ax.set_ylabel("false-alarm rate on the innocent knob")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("The deliberate breaker: unlogged adaptive assignment false-alarms 100% under the i.i.d. CRT;\n"
                 "logging the batch (block-CRT) or the hidden state (state-CRT) fixes it to 0% -- this defines the contract",
                 fontsize=8.8)
    ax.legend(frameon=False, fontsize=7.8, loc="center right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=RULE, lw=0.6, alpha=0.5)
    fig.tight_layout()
    _finish(fig, "fig8_break_fix.png")


if __name__ == "__main__":
    fig1_axisA_calibration()
    fig2_axisA_recovery()
    fig3_axisB_identifiability()
    fig4_diagnostics()
    fig5_certificate_concordance()
    fig6_crt_validity_power()
    fig7_noise_robustness()
    fig8_break_fix()
    print("done ->", OUT)
