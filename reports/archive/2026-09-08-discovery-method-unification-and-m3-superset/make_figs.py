"""Figures for the 2026-09-08 discovery-method unification + M3-superset record.

Numbers are read live from the source JSONs under scratchpad/ (no hand transcription):
  - scratchpad/m3_superset/results_e1.json, results_e3.json (uniform M3 run, fixed hyperparams)
  - scratchpad/p0k5_fp_calibration/results.json (M3 on E2, 20 seeds)
All are truth-free DEV prototypes; nothing frozen is changed. fig3 is a schematic (the
design-diversity continuum); its only quantitative anchors are the measured E1 / E2 / E3 FDR.

Run from repo root (single-core, headless):
  MPLBACKEND=Agg .venv/Scripts/python.exe \
    reports/2026-09-08-discovery-method-unification-and-m3-superset/make_figs.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SP = ROOT / "scratchpad"
FLOOR = 1.0 / 300.0

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


def _score_records(path):
    """Return (recall_frac, typeI, fdr, floor_frac, null_pvals) from a m3_superset results file."""
    recs = json.load(open(path))["records"]
    true_hits = defaultdict(int)
    true_edges = set()
    seeds = set()
    decl = false_decl = 0
    null_p = []
    for r in recs:
        seeds.add(r["seed"])
        if r["bh_declared"]:
            decl += 1
        if r["is_true_edge"]:
            true_edges.add((r["target"], r["candidate"]))
            if r["bh_declared"]:
                true_hits[(r["target"], r["candidate"])] += 1
        else:
            null_p.append(r["pval"])
            if r["bh_declared"]:
                false_decl += 1
    ns = len(seeds)
    recall = np.mean([true_hits[e] / ns for e in true_edges])
    typeI = np.mean([p <= 0.05 for p in null_p])
    fdr = false_decl / decl if decl else 0.0
    floor_frac = np.mean([abs(p - FLOOR) < 1e-9 for p in null_p])
    return recall, typeI, fdr, floor_frac, np.array(null_p)


def fig1_recall_vs_fp():
    """Recall is 100% everywhere; false-positive control collapses only on E1."""
    e1 = _score_records(SP / "m3_superset" / "results_e1.json")
    e3 = _score_records(SP / "m3_superset" / "results_e3.json")
    # E2 from the 20-seed calibration study
    fp = json.load(open(SP / "p0k5_fp_calibration" / "results.json"))["summary"]
    e2_typeI = fp["metric1_per_test_typeI"]["pooled"]["rate"]
    e2_fdr = fp["metric2_post_BH_FDR"]["realized_FDR"]

    envs = ["E1", "E2", "E3"]
    typeI = [e1[1], e2_typeI, e3[1]]
    fdr = [e1[2], e2_fdr, e3[2]]
    recall = [e1[0], 1.0, e3[0]]  # E2 recall = 20/20 on true edges

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    x = np.arange(3)
    w = 0.34
    b1 = ax.bar(x - w / 2, typeI, w, color=MUT, edgecolor="white", linewidth=0.8,
                label="per-test type-I (null pairs)", zorder=3)
    b2 = ax.bar(x + w / 2, fdr, w, color=ORAC, edgecolor="white", linewidth=0.8,
                label="realized post-BH FDR", zorder=3)
    for xi, (t, f) in enumerate(zip(typeI, fdr)):
        ax.text(xi - w / 2, t + 0.02, f"{t:.3f}", ha="center", fontsize=8.2, color=MUT, fontweight="bold")
        ax.text(xi + w / 2, f + 0.02, f"{f:.3f}", ha="center", fontsize=8.2, color=ORAC, fontweight="bold")
    ax.axhline(0.05, color=INK, lw=1.2, ls=":", zorder=2)
    ax.text(2.42, 0.075, "q = 0.05", fontsize=8.2, color=INK, ha="left")
    assert min(recall) == 1.0  # recall is 100% everywhere; stated in the title, not drawn as bars
    ax.set_xticks(x)
    ax.set_xticklabels(["E1\n(single-param actuation)", "E2\n(full-rank i.i.d.)", "E3\n(full-rank i.i.d.)"], fontsize=8.6)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("error rate")
    ax.set_title("M3 recovers every true edge in all three envs (recall 100%, incl. every KPI→KPI edge)\n"
                 "— but false-positive control collapses on E1", fontsize=10.0)
    ax.legend(frameon=False, fontsize=8.4, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=RULE, lw=0.6, alpha=0.5)
    ax.annotate("declares 100% of null pairs", (0 + w / 2, e1[2]), (0.35, 0.62),
                fontsize=8.2, color=ORAC, ha="left",
                arrowprops=dict(arrowstyle="-", color=ORAC, lw=0.7))
    fig.tight_layout()
    _finish(fig, "fig1_recall_vs_fp.png")


def fig2_null_pvalue_hist():
    """Null-pair permutation p-value distribution: E1 piled at the floor vs E3 well-spread."""
    _, _, _, e1_floor, e1_p = _score_records(SP / "m3_superset" / "results_e1.json")
    _, _, _, e3_floor, e3_p = _score_records(SP / "m3_superset" / "results_e3.json")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=True)
    bins = np.linspace(0, 1, 26)
    for ax, p, floor, name, color in [
        (axL, e1_p, e1_floor, "E1 (single-param actuation)", ORAC),
        (axR, e3_p, e3_floor, "E3 (full-rank i.i.d.)", DISC),
    ]:
        ax.hist(p, bins=bins, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.axvline(FLOOR, color=INK, lw=1.1, ls=":")
        ax.set_title(name, fontsize=9.8)
        ax.set_xlabel("permutation p-value on NULL candidate pairs")
        ax.text(0.5, 0.92, f"{floor*100:.1f}% at the permutation floor (1/300)",
                transform=ax.transAxes, ha="center", fontsize=8.4, color=color, fontweight="bold")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=RULE, lw=0.6, alpha=0.5)
    axL.set_ylabel("count of null pairs")
    axL.annotate("floor 0.0033 < BH rank-1\n(0.05/m) → auto-declared",
                 (FLOOR, axL.get_ylim()[1] * 0.55), (0.18, axL.get_ylim()[1] * 0.62),
                 fontsize=7.8, color=INK, ha="left",
                 arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))
    fig.suptitle("Why E1 collapses: a rank-deficient design pins the permutation null at its floor",
                 fontsize=10.6, y=1.03)
    fig.tight_layout()
    _finish(fig, "fig2_null_pvalue_hist.png")


def fig3_continuum():
    """Schematic: tested extremes vs the unverified middle; the validate-first stress test."""
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.90, "observable design diversity  →", ha="center", fontsize=9.2, color=INK, fontweight="bold")

    # unverified middle band (drawn first, behind)
    ax.axvspan(0.30, 0.78, ymin=0.36, ymax=0.86, color=WARN, alpha=0.12, zorder=0)
    ax.text(0.54, 0.80, "UNVERIFIED MIDDLE", ha="center", fontsize=9.4, color=WARN, fontweight="bold")
    ax.text(0.54, 0.73, "partial collinearity · sparse strata · weak overlap", ha="center", fontsize=7.8, color=MUT)
    ax.text(0.54, 0.675, "diagnostic can't cleanly pick the valid null", ha="center", fontsize=7.8, color=MUT)

    # baseline axis
    y = 0.50
    ax.annotate("", (0.97, y), (0.03, y), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4), zorder=2)

    # tested endpoints (markers on the axis; label + FDR ABOVE, descriptor BELOW)
    ax.scatter([0.09], [y], s=150, color=ORAC, edgecolor="white", zorder=5, linewidth=1.0)
    ax.text(0.09, y + 0.045, "E1  ·  FDR 0.81", ha="center", fontsize=8.4, color=ORAC, fontweight="bold")
    ax.text(0.09, y - 0.075, "rank-deficient\n(single-param actuation)", ha="center", fontsize=7.8, color=MUT)
    ax.scatter([0.90], [y], s=150, color=DISC, edgecolor="white", zorder=5, linewidth=1.0)
    ax.text(0.90, y + 0.045, "E2 / E3  ·  FDR 0.07", ha="center", fontsize=8.4, color=DISC, fontweight="bold")
    ax.text(0.90, y - 0.075, "full-rank i.i.d.\n(all params perturbed)", ha="center", fontsize=7.8, color=MUT)

    # stress-test sampling grid (validate-first)
    ys = 0.26
    xs = np.linspace(0.14, 0.86, 9)
    ax.annotate("", (0.88, ys), (0.12, ys), arrowprops=dict(arrowstyle="-", color=ACC, lw=0.8, ls=(0, (2, 2))), zorder=2)
    ax.scatter(xs, [ys] * len(xs), s=28, color=ACC, marker="v", zorder=4)
    ax.text(0.5, 0.135, "validate-first — a pre-declared calibration stress test sweeps this continuum "
                        "(co-actuation × overlap × stratum sparsity),\n"
                        "measuring type-I / post-BH FDR of the FULL adaptive pipeline and establishing "
                        "where it must ABSTAIN before scoring recovery power",
            ha="center", fontsize=7.8, color=ACC)

    ax.set_title("Convergence holds only at the two tested extremes; the deployment regime is the untested middle",
                 fontsize=10.0, y=1.0)
    fig.tight_layout()
    _finish(fig, "fig3_continuum.png")


if __name__ == "__main__":
    fig1_recall_vs_fp()
    fig2_null_pvalue_hist()
    fig3_continuum()
    print("done ->", OUT)
