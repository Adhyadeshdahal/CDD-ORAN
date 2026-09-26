"""Figures for the 2026-09-06 E2 discovery-method / live-trap / end-to-end thesis report.

All numbers are transcribed verbatim from the consolidated technical record
(reports/2026-09-06-e2-discovery-method-and-livetrap.md), which is itself the as-run
provenance for the frozen runs (RCoT-v2 protocol_commit_v2 8052e10; live-trap env 9d87a60)
and for the two DEV prototypes (end-to-end thesis spine; stratified P0->K5 recovery). No
number is invented; where the source records a range or "~" estimate it is reproduced as-is.

Run from repo root (single-core, Agg):
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  MPLBACKEND=Agg ./.venv/Scripts/python.exe \
  reports/2026-09-06-e2-discovery-method-and-livetrap/make_figs.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parent

# palette (matches the report css)
INK, MUT, FAINT, RULE = "#1a1d22", "#4c545e", "#79818c", "#d8dce2"
DISC, ORAC, WARN = "#0E7A73", "#A8412F", "#C8792B"
NEUTRAL = "#b9c0c9"

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


def fig1_method_arc():
    """The discovery-method arc: NCP->KPI recall and KPI->KPI over-selection across methods."""
    # (label, NCP recall, KPI->KPI FP rate per 36 candidates, color, env tag)
    rows = [
        ("pdCor\n(retired, seed-0)", 0.813, 17 / 36, ORAC, "OLD"),
        ("RCoT-v1\nB=99", 0.163, 6 / 360, NEUTRAL, "OLD"),
        ("RCoT-v2\nB=299", 0.769, 6 / 360, DISC, "OLD"),
        ("RCoT-v2\nB=299 (live trap)", 0.650, 6 / 360, DISC, "REDESIGN"),
    ]
    labels = [r[0] for r in rows]
    recall = [r[1] for r in rows]
    fp = [r[2] for r in rows]
    colors = [r[3] for r in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.5))

    ax = axes[0]
    bars = ax.bar(x, recall, width=0.62, color=colors, edgecolor="white", linewidth=0.8)
    # hatch the live-trap (redesigned-env) bar to mark the env change
    bars[3].set_hatch("////")
    bars[3].set_edgecolor(DISC)
    for xi, v in zip(x, recall):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=8.6, color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.8)
    ax.set_ylabel("NCP→KPI recall  (16 true edges)")
    ax.set_title("Recall: over-selection fixed, then\nrecall restored at B=299", fontsize=9.6)
    ax.grid(axis="y", color=RULE, lw=0.6, alpha=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    bars = ax.bar(x, fp, width=0.62, color=colors, edgecolor="white", linewidth=0.8)
    bars[3].set_hatch("////")
    bars[3].set_edgecolor(DISC)
    fp_lbl = ["17/36", "6/360", "6/360", "6/360"]
    for xi, v, t in zip(x, fp, fp_lbl):
        ax.text(xi, v + 0.012, t, ha="center", fontsize=8.4, color=INK)
    ax.set_ylim(0, 0.52)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.8)
    ax.set_ylabel("KPI→KPI false-positive rate  (of 36)")
    ax.set_title("Over-selection: block-perm null\ncrushes the 9× pdCor FDR", fontsize=9.6)
    ax.grid(axis="y", color=RULE, lw=0.6, alpha=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.suptitle("E2 discovery-method arc:  pdCor → RCoT-v1 (B=99) → RCoT-v2 (B=299)",
                 fontsize=10.6, y=1.03)
    fig.tight_layout()
    _finish(fig, "fig1_method_arc.png")


def fig2_p0k5_recovery():
    """P0->K5 harmful-edge recovery by method (out of 10 seeds) + NCP recall context."""
    # (label, P0->K5 recovered /10, NCP recall /16, color)
    rows = [
        ("M1 frozen\nRCoT-v2", 0, 10.5, NEUTRAL),
        ("M2 do(P0)-\nmarginal", 0, 11.2, NEUTRAL),
        ("M3 STRATIFIED\n[primary]", 10, 15.3, DISC),
        ("M4 superset\n+prune", 0, 11.2, NEUTRAL),
    ]
    labels = [r[0] for r in rows]
    p0k5 = [r[1] for r in rows]
    ncp = [r[2] for r in rows]
    colors = [r[3] for r in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.5))

    ax = axes[0]
    ax.bar(x, p0k5, width=0.6, color=colors, edgecolor="white", linewidth=0.8)
    for xi, v in zip(x, p0k5):
        ax.text(xi, v + 0.2, f"{v}/10", ha="center", fontsize=8.8,
                color=DISC if v else FAINT, fontweight="bold" if v else "normal")
    ax.set_ylim(0, 11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.8)
    ax.set_ylabel("harmful P0→K5 recovered  (of 10 seeds)")
    ax.set_title("Only stratified conditioning recovers\nthe co-parent-gated harmful edge", fontsize=9.6)
    ax.grid(axis="y", color=RULE, lw=0.6, alpha=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    ax.bar(x, ncp, width=0.6, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(16, color=FAINT, lw=1.0, ls="--")
    ax.text(3.35, 16, "16 true", ha="right", va="bottom", fontsize=7.6, color=FAINT)
    ncp_lbl = ["~10.5", "11.2", "15.3", "11.2"]
    for xi, v, t in zip(x, ncp, ncp_lbl):
        ax.text(xi, v + 0.25, t, ha="center", fontsize=8.4, color=INK)
    ax.set_ylim(0, 17.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.8)
    ax.set_ylabel("NCP→KPI recall  (mean count of 16)")
    ax.set_title("M3 raises overall recall while keeping\nKPI→KPI FP low (0.30)", fontsize=9.6)
    ax.grid(axis="y", color=RULE, lw=0.6, alpha=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.suptitle("P0→K5 recovery experiment (DEV prototype, 10 seeds, n=4000, truth-free, q=0.05)",
                 fontsize=10.2, y=1.03)
    fig.tight_layout()
    _finish(fig, "fig2_p0k5_recovery.png")


def fig3_endtoend_thesis():
    """3-arm end-to-end thesis over the 32 K5-positive states: realized R, regret, trap-fall."""
    # (arm, mean realized R, mean regret, trap-fall /32, true-K5 satisfied /32, color)
    arms = [
        ("do-nothing\n(floor)", -102.56, 60.23, 32, 0, NEUTRAL),
        ("oracle /\ncomplete fan-out", -42.33, 0.00, 6, 26, ORAC),
        ("discovered\n(decoy)", -57.48, 15.15, 30, 2, DISC),
    ]
    labels = [a[0] for a in arms]
    realized = [a[1] for a in arms]
    regret = [a[2] for a in arms]
    trapfall = [a[3] / 32 for a in arms]
    colors = [a[5] for a in arms]
    x = np.arange(len(arms))

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.6))

    ax = axes[0]
    ax.bar(x, realized, width=0.6, color=colors, edgecolor="white", linewidth=0.8)
    for xi, v, reg in zip(x, realized, regret):
        ax.text(xi, v + 1.5, f"{v:.2f}", ha="center", va="bottom", fontsize=8.6, color=INK)
        tag = "regret 0" if reg == 0 else f"regret +{reg:.2f}"
        ax.text(xi, 7.0, tag, ha="center", fontsize=8.0,
                color=DISC if reg == 0 else (WARN if reg < 30 else FAINT))
    ax.axhline(0, color=RULE, lw=0.8)
    ax.set_ylim(-120, 16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.0)
    ax.set_ylabel("mean realized R on TRUE env")
    ax.set_title("Realized objective over 32 K5-positive states\n(higher = better)", fontsize=9.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    ax.bar(x, trapfall, width=0.6, color=colors, edgecolor="white", linewidth=0.8)
    tf_lbl = ["32/32", "6/32", "30/32"]
    for xi, v, t in zip(x, trapfall, tf_lbl):
        ax.text(xi, v + 0.02, t, ha="center", fontsize=8.6, color=INK)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.0)
    ax.set_ylabel("trap-fall fraction  (of 32)")
    ax.set_title("Complete fan-out avoids the trap;\ndiscovered decoy falls in 30/32", fontsize=9.4)
    ax.grid(axis="y", color=RULE, lw=0.6, alpha=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    leg = [Patch(facecolor=ORAC, label="oracle / complete fan-out (reference)"),
           Patch(facecolor=DISC, label="discovered structure (decoy, missing P0→K5)"),
           Patch(facecolor=NEUTRAL, label="do-nothing floor")]
    fig.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=1,
               frameon=False, fontsize=8.0, handlelength=1.2)
    fig.suptitle("End-to-end thesis on the LIVE trap (DEV spine): discovered structure → "
                 "world-model → decision", fontsize=9.4, y=1.03)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _finish(fig, "fig3_endtoend_thesis.png")


if __name__ == "__main__":
    fig1_method_arc()
    fig2_p0k5_recovery()
    fig3_endtoend_thesis()
    print("done ->", OUT)
