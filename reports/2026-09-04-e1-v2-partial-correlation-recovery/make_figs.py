"""Figures for the 2026-09-04 E1 v2 partial-correlation recovery delta report.

Reads committed provenance plus the local (gitignored) v2 003 discovery artifact:
  - docs/benchmark/plan003_frozen_artifacts/discovery.json   (v1 |beta| scores + global cut)
  - docs/benchmark/plan006_recovery_digest/summary.json      (v2 10-seed recovery envelope)
  - runs/e1slice/plan003_frozen/discovery_v2.json            (v2 |partial_corr| scores + per-target cuts)

Regenerate the last if absent:
  uv run python -m scripts.e1_slice discover-v2 --dataset runs/e1slice/plan003_frozen --force

Run from the repo root:
  uv run python reports/2026-09-04-e1-v2-partial-correlation-recovery/make_figs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
V1 = json.loads((ROOT / "docs/benchmark/plan003_frozen_artifacts/discovery.json").read_text())
SUMMARY = json.loads((ROOT / "docs/benchmark/plan006_recovery_digest/summary.json").read_text())
V2 = json.loads((ROOT / "runs/e1slice/plan003_frozen/discovery_v2.json").read_text())

# palette (matches the report css)
INK, MUT, FAINT, RULE = "#1a1d22", "#4c545e", "#79818c", "#d8dce2"
DISCV, ORAC = "#0E7A73", "#A8412F"
BAND_NCP, BAND_KPI, ZERO = "#0E7A73", "#C8792B", "#b9c0c9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10.5, "axes.edgecolor": RULE, "axes.linewidth": 0.9,
    "axes.labelcolor": MUT, "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
    "axes.titlecolor": INK, "figure.dpi": 150, "savefig.dpi": 150,
})

INPUTS = ["P0", "P1", "P2", "P3", "K0", "K1", "K2", "K3"]
KPIS = ["K0", "K1", "K2", "K3"]
# true E1 parents as (kpi_row, input_col); inputs = [P0 P1 P2 P3 K0 K1 K2 K3] -> K0=col4, K1=col5
TRUE = {(0, 0), (1, 1), (2, 2), (2, 4), (3, 3), (3, 5)}   # K0<-P0,K1<-P1,K2<-{P2,K0},K3<-{P3,K1}
KPI_KPI = {(2, 4), (3, 5)}                                  # K2<-K0, K3<-K1


def _finish(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def _grid(ax, mask, title, recall):
    for r in range(4):
        for c in range(8):
            true = (r, c) in TRUE
            got = bool(mask[r][c])
            if got and true:
                fc, ec, hatch = DISCV, DISCV, None      # true positive
            elif true and not got:
                fc, ec, hatch = "white", ORAC, "////"   # false negative (missed)
            else:
                fc, ec, hatch = "#f2f4f6", RULE, None    # true negative
            ax.add_patch(Rectangle((c, 3 - r), 1, 1, facecolor=fc, edgecolor=ec,
                                   linewidth=1.6 if true else 0.8, hatch=hatch))
    ax.set_xlim(0, 8); ax.set_ylim(0, 4)
    ax.set_xticks(np.arange(8) + 0.5); ax.set_xticklabels(INPUTS, fontsize=8.5)
    ax.set_yticks(np.arange(4) + 0.5); ax.set_yticklabels(KPIS[::-1], fontsize=8.5)
    ax.set_xlabel("input  [ NCP params | KPIs ]", fontsize=9)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{title}  ·  recall {recall}", fontsize=10.5)


def fig1_recovery_grid():
    v1_mask = V1["binary_mask"]
    v2_mask = SUMMARY["replicates"][0]["binary_mask"]  # 003 == replicate-00; v2 == oracle
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    _grid(axes[0], v1_mask, "v1  |β| + global cut", "0.667")
    axes[0].set_ylabel("predicted KPI", fontsize=9)
    _grid(axes[1], v2_mask, "v2  |partial corr| + per-target cut", "1.000")
    leg = [Patch(facecolor=DISCV, edgecolor=DISCV, label="recovered edge"),
           Patch(facecolor="white", edgecolor=ORAC, hatch="////", label="missed (KPI→KPI)"),
           Patch(facecolor="#f2f4f6", edgecolor=RULE, label="true negative")]
    fig.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3,
               frameon=False, fontsize=8.6, handlelength=1.3)
    fig.suptitle("v1 misses both KPI→KPI edges (K2←K0, K3←K1);  v2 recovers all six",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    _finish(fig, "fig1_recovery_grid.png")


def _classify(scores):
    ncp, kpi, zero = [], [], []
    for r in range(4):
        for c in range(8):
            v = abs(scores[r][c])
            if (r, c) in KPI_KPI:
                kpi.append(v)
            elif (r, c) in TRUE:
                ncp.append(v)
            else:
                zero.append(v)
    return ncp, kpi, zero


def fig2_score_gap():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9), sharey=True)
    rng = np.random.default_rng(0)

    def strip(ax, vals, y, color):
        x = np.array(vals)
        ax.scatter(x, y + (rng.random(len(x)) - 0.5) * 0.16, s=44, color=color,
                   edgecolor="white", linewidth=0.6, zorder=3)

    # v1: |beta|, single global cut (0.674) lands between the KPI band (~0.44) and NCP band (~0.9)
    n1, k1, z1 = _classify(V1["scores"])
    thr1 = V1["threshold"]
    strip(axes[0], z1, 0.0, ZERO); strip(axes[0], k1, 1.0, BAND_KPI); strip(axes[0], n1, 2.0, BAND_NCP)
    axes[0].axvline(thr1, color=INK, lw=1.4, ls="--", zorder=2)
    axes[0].text(thr1, 2.62, f"global cut {thr1:.3f}", color=INK, fontsize=8.4, ha="center")
    axes[0].annotate("both KPI→KPI edges\nfall below the cut", (float(np.mean(k1)), 1.0),
                     (0.06, 1.52), fontsize=7.8, color=BAND_KPI, ha="left",
                     arrowprops=dict(arrowstyle="-", color=BAND_KPI, lw=0.7))
    axes[0].set_title("v1  ·  score = |standardized OLS coef|", fontsize=10)
    axes[0].set_xlabel("edge score")

    # v2: |partial_corr|, per-target cuts (~0.5); clean 0-vs-1 separation
    n2, k2, z2 = _classify(V2["scores"])
    thr2 = V2["threshold"]
    strip(axes[1], z2, 0.0, ZERO); strip(axes[1], k2, 1.0, BAND_KPI); strip(axes[1], n2, 2.0, BAND_NCP)
    axes[1].axvspan(min(thr2) - 1e-4, max(thr2) + 1e-4, color=FAINT, alpha=0.16, zorder=1)
    axes[1].text(float(np.mean(thr2)), 2.62, "per-target cuts ≈ 0.500", color=INK, fontsize=8.4,
                 ha="center")
    axes[1].set_title("v2  ·  score = |partial correlation|", fontsize=10)
    axes[1].set_xlabel("edge score")

    for ax in axes:
        ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["non-edge", "KPI→KPI", "NCP→KPI"])
        ax.set_ylim(-0.7, 3.0); ax.set_xlim(-0.05, 1.08)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("The score-scale fix: partial correlation lifts KPI→KPI edges to ≈1, clearing the cut",
                 fontsize=10.6, y=1.02)
    fig.tight_layout()
    _finish(fig, "fig2_score_gap.png")


def fig3_recovery_stability():
    g = SUMMARY["graph_recovery"]
    blocks = [("overall", INK), ("ncp_kpi", DISCV), ("kpi_kpi", ORAC)]
    fig, ax = plt.subplots(figsize=(6.8, 3.1))
    label_y = {"overall": 1.22, "ncp_kpi": 1.15, "kpi_kpi": 1.08}
    for b, color in blocks:
        vals = np.array(g[b]["recall"]["values"], dtype=float)
        jitter = {"overall": 0.0, "ncp_kpi": -0.022, "kpi_kpi": 0.022}[b]
        ax.scatter(range(10), vals + jitter, s=42, color=color, edgecolor="white",
                   linewidth=0.6, zorder=3)
        ax.plot(range(10), vals + jitter, color=color, lw=1.3, alpha=0.5)
        ax.text(0.0, label_y[b], f"{b.replace('_', '→')}:  recall {vals.mean():.3f} (σ = 0, min = max = 1)",
                color=color, fontsize=8.4, va="center", fontweight="bold")
    ax.set_xticks(range(10)); ax.set_xticklabels([f"s{r}" for r in range(10)])
    ax.set_ylim(-0.08, 1.30); ax.set_xlim(-0.4, 9.4)
    ax.set_ylabel("edge recall"); ax.set_xlabel("replicate (env_seed = weight_seed = r, split_seed = 0)")
    ax.set_title("v2 recovery is 1.000 on every block, every seed — zero variance", fontsize=10.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=RULE, lw=0.6, alpha=0.5)
    fig.tight_layout()
    _finish(fig, "fig3_recovery_stability.png")


if __name__ == "__main__":
    fig1_recovery_grid()
    fig2_score_gap()
    fig3_recovery_stability()
    print("done ->", OUT)
