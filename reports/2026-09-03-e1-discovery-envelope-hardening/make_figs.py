"""Figures for the 2026-09-03 E1 delta report.

Reads only committed/tracked provenance where possible:
  - docs/benchmark/plan003_frozen_artifacts/discovery.json  (single-seed labelled scores)
  - docs/benchmark/plan004_envelope_digest/summary.json      (10-seed envelope)
and the local (gitignored) per-seed discovery.json under runs/ for the per-seed threshold band.

Run from the repo root:  uv run python reports/2026-09-03-e1-discovery-envelope-hardening/make_figs.py
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
DISC = json.loads((ROOT / "docs/benchmark/plan003_frozen_artifacts/discovery.json").read_text())
SUMMARY = json.loads((ROOT / "docs/benchmark/plan004_envelope_digest/summary.json").read_text())

# palette (matches the report css)
INK, MUT, FAINT, RULE = "#1a1d22", "#4c545e", "#79818c", "#d8dce2"
ORAC, DENSE, DISCV = "#A8412F", "#2f6d92", "#0E7A73"
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
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def fig1_recovery_graph():
    mask = np.array(DISC["binary_mask"])              # (4,8) discovered
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    for r in range(4):
        for c in range(8):
            true = (r, c) in TRUE
            disc = bool(mask[r, c])
            if disc and true:
                fc, ec, hatch = DISCV, DISCV, None      # true positive
            elif true and not disc:
                fc, ec, hatch = "white", ORAC, "////"   # false negative (missed)
            else:
                fc, ec, hatch = "#f2f4f6", RULE, None
            ax.add_patch(Rectangle((c, 3 - r), 1, 1, facecolor=fc, edgecolor=ec,
                                   linewidth=1.6 if true else 0.8, hatch=hatch))
    ax.set_xlim(0, 8); ax.set_ylim(0, 4)
    ax.set_xticks(np.arange(8) + 0.5); ax.set_xticklabels(INPUTS)
    ax.set_yticks(np.arange(4) + 0.5); ax.set_yticklabels(KPIS[::-1])
    ax.set_xlabel("input  [ NCP params | KPIs ]"); ax.set_ylabel("predicted KPI")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Discovered vs. true E1 structure  ·  precision 1.00, recall 0.667", fontsize=11)
    leg = [Patch(facecolor=DISCV, edgecolor=DISCV, label="recovered (NCP→KPI, 4)"),
           Patch(facecolor="white", edgecolor=ORAC, hatch="////", label="missed (KPI→KPI, 2)"),
           Patch(facecolor="#f2f4f6", edgecolor=RULE, label="true negative")]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
              frameon=False, fontsize=8.6, handlelength=1.3)
    _finish(fig, "fig1_recovery_graph.png")


def fig2_score_gap():
    scores = np.array(DISC["scores"])
    thr = DISC["threshold"]
    # classify the 8x4 cells
    ncp, kpi, zero = [], [], []
    for r in range(4):
        for c in range(8):
            v = scores[r, c]
            if (r, c) in KPI_KPI:
                kpi.append(v)
            elif (r, c) in TRUE:
                ncp.append(v)
            else:
                zero.append(v)
    # per-seed thresholds (local runs, gitignored) for the stability band
    seed_thr = []
    for d in sorted((ROOT / "runs/e1slice-v2-envelope").glob("replicate-*/discovery.json")):
        try:
            seed_thr.append(json.loads(d.read_text())["threshold"])
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    rng = np.random.default_rng(0)

    def strip(vals, y, color, label):
        x = np.array(vals)
        ax.scatter(x, y + (rng.random(len(x)) - 0.5) * 0.14, s=46, color=color,
                   edgecolor="white", linewidth=0.6, zorder=3, label=label)

    strip(zero, 0.0, ZERO, "non-edge (|coef|≈0)")
    strip(kpi, 1.0, BAND_KPI, "KPI→KPI edge (true parent)")
    strip(ncp, 2.0, BAND_NCP, "NCP→KPI edge (true parent)")

    if seed_thr:
        ax.axvspan(min(seed_thr), max(seed_thr), color=FAINT, alpha=0.12, zorder=0)
    ax.axvline(thr, color=INK, lw=1.4, ls="--", zorder=2)
    ax.text(thr, 2.55, f" largest_gap cut = {thr:.3f}", color=INK, fontsize=8.6, va="center")
    if seed_thr:
        ax.text(np.mean(seed_thr), -0.62, f"10-seed cut range {min(seed_thr):.3f}–{max(seed_thr):.3f}",
                color=FAINT, fontsize=8, ha="center")
    ax.annotate("K0→K2 = 0.439", (0.439, 1.0), (0.30, 1.42), fontsize=8, color=BAND_KPI,
                arrowprops=dict(arrowstyle="-", color=BAND_KPI, lw=0.8))
    ax.annotate("K1→K3 = 0.456", (0.456, 1.0), (0.52, 1.42), fontsize=8, color=BAND_KPI,
                arrowprops=dict(arrowstyle="-", color=BAND_KPI, lw=0.8))
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["non-edge", "KPI→KPI", "NCP→KPI"])
    ax.set_ylim(-0.8, 2.9); ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("edge score  = |standardized OLS coefficient|")
    ax.set_title("Why the KPI→KPI edges are missed: the global cut lands in the band gap", fontsize=10.5)
    ax.legend(loc="upper left", frameon=False, fontsize=8, handletextpad=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _finish(fig, "fig2_score_gap.png")


def fig3_envelope_mse():
    pred = SUMMARY["prediction"]
    arms = [("oracle", ORAC), ("dense", DENSE), ("discovered", DISCV)]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    rng = np.random.default_rng(1)
    for i, (arm, color) in enumerate(arms):
        vals = np.array(pred[arm]["mse"]["values"])
        ax.scatter(i + (rng.random(len(vals)) - 0.5) * 0.22, vals, s=40, color=color,
                   edgecolor="white", linewidth=0.6, zorder=3)
        ax.plot([i - 0.30, i + 0.30], [vals.mean()] * 2, color=color, lw=2.4, zorder=4)
        ax.text(i + 0.34, vals.mean(), f"mean {vals.mean():.2e}", ha="left", va="center",
                fontsize=8, color=color, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(1e-9, 4e-2)
    ax.set_xticks(range(3)); ax.set_xticklabels([a for a, _ in arms])
    ax.set_ylabel("held-out one-step MSE  (log)")
    ax.set_title("Three-arm prediction envelope across 10 seeds  ·  oracle < dense < discovered (all 10)",
                 fontsize=10.3)
    ax.set_xlim(-0.5, 2.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=RULE, lw=0.6, alpha=0.6)
    _finish(fig, "fig3_envelope_mse.png")


def fig4_recovery_stability():
    g = SUMMARY["graph_recovery"]
    blocks = [("overall", INK), ("ncp_kpi", DISCV), ("kpi_kpi", ORAC)]
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    for i, (b, color) in enumerate(blocks):
        vals = np.array(g[b]["recall"]["values"])
        ax.scatter(range(10), vals, s=40, color=color, edgecolor="white", linewidth=0.6, zorder=3)
        ax.plot(range(10), vals, color=color, lw=1.3, alpha=0.5)
        ax.text(9.25, vals[-1], f"{b}  R={vals.mean():.3f} (σ=0)", color=color, fontsize=8.4, va="center")
    ax.set_xticks(range(10)); ax.set_xticklabels([f"s{r}" for r in range(10)])
    ax.set_ylim(-0.08, 1.08); ax.set_xlim(-0.4, 12.6)
    ax.set_ylabel("edge recall"); ax.set_xlabel("replicate (env_seed = weight_seed = r, split_seed = 0)")
    ax.set_title("Recovery is deterministic across seeds: zero variance on every block", fontsize=10.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=RULE, lw=0.6, alpha=0.5)
    _finish(fig, "fig4_recovery_stability.png")


if __name__ == "__main__":
    fig1_recovery_graph()
    fig2_score_gap()
    fig3_envelope_mse()
    fig4_recovery_stability()
    print("done ->", OUT)
