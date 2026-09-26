"""Figures for the 2026-09-07 E-series learned-arm + E2 design-decision record.

All numbers are transcribed directly from the consolidated 2026-09-07 record
(truth-free DEV prototypes; scratchpad `fm_eseries/` and `e4_feas/`). Nothing is
invented for the plot: only values that appear in the source are drawn, and
where only endpoint measurements exist that is stated in the caption.

Run from the repo root (single-core, headless):
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    MPLBACKEND=Agg .venv/Scripts/python.exe \
    reports/2026-09-07-eseries-learned-arms-and-e2-design-decision/make_figs.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent

# palette (matches the report css)
INK, MUT, FAINT, RULE = "#1a1d22", "#4c545e", "#79818c", "#d8dce2"
DISC, ORAC, WARN = "#0E7A73", "#A8412F", "#C8792B"

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


def fig1_fm1_lambda_sweep():
    """FM-1 utility_weight lambda-sweep realized R: a NULL.

    Only the endpoint realized_pos values are given in the source (lambda=0 and
    lambda=0.5); the pre-declared grid is {0, 0.1, 0.25, 0.5}. Endpoints are
    drawn as measured markers; the connecting segment is interpolation across the
    grid (source: 'flat then mildly harmful' / 'strictly worse'), flagged in the
    caption. No intermediate value is fabricated.
    """
    lam = [0.0, 0.5]
    perfect = [-42.333, -42.379]      # perfect world-model realized_pos
    decoy = [-57.48, -58.01]          # decoy world-model realized_pos

    fig, ax = plt.subplots(figsize=(6.9, 3.8))

    ax.plot(lam, perfect, color=DISC, lw=1.4, ls="--", alpha=0.55, zorder=2)
    ax.scatter(lam, perfect, s=64, color=DISC, edgecolor="white", linewidth=0.8,
               zorder=3, label="perfect world-model")
    ax.plot(lam, decoy, color=ORAC, lw=1.4, ls="--", alpha=0.55, zorder=2)
    ax.scatter(lam, decoy, s=64, color=ORAC, edgecolor="white", linewidth=0.8,
               zorder=3, label="decoy world-model")

    # annotate the null: lambda=0 already at/near oracle, lambda>0 never helps
    ax.annotate("λ=0 realizes the oracle\n(norm_regret 0.0)",
                (0.0, -42.333), (0.04, -44.6), fontsize=8.0, color=DISC, ha="left",
                arrowprops=dict(arrowstyle="-", color=DISC, lw=0.7))
    ax.annotate("λ=0 regret 0.19;\nλ>0 strictly worse",
                (0.5, -58.01), (0.20, -55.6), fontsize=8.0, color=ORAC, ha="left",
                arrowprops=dict(arrowstyle="-", color=ORAC, lw=0.7))
    ax.text(0.25, -42.28,
            "Δ realized R over λ in {0..0.5}: perfect −42.333 → −42.379   ·   decoy −57.48 → −58.01",
            fontsize=7.8, color=MUT, ha="center")

    ax.set_xlim(-0.03, 0.55)
    ax.set_xticks([0.0, 0.1, 0.25, 0.5])
    ax.set_xticklabels(["0", "0.1", "0.25", "0.5"])
    ax.set_xlabel("utility_weight  lambda  (pre-declared grid; endpoints measured)")
    ax.set_ylabel("realized R  (realized_pos)")
    ax.set_title("FM-1 is a NULL on the E-series: adding λ never improves realized R",
                 fontsize=10.3)
    ax.legend(frameon=False, fontsize=8.6, loc="center right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=RULE, lw=0.6, alpha=0.5)
    fig.tight_layout()
    _finish(fig, "fig1_fm1_lambda_sweep.png")


def fig2_e4_deconfound():
    """E4 deconfounding: point-with-CI, sign-flip made visible."""
    # (label, slope, lo, hi, color, note)  -- hi/lo None => reference point, no CI
    rows = [
        ("correlational pooler\n(K~A on 90:10 pool)", 3.3046, 3.29, 3.32, ORAC,
         "reproduces frozen +3.2916\n(wrong sign)"),
        ("deconfounded\n(K~A on D=1 do-subset)", -1.0423, -1.201, -0.883, DISC,
         "CI excludes 0; recovers α=−1"),
        ("oracle backdoor\n(K~A+Z, reference)", -1.0000, None, None, MUT,
         "confirms {Z} is a valid backdoor"),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    ys = list(range(len(rows)))[::-1]

    for y, (lab, slope, lo, hi, color, note) in zip(ys, rows):
        if lo is not None:
            ax.plot([lo, hi], [y, y], color=color, lw=2.6, solid_capstyle="round",
                    zorder=2)
            for x in (lo, hi):
                ax.plot([x, x], [y - 0.08, y + 0.08], color=color, lw=1.6, zorder=2)
        ax.scatter([slope], [y], s=90, color=color, edgecolor="white", linewidth=0.9,
                   zorder=3)
        ax.text(slope, y + 0.20, f"{slope:+.3f}", color=color, fontsize=9.2,
                ha="center", fontweight="bold")
        ax.text(3.85, y, note, color=MUT, fontsize=7.6, va="center", ha="left")

    # reference lines: true alpha = -1 and zero (no effect)
    ax.axvline(-1.0, color=DISC, lw=1.2, ls=":", zorder=1)
    ax.text(-1.0, 2.62, "true α = −1", color=DISC, fontsize=8.2, ha="center")
    ax.axvline(0.0, color=FAINT, lw=1.0, ls="--", zorder=1)
    ax.text(0.0, -0.55, "no effect", color=FAINT, fontsize=7.8, ha="center")

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.4)
    ax.set_ylim(-0.7, 3.0)
    ax.set_xlim(-1.9, 6.0)
    ax.set_xlabel("estimated slope on A → K_out   (95% CI where shown; 10 seeds)")
    ax.set_title("E4: observational correlation flips the sign; the do-subset recovers α=−1",
                 fontsize=10.2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="x", color=RULE, lw=0.6, alpha=0.5)
    fig.tight_layout()
    _finish(fig, "fig2_e4_deconfound.png")


if __name__ == "__main__":
    fig1_fm1_lambda_sweep()
    fig2_e4_deconfound()
    print("done ->", OUT)
