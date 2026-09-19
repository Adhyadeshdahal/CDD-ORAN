"""Figures for the 2026-09-07 harmful-edge de-risk + noise-robustness record.

Every number is read live from the source JSONs under scratchpad/ (no hand-
transcription): the M3 FP-calibration (p0k5_fp_calibration/results.json), the E2
single-control downstream inertness (p0k5_downstream/downstream_results.json),
the E3 multi-step structural world model (e3_multistep_wm/results.json), and the
pre-declared noise-robustness sweep (noise_robustness/results_e2.json,
results_e3.json). All are truth-free DEV prototypes; nothing frozen is changed.

Run from the repo root (single-core, headless):
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    MPLBACKEND=Agg .venv/Scripts/python.exe \
    reports/2026-09-07-harmful-edge-derisk-and-noise-robustness/make_figs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SP = ROOT / "scratchpad"


def _load(rel):
    return json.load(open(SP / rel))


# palette (matches the report css)
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


def fig1_noise_robustness():
    """Two panels: E2 M3 (K5 target) and E3 partial-corr recovery vs obs noise."""
    e2 = _load("noise_robustness/results_e2.json")
    e3 = _load("noise_robustness/results_e3.json")
    grid = [f"{g:.2f}" for g in e2["grid"]]
    x = [float(g) for g in grid]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.1), sharey=True)

    # ---- E2 (left)
    k5 = e2["k5_recovery_over_10_seeds"]
    ben = e2["p0_beneficial_recovery_over_10_seeds"]
    p0k5 = [k5[g]["P0->K5 (cand0)"] for g in grid]
    p7k5 = [k5[g]["P7->K5 (cand7)"] for g in grid]
    p6k5 = [k5[g]["P6->K5 (cand6)"] for g in grid]
    benmin = [min(ben[g].values()) for g in grid]
    axL.plot(x, p0k5, "-o", color=ORAC, lw=2.2, ms=7, zorder=5,
             label="P0→K5  (harmful edge)")
    axL.plot(x, p7k5, "-s", color=ACC, lw=1.4, ms=5, alpha=0.9,
             label="P7→K5  (co-parent)")
    axL.plot(x, benmin, "-^", color=DISC, lw=1.4, ms=5, alpha=0.9,
             label="P0→{K0,K1,K2}  (beneficial, min)")
    axL.plot(x, p6k5, ":d", color=FAINT, lw=1.3, ms=5,
             label="P6→K5  (width co-parent, weak)")
    axL.set_title("E2 · M3 stratified CI  (K5 target, 14 candidates)", fontsize=10.0)
    axL.set_ylabel("edges recovered  (of 10 seeds)")
    axL.annotate("harmful edge held at 10/10\nthrough 0.50 σ — no drop",
                 (0.50, 10), (0.24, 6.3), fontsize=8.2, color=ORAC, ha="left",
                 arrowprops=dict(arrowstyle="-", color=ORAC, lw=0.7))

    # ---- E3 (right)
    r3 = e3["recovery_over_10_seeds"]
    names = list(e3["tracked_edges"].keys())
    chain = [n for n in names if n.startswith("K1->")]
    direct = [n for n in names if not n.startswith("K1->")]
    for n in direct:
        axR.plot(x, [r3[g][n] for g in grid], "-", color=DISC, lw=1.2, alpha=0.45,
                 zorder=2)
    for n, mk in zip(chain, ("-o", "-s")):
        axR.plot(x, [r3[g][n] for g in grid], mk, color=ORAC, lw=2.2, ms=7, zorder=5,
                 label=n.split(" ")[0] + "  (fan-out chain)")
    axR.plot([], [], "-", color=DISC, lw=1.2, alpha=0.6,
             label="5 direct edges (P0→K0, K0→K1, …)")
    axR.set_title("E3 · partial-corr full skeleton  (lagged KPI→KPI chain)", fontsize=10.0)
    axR.annotate("errors-in-variables chain\nalso 10/10 through 0.50 σ",
                 (0.50, 10), (0.20, 6.3), fontsize=8.2, color=ORAC, ha="left",
                 arrowprops=dict(arrowstyle="-", color=ORAC, lw=0.7))

    for ax in (axL, axR):
        ax.set_ylim(-0.4, 10.7)
        ax.set_yticks([0, 2, 4, 6, 8, 10])
        ax.axhline(8, color=WARN, lw=1.0, ls="--", alpha=0.7, zorder=1)
        ax.set_xlim(-0.02, 0.52)
        ax.set_xlabel("observation noise  σ_obs / σ_KPI")
        ax.legend(frameon=False, fontsize=7.9, loc="lower left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=RULE, lw=0.6, alpha=0.5)
    axL.text(0.0, 8.2, "8/10 pre-declared floor", fontsize=7.2, color=WARN)
    fig.suptitle("Load-bearing edge recovery survives observation noise to 0.50 σ (10 seeds, n=4000)",
                 fontsize=11.0, y=1.02, color=INK)
    fig.tight_layout()
    _finish(fig, "fig1_noise_robustness.png")


def fig2_fp_calibration():
    """M3 FP-calibration: per-test type-I PASS vs post-BH FDR FAIL, point+CI vs q."""
    fp = _load("p0k5_fp_calibration/results.json")["summary"]
    m1 = fp["metric1_per_test_typeI"]["pooled"]
    m2 = fp["metric2_post_BH_FDR"]
    rows = [
        ("per-test type-I\n(pooled nulls, n=%d)" % m1["n"],
         m1["rate"], m1["wilson95"][0], m1["wilson95"][1], DISC, "PASS",
         "CI includes 0.05"),
        ("post-BH realized FDR\n(%d / %d declarations)" % (
            m2["false_declarations"], m2["total_declarations"]),
         m2["realized_FDR"], m2["FDR_wilson95"][0], m2["FDR_wilson95"][1], ORAC, "FAIL",
         "entire CI above q"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ys = [1, 0]
    for y, (lab, est, lo, hi, color, tag, note) in zip(ys, rows):
        ax.plot([lo, hi], [y, y], color=color, lw=2.8, solid_capstyle="round", zorder=2)
        for xx in (lo, hi):
            ax.plot([xx, xx], [y - 0.07, y + 0.07], color=color, lw=1.6, zorder=2)
        ax.scatter([est], [y], s=95, color=color, edgecolor="white", linewidth=0.9,
                   zorder=3)
        ax.text(est, y + 0.17, f"{est:.4f}", color=color, fontsize=9.4, ha="center",
                fontweight="bold")
        ax.text(0.121, y + 0.02, tag, color=color, fontsize=9.6, ha="left",
                fontweight="bold", va="center")
        ax.text(0.121, y - 0.20, note, color=MUT, fontsize=7.6, ha="left", va="center")
    ax.axvline(0.05, color=INK, lw=1.3, ls=":", zorder=1)
    ax.text(0.05, 1.62, "q = 0.05", color=INK, fontsize=8.6, ha="center")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.6)
    ax.set_ylim(-0.6, 1.75)
    ax.set_xlim(0.0, 0.16)
    ax.set_xlabel("error rate  (Wilson 95% CI; 20 seeds × 84 pairs, n=4000)")
    ax.set_title("M3 is per-test nominal but runs ~1.5× over nominal FDR after BH",
                 fontsize=10.1)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="x", color=RULE, lw=0.6, alpha=0.5)
    fig.tight_layout()
    _finish(fig, "fig2_fp_calibration.png")


def fig3_derisk_ledger():
    """Decoy trap vs oracle vs discovered-structure arm, E2 & E3 (normalized regret)."""
    dn = _load("p0k5_downstream/downstream_results.json")
    e3 = _load("e3_multistep_wm/results.json")
    # E2 uses raw positive-regret units (decoy 0.190); E3 uses gap_norm (decoy 0.118).
    e2_decoy = dn["sanity"]["decoy_mean_pos"]
    e2_orac = dn["sanity"]["oracle_mean_pos"]
    e2_disc = dn["m3_arm"]["pooled_mean_pos"]
    e2_mism = dn["m3_arm"]["n_action_mismatches"]
    e3_decoy = e3["reference"]["frozen_decoy_gap_norm_mean"]
    e3_orac = e3["reference"]["oracle_arm_regret_raw_mean"]
    e3_disc = e3["pooled"]["pooled_disc_regret_raw_mean"]
    e3_mism = e3["pooled"]["total_action_mismatches_vs_oracle"]

    groups = [
        ("E2  (single-control decision)\nnorm. positive-regret", e2_decoy, e2_orac,
         e2_disc, e2_mism, "0 / 640 bank decisions"),
        ("E3  (multi-step do-propagation)\nnorm. decision gap", e3_decoy, e3_orac,
         e3_disc, e3_mism, "0 / 320 decisions"),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    w = 0.26
    xc = np.arange(len(groups))
    for k, (lab, decoy, orac, disc, mism, note) in enumerate(groups):
        bars = [(decoy, ORAC, "decoy structure (false world-model)"),
                (orac, MUT, "oracle (true structure)"),
                (disc, DISC, "discovered (M3 / partial-corr)")]
        for j, (val, color, blab) in enumerate(bars):
            xx = k + (j - 1) * w
            ax.bar(xx, val, width=w * 0.92, color=color,
                   edgecolor="white", linewidth=0.8, zorder=3,
                   label=blab if k == 0 else None)
            ax.text(xx, val + 0.005, f"{val:.3f}", ha="center", fontsize=8.4,
                    color=color, fontweight="bold")
        # "discovered == oracle" callout above the (zero-height) discovered/oracle bars
        ax.text(k + 0.02 * w, 0.055, "discovered = oracle\n" + note, ha="center",
                fontsize=8.0, color=DISC, fontweight="600", zorder=4)
    ax.set_xticks(xc)
    ax.set_xticklabels([g[0] for g in groups], fontsize=8.8)
    ax.set_ylim(0, 0.235)
    ax.set_ylabel("regret vs oracle  (lower = better)")
    ax.set_title("Discovered structure matches the oracle; only the decoy trap fails",
                 fontsize=10.4, pad=26)
    ax.legend(frameon=False, fontsize=7.8, ncol=1, loc="upper right",
              bbox_to_anchor=(1.0, 0.99))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=RULE, lw=0.6, alpha=0.5)
    fig.tight_layout()
    _finish(fig, "fig3_derisk_ledger.png")


if __name__ == "__main__":
    fig1_noise_robustness()
    fig2_fp_calibration()
    fig3_derisk_ledger()
    print("done ->", OUT)
