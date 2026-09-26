"""Generate the four figures for the 2026-09-02 E1 recovery-control-pipeline report.

Runs the real e1slice pipeline (generate -> split -> train oracle+dense -> predict) and
renders figures from the actual held-out predictions. Prints the exact metrics used in
the report text so the HTML can cite them verbatim.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from cdd_oran.e1slice.dataset import E1DatasetConfig, dataset_hash, generate_rows, scm_hash
from cdd_oran.e1slice.evaluate import predict, regression_metrics
from cdd_oran.e1slice.model import ModelConfig, arm_mask, build_model, train_arm
from cdd_oran.e1slice.split import SplitConfig, make_split, row_indices_for

OUT = Path("reports/2026-09-02-e1-recovery-control-pipeline")
ORAC, DENSE, INK, MUT, RULE, BAND = "#A8412F", "#0E7A73", "#1a1d22", "#79818c", "#d8dce2", "#f5f6f8"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "axes.edgecolor": "#c8ccd2", "axes.linewidth": 0.8, "axes.titlesize": 10,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
    "figure.facecolor": "white", "savefig.facecolor": "white", "axes.grid": False,
})

# --- run the real pipeline ---------------------------------------------------
cfg = E1DatasetConfig(n_episodes=48, steps_per_episode=16, warmup=2, env_seed=0)
rows = generate_rows(cfg)
split = make_split(rows.episode.tolist(), SplitConfig(test_fraction=0.25, split_seed=0))
mcfg = ModelConfig(hidden=(16,), lr=1e-2, epochs=300, batch_size=64, weight_seed=0)

models, metas = {}, {}
for arm in ("oracle", "dense"):
    m, meta = train_arm(rows, split["train"], arm, mcfg)
    m.eval()
    models[arm], metas[arm] = m, meta

test_idx = row_indices_for(rows, split["test"])
y = rows.y_kpis[test_idx]
preds = {a: predict(models[a], rows, split["test"]) for a in models}
mets = {a: regression_metrics(preds[a], y) for a in preds}

summary = {
    "n_rows": rows.n, "n_train_ep": len(split["train"]), "n_test_ep": len(split["test"]),
    "n_test_rows": int(test_idx.shape[0]),
    "params": {a: metas[a]["capacity"]["num_parameters"] for a in metas},
    "train_mse": {a: metas[a]["final_train_mse"] for a in metas},
    "test_mse": {a: mets[a]["mse"] for a in mets},
    "test_mae": {a: mets[a]["mae"] for a in mets},
    "per_kpi_mse": {a: mets[a]["per_kpi_mse"] for a in mets},
    "dataset_hash": dataset_hash(rows)[:12], "scm_hash": scm_hash(cfg)[:12],
}
print(json.dumps(summary, indent=2))

# ============================ Figure 1: pipeline =============================
fig, ax = plt.subplots(figsize=(7.2, 2.55))
ax.set_xlim(0, 10); ax.set_ylim(0, 3.4); ax.axis("off")
stages = [
    ("generate", "rows.npz\nmanifest.json", "768 rows / 48 ep"),
    ("split", "split.json", "episode-level\n36 / 12 ep"),
    ("train", "arms/*/model.pt", "oracle + dense\nidentical rows+ids"),
    ("eval", "metrics.json", "held-out 192 rows"),
    ("verify", "PASS", "reload ≤ 1e-6"),
]
w, gap = 1.62, 0.28
x0 = 0.15
for i, (cmd, art, desc) in enumerate(stages):
    x = x0 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, 1.5), w, 1.35, boxstyle="round,pad=0.02,rounding_size=0.08",
                                 fc="white", ec="#c8ccd2", lw=1.0))
    ax.plot([x, x], [1.5, 2.85], color=ORAC, lw=2.4, solid_capstyle="round")
    ax.text(x + w / 2 + 0.03, 2.55, cmd, ha="center", va="center", fontsize=10,
            fontweight="bold", color=ORAC, family="monospace")
    ax.text(x + w / 2 + 0.03, 2.02, art, ha="center", va="center", fontsize=6.6,
            color=INK, family="monospace")
    ax.text(x + w / 2 + 0.03, 1.02, desc, ha="center", va="top", fontsize=6.8, color=MUT)
    if i < len(stages) - 1:
        ax.add_patch(FancyArrowPatch((x + w, 2.17), (x + w + gap, 2.17),
                     arrowstyle="-|>", mutation_scale=11, color="#9aa2ad", lw=1.1))
# provenance band
ax.add_patch(FancyBboxPatch((x0, 0.12), 5 * w + 4 * gap, 0.5,
             boxstyle="round,pad=0.02,rounding_size=0.06", fc=BAND, ec="#dde1e6", lw=0.8))
ax.text(x0 + (5 * w + 4 * gap) / 2, 0.37,
        "provenance carried through: dataset_hash · scm_hash · split_hash · "
        "metrics_hash · git SHA · schema_version",
        ha="center", va="center", fontsize=6.4, color=MUT, family="monospace")
fig.tight_layout(pad=0.4)
fig.savefig(OUT / "fig1_pipeline.png", bbox_inches="tight")
plt.close(fig)

# ===================== Figure 2: oracle vs dense input mask ==================
labels_in = ["P0", "P1", "P2", "P3", "K0", "K1", "K2", "K3"]
labels_out = ["K0", "K1", "K2", "K3"]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
for ax, arm, col in zip(axes, ("oracle", "dense"), (ORAC, DENSE), strict=True):
    mask = arm_mask(arm).numpy()
    cmap = matplotlib.colors.ListedColormap(["#ffffff", col])
    ax.imshow(mask, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(8)); ax.set_xticklabels(labels_in, fontsize=7.5)
    ax.set_yticks(range(4)); ax.set_yticklabels(labels_out, fontsize=7.5)
    ax.set_xticks(np.arange(-.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
    ax.grid(which="minor", color="#d8dce2", lw=0.7)
    ax.tick_params(which="both", length=0)
    n_active = int(mask.sum())
    ax.set_title(f"{arm}  ({n_active} active inputs)", color=col, fontsize=9.5, fontweight="bold")
    ax.set_xlabel("input feature  [params | KPIs]", fontsize=7.5)
axes[0].set_ylabel("predicted KPI", fontsize=7.5)
fig.tight_layout(pad=0.6)
fig.savefig(OUT / "fig2_mask.png", bbox_inches="tight")
plt.close(fig)

# ===================== Figure 3: held-out parity ============================
fig, ax = plt.subplots(figsize=(4.0, 4.0))
lo = float(min(y.min(), min(preds[a].min() for a in preds)))
hi = float(max(y.max(), max(preds[a].max() for a in preds)))
pad = 0.04 * (hi - lo)
ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUT, lw=1.0, ls="--", zorder=1)
for arm, col in (("dense", DENSE), ("oracle", ORAC)):
    ax.scatter(y.ravel(), preds[arm].ravel(), s=7, color=col, alpha=0.45,
               edgecolors="none", label=arm, zorder=2)
ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
ax.set_aspect("equal")
ax.set_xlabel("true next-step KPI  (latent)", fontsize=8.5)
ax.set_ylabel("predicted next-step KPI", fontsize=8.5)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
fig.tight_layout(pad=0.5)
fig.savefig(OUT / "fig3_parity.png", bbox_inches="tight")
plt.close(fig)

# ===================== Figure 4: per-KPI held-out MSE (log) =================
fig, ax = plt.subplots(figsize=(5.2, 3.1))
x = np.arange(4); bw = 0.38
for i, (arm, col) in enumerate((("oracle", ORAC), ("dense", DENSE))):
    vals = np.asarray(mets[arm]["per_kpi_mse"])
    ax.bar(x + (i - 0.5) * bw, vals, bw, label=arm, color=col, alpha=0.9)
ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(["K0", "K1", "K2", "K3"])
ax.set_ylabel("held-out MSE  (log scale)", fontsize=8.5)
ax.set_xlabel("predicted KPI", fontsize=8.5)
ax.axhline(1e-6, color=MUT, lw=0.8, ls=":")
ax.text(3.45, 1.15e-6, "≈ 1e-6 sanity floor", fontsize=6.8, color=MUT, ha="right")
ax.legend(frameon=False, fontsize=8.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=0.5)
fig.savefig(OUT / "fig4_per_kpi_mse.png", bbox_inches="tight")
plt.close(fig)

print("wrote 4 figures to", OUT)
