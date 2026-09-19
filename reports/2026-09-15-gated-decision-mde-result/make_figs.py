"""Figures for the 2026-09-15 gated decision-MDE result report.

Reads the 1600 raw per-cell records in scratchpad/design_adaptive_m3/gated_mde_records/ LIVE (no torch,
no freeze_config -- just json + numpy + matplotlib; pre-registered constants hardcoded from PREDECLARE v3
addendum). Aggregates per (q, delta) over the 40 discovery seeds and emits fig1..fig4 (PNG).

Estimands (matched-arm, all-deployment normalized regret over the D_E2_ID bank; see PREDECLARE_phase4b_
gated_mde_v3*.md): learned_omit (incomplete pipeline, P0 forced out of K5), learned_discovered (complete
fan-out), learned_oracle (true structure {P0,P6,P7} ceiling), exact_decoy (synthetic base_p0free, NO learner
-> intrinsic planted-trap severity). Prevention success per seed = G_omit^deploy > TOL_HARM AND
(G_omit-G_disc)^deploy >= RHO_MIN * G_omit^deploy. Decision power = #success/40. Edge power = P0->K5
declared fraction.
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "scratchpad", "design_adaptive_m3", "gated_mde_records")
OUT = os.path.dirname(os.path.abspath(__file__))
QS = [0.02, 0.05, 0.10, 0.20, 1.0]
DS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
TOL_HARM, RHO_MIN, MAX_OFFGATE = 0.02, 0.80, 0.05

INK = "#23201b"; SOFT = "#6b6459"; GOOD = "#1f8a5b"; BAD = "#c0492f"; ACC = "#3a5a8c"; AMB = "#b7791f"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#c9c2b5",
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": SOFT, "ytick.color": SOFT,
                     "axes.titlesize": 12.5, "figure.facecolor": "white", "axes.facecolor": "white"})


def load():
    cells = {}
    for f in glob.glob(os.path.join(REC, "cell_*.json")):
        r = json.load(open(f)); cells.setdefault((r["q"], r["delta"]), []).append(r)
    return cells


def _mean(vs):
    vs = [v for v in vs if v is not None]
    return float(np.mean(vs)) if vs else np.nan


def aggregate(cells):
    A = {}
    for q in QS:
        for d in DS:
            recs = cells[(q, d)]
            row = {"n": len(recs)}
            for arm in ("G_omit", "G_disc", "G_oracle", "G_exact_decoy"):
                for lvl in ("deploy", "cond", "off"):
                    row[f"{arm}_{lvl}"] = _mean([r[arm][lvl] for r in recs])
            row["edge_power"] = _mean([1.0 if r["p0_declared"] else 0.0 for r in recs])
            succ = [(r["G_omit"]["deploy"] > TOL_HARM
                     and (r["G_omit"]["deploy"] - r["G_disc"]["deploy"]) >= RHO_MIN * r["G_omit"]["deploy"])
                    for r in recs]
            row["decision_power"] = float(np.mean(succ))
            row["decoy_worse"] = _mean([r["decoy_worse_frac_ingate"] for r in recs])
            A[(q, d)] = row
    return A


def _grid(A, key):
    return np.array([[A[(q, d)][key] for d in DS] for q in QS])


def qlabel(q):
    return "100%" if q >= 1.0 else f"{int(q*100)}%"


# ---------- fig1: decision power (the headline) ----------
def fig1(A):
    M = _grid(A, "decision_power")
    fig, ax = plt.subplots(figsize=(8.6, 3.5))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(DS))); ax.set_xticklabels(DS)
    ax.set_yticks(range(len(QS))); ax.set_yticklabels([qlabel(q) for q in QS])
    ax.set_xlabel("harm strength  δ  (delta)"); ax.set_ylabel("occupancy  (how often harm is active)")
    for i in range(len(QS)):
        for j in range(len(DS)):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if (v < 0.25 or v > 0.85) else INK, fontsize=9)
    # mark the only crossing >=0.80
    for i, q in enumerate(QS):
        for j, d in enumerate(DS):
            if M[i, j] >= 0.80:
                ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False, ec=ACC, lw=2.4))
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label("prevention power (fraction of 40 seeds)", fontsize=9)
    ax.set_title("Fig 1 — Does complete fan-out reliably PREVENT the harm? (blue box = ≥80% power)",
                 loc="left", fontweight="bold")
    fig.text(0.012, -0.02, "Decision-MDE exists ONLY at 100% occupancy (δ≥0.4, power 0.925). "
             "Null at every realistic occupancy (2–20%): prevention never reaches 80%.",
             fontsize=9.5, color=SOFT)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_decision_power.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------- fig2: edge power (discovery works everywhere) ----------
def fig2(A):
    M = _grid(A, "edge_power")
    fig, ax = plt.subplots(figsize=(8.6, 3.5))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(DS))); ax.set_xticklabels(DS)
    ax.set_yticks(range(len(QS))); ax.set_yticklabels([qlabel(q) for q in QS])
    ax.set_xlabel("harm strength  δ"); ax.set_ylabel("occupancy")
    for i in range(len(QS)):
        for j in range(len(DS)):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.6 else INK, fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label("edge power: P0→K5 discovered (frac)", fontsize=9)
    ax.set_title("Fig 2 — Does discovery FIND the hidden harmful link? (Edge power)",
                 loc="left", fontweight="bold")
    fig.text(0.012, -0.02, "Discovery reaches near-certainty at every occupancy as δ grows — "
             "finding the danger is NOT the bottleneck (contrast Fig 1).", fontsize=9.5, color=SOFT)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_edge_power.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------- fig3: mechanism at q=20% (in-gate helps, off-gate hurts; baseline off-gate error) ----------
def fig3(A):
    q = 0.20
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=True)
    styles = [("G_omit", "incomplete (omit P0)", SOFT, "o", "-"),
              ("G_disc", "complete (discovered)", BAD, "s", "-"),
              ("G_oracle", "true structure (oracle)", ACC, "^", "--")]
    for arm, lab, c, mk, ls in styles:
        axL.plot(DS, [A[(q, d)][f"{arm}_cond"] for d in DS], mk+ls, color=c, label=lab, ms=5, lw=1.8)
        axR.plot(DS, [A[(q, d)][f"{arm}_off"] for d in DS], mk+ls, color=c, label=lab, ms=5, lw=1.8)
    axL.set_title("IN-GATE (harm active)  —  complete fan-out helps", loc="left", fontsize=11)
    axR.set_title("OFF-GATE (harm inactive)  —  complete fan-out hurts", loc="left", fontsize=11)
    for ax in (axL, axR):
        ax.set_xlabel("harm strength  δ"); ax.grid(alpha=0.25)
    axL.set_ylabel("mean normalized regret  (lower = better)")
    axR.axhline(MAX_OFFGATE, color=AMB, ls=":", lw=1.4)
    axR.text(3.25, MAX_OFFGATE+0.006, "off-gate control limit 0.05", color=AMB, fontsize=8.5, ha="right")
    axL.legend(fontsize=8.7, loc="upper right", framealpha=0.9)
    fig.suptitle("Fig 3 — Why it fails at 20% occupancy: the learned world-model's off-gate cost",
                 fontweight="bold", x=0.012, ha="left", y=1.02, fontsize=12.5)
    fig.text(0.012, -0.04,
             "In-gate: complete fan-out (red) cuts regret sharply as δ grows. Off-gate: it RISES with δ "
             "(P0 collateral). Crucially, off-gate regret is already ~0.13 at δ=0 for ALL arms incl. omit "
             "— a large BASELINE learned-model error, far above the 0.05 limit, independent of P0.",
             fontsize=9.5, color=SOFT)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_mechanism_q20.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------- fig4: severity honesty + oracle also fails ----------
def fig4(A):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 3.8))
    # A: intrinsic (exact_decoy) severity vs the learner-inflated barG_omit, at q=20% and q=100%
    for q, c in [(0.20, ACC), (1.0, AMB)]:
        axA.plot(DS, [A[(q, d)]["G_omit_deploy"] for d in DS], "-o", color=c, ms=4, lw=1.8,
                 label=f"barG_omit (learner)  q={qlabel(q)}")
        axA.plot(DS, [A[(q, d)]["G_exact_decoy_deploy"] for d in DS], "--s", color=c, ms=4, lw=1.4, alpha=0.75,
                 label=f"intrinsic trap (no learner)  q={qlabel(q)}")
    axA.set_xlabel("harm strength  δ"); axA.set_ylabel("mean all-deployment regret")
    axA.set_title("Severity: learner-inflated vs intrinsic", loc="left", fontsize=11)
    axA.legend(fontsize=8, loc="upper left", framealpha=0.9); axA.grid(alpha=0.25)
    # B: deploy regret omit vs disc vs oracle at q=20% -> oracle doesn't rescue it
    q = 0.20
    for arm, lab, c, mk in [("G_omit", "incomplete", SOFT, "o"), ("G_disc", "complete", BAD, "s"),
                            ("G_oracle", "true structure", ACC, "^")]:
        axB.plot(DS, [A[(q, d)][f"{arm}_deploy"] for d in DS], "-"+mk, color=c, ms=5, lw=1.8, label=lab)
    axB.set_xlabel("harm strength  δ"); axB.set_ylabel("mean all-deployment regret  (q=20%)")
    axB.set_title("Even TRUE structure doesn't prevent it", loc="left", fontsize=11)
    axB.legend(fontsize=8.7, framealpha=0.9); axB.grid(alpha=0.25)
    fig.suptitle("Fig 4 — The severity is mostly learner noise, and the ceiling arm fails too",
                 fontweight="bold", x=0.012, ha="left", y=1.02, fontsize=12.5)
    fig.text(0.012, -0.04,
             "Left: the plotted 'severity' barG_omit (~0.13) is dominated by baseline learner regret; the "
             "intrinsic planted trap (exact-decoy, no learner) is ≤0.029. Right: the oracle arm handed the "
             "TRUE map tracks the others at 20% occupancy — so the bottleneck is the learned world-model, "
             "not discovery.", fontsize=9.5, color=SOFT)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_severity_oracle.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    cells = load()
    assert sum(len(v) for v in cells.values()) == 1600, "expected 1600 records"
    A = aggregate(cells)
    fig1(A); fig2(A); fig3(A); fig4(A)
    # emit a compact numeric appendix the report cites
    lines = []
    for q in QS:
        for d in DS:
            r = A[(q, d)]
            lines.append(dict(q=q, delta=d, edge_power=round(r["edge_power"], 3),
                              decision_power=round(r["decision_power"], 3),
                              barG_omit=round(r["G_omit_deploy"], 4),
                              exact_decoy_deploy=round(r["G_exact_decoy_deploy"], 4),
                              G_disc_cond=round(r["G_disc_cond"], 4), G_omit_cond=round(r["G_omit_cond"], 4),
                              G_disc_off=round(r["G_disc_off"], 4), G_omit_off=round(r["G_omit_off"], 4),
                              G_oracle_off=round(r["G_oracle_off"], 4), decoy_worse=round(r["decoy_worse"], 3)))
    json.dump(lines, open(os.path.join(OUT, "figures_data.json"), "w"), indent=1)
    print("wrote fig1..fig4 + figures_data.json to", OUT)


if __name__ == "__main__":
    main()
