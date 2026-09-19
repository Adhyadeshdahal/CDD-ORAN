"""Figures for the 2026-09-19 Thread B (world-model off-gate generalization) result report.

Reads the 1000 raw per-cell records in scratchpad/design_adaptive_m3/gated_mde_B_records/ LIVE (json +
numpy + matplotlib only; no torch, no freeze_config). Recomputes the PRE-REGISTERED B success logic
(PREDECLARE_B_worldmodel_offgate_v3 + addendum A1-A10) independently from the raw records -- forced
mechanism event M_r, discovered end-to-end event N_r, cell booleans MECH_cell / EE_cell with the paired
seed-level bootstrap -- and emits fig1..fig4 (PNG) + figures_data.json.

Constants (frozen, from the pre-registration): EQ=TOL=IN_ABS=0.02, IN_REL=0.50, NI=0.01, FRAC=0.80;
strong deltas {0.8,1.6,3.2}; paired bootstrap n_boot=10000, seed 800001. Tracks: FORCED (K5 parents
{0,6,7}, CL64 action vs own slope-0 twin) and DISCOVERED (M3-declared, native-vs-omit + matched-vs-null).
Deploy = occ_B[q]*cond + (1-occ_B[q])*off. q=1.0 = REFERENCE row (off NOT_APPLICABLE; never a success cell).
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
                   "scratchpad", "design_adaptive_m3", "gated_mde_B_records")
OUT = os.path.dirname(os.path.abspath(__file__))
QS = [0.02, 0.05, 0.10, 0.20, 1.0]
DS = [0.0, 0.4, 0.8, 1.6, 3.2]
STRONG = [0.8, 1.6, 3.2]
EQ = TOL = IN_ABS = 0.02
IN_REL, NI, FRAC = 0.50, 0.01, 0.80
N_BOOT, BOOT_SEED = 10000, 800001

INK = "#23201b"; SOFT = "#6b6459"; GOOD = "#1f8a5b"; BAD = "#c0492f"; ACC = "#3a5a8c"; AMB = "#b7791f"
VIO = "#6d4a9c"; TEAL = "#0d7d74"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#c9c2b5",
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": SOFT, "ytick.color": SOFT,
                     "axes.titlesize": 12.5, "figure.facecolor": "white", "axes.facecolor": "white"})


def _num(x):
    return None if (x is None or x == "NOT_APPLICABLE") else float(x)


def load():
    cells = {}
    for f in glob.glob(os.path.join(REC, "cell_*.json")):
        r = json.load(open(f)); cells.setdefault((r["q"], r["delta"]), []).append(r)
    for k in cells:
        cells[k].sort(key=lambda r: r["seed"])
    return cells


def _boot(vals, side):
    """Paired seed-level percentile bootstrap; side 'U'->97.5th, 'L'->2.5th. Filters None."""
    v = np.array([x for x in vals if x is not None], float)
    if v.size == 0:
        return None
    rng = np.random.default_rng(BOOT_SEED); n = v.size
    b = np.array([v[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    return float(np.quantile(b, 0.975 if side == "U" else 0.025, method="linear"))


def forced(rec):
    act, nul = rec["G_CL64f_act"], rec["G_CL64f_null"]
    dOff = None if _num(act["off"]) is None or _num(nul["off"]) is None else _num(act["off"]) - _num(nul["off"])
    cn = _num(nul["cond"]); E = cn is not None and cn > TOL
    benA = None if cn is None else cn - _num(act["cond"])
    benR = (benA / cn) if (E and benA is not None) else None
    M = bool(E and dOff is not None and dOff <= EQ and benA is not None and benA >= IN_ABS
             and benR is not None and benR >= IN_REL)
    return dOff, benA, M


def discovered(rec):
    nat, mat, nul, omit = rec["G_act_native"], rec["G_act_matched"], rec["G_null_disc"], rec["G_omit"]
    netOmit = None if _num(omit["deploy"]) is None or _num(nat["deploy"]) is None else _num(omit["deploy"]) - _num(nat["deploy"])
    netNull = None if _num(nul["deploy"]) is None or _num(mat["deploy"]) is None else _num(nul["deploy"]) - _num(mat["deploy"])
    cn = _num(nul["cond"]); Ed = cn is not None and cn > TOL
    bA = None if cn is None else cn - _num(mat["cond"])
    bR = (bA / cn) if (Ed and bA is not None) else None
    D = bool(rec["p0_declared"])
    N = bool(D and Ed and netOmit is not None and netOmit >= -NI and netNull is not None and netNull >= -NI
             and bA is not None and bA >= IN_ABS and bR is not None and bR >= IN_REL)
    return netOmit, netNull, N


def sec_off(rec, cls):
    a = _num(rec["secondary"][f"G_{cls}_act"]["off"]); n = _num(rec["secondary"][f"G_{cls}_null"]["off"])
    return None if a is None or n is None else a - n


def aggregate(cells):
    A = {}
    for q in QS:
        for d in DS:
            rs = cells[(q, d)]
            fc = [forced(r) for r in rs]; dc = [discovered(r) for r in rs]
            dOff = [x[0] for x in fc]; benA = [x[1] for x in fc]; M = [x[2] for x in fc]
            nOmit = [x[0] for x in dc]; nNull = [x[1] for x in dc]; N = [x[2] for x in dc]
            mdo = [x for x in dOff if x is not None]; mba = [x for x in benA if x is not None]
            row = {
                "n": len(rs),
                "mean_dOff": (float(np.mean(mdo)) if mdo else np.nan),
                "U95_dOff": _boot(dOff, "U"),
                "frac_M": float(np.mean(M)),
                "mean_benA": (float(np.mean(mba)) if mba else np.nan),
                "mean_netOmit": float(np.mean([x for x in nOmit if x is not None])) if any(x is not None for x in nOmit) else np.nan,
                "L95_netOmit": _boot(nOmit, "L"),
                "mean_netNull": float(np.mean([x for x in nNull if x is not None])) if any(x is not None for x in nNull) else np.nan,
                "L95_netNull": _boot(nNull, "L"),
                "frac_N": float(np.mean(N)),
                "p0_decl_rate": float(np.mean([r["p0_declared"] for r in rs])),
                "dispatch_rate": float(np.mean([r["dispatched"] for r in rs])),
            }
            u = row["U95_dOff"]
            row["MECH_cell"] = bool(not np.isnan(row["mean_dOff"]) and row["mean_dOff"] <= EQ
                                    and u is not None and u <= EQ and row["frac_M"] >= FRAC)
            lo, ln = row["L95_netOmit"], row["L95_netNull"]
            row["EE_cell"] = bool(q < 1.0 and not np.isnan(row["mean_netOmit"]) and row["mean_netOmit"] >= -NI
                                  and lo is not None and lo >= -NI and not np.isnan(row["mean_netNull"])
                                  and row["mean_netNull"] >= -NI and ln is not None and ln >= -NI
                                  and row["frac_N"] >= FRAC)
            for cls in ("CL64f", "CL256", "GOLS", "ANISO"):
                if cls == "CL64f":
                    row[f"off_{cls}"] = row["mean_dOff"]
                else:
                    vs = [sec_off(r, cls) for r in rs]
                    row[f"off_{cls}"] = float(np.mean([v for v in vs if v is not None])) if any(v is not None for v in vs) else np.nan
            A[(q, d)] = row
    return A


def qlabel(q):
    return "100%" if q >= 1.0 else f"{int(q*100)}%"


# ---------- fig1: mechanism headline heatmap (off-gate collateral vs the 0.02 bar) ----------
def fig1(A):
    M = np.array([[A[(q, d)]["mean_dOff"] for d in DS] for q in QS])
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    Mm = np.ma.masked_invalid(M)
    im = ax.imshow(Mm, cmap="RdYlGn_r", vmin=0, vmax=0.05, aspect="auto")
    im.cmap.set_bad("#e8e5dd")
    ax.set_xticks(range(len(DS))); ax.set_xticklabels(DS)
    ax.set_yticks(range(len(QS))); ax.set_yticklabels([qlabel(q) for q in QS])
    ax.set_xlabel("harm strength  δ  (delta)"); ax.set_ylabel("occupancy  (how often harm is active)")
    for i, q in enumerate(QS):
        for j, d in enumerate(DS):
            v = M[i, j]
            s = "n/a" if np.isnan(v) else f"{v:.3f}"
            ax.text(j, i, s, ha="center", va="center", fontsize=8.6,
                    color="white" if (not np.isnan(v) and v > 0.032) else INK)
            if A[(q, d)]["MECH_cell"]:
                ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False, ec=ACC, lw=2.6))
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("incremental off-gate collateral  dOff  (bar = 0.02)", fontsize=9)
    ax.set_title("Fig 1 — CL64 off-gate collateral vs trap strength (blue box = passes the 0.02 bar)",
                 loc="left", fontweight="bold")
    fig.text(0.012, -0.03, "Collateral stays under 0.02 only at moderate δ; it exceeds the bar at every "
             "δ≥1.6. No occupancy passes the headline (all three strong δ required). q=100% off-gate = n/a.",
             fontsize=9.5, color=SOFT)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_mech_heatmap.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------- fig2: the two halves of the partial win ----------
def fig2(A):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 3.8))
    cols = {0.02: GOOD, 0.05: TEAL, 0.10: ACC, 0.20: VIO}
    for q in [0.02, 0.05, 0.10, 0.20]:
        axL.plot(STRONG_X(), [A[(q, d)]["mean_dOff"] for d in [0.4, 0.8, 1.6, 3.2]],
                 "-o", color=cols[q], ms=4, lw=1.9, label=f"q={qlabel(q)}")
        axR.plot(STRONG_X(), [A[(q, d)]["mean_benA"] for d in [0.4, 0.8, 1.6, 3.2]],
                 "-o", color=cols[q], ms=4, lw=1.9, label=f"q={qlabel(q)}")
    axL.axhline(EQ, color=BAD, ls="--", lw=1.5); axL.text(3.2, EQ+0.001, "safety bar 0.02", color=BAD, fontsize=8.5, ha="right")
    axL.set_title("OFF-GATE collateral (lower = safer)", loc="left", fontsize=11)
    axR.set_title("IN-GATE benefit (higher = better)", loc="left", fontsize=11)
    for ax in (axL, axR):
        ax.set_xlabel("harm strength  δ"); ax.set_xticks([0.4, 0.8, 1.6, 3.2]); ax.grid(alpha=0.25)
    axL.set_ylabel("incremental collateral  dOff"); axR.set_ylabel("in-gate benefit  benA")
    axL.legend(fontsize=8.5, framealpha=0.9)
    fig.suptitle("Fig 2 — Why CL64 is a genuine PARTIAL win: real in-gate benefit, but off-gate collateral crosses the bar",
                 fontweight="bold", x=0.012, ha="left", y=1.03, fontsize=12)
    fig.text(0.012, -0.04, "Right: CL64 delivers large, growing in-gate decision benefit (it captures the "
             "action effect). Left: but its off-gate collateral rises and saturates ABOVE the 0.02 bar by δ=1.6 "
             "-- so the mechanism headline fails at every occupancy.", fontsize=9.5, color=SOFT)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_partial_win.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def STRONG_X():
    return [0.4, 0.8, 1.6, 3.2]


# ---------- fig3: end-to-end occupancy dependence ----------
def fig3(A):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 3.8), gridspec_kw={"width_ratios": [1.35, 1]})
    cols = {0.02: GOOD, 0.05: TEAL, 0.10: ACC, 0.20: VIO}
    for q in [0.02, 0.05, 0.10, 0.20]:
        axL.plot([0.4, 0.8, 1.6, 3.2], [A[(q, d)]["mean_netOmit"] for d in [0.4, 0.8, 1.6, 3.2]],
                 "-o", color=cols[q], ms=4, lw=1.9, label=f"q={qlabel(q)}")
    axL.axhline(-NI, color=AMB, ls="--", lw=1.5); axL.text(3.2, -NI-0.003, "non-inferiority line -0.01", color=AMB, fontsize=8.5, ha="right")
    axL.axhline(0, color="#cfc9bd", lw=1)
    axL.set_xlabel("harm strength  δ"); axL.set_ylabel("net vs incomplete (omit) baseline  [deploy]")
    axL.set_xticks([0.4, 0.8, 1.6, 3.2]); axL.grid(alpha=0.25); axL.legend(fontsize=8.5, framealpha=0.9)
    axL.set_title("FINITE occupancy: net benefit turns negative as δ grows", loc="left", fontsize=11)
    # right: mean netOmit over strong deltas across ALL q incl 100% -> occupancy monotone jump
    qs = [0.02, 0.05, 0.10, 0.20, 1.0]
    vals = [float(np.mean([A[(q, d)]["mean_netOmit"] for d in STRONG])) for q in qs]
    barcol = [BAD if v < -NI else (AMB if v < 0 else GOOD) for v in vals]
    axR.bar([qlabel(q) for q in qs], vals, color=barcol, width=0.62)
    axR.axhline(-NI, color=AMB, ls="--", lw=1.3); axR.axhline(0, color="#cfc9bd", lw=1)
    for i, v in enumerate(vals):
        axR.text(i, v + (0.006 if v >= 0 else -0.012), f"{v:+.3f}", ha="center", fontsize=8.4, color=INK)
    axR.set_ylabel("mean net vs omit  (strong δ)"); axR.set_xlabel("occupancy")
    axR.set_title("Net benefit vs occupancy", loc="left", fontsize=11); axR.grid(alpha=0.2, axis="y")
    fig.suptitle("Fig 3 — End-to-end: the discovered complete arm helps only at high occupancy",
                 fontweight="bold", x=0.012, ha="left", y=1.03, fontsize=12)
    fig.text(0.012, -0.04, "Left: at finite occupancy, net benefit vs the incomplete baseline falls below the "
             "-0.01 line as the trap strengthens. Right: averaged over strong δ, the arm is strongly positive "
             "at 100% occupancy (+0.17) and monotonically worse as the danger zone gets rarer.", fontsize=9.5, color=SOFT)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_endtoend_occupancy.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------- fig4: secondary class comparison ----------
def fig4(A):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=True)
    classes = [("CL64f", "CL64", ACC), ("CL256", "CL256", VIO), ("GOLS", "GOLS", BAD), ("ANISO", "ANISO", GOOD)]
    for ax, q in zip(axes, [0.10, 0.20]):
        for key, lab, c in classes:
            ax.plot([0.4, 0.8, 1.6, 3.2], [min(A[(q, d)][f"off_{key}"], 0.09) for d in [0.4, 0.8, 1.6, 3.2]],
                    "-o", color=c, ms=4, lw=1.8, label=lab)
        ax.axhline(EQ, color="#999", ls="--", lw=1.3)
        ax.set_title(f"occupancy {qlabel(q)}", loc="left", fontsize=11)
        ax.set_xlabel("harm strength  δ"); ax.set_xticks([0.4, 0.8, 1.6, 3.2]); ax.grid(alpha=0.25)
    axes[0].set_ylabel("off-gate collateral  (clipped at 0.09)"); axes[0].legend(fontsize=8.5, framealpha=0.9)
    axes[0].text(0.42, 0.083, "GOLS off-scale (0.18–0.58)", color=BAD, fontsize=8)
    fig.suptitle("Fig 4 — Which world-model class leaks least off-gate? ANISO leads (a lead for a C-bank test)",
                 fontweight="bold", x=0.012, ha="left", y=1.03, fontsize=12)
    fig.text(0.012, -0.04, "ANISO (down-weight P0 in the similarity metric) has the lowest collateral of all "
             "classes, though it still breaches 0.02 at the strongest δ. CL256 over-fits; GOLS (global OLS) is "
             "unstable and runs off-scale. None is a confirmatory winner on this bank.", fontsize=9.5, color=SOFT)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_secondary_classes.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    cells = load()
    total = sum(len(v) for v in cells.values())
    assert total == 1000, f"expected 1000 records, got {total}"
    A = aggregate(cells)
    fig1(A); fig2(A); fig3(A); fig4(A)
    rows = []
    for q in QS:
        for d in DS:
            r = A[(q, d)]
            rows.append(dict(q=q, delta=d, mean_dOff=_r(r["mean_dOff"]), U95_dOff=_r(r["U95_dOff"]),
                             frac_M=_r(r["frac_M"]), MECH_cell=r["MECH_cell"], mean_benA=_r(r["mean_benA"]),
                             mean_netOmit=_r(r["mean_netOmit"]), L95_netOmit=_r(r["L95_netOmit"]),
                             mean_netNull=_r(r["mean_netNull"]), frac_N=_r(r["frac_N"]), EE_cell=r["EE_cell"],
                             p0_decl_rate=_r(r["p0_decl_rate"]), off_ANISO=_r(r["off_ANISO"]),
                             off_CL256=_r(r["off_CL256"]), off_GOLS=_r(r["off_GOLS"])))
    # headline booleans
    head = {}
    for q in QS:
        head[qlabel(q)] = dict(
            MECH_headline=(bool(all(A[(q, d)]["MECH_cell"] for d in STRONG)) if q < 1.0 else "REFERENCE"),
            EE_headline=(bool(all(A[(q, d)]["EE_cell"] for d in STRONG)) if q < 1.0 else "REFERENCE"))
    json.dump({"n_records": total, "headline": head, "per_cell": rows},
              open(os.path.join(OUT, "figures_data.json"), "w"), indent=1)
    print("wrote fig1..fig4 + figures_data.json to", OUT)
    print("headline:", json.dumps(head))


def _r(x):
    return None if (x is None or (isinstance(x, float) and np.isnan(x))) else round(float(x), 4)


if __name__ == "__main__":
    main()
