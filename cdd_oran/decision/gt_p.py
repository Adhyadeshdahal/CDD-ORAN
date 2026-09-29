"""E6-P discovery knockout ground truth (scratchpad/e6_dev/decision/STEP1_MSCR_PLAN.md, "Ground truth").

Source: CRN knockout labels on the GT seeds. A GT episode runs the discovery collection (pi0 base path); a sampled
unit u = (c, x, t0) (``GT_RATE`` per knob family, draw = ``labels_p.sampled`` = tag 6613) is labelled by
``labels_p.Labeller(ks=GT_KS, H=GT_H, cells=True)`` with modes accept and reject: for each reseed k in {1, 2, 3} the
env is copied at t0 (same re-drawn tape k in both arms), u's mode is held for its T, every other unit is ACCEPTED
(AA continuation, the estimand of this GT; the discovery CRT runs under pi0 instead: declared mismatch), rolled out
H = 90 s. Per-cell outcome = scored cumulative KPI over the rollout (``labels_p.CELL_KPIS``).

Unit effect (per cell q, KPI y, reseed k):  D[k, y, q] = y(accept) - y(reject)  (= -label["cell"]["reject"]).
Orientation (``ORIENTS``):
  "act"  D as is: the effect of APPLYING the xApp's requested change (the plan's wording: "ptx -> own e (-)" is a
         PowerES DEcrease applied);
  "dir"  D * sign(prop - cur) of the unit's opening request: the effect per knob-value INCREASE (sleep: 1 = asleep;
         carrier: active carriers; ptx: dB; prot_min: share). Units whose sleep / wake, up / down requests would
         cancel in "act" line up in "dir". PRIMARY = "dir" (``PRIMARY_ORIENT``).
Relations of a unit on cell c (exposure set N(c) = ``units_p.exposure_sets``, obs-only): own = {c}; nbr = N(c) - {c};
far = every cell not in N(c) (negative control; the 2-hop closure covers the whole 24-cell layout, so 1-hop is the
only possible exclusion). Unit value y_u(f, rel, kpi) = mean over k of sum over the relation's cells of D.
Knob family f = the unit's knob (carrier | sleep | ptx | prot_min). KPIs ``GT_KPIS`` = pv (protected violated UE-s),
v (violated UE-s, all slices), e (energy J), rlf (count), load (served UE-s).

Classification per (f, rel, kpi) (``gt_table``):
  mean = pooled unit mean; 95 % CI = episode-cluster percentile bootstrap (resample the GT episodes that contribute
  units of f, with replacement; ``N_BOOT`` draws; ``default_rng([GT_TAG, f_idx, rel_idx, kpi_idx, orient_idx])``,
  GT_TAG = 6617);
  delta[kpi] = max(REL_FRAC * max over (f, rel) of |mean|, KPI_FLOOR[kpi]) (REL_FRAC = 5 %, pv floor 0.5 UE-s);
  TRUE(sign)  CI excludes 0 and |mean| >= delta[kpi]      (sign = sign(mean));
  NULL        CI inside [-delta, +delta];
  INDET       otherwise, and whenever unsupported (< GT_MIN_UNITS units or < GT_MIN_EPS episodes): excluded from
              scoring and counted.
Receiving cells (``receiving_shares``): per (seed, pico), over the pico's SLEEP units (request step > 0), the "dir"
load effect summed over the units: share of each other cell in the positive load gain, top-1 receiver vs the ES
candidate macro (``gt_static["cand"]``, privileged).

Input "episode" dicts (the driver decodes its JSONL records into these; see scratchpad/e6_dev/e6p_discovery.py):
  {"seed": int, "n_cells": int, "gt_static": {"cand": {pico: macro}, ...},
   "gt_labels": [{"c": int, "knob": str, "exp": [cells], "step": float, "kpis": [CELL_KPIS],
                  "delta": ndarray (len(ks), len(kpis), n_cells) = accept - reject}, ...]}
"""
from __future__ import annotations

import numpy as np

from .labels_p import CELL_KPIS

GT_TAG = 6617
GT_KS = (1, 2, 3)
GT_H = 90
GT_MODES = ("accept", "reject")
FAMILIES = ("carrier", "sleep", "ptx", "prot_min")
RELATIONS = ("own", "nbr", "far")
GT_KPIS = ("pv", "v", "e", "rlf", "load")
ORIENTS = ("dir", "act")
PRIMARY_ORIENT = "dir"
REL_FRAC = 0.05
KPI_FLOOR = {"pv": 0.5}
N_BOOT = 4000
CI = (0.025, 0.975)
GT_MIN_UNITS, GT_MIN_EPS = 10, 3
# per-family label sampling rate on GT episodes (cost cap; draw = labels_p.sampled, tag 6613)
GT_RATE = {"carrier": 1.0, "sleep": 1.0, "ptx": 1.0, "prot_min": 0.3}


def relation_cells(c: int, exp, n_cells: int) -> dict:
    """{"own": [c], "nbr": N(c) - {c}, "far": cells not in N(c)} (sorted lists)."""
    ex = {int(v) for v in exp}
    return {"own": [int(c)], "nbr": sorted(ex - {int(c)}), "far": sorted(set(range(int(n_cells))) - ex)}


def unit_delta(label) -> np.ndarray:
    """(len(ks), len(CELL_KPIS), cells) accept - reject from a ``Labeller(cells=True)`` label."""
    if "cell" not in label:
        raise ValueError("label has no per-cell contrasts (Labeller(cells=True))")
    return -np.asarray(label["cell"]["reject"], float)


def unit_rows(episodes, kpis=GT_KPIS) -> list:
    """One row per GT label: {"ep", "family", "c", "t0", "step", "val": {orient: {rel: {kpi: y_u}}},
    "cells": {orient: (len(kpis), n_cells) mean-over-k D}}."""
    rows = []
    for ep in episodes:
        n = int(ep["n_cells"])
        for L in ep["gt_labels"]:
            D = np.asarray(L["delta"], float)
            ki = [list(L["kpis"]).index(k) for k in kpis]
            Dm = D[:, ki, :].mean(0)                                     # (kpis, cells), mean over k
            sgn = float(np.sign(L["step"])) or 1.0
            rel = relation_cells(L["c"], L["exp"], n)
            cells = {"act": Dm, "dir": Dm * sgn}
            val = {o: {r: {k: float(cells[o][j, idx].sum()) for j, k in enumerate(kpis)} for r, idx in rel.items()}
                   for o in ORIENTS}
            rows.append({"ep": int(ep["seed"]), "family": L["knob"], "c": int(L["c"]), "t0": L.get("t0"),
                         "step": float(L["step"]), "val": val, "cells": cells})
    return rows


def cluster_boot(y, ep, rng, n_boot: int = N_BOOT) -> tuple[float, tuple[float, float], int, int]:
    """(pooled mean, percentile CI, n units, n episodes) with an episode-cluster bootstrap."""
    y, ep = np.asarray(y, float), np.asarray(ep)
    if len(y) == 0:
        return float("nan"), (float("nan"), float("nan")), 0, 0
    eps, inv = np.unique(ep, return_inverse=True)
    S = np.bincount(inv, y, len(eps))
    N = np.bincount(inv, minlength=len(eps)).astype(float)
    idx = rng.integers(len(eps), size=(n_boot, len(eps)))
    bm = S[idx].sum(1) / N[idx].sum(1)
    lo, hi = np.quantile(bm, CI)
    return float(y.mean()), (float(lo), float(hi)), len(y), len(eps)


def classify(mean, ci, delta, n, n_eps, min_units=GT_MIN_UNITS, min_eps=GT_MIN_EPS) -> dict:
    """TRUE(sign) / NULL / INDET (module docstring)."""
    if n < min_units or n_eps < min_eps or not np.isfinite(mean):
        return {"status": "INDET", "sign": 0, "why": f"unsupported (n={n}, episodes={n_eps})"}
    lo, hi = ci
    if (lo > 0 or hi < 0) and abs(mean) >= delta:
        return {"status": "TRUE", "sign": int(np.sign(mean)), "why": "CI excludes 0, |mean| >= delta"}
    if -delta <= lo and hi <= delta:
        return {"status": "NULL", "sign": 0, "why": "CI within +-delta"}
    return {"status": "INDET", "sign": 0, "why": "neither rule"}


def gt_table(rows, orient: str = PRIMARY_ORIENT, kpis=GT_KPIS, families=FAMILIES, relations=RELATIONS,
             n_boot: int = N_BOOT, rel_frac: float = REL_FRAC, floors=None, min_units=GT_MIN_UNITS,
             min_eps=GT_MIN_EPS) -> dict:
    """{"orient", "delta": {kpi: d}, "cells": [{family, relation, kpi, mean, ci, n, n_eps, status, sign, why}],
    "counts": {status: n}} over families x relations x kpis."""
    floors = KPI_FLOOR if floors is None else floors
    oi = ORIENTS.index(orient)
    stats = {}
    for fi, f in enumerate(families):
        fr = [r for r in rows if r["family"] == f]
        ep = [r["ep"] for r in fr]
        for ri, rel in enumerate(relations):
            for ki, k in enumerate(kpis):
                y = [r["val"][orient][rel][k] for r in fr]
                rng = np.random.default_rng([GT_TAG, fi, ri, ki, oi])
                stats[(f, rel, k)] = cluster_boot(y, ep, rng, n_boot)
    delta = {}
    for k in kpis:
        m = [abs(v[0]) for (f, rel, kk), v in stats.items() if kk == k and np.isfinite(v[0])]
        delta[k] = max(rel_frac * (max(m) if m else 0.0), float(floors.get(k, 0.0)))
    cells, counts = [], {"TRUE": 0, "NULL": 0, "INDET": 0}
    for (f, rel, k), (mean, ci, n, n_eps) in stats.items():
        c = classify(mean, ci, delta[k], n, n_eps, min_units, min_eps)
        counts[c["status"]] += 1
        cells.append({"family": f, "relation": rel, "kpi": k, "mean": mean, "ci": list(ci), "n": n, "n_eps": n_eps,
                      "delta": delta[k], **c})
    return {"orient": orient, "delta": delta, "cells": cells, "counts": counts,
            "rule": f"TRUE: CI{list(CI)} excludes 0 and |mean| >= delta; NULL: CI within +-delta; delta = "
                    f"max({rel_frac} * max|mean| of the KPI, floor {dict(floors)}); INDET otherwise / unsupported "
                    f"(< {min_units} units or < {min_eps} episodes); episode-cluster bootstrap B={n_boot}, tag "
                    f"{GT_TAG}"}


def receiving_shares(rows, gt_static_by_seed: dict, kpis=GT_KPIS, top: int = 5) -> list:
    """Per (seed, pico): receiving-cell shares of the "dir" load gain over the pico's sleep units (step > 0)."""
    li = list(kpis).index("load")
    out = []
    groups = {}
    for r in rows:
        if r["family"] == "sleep" and r["step"] > 0:
            groups.setdefault((r["ep"], r["c"]), []).append(r["cells"]["dir"][li])
    for (seed, pico), vs in sorted(groups.items()):
        d = np.sum(vs, 0)
        pos = np.where(np.arange(len(d)) == pico, 0.0, np.maximum(d, 0.0))
        tot = float(pos.sum())
        cand = (gt_static_by_seed.get(seed) or {}).get("cand", {})
        cand = cand.get(pico, cand.get(str(pico)))
        cand = int(cand) if cand is not None else None
        shares = pos / tot if tot > 0 else np.zeros_like(pos)
        order = [int(q) for q in np.argsort(-shares, kind="stable")[:top] if shares[q] > 0]
        top1 = order[0] if order else None
        out.append({"seed": int(seed), "pico": int(pico), "n_units": len(vs), "cand": cand, "top1": top1,
                    "top1_is_cand": top1 is not None and top1 == cand,
                    "share_cand": float(shares[cand]) if cand is not None else None,
                    "shares": {str(q): float(shares[q]) for q in order}, "pico_load_delta": float(d[pico]),
                    "gain_total": tot})
    return out


def summarize(episodes, orients=ORIENTS, **kw) -> dict:
    rows = unit_rows(episodes)
    per = {}
    for r in rows:
        per[r["family"]] = per.get(r["family"], 0) + 1
    return {"n_labels": len(rows), "labels_per_family": per, "tables": {o: gt_table(rows, o, **kw) for o in orients},
            "receiving": receiving_shares(rows, {int(e["seed"]): e.get("gt_static") or {} for e in episodes})}


__all__ = ["CELL_KPIS", "CI", "FAMILIES", "GT_H", "GT_KPIS", "GT_KS", "GT_MIN_EPS", "GT_MIN_UNITS", "GT_MODES",
           "GT_RATE", "GT_TAG", "KPI_FLOOR", "N_BOOT", "ORIENTS", "PRIMARY_ORIENT", "REL_FRAC", "RELATIONS",
           "classify", "cluster_boot", "gt_table", "receiving_shares", "relation_cells", "summarize", "unit_delta",
           "unit_rows"]
