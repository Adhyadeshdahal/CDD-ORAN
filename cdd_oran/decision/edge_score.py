"""Edge-recovery scoring for the E6-P discovery study (scratchpad/e6_dev/decision/STEP1_MSCR_PLAN.md, "Metrics";
protocol docs/benchmark/E6P_DISCOVERY_PROTOCOL.md).

Edge space: the 60 hypotheses (knob family f, relation rel, KPI k) = ``crt_units.HYPOTHESES``, "dir" orientation.
A method's output (``decl``): {(f, rel, k): {"declared": bool | None, "sign": int, ...}}; ``declared`` None (or the
key absent) = the method cannot express that hypothesis ("unmapped": excluded from its scores and counted).
Reference (``ref``): {(f, rel, k): {"status": TRUE | NULL | INDET, "sign": int}} from the knockout GT
(``gt_reference(gt_p.gt_table(...))``) or, in a dry run only, from the declared physics PROXY (``physics_reference``,
never a verdict input).

Scores over TRUE u NULL (INDET excluded and counted):
  TP = declared and TRUE (whatever the sign); FP = declared and NULL; FN = TRUE and not declared;
  precision = TP / (TP + FP) (NaN if nothing declared), recall = TP / (TP + FN) (NaN if no TRUE), F1 = 2PR / (P + R)
  (0 when TP = 0 and it is defined), sign accuracy = share of TP whose declared sign equals the GT sign (NaN if no TP);
  computed overall, for the indirect relation (nbr) and for own / far;
  far_declared = number of declared far hypotheses (ANY GT status; P1's "far declarations"), far_fp = declared & NULL.
Receiving cell: top-1 accuracy of a method's localisation {(seed, pico): top1 cell} vs ``gt_static["cand"]`` of that
seed (privileged; scorer only); ``gt_receiving_summary`` = how often the knockout GT's own top-1 receiver IS the
candidate (gt_p.receiving_shares, on the GT seeds).
"""
from __future__ import annotations

import numpy as np

from .gt_p import FAMILIES, RELATIONS

KPIS = ("pv", "v", "e", "rlf", "load")
HYPOTHESES = tuple((f, r, k) for f in FAMILIES for r in RELATIONS for k in KPIS)

# Declared PHYSICS PROXY ("dir" orientation: effect of a knob-value increase), from the plan's "Expected" line and the
# plant mechanisms tested in tests/test_e6p_gt.py. DRY-RUN SCORING and the tau-tuning SENSITIVITY only; listed edges
# TRUE with the sign, every other hypothesis NULL. Not ground truth.
PHYSICS_PRIOR = {("sleep", "nbr", "load"): 1, ("sleep", "nbr", "pv"): 1, ("sleep", "nbr", "v"): 1,
                 ("sleep", "own", "load"): -1, ("sleep", "own", "e"): -1,
                 ("carrier", "own", "e"): 1, ("carrier", "own", "pv"): -1, ("carrier", "own", "v"): -1,
                 ("ptx", "own", "e"): 1, ("ptx", "own", "load"): 1, ("ptx", "nbr", "load"): -1,
                 ("prot_min", "own", "pv"): -1}


def gt_reference(table: dict) -> dict:
    """Reference from a ``gt_p.gt_table`` (primary orientation "dir")."""
    return {(c["family"], c["relation"], c["kpi"]): {"status": c["status"], "sign": int(c["sign"])}
            for c in table["cells"]}


def physics_reference(prior: dict | None = None) -> dict:
    prior = PHYSICS_PRIOR if prior is None else prior
    return {h: ({"status": "TRUE", "sign": int(prior[h])} if h in prior else {"status": "NULL", "sign": 0})
            for h in HYPOTHESES}


def _safe(a, b):
    return float(a / b) if b else float("nan")


def confusion(decl: dict, ref: dict, relation: str | None = None) -> dict:
    tp = fp = fn = tn = sign_ok = n_indet = n_unmapped = 0
    for h in HYPOTHESES:
        if relation is not None and h[1] != relation:
            continue
        g = ref.get(h, {"status": "INDET", "sign": 0})
        d = decl.get(h)
        if g["status"] == "INDET":
            n_indet += 1
            continue
        if d is None or d.get("declared") is None:
            n_unmapped += 1
            continue
        if d["declared"]:
            if g["status"] == "TRUE":
                tp += 1
                sign_ok += int(int(d.get("sign", 0)) == g["sign"])
            else:
                fp += 1
        elif g["status"] == "TRUE":
            fn += 1
        else:
            tn += 1
    prec, rec = _safe(tp, tp + fp), _safe(tp, tp + fn)
    if tp == 0:
        f1 = 0.0 if (tp + fp + fn) > 0 else float("nan")
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1,
            "sign_acc": _safe(sign_ok, tp), "n_true": tp + fn, "n_null": fp + tn, "n_indet": n_indet,
            "n_unmapped": n_unmapped}


def score_method(decl: dict, ref: dict) -> dict:
    far_dec = sum(1 for h in HYPOTHESES if h[1] == "far" and (decl.get(h) or {}).get("declared"))
    far_fp = sum(1 for h in HYPOTHESES if h[1] == "far" and (decl.get(h) or {}).get("declared")
                 and ref.get(h, {}).get("status") == "NULL")
    return {"overall": confusion(decl, ref), "indirect": confusion(decl, ref, "nbr"),
            "own": confusion(decl, ref, "own"), "far": confusion(decl, ref, "far"), "far_declared": far_dec,
            "far_fp": far_fp, "n_declared": sum(1 for d in decl.values() if d and d.get("declared"))}


def p1_check(m: dict, rules: dict | None = None) -> dict:
    """P1 of the plan on a ``score_method`` output (MSCR-CRT pooled EVAL)."""
    r = {"ind_recall": 2.0 / 3.0, "ind_precision": 0.80, "f1": 0.60, "sign_acc": 0.90, "max_far": 1}
    r.update(rules or {})
    ind, ov = m["indirect"], m["overall"]

    def ge(v, t):
        return bool(np.isfinite(v) and v >= t - 1e-12)
    parts = {"indirect_recall": ge(ind["recall"], r["ind_recall"]),
             "indirect_precision": ge(ind["precision"], r["ind_precision"]),
             "overall_f1": ge(ov["f1"], r["f1"]), "sign_accuracy": ge(ov["sign_acc"], r["sign_acc"]),
             "far_declarations": m["far_declared"] <= r["max_far"]}
    return {"pass": all(parts.values()), "parts": parts, "rules": r,
            "values": {"indirect_recall": ind["recall"], "indirect_precision": ind["precision"],
                       "overall_f1": ov["f1"], "sign_accuracy": ov["sign_acc"], "far_declared": m["far_declared"]}}


def p2_check(mscr_f1: dict, baseline_f1: dict, folds=("fold0", "fold1", "fold2"), min_folds: int = 2) -> dict:
    """P2: MSCR indirect F1 >= the best DEV-tuned baseline's, pooled and in >= min_folds folds.
    ``mscr_f1`` {split: F1}; ``baseline_f1`` {method: {split: F1}} (NaN F1 counts as 0 for a baseline and fails MSCR)."""
    def best(split):
        vals = [v.get(split, float("nan")) for v in baseline_f1.values()]
        vals = [0.0 if not np.isfinite(v) else v for v in vals]
        return max(vals) if vals else 0.0

    def ok(split):
        m = mscr_f1.get(split, float("nan"))
        return bool(np.isfinite(m) and m >= best(split) - 1e-12)
    per = {s: {"mscr": mscr_f1.get(s), "best_baseline": best(s),
               "best_method": max(baseline_f1, key=lambda b: (np.nan_to_num(baseline_f1[b].get(s, 0.0)), b))
               if baseline_f1 else None, "ok": ok(s)} for s in ("pooled", *folds)}
    n_f = sum(per[s]["ok"] for s in folds)
    return {"pass": per["pooled"]["ok"] and n_f >= min_folds, "folds_ok": n_f, "per_split": per}


def top1_accuracy(loc: dict, gt_static_by_seed: dict) -> dict:
    """``loc`` {(seed, pico): {"top1": cell | None, ...}} vs the seed's ES candidate macro of that pico."""
    hits, n, none = 0, 0, 0
    for (seed, pico), v in loc.items():
        cand = (gt_static_by_seed.get(seed) or {}).get("cand", {})
        cand = cand.get(str(pico), cand.get(pico))
        if cand is None:
            continue
        if v.get("top1") is None:
            none += 1
            continue
        n += 1
        hits += int(int(v["top1"]) == int(cand))
    return {"n": n, "hits": hits, "acc": _safe(hits, n), "no_localisation": none}


def gt_receiving_summary(receiving: list) -> dict:
    ok = [r for r in receiving if r.get("cand") is not None and r.get("top1") is not None]
    return {"n": len(ok), "top1_is_cand": sum(r["top1_is_cand"] for r in ok),
            "rate": _safe(sum(r["top1_is_cand"] for r in ok), len(ok)),
            "mean_share_cand": float(np.mean([r["share_cand"] for r in ok])) if ok else float("nan")}


def score_splits(decl_by_split: dict, ref: dict) -> dict:
    """{split: score_method(...)} for {split: decl}."""
    return {s: score_method(d, ref) for s, d in decl_by_split.items()}


__all__ = ["HYPOTHESES", "KPIS", "PHYSICS_PRIOR", "confusion", "gt_receiving_summary", "gt_reference", "p1_check",
           "p2_check", "physics_reference", "score_method", "score_splits", "top1_accuracy"]
