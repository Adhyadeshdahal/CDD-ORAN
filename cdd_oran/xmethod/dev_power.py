"""Seed-count calculations on DEV aggregates (brief dev-runs step 4; ruling R-12). Descriptive only.

1. Paired recall gap: for two arms A, B of one cell (same measurement seeds), d_s = recall_A(s) - recall_B(s);
   the number of EVAL seeds S to detect |mean d| = delta (default .15) with a two-sided paired t-test at alpha
   (.05) and power (.8) is the smallest S >= 2 with
       S >= ((t_{1-alpha/2, S-1} + t_{power, S-1}) * sd_d / delta)^2
   (iterated from the normal approximation). sd_d = 0 (identical recalls on every seed) gives S = 2 and a note.
   Only arms that pass validity (R-20: neither the truth-null nor the placebo rate significantly above .05) and
   seeds where both recalls are defined enter.
2. Null-rate precision: seeds S so that the 95 % CI half-width of a rejection rate at p0 (.05) is <= h (.02),
   with m tests per seed and the DEV seed-cluster design effect deff (>= 1):
       S = ceil(z^2 * deff * p0 (1 - p0) / (h^2 m)).
"""
from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
from scipy import stats


def paired_seeds(sd_d: float, delta: float = 0.15, alpha: float = 0.05, power: float = 0.8,
                 s_max: int = 100_000) -> int:
    if not np.isfinite(sd_d) or sd_d <= 0:
        return 2
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    s = max(2, math.ceil(((za + zb) * sd_d / delta) ** 2))
    while s < s_max:
        need = ((stats.t.ppf(1 - alpha / 2, s - 1) + stats.t.ppf(power, s - 1)) * sd_d / delta) ** 2
        if s >= need:
            return s
        s += 1
    return s_max


def null_rate_seeds(m_per_seed: float, deff: float = 1.0, p0: float = 0.05, half: float = 0.02,
                    z: float = 1.959964) -> int | None:
    if not m_per_seed or m_per_seed <= 0:
        return None
    return math.ceil(z * z * max(deff, 1.0) * p0 * (1 - p0) / (half * half * m_per_seed))


def design_effect(hits: list[float], cnt: list[float]) -> float:
    """Seed-cluster design effect of a pooled rate (>= 1; 1 when undefined)."""
    h, c = np.asarray(hits, float), np.asarray(cnt, float)
    N, k = c.sum(), len(c)
    if N <= 0 or k < 2:
        return 1.0
    p = h.sum() / N
    if not 0 < p < 1:
        return 1.0
    v_clu = k / (k - 1) * float(np.sum((h - p * c) ** 2)) / N ** 2
    return max(1.0, v_clu / (p * (1 - p) / N))


def _valid(entry: dict) -> bool:
    p = entry.get("primary")
    return bool(p) and p.get("valid_flag") == "ok"


def _cell_of(key: str) -> str:
    return key.split("|", 1)[1]


def recall_power(agg: dict, delta: float = 0.15, alpha: float = 0.05, power: float = 0.8,
                 focal: str | None = "pmrt_eq") -> dict[str, Any]:
    """Per cell (world, regime, lam, kappa, n): seeds needed for every pair of valid arms; ``focal`` pairs (the
    proposed method vs each other arm) are the primary comparison, all pairs reported too."""
    by_cell: dict[str, dict[str, dict]] = {}
    for k, e in agg["cells"].items():
        by_cell.setdefault(_cell_of(k), {})[e["arm"]] = e
    out = {}
    for cell, arms in sorted(by_cell.items()):
        valid = sorted(a for a, e in arms.items() if _valid(e))
        pairs = []
        for a, b in itertools.combinations(valid, 2):
            ra, rb = arms[a]["primary"]["per_seed"], arms[b]["primary"]["per_seed"]
            seeds = sorted(s for s in set(ra) & set(rb)
                           if ra[s]["recall"] is not None and rb[s]["recall"] is not None)
            if len(seeds) < 2:
                continue
            d = np.array([ra[s]["recall"] - rb[s]["recall"] for s in seeds])
            sd = float(np.std(d, ddof=1))
            pairs.append({"a": a, "b": b, "n_seeds": len(seeds), "mean_d": float(d.mean()), "sd_d": sd,
                          "seeds_needed": paired_seeds(sd, delta, alpha, power), "sd_zero": sd == 0.0})
        foc = [p for p in pairs if focal in (p["a"], p["b"])] if focal else pairs
        out[cell] = {"valid_arms": valid, "invalid_arms": sorted(set(arms) - set(valid)), "pairs": pairs,
                     "max_focal": max((p["seeds_needed"] for p in foc), default=None),
                     "max_all": max((p["seeds_needed"] for p in pairs), default=None)}
    allmax_f = max((v["max_focal"] for v in out.values() if v["max_focal"]), default=None)
    allmax = max((v["max_all"] for v in out.values() if v["max_all"]), default=None)
    return {"delta": delta, "alpha": alpha, "power": power, "focal": focal, "cells": out,
            "max_over_cells_focal": allmax_f, "max_over_cells_all": allmax}


def null_precision(agg: dict, half: float = 0.02, p0: float = 0.05) -> dict[str, Any]:
    """Seeds per cell so that the truth-null (raw p, or declaration for tau arms) and placebo rate CIs have
    half-width <= ``half`` at ``p0``; deff from the DEV per-seed counts when available (else 1)."""
    out = {}
    for k, e in sorted(agg["cells"].items()):
        p = e.get("primary")
        if not p:
            continue
        row = {}
        for name in ("null_raw05", "null_decl", "plac_raw05", "plac_decl"):
            r = p.get(name)
            if not r or not r.get("n_seeds"):
                continue
            m = r["n"] / r["n_seeds"]
            row[name] = {"m_per_seed": m, "seeds_needed": null_rate_seeds(m, r.get("deff", 1.0), p0, half)}
        out[k] = row
    return out
