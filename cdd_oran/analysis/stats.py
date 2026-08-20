"""Significance stats for CDL-vs-MLP per-planner utilities. numpy only (no scipy).

Given paired per-seed utilities for two conditions (e.g. CDL vs MLP) on the same
planner, computes:
  - bootstrap 95% CI of the mean difference (CDL - MLP),
  - paired Wilcoxon signed-rank test (two-sided, normal approximation with
    tie + continuity correction),
  - Cohen's d for paired samples.

Self-check: `uv run python -m cdd_oran.analysis.stats`
"""

from __future__ import annotations

import math

import numpy as np


def bootstrap_ci(diff: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    """Bootstrap CI of the mean of paired differences."""
    diff = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = diff[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean_diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray):
    """Two-sided paired Wilcoxon signed-rank test, normal approximation.

    Handles ties via average ranks and a tie correction on the variance, plus a
    continuity correction. Zero differences are dropped (Wilcoxon convention).
    Returns W statistic (min of signed rank sums), z, and two-sided p-value.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x - y
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return {"W": float("nan"), "z": 0.0, "p_value": 1.0, "n": 0}

    ranks = _average_ranks(np.abs(d))
    r_plus = float(ranks[d > 0].sum())
    r_minus = float(ranks[d < 0].sum())
    W = min(r_plus, r_minus)

    mean_w = n * (n + 1) / 4.0
    # tie correction
    _, counts = np.unique(np.abs(d), return_counts=True)
    tie_term = float(np.sum(counts**3 - counts))
    var_w = (n * (n + 1) * (2 * n + 1) - tie_term / 2.0) / 24.0
    if var_w <= 0:
        return {"W": W, "z": 0.0, "p_value": 1.0, "n": n}
    # continuity correction toward the mean
    cc = 0.5 * np.sign(mean_w - W)
    z = (W - mean_w + cc) / math.sqrt(var_w)
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return {"W": float(W), "z": float(z), "p_value": float(min(1.0, p)), "n": n}


def cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d for paired samples: mean(diff) / std(diff)."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0


def compare(cdl: np.ndarray, mlp: np.ndarray, seed: int = 0):
    """Full comparison for one planner (paired arrays: cdl[i], mlp[i] same seed)."""
    cdl = np.asarray(cdl, dtype=float)
    mlp = np.asarray(mlp, dtype=float)
    return {
        "n_pairs": len(cdl),
        "cdl_mean": float(cdl.mean()),
        "mlp_mean": float(mlp.mean()),
        **bootstrap_ci(cdl - mlp, seed=seed),
        "wilcoxon": wilcoxon_signed_rank(cdl, mlp),
        "cohens_d": cohens_d_paired(cdl, mlp),
    }


def _average_ranks(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j < len(a) and sorted_a[j] == sorted_a[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0  # 1-based average rank
        i = j
    return ranks


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _demo() -> None:
    rng = np.random.default_rng(42)
    # Clear positive effect: cdl consistently above mlp.
    mlp = rng.normal(1.0, 0.1, size=30)
    cdl = mlp + rng.normal(0.5, 0.05, size=30)
    res = compare(cdl, mlp)
    assert res["mean_diff"] > 0.3, res
    assert res["ci_low"] > 0, res  # CI excludes 0 -> significant
    assert res["wilcoxon"]["p_value"] < 0.05, res
    assert res["cohens_d"] > 0.8, res  # large effect

    # No effect: same distribution -> not significant.
    a = rng.normal(0.0, 1.0, size=40)
    b = rng.normal(0.0, 1.0, size=40)
    res0 = compare(a, b)
    assert res0["wilcoxon"]["p_value"] > 0.05, res0
    assert res0["ci_low"] < 0 < res0["ci_high"], res0

    # Average-rank helper sanity on a tie.
    r = _average_ranks(np.array([1.0, 2.0, 2.0, 4.0]))
    assert list(r) == [1.0, 2.5, 2.5, 4.0], r

    print("stats.py self-check passed")


if __name__ == "__main__":
    _demo()
