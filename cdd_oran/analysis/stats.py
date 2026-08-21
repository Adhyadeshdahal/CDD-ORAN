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

import json
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


# ---------------------------------------------------------------------------
# Robust, seed-based effect sizes and multiple-comparison corrections (M1).
# The paper's lead metric (Cohen's d on one mitigation seed) is pseudo-replication
# on bounded, skewed utility. These are nonparametric and rank-based, so they do
# not assume normality, and they pair naturally with the Wilcoxon test above.
# ---------------------------------------------------------------------------


def _cliffs_label(delta: float) -> str:
    """Romano et al. magnitude thresholds for |Cliff's delta|."""
    d = abs(delta)
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def cliffs_delta(a: np.ndarray, b: np.ndarray):
    """Cliff's delta: nonparametric dominance effect size in [-1, 1].

    delta = (#(a > b) - #(a < b)) / (n_a * n_b) over all cross pairs. +1 means every
    a beats every b, -1 the reverse, 0 no dominance. Robust to non-normal / bounded
    data; pairs with the Wilcoxon test. Returns (delta, magnitude_label).
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        return 0.0, _cliffs_label(0.0)
    diff = a[:, None] - b[None, :]
    greater = int(np.count_nonzero(diff > 0))
    less = int(np.count_nonzero(diff < 0))
    delta = (greater - less) / (a.size * b.size)
    return float(delta), _cliffs_label(delta)


def prob_superiority(a: np.ndarray, b: np.ndarray) -> float:
    """Common-language effect size: P(a > b) + 0.5 * P(a == b) over cross pairs."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        return 0.5
    diff = a[:, None] - b[None, :]
    greater = int(np.count_nonzero(diff > 0))
    equal = int(np.count_nonzero(diff == 0))
    return float((greater + 0.5 * equal) / (a.size * b.size))


def holm_correction(pvalues) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values for a family of tests.

    Controls the family-wise error rate (e.g. the 4-planner x 2-env comparisons).
    Each adjusted p is >= its raw p, and the adjusted values are monotone in the raw
    p ordering. Adjusted p_(i) = max over k<=i of (m - k + 1) * p_(k), clipped to 1.
    """
    p = np.asarray(pvalues, dtype=float).ravel()
    m = p.size
    if m == 0:
        return p.copy()
    order = np.argsort(p, kind="mergesort")
    adj_sorted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj_sorted[rank] = min(running, 1.0)
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj


def bh_fdr(pvalues) -> np.ndarray:
    """Benjamini-Hochberg step-up adjusted p-values (controls the FDR).

    Less conservative than Holm; use when reporting many exploratory comparisons.
    Adjusted p_(i) = min over k>=i of (m / k) * p_(k), clipped to 1.
    """
    p = np.asarray(pvalues, dtype=float).ravel()
    m = p.size
    if m == 0:
        return p.copy()
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    ranks = np.arange(1, m + 1)
    raw = sorted_p * m / ranks
    # step-up: enforce monotone non-decreasing along the sorted p order
    adj_sorted = np.minimum.accumulate(raw[::-1])[::-1]
    adj_sorted = np.minimum(adj_sorted, 1.0)
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj


def compare_seeds(cdl: np.ndarray, mlp: np.ndarray, seed: int = 0):
    """Robust comparison for one planner from PER-SEED PAIRED MEANS.

    REPLICATION UNIT: independent mitigation seeds, NOT within-run samples. Pass one
    scalar per seed for each condition -- e.g. cdl[k] is the mean utility of run k
    under CDL, mlp[k] the mean utility of the paired run k under MLP. Using within-run
    samples here would be pseudo-replication and inflate significance.

    Returns the paired mean diff with a 95% bootstrap CI, the Wilcoxon p-value, and
    three effect sizes: Cohen's d (kept, but assumption-bound -- normality of the
    paired diffs), Cliff's delta (nonparametric dominance), and P(CDL > MLP)
    (common-language). Lead with the nonparametric trio; treat Cohen's d as secondary.
    """
    cdl = np.asarray(cdl, dtype=float).ravel()
    mlp = np.asarray(mlp, dtype=float).ravel()
    delta, delta_label = cliffs_delta(cdl, mlp)
    return {
        "n_seeds": int(cdl.size),
        "replication_unit": "independent seeds (paired per-seed means)",
        "cdl_mean": float(cdl.mean()) if cdl.size else float("nan"),
        "mlp_mean": float(mlp.mean()) if mlp.size else float("nan"),
        **bootstrap_ci(cdl - mlp, seed=seed),
        "wilcoxon": wilcoxon_signed_rank(cdl, mlp),
        "cohens_d": cohens_d_paired(cdl, mlp),
        "cohens_d_note": "assumption-bound (normality of paired diffs); secondary",
        "cliffs_delta": delta,
        "cliffs_label": delta_label,
        "prob_cdl_gt_mlp": prob_superiority(cdl, mlp),
    }


def _satisfied(utility: float, norm_threshold: float, direction: int) -> bool:
    """Satisfaction indicator, matching cost.weighted_distance's `satisfied`.

    direction 0 (maximiser): satisfied when utility >= norm_threshold.
    direction 1 (minimiser): satisfied when utility <= norm_threshold.
    Inlined (not imported) so stats.py stays numpy-only; cost.py pulls in torch.
    """
    if direction == 0:
        return utility >= norm_threshold
    return utility <= norm_threshold


def satisfaction_rate(
    utilities_json_path: str,
    directions=None,
    norm_thresholds=None,
):
    """Mean xApp satisfaction rate per planner from a run's utilities.json.

    Satisfaction = fraction of conflicting xApps whose (normalised) utility meets its
    threshold, using the same `satisfied` indicator as cost.weighted_distance. The
    per-panel `planner_utilities[planner]` list holds one normalised utility per
    conflicting xApp, aligned to `curves[].xapp_id`.

    MISSING FIELDS (utilities.json schema v1): the satisfied indicator needs, per
    xApp, (a) the optimisation `direction` and (b) the normalised threshold
    `(threshold - mean) / std`. v1 persists only per-KPI `kpi_thresholds` and
    `mean_std_kpis` and no xApp->KPI index mapping, so neither is derivable from the
    file alone. Rather than guess (e.g. assume xapp_id == kpi_id, which is false for
    multi-KPI xApps in env_ii/iii, and direction is absent entirely), pass them in:
      directions:      {xapp_id: 0|1}
      norm_thresholds: {xapp_id: (threshold - mean) / std}
    both obtainable from the env (env.xapps[i].direction / .threshold / .mean / .std).

    Without them, returns {"ok": False, "missing": [...]} naming the gap, so
    evaluate.py can be extended to emit a per-xApp `satisfied` (or `direction` +
    `norm_threshold`) field. With them, returns {"ok": True, "satisfaction": {planner:
    rate}, "n_evaluations": {planner: count}}.
    """
    with open(utilities_json_path, encoding="utf-8") as handle:
        data = json.load(handle)

    if directions is None or norm_thresholds is None:
        return {
            "ok": False,
            "missing": [
                "per-xApp `direction` (minimise/maximise) -- absent from utilities.json",
                "per-xApp `norm_threshold` = (threshold - mean)/std -- only per-KPI "
                "values and no xApp->KPI mapping are stored",
            ],
            "note": (
                "utilities.json v%s lacks the satisfaction inputs; pass directions= and "
                "norm_thresholds= from the env, or extend evaluate.py to persist a "
                "per-xApp `satisfied` field." % str(data.get("version"))
            ),
            "planners": list(data.get("algorithm_names", [])),
        }

    hits: dict[str, int] = {}
    total: dict[str, int] = {}
    skipped = 0
    for step in data.get("steps", []):
        for panel in step.get("panels", []):
            xapp_ids = [curve["xapp_id"] for curve in panel.get("curves", [])]
            for planner, utilities in panel.get("planner_utilities", {}).items():
                if len(utilities) != len(xapp_ids):
                    skipped += 1
                    continue
                for xid, utility in zip(xapp_ids, utilities, strict=True):
                    if xid not in directions or xid not in norm_thresholds:
                        skipped += 1
                        continue
                    total[planner] = total.get(planner, 0) + 1
                    if _satisfied(utility, norm_thresholds[xid], directions[xid]):
                        hits[planner] = hits.get(planner, 0) + 1

    satisfaction = {
        planner: (hits.get(planner, 0) / n if n else float("nan"))
        for planner, n in total.items()
    }
    return {
        "ok": True,
        "satisfaction": satisfaction,
        "n_evaluations": total,
        "skipped_xapp_evaluations": skipped,
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

    # --- M1 robust metrics ---
    hi = np.array([10.0, 11.0, 12.0, 13.0])
    lo = np.array([1.0, 2.0, 3.0, 4.0])
    d_hi, lbl_hi = cliffs_delta(hi, lo)
    assert d_hi == 1.0 and lbl_hi == "large", (d_hi, lbl_hi)
    d_same, lbl_same = cliffs_delta(hi, hi.copy())
    assert d_same == 0.0 and lbl_same == "negligible", (d_same, lbl_same)
    assert cliffs_delta(lo, hi)[0] == -1.0  # antisymmetric

    assert prob_superiority(hi, lo) == 1.0
    assert prob_superiority(hi, hi.copy()) == 0.5  # all ties -> 0.5
    assert abs(prob_superiority(lo, hi)) == 0.0

    # Holm / BH: adjusted >= raw, monotone in raw-p order, clipped to 1.
    raw_p = np.array([0.001, 0.013, 0.02, 0.04, 0.5])  # 8 comparisons family, unsorted mix
    for adj in (holm_correction(raw_p), bh_fdr(raw_p)):
        assert np.all(adj >= raw_p - 1e-12), adj  # never smaller than raw
        assert np.all(adj <= 1.0 + 1e-12), adj
        ordered = adj[np.argsort(raw_p, kind="mergesort")]
        assert np.all(np.diff(ordered) >= -1e-12), ordered  # monotone non-decreasing
    assert np.all(holm_correction(raw_p) >= bh_fdr(raw_p) - 1e-12)  # Holm >= BH (more conservative)

    # compare_seeds on synthetic paired seed means: CDL above MLP each seed.
    rng2 = np.random.default_rng(7)
    mlp_seeds = rng2.normal(0.6, 0.02, size=8)
    cdl_seeds = rng2.normal(0.9, 0.02, size=8)  # cleanly separated -> full dominance
    cs = compare_seeds(cdl_seeds, mlp_seeds)
    for field in (
        "n_seeds", "mean_diff", "ci_low", "ci_high", "wilcoxon", "cohens_d",
        "cliffs_delta", "cliffs_label", "prob_cdl_gt_mlp", "replication_unit",
    ):
        assert field in cs, (field, cs)
    assert cs["n_seeds"] == 8
    assert cs["mean_diff"] > 0 and cs["ci_low"] > 0
    assert cs["cliffs_delta"] == 1.0 and cs["prob_cdl_gt_mlp"] == 1.0

    # satisfaction_rate: without env fields it reports the missing inputs, not a guess.
    import glob

    run_files = glob.glob("runs/**/utilities.json", recursive=True)
    if run_files:
        miss = satisfaction_rate(run_files[0])
        assert miss["ok"] is False and miss["missing"], miss
        # With directions + norm_thresholds supplied, it computes rates in [0, 1].
        import json as _json

        d0 = _json.load(open(run_files[0], encoding="utf-8"))
        xids = {c["xapp_id"] for s in d0["steps"] for p in s["panels"] for c in p["curves"]}
        directions = {i: 0 for i in xids}
        norm_thresholds = {i: 0.0 for i in xids}
        got = satisfaction_rate(run_files[0], directions, norm_thresholds)
        assert got["ok"] is True, got
        assert all(0.0 <= v <= 1.0 for v in got["satisfaction"].values()), got

    print("stats.py self-check passed")


if __name__ == "__main__":
    _demo()
