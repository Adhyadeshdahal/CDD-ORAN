"""MSCR: Max-Stratified Correlation-Ratio test for parameter->KPI edge discovery (v2 selection rule).

What it is
----------
For a target KPI ``y`` and candidate column ``i``, and for each OTHER column ``g`` (the conditioner):
split rows into ``nc`` equal-count strata of ``g`` (drop strata with fewer than ``min_stratum`` rows);
within each stratum, split rows into ``nb`` equal-count bins of candidate ``i``. The pooled conditional
correlation ratio is ``S_g = sum_s BSS_s / sum_s TSS_s`` (between-bin over total sum of squares of y),
and the test statistic is ``S* = max_g S_g``. The max over single conditioners is the "gate search":
it recovers co-parent-GATED edges (e.g. E2's P0->K5, E5's P0->K_harm) that kernel/distance CI tests
(RCoT, pdCor) dilute away.

Null and p-value: within each g-stratum, ``n_perm`` random equal-size partitions of y give null draws of
S_g. Under equal-count binning those draws depend on the candidate only through the bin sizes, so one
bank per (target, g) is shared across candidates. ``S*_null = max_{g != i}`` of the bank draws, and
``p = (1 + #{S*_null >= S*}) / (n_perm + 1)``. The shared bank is NOT a joint null across candidates or
across g. Only marginal p-values are used, and BY allows arbitrary dependence among them.

Selection (v2): per target, Benjamini-Yekutieli at ``q`` over the PARAMETER candidates only (the
intervention estimand for conflict mitigation). Lagged-KPI columns, if supplied, act as conditioners
but are never tested.

History: v1 (B=299, BH over all 14 E2 candidates) failed its preregistered post-selection FDR gate
(pooled 25/329; family-level 0.062, CI [0.041, 0.083], a warning, not a proved excess). v2 = B=2999
+ per-target BY over params, preregistered (``scratchpad/p0k5_fp_calibration/PREDECLARE_v2.md``) and
confirmed on 100 fresh E2 seeds (800000-800099): family-level FDR 0.0082 (one-sided 95% UB 0.0121),
P0->K5 recall 100/100, total param-edge recall 0.945.

SCOPE: binding, from the sol review of the v2 result. Read before using the output.
----------------------------------------------------------------------------------------
- VALIDATED ON RANDOMIZED DESIGNS ONLY: exogenous, independently randomized parameter candidates,
  noiseless E2, n=4000, the frozen constants below. The FDR evidence is empirical, finite-simulation
  evidence consistent with control under that design (BY's marginal-validity premise was checked
  empirically, not proved). There is NO general FDR-control guarantee and NO general
  conditional-independence claim for the max statistic.
- Known failure: single-parameter-actuation / low-diversity designs (E1: FDR 0.81). Not validated on
  observational, noisy, or confounded data, or on E5 (E5 needs its own calibration check plus fresh
  seeds). E5 CORE dev probe: P0->K_harm 7/10 at n=6000, 10/10 at n>=12000 (power-limited; fix n).
- Declares param->KPI ASSOCIATION edges. A declared edge alone does not identify an intervention
  effect, and there is no KPI->KPI discovery claim.
- Recall limitation: weak co-parent edges can be missed (E2 P6->K5 recall 12%). State this whenever
  recall or graph recovery is quoted.
- Statistic limits: one conditioner at a time (joint AND-gates are caught only when one axis isolates
  enough of the gate); mean-shift dependence only; coarse equal-count bins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MSCR_VERSION = "mscr-v2"


@dataclass(frozen=True)
class MSCRConfig:
    nc: int = 6               # equal-count strata of the conditioner g
    nb: int = 8               # equal-count bins of the candidate within a stratum
    min_stratum: int = 40     # drop strata with fewer rows
    n_perm: int = 2999        # permutations per (target, g) bank; p floor 1/(n_perm+1)
    q: float = 0.05           # BY level, per target, over the parameter family


def frozen_config() -> MSCRConfig:
    """The confirmed v2 constants (NC=6, NB=8, MIN=40, B=2999, BY q=0.05)."""
    return MSCRConfig()


@dataclass(frozen=True)
class MSCRResult:
    pvals: np.ndarray       # (n_targets, n_params) permutation p-values
    s_star: np.ndarray      # (n_targets, n_params) observed S*
    declared: np.ndarray    # (n_targets, n_params) bool, per-target BY selection
    config: MSCRConfig
    version: str = MSCR_VERSION

    def param_edges(self) -> set[tuple[int, int]]:
        """Declared edges as ``(kpi_index, param_index)``, the convention of ``_TRUE_ADJACENCY``."""
        return {(int(k), int(p)) for k, p in zip(*np.nonzero(self.declared), strict=True)}


def equal_count_labels(values: np.ndarray, nbins: int) -> np.ndarray:
    """Rank-based equal-count bin labels 0..nbins-1 (sizes differ by at most 1, stable ties)."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values))
    return (ranks * nbins) // len(values)


def _between_ss(y: np.ndarray, labels: np.ndarray, nbins: int) -> float:
    total = y.sum()
    tb = np.bincount(labels, weights=y, minlength=nbins)
    nb = np.bincount(labels, minlength=nbins).astype(np.float64)
    mask = nb > 0
    return float(np.sum(tb[mask] ** 2 / nb[mask]) - total * total / len(y))


def _build_bank(x: np.ndarray, y: np.ndarray, cfg: MSCRConfig, rng: np.random.Generator):
    """Per conditioner g: strata rows, pooled TSS, and n_perm shared null draws of S_g."""
    n_cols = x.shape[1]
    denom = np.zeros(n_cols)
    null = np.zeros((n_cols, cfg.n_perm))
    strata = []
    for g in range(n_cols):
        lab = equal_count_labels(x[:, g], cfg.nc)
        bss_null = np.zeros(cfg.n_perm)
        tss = 0.0
        info = []
        for s in range(cfg.nc):
            rows = np.nonzero(lab == s)[0]
            if len(rows) < cfg.min_stratum:
                continue
            y_s = y[rows]
            ns = len(y_s)
            sizes = np.bincount((np.arange(ns) * cfg.nb) // ns, minlength=cfg.nb).astype(np.int64)
            offsets = np.concatenate(([0], np.cumsum(sizes)[:-1]))
            t_s = y_s.sum()
            tss += float(np.dot(y_s, y_s)) - t_s * t_s / ns
            perm = np.argsort(rng.random((cfg.n_perm, ns)), axis=1)
            groupsums = np.add.reduceat(y_s[perm], offsets, axis=1)
            bss_null += np.sum(groupsums ** 2 / sizes[None, :], axis=1) - t_s * t_s / ns
            info.append((rows, y_s))
        denom[g] = tss
        if tss > 0:
            null[g] = bss_null / tss
        strata.append(info)
    return denom, null, strata


def _s_star(col: np.ndarray, i: int, denom, strata, cfg: MSCRConfig) -> float:
    best = -np.inf
    for g, info in enumerate(strata):
        if g == i or denom[g] <= 0:
            continue
        bss = sum(_between_ss(y_s, equal_count_labels(col[rows], cfg.nb), cfg.nb) for rows, y_s in info)
        best = max(best, bss / denom[g])
    return best


def _pval(s_obs: float, i: int, null: np.ndarray) -> float:
    keep = np.ones(null.shape[0], dtype=bool)
    keep[i] = False
    s_null = np.max(null[keep], axis=0)
    return (1.0 + np.count_nonzero(s_null >= s_obs)) / (null.shape[1] + 1.0)


def by_declare(pvals, q: float) -> np.ndarray:
    """Benjamini-Yekutieli step-up at level q (valid under arbitrary dependence of valid p-values)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    c_m = sum(1.0 / k for k in range(1, m + 1))
    order = np.argsort(p, kind="stable")
    passed = p[order] <= np.arange(1, m + 1) * q / (m * c_m)
    declared = np.zeros(m, dtype=bool)
    if passed.any():
        declared[order[: np.nonzero(passed)[0].max() + 1]] = True
    return declared


def discover_mscr(
    x: np.ndarray,
    y: np.ndarray,
    n_params: int,
    seed: int,
    config: MSCRConfig | None = None,
) -> MSCRResult:
    """Run MSCR-v2 param->KPI discovery.

    x : (n, n_cols) candidate matrix whose FIRST ``n_params`` columns are the randomized parameters
        (the tested family); any further columns (e.g. lagged KPIs) are conditioners only.
    y : (n, n_targets) target KPIs.
    seed : dataset seed; the bank for target j uses ``default_rng([seed, j, n_perm])``, reproducing
        the v2 confirmatory run bit-for-bit.
    """
    cfg = config or frozen_config()
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"row mismatch: x {x.shape} vs y {y.shape}")
    if not 1 <= n_params <= x.shape[1] or x.shape[1] < 2:
        raise ValueError(f"need 1 <= n_params <= n_cols and n_cols >= 2, got {n_params}, {x.shape[1]}")

    n_t = y.shape[1]
    pvals = np.zeros((n_t, n_params))
    s_star = np.zeros((n_t, n_params))
    declared = np.zeros((n_t, n_params), dtype=bool)
    for j in range(n_t):
        denom, null, strata = _build_bank(x, y[:, j], cfg, np.random.default_rng([int(seed), j, cfg.n_perm]))
        for i in range(n_params):
            s_star[j, i] = _s_star(x[:, i], i, denom, strata, cfg)
            pvals[j, i] = _pval(s_star[j, i], i, null)
        declared[j] = by_declare(pvals[j], cfg.q)
    return MSCRResult(pvals=pvals, s_star=s_star, declared=declared, config=cfg)
