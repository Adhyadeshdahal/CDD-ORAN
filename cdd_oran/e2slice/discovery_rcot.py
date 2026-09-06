"""Label-free nonlinear E2 graph discovery via RCoT (Randomized Conditional Correlation Test).

**FROZEN** -- pre-registered in ``docs/benchmark/E2_RCOT_DISCOVERY_PROTOCOL.md`` (committed as
``PROTOCOL_COMMIT`` below, BEFORE any E2 recovery result was computed). This module is a SIBLING to
the frozen pdCor method in ``discovery.py``. It reads NO ground truth and scores against NO truth;
ground truth is read only in the separate recovery step, after the mask is persisted and hashed.
The primary null is ``block_perm`` (human-approved 2026-09-06; analytic_hbe is a sensitivity variant,
never the selector). The canonical persist path uses ``frozen_config()`` and ``load_discovery_rcot``
re-derives every frozen constant and the ``protocol_commit`` -- so a retuned mask can never be
persisted and loaded under this ``protocol_commit``.

Why RCoT instead of the frozen marginal-permutation pdCor
---------------------------------------------------------
The frozen method (``discovery.py``) over-selects lagged KPI->KPI edges ~9x. A verified calibration
study (``runs/calib-study/calib_study4_correct.py``, on a testbed byte-faithful to the real E2 KPI
mechanism) traced this to a MARGINAL-PERMUTATION MISCALIBRATION: a lagged-KPI candidate shares its
generating params with sibling lagged KPIs that live in the conditioning set Z, so a marginal
(global) permutation destroys the candidate|Z coupling and the null collapses too tight -> a tiny
partial-dCor rejects. RCoT is a genuine CONDITIONAL-independence test: it residualizes both the
candidate and the target on random Fourier features (RFF) of Z, then tests the residuals for
dependence, so the candidate|Z coupling is preserved and the null is (near-)nominal and FLAT in n.

The RCoT numerics (median-heuristic bandwidth, RFF residualization on Z, the RFF kernel
cross-covariance statistic ``T = n * ||Cxy||_F^2``, and the analytic Hall-Buckley-Eagleson (HBE)
weighted-chi^2 null via a regularized-incomplete-gamma survival) are LIFTED verbatim from the
verified reference ``runs/calib-study/calib_study.py`` (imported UNCHANGED by ``calib_study4``).
The block-conditional PERMUTATION null variant is lifted from ``calib_study4_correct.rcot_permcpt``
+ ``calib_study.z_blocks``.

Contract compatibility with the frozen slice
--------------------------------------------
- Same candidate/target layout: targets = 6 current KPIs, candidates = 14 = [8 params, 6 lagged
  KPIs]; the conditioning set Z for candidate i is the OTHER 13 candidates -- IDENTICAL to what
  ``discovery.py`` conditions on (``np.delete(xs, i, axis=1)``).
- Same selection: per-target Benjamini-Hochberg FDR at ``q`` over the ``m = 14`` candidate p-values
  (the exact ``bh_fdr_reject`` from ``discovery.py`` is imported and reused).
- Same degeneracy philosophy: a RELATIVE residual-variance-collapse guard fires when the candidate
  and target are (near-)fully explained by Z -> ``p = 1``, never selected (analogous to
  ``discovery.py``'s relative denominator guard).
- Same guard-fraction HALT: ``discover_graph_rcot`` HALTs when the residual-collapse guard fires on
  ``>= ceil(max_guard_fraction * 84) = 5`` of the 84 candidates -- the SAME mechanism and 0.05
  threshold as ``discovery.py``'s §11 guard-fraction HALT. (``discovery.py``'s degenerate-null HALT
  has no dangerous RCoT analogue; see the note at the end of ``discover_graph_rcot``.)
- Same persistence style (canonical JSON, atomic write, git sha, schema version, per-candidate
  moments) -- but written to ``discovery_rcot.json`` (never ``discovery.json``) so a later evaluate
  step can score it, and this module reads/compares NO truth.

Numerical note / deviation from the reference, flagged explicitly
-----------------------------------------------------------------
The reference ``rcot_test`` does not standardize its inputs (its synthetic testbeds were already
~unit-scale). Real E2 columns span wildly different scales (params ~+-100, KPIs ~+-40), which would
let the median-heuristic Z-bandwidth be dominated by the largest-scale column. So, exactly as
``discovery.py`` does, this module z-standardizes every candidate and target column ONCE up front
(population std, ddof=0) before the RCoT test.

This per-column z-standardization is a DELIBERATE, non-invariant preconditioning of the multivariate
conditioner Z -- NOT an affine no-op. The 13-dim Z bandwidth is a SINGLE joint Euclidean
median-heuristic ``sig`` over all columns at once (``_median_sigma`` on the (n, 13) block), so
rescaling each column by its own std genuinely REBALANCES that joint bandwidth across the mixed-scale
columns: without it the ~+-100 param columns dominate the median pairwise distance and swamp the
~+-40 KPI columns; with it every column contributes on a comparable scale. An adversarial review
verified this is NOT invariant for the analytic joint-Z p-value (per-column rescaling of Z shifts the
HBE weighted-chi^2 null; measured Delta p up to 0.307 on the same statistic). Only the 1-D candidate
and 1-D target legs are affine-invariant (their per-variable ``sig`` rescales with the data); the
multivariate-Z leg is not. The review further verified that standardization KEEPS/IMPROVES calibration
on the faithful E2 null (e.g. NULL_k3 FP@.05 0.093 -> 0.067 at n=400): so it stays because it is
beneficial and correct, not because it is numerically inert. This is the only intentional deviation
from the reference numerics, and it is intentional precisely because it is not a no-op.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from cdd_oran.e2slice import SCHEMA_VERSION
from cdd_oran.e2slice.dataset import (
    E2Rows,
    _atomic_write_text,
    _git_sha,
    canonical_json,
    guard_descendants,
    load_dataset,
)

# BH-FDR is method-agnostic selection; reuse the EXACT frozen helper (§7.2 of the pdCor protocol).
from cdd_oran.e2slice.discovery import bh_fdr_reject

# The candidate-graph layout: (num_kpis, num_params + num_kpis) = (6, 14). A local constant matching
# the frozen slice; imports no env truth. 48 param->KPI + 36 lagged KPI->KPI = 84 candidates.
_NUM_TARGETS = 6
_NUM_CANDIDATES = 14
_CANDIDATE_SHAPE = (_NUM_TARGETS, _NUM_CANDIDATES)

# Schema tag for the sibling artifact; kept distinct from the frozen discovery.json record.
RCOT_SCHEMA_VERSION = f"{SCHEMA_VERSION}.rcot"

# Full SHA of the frozen protocol commit (``docs: freeze E2 RCoT discovery protocol (008)``). Recorded
# in every discovery_rcot.json so the frozen contract provably predates any recovery result, and
# re-derived on load. Mirrors ``discovery.py``'s ``PROTOCOL_COMMIT``.
PROTOCOL_COMMIT = "eba381a195c4b341cc35e79b600f6aef815f54ea"

# The frozen score-method tag (recorded + re-derived on load).
FROZEN_SCORE_METHOD = "rcot_conditional_independence"

_NULL_ANALYTIC = "analytic_hbe"
_NULL_BLOCK_PERM = "block_perm"


@dataclass(frozen=True)
class RCoTDiscoveryConfig:
    """RCoT discovery constants. Defaults are the FROZEN §11 protocol values.

    NOT freeze-locked in ``__post_init__`` (so a truth-free smoke test may build a throwaway config,
    and the analytic_hbe sensitivity variant may be run directly): the CANONICAL persist path
    ``write_discovery_rcot`` ALWAYS uses ``frozen_config()``, and ``load_discovery_rcot`` re-derives
    every persisted constant and the ``protocol_commit`` against these module values -- so a retuned
    mask can never be persisted AND loaded under this ``protocol_commit``.

    ``dz`` is the number of RFF features used to represent the 13-dim conditioner Z. ``25`` is the
    value the verified calibration study (calib_study4) selected: near-nominal FP, FLAT in n, power
    ~0.98 on true parents. It is a plain config field, NOT a knob to be raised as a "fix": higher
    ``dz`` OVER-REJECTS at small n (more RFF capacity -> the residualization overfits the null
    coupling and the statistic inflates), which is exactly the failure mode we are avoiding. Leave it
    at 25 unless a fresh calibration study says otherwise.

    ``null_method``:
      * ``"block_perm"`` (FROZEN default / LIVE selector) -- block-conditional permutation null
        (permute the candidate WITHIN Z-neighbour blocks, recompute the RCoT statistic with a FIXED
        RFF basis). EXACT-nominal and flat in n (calib_study4 + shipping-path study); ~3.2 min/seed
        at n=4000 single-core. This is the human-approved primary null.
      * ``"analytic_hbe"`` -- analytic HBE weighted-chi^2 null. Fast (~0.8 s/seed) but mildly liberal
        and its liberality GROWS with n on strong-sibling nulls (FP 0.07-0.16). Retained as a labelled
        SENSITIVITY variant only, never the frozen selector.
    """

    dz: int = 25                       # RFF features for the Z conditioner (calib_study4 value)
    dxy: int = 5                       # RFF features for candidate/target (reference default)
    ridge: float = 1e-6                # ridge for the fz residualization least-squares
    q: float = 0.05                    # per-target BH-FDR level, m = 14
    null_method: str = _NULL_BLOCK_PERM  # FROZEN primary; "analytic_hbe" = sensitivity variant only
    block_perm_reps: int = 99          # permutations for the block_perm null
    block_size: int = 25               # Z-neighbour block length for the block_perm null
    residual_epsilon: float = 1e-12    # relative residual-collapse guard threshold
    max_guard_fraction: float = 0.05   # §11 guard-fraction HALT: halt at >= ceil(0.05*84)=5 of 84
    rng_seed: int = 0                  # base RNG seed for deterministic RFF draws

    def validate(self) -> None:
        if self.null_method not in (_NULL_ANALYTIC, _NULL_BLOCK_PERM):
            raise ValueError(
                f"RCoTDiscoveryConfig: null_method must be {_NULL_ANALYTIC!r} or "
                f"{_NULL_BLOCK_PERM!r}, got {self.null_method!r}"
            )
        if self.dz < 1 or self.dxy < 1:
            raise ValueError("RCoTDiscoveryConfig: dz and dxy must be >= 1")


def frozen_config() -> RCoTDiscoveryConfig:
    """The frozen §11 protocol config (used by the canonical persist path ``write_discovery_rcot``)."""
    return RCoTDiscoveryConfig()


@dataclass(frozen=True)
class RCoTDiscoveryResult:
    statistic: npt.NDArray[np.float64]     # (6, 14) RCoT statistic T = n*||Cxy||^2, NaN where guarded
    pvalues: npt.NDArray[np.float64]       # (6, 14) conditional-independence p-values, 1.0 where guarded
    guarded_mask: npt.NDArray[np.int64]    # (6, 14) {0,1} residual-collapse guard fires
    binary_mask: npt.NDArray[np.int64]     # (6, 14) {0,1} per-target BH-FDR selection
    n_rows: int
    null_method: str
    mean_candidates: npt.NDArray[np.float64]  # (14,)
    std_candidates: npt.NDArray[np.float64]   # (14,)
    mean_targets: npt.NDArray[np.float64]     # (6,)
    std_targets: npt.NDArray[np.float64]      # (6,)


# ---------------------------------------------------------------------------
# RCoT numerics -- LIFTED VERBATIM from runs/calib-study/calib_study.py.
# (dist geometry, median heuristic, RFF, regularized-incomplete-gamma survival.)
# ---------------------------------------------------------------------------
def _dist_matrix_1d(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.abs(v[:, None] - v[None, :])


def _dist_matrix_euclidean(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    sq = np.einsum("ij,ij->i", z, z)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (z @ z.T)
    np.maximum(d2, 0.0, out=d2)
    d = np.sqrt(d2)
    np.fill_diagonal(d, 0.0)
    return d


def _median_sigma(V: npt.NDArray[np.float64]) -> float:
    """Median-heuristic bandwidth over the first <=200 rows (verbatim: calib_study._median_sigma)."""
    n = V.shape[0]
    m = min(n, 200)
    Vs = V[:m]
    d = _dist_matrix_euclidean(Vs) if Vs.ndim == 2 and Vs.shape[1] > 1 else _dist_matrix_1d(Vs.ravel())
    iu = np.triu_indices(m, 1)
    med = np.median(d[iu])
    return med if med > 0 else 1.0


def _rff(V: npt.NDArray[np.float64], D: int, sig: float, rng: np.random.Generator) -> npt.NDArray[np.float64]:
    """Random Fourier features (verbatim: calib_study._rff). Mean-centered columns."""
    if V.ndim == 1:
        V = V[:, None]
    dim = V.shape[1]
    W = rng.normal(size=(dim, D)) / sig
    b = rng.uniform(0, 2 * np.pi, size=D)
    f = math.sqrt(2.0 / D) * np.cos(V @ W + b)
    f = f - f.mean(0)
    return f


def _gamma_sf(x: float, k_shape: float, theta: float) -> float:
    """Survival of Gamma(shape=k_shape, scale=theta) (verbatim: calib_study._gamma_sf)."""
    if x <= 0:
        return 1.0
    a = k_shape
    xx = x / theta
    gln = math.lgamma(a)
    if xx < a + 1.0:  # series for the regularized lower incomplete gamma P(a, xx)
        ap = a
        summ = 1.0 / a
        delv = summ
        for _ in range(1000):
            ap += 1.0
            delv *= xx / ap
            summ += delv
            if abs(delv) < abs(summ) * 1e-14:
                break
        P = summ * math.exp(-xx + a * math.log(xx) - gln)
        return 1.0 - P
    else:            # continued fraction for the upper tail Q
        b0 = xx + 1.0 - a
        c = 1e300
        d = 1.0 / b0
        h = d
        for i in range(1, 1000):
            an = -i * (i - a)
            b0 += 2.0
            d = an * d + b0
            if abs(d) < 1e-300:
                d = 1e-300
            c = b0 + an / c
            if abs(c) < 1e-300:
                c = 1e-300
            d = 1.0 / d
            delv = d * c
            h *= delv
            if abs(delv - 1.0) < 1e-14:
                break
        Q = math.exp(-xx + a * math.log(xx) - gln) * h
        return Q


def _rff_residuals(
    X: npt.NDArray[np.float64],
    Y: npt.NDArray[np.float64],
    Z: npt.NDArray[np.float64],
    dxy: int,
    dz: int,
    ridge: float,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Draw RFF(X), RFF(Y), RFF(Z) and ridge-residualize fx, fy on fz.

    Returns ``(fx, fy, rx, ry)`` (all mean-centered). Faithful to the residualization block of
    ``calib_study.rcot_test``.
    """
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
    n = X.shape[0]
    fx = _rff(X, dxy, _median_sigma(X), rng)
    fy = _rff(Y, dxy, _median_sigma(Y), rng)
    fz = _rff(Z, dz, _median_sigma(Z), rng)
    G = fz.T @ fz + ridge * n * np.eye(fz.shape[1])
    Bx = np.linalg.solve(G, fz.T @ fx)
    By = np.linalg.solve(G, fz.T @ fy)
    rx = fx - fz @ Bx
    ry = fy - fz @ By
    return fx, fy, rx, ry


def _rcot_statistic(rx: npt.NDArray[np.float64], ry: npt.NDArray[np.float64]) -> tuple[
    float, npt.NDArray[np.float64], npt.NDArray[np.float64]
]:
    """RCoT cross-covariance statistic ``T = n * ||Cxy||_F^2`` and the centered product matrix.

    Verbatim from the statistic block of ``calib_study.rcot_test``.
    """
    n = rx.shape[0]
    prod = np.einsum("ni,nj->nij", rx, ry).reshape(n, -1)  # per-row kron(rx, ry)
    cmean = prod.mean(0)
    stat = n * float(cmean @ cmean)
    prodc = prod - cmean
    return stat, prod, prodc


def _hbe_pvalue(stat: float, prodc: npt.NDArray[np.float64]) -> float:
    """Analytic HBE weighted-chi^2 p-value from the centered products (verbatim: calib_study)."""
    n = prodc.shape[0]
    Sigma = (prodc.T @ prodc) / n
    w = np.linalg.eigvalsh(Sigma)
    w = w[w > 1e-12]
    if w.size == 0:
        return 1.0
    mean = float(w.sum())
    var = float(2.0 * (w ** 2).sum())
    g = var / (2.0 * mean)
    kdf = 2.0 * mean * mean / var
    return _gamma_sf(stat, kdf / 2.0, 2.0 * g)


def _residual_collapse(
    fx: npt.NDArray[np.float64], fy: npt.NDArray[np.float64],
    rx: npt.NDArray[np.float64], ry: npt.NDArray[np.float64], eps: float
) -> bool:
    """RELATIVE residual-collapse guard: candidate/target (near-)fully explained by Z.

    Fires when ``sqrt((||rx||^2/||fx||^2) * (||ry||^2/||fy||^2)) <= eps`` -- the RFF residuals of BOTH
    the candidate and the target have collapsed relative to their raw RFF energy (the RCoT analogue of
    ``discovery.py``'s relative denominator guard). Guarded pairs get ``p = 1`` and are never selected.
    """
    ssx = float((fx * fx).sum())
    ssy = float((fy * fy).sum())
    if ssx <= 0.0 or ssy <= 0.0:
        return True
    ratio_x = float((rx * rx).sum()) / ssx
    ratio_y = float((ry * ry).sum()) / ssy
    return math.sqrt(max(ratio_x, 0.0) * max(ratio_y, 0.0)) <= eps


def z_blocks(Zs: npt.NDArray[np.float64], L: int = 25) -> list[npt.NDArray[np.int64]]:
    """Greedy nearest-neighbour tour of Z split into contiguous blocks of size ``L``.

    Verbatim from ``calib_study.z_blocks``. Consecutive tour entries have similar Z, so within-block
    permutation preserves the candidate|Z coupling (the basis of the block-conditional perm null).
    """
    n = Zs.shape[0]
    D = _dist_matrix_euclidean(Zs).copy()
    np.fill_diagonal(D, np.inf)
    order = np.empty(n, dtype=int)
    cur = 0
    order[0] = 0
    for t in range(1, n):
        D[:, cur] = np.inf
        cur = int(np.argmin(D[cur]))
        order[t] = cur
    return [order[i:i + L] for i in range(0, n, L)]


def rcot_pvalue_analytic(
    cand: npt.NDArray[np.float64], target: npt.NDArray[np.float64], Z: npt.NDArray[np.float64],
    cfg: RCoTDiscoveryConfig, seed: int
) -> tuple[float, float, bool]:
    """RCoT analytic-HBE p-value for ``cand ⟂ target | Z``. Returns ``(statistic, pvalue, guarded)``."""
    rng = np.random.default_rng(seed)
    fx, fy, rx, ry = _rff_residuals(cand, target, Z, cfg.dxy, cfg.dz, cfg.ridge, rng)
    if _residual_collapse(fx, fy, rx, ry, cfg.residual_epsilon):
        return math.nan, 1.0, True
    stat, _prod, prodc = _rcot_statistic(rx, ry)
    return stat, _hbe_pvalue(stat, prodc), False


def rcot_pvalue_block_perm(
    cand: npt.NDArray[np.float64], target: npt.NDArray[np.float64], Z: npt.NDArray[np.float64],
    cfg: RCoTDiscoveryConfig, seed: int
) -> tuple[float, float, bool]:
    """RCoT block-conditional PERMUTATION p-value. Returns ``(statistic, pvalue, guarded)``.

    Permutes the candidate WITHIN Z-neighbour blocks (preserving cand|Z), recomputing the RCoT
    statistic each time with a FIXED RFF basis (same ``seed``) -- lifted from
    ``calib_study4_correct.rcot_permcpt``. As in the reference, the RFF basis is regenerated from
    ``default_rng(seed)`` on every statistic evaluation (identical W, b, and draw order fx->fy->fz),
    so permuting the candidate input changes only fx while the projection basis is held fixed.
    """
    cand = np.asarray(cand, dtype=float)
    target = np.asarray(target, dtype=float)
    n = cand.shape[0]
    # Standardized Z is used ONLY to build the neighbour blocks (matches calib_study4).
    Zs = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1.0)
    blocks = z_blocks(Zs, L=cfg.block_size)

    def _eval(cand_vec: npt.NDArray[np.float64]) -> tuple[float, bool]:
        rng = np.random.default_rng(seed)  # fixed basis: identical W,b every call (matches reference)
        fx, fy, rx, ry = _rff_residuals(cand_vec, target, Z, cfg.dxy, cfg.dz, cfg.ridge, rng)
        collapse = _residual_collapse(fx, fy, rx, ry, cfg.residual_epsilon)
        stat, _prod, _prodc = _rcot_statistic(rx, ry)
        return stat, collapse

    stat0, guard0 = _eval(cand)
    if guard0:
        return math.nan, 1.0, True

    perm_rng = np.random.default_rng(seed * 7 + 99991)
    cnt = 0
    for _ in range(cfg.block_perm_reps):
        perm = np.arange(n)
        for blk in blocks:
            perm[blk] = blk[perm_rng.permutation(blk.size)]
        st, _g = _eval(cand[perm])
        if st >= stat0:
            cnt += 1
    pval = (1 + cnt) / (cfg.block_perm_reps + 1)
    return stat0, pval, False


def _pair_seed(base: int, j: int, i: int) -> int:
    """Deterministic, well-mixed per-(target j, candidate i) RFF seed."""
    return int(np.random.SeedSequence(entropy=int(base), spawn_key=(int(j), int(i))).generate_state(1)[0])


def _standardize(a: npt.NDArray[np.float64]) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]:
    """z-standardize columns (population std, ddof=0); reject zero-variance columns (matches discovery.py)."""
    mean = a.mean(axis=0)
    std = a.std(axis=0)
    if (std == 0.0).any():
        raise ValueError(
            f"discovery_rcot: zero-variance column(s) at {np.where(std == 0.0)[0].tolist()}; cannot "
            "z-standardize."
        )
    return (a - mean) / std, mean, std


def discover_graph_rcot(
    rows: E2Rows, cfg: RCoTDiscoveryConfig | None = None
) -> RCoTDiscoveryResult:
    """Select the E2 candidate graph from the whole dataset via RCoT (no labels, no env truth).

    For each of the 84 (target KPI j, candidate i) pairs, test ``candidate_i ⟂ target_j | Z`` with
    ``Z = the other 13 candidates`` (identical conditioning to ``discovery.py``), using the RCoT
    null selected by ``cfg.null_method``. Selection is per-target BH-FDR at ``cfg.q`` (m = 14),
    reusing the frozen ``bh_fdr_reject``. Guarded pairs (residual collapse) carry ``p = 1`` and are
    never selected.
    """
    cfg = cfg if cfg is not None else RCoTDiscoveryConfig()
    cfg.validate()
    x = np.concatenate([rows.x_params, rows.x_kpis], axis=1).astype(np.float64)  # (n, 14)
    y = rows.y_kpis.astype(np.float64)                                           # (n, 6)
    n, d = x.shape
    k = y.shape[1]
    if (d, k) != (_NUM_CANDIDATES, _NUM_TARGETS):
        raise ValueError(
            f"discovery_rcot: candidate/target layout ({k}, {d}) != {_CANDIDATE_SHAPE}"
        )
    if n < 4:
        raise ValueError(f"discovery_rcot: need n >= 4, got n = {n}")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("discovery_rcot: dataset rows contain non-finite values")

    xs, mean_c, std_c = _standardize(x)
    ys, mean_t, std_t = _standardize(y)

    pvalue_fn = rcot_pvalue_analytic if cfg.null_method == _NULL_ANALYTIC else rcot_pvalue_block_perm

    statistic = np.full((k, d), np.nan, dtype=np.float64)
    pvalues = np.ones((k, d), dtype=np.float64)
    guarded = np.zeros((k, d), dtype=np.int64)

    for i in range(d):
        z = np.delete(xs, i, axis=1)   # 13-dim conditioner = the OTHER 13 candidates
        xi = xs[:, i]
        for j in range(k):
            seed = _pair_seed(cfg.rng_seed, j, i)
            stat, pval, is_guarded = pvalue_fn(xi, ys[:, j], z, cfg, seed)
            if is_guarded:
                guarded[j, i] = 1
                continue
            statistic[j, i] = stat
            pvalues[j, i] = pval

    if not np.isfinite(pvalues).all():
        raise ValueError("discovery_rcot: non-finite p-value outside the guard path")

    # §11 guard-fraction HALT -- PARITY with discovery.py: halt when the residual-collapse guard
    # fires on >= ceil(max_guard_fraction * 84) = 5 of the 84 candidates (same mechanism + 0.05
    # threshold as the frozen sibling). A guard rate this high means the distance geometry /
    # standardization is degenerate, not that 5+ edges are genuinely unidentifiable.
    n_candidates = k * d
    guard_halt = math.ceil(cfg.max_guard_fraction * n_candidates)
    n_guarded = int(guarded.sum())
    if n_guarded >= guard_halt:
        raise ValueError(
            f"discovery_rcot: residual-collapse guard fired on {n_guarded} of {n_candidates} "
            f"candidates >= HALT threshold {guard_halt}; diagnose the RFF residualization / "
            "standardization (§11 guard-fraction HALT, parity with discovery.py)."
        )

    binary = np.zeros((k, d), dtype=np.int64)
    for j in range(k):
        binary[j] = bh_fdr_reject(pvalues[j], cfg.q)
    binary[guarded == 1] = 0  # fail-closed: guarded is never selected

    # discovery.py additionally HALTs on a DEGENERATE (zero-variance) permutation null for a SELECTED
    # candidate: there, a marginal permutation that destroys the candidate|Z coupling can collapse the
    # null and let a tiny statistic reject -- the exact failure this RCoT method exists to avoid, so a
    # degenerate null there is dangerous (a false selection) and must HALT. Neither RCoT null has a
    # clean analogue that is dangerous in that direction:
    #   * analytic_hbe -- no permutation null at all; the only degenerate case (all HBE eigenweights
    #     <= 1e-12 -> empty spectrum) is already returned as p = 1 by ``_hbe_pvalue`` (never selected).
    #   * block_perm  -- a degenerate null (every within-Z-block permutation reproduces stat0) makes
    #     ``st >= stat0`` hold for all reps, so pval -> 1.0: a degenerate block-perm null is
    #     fail-SAFE (never selects), the opposite of discovery.py's fail-dangerous case. No HALT needed.

    return RCoTDiscoveryResult(
        statistic=statistic,
        pvalues=pvalues,
        guarded_mask=guarded,
        binary_mask=binary,
        n_rows=n,
        null_method=cfg.null_method,
        mean_candidates=mean_c,
        std_candidates=std_c,
        mean_targets=mean_t,
        std_targets=std_t,
    )


# --- persistence (mirrors discovery.py; UNLOCKED, truth-free) ----------------
_DISCOVER_RCOT_DESCENDANTS = ("recovery_rcot.json",)


def _json_num(v: float) -> float | None:
    return None if (isinstance(v, float) and math.isnan(v)) else float(v)


def _grid(arr: npt.NDArray[np.float64]) -> list[list[float | None]]:
    return [[_json_num(float(v)) for v in row] for row in arr]


def _grid_array(grid: list[list[float | None]]) -> npt.NDArray[np.float64]:
    return np.asarray(
        [[np.nan if v is None else float(v) for v in row] for row in grid], dtype=np.float64
    )


def build_rcot_discovery_record(
    result: RCoTDiscoveryResult, dataset_hash: str, cfg: RCoTDiscoveryConfig
) -> dict[str, Any]:
    """Assemble the persisted, hash-bound RCoT discovery record (content_hash added last)."""
    sha, dirty = _git_sha()
    record: dict[str, Any] = {
        "schema_version": RCOT_SCHEMA_VERSION,
        "frozen": True,
        "protocol_commit": PROTOCOL_COMMIT,
        "dataset_hash": dataset_hash,
        "n_rows": int(result.n_rows),
        "seed": None,  # filled by write_discovery_rcot from the manifest
        "moments": {
            "mean_candidates": [float(v) for v in result.mean_candidates],
            "std_candidates": [float(v) for v in result.std_candidates],
            "mean_targets": [float(v) for v in result.mean_targets],
            "std_targets": [float(v) for v in result.std_targets],
        },
        "score_method": "rcot_conditional_independence",
        "null_method": result.null_method,
        "dz": int(cfg.dz),
        "dxy": int(cfg.dxy),
        "ridge": float(cfg.ridge),
        "q": float(cfg.q),
        "block_perm_reps": int(cfg.block_perm_reps),
        "block_size": int(cfg.block_size),
        "residual_epsilon": float(cfg.residual_epsilon),
        "rng_seed": int(cfg.rng_seed),
        "statistic": _grid(result.statistic),
        "pvalues": _grid(result.pvalues),
        "guarded_mask": result.guarded_mask.astype(int).tolist(),
        "binary_mask": result.binary_mask.astype(int).tolist(),
        "candidate_shape": list(result.binary_mask.shape),
        "git_sha": sha,
        "git_dirty": dirty,
    }
    return record


def _finalize_record(record: dict[str, Any]) -> dict[str, Any]:
    record["content_hash"] = hashlib.sha256(canonical_json(record).encode()).hexdigest()
    return record


def write_discovery_rcot(
    dataset_dir: str | Path, cfg: RCoTDiscoveryConfig | None = None, force: bool = False
) -> dict[str, Any]:
    """Discover the E2 graph via RCoT from the whole dataset and write ``discovery_rcot.json``.

    The canonical FROZEN persist path: when ``cfg`` is None it uses ``frozen_config()`` (block_perm,
    the §11 constants). Binds to the current dataset and reads/compares NO truth. Writes atomically. A
    non-frozen ``cfg`` may be passed for a throwaway smoke run, but its mask will NOT re-load via
    ``load_discovery_rcot`` (which re-derives every frozen constant + the protocol_commit).
    """
    cfg = cfg if cfg is not None else frozen_config()
    out = Path(dataset_dir)
    rows, manifest = load_dataset(out)
    result = discover_graph_rcot(rows, cfg)
    record = build_rcot_discovery_record(result, manifest["dataset_hash"], cfg)
    record["seed"] = int(manifest["config"]["seed"])
    record = _finalize_record(record)
    guard_descendants(out, _DISCOVER_RCOT_DESCENDANTS, force, "discover_rcot")
    _atomic_write_text(out / "discovery_rcot.json", json.dumps(record, indent=2, sort_keys=True))
    return record


def load_discovery_rcot(dataset_dir: str | Path) -> dict[str, Any]:
    """Load ``discovery_rcot.json`` and verify the content hash, frozen constants, and shape contract.

    FROZEN, fail-closed (mirrors ``discovery.py``'s ``load_discovery``): verifies the content hash,
    re-derives every frozen §11 constant and the ``protocol_commit`` against this module's values
    (refusing to load on any mismatch -- so a retuned mask cannot masquerade under this protocol),
    checks the persisted grids are the (6, 14) shape, and that guarded cells are never selected.
    """
    disc_file = Path(dataset_dir) / "discovery_rcot.json"
    record = json.loads(disc_file.read_text())

    if record.get("schema_version") != RCOT_SCHEMA_VERSION:
        raise ValueError(
            f"{disc_file}: schema_version {record.get('schema_version')!r} != {RCOT_SCHEMA_VERSION!r}"
        )
    for key in ("dataset_hash", "score_method", "null_method", "statistic", "pvalues",
                "guarded_mask", "binary_mask", "candidate_shape", "content_hash",
                "protocol_commit", "frozen", "dz", "dxy", "ridge", "q", "block_perm_reps",
                "block_size", "residual_epsilon", "rng_seed"):
        if key not in record:
            raise ValueError(f"{disc_file}: missing required field '{key}'")

    stored = record["content_hash"]
    payload = {key: value for key, value in record.items() if key != "content_hash"}
    recomputed = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    if recomputed != stored:
        raise ValueError(f"{disc_file}: content_hash {stored} != recomputed {recomputed}")

    # Re-derive the frozen contract: refuse any mask whose protocol_commit or §11 constants differ
    # from this module's frozen values (mirrors discovery.py's frozen-constant re-derivation).
    fc = frozen_config()
    if record["protocol_commit"] != PROTOCOL_COMMIT:
        raise ValueError(
            f"{disc_file}: protocol_commit {record['protocol_commit']!r} != frozen {PROTOCOL_COMMIT!r}")
    if record["frozen"] is not True:
        raise ValueError(f"{disc_file}: frozen flag is {record['frozen']!r}, expected True")
    if record["score_method"] != FROZEN_SCORE_METHOD:
        raise ValueError(
            f"{disc_file}: score_method {record['score_method']!r} != frozen {FROZEN_SCORE_METHOD!r}")
    if record["null_method"] != fc.null_method:
        raise ValueError(
            f"{disc_file}: null_method {record['null_method']!r} != frozen {fc.null_method!r} "
            "(the frozen selector is block_perm; analytic_hbe is a sensitivity variant only)")
    for key, frozen_val in (("dz", fc.dz), ("dxy", fc.dxy), ("block_perm_reps", fc.block_perm_reps),
                            ("block_size", fc.block_size), ("rng_seed", fc.rng_seed)):
        if record[key] != frozen_val:
            raise ValueError(f"{disc_file}: {key} {record[key]!r} != frozen {frozen_val!r}")
    for key, frozen_val in (("ridge", fc.ridge), ("q", fc.q), ("residual_epsilon", fc.residual_epsilon)):
        if not math.isclose(float(record[key]), frozen_val, rel_tol=0.0, abs_tol=1e-18):
            raise ValueError(f"{disc_file}: {key} {record[key]!r} != frozen {frozen_val!r}")

    guarded = np.asarray(record["guarded_mask"], dtype=np.int64)
    mask = np.asarray(record["binary_mask"], dtype=np.int64)
    for name, arr in (("statistic", _grid_array(record["statistic"])),
                      ("pvalues", _grid_array(record["pvalues"])),
                      ("guarded_mask", guarded), ("binary_mask", mask)):
        if arr.shape != _CANDIDATE_SHAPE:
            raise ValueError(f"{disc_file}: {name} shape {arr.shape} != {_CANDIDATE_SHAPE}")
    if tuple(record["candidate_shape"]) != _CANDIDATE_SHAPE:
        raise ValueError(f"{disc_file}: candidate_shape != {list(_CANDIDATE_SHAPE)}")
    if (mask[guarded == 1] != 0).any():
        raise ValueError(f"{disc_file}: binary_mask is 1 on a guarded cell (must be fail-closed 0)")
    return record


def discovered_mask_array(record: dict[str, Any]) -> npt.NDArray[np.int64]:
    """The (6, 14) binary mask as an int array (evaluate-compatible with the frozen slice)."""
    return np.asarray(record["binary_mask"], dtype=np.int64)
