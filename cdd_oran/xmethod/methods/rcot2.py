"""xmethod adapter ``rcot2``: RCoT-v2 (``e2slice.discovery_rcot`` numerics + ``discovery_rcot_v2`` config, wrapped).

Per candidate: the frozen ``rcot_pvalue_block_perm`` loop (verbatim in ``block_perm_sequential``): RFF residualization on Z (dz = 25, dxy = 5, ridge 1e-6,
median-heuristic bandwidths), T = n ||Cxy||_F^2, block-conditional permutation null (Z-neighbour tour blocks of 25,
B = 299), relative residual-collapse guard. Columns are z-standardized once (as the frozen module). Per-pair RFF
seed = ``_pair_seed(rng_seed, j, i)`` with rng_seed derived from the citests tag (frozen: rng_seed = 0).
PRIMARY null (R-13, the authors' own): Strobl et al.'s RCoT approximate null "lpd4" = 1 - Lindsay-Pilla-Basak
cdf of the weighted chi^2 with weights = eigenvalues of the covariance of the residual feature products, evaluated at
T; on failure 1 - HBE; clipped at 0 (as RCIT R/RCoT.R, github ericstrobl/RCIT 7a7fb2b). LPB4 / HBE come from the
MIT-licensed ``momentchi2`` 0.1.8 (PyPI; D. Bodenham's Python port of his R package). Features: the AUTHORS'
default num_f = 100 conditioning features (RCIT; frozen E2 used dz = 25, now a cross-check), num_f2 = dxy = 5. The
RFF / residualization / statistic numerics stay the frozen e2slice ones (``_rff_residuals``, ``_rcot_statistic``,
eigenvalues > 1e-12 as the frozen ``_hbe_pvalue``), so the null is the authors' and the statistic the frozen port.
Cross-checks: ``block_perm_config()`` (frozen block permutation, dz 25, B raised from native 299 to sequential
9999 for pooled-BY resolution, R-9) and ``native_config()`` (block permutation, dz 25, B 299 fixed: equals the
frozen function).
Frozen guard-fraction HALT kept (>= ceil(.05 m) guarded -> STOP). Native rule: per-target BH q = .05.
Memory: ``z_blocks`` forms an n x n distance matrix (+ a copy) per candidate: ~2 x 8 n^2 bytes.
"""
from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from cdd_oran.e2slice.discovery import bh_fdr_reject
from cdd_oran.e2slice.discovery_rcot import (
    _pair_seed,
    _rcot_statistic,
    _residual_collapse,
    _rff_residuals,
    _standardize,
    rcot_pvalue_analytic,
    z_blocks,
)
from cdd_oran.e2slice.discovery_rcot_v2 import frozen_config_v2
from cdd_oran.xmethod.methods._citests_common import (
    B_MAX,
    BC_H,
    CITestBase,
    Prepared,
    SequentialP,
    method_seed,
    progress,
)


def _momentchi2():
    """momentchi2 0.1.8 calls ``np.math.factorial`` (removed in numpy 2; ``np.math`` was the stdlib ``math``).
    Compatibility shim, local to momentchi2: its module-level ``np`` becomes a proxy with ``math`` added; numpy's
    own namespace is untouched."""
    import types

    import momentchi2
    import momentchi2.utilities as mu

    if not hasattr(mu.np, "math"):
        proxy = types.ModuleType("numpy_with_math")
        proxy.__getattr__ = lambda name: getattr(np, name)
        proxy.math = math
        mu.np = proxy
    return momentchi2.hbe, momentchi2.lpb4


def rcot_pvalue_lpd4(cand, target, Z, cfg, seed):
    """``discovery_rcot.rcot_pvalue_analytic`` with the HBE null replaced by RCIT's "lpd4" (LPB4, HBE fallback).
    Returns ``(statistic, pvalue, guarded, approximation)``."""
    hbe, lpb4 = _momentchi2()

    rng = np.random.default_rng(seed)
    fx, fy, rx, ry = _rff_residuals(cand, target, Z, cfg.dxy, cfg.dz, cfg.ridge, rng)
    if _residual_collapse(fx, fy, rx, ry, cfg.residual_epsilon):
        return math.nan, 1.0, True, "guard"
    stat, _prod, prodc = _rcot_statistic(rx, ry)
    n = prodc.shape[0]
    w = np.linalg.eigvalsh((prodc.T @ prodc) / n)
    w = w[w > 1e-12]
    if w.size == 0:
        return stat, 1.0, False, "empty"
    how = "lpb4"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = 1.0 - float(lpb4(w, float(stat)))
        if not np.isfinite(p):
            raise ValueError("lpb4 nan")
    except Exception:  # noqa: BLE001  (RCIT: try(lpb4) -> hbe on any error)
        how = "hbe_fallback"
        p = 1.0 - float(hbe(w, float(stat)))
    return stat, max(p, 0.0), False, how


def block_perm_sequential(cand, target, Z, cfg, seed, h):
    """``discovery_rcot.rcot_pvalue_block_perm`` verbatim, with the fixed-B loop replaced by Besag-Clifford
    stopping (``h=None``: the frozen fixed-B p). Returns ``(statistic, pvalue, guarded, draws)``."""
    cand = np.asarray(cand, dtype=float)
    target = np.asarray(target, dtype=float)
    n = cand.shape[0]
    Zs = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1.0)
    blocks = z_blocks(Zs, L=cfg.block_size)

    def _eval(cand_vec):
        rng = np.random.default_rng(seed)
        fx, fy, rx, ry = _rff_residuals(cand_vec, target, Z, cfg.dxy, cfg.dz, cfg.ridge, rng)
        collapse = _residual_collapse(fx, fy, rx, ry, cfg.residual_epsilon)
        stat, _prod, _prodc = _rcot_statistic(rx, ry)
        return stat, collapse

    stat0, guard0 = _eval(cand)
    if guard0:
        return math.nan, 1.0, True, 0
    perm_rng = np.random.default_rng(seed * 7 + 99991)
    seq = SequentialP(stat0, cfg.block_perm_reps, h)
    while not seq.done:
        perm = np.arange(n)
        for blk in blocks:
            perm[blk] = blk[perm_rng.permutation(blk.size)]
        st, _g = _eval(cand[perm])
        seq.add(st)
    return stat0, seq.p, False, seq.k


class RCoT2Method(CITestBase):
    name = "rcot2"
    version = "xm-rcot-v2-lpd4"
    method_key = 4

    def default_config(self) -> dict[str, Any]:
        c = frozen_config_v2()
        return {"null_method": "lpd4", "block_perm_reps": None, "bc_h": None, "dz": 100, "dxy": c.dxy,
                "block_size": c.block_size, "rng_seed": None}

    def block_perm_config(self) -> dict[str, Any]:
        c = frozen_config_v2()
        return {**self.default_config(), "null_method": c.null_method, "dz": c.dz, "block_perm_reps": B_MAX,
                "bc_h": BC_H}

    def native_config(self) -> dict[str, Any]:
        return {**self.block_perm_config(), "block_perm_reps": frozen_config_v2().block_perm_reps, "bc_h": None,
                "arm": "native"}

    def _test(self, prep: Prepared, data, cfg: dict[str, Any]) -> dict[str, Any]:
        base = frozen_config_v2()
        lpd4 = cfg["null_method"] == "lpd4"
        rcfg = type(base)(dz=cfg["dz"], dxy=cfg["dxy"], ridge=base.ridge, q=base.q,
                          null_method="analytic_hbe" if lpd4 else cfg["null_method"],
                          block_perm_reps=cfg["block_perm_reps"] or base.block_perm_reps, block_size=cfg["block_size"],
                          residual_epsilon=base.residual_epsilon, max_guard_fraction=base.max_guard_fraction,
                          rng_seed=cfg["rng_seed"] if cfg.get("rng_seed") is not None
                          else method_seed(data.seed, self.method_key))
        rcfg.validate()
        xs, _, _ = _standardize(prep.S)
        ys, _, _ = _standardize(prep.Y)
        m = len(prep.pairs)
        stat, p, guarded = np.full(m, np.nan), np.ones(m), np.zeros(m, dtype=bool)
        k_used = np.zeros(m, dtype=int)
        approx: dict[str, int] = {}
        for e, (i, j) in enumerate(prep.pairs):
            args = (xs[:, i], ys[:, j], prep.z_of(xs, i), rcfg, _pair_seed(rcfg.rng_seed, j, i))
            if lpd4:
                st, pv, g, how = rcot_pvalue_lpd4(*args)
                approx[how] = approx.get(how, 0) + 1
            elif rcfg.null_method == "analytic_hbe":
                st, pv, g = rcot_pvalue_analytic(*args)
            else:
                st, pv, g, k_used[e] = block_perm_sequential(*args, h=cfg["bc_h"])
            progress(cfg, e + 1, m)
            if g:
                guarded[e] = True
                continue
            stat[e], p[e] = st, pv
        halt = math.ceil(rcfg.max_guard_fraction * m)
        if int(guarded.sum()) >= halt:
            raise ValueError(f"rcot2: residual-collapse guard fired on {int(guarded.sum())} of {m} >= {halt} (HALT)")
        native = np.zeros(m, dtype=bool)
        for j in {j for _, j in prep.pairs}:
            es = [e for e, (_, jj) in enumerate(prep.pairs) if jj == j]
            native[es] = bh_fdr_reject(p[es], rcfg.q).astype(bool)
        native[guarded] = False
        p[guarded] = np.nan                     # audit L2: not testable, outside the BY families (native rule: p = 1)
        return {"score": stat, "p": p, "declared_native": native,
                "not_testable": {int(e): "guard: residual collapse" for e in np.nonzero(guarded)[0]},
                "notes": {"n_guarded": int(guarded.sum()), "rng_seed": rcfg.rng_seed, "mc_draws": k_used.tolist(),
                          "null": cfg["null_method"], "approximations": approx,
                          "native_rule": "per-target BH q=.05 over the target's candidates"}}
