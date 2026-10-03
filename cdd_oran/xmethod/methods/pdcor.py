"""xmethod adapter ``pdcor``: U-centred partial distance correlation (``e2slice.discovery``, numerics wrapped).

PRIMARY null (R-13 / Q9, the authors' own): Szekely & Rizzo (2014, Ann. Statist. 42(6), sec 4.3, 5.1) give no
non-permutation test; their implementation (R energy 1.7-12 ``pdcov.test``; python dcor 0.7
``partial_distance_covariance_test``) permutes the PROJECTED matrix P_z(x) (``Pxz[i, i]``) with statistic
T = n <Pxz, Pyz>, one-sided (large T = dependence). Here: the same, with Besag-Clifford B 9999 (R-9; energy counts
replicates > T, here >= T, the conservative tie rule), score = signed pdCor (the authors' estimate). Caveat (paper
sec 4.2): pdCor = 0 is NOT equivalent to conditional independence outside the Gaussian case.
Cross-check (config ``null="x_perm"``, ``frozen_config()``): the frozen E2 null below.

Score: the SIGNED pdCor(source, target; Z) under proj_perm (primary); |pdCor| under the x_perm cross-check. From
the frozen helpers (``u_center``, ``dist_matrix_1d``, ``dist_matrix_euclidean``, ``partial_distance_correlation``:
alpha = 1, relative denominator guard 1e-12). Inputs are z-standardized first (source, target and every Z column;
frozen E2): energy / dcor use raw Z, so the Euclidean metric on Z differs when |Z| > 1 (F6). Ruling R-40: pdcor is
reported as a DEPENDENCE test (pdCor = 0 is not CI), not a CI test. x_perm null:
the frozen per-candidate MARGINAL permutation of the source, one RNG stream per (target j, source i) exactly as
the frozen loop (``SeedSequence(entropy=base, spawn_key=(j, i))``), base derived from the citests tag.
F6 deviation (R-9): B raised from native 999 to sequential 9999 (Besag-Clifford, h = 20) for pooled-BY resolution;
``native_config()`` = B 999 without stopping (fidelity-gate check only).

Execution change (not a method change): a permuted source's U-centred matrix is the permuted U-centred matrix
(U-centring is permutation-equivariant), so the loop gathers ``A[perm][:, perm]`` instead of recomputing distances
and U-centring, and uses <Pxz_p, Pxz_p> = <A, A> - <A_p, C>^2 / <C, C>. Equal to the frozen loop up to floating
point summation order (F2 check: ``frozen_loop_pvalue``). O(n^2) memory (about 5 n x n float64 matrices live) and
O(B n^2) time per candidate: not scalable beyond n ~ 4000.

Frozen HALTs kept: guard fires on >= ceil(.05 m) candidates -> STOP. Native rule: per-target BH at q = .05
(``bh_fdr_reject``), kept as ``declared_native``.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from cdd_oran.e2slice.discovery import (
    FROZEN_B_PERM,
    FROZEN_DENOMINATOR_EPSILON,
    FROZEN_MAX_GUARD_FRACTION,
    FROZEN_Q,
    _project_out,
    bh_fdr_reject,
    dist_matrix_1d,
    dist_matrix_euclidean,
    partial_distance_correlation,
    u_center,
    u_inner,
)
from cdd_oran.xmethod.methods._citests_common import (
    B_MAX,
    BC_H,
    CITestBase,
    Prepared,
    SequentialP,
    method_seed,
    progress,
)


def _zstd(a: np.ndarray, what: str) -> np.ndarray:
    sd = a.std(axis=0)
    if (sd == 0).any():
        raise ValueError(f"pdcor: zero-variance {what} column(s) {np.nonzero(sd == 0)[0].tolist()}")
    return (a - a.mean(axis=0)) / sd


def _perm_null(cand_u, cand_self, cond_u, cond_self, pyz, vb_proj, rng, seq: SequentialP) -> list[float]:
    """|pdCor| of marginal permutations of the source until ``seq`` stops (fast, equivalent form of the frozen
    loop; same permutation stream)."""
    n = cand_u.shape[0]
    c_pyz = u_inner(cond_u, pyz)
    null = []
    while not seq.done:
        perm = rng.permutation(n)
        a_p = cand_u[np.ix_(perm, perm)]
        a_c = u_inner(a_p, cond_u)
        va = cand_self - a_c * a_c / cond_self
        den = math.sqrt(va * vb_proj) if va > 0 else 0.0
        num = u_inner(a_p, pyz) - (a_c / cond_self) * c_pyz
        null.append(abs(num / den) if den > 0.0 else 0.0)
        seq.add(null[-1])
    return null


def _proj_perm_null(pxz, pyz, rng, seq: SequentialP) -> None:
    """Authors' null: n <Pxz[perm][:, perm], Pyz> until ``seq`` stops."""
    n = pxz.shape[0]
    while not seq.done:
        perm = rng.permutation(n)
        seq.add(n * u_inner(pxz[np.ix_(perm, perm)], pyz))


def frozen_loop_null(xi, cond_u, cond_self, pyz, vb_proj, rng, b_perm):
    """The frozen ``discover_graph`` permutation loop verbatim (reference for the F2 equivalence check)."""
    n = xi.shape[0]
    null = np.empty(b_perm)
    for b in range(b_perm):
        cand_u_p = u_center(dist_matrix_1d(xi[rng.permutation(n)]))
        pxz_p = _project_out(cand_u_p, cond_u, cond_self, u_inner(cand_u_p, cond_u))
        va_proj_p = u_inner(pxz_p, pxz_p)
        denom_p = math.sqrt(va_proj_p * vb_proj)
        null[b] = abs(u_inner(pxz_p, pyz) / denom_p) if denom_p > 0.0 else 0.0
    return null


class PDCorMethod(CITestBase):
    name = "pdcor"
    version = "xm-pdcor-v2-projperm"
    method_key = 3

    def default_config(self) -> dict[str, Any]:
        return {"null": "proj_perm", "b_perm": B_MAX, "bc_h": BC_H,
                "denominator_epsilon": FROZEN_DENOMINATOR_EPSILON, "permutation_seed": None}

    def frozen_config(self) -> dict[str, Any]:
        """Cross-check: the frozen E2 null (permute the source, re-project, two-sided |pdCor|), B 9999 BC."""
        return {**self.default_config(), "null": "x_perm"}

    def native_config(self) -> dict[str, Any]:
        return {**self.frozen_config(), "b_perm": FROZEN_B_PERM, "bc_h": None, "arm": "native"}

    def _test(self, prep: Prepared, data, cfg: dict[str, Any], _loop=None) -> dict[str, Any]:
        xs, ys = _zstd(prep.S, "source"), _zstd(prep.Y, "target")
        n = xs.shape[0]
        if n < 4:
            raise ValueError("pdcor: need n >= 4")
        eps, b_perm = float(cfg["denominator_epsilon"]), int(cfg["b_perm"])
        base = cfg["permutation_seed"]
        if base is None:
            base = method_seed(data.seed, self.method_key)
        m = len(prep.pairs)
        signed, score, p = np.full(m, np.nan), np.full(m, np.nan), np.ones(m)
        guarded = np.zeros(m, dtype=bool)
        null_var = np.full(m, np.nan)
        k_used = np.zeros(m, dtype=int)
        by_i: dict[int, list[int]] = {}
        for e, (i, _) in enumerate(prep.pairs):
            by_i.setdefault(i, []).append(e)
        for i, es in by_i.items():
            cand_u = u_center(dist_matrix_1d(xs[:, i]))
            cand_self = u_inner(cand_u, cand_u)
            cond_u = u_center(dist_matrix_euclidean(prep.z_of(xs, i)))
            cond_self = u_inner(cond_u, cond_u)
            if cond_self <= 0.0:
                raise ValueError(f"pdcor: degenerate conditioning self-norm for source {i}")
            for e in es:
                j = prep.pairs[e][1]
                tgt_u = u_center(dist_matrix_1d(ys[:, j]))
                tgt_self = u_inner(tgt_u, tgt_u)
                s_ij, pxz, pyz, _va, vb_proj, g = partial_distance_correlation(
                    cand_u, cand_self, tgt_u, tgt_self, cond_u, cond_self, eps)
                del tgt_u
                if g:
                    guarded[e] = True
                    continue
                signed[e] = s_ij
                rng = np.random.default_rng(np.random.SeedSequence(entropy=base, spawn_key=(int(j), int(i))))
                if cfg.get("null", "x_perm") == "proj_perm":
                    score[e] = s_ij
                    seq = SequentialP(n * u_inner(pxz, pyz), b_perm, cfg["bc_h"])
                    _proj_perm_null(pxz, pyz, rng, seq)
                    p[e], k_used[e] = seq.p, seq.k
                    progress(cfg, int(np.count_nonzero(k_used)), m)
                    continue
                score[e] = abs(s_ij)
                seq = SequentialP(score[e], b_perm, cfg["bc_h"])
                if _loop == "frozen":
                    null = frozen_loop_null(xs[:, i], cond_u, cond_self, pyz, vb_proj, rng, b_perm)
                    for v in null:
                        if seq.add(v):
                            break
                else:
                    null = _perm_null(cand_u, cand_self, cond_u, cond_self, pyz, vb_proj, rng, seq)
                null_var[e] = float(np.var(null))
                p[e], k_used[e] = seq.p, seq.k
                progress(cfg, int(np.count_nonzero(k_used)), m)
        halt = math.ceil(FROZEN_MAX_GUARD_FRACTION * m)
        if int(guarded.sum()) >= halt:
            raise ValueError(f"pdcor: denominator guard fired on {int(guarded.sum())} of {m} >= {halt} (HALT)")
        native = np.zeros(m, dtype=bool)
        for j in {j for _, j in prep.pairs}:
            es = [e for e, (_, jj) in enumerate(prep.pairs) if jj == j]
            native[es] = bh_fdr_reject(p[es], FROZEN_Q).astype(bool)
        native[guarded] = False
        p[guarded] = np.nan                     # audit L2: not testable, outside the BY families (native rule: p = 1)
        return {"score": score, "p": p, "declared_native": native,
                "not_testable": {int(e): "guard: pdCor denominator" for e in np.nonzero(guarded)[0]},
                "notes": {"n_guarded": int(guarded.sum()), "signed_pdcor": [float(v) for v in signed],
                          "n_zero_var_null": int(np.sum(null_var == 0.0)), "mc_draws": k_used.tolist(),
                          "native_rule": "per-target BH q=.05 over the target's candidates"}}
