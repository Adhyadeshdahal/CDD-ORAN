"""Concrete inputs for the MSCR logic audit.

    PYTHONPATH=. .venv/Scripts/python.exe scratchpad/perf_mscr/audit_cases.py
"""
from __future__ import annotations

import numpy as np

from cdd_oran.discovery.mscr import MSCRConfig, _between_ss, discover_mscr, equal_count_labels

CFG = MSCRConfig(n_perm=299)


def case_tied_candidate_row_order():
    """A1: a candidate with ties is binned by ROW ORDER (stable argsort). A constant (never-actuated)
    parameter column then encodes time, and any time trend / serial dependence in y is 'discovered'."""
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.uniform(size=(n, 4))
    x[:, 0] = 0.5                                     # param 0 held fixed for the whole corpus
    trend = np.linspace(0.0, 1.0, n)                  # slow drift in the KPI (e.g. non-stationary load)
    y = np.column_stack([trend + 0.3 * rng.normal(size=n)])
    res = discover_mscr(x, y, n_params=4, seed=0, config=CFG)
    print("A1 constant param col vs trending KPI: p =", res.pvals[0].round(4), "declared =", res.declared[0])
    # block-wise actuation (E1-like single-parameter actuation): x0 only varies in the 2nd half, fixed at 0 before
    x2 = rng.uniform(size=(n, 4))
    x2[: n // 2, 1] = 0.0
    y2 = np.column_stack([np.r_[np.zeros(n // 2), np.ones(n // 2)] + 0.3 * rng.normal(size=n)])  # regime shift
    x2[:, 2] = 0.3
    res2 = discover_mscr(x2, y2, n_params=4, seed=0, config=CFG)
    print("A1b constant col 2 under a regime shift: p =", res2.pvals[0].round(4), "declared =", res2.declared[0])


def case_constant_target():
    """A2: a constant KPI. TSS = dot(y,y) - t^2/n is rounding noise, may be > 0, and S is then a ratio of
    rounding errors computed along two DIFFERENT summation paths (bincount vs reduceat)."""
    rng = np.random.default_rng(1)
    n = 1200
    x = rng.uniform(size=(n, 5))
    hits = 0
    for c in (0.1, 0.3, 0.7, 1.1, 3.3, 12.7, 1e3 + 0.1):
        y = np.full((n, 1), c)
        res = discover_mscr(x, y, n_params=5, seed=0, config=CFG)
        hits += int(res.declared.sum())
        print(f"A2 constant y={c}: s_star={res.s_star[0].round(3)} p={res.pvals[0].round(4)} "
              f"declared={res.declared[0].astype(int)}")
    print("A2 total declared edges on constant targets:", hits)


def case_large_offset():
    """A3: y = big offset + signal. BSS = sum T_k^2/n_k - T^2/n cancels catastrophically."""
    rng = np.random.default_rng(2)
    n = 1200
    x = rng.uniform(size=(n, 4))
    base = 0.05 * x[:, 0] + 0.05 * rng.normal(size=n)
    for off in (0.0, 1e4, 1e6, 1e8):
        res = discover_mscr(x, (base + off)[:, None], n_params=4, seed=0, config=CFG)
        print(f"A3 offset {off:g}: s_star={res.s_star[0].round(5)} p={res.pvals[0].round(4)}")


def case_ulp_tie_convention():
    """A4: identical partitions, observed path vs bank path, discrete y. The '>=' in the p-value must count
    exact-arithmetic ties; here it depends on rounding along two summation orders."""
    rng = np.random.default_rng(3)
    ns, nb = 200, 8
    worse = better = 0
    for _ in range(2000):
        y = rng.integers(0, 3, size=ns).astype(float) * 0.1   # discrete KPI levels
        lab = equal_count_labels(rng.uniform(size=ns), nb)
        obs = _between_ss(y, lab, nb)
        perm = np.argsort(lab, kind="stable")                  # same partition, bank's summation order
        sizes = np.bincount((np.arange(ns) * nb) // ns, minlength=nb)
        offsets = np.concatenate(([0], np.cumsum(sizes)[:-1]))
        gs = np.add.reduceat(y[perm], offsets)
        bank = float(np.sum(gs ** 2 / sizes) - y.sum() ** 2 / ns)
        worse += bank < obs
        better += bank > obs
    print(f"A4 same partition, bank < observed in {worse}/2000, bank > observed in {better}/2000 "
          "(bank<observed => an exact tie is NOT counted as >=)")


def case_nan():
    """A5: NaN in a target row silently turns the whole target into p=1 (all edges missed, no error)."""
    rng = np.random.default_rng(4)
    n = 1200
    x = rng.uniform(size=(n, 4))
    y = (x[:, 0] + 0.1 * rng.normal(size=n))[:, None]
    y[17, 0] = np.nan
    res = discover_mscr(x, y, n_params=4, seed=0, config=CFG)
    print("A5 one NaN in y: s_star", res.s_star[0], "p", res.pvals[0], "declared", res.declared[0])


def case_discrete_null_calibration():
    """A6: null calibration with a binary target and exact column-permutation nulls (how anti-conservative
    the rounding-resolved ties make p under a discrete KPI)."""
    from cdd_oran.discovery.mscr import _build_bank, _pval, _s_star
    rng = np.random.default_rng(5)
    n, reps = 480, 1000
    x = rng.uniform(size=(n, 3))
    y = (rng.uniform(size=n) < 0.05).astype(float)
    cfg = MSCRConfig(n_perm=199)
    denom, null, strata = _build_bank(x, y, cfg, np.random.default_rng(0))
    keep = np.ones(3, dtype=bool)
    keep[0] = False
    s_null = null[keep].max(axis=0)
    ps, ps_tol = [], []
    for _ in range(reps):
        s = _s_star(x[rng.permutation(n), 0], 0, denom, strata, cfg)
        ps.append(_pval(s, 0, null))
        ps_tol.append((1.0 + np.count_nonzero(s_null >= s - 1e-9 * abs(s))) / (cfg.n_perm + 1.0))
    ps, ps_tol = np.array(ps), np.array(ps_tol)
    print(f"A6 binary y (5% ones), n={n}: distinct null S values {len(np.unique(s_null.round(12)))}/{cfg.n_perm}")
    for a in (0.01, 0.05, 0.1, 0.2):
        print(f"A6 P(p<={a}): as coded {np.mean(ps <= a + 1e-12):.3f}   ties counted {np.mean(ps_tol <= a + 1e-12):.3f}"
              f"   (reps={reps})")
    print(f"A6 mean p as coded {ps.mean():.4f} vs ties counted {ps_tol.mean():.4f}; "
          f"reps where p differs: {np.mean(ps != ps_tol):.3f}")


if __name__ == "__main__":
    case_tied_candidate_row_order()
    case_constant_target()
    case_large_offset()
    case_ulp_tie_convention()
    case_nan()
    case_discrete_null_calibration()
