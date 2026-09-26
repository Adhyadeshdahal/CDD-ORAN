"""Unit self-checks for the RCoT E2 discovery sibling (``discovery_rcot.py``).

**PRE-TRUTH**: these tests read NO ground truth and score against NO truth. They prove the RCoT
CI-test is CALIBRATED (null ~Uniform, power ~1) before the method is ever pointed at real E2 data,
and that the module honours the ``(6, 14)`` mask contract + persistence round-trip. The null /
power self-checks are ports of the reference ``calib_study.rcot_selfcheck`` (independent X,Y,Z; a
linear-Gaussian conditional null X⟂Y|Z; and an X->Y power case), run at modest reps so the suite
completes in well under a minute single-core.

Thread caps are pinned to 1 at import (thermal constraint on this laptop).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from cdd_oran.e2slice.dataset import E2DatasetConfig, E2Rows, write_dataset  # noqa: E402
from cdd_oran.e2slice.discovery_rcot import (  # noqa: E402
    PROTOCOL_COMMIT,
    RCOT_SCHEMA_VERSION,
    RCoTDiscoveryConfig,
    discover_graph_rcot,
    discovered_mask_array,
    load_discovery_rcot,
    rcot_pvalue_analytic,
    rcot_pvalue_block_perm,
    write_discovery_rcot,
)


def _ks_uniform(p: np.ndarray) -> float:
    p = np.sort(np.asarray(p))
    n = p.size
    cdf = np.arange(1, n + 1) / n
    return float(np.max(np.abs(cdf - p)))


# --- RCoT null / power self-checks (ports of calib_study.rcot_selfcheck) ------
@pytest.fixture(scope="module")
def _selfcheck():
    """Run the three reference self-check cases once; reused across assertions."""
    cfg = RCoTDiscoveryConfig()
    reps, n = 200, 200
    rng = np.random.default_rng(123)
    pA, pB, pC = [], [], []
    for r in range(reps):
        # A: X, Y, Z fully independent Gaussians -> uniform null.
        Z = rng.normal(size=(n, 3))
        X = rng.normal(size=n)
        Y = rng.normal(size=n)
        pA.append(rcot_pvalue_analytic(X, Y, Z, cfg, seed=r)[1])
        # B: X _|_ Y | Z, linear-Gaussian (both depend on Z -> marginally dependent) -> uniform null.
        Z2 = rng.normal(size=(n, 3))
        bx = rng.normal(size=3)
        by = rng.normal(size=3)
        Xb = Z2 @ bx + rng.normal(size=n)
        Yb = Z2 @ by + rng.normal(size=n)
        pB.append(rcot_pvalue_analytic(Xb, Yb, Z2, cfg, seed=1000 + r)[1])
        # C: X -> Y direct edge, Z independent -> power near 1.
        Z3 = rng.normal(size=(n, 3))
        Xc = rng.normal(size=n)
        Yc = 0.7 * Xc + rng.normal(size=n)
        pC.append(rcot_pvalue_analytic(Xc, Yc, Z3, cfg, seed=2000 + r)[1])
    return {k: np.asarray(v) for k, v in (("A", pA), ("B", pB), ("C", pC))}


def test_null_independent_is_calibrated(_selfcheck):
    p = _selfcheck["A"]
    fp = float(np.mean(p < 0.05))
    mean = float(p.mean())
    ks = _ks_uniform(p)
    # calib_study4 reported the analytic HBE null as mildly liberal (FP 0.07-0.13); allow headroom.
    assert fp <= 0.15, f"A_indep FP@.05={fp:.3f} too high"
    assert 0.40 <= mean <= 0.60, f"A_indep mean p={mean:.3f} not ~0.5"
    assert ks <= 0.15, f"A_indep KS={ks:.3f} too large"


def test_conditional_null_lingauss_is_calibrated(_selfcheck):
    p = _selfcheck["B"]
    fp = float(np.mean(p < 0.05))
    ks = _ks_uniform(p)
    assert fp <= 0.15, f"B_ci_linGauss FP@.05={fp:.3f} too high"
    assert 0.40 <= float(p.mean()) <= 0.60
    assert ks <= 0.15, f"B_ci_linGauss KS={ks:.3f} too large"


def test_power_direct_edge(_selfcheck):
    p = _selfcheck["C"]
    power = float(np.mean(p < 0.05))
    assert power >= 0.90, f"C_power_XtoY power@.05={power:.3f} too low"


def test_both_null_variants_give_valid_pvalues():
    cfg_hbe = RCoTDiscoveryConfig(null_method="analytic_hbe")
    cfg_perm = RCoTDiscoveryConfig(null_method="block_perm", block_perm_reps=49, block_size=15)
    rng = np.random.default_rng(5)
    n = 120
    for r in range(6):
        Z = rng.normal(size=(n, 3))
        X = rng.normal(size=n)
        Y = rng.normal(size=n)
        _s_h, p_h, g_h = rcot_pvalue_analytic(X, Y, Z, cfg_hbe, seed=r)
        _s_p, p_p, g_p = rcot_pvalue_block_perm(X, Y, Z, cfg_perm, seed=r)
        assert 0.0 <= p_h <= 1.0 and not g_h
        assert 0.0 <= p_p <= 1.0 and not g_p


# --- shape / contract + determinism ------------------------------------------
def _synthetic_rows(seed: int, n: int = 120):
    """Even (quadratic) support: target K_j depends on candidate j through an even function -- a
    genuine nonlinear conditional dependence that a linear score misses. Truth-free construction."""
    rng = np.random.default_rng(seed)
    cand = rng.standard_normal((n, 14))
    y = np.empty((n, 6))
    for j in range(6):
        y[:, j] = cand[:, j] ** 2 + 0.1 * rng.standard_normal(n)
    return E2Rows(x_params=cand[:, :8], x_kpis=cand[:, 8:], y_kpis=y)


def test_mask_shape_and_dtype_contract():
    result = discover_graph_rcot(_synthetic_rows(7))
    assert result.binary_mask.shape == (6, 14)
    assert result.binary_mask.dtype == np.int64
    assert set(np.unique(result.binary_mask).tolist()).issubset({0, 1})
    assert result.pvalues.shape == (6, 14)
    assert ((result.pvalues > 0) & (result.pvalues <= 1.0)).all()


def test_guarded_entries_never_selected():
    # A high residual-collapse threshold forces the guard to fire on a controlled subset of cells
    # (residual_epsilon=0.7 -> ~9 of 84 here). Raise max_guard_fraction so this exercises the RETURN
    # path (fail-closed selection) rather than the guard-fraction HALT (tested separately below).
    result = discover_graph_rcot(
        _synthetic_rows(7), RCoTDiscoveryConfig(residual_epsilon=0.7, max_guard_fraction=2.0)
    )
    assert result.guarded_mask.sum() >= 1  # the config really does guard some cells
    # Wherever the guard fired, that cell is never selected and carries p = 1.
    assert np.all(result.binary_mask[result.guarded_mask == 1] == 0)
    assert np.all(result.pvalues[result.guarded_mask == 1] == 1.0)


def test_guard_fraction_halt_fires():
    """§11 guard-fraction HALT (parity with discovery.py): >= ceil(0.05*84)=5 guarded cells HALT.

    residual_epsilon=0.9 collapses the guard on every candidate (>> the threshold 5), so the default
    max_guard_fraction=0.05 must raise the parity HALT rather than return a mask.
    """
    with pytest.raises(ValueError, match="HALT"):
        discover_graph_rcot(_synthetic_rows(7), RCoTDiscoveryConfig(residual_epsilon=0.9))


def test_discovery_is_deterministic():
    rows = _synthetic_rows(11, n=80)
    a = discover_graph_rcot(rows)
    b = discover_graph_rcot(rows)
    np.testing.assert_array_equal(a.binary_mask, b.binary_mask)
    np.testing.assert_allclose(a.pvalues, b.pvalues, rtol=0, atol=0)
    np.testing.assert_allclose(a.statistic, b.statistic, rtol=0, atol=0, equal_nan=True)


def test_block_perm_null_method_runs_end_to_end():
    rows = _synthetic_rows(3, n=80)
    cfg = RCoTDiscoveryConfig(null_method="block_perm", block_perm_reps=49, block_size=15)
    result = discover_graph_rcot(rows, cfg)
    assert result.binary_mask.shape == (6, 14)
    assert result.null_method == "block_perm"
    assert ((result.pvalues > 0) & (result.pvalues <= 1.0)).all()


def test_rejects_zero_variance_column():
    rng = np.random.default_rng(5)
    n = 40
    cand = rng.standard_normal((n, 14))
    cand[:, 3] = 2.0
    rows = E2Rows(x_params=cand[:, :8], x_kpis=cand[:, 8:], y_kpis=rng.standard_normal((n, 6)))
    with pytest.raises(ValueError, match="zero-variance"):
        discover_graph_rcot(rows)


def test_rejects_wrong_layout():
    rng = np.random.default_rng(6)
    rows = E2Rows(x_params=rng.standard_normal((30, 7)),
                  x_kpis=rng.standard_normal((30, 6)),
                  y_kpis=rng.standard_normal((30, 6)))
    with pytest.raises(ValueError, match="layout"):
        discover_graph_rcot(rows)


def test_invalid_null_method_rejected():
    with pytest.raises(ValueError, match="null_method"):
        discover_graph_rcot(_synthetic_rows(1, n=40), RCoTDiscoveryConfig(null_method="bogus"))


# --- persistence round-trip (truth-free) -------------------------------------
def test_persist_and_load_round_trips(tmp_path: Path):
    d = tmp_path / "e2rcot"
    write_dataset(E2DatasetConfig(n_rows_per_seed=64, seed=0, sampling_seed=0), d)
    record = write_discovery_rcot(d)
    assert (d / "discovery_rcot.json").exists()
    assert not (d / "discovery.json").exists()  # never clobbers the frozen artifact name
    assert record["schema_version"] == RCOT_SCHEMA_VERSION
    assert record["frozen"] is True
    assert record["protocol_commit"] == PROTOCOL_COMMIT
    assert record["score_method"] == "rcot_conditional_independence"
    assert record["null_method"] == "block_perm"  # FROZEN primary selector
    assert record["dz"] == 25 and record["q"] == 0.05
    manifest = json.loads((d / "manifest.json").read_text())
    assert record["dataset_hash"] == manifest["dataset_hash"]

    before = (d / "discovery_rcot.json").read_text()
    loaded = load_discovery_rcot(d)
    assert (d / "discovery_rcot.json").read_text() == before  # load does not mutate
    mask = discovered_mask_array(loaded)
    assert mask.shape == (6, 14)
    assert set(np.unique(mask).tolist()).issubset({0, 1})


def test_load_rejects_tampered_content_hash(tmp_path: Path):
    d = tmp_path / "e2rcot2"
    write_dataset(E2DatasetConfig(n_rows_per_seed=64, seed=0, sampling_seed=0), d)
    write_discovery_rcot(d)
    record = json.loads((d / "discovery_rcot.json").read_text())
    record["pvalues"][0][0] = 0.123456  # tamper without re-hashing
    (d / "discovery_rcot.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    with pytest.raises(ValueError, match="content_hash"):
        load_discovery_rcot(d)


# --- SHIPPING-PATH calibration on the faithful E2 kpi_fn testbed --------------
# TRUTH-FREE. A byte-faithful standalone copy of the real E2 KPI mechanism
# (runs/calib-study/calib_study4_correct.py `kpi_fn`, cross-checked there against E2V2Env by
# `_verify_kpifn.py`). It reads NO real ground-truth adjacency: these are self-checks that the
# STANDARDIZED SHIPPING numerics (the per-column z-standardization discover_graph_rcot applies up
# front, which the adversarial review PROVED is NOT an affine no-op for the joint-Z p-value) are
# calibrated on the real KPI->KPI geometry -- near-nominal FP on a true null, high power on a true
# parent -- for BOTH null variants, BEFORE the method is ever pointed at real E2 data.
_ID_RANGES = [(-100.0, 100.0), (-10.0, 50.0), (-20.0, 20.0), (-60.0, 60.0),
              (-20.0, 20.0), (-50.0, 150.0), (-60.0, 65.0), (-100.0, 150.0)]


def _safe_exp(x):
    return np.where(np.abs(x) > 1e-1, x, 1e-1)


def _kpi_fn(p):
    """(n,8) params -> (n,6) KPIs; byte-faithful to E2V2Env._update_kpis (decoy OFF)."""
    p0, p1, p2, p3, p4, p5, p6, p7 = (p[:, i] for i in range(8))
    se1 = _safe_exp(p1)
    se3 = _safe_exp(p3)
    se4 = _safe_exp(p4)
    se6 = _safe_exp(p6)
    k0 = 80.0 * np.exp(-(p0 ** 2) / (2.0 * se1 ** 2))
    k1 = 100.0 * np.exp(-((p0 + p2) ** 2) / (2.0 * se1 ** 2))
    k2 = 120.0 * np.exp(-((p0 + 45.0) ** 2) / (2.0 * se3 ** 2))
    k3 = 120.0 * np.exp(-((p5 + p1 - 30.0) ** 2) / (2.0 * se4 ** 2))
    k4 = 150.0 * np.exp(-((p5 + p1 - 50.0) ** 2) / (2.0 * se4 ** 2))
    k5 = -35.0 * np.exp(-((p7 + p0 - 25.0) ** 2) / (2.0 * se6 ** 2))
    return np.stack([k0, k1, k2, k3, k4, k5], axis=1)


def _draw_params(n, rng):
    return np.column_stack([rng.uniform(lo, hi, size=n) for (lo, hi) in _ID_RANGES])


def _std_col(v):
    """Population-std (ddof=0) z-standardize a 1-D column -- matches discovery_rcot._standardize."""
    s = float(v.std())
    return (v - v.mean()) / (s if s > 0 else 1.0)


def _std_mat(Z):
    return np.column_stack([_std_col(Z[:, c]) for c in range(Z.shape[1])])


def _ship_pval(pvfn, cand, target, Z, cfg, seed):
    """Exercise the STANDARDIZED shipping numerics: apply the discover path's per-column
    z-standardization to (cand, target, each Z column), then call the null variant's p-value fn.
    Per-column standardization of Z == what discover_graph_rcot does to xs before Z = delete(xs, i).
    """
    return pvfn(_std_col(cand), _std_col(target), _std_mat(Z), cfg, seed)


@pytest.mark.parametrize(
    "null_method,fp_reps,power_reps,extra_cfg",
    [
        ("analytic_hbe", 120, 120, {}),
        ("block_perm", 30, 40, {"block_perm_reps": 39, "block_size": 20}),
    ],
)
def test_shipping_path_calibrated_on_faithful_e2(null_method, fp_reps, power_reps, extra_cfg):
    """Shipping standardized path: near-nominal FP on a faithful E2 true null (lagged k3 vs cur k0,
    strong sibling k4 in Z) and high power on a true parent (cur p0 -> cur k0), for BOTH null
    variants.

    Tolerances are deliberately LOOSE at these modest single-core reps: calib_study4 measured the
    true faithful-null FP at ~0.07 (analytic, standardized) / ~0.06 (block_perm) and true-parent
    power ~0.9-0.95 at n=400, so the FP<=0.20 / power>=0.80 bands guard against GROSS miscalibration
    -- the marginal-pdCor bug this method replaces over-selected ~9x (FP ~0.5-0.9) -- not the final
    calibration point. The high-rep pre-registration numbers live in the scratchpad calibration
    table, not in the unit suite.
    """
    cfg = RCoTDiscoveryConfig(null_method=null_method, **extra_cfg)
    n = 400
    pvfn = rcot_pvalue_analytic if null_method == "analytic_hbe" else rcot_pvalue_block_perm

    # True null: lagged KPI k3 (independent of the current KPI target, but strongly Z-coupled via its
    # sibling lagged KPIs in Z). A miscalibrated conditional test over-rejects here.
    fp = []
    for r in range(fp_reps):
        rng = np.random.default_rng(4000 + r)
        u_prev = _draw_params(n, rng)
        u_cur = _draw_params(n, rng)
        lagged = _kpi_fn(u_prev)
        cur = _kpi_fn(u_cur)
        z = np.column_stack([u_cur, np.delete(lagged, 3, axis=1)])   # 8 params + 5 other lagged = 13
        _s, p, g = _ship_pval(pvfn, lagged[:, 3], cur[:, 0], z, cfg, seed=r * 17 + 1)
        if not g:
            fp.append(p)
    fp_rate = float(np.mean(np.asarray(fp) < 0.05))
    assert fp_rate <= 0.20, f"{null_method} faithful true-null FP@.05={fp_rate:.3f} > 0.20 tolerance"

    # True parent: current param p0 genuinely drives current KPI k0 -> power must be high.
    pw = []
    for r in range(power_reps):
        rng = np.random.default_rng(6000 + r)
        u_prev = _draw_params(n, rng)
        u_cur = _draw_params(n, rng)
        lagged = _kpi_fn(u_prev)
        cur = _kpi_fn(u_cur)
        z = np.column_stack([np.delete(u_cur, 0, axis=1), lagged])   # other 7 params + 6 lagged = 13
        _s, p, g = _ship_pval(pvfn, u_cur[:, 0], cur[:, 0], z, cfg, seed=r * 23 + 5)
        if not g:
            pw.append(p)
    power = float(np.mean(np.asarray(pw) < 0.05))
    assert power >= 0.80, f"{null_method} faithful true-parent power@.05={power:.3f} < 0.80"
