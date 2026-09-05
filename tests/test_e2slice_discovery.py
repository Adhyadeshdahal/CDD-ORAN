"""Tests for the frozen E2 discovery layer (U-centered partial distance correlation).

Recovery quality on the REAL E2 SCM is an empirical result, not a unit-test constant. These tests
prove the MECHANISM on synthetic data with a known support and hand-verified algebra:

- U-centering + U-inner-product match a brute-force transcription of the §6.2 formulas;
- the partial projection is orthogonal to the conditioning set and ``|pdCor| <= 1``;
- the score is ``abs(signed)`` and negative signed estimates are never clipped;
- the RELATIVE denominator guard predicate fires / does-not-fire correctly (§6.3);
- the permutation p-value formula and per-target BH-FDR selection are exact (§7);
- discovery recovers a KNOWN nonlinear (even/quadratic) support a linear score would miss;
- persistence round-trips and load fails closed on tamper / retune / mismatch (§8);
- the discovery path imports NO env truth.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cdd_oran.e2slice.dataset import (
    E2DatasetConfig,
    E2Rows,
    canonical_json,
    write_dataset,
)
from cdd_oran.e2slice.discovery import (
    PROTOCOL_COMMIT,
    E2DiscoveryConfig,
    bh_fdr_reject,
    denominator_guard_fires,
    discover_graph,
    dist_matrix_1d,
    dist_matrix_euclidean,
    frozen_config,
    load_discovery,
    partial_distance_correlation,
    u_center,
    u_inner,
    write_discovery,
)


# --- U-centering + U-inner product (hand / brute-force verification, §6.2) ----
def _brute_u_center(a: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    row = a.sum(axis=1)
    col = a.sum(axis=0)
    total = a.sum()
    u = np.zeros_like(a)
    for kk in range(n):
        for ll in range(n):
            if kk == ll:
                continue
            u[kk, ll] = a[kk, ll] - row[kk] / (n - 2) - col[ll] / (n - 2) + total / ((n - 1) * (n - 2))
    return u


def _brute_u_inner(a: np.ndarray, b: np.ndarray) -> float:
    n = a.shape[0]
    s = 0.0
    for kk in range(n):
        for ll in range(n):
            if kk != ll:
                s += a[kk, ll] * b[kk, ll]
    return s / (n * (n - 3))


def test_u_center_matches_brute_force():
    rng = np.random.default_rng(0)
    a = dist_matrix_1d(rng.standard_normal(6))
    np.testing.assert_allclose(u_center(a), _brute_u_center(a), rtol=0.0, atol=1e-12)


def test_u_center_zero_diagonal():
    a = dist_matrix_1d(np.random.default_rng(1).standard_normal(7))
    assert np.allclose(np.diag(u_center(a)), 0.0)


def test_u_inner_matches_brute_force():
    rng = np.random.default_rng(2)
    a = u_center(dist_matrix_1d(rng.standard_normal(6)))
    b = u_center(dist_matrix_1d(rng.standard_normal(6)))
    assert abs(u_inner(a, b) - _brute_u_inner(a, b)) < 1e-12


def test_u_inner_self_is_nonnegative():
    # <A,A> is a sum of squares / (n(n-3)) >= 0 (a genuine inner product, Szekely-Rizzo 2014).
    rng = np.random.default_rng(3)
    for _ in range(20):
        a = u_center(dist_matrix_1d(rng.standard_normal(8)))
        assert u_inner(a, a) >= 0.0


# --- partial-projection algebra (§6.2) ---------------------------------------
def _ucentered_triple(seed: int, n: int = 40):
    rng = np.random.default_rng(seed)
    xi = rng.standard_normal(n)
    yj = rng.standard_normal(n)
    z = rng.standard_normal((n, 13))
    cand_u = u_center(dist_matrix_1d(xi))
    tgt_u = u_center(dist_matrix_1d(yj))
    cond_u = u_center(dist_matrix_euclidean(z))
    return cand_u, tgt_u, cond_u


def test_projection_is_orthogonal_to_conditioning():
    cand_u, tgt_u, cond_u = _ucentered_triple(4)
    cand_self = u_inner(cand_u, cand_u)
    tgt_self = u_inner(tgt_u, tgt_u)
    cond_self = u_inner(cond_u, cond_u)
    signed, pxz, pyz, _va, _vb, guarded = partial_distance_correlation(
        cand_u, cand_self, tgt_u, tgt_self, cond_u, cond_self, 1e-12
    )
    assert not guarded
    # Pxz and Pyz live in the orthogonal complement of C~.
    assert abs(u_inner(pxz, cond_u)) < 1e-9
    assert abs(u_inner(pyz, cond_u)) < 1e-9
    assert abs(signed) <= 1.0 + 1e-9  # cosine, Cauchy-Schwarz bound


def test_score_is_abs_of_signed_and_negatives_are_retained():
    # Over many independent triples the signed partial dCor scatters around 0 incl. negatives;
    # the score is abs(signed) and negatives are never clipped to 0.
    found_negative = False
    for seed in range(40):
        cand_u, tgt_u, cond_u = _ucentered_triple(seed)
        signed, *_rest = partial_distance_correlation(
            cand_u, u_inner(cand_u, cand_u), tgt_u, u_inner(tgt_u, tgt_u),
            cond_u, u_inner(cond_u, cond_u), 1e-12,
        )
        if signed < 0:
            found_negative = True
    assert found_negative, "expected some negative signed pdCor estimates on independent data"


# --- denominator guard (§6.3) ------------------------------------------------
def test_denominator_guard_predicate_fires_and_does_not_fire():
    eps = 1e-12
    # Projected self-norms collapse relative to raw -> fires.
    assert denominator_guard_fires(1e-30, 1e-30, 1.0, 1.0, eps)
    # Healthy projected norms -> does not fire.
    assert not denominator_guard_fires(0.5, 0.5, 1.0, 1.0, eps)
    # RELATIVE: scaling both projected and raw by the same factor is invariant.
    assert denominator_guard_fires(1e-30 * 1e6, 1e-30 * 1e6, 1.0 * 1e6, 1.0 * 1e6, eps) == \
        denominator_guard_fires(1e-30, 1e-30, 1.0, 1.0, eps)


def test_guard_fires_and_returns_nan_when_projection_collapses():
    # A candidate identical to the target, both fully explained by the conditioning set, drives the
    # projected self-norms to (near) zero -> guard fires -> signed pdCor is NaN.
    n = 30
    rng = np.random.default_rng(9)
    base = rng.standard_normal(n)
    cand_u = u_center(dist_matrix_1d(base))
    tgt_u = u_center(dist_matrix_1d(base))
    # Conditioning stack that CONTAINS the base direction -> C~ explains cand/tgt almost fully.
    z = np.column_stack([base] + [base + 1e-9 * rng.standard_normal(n) for _ in range(12)])
    cond_u = u_center(dist_matrix_euclidean(z))
    signed, _pxz, _pyz, _va, _vb, guarded = partial_distance_correlation(
        cand_u, u_inner(cand_u, cand_u), tgt_u, u_inner(tgt_u, tgt_u),
        cond_u, u_inner(cond_u, cond_u), 1e-12,
    )
    assert guarded
    assert np.isnan(signed)


# --- permutation p-value + BH-FDR (§7) ---------------------------------------
def test_bh_fdr_hand_cases():
    q = 0.05
    # Only the single tiny p-value passes.
    np.testing.assert_array_equal(
        bh_fdr_reject(np.array([0.001, 0.2, 0.5, 0.9]), q), np.array([1, 0, 0, 0])
    )
    # Step-up rejects the first three.
    np.testing.assert_array_equal(
        bh_fdr_reject(np.array([0.01, 0.02, 0.03, 0.9]), q), np.array([1, 1, 1, 0])
    )
    # Nothing passes.
    np.testing.assert_array_equal(
        bh_fdr_reject(np.ones(4), q), np.zeros(4, dtype=int)
    )


def test_bh_fdr_on_a_14_wide_row():
    q = 0.05
    p = np.full(14, 0.9)
    p[3] = 0.001  # 0.001 <= (1/14)*0.05 = 0.00357 -> passes
    mask = bh_fdr_reject(p, q)
    assert mask[3] == 1 and mask.sum() == 1


def test_permutation_pvalue_is_bounded_and_selected_true_parent(_recovery_result):
    result, expected_mask = _recovery_result
    # p-values are in (0, 1]; guarded cells carry p = 1.
    assert (result.perm_pvalues > 0).all() and (result.perm_pvalues <= 1.0).all()
    assert np.all(result.perm_pvalues[result.guarded_mask == 1] == 1.0)


# --- end-to-end nonlinear recovery -------------------------------------------
@pytest.fixture(scope="module")
def _recovery_result():
    # Synthetic support: target K_j depends on candidate P_j through an EVEN (quadratic) function --
    # linear/rank partial correlation is ~0 while the functional dependence is total (§3). A small
    # additive noise keeps each target from being perfectly determined by the conditioning set (so
    # the §15.3 deterministic-collapse does not blanket-guard the non-parents in this unit test).
    # B_perm must be large enough that the best achievable p = 1/(B+1) clears the per-target BH
    # rank-1 threshold q/m = 0.05/14 = 0.00357 -- i.e. B >= 280. This is exactly why the frozen
    # §14 B_perm = 999 is load-bearing, so the test uses it (min p = 0.001, robust to a few null
    # exceedances).
    rng = np.random.default_rng(7)
    n = 120
    cand = rng.standard_normal((n, 14))
    y = np.empty((n, 6))
    for j in range(6):
        y[:, j] = cand[:, j] ** 2 + 0.1 * rng.standard_normal(n)
    rows = E2Rows(x_params=cand[:, :8], x_kpis=cand[:, 8:], y_kpis=y)
    cfg = frozen_config()
    result = discover_graph(rows, cfg)
    expected = np.zeros((6, 14), dtype=int)
    for j in range(6):
        expected[j, j] = 1
    return result, expected


def test_recovers_known_even_nonlinear_support(_recovery_result):
    result, expected = _recovery_result
    # Each target's true even-quadratic parent is selected; no false positives.
    np.testing.assert_array_equal(result.binary_mask, expected)


def test_no_guard_fires_on_healthy_synthetic(_recovery_result):
    result, _expected = _recovery_result
    assert int(result.guarded_mask.sum()) == 0
    assert np.array_equal(result.scores[result.guarded_mask == 0],
                          np.abs(result.signed_pdcor[result.guarded_mask == 0]))


def test_discovery_is_deterministic():
    rng = np.random.default_rng(11)
    n = 60
    cand = rng.standard_normal((n, 14))
    y = np.column_stack([cand[:, j] ** 2 for j in range(6)])
    rows = E2Rows(x_params=cand[:, :8], x_kpis=cand[:, 8:], y_kpis=y)
    cfg = replace(frozen_config(), b_perm=49)
    a = discover_graph(rows, cfg)
    b = discover_graph(rows, cfg)
    np.testing.assert_array_equal(a.binary_mask, b.binary_mask)
    np.testing.assert_array_equal(a.perm_pvalues, b.perm_pvalues)
    np.testing.assert_allclose(a.signed_pdcor, b.signed_pdcor, rtol=0, atol=0, equal_nan=True)


def test_rejects_zero_variance_column():
    rng = np.random.default_rng(5)
    n = 40
    cand = rng.standard_normal((n, 14))
    cand[:, 3] = 2.0  # dead candidate column
    rows = E2Rows(x_params=cand[:, :8], x_kpis=cand[:, 8:],
                  y_kpis=rng.standard_normal((n, 6)))
    with pytest.raises(ValueError, match="zero-variance"):
        discover_graph(rows, replace(frozen_config(), b_perm=19))


def test_rejects_wrong_candidate_layout():
    rng = np.random.default_rng(6)
    rows = E2Rows(x_params=rng.standard_normal((30, 7)),  # 7 params, not 8
                  x_kpis=rng.standard_normal((30, 6)),
                  y_kpis=rng.standard_normal((30, 6)))
    with pytest.raises(ValueError, match="layout"):
        discover_graph(rows, replace(frozen_config(), b_perm=19))


# --- firewall: no env truth in the discovery path ----------------------------
def test_discovery_imports_no_env_truth():
    import cdd_oran.e2slice.discovery as disc

    tree = ast.parse(Path(disc.__file__).read_text())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert not any("envs" in name or "E2V2Env" in name or "true_adj" in name for name in imported)
    assert not hasattr(disc, "E2V2Env")


# --- persistence + fail-closed load (§8) -------------------------------------
_DS_CFG = E2DatasetConfig(n_rows_per_seed=48, seed=0, sampling_seed=0)


@pytest.fixture(scope="module")
def _persisted(tmp_path_factory):
    """Run the FROZEN discovery once on a tiny dataset; reused across the fail-closed tests."""
    d = tmp_path_factory.mktemp("e2disc")
    write_dataset(_DS_CFG, d)
    record = write_discovery(d)
    return d, record


def test_write_discovery_binds_and_stamps_protocol(_persisted):
    d, record = _persisted
    assert (d / "discovery.json").exists()
    assert record["protocol_commit"] == PROTOCOL_COMMIT
    assert record["score_method"] == "u_centered_partial_distance_correlation"
    assert record["threshold_method"] == "per_candidate_permutation_per_target_bh_fdr"
    assert record["alpha"] == 1 and record["denominator_epsilon"] == 1e-12
    assert record["b_perm"] == 999 and record["permutation_seed"] == 0 and record["q"] == 0.05
    assert "git_sha" in record
    manifest = json.loads((d / "manifest.json").read_text())
    assert record["dataset_hash"] == manifest["dataset_hash"]
    loaded = load_discovery(d, expected_dataset_hash=manifest["dataset_hash"])
    assert np.asarray(loaded["binary_mask"]).shape == (6, 14)


def test_write_discovery_refuses_throwaway_config(_persisted):
    d, _record = _persisted
    with pytest.raises(ValueError, match="FROZEN"):
        write_discovery(d, E2DiscoveryConfig(b_perm=49), force=True)


def test_valid_discovery_loads_after_rederivation(_persisted):
    d, _record = _persisted
    before = (d / "discovery.json").read_text()
    load_discovery(d)
    assert (d / "discovery.json").read_text() == before


def _rehash(record: dict) -> None:
    payload = {k: v for k, v in record.items() if k != "content_hash"}
    record["content_hash"] = hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def _load_mutated(persisted, tmp_path: Path, mutate, rehash: bool) -> dict:
    d, _record = persisted
    record = json.loads((d / "discovery.json").read_text())
    mutate(record)
    if rehash:
        _rehash(record)
    (tmp_path / "discovery.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    return record


def test_load_rejects_tampered_content_hash(_persisted, tmp_path: Path):
    _load_mutated(_persisted, tmp_path,
                  lambda r: r["scores"][0].__setitem__(0, 0.123456), rehash=False)
    with pytest.raises(ValueError, match="content_hash"):
        load_discovery(tmp_path)


def test_load_rejects_mismatched_parent_hash(_persisted, tmp_path: Path):
    _load_mutated(_persisted, tmp_path, lambda r: r.__setitem__("dataset_hash", "0" * 64),
                  rehash=True)
    with pytest.raises(ValueError, match="dataset_hash"):
        load_discovery(tmp_path, expected_dataset_hash="f" * 64)


def test_load_rejects_wrong_protocol_commit(_persisted, tmp_path: Path):
    _load_mutated(_persisted, tmp_path, lambda r: r.__setitem__("protocol_commit", "0" * 40),
                  rehash=True)
    with pytest.raises(ValueError, match="protocol_commit"):
        load_discovery(tmp_path)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda r: r.__setitem__("b_perm", 99), "b_perm"),
        (lambda r: r.__setitem__("q", 0.10), "q"),
        (lambda r: r.__setitem__("alpha", 2), "alpha"),
        (lambda r: r.__setitem__("denominator_epsilon", 1e-6), "denominator_epsilon"),
        (lambda r: r.__setitem__("score_method", "pearson"), "score_method"),
    ],
)
def test_load_rejects_frozen_constant_violation(_persisted, tmp_path: Path, mutate, match):
    _load_mutated(_persisted, tmp_path, mutate, rehash=True)
    with pytest.raises(ValueError, match=match):
        load_discovery(tmp_path)


def test_load_rejects_wrong_shape(_persisted, tmp_path: Path):
    def mutate(r):
        for key in ("signed_pdcor", "scores", "perm_pvalues", "guarded_mask", "binary_mask"):
            r[key] = r[key][:3]  # drop child rows -> (3, 14)
        r["candidate_shape"] = [3, 14]
    _load_mutated(_persisted, tmp_path, mutate, rehash=True)
    with pytest.raises(ValueError, match="shape"):
        load_discovery(tmp_path)


def test_load_rejects_retuned_mask(_persisted, tmp_path: Path):
    # Flip one BH decision while keeping content_hash consistent -> the BH re-derivation rejects it.
    def mutate(r):
        r["binary_mask"][0][0] = 1 - int(r["binary_mask"][0][0])
    _load_mutated(_persisted, tmp_path, mutate, rehash=True)
    with pytest.raises(ValueError, match="BH-FDR|binary_mask"):
        load_discovery(tmp_path)


def test_discover_refuses_stale_downstream_without_force(_persisted, tmp_path: Path):
    # Fresh dataset in tmp (the module fixture dir must stay valid for other tests).
    write_dataset(_DS_CFG, tmp_path)
    write_discovery(tmp_path)
    (tmp_path / "recovery.json").write_text("{}")
    with pytest.raises(ValueError, match="downstream artifacts already exist"):
        write_discovery(tmp_path)
    write_discovery(tmp_path, force=True)
    assert not (tmp_path / "recovery.json").exists()
