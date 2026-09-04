"""Tests for the frozen E1 discovery v2 layer (partial-correlation, per-target largest_gap).

Recovery quality on the REAL E1 SCM is an empirical result, not a unit-test constant; these tests
use synthetic equations with a KNOWN support to prove the mechanism:

- a true single parent -> |rho| ~ 1, a non-parent -> exactly 0 (§3.4 branch 1);
- a **two-parent** target recovers BOTH co-parents (the FROZEN_FLOOR=0.0 fix: the 0->signal gap is
  eligible so the per-target cut lands ~0.5, not inside the ~1 cluster);
- train-only fit, determinism, invalid-input rejection, the collinearity STOP guard, and
  fail-closed/tamper handling of discovery_v2.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cdd_oran.e1slice.dataset import E1DatasetConfig, E1Rows, canonical_json, write_dataset
from cdd_oran.e1slice.discovery_v2 import (
    PROTOCOL_COMMIT,
    DiscoveryV2Config,
    discover_graph_v2,
    load_discovery_v2,
    write_discovery_v2,
)
from cdd_oran.e1slice.split import SplitConfig, load_split, write_split

_N_EP = 12
_PER = 40


def _rows_from_feats(feats: np.ndarray, y: np.ndarray) -> E1Rows:
    n = feats.shape[0]
    episode = np.repeat(np.arange(_N_EP), n // _N_EP).astype(np.int64)
    time = np.tile(np.arange(n // _N_EP), _N_EP).astype(np.int64)
    return E1Rows(
        episode=episode,
        time=time,
        x_params=feats[:, :4].astype(np.float64),
        x_kpis=feats[:, 4:].astype(np.float64),
        y_kpis=y.astype(np.float64),
    )


# Known synthetic support mirroring E1's structure: outputs 0,1 are SINGLE-parent (P0->K0, P1->K1);
# outputs 2,3 are TWO-parent, each a param + a KPI co-parent (P2 + 0.5*K0, P3 + 0.5*K1) -- the
# noiseless, exact-linear case where v1's global cut and a positive-floor per-row cut both miss the
# weaker co-parent. All features are iid so non-parents have exactly-zero residual association.
_STRONG = {0: 0, 1: 1}                    # output -> single param parent
_COPARENTS = {2: (2, 4), 3: (3, 5)}       # output -> (param parent, kpi co-parent col)


def _synthetic_rows(seed: int = 0, tweak_test_labels: bool = False) -> E1Rows:
    rng = np.random.default_rng(seed)
    n = _N_EP * _PER
    feats = rng.standard_normal((n, 8))
    y = np.zeros((n, 4), dtype=np.float64)
    for out, col in _STRONG.items():
        y[:, out] = feats[:, col]
    for out, (p_col, k_col) in _COPARENTS.items():
        y[:, out] = feats[:, p_col] + 0.5 * feats[:, k_col]
    rows = _rows_from_feats(feats, y)
    if tweak_test_labels:
        # Corrupt ONLY the held-out episodes' labels; train-only discovery must be unaffected.
        y = rows.y_kpis.copy()
        y[np.isin(rows.episode, [10, 11])] += 1000.0
        rows = E1Rows(rows.episode, rows.time, rows.x_params, rows.x_kpis, y)
    return rows


_TRAIN = list(range(10))
_TEST = [10, 11]


def _expected_mask() -> np.ndarray:
    mask = np.zeros((4, 8), dtype=int)
    for out, col in _STRONG.items():
        mask[out, col] = 1
    for out, (p_col, k_col) in _COPARENTS.items():
        mask[out, p_col] = 1
        mask[out, k_col] = 1
    return mask


def test_recovers_known_support_including_both_coparents():
    result = discover_graph_v2(_synthetic_rows(), _TRAIN)
    np.testing.assert_array_equal(result.binary_mask, _expected_mask())
    # The multi-parent rows must select EXACTLY two edges each (both co-parents, no float drop).
    for out in _COPARENTS:
        assert int(result.binary_mask[out].sum()) == 2, f"target {out} did not recover both co-parents"


def test_true_edges_near_one_and_non_edges_exactly_zero():
    result = discover_graph_v2(_synthetic_rows(), _TRAIN)
    exp = _expected_mask()
    scores = result.scores
    # Every true edge scores ~1; every non-edge is exactly 0 (branch 1 definition).
    assert scores[exp == 1].min() > 0.99
    assert np.count_nonzero(scores[exp == 0]) == 0


def test_per_target_threshold_lands_between_zero_and_signal():
    result = discover_graph_v2(_synthetic_rows(), _TRAIN)
    # FROZEN_FLOOR=0.0 makes the 0->signal gap eligible, so each per-target cut sits well inside
    # (0, 1) -- not pushed up into the ~1 signal cluster.
    assert result.threshold.shape == (4,)
    assert np.all(result.threshold > 0.0) and np.all(result.threshold < 1.0)


def test_row_order_shuffle_does_not_change_mask():
    rows = _synthetic_rows()
    base = discover_graph_v2(rows, _TRAIN).binary_mask
    perm = np.random.default_rng(1).permutation(rows.n)
    shuffled = E1Rows(
        episode=rows.episode[perm],
        time=rows.time[perm],
        x_params=rows.x_params[perm],
        x_kpis=rows.x_kpis[perm],
        y_kpis=rows.y_kpis[perm],
    )
    np.testing.assert_array_equal(discover_graph_v2(shuffled, _TRAIN).binary_mask, base)


def test_discovery_is_deterministic():
    a = discover_graph_v2(_synthetic_rows(), _TRAIN)
    b = discover_graph_v2(_synthetic_rows(), _TRAIN)
    np.testing.assert_array_equal(a.binary_mask, b.binary_mask)
    np.testing.assert_array_equal(a.partial_corr, b.partial_corr)
    np.testing.assert_array_equal(a.threshold, b.threshold)


def test_changing_held_out_rows_cannot_change_the_mask():
    clean = discover_graph_v2(_synthetic_rows(tweak_test_labels=False), _TRAIN).binary_mask
    tweaked = discover_graph_v2(_synthetic_rows(tweak_test_labels=True), _TRAIN).binary_mask
    np.testing.assert_array_equal(clean, tweaked)


def test_rejects_zero_variance_feature():
    rows = _synthetic_rows()
    dead = rows.x_params.copy()
    dead[:, 3] = 1.0
    rows = E1Rows(rows.episode, rows.time, dead, rows.x_kpis, rows.y_kpis)
    with pytest.raises(ValueError, match="zero-variance"):
        discover_graph_v2(rows, _TRAIN)


def test_rejects_rank_deficient_design():
    rows = _synthetic_rows()
    dup = rows.x_kpis.copy()
    dup[:, 3] = rows.x_params[:, 3]  # K3 input == P3 input -> exact collinearity in the design
    rows = E1Rows(rows.episode, rows.time, rows.x_params, dup, rows.y_kpis)
    with pytest.raises(ValueError, match="rank-deficient"):
        discover_graph_v2(rows, _TRAIN)


def test_near_collinear_input_is_a_stop_not_an_absent_edge():
    # Branch 2 (§3.4): an input NEARLY (not exactly) determined by the others, while the target is
    # NOT fully explained without it -> recorded STOP (distinct from an absent edge). x7 = x6 + tiny
    # noise makes the pair near-collinear (feature residual << EPS_COND, but not exactly rank-
    # deficient). A target built from the pair's ORTHOGONAL part (x7 - x6) genuinely needs that tiny
    # unique direction, so its target residual stays full -> branch 1 does NOT pre-empt -> STOP.
    rng = np.random.default_rng(3)
    n = _N_EP * _PER
    feats = rng.standard_normal((n, 8))
    feats[:, 7] = feats[:, 6] + 3e-5 * rng.standard_normal(n)  # near-collinear pair (not exact)
    y = np.zeros((n, 4))
    y[:, 0] = feats[:, 0]
    y[:, 1] = feats[:, 1]
    y[:, 2] = feats[:, 2]
    y[:, 3] = feats[:, 7] - feats[:, 6]  # the pair's orthogonal part: needs the near-collinear dir
    rows = _rows_from_feats(feats, y)
    with pytest.raises(ValueError, match="near-collinear"):
        discover_graph_v2(rows, _TRAIN)


def test_rejects_empty_training_rows():
    with pytest.raises(ValueError, match="no training rows"):
        discover_graph_v2(_synthetic_rows(), [])


def test_config_freezes_floor_score_and_threshold_method():
    cfg = DiscoveryV2Config()
    assert cfg.floor == 0.0
    assert cfg.score_method == "partial_correlation"
    assert cfg.threshold_method == "per_target_largest_gap"
    with pytest.raises(ValueError, match="floor is frozen"):
        DiscoveryV2Config(floor=1e-3)
    with pytest.raises(ValueError, match="score_method is frozen"):
        DiscoveryV2Config(score_method="ols_beta")
    with pytest.raises(ValueError, match="threshold_method is frozen"):
        DiscoveryV2Config(threshold_method="pooled_largest_gap")


def test_does_not_import_env_truth():
    import ast

    import cdd_oran.e1slice.discovery_v2 as disc

    tree = ast.parse(Path(disc.__file__).read_text())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert not any("envs" in name or "E1V2Env" in name or "true_adj" in name for name in imported)
    assert not hasattr(disc, "E1V2Env")


@pytest.mark.parametrize("env_seed", list(range(10)))
def test_discovered_mask_equals_oracle_true_mask_on_real_e1(tmp_path: Path, env_seed: int):
    # Phase-3 mask-identity guard: on the REAL E1 SCM, v2 recovers the exact true graph, so the
    # discovered (K, P+K) mask is bit-for-bit the ORACLE mask (E1 true adjacency in the candidate
    # layout). arm_mask reads env truth (it is the oracle reference), used HERE only for comparison;
    # discovery itself reads no truth. Guards the by-construction MSE-parity argument in phase 3.
    from cdd_oran.e1slice.model import arm_mask

    write_dataset(E1DatasetConfig(n_episodes=16, steps_per_episode=12, warmup=2, env_seed=env_seed), tmp_path)
    write_split(tmp_path, SplitConfig(test_fraction=0.25, split_seed=0))
    record = write_discovery_v2(tmp_path)
    discovered = np.asarray(record["binary_mask"], dtype=int)
    oracle = arm_mask("oracle").cpu().numpy().astype(int)
    np.testing.assert_array_equal(discovered, oracle)


_CFG = E1DatasetConfig(n_episodes=12, steps_per_episode=8, warmup=2, env_seed=0)


def _prepare_dataset(tmp_path: Path) -> dict:
    write_dataset(_CFG, tmp_path)
    write_split(tmp_path, SplitConfig(test_fraction=0.25, split_seed=0))
    return write_discovery_v2(tmp_path)


def test_write_discovery_v2_binds_and_records_protocol(tmp_path: Path):
    record = _prepare_dataset(tmp_path)
    assert (tmp_path / "discovery_v2.json").exists()
    assert record["protocol_commit"] == PROTOCOL_COMMIT
    assert record["score_method"] == "partial_correlation"
    assert record["threshold_method"] == "per_target_largest_gap"
    assert record["floor"] == 0.0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    split = load_split(tmp_path)
    assert record["dataset_hash"] == manifest["dataset_hash"]
    assert record["split_hash"] == split["split_hash"]
    loaded = load_discovery_v2(
        tmp_path,
        expected_dataset_hash=manifest["dataset_hash"],
        expected_split_hash=split["split_hash"],
    )
    assert np.asarray(loaded["binary_mask"]).shape == (4, 8)
    assert np.asarray(loaded["threshold"]).shape == (4,)


def test_load_discovery_v2_rejects_mismatched_binding(tmp_path: Path):
    _prepare_dataset(tmp_path)
    with pytest.raises(ValueError, match="dataset_hash"):
        load_discovery_v2(tmp_path, expected_dataset_hash="f" * 64)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    with pytest.raises(ValueError, match="split_hash"):
        load_discovery_v2(
            tmp_path, expected_dataset_hash=manifest["dataset_hash"], expected_split_hash="0" * 64
        )


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda d: d["partial_corr"][0].__setitem__(0, d["partial_corr"][0][0] + 1.0), id="partial_corr"),
        pytest.param(lambda d: d["threshold"].__setitem__(0, float(d["threshold"][0]) + 1.0), id="threshold"),
        pytest.param(lambda d: d["binary_mask"][0].__setitem__(3, 1 - d["binary_mask"][0][3]), id="mask"),
        pytest.param(lambda d: d.__setitem__("dataset_hash", "0" * 64), id="parent_hash"),
    ],
)
def test_corrupted_discovery_v2_is_rejected(tmp_path: Path, mutate):
    _prepare_dataset(tmp_path)
    record = json.loads((tmp_path / "discovery_v2.json").read_text())
    mutate(record)  # mutate a field WITHOUT recomputing content_hash
    (tmp_path / "discovery_v2.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    with pytest.raises(ValueError):
        load_discovery_v2(tmp_path)


def test_load_discovery_v2_rejects_wrong_candidate_shape(tmp_path: Path):
    _prepare_dataset(tmp_path)
    record = json.loads((tmp_path / "discovery_v2.json").read_text())
    for key in ("binary_mask", "partial_corr", "scores"):
        record[key] = record[key][:3]  # drop the last child row -> (3, 8)
    record["threshold"] = record["threshold"][:3]
    record["candidate_shape"] = [3, 8]
    _rehash(record)
    (tmp_path / "discovery_v2.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    with pytest.raises(ValueError, match="candidate shape"):
        load_discovery_v2(tmp_path)


def test_discover_v2_refuses_stale_downstream_without_force(tmp_path: Path):
    _prepare_dataset(tmp_path)
    (tmp_path / "recovery_v2.json").write_text("{}")
    with pytest.raises(ValueError, match="downstream artifacts already exist"):
        write_discovery_v2(tmp_path, DiscoveryV2Config())
    write_discovery_v2(tmp_path, DiscoveryV2Config(), force=True)
    assert not (tmp_path / "recovery_v2.json").exists()


def _rehash(record: dict) -> None:
    """Recompute content_hash so a targeted guard (not the content_hash guard) is what rejects."""
    payload = {k: v for k, v in record.items() if k != "content_hash"}
    record["content_hash"] = hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def test_valid_discovery_v2_loads_after_rederivation(tmp_path: Path):
    _prepare_dataset(tmp_path)
    before = (tmp_path / "discovery_v2.json").read_text()
    load_discovery_v2(tmp_path)
    assert (tmp_path / "discovery_v2.json").read_text() == before


@pytest.mark.parametrize(
    "mutate, match",
    [
        pytest.param(lambda d: d.__setitem__("protocol_commit", "0" * 40), "protocol_commit",
                     id="protocol_commit"),
        pytest.param(lambda d: d.__setitem__("floor", 1e-3), "floor", id="floor"),
        pytest.param(lambda d: d.__setitem__("score_method", "ols_beta"), "score_method",
                     id="score_method"),
        pytest.param(lambda d: d.__setitem__("threshold_method", "pooled_largest_gap"),
                     "threshold_method", id="threshold_method"),
        pytest.param(
            lambda d: d["partial_corr"][0].__setitem__(0, d["partial_corr"][0][0] + 1.0),
            "absolute partial correlations", id="partial_corr_relation",
        ),
    ],
)
def test_load_discovery_v2_rejects_frozen_constant_or_relation_violation(
    tmp_path: Path, mutate, match: str
):
    _prepare_dataset(tmp_path)
    record = json.loads((tmp_path / "discovery_v2.json").read_text())
    mutate(record)
    _rehash(record)  # keep content_hash consistent so the NEW guard is what fires
    (tmp_path / "discovery_v2.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    with pytest.raises(ValueError, match=match):
        load_discovery_v2(tmp_path)


def test_load_discovery_v2_rejects_retuned_threshold(tmp_path: Path):
    # A rerun that quietly retunes a per-target threshold (but keeps that row's mask self-consistent
    # and re-hashes) is caught by the per-row largest_gap re-derivation.
    _prepare_dataset(tmp_path)
    record = json.loads((tmp_path / "discovery_v2.json").read_text())
    scores = np.asarray(record["scores"], dtype=np.float64)
    new_thr = float(record["threshold"][0]) + 1.0
    record["threshold"][0] = new_thr
    record["binary_mask"][0] = (scores[0] >= new_thr).astype(int).tolist()  # keep row 0 consistent
    _rehash(record)
    (tmp_path / "discovery_v2.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    with pytest.raises(ValueError, match="largest_gap"):
        load_discovery_v2(tmp_path)
