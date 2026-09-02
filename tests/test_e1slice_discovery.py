"""Tests for the frozen E1 discovery layer.

Recovery quality on the REAL E1 SCM is an empirical result, not a unit-test constant; these
tests use synthetic equations with a KNOWN support to prove the mechanism (train-only fit,
label-free thresholding), determinism, train-only behaviour, invalid-input rejection, and
artifact corruption handling.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cdd_oran.e1slice.dataset import E1DatasetConfig, E1Rows, canonical_json, write_dataset
from cdd_oran.e1slice.discovery import (
    PROTOCOL_COMMIT,
    DiscoveryConfig,
    discover_graph,
    load_discovery,
    write_discovery,
)
from cdd_oran.e1slice.split import SplitConfig, load_split, write_split

_N_EP = 10
_PER = 30
# Known synthetic support: each output has ONE strong single parent; a weak shared false feature
# (col 1) and the pure-noise features must be excluded by the label-free threshold.
_STRONG_PARENT = {0: 0, 1: 5, 2: 2, 3: 7}  # output -> input column (0..3 params, 4..7 kpis)


def _synthetic_rows(seed: int = 0, tweak_test_labels: bool = False) -> E1Rows:
    """10 episodes x 30 steps of iid features with a known sparse linear label map."""
    rng = np.random.default_rng(seed)
    n = _N_EP * _PER
    feats = rng.standard_normal((n, 8))
    y = np.zeros((n, 4), dtype=np.float64)
    for out, col in _STRONG_PARENT.items():
        y[:, out] = feats[:, col] + 0.02 * feats[:, 1]  # strong parent + weak false feature
    episode = np.repeat(np.arange(_N_EP), _PER).astype(np.int64)
    time = np.tile(np.arange(_PER), _N_EP).astype(np.int64)
    if tweak_test_labels:
        # Corrupt ONLY the held-out episodes' labels; train-only discovery must be unaffected.
        y[np.isin(episode, [8, 9])] += 1000.0
    return E1Rows(
        episode=episode,
        time=time,
        x_params=feats[:, :4].astype(np.float64),
        x_kpis=feats[:, 4:].astype(np.float64),
        y_kpis=y,
    )


_TRAIN = list(range(8))
_TEST = [8, 9]


def _expected_mask() -> np.ndarray:
    mask = np.zeros((4, 8), dtype=int)
    for out, col in _STRONG_PARENT.items():
        mask[out, col] = 1
    return mask


def test_recovers_known_sparse_support():
    result = discover_graph(_synthetic_rows(), _TRAIN)
    np.testing.assert_array_equal(result.binary_mask, _expected_mask())


def test_row_order_shuffle_does_not_change_mask():
    rows = _synthetic_rows()
    base = discover_graph(rows, _TRAIN).binary_mask
    perm = np.random.default_rng(1).permutation(rows.n)
    shuffled = E1Rows(
        episode=rows.episode[perm],
        time=rows.time[perm],
        x_params=rows.x_params[perm],
        x_kpis=rows.x_kpis[perm],
        y_kpis=rows.y_kpis[perm],
    )
    np.testing.assert_array_equal(discover_graph(shuffled, _TRAIN).binary_mask, base)


def test_discovery_is_deterministic():
    a = discover_graph(_synthetic_rows(), _TRAIN)
    b = discover_graph(_synthetic_rows(), _TRAIN)
    np.testing.assert_array_equal(a.binary_mask, b.binary_mask)
    np.testing.assert_array_equal(a.coefficients, b.coefficients)
    assert a.threshold == b.threshold


def test_changing_held_out_rows_cannot_change_the_mask():
    # Discovery reads TRAIN rows only, so wrecking the held-out labels leaves the mask identical.
    clean = discover_graph(_synthetic_rows(tweak_test_labels=False), _TRAIN).binary_mask
    tweaked = discover_graph(_synthetic_rows(tweak_test_labels=True), _TRAIN).binary_mask
    np.testing.assert_array_equal(clean, tweaked)


def test_rejects_zero_variance_feature():
    rows = _synthetic_rows()
    dead = rows.x_params.copy()
    dead[:, 3] = 1.0  # constant column -> zero variance
    rows = E1Rows(rows.episode, rows.time, dead, rows.x_kpis, rows.y_kpis)
    with pytest.raises(ValueError, match="zero-variance"):
        discover_graph(rows, _TRAIN)


def test_rejects_rank_deficient_design():
    rows = _synthetic_rows()
    dup = rows.x_kpis.copy()
    dup[:, 0] = rows.x_params[:, 0]  # exact duplicate of an input column -> rank deficiency
    rows = E1Rows(rows.episode, rows.time, rows.x_params, dup, rows.y_kpis)
    with pytest.raises(ValueError, match="rank-deficient"):
        discover_graph(rows, _TRAIN)


def test_rejects_empty_training_rows():
    with pytest.raises(ValueError, match="no training rows"):
        discover_graph(_synthetic_rows(), [])


def test_discovery_config_freezes_floor_and_method():
    # The primary threshold is not tunable: only the frozen floor/method are accepted.
    assert DiscoveryConfig().floor == 1e-3 and DiscoveryConfig().method == "largest_gap"
    with pytest.raises(ValueError, match="floor is frozen"):
        DiscoveryConfig(floor=0.5)
    with pytest.raises(ValueError, match="method is frozen"):
        DiscoveryConfig(method="otsu")


def test_does_not_import_env_truth():
    import ast

    import cdd_oran.e1slice.discovery as disc

    tree = ast.parse(Path(disc.__file__).read_text())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    # The discovery module must not import the env or any true-adjacency symbol.
    assert not any("envs" in name or "E1V2Env" in name or "true_adj" in name for name in imported)
    # And the loaded module has no env truth bound in its namespace.
    assert not hasattr(disc, "E1V2Env")


# --- persistence + binding + corruption (real small dataset) ---------------------------------

_CFG = E1DatasetConfig(n_episodes=12, steps_per_episode=8, warmup=2, env_seed=0)


def _prepare_dataset(tmp_path: Path) -> dict:
    write_dataset(_CFG, tmp_path)
    write_split(tmp_path, SplitConfig(test_fraction=0.25, split_seed=0))
    return write_discovery(tmp_path)


def test_write_discovery_binds_and_records_protocol(tmp_path: Path):
    record = _prepare_dataset(tmp_path)
    assert (tmp_path / "discovery.json").exists()
    assert record["protocol_commit"] == PROTOCOL_COMMIT
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    split = load_split(tmp_path)
    assert record["dataset_hash"] == manifest["dataset_hash"]
    assert record["split_hash"] == split["split_hash"]
    # Fully-bound load succeeds; mask is (num_kpis, num_params+num_kpis).
    loaded = load_discovery(
        tmp_path,
        expected_dataset_hash=manifest["dataset_hash"],
        expected_split_hash=split["split_hash"],
    )
    assert np.asarray(loaded["binary_mask"]).shape == (4, 8)


def test_load_discovery_rejects_mismatched_binding(tmp_path: Path):
    _prepare_dataset(tmp_path)
    with pytest.raises(ValueError, match="dataset_hash"):
        load_discovery(tmp_path, expected_dataset_hash="f" * 64)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    with pytest.raises(ValueError, match="split_hash"):
        load_discovery(
            tmp_path, expected_dataset_hash=manifest["dataset_hash"], expected_split_hash="0" * 64
        )


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda d: d["coefficients"][0].__setitem__(0, d["coefficients"][0][0] + 1.0), id="coefficients"),
        pytest.param(lambda d: d.__setitem__("threshold", float(d["threshold"]) + 1.0), id="threshold"),
        pytest.param(lambda d: d["binary_mask"][0].__setitem__(3, 1 - d["binary_mask"][0][3]), id="mask"),
        pytest.param(lambda d: d.__setitem__("dataset_hash", "0" * 64), id="parent_hash"),
    ],
)
def test_corrupted_discovery_is_rejected(tmp_path: Path, mutate):
    _prepare_dataset(tmp_path)
    record = json.loads((tmp_path / "discovery.json").read_text())
    mutate(record)  # mutate a field WITHOUT recomputing content_hash
    (tmp_path / "discovery.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    with pytest.raises(ValueError):
        load_discovery(tmp_path)


def test_load_discovery_rejects_wrong_candidate_shape(tmp_path: Path):
    _prepare_dataset(tmp_path)
    record = json.loads((tmp_path / "discovery.json").read_text())
    # Drop the last child row -> (3, 8); keep the payload internally consistent and re-hash so the
    # SHAPE guard (not the content_hash guard) is what rejects it.
    for key in ("binary_mask", "coefficients", "scores"):
        record[key] = record[key][:3]
    record["candidate_shape"] = [3, 8]
    payload = {k: v for k, v in record.items() if k != "content_hash"}
    record["content_hash"] = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    (tmp_path / "discovery.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    with pytest.raises(ValueError, match="candidate shape"):
        load_discovery(tmp_path)


def test_discover_refuses_stale_downstream_without_force(tmp_path: Path):
    _prepare_dataset(tmp_path)
    (tmp_path / "metrics.json").write_text("{}")  # stale downstream artifact
    with pytest.raises(ValueError, match="downstream artifacts already exist"):
        write_discovery(tmp_path, DiscoveryConfig())
    write_discovery(tmp_path, DiscoveryConfig(), force=True)
    assert not (tmp_path / "metrics.json").exists()
