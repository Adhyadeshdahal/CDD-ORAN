"""Tests for the E1 slice eval + verify layers: held-out metrics, immutability, reload."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cdd_oran.e1slice.dataset import E1DatasetConfig, load_dataset, write_dataset
from cdd_oran.e1slice.evaluate import (
    evaluate_dataset,
    predict,
    regression_metrics,
    verify_dataset,
)
from cdd_oran.e1slice.model import Arm, ModelConfig, save_arm, train_arm
from cdd_oran.e1slice.split import SplitConfig, load_split, write_split

_CFG = E1DatasetConfig(n_episodes=12, steps_per_episode=8, warmup=2, env_seed=0)
_MODEL = ModelConfig(hidden=(16,), lr=1e-2, epochs=120, batch_size=64, weight_seed=0)
_ARMS: tuple[Arm, ...] = ("oracle", "dense")


def _prepare(dataset_dir: Path) -> None:
    """Full pipeline: generate -> split -> train (weights + reference captured together)."""
    write_dataset(_CFG, dataset_dir)
    write_split(dataset_dir, SplitConfig(test_fraction=0.25, split_seed=0))
    rows, manifest = load_dataset(dataset_dir)
    split = load_split(dataset_dir)
    for arm in _ARMS:
        model, meta = train_arm(rows, split["train_episodes"], arm, _MODEL)
        model.eval()
        reference = predict(model, rows, split["test_episodes"])
        save_arm(dataset_dir, model, meta, manifest["dataset_hash"], split["split_hash"], reference)


def test_regression_metrics_are_zero_on_perfect_prediction():
    y = np.random.default_rng(0).standard_normal((10, 4))
    m = regression_metrics(y.copy(), y)
    assert m["mse"] == 0.0 and m["mae"] == 0.0
    assert m["per_kpi_mse"] == [0.0, 0.0, 0.0, 0.0]


def test_eval_writes_metrics_over_held_out_episodes(tmp_path: Path):
    _prepare(tmp_path)
    record = evaluate_dataset(tmp_path)
    assert (tmp_path / "metrics.json").exists()
    # Metrics computed only over the held-out test episodes.
    rows, _ = load_dataset(tmp_path)
    split = load_split(tmp_path)
    n_test_rows = int(np.isin(rows.episode, split["test_episodes"]).sum())
    for arm in _ARMS:
        assert record["arms"][arm]["n_test_rows"] == n_test_rows
        assert record["arms"][arm]["mse"] < 1e-2  # E1 recovery control: both generalize
    assert record["dataset_hash"] == json.loads(
        (tmp_path / "manifest.json").read_text()
    )["dataset_hash"]


def test_eval_is_immutable_deterministic(tmp_path: Path):
    _prepare(tmp_path)
    a = evaluate_dataset(tmp_path)
    b = evaluate_dataset(tmp_path)
    assert a["metrics_hash"] == b["metrics_hash"]
    assert a["arms"] == b["arms"]


def test_verify_passes_for_a_faithful_roundtrip(tmp_path: Path):
    _prepare(tmp_path)
    results = verify_dataset(tmp_path, tol=1e-6)
    assert {r.arm for r in results} == set(_ARMS)
    for r in results:
        assert r.passed
        assert r.max_abs_diff <= 1e-6


def test_verify_rejects_a_corrupted_reference(tmp_path: Path):
    _prepare(tmp_path)
    # Corrupt the oracle reference: its recorded ref_sha256 no longer matches the bytes.
    ref_path = tmp_path / "arms" / "oracle" / "test_pred_ref.npz"
    with np.load(ref_path) as data:
        bad = data["pred"] + 1.0
    np.savez(ref_path, pred=bad)
    with pytest.raises(ValueError, match="ref_sha256"):
        verify_dataset(tmp_path, tol=1e-6)


def test_eval_rejects_a_corrupted_reference(tmp_path: Path):
    _prepare(tmp_path)
    ref_path = tmp_path / "arms" / "dense" / "test_pred_ref.npz"
    with np.load(ref_path) as data:
        bad = data["pred"] + 1.0
    np.savez(ref_path, pred=bad)
    with pytest.raises(ValueError, match="ref_sha256"):
        evaluate_dataset(tmp_path)


def test_changed_split_is_rejected_by_eval_and_verify(tmp_path: Path):
    from cdd_oran.e1slice.split import build_split_record, make_split

    _prepare(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    rows, _ = load_dataset(tmp_path)
    # Rewrite split.json with a DIFFERENT seed (new, internally-valid split_hash) while the
    # trained arms stay on disk, still bound to the ORIGINAL split_hash.
    other = SplitConfig(test_fraction=0.25, split_seed=7)
    record = build_split_record(make_split(rows.episode.tolist(), other), other, manifest["dataset_hash"])
    (tmp_path / "split.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    # The arms bind to the original split_hash; both consumers must refuse before producing results.
    with pytest.raises(ValueError, match="split_hash"):
        evaluate_dataset(tmp_path)
    with pytest.raises(ValueError, match="split_hash"):
        verify_dataset(tmp_path)


def test_eval_refuses_stale_metrics_without_force(tmp_path: Path):
    _prepare(tmp_path)
    evaluate_dataset(tmp_path)
    # A stale metrics.json is a downstream artifact of train; retraining must refuse it.
    rows, manifest = load_dataset(tmp_path)
    split = load_split(tmp_path)
    model, meta = train_arm(rows, split["train_episodes"], "oracle", _MODEL)
    model.eval()
    reference = predict(model, rows, split["test_episodes"])
    with pytest.raises(ValueError, match="metrics.json"):
        save_arm(tmp_path, model, meta, manifest["dataset_hash"], split["split_hash"], reference)


@pytest.mark.parametrize(
    "pred, target",
    [
        (np.zeros((2, 3)), np.zeros((2, 4))),  # shape mismatch
        (np.zeros((0, 4)), np.zeros((0, 4))),  # empty
        (np.zeros(4), np.zeros(4)),            # not 2-D
        (np.full((2, 4), np.nan), np.zeros((2, 4))),  # non-finite
    ],
)
def test_regression_metrics_rejects_bad_inputs(pred, target):
    with pytest.raises(ValueError):
        regression_metrics(pred, target)
