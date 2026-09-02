"""Tests for the E1 slice eval + verify layers: held-out metrics, immutability, reload."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cdd_oran.e1slice.dataset import E1DatasetConfig, load_dataset, write_dataset
from cdd_oran.e1slice.evaluate import (
    evaluate_dataset,
    predict,
    regression_metrics,
    save_reference_predictions,
    verify_dataset,
)
from cdd_oran.e1slice.model import Arm, ModelConfig, save_arm, train_arm
from cdd_oran.e1slice.split import SplitConfig, load_split, write_split

_CFG = E1DatasetConfig(n_episodes=12, steps_per_episode=8, warmup=2, env_seed=0)
_MODEL = ModelConfig(hidden=(16,), lr=1e-2, epochs=120, batch_size=64, weight_seed=0)
_ARMS: tuple[Arm, ...] = ("oracle", "dense")


def _prepare(dataset_dir: Path) -> None:
    """Full pipeline: generate -> split -> train + save reference predictions."""
    write_dataset(_CFG, dataset_dir)
    write_split(dataset_dir, SplitConfig(test_fraction=0.25, split_seed=0))
    rows, manifest = load_dataset(dataset_dir)
    split = load_split(dataset_dir)
    for arm in _ARMS:
        model, meta = train_arm(rows, split["train_episodes"], arm, _MODEL)
        save_arm(dataset_dir, model, meta, manifest["dataset_hash"], split["split_hash"])
        model.eval()
        save_reference_predictions(dataset_dir, arm, predict(model, rows, split["test_episodes"]))


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


def test_verify_fails_if_the_reference_is_corrupted(tmp_path: Path):
    _prepare(tmp_path)
    # Corrupt the oracle reference so the reloaded artifact no longer matches it.
    ref_path = tmp_path / "arms" / "oracle" / "test_pred_ref.npz"
    with np.load(ref_path) as data:
        bad = data["pred"] + 1.0
    np.savez(ref_path, pred=bad)
    results = {r.arm: r for r in verify_dataset(tmp_path, tol=1e-6)}
    assert not results["oracle"].passed
    assert results["dense"].passed  # untouched arm still round-trips
