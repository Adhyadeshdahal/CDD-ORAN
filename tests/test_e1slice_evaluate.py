"""Tests for the E1 slice eval + verify layers: held-out metrics, immutability, reload."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cdd_oran.analysis.recovery_metrics import recovery_by_edge_type
from cdd_oran.e1slice.dataset import E1DatasetConfig, load_dataset, write_dataset
from cdd_oran.e1slice.discovery import discovered_mask_array, load_discovery, write_discovery
from cdd_oran.e1slice.evaluate import (
    evaluate_dataset,
    predict,
    regression_metrics,
    score_recovery,
    verify_dataset,
)
from cdd_oran.e1slice.model import Arm, ModelConfig, save_arm, train_arm
from cdd_oran.e1slice.split import SplitConfig, load_split, write_split

_CFG = E1DatasetConfig(n_episodes=12, steps_per_episode=8, warmup=2, env_seed=0)
_MODEL = ModelConfig(hidden=(16,), lr=1e-2, epochs=120, batch_size=64, weight_seed=0)
_ARMS: tuple[Arm, ...] = ("oracle", "dense", "discovered")
# Arms whose (correct/complete) mask must fit the linear E1 mechanism; the discovered arm's fit
# depends on empirical recovery quality, so it is NOT asserted to reach the same MSE floor.
_FITTING_ARMS: tuple[Arm, ...] = ("oracle", "dense")


def _prepare(dataset_dir: Path) -> None:
    """Full pipeline: generate -> split -> discover -> train (all three arms) with references."""
    write_dataset(_CFG, dataset_dir)
    write_split(dataset_dir, SplitConfig(test_fraction=0.25, split_seed=0))
    write_discovery(dataset_dir)
    rows, manifest = load_dataset(dataset_dir)
    split = load_split(dataset_dir)
    disc = load_discovery(dataset_dir)
    disc_mask = discovered_mask_array(disc)
    for arm in _ARMS:
        mask = disc_mask if arm == "discovered" else None
        model, meta = train_arm(rows, split["train_episodes"], arm, _MODEL, mask=mask)
        model.eval()
        reference = predict(model, rows, split["test_episodes"])
        save_arm(
            dataset_dir, model, meta, manifest["dataset_hash"], split["split_hash"], reference,
            discovery_hash=disc["content_hash"] if arm == "discovered" else None,
        )


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
        assert np.isfinite(record["arms"][arm]["mse"])
    for arm in _FITTING_ARMS:
        assert record["arms"][arm]["mse"] < 1e-2  # correct/complete mask fits the linear E1
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


def test_all_three_arms_share_identical_capacity(tmp_path: Path):
    _prepare(tmp_path)
    counts = set()
    for arm in _ARMS:
        meta = json.loads((tmp_path / "arms" / arm / "arm_meta.json").read_text())
        counts.add(meta["capacity"]["num_parameters"])
    assert len(counts) == 1  # identical architecture; only the fixed mask differs


def test_discovered_arm_mask_equals_persisted_graph(tmp_path: Path):
    from cdd_oran.e1slice.model import load_arm

    _prepare(tmp_path)
    disc = load_discovery(tmp_path)
    model, _ = load_arm(tmp_path, "discovered")
    np.testing.assert_array_equal(
        model.mask.numpy().astype(int), np.asarray(disc["binary_mask"], dtype=int)
    )


def test_changed_discovery_artifact_invalidates_eval_and_verify(tmp_path: Path):
    _prepare(tmp_path)
    # Corrupt discovery.json after training: its content_hash no longer matches its bytes, and
    # the discovered arm is bound to the original graph. Both consumers must fail closed.
    disc = json.loads((tmp_path / "discovery.json").read_text())
    disc["coefficients"][0][0] = float(disc["coefficients"][0][0]) + 1.0
    (tmp_path / "discovery.json").write_text(json.dumps(disc, indent=2, sort_keys=True))
    with pytest.raises(ValueError):
        evaluate_dataset(tmp_path)
    with pytest.raises(ValueError):
        verify_dataset(tmp_path)


def test_score_recovery_reads_truth_only_after_persistence(tmp_path: Path):
    _prepare(tmp_path)
    record = score_recovery(tmp_path)
    assert (tmp_path / "recovery.json").exists()
    # Recovery is bound to the frozen discovery artifact and E1's true edge count (6).
    disc = load_discovery(tmp_path)
    assert record["discovery_hash"] == disc["content_hash"]
    assert record["protocol_commit"] == disc["protocol_commit"]
    assert record["gt_edge_count"] == 6
    rec = record["recovery"]
    for block in ("overall", "ncp_kpi", "kpi_kpi"):
        assert 0.0 <= rec[block]["recall"] <= 1.0
        assert 0.0 <= rec[block]["precision"] <= 1.0


def test_full_graph_mapping_and_imperfect_recovery_fixture():
    from cdd_oran.e1slice.evaluate import full_graph_from_discovered_mask
    from cdd_oran.envs.v2.e1 import E1V2Env

    # A deliberately imperfect (4,8) discovered mask: correct NCP->KPI edges, but it MISSES the
    # KPI->KPI edge K2<-K0 and adds one false positive K0<-P1.
    mask = np.zeros((4, 8), dtype=int)
    mask[0, 0] = 1  # K0<-P0 (true)
    mask[0, 1] = 1  # K0<-P1 (FALSE POSITIVE)
    mask[1, 1] = 1  # K1<-P1 (true)
    mask[2, 2] = 1  # K2<-P2 (true), but K2<-K0 (col 4) intentionally MISSED
    mask[3, 3] = 1  # K3<-P3 (true)
    mask[3, 5] = 1  # K3<-K1 (true, col 5)

    pred = full_graph_from_discovered_mask(mask)
    gt = E1V2Env(env_seed=0).true_adj_matrix().astype(int)
    out = recovery_by_edge_type(pred, gt, E1V2Env.num_params)

    # The false positive K0<-P1 lands in the NCP->KPI block.
    assert out["ncp_kpi"]["fp"] == 1
    # The missed KPI->KPI edge is K2<-K0 (child row 6, parent col 4 in the (8,8) graph).
    missed = {(m["child"], m["parent"], m["type"]) for m in out["missed"]}
    assert (6, 4, "kpi_kpi") in missed
    assert out["kpi_kpi"]["fn"] >= 1
