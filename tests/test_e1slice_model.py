"""Tests for the E1 slice model layer: matched capacity, oracle masking, determinism."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from cdd_oran.e1slice.dataset import (
    E1DatasetConfig,
    generate_rows,
    load_dataset,
    write_dataset,
)
from cdd_oran.e1slice.evaluate import predict
from cdd_oran.e1slice.model import (
    Arm,
    ModelConfig,
    OneStepPredictor,
    arm_mask,
    build_model,
    load_arm,
    save_arm,
    train_arm,
    validate_explicit_mask,
)
from cdd_oran.e1slice.split import SplitConfig, load_split, make_split, write_split
from cdd_oran.envs.v2.e1 import E1V2Env

_CFG = E1DatasetConfig(n_episodes=12, steps_per_episode=8, warmup=2, env_seed=0)
_MODEL = ModelConfig(hidden=(16,), lr=1e-2, epochs=120, batch_size=64, weight_seed=0)
_ARMS: tuple[Arm, ...] = ("oracle", "dense")
_IN_DIM = E1V2Env.num_params + E1V2Env.num_kpis


def test_oracle_mask_matches_the_true_graph():
    mask = arm_mask("oracle").numpy()
    p, k = E1V2Env.num_params, E1V2Env.num_kpis
    expected = E1V2Env(env_seed=0).true_adj_matrix()[p : p + k, : p + k]
    np.testing.assert_array_equal(mask, expected)
    # E1 parents: K0<-P0, K1<-P1, K2<-{P2,K0}, K3<-{P3,K1}.
    assert mask[0].tolist() == [1, 0, 0, 0, 0, 0, 0, 0]
    assert mask[2].tolist() == [0, 0, 1, 0, 1, 0, 0, 0]


def test_dense_sees_everything():
    assert arm_mask("dense").sum().item() == E1V2Env.num_kpis * _IN_DIM


def test_arms_have_identical_capacity():
    # Same architecture -> identical parameter budget; only the input mask differs.
    oracle = build_model("oracle", _MODEL)
    dense = build_model("dense", _MODEL)
    # A discovered arm is built from an EXPLICIT mask (never env truth) and matches capacity.
    discovered = build_model("discovered", _MODEL, mask=arm_mask("oracle").numpy())
    assert oracle.num_parameters() == dense.num_parameters() == discovered.num_parameters()


def test_arm_mask_refuses_to_derive_discovered():
    # The discovered mask must come from discovery.json, never from environment truth.
    with pytest.raises(ValueError):
        arm_mask("discovered")
    with pytest.raises(ValueError):
        build_model("discovered", _MODEL)


@pytest.mark.parametrize(
    "bad",
    [
        np.ones((E1V2Env.num_kpis, _IN_DIM + 1), dtype=np.float32),
        np.full((E1V2Env.num_kpis, _IN_DIM), 2.0, dtype=np.float32),
    ],
)
def test_validate_explicit_mask_rejects_bad_masks(bad):
    with pytest.raises(ValueError):
        validate_explicit_mask(bad)


def test_oracle_ignores_non_parent_inputs():
    # Perturbing a NON-parent input must not change an oracle head's output.
    model = build_model("oracle", _MODEL)
    model.eval()
    x = torch.zeros(1, _IN_DIM)
    base = model(x)
    x_perturbed = x.clone()
    x_perturbed[0, 1] = 5.0  # P1 is NOT a parent of K0 or K2
    out = model(x_perturbed)
    assert torch.allclose(base[:, 0], out[:, 0])
    assert torch.allclose(base[:, 2], out[:, 2])


def test_training_fits_the_linear_mechanism():
    rows = generate_rows(_CFG)
    split = make_split(rows.episode.tolist(), SplitConfig(test_fraction=0.25, split_seed=0))
    for arm in _ARMS:
        _, meta = train_arm(rows, split["train"], arm, _MODEL)
        # E1 is linear; both arms should drive training MSE low (recovery control).
        assert meta["final_train_mse"] < 1e-2


def test_training_is_deterministic():
    rows = generate_rows(_CFG)
    split = make_split(rows.episode.tolist(), SplitConfig(test_fraction=0.25, split_seed=0))
    m1, _ = train_arm(rows, split["train"], "oracle", _MODEL)
    m2, _ = train_arm(rows, split["train"], "oracle", _MODEL)
    x = torch.randn(5, _IN_DIM)
    m1.eval()
    m2.eval()
    assert torch.allclose(m1(x), m2(x))


def test_save_load_reproduces_predictions(tmp_path: Path):
    write_dataset(_CFG, tmp_path)
    write_split(tmp_path, SplitConfig(test_fraction=0.25, split_seed=0))
    rows, ds_manifest = load_dataset(tmp_path)
    split = load_split(tmp_path)

    model, meta = train_arm(rows, split["train_episodes"], "oracle", _MODEL)
    model.eval()
    reference = predict(model, rows, split["test_episodes"])
    save_arm(tmp_path, model, meta, ds_manifest["dataset_hash"], split["split_hash"], reference)

    reloaded, reloaded_meta = load_arm(tmp_path, "oracle")
    x = torch.randn(7, _IN_DIM)
    assert torch.allclose(model(x), reloaded(x), atol=1e-6)
    assert isinstance(reloaded, OneStepPredictor)
    # Digests for both persisted artifacts are recorded and verified on reload.
    assert reloaded_meta["model_sha256"] and reloaded_meta["ref_sha256"]


def test_load_arm_rejects_tampered_weights(tmp_path: Path):
    write_dataset(_CFG, tmp_path)
    write_split(tmp_path, SplitConfig(test_fraction=0.25, split_seed=0))
    rows, ds_manifest = load_dataset(tmp_path)
    split = load_split(tmp_path)
    model, meta = train_arm(rows, split["train_episodes"], "oracle", _MODEL)
    model.eval()
    reference = predict(model, rows, split["test_episodes"])
    save_arm(tmp_path, model, meta, ds_manifest["dataset_hash"], split["split_hash"], reference)

    # Flip one byte of the saved weights: the recorded model_sha256 no longer matches.
    weights = tmp_path / "arms" / "oracle" / "model.pt"
    raw = bytearray(weights.read_bytes())
    raw[-1] ^= 0x01
    weights.write_bytes(raw)
    with pytest.raises(ValueError, match="model_sha256"):
        load_arm(tmp_path, "oracle")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden": ()},
        {"hidden": (0,)},
        {"lr": 0.0},
        {"lr": float("nan")},
        {"epochs": 0},
        {"batch_size": 0},
    ],
)
def test_build_model_rejects_invalid_config(kwargs):
    from dataclasses import replace

    with pytest.raises(ValueError):
        build_model("dense", replace(_MODEL, **kwargs))
