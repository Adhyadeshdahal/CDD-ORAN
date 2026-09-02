"""Tests for the E1 slice model layer: matched capacity, oracle masking, determinism."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cdd_oran.e1slice.dataset import (
    E1DatasetConfig,
    generate_rows,
    load_dataset,
    write_dataset,
)
from cdd_oran.e1slice.model import (
    Arm,
    ModelConfig,
    OneStepPredictor,
    arm_mask,
    build_model,
    load_arm,
    save_arm,
    train_arm,
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
    assert oracle.num_parameters() == dense.num_parameters()


def test_oracle_ignores_non_parent_inputs():
    # Perturbing a NON-parent input must not change an oracle head's output.
    model = build_model("oracle", _MODEL)
    model.eval()
    x = torch.zeros(1, _IN_DIM)
    base = model(x)
    x_perturbed = x.clone()
    x_perturbed[0, 1] = 5.0  # P1 is NOT a parent of K0 or K2
    out = model(x_perturbed)
    assert torch.allclose(base[:, 0], out[:, 0])  # K0 unchanged
    assert torch.allclose(base[:, 2], out[:, 2])  # K2 unchanged


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
    save_arm(tmp_path, model, meta, ds_manifest["dataset_hash"], split["split_hash"])

    reloaded, _ = load_arm(tmp_path, "oracle")
    x = torch.randn(7, _IN_DIM)
    model.eval()
    assert torch.allclose(model(x), reloaded(x), atol=1e-6)
    assert isinstance(reloaded, OneStepPredictor)
