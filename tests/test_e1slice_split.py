"""Tests for the E1 slice split layer: episode-level, disjoint, deterministic, bound."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cdd_oran.e1slice.dataset import E1DatasetConfig, generate_rows, load_dataset, write_dataset
from cdd_oran.e1slice.split import (
    SplitConfig,
    load_split,
    make_split,
    row_indices_for,
    write_split,
)

_CFG = E1DatasetConfig(n_episodes=10, steps_per_episode=5, warmup=1, env_seed=0)


def test_split_is_disjoint_and_covers_all_episodes():
    episodes = list(range(_CFG.n_episodes))
    split = make_split(episodes, SplitConfig(test_fraction=0.2, split_seed=0))
    train, test = set(split["train"]), set(split["test"])
    assert train.isdisjoint(test)
    assert train | test == set(episodes)
    assert len(test) == 2  # round(10 * 0.2)


def test_split_is_deterministic_and_seed_sensitive():
    episodes = list(range(_CFG.n_episodes))
    a = make_split(episodes, SplitConfig(test_fraction=0.3, split_seed=0))
    b = make_split(episodes, SplitConfig(test_fraction=0.3, split_seed=0))
    assert a == b
    c = make_split(episodes, SplitConfig(test_fraction=0.3, split_seed=1))
    # A different seed generally reassigns membership (same sizes, different members).
    assert set(a["test"]) != set(c["test"])


def test_both_splits_nonempty_for_valid_fraction():
    split = make_split(list(range(4)), SplitConfig(test_fraction=0.01, split_seed=0))
    assert len(split["train"]) >= 1 and len(split["test"]) >= 1


def test_row_indices_never_cross_the_split():
    rows = generate_rows(_CFG)
    split = make_split(rows.episode.tolist(), SplitConfig(test_fraction=0.2, split_seed=0))
    train_idx = row_indices_for(rows, split["train"])
    test_idx = row_indices_for(rows, split["test"])
    # No row index appears in both, and together they cover every row exactly once.
    assert set(train_idx.tolist()).isdisjoint(test_idx.tolist())
    assert len(train_idx) + len(test_idx) == rows.n
    # Every selected train row belongs to a train episode (no transition-level leakage).
    assert set(rows.episode[train_idx].tolist()) == set(split["train"])
    assert set(rows.episode[test_idx].tolist()) == set(split["test"])


def test_write_split_binds_to_dataset_hash(tmp_path: Path):
    manifest = write_dataset(_CFG, tmp_path)
    record = write_split(tmp_path, SplitConfig(test_fraction=0.2, split_seed=0))
    assert (tmp_path / "split.json").exists()
    assert record["dataset_hash"] == manifest["dataset_hash"]
    assert load_split(tmp_path) == record
    # Split partitions the dataset's episodes with no overlap.
    assert set(record["train_episodes"]).isdisjoint(record["test_episodes"])
    assert (
        set(record["train_episodes"]) | set(record["test_episodes"])
        == set(np.arange(_CFG.n_episodes).tolist())
    )


def test_make_split_rejects_invalid_fraction():
    for frac in (0.0, 1.0, float("nan")):
        with pytest.raises(ValueError, match="test_fraction"):
            make_split(list(range(_CFG.n_episodes)), SplitConfig(test_fraction=frac, split_seed=0))


def test_load_split_rejects_a_tampered_hash(tmp_path: Path):
    write_dataset(_CFG, tmp_path)
    write_split(tmp_path, SplitConfig(test_fraction=0.2, split_seed=0))
    record = json.loads((tmp_path / "split.json").read_text())
    # Move an episode across the partition without recomputing split_hash.
    moved = record["train_episodes"].pop()
    record["test_episodes"].append(moved)
    (tmp_path / "split.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    with pytest.raises(ValueError, match="split_hash"):
        load_split(tmp_path)


def test_load_split_rejects_mismatched_dataset_binding(tmp_path: Path):
    manifest = write_dataset(_CFG, tmp_path)
    write_split(tmp_path, SplitConfig(test_fraction=0.2, split_seed=0))
    # Self-validation passes, but binding to a different dataset_hash must be refused.
    load_split(tmp_path)  # no binding args -> OK
    with pytest.raises(ValueError, match="dataset_hash"):
        load_split(tmp_path, expected_dataset_hash="f" * 64)
    # Coverage binding: a missing episode id in the union is rejected.
    episode_ids = set(range(_CFG.n_episodes)) | {999}
    with pytest.raises(ValueError, match="cover the dataset episodes"):
        load_split(tmp_path, expected_dataset_hash=manifest["dataset_hash"], episode_ids=episode_ids)


def test_write_split_refuses_stale_downstream_without_force(tmp_path: Path):
    write_dataset(_CFG, tmp_path)
    write_split(tmp_path, SplitConfig(test_fraction=0.2, split_seed=0))
    (tmp_path / "metrics.json").write_text("{}")  # stale downstream artifact
    with pytest.raises(ValueError, match="downstream artifacts already exist"):
        write_split(tmp_path, SplitConfig(test_fraction=0.3, split_seed=1))
    write_split(tmp_path, SplitConfig(test_fraction=0.3, split_seed=1), force=True)
    assert not (tmp_path / "metrics.json").exists()


def test_write_split_binds_union_to_dataset_episodes(tmp_path: Path):
    manifest = write_dataset(_CFG, tmp_path)
    write_split(tmp_path, SplitConfig(test_fraction=0.2, split_seed=0))
    rows, _ = load_dataset(tmp_path)
    ids = set(int(e) for e in rows.episode.tolist())
    record = load_split(tmp_path, expected_dataset_hash=manifest["dataset_hash"], episode_ids=ids)
    assert set(record["train_episodes"]) | set(record["test_episodes"]) == ids
