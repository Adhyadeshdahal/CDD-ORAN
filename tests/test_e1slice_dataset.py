"""Tests for the E1 slice dataset layer: determinism, schema, provenance, round-trip."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cdd_oran.e1slice import SCHEMA_VERSION
from cdd_oran.e1slice.dataset import (
    E1DatasetConfig,
    dataset_hash,
    generate_rows,
    load_dataset,
    scm_hash,
    write_dataset,
)
from cdd_oran.envs.v2.e1 import E1V2Env

_CFG = E1DatasetConfig(n_episodes=8, steps_per_episode=6, warmup=2, env_seed=0)


def test_row_shapes_and_count():
    rows = generate_rows(_CFG)
    assert rows.n == _CFG.n_episodes * _CFG.steps_per_episode
    assert rows.x_params.shape == (rows.n, E1V2Env.num_params)
    assert rows.x_kpis.shape == (rows.n, E1V2Env.num_kpis)
    assert rows.y_kpis.shape == (rows.n, E1V2Env.num_kpis)
    # Every episode contributes exactly steps_per_episode rows, at contiguous time coordinates.
    assert sorted(set(rows.episode.tolist())) == list(range(_CFG.n_episodes))
    assert np.bincount(rows.episode).tolist() == [_CFG.steps_per_episode] * _CFG.n_episodes


def test_generation_is_deterministic():
    a = generate_rows(_CFG)
    b = generate_rows(_CFG)
    assert dataset_hash(a) == dataset_hash(b)
    np.testing.assert_array_equal(a.x_params, b.x_params)
    np.testing.assert_array_equal(a.y_kpis, b.y_kpis)


def test_label_matches_the_scm_mechanism():
    # y must equal the exact E1 mechanism applied to the recorded (x_params, x_kpis).
    rows = generate_rows(_CFG)
    probe = E1V2Env(env_seed=_CFG.env_seed, obs_noise_scale=0.0)
    for i in range(rows.n):
        expected = probe._update_kpis(rows.x_params[i], rows.x_kpis[i])
        np.testing.assert_allclose(rows.y_kpis[i], expected, rtol=0.0, atol=1e-12)


def test_noise_off_by_default():
    assert _CFG.obs_noise_scale == 0.0  # latent == observed for the recovery control


def test_scm_hash_tracks_the_mechanism():
    base = scm_hash(_CFG)
    # A different SCM coefficient (a shadow SCM) must change the hash.
    from dataclasses import replace

    assert scm_hash(replace(_CFG, obs_noise_scale=0.5)) != base


def test_write_load_roundtrip_and_manifest(tmp_path: Path):
    manifest = write_dataset(_CFG, tmp_path)
    assert (tmp_path / "rows.npz").exists()
    assert (tmp_path / "manifest.json").exists()

    # Manifest carries the full provenance contract.
    for key in (
        "schema_version",
        "n_rows",
        "config",
        "seeds",
        "dataset_hash",
        "scm_hash",
        "scm_identity",
        "git_sha",
    ):
        assert key in manifest
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["n_rows"] == _CFG.n_episodes * _CFG.steps_per_episode

    rows, loaded_manifest = load_dataset(tmp_path)
    assert loaded_manifest == json.loads((tmp_path / "manifest.json").read_text())
    # Reloaded rows reproduce the persisted content hash exactly.
    assert dataset_hash(rows) == manifest["dataset_hash"]


def test_load_rejects_a_single_tampered_row_byte(tmp_path: Path):
    write_dataset(_CFG, tmp_path)
    # Flip one value in the persisted rows: the recomputed dataset_hash must no longer match.
    with np.load(tmp_path / "rows.npz") as data:
        cols = {name: data[name] for name in data.files}
    cols["x_params"][0, 0] += 1.0
    np.savez(tmp_path / "rows.npz", **cols)
    with pytest.raises(ValueError, match="dataset_hash"):
        load_dataset(tmp_path)


def test_load_rejects_a_manifest_hash_mismatch(tmp_path: Path):
    write_dataset(_CFG, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["dataset_hash"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    with pytest.raises(ValueError, match="dataset_hash"):
        load_dataset(tmp_path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_episodes": 1},
        {"steps_per_episode": 0},
        {"warmup": -1},
        {"obs_noise_scale": 0.5},
    ],
)
def test_generate_rejects_invalid_config(kwargs):
    with pytest.raises(ValueError):
        generate_rows(replace(_CFG, **kwargs))


def test_nonzero_observation_noise_is_rejected():
    # This latent recovery slice stays noiseless: nonzero obs noise is an error, not a feature.
    with pytest.raises(ValueError, match="obs_noise_scale"):
        write_dataset(replace(_CFG, obs_noise_scale=1e-3), Path("unused"))


def test_generate_refuses_to_clobber_downstream_without_force(tmp_path: Path):
    write_dataset(_CFG, tmp_path)
    (tmp_path / "split.json").write_text("{}")  # a stale downstream artifact
    with pytest.raises(ValueError, match="downstream artifacts already exist"):
        write_dataset(_CFG, tmp_path)
    # With force, the known descendant is cleared and generation republishes cleanly.
    write_dataset(_CFG, tmp_path, force=True)
    assert not (tmp_path / "split.json").exists()


def test_failed_write_leaves_no_partial_final(tmp_path: Path, monkeypatch):
    write_dataset(_CFG, tmp_path)
    good = (tmp_path / "manifest.json").read_text()

    import cdd_oran.e1slice.dataset as ds

    def boom(path, text):
        raise OSError("simulated write failure")

    monkeypatch.setattr(ds, "_atomic_write_text", boom)
    with pytest.raises(OSError):
        write_dataset(_CFG, tmp_path, force=True)
    # rows.npz was republished, but the manifest publish failed: the old manifest is intact,
    # not a half-written file, and no *.tmp final artifact was left behind.
    assert (tmp_path / "manifest.json").read_text() == good
    assert not (tmp_path / "manifest.json.tmp").exists()
