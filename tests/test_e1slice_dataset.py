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


def test_forced_generate_clears_stale_discovery_and_recovery(tmp_path: Path):
    # Finding #7: a forced regenerate must not leave discovery/recovery bound to the old ancestor.
    write_dataset(_CFG, tmp_path)
    (tmp_path / "discovery.json").write_text("{}")
    (tmp_path / "recovery.json").write_text("{}")
    # Without force, their presence blocks the rerun.
    with pytest.raises(ValueError, match="downstream artifacts already exist"):
        write_dataset(_CFG, tmp_path)
    # With force, guard_descendants removes both stale outputs.
    write_dataset(_CFG, tmp_path, force=True)
    assert not (tmp_path / "discovery.json").exists()
    assert not (tmp_path / "recovery.json").exists()


# --- Finding #5: dataset provenance rehash must cover scm_hash + seed consistency ----------
def test_load_rejects_scm_hash_mismatch(tmp_path: Path):
    write_dataset(_CFG, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["scm_hash"] = "0" * 64  # rows + dataset_hash stay valid; only scm_hash is wrong
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    with pytest.raises(ValueError, match="scm_hash"):
        load_dataset(tmp_path)


def test_load_rejects_inconsistent_env_seed(tmp_path: Path):
    write_dataset(_CFG, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["seeds"]["env_seed"] = int(manifest["config"]["env_seed"]) + 1
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    with pytest.raises(ValueError, match="env_seed"):
        load_dataset(tmp_path)


def test_load_rejects_manifest_missing_warmup(tmp_path: Path):
    write_dataset(_CFG, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    del manifest["config"]["warmup"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    with pytest.raises(ValueError, match="warmup"):
        load_dataset(tmp_path)


def test_valid_manifest_still_loads_after_rederivation(tmp_path: Path):
    # The new re-derivations (scm_hash, seed consistency) leave a correct manifest byte-identical.
    manifest = write_dataset(_CFG, tmp_path)
    rows, loaded = load_dataset(tmp_path)
    assert loaded == json.loads((tmp_path / "manifest.json").read_text())
    assert loaded["scm_hash"] == manifest["scm_hash"] == scm_hash(_CFG)
    assert dataset_hash(rows) == manifest["dataset_hash"]


# --- Finding #6: rows.npz + manifest.json publish is an atomic pair ------------------------
def test_manifest_write_failure_after_config_change_keeps_prior_pair(tmp_path: Path, monkeypatch):
    import cdd_oran.e1slice.dataset as ds

    # A valid dataset from config A is on disk.
    write_dataset(_CFG, tmp_path)
    old_rows = (tmp_path / "rows.npz").read_bytes()
    old_manifest = (tmp_path / "manifest.json").read_text()
    old_dataset_hash = json.loads(old_manifest)["dataset_hash"]

    # Republish a DIFFERENT config B (its rows differ), but the manifest staging fails mid-write.
    cfg_b = replace(_CFG, n_episodes=_CFG.n_episodes + 2)

    def boom(tmp, text):
        raise OSError("simulated manifest write failure")

    monkeypatch.setattr(ds, "_stage_text", boom)
    with pytest.raises(OSError):
        write_dataset(cfg_b, tmp_path, force=True)

    # The prior valid pair survives whole: never a NEW-rows + OLD-manifest state, and no temp
    # artifacts left behind.
    assert (tmp_path / "rows.npz").read_bytes() == old_rows
    assert (tmp_path / "manifest.json").read_text() == old_manifest
    assert not (tmp_path / "rows.npz.tmp").exists()
    assert not (tmp_path / "manifest.json.tmp").exists()
    # And the surviving pair still loads and binds.
    _, manifest = load_dataset(tmp_path)
    assert manifest["dataset_hash"] == old_dataset_hash
