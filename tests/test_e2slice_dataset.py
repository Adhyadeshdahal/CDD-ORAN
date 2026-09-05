"""Tests for the E2 slice dataset layer: §4 generation, determinism, provenance, round-trip."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cdd_oran.e2slice import SCHEMA_VERSION
from cdd_oran.e2slice.dataset import (
    E2DatasetConfig,
    dataset_hash,
    generate_rows,
    load_dataset,
    scm_hash,
    write_dataset,
)
from cdd_oran.envs.v2.e2 import E2V2Env

_CFG = E2DatasetConfig(n_rows_per_seed=64, seed=0, sampling_seed=0)


def test_row_shapes_and_count():
    rows = generate_rows(_CFG)
    assert rows.n == _CFG.n_rows_per_seed
    assert rows.x_params.shape == (rows.n, E2V2Env.num_params)  # (n, 8)
    assert rows.x_kpis.shape == (rows.n, E2V2Env.num_kpis)      # (n, 6)
    assert rows.y_kpis.shape == (rows.n, E2V2Env.num_kpis)      # (n, 6)


def test_params_are_within_id_ranges():
    rows = generate_rows(_CFG)
    for i, (low, high) in enumerate(E2V2Env.id_ranges):
        assert rows.x_params[:, i].min() >= low
        assert rows.x_params[:, i].max() <= high


def test_generation_is_deterministic():
    a = generate_rows(_CFG)
    b = generate_rows(_CFG)
    assert dataset_hash(a) == dataset_hash(b)
    np.testing.assert_array_equal(a.x_params, b.x_params)
    np.testing.assert_array_equal(a.y_kpis, b.y_kpis)


def test_label_matches_the_scm_mechanism_of_x_params():
    # §4: Y = f(P_t) under the TRUE SCM (decoy OFF). The label equals the exact E2 mechanism applied
    # to the recorded x_params (the mechanism ignores lagged KPIs), NOT to x_kpis.
    rows = generate_rows(_CFG)
    probe = E2V2Env(env_seed=_CFG.seed, obs_noise_scale=0.0, decoy_omit_p0_k5=False)
    zeros = np.zeros(E2V2Env.num_kpis)
    for i in range(rows.n):
        expected = probe._update_kpis(rows.x_params[i], zeros)
        np.testing.assert_allclose(rows.y_kpis[i], expected, rtol=0.0, atol=1e-12)


def test_lagged_kpi_is_the_previous_rows_mechanism_output():
    # §4 actuation latency: x_kpis[m] (K_t) == f(P_{m-1}) == y_kpis[m-1] for m >= 1.
    rows = generate_rows(_CFG)
    np.testing.assert_allclose(rows.x_kpis[1:], rows.y_kpis[:-1], rtol=0.0, atol=1e-12)


def test_lagged_kpi_is_independent_of_x_params_in_expectation():
    # The lagged KPI is a function of the PREVIOUS (independent) param draw, so it is uncorrelated
    # with the current x_params -- the 36 lagged KPI->KPI candidates are true-negatives.
    rows = generate_rows(replace(_CFG, n_rows_per_seed=2000))
    corr = np.corrcoef(rows.x_params[:, 0], rows.x_kpis[:, 0])[0, 1]
    assert abs(corr) < 0.1  # near-zero on a large sample (exact-zero population dependence)


def test_seeds_differ_across_the_envelope():
    a = generate_rows(replace(_CFG, seed=0))
    b = generate_rows(replace(_CFG, seed=1))
    assert dataset_hash(a) != dataset_hash(b)


def test_noise_off_and_decoy_off_by_default():
    assert _CFG.obs_noise_scale == 0.0
    assert _CFG.decoy_omit_p0_k5 is False


def test_scm_hash_tracks_the_mechanism():
    base = scm_hash(_CFG)
    assert scm_hash(replace(_CFG, obs_noise_scale=0.0)) == base  # unchanged config -> same hash


def test_write_load_roundtrip_and_manifest(tmp_path: Path):
    manifest = write_dataset(_CFG, tmp_path)
    assert (tmp_path / "rows.npz").exists()
    assert (tmp_path / "manifest.json").exists()
    for key in ("schema_version", "n_rows", "config", "seeds", "dataset_hash", "scm_hash",
                "scm_identity", "git_sha"):
        assert key in manifest
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["n_rows"] == _CFG.n_rows_per_seed
    rows, loaded_manifest = load_dataset(tmp_path)
    assert loaded_manifest == json.loads((tmp_path / "manifest.json").read_text())
    assert dataset_hash(rows) == manifest["dataset_hash"]


def test_manifest_scm_identity_does_not_enumerate_truth():
    # The manifest is loaded by the discovery path; it must carry no adjacency edge list.
    manifest_identity = json.loads(json.dumps(_CFG_scm_identity()))
    assert "adjacency_edges" not in manifest_identity
    assert "mechanism_fingerprint" in manifest_identity


def _CFG_scm_identity():
    from cdd_oran.e2slice.dataset import scm_identity

    return scm_identity(_CFG)


def test_load_rejects_a_single_tampered_row_byte(tmp_path: Path):
    write_dataset(_CFG, tmp_path)
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


def test_load_rejects_scm_hash_mismatch(tmp_path: Path):
    write_dataset(_CFG, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["scm_hash"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    with pytest.raises(ValueError, match="scm_hash"):
        load_dataset(tmp_path)


def test_load_rejects_tampered_scm_identity_fingerprint(tmp_path: Path):
    write_dataset(_CFG, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["scm_identity"]["mechanism_fingerprint"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    with pytest.raises(ValueError, match="scm_identity"):
        load_dataset(tmp_path)


def test_load_rejects_inconsistent_seed(tmp_path: Path):
    write_dataset(_CFG, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["seeds"]["seed"] = int(manifest["config"]["seed"]) + 1
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    with pytest.raises(ValueError, match="seed"):
        load_dataset(tmp_path)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"n_rows_per_seed": 3}, "n_rows_per_seed"),
        ({"obs_noise_scale": 0.5}, "obs_noise_scale"),
        ({"decoy_omit_p0_k5": True}, "decoy_omit_p0_k5"),
    ],
)
def test_generate_rejects_invalid_config(kwargs, match):
    with pytest.raises(ValueError, match=match):
        generate_rows(replace(_CFG, **kwargs))


def test_generate_refuses_to_clobber_downstream_without_force(tmp_path: Path):
    write_dataset(_CFG, tmp_path)
    (tmp_path / "discovery.json").write_text("{}")
    with pytest.raises(ValueError, match="downstream artifacts already exist"):
        write_dataset(_CFG, tmp_path)
    write_dataset(_CFG, tmp_path, force=True)
    assert not (tmp_path / "discovery.json").exists()


def test_valid_manifest_loads_after_rederivation(tmp_path: Path):
    manifest = write_dataset(_CFG, tmp_path)
    rows, loaded = load_dataset(tmp_path)
    assert loaded == json.loads((tmp_path / "manifest.json").read_text())
    assert loaded["scm_hash"] == manifest["scm_hash"] == scm_hash(_CFG)
    assert dataset_hash(rows) == manifest["dataset_hash"]
