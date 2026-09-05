"""Tests for the E2 recovery scoring layer (§9). This is the recovery path, which reads truth;
the DISCOVERY path (tested elsewhere) reads none."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cdd_oran.e2slice.dataset import E2DatasetConfig, write_dataset
from cdd_oran.e2slice.discovery import PROTOCOL_COMMIT, write_discovery
from cdd_oran.e2slice.evaluate import (
    full_graph_from_discovered_mask,
    load_recovery,
    score_recovery,
)
from cdd_oran.envs.v2.e2 import E2V2Env


def test_full_graph_mapping_places_kpi_rows_only():
    mask = np.zeros((6, 14), dtype=int)
    mask[0, 0] = 1   # K0 <- P0
    mask[5, 13] = 1  # K5 <- K5_lagged (a KPI->KPI candidate)
    full = full_graph_from_discovered_mask(mask)
    assert full.shape == (14, 14)
    # Param-child rows (0..7) are empty; KPI child rows (8..13) carry the mask verbatim.
    assert full[:8, :].sum() == 0
    assert full[8, 0] == 1        # K0 (row 8) <- P0 (col 0)
    assert full[13, 13] == 1      # K5 (row 13) <- K5_lagged (col 13)
    np.testing.assert_array_equal(full[8:14, :], mask)


def test_full_graph_rejects_wrong_shape():
    with pytest.raises(ValueError, match="mask shape"):
        full_graph_from_discovered_mask(np.zeros((6, 8), dtype=int))


def test_true_adj_has_16_ncp_kpi_and_0_kpi_kpi_edges():
    # §5 expected counts (decoy OFF), sanity-checked against the env so the recovery denominators
    # are what the protocol declares.
    gt = E2V2Env(env_seed=0).true_adj_matrix().astype(int)
    p = E2V2Env.num_params
    assert int(gt.sum()) == 16
    assert int(gt[p:, :p].sum()) == 16    # all true edges are NCP->KPI
    assert int(gt[p:, p:].sum()) == 0     # no KPI->KPI edges


@pytest.fixture(scope="module")
def _recovered(tmp_path_factory):
    d = tmp_path_factory.mktemp("e2recover")
    write_dataset(E2DatasetConfig(n_rows_per_seed=48, seed=0, sampling_seed=0), d)
    write_discovery(d)
    record = score_recovery(d)
    return d, record


def test_score_recovery_reports_all_blocks_and_kpi_kpi(_recovered):
    _d, record = _recovered
    assert record["protocol_commit"] == PROTOCOL_COMMIT
    for block in ("overall", "ncp_kpi", "kpi_kpi"):
        for key in ("precision", "recall", "f1", "tp", "fp", "fn"):
            assert key in record["recovery"][block]
    assert record["kpi_kpi_candidates"] == 36
    assert 0 <= record["kpi_kpi_fp"] <= 36
    expected_rate = 1.0 - record["kpi_kpi_fp"] / 36
    assert abs(record["kpi_kpi_rejection_rate"] - expected_rate) < 1e-12
    assert len(record["per_target"]) == 6


def test_kpi_kpi_fp_matches_mask(_recovered):
    d, record = _recovered
    disc = json.loads((d / "discovery.json").read_text())
    mask = np.asarray(disc["binary_mask"], dtype=int)
    assert record["kpi_kpi_fp"] == int(mask[:, E2V2Env.num_params:].sum())


def test_recovery_loads_fail_closed(_recovered):
    d, record = _recovered
    manifest = json.loads((d / "manifest.json").read_text())
    disc = json.loads((d / "discovery.json").read_text())
    loaded = load_recovery(
        d,
        expected_dataset_hash=manifest["dataset_hash"],
        expected_discovery_hash=disc["content_hash"],
    )
    assert loaded["content_hash"] == record["content_hash"]
    with pytest.raises(ValueError, match="dataset_hash"):
        load_recovery(d, expected_dataset_hash="0" * 64)
    with pytest.raises(ValueError, match="discovery_hash"):
        load_recovery(d, expected_discovery_hash="0" * 64)


def test_recovery_rejects_tampered_content_hash(_recovered, tmp_path: Path):
    d, _record = _recovered
    record = json.loads((d / "recovery.json").read_text())
    record["kpi_kpi_fp"] = record["kpi_kpi_fp"] + 1  # no rehash -> content_hash mismatch
    (tmp_path / "recovery.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    with pytest.raises(ValueError, match="content_hash"):
        load_recovery(tmp_path)
