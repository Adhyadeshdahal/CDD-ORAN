"""Tests for the E2 RCoT recovery scoring layer (§9). This reads truth; the RCoT DISCOVERY path
reads none. Sibling to ``test_e2slice_evaluate.py`` for the frozen RCoT method.

Thread caps are pinned to 1 at import (thermal constraint on this laptop)."""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from cdd_oran.e2slice.dataset import E2DatasetConfig, write_dataset  # noqa: E402
from cdd_oran.e2slice.discovery_rcot import PROTOCOL_COMMIT, write_discovery_rcot  # noqa: E402
from cdd_oran.e2slice.evaluate_rcot import (  # noqa: E402
    load_recovery_rcot,
    score_recovery_rcot,
)
from cdd_oran.envs.v2.e2 import E2V2Env  # noqa: E402


@pytest.fixture(scope="module")
def _recovered(tmp_path_factory):
    """Discover an RCoT mask on a tiny truth-free dataset, then score recovery against E2 truth."""
    d = tmp_path_factory.mktemp("e2rcot_recover")
    write_dataset(E2DatasetConfig(n_rows_per_seed=64, seed=0, sampling_seed=0), d)
    write_discovery_rcot(d)  # frozen block_perm; tiny n keeps it fast + single-core
    record = score_recovery_rcot(d)
    return d, record


def test_score_recovery_rcot_reports_all_blocks_and_kpi_kpi(_recovered):
    _d, record = _recovered
    assert record["protocol_commit"] == PROTOCOL_COMMIT
    assert record["score_method"] == "rcot_conditional_independence"
    assert record["null_method"] == "block_perm"
    for block in ("overall", "ncp_kpi", "kpi_kpi"):
        for key in ("precision", "recall", "f1", "tp", "fp", "fn"):
            assert key in record["recovery"][block]
    assert record["kpi_kpi_candidates"] == 36
    assert 0 <= record["kpi_kpi_fp"] <= 36
    expected_rate = 1.0 - record["kpi_kpi_fp"] / 36
    assert abs(record["kpi_kpi_rejection_rate"] - expected_rate) < 1e-12
    assert len(record["per_target"]) == 6
    # Writes the RCoT-named artifact, never the pdCor one.
    assert (_d / "recovery_rcot.json").exists()
    assert not (_d / "recovery.json").exists()


def test_kpi_kpi_fp_matches_mask(_recovered):
    d, record = _recovered
    disc = json.loads((d / "discovery_rcot.json").read_text())
    mask = np.asarray(disc["binary_mask"], dtype=int)
    assert record["kpi_kpi_fp"] == int(mask[:, E2V2Env.num_params:].sum())


def test_full_graph_embedding_places_kpi_rows_only(_recovered):
    """The (6,14)->(14,14) embedding: the recovered graph's TP/FP/FN must live only in KPI rows."""
    d, record = _recovered
    disc = json.loads((d / "discovery_rcot.json").read_text())
    mask = np.asarray(disc["binary_mask"], dtype=int)
    p = E2V2Env.num_params
    # Every predicted edge (mask sum) is embedded in a KPI child row -> overall predicted edge count
    # equals the mask sum, and equals ncp_kpi.edges + kpi_kpi_fp.
    assert record["recovery"]["overall"]["edges"] == int(mask.sum())
    assert record["recovery"]["ncp_kpi"]["edges"] + record["kpi_kpi_fp"] == int(mask.sum())
    # True graph has 0 KPI->KPI edges, so kpi_kpi tp/fn are 0 and every KPI->KPI prediction is an FP.
    assert record["recovery"]["kpi_kpi"]["tp"] == 0
    assert record["recovery"]["kpi_kpi"]["fn"] == 0
    assert record["recovery"]["kpi_kpi"]["fp"] == record["kpi_kpi_fp"]
    assert record["recovery"]["overall"]["fp"] == int(mask[:, p:].sum()) + record["recovery"]["ncp_kpi"]["fp"]


def test_recovery_rcot_loads_fail_closed(_recovered):
    d, record = _recovered
    manifest = json.loads((d / "manifest.json").read_text())
    disc = json.loads((d / "discovery_rcot.json").read_text())
    loaded = load_recovery_rcot(
        d,
        expected_dataset_hash=manifest["dataset_hash"],
        expected_discovery_hash=disc["content_hash"],
    )
    assert loaded["content_hash"] == record["content_hash"]
    with pytest.raises(ValueError, match="dataset_hash"):
        load_recovery_rcot(d, expected_dataset_hash="0" * 64)
    with pytest.raises(ValueError, match="discovery_hash"):
        load_recovery_rcot(d, expected_discovery_hash="0" * 64)


def test_recovery_rcot_rejects_tampered_content_hash(_recovered, tmp_path: Path):
    d, _record = _recovered
    record = json.loads((d / "recovery_rcot.json").read_text())
    record["kpi_kpi_fp"] = record["kpi_kpi_fp"] + 1  # no rehash -> content_hash mismatch
    (tmp_path / "recovery_rcot.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    with pytest.raises(ValueError, match="content_hash"):
        load_recovery_rcot(tmp_path)


def test_recovery_rcot_rejects_wrong_protocol_commit(_recovered, tmp_path: Path):
    d, _record = _recovered
    record = json.loads((d / "recovery_rcot.json").read_text())
    record["protocol_commit"] = "0" * 40
    import hashlib

    from cdd_oran.e2slice.dataset import canonical_json
    payload = {k: v for k, v in record.items() if k != "content_hash"}
    record["content_hash"] = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    (tmp_path / "recovery_rcot.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    with pytest.raises(ValueError, match="protocol_commit"):
        load_recovery_rcot(tmp_path)
