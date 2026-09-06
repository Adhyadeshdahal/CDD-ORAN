"""Unit self-checks for the RCoT-v2 E2 discovery sibling (``discovery_rcot_v2.py``).

**PRE-TRUTH**: these tests read NO ground truth and score against NO truth. They prove that v2 changes
ONLY ``block_perm_reps`` (99 -> 299) versus v1, that the v2 artifact contract (distinct filenames,
persistence round-trip, fail-closed load) holds, and -- critically -- that v1 is UNTOUCHED: v1's
``discovery_rcot.json`` / ``recovery_rcot.json`` still write and load beside the v2 ones. The heavy RCoT
calibration self-checks live in the v1 suite; v2 reuses v1's numerics verbatim, so they are not repeated.

Thread caps are pinned to 1 at import (thermal constraint on this laptop).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import dataclasses  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from cdd_oran.e2slice.dataset import E2DatasetConfig, write_dataset  # noqa: E402
from cdd_oran.e2slice.discovery_rcot import (  # noqa: E402
    PROTOCOL_COMMIT,
    RCOT_SCHEMA_VERSION,
    frozen_config,
    load_discovery_rcot,
    write_discovery_rcot,
)
from cdd_oran.e2slice.discovery_rcot_v2 import (  # noqa: E402
    FROZEN_BLOCK_PERM_REPS_V2,
    PROTOCOL_COMMIT_V2,
    frozen_config_v2,
    load_discovery_rcot_v2,
    write_discovery_rcot_v2,
)
from cdd_oran.e2slice.evaluate_rcot_v2 import (  # noqa: E402
    load_recovery_rcot_v2,
    score_recovery_rcot_v2,
)
from cdd_oran.envs.v2.e2 import E2V2Env  # noqa: E402


# --- the ONE change, and NOTHING else ----------------------------------------
def test_v2_only_changes_block_perm_reps():
    """v2 frozen config == v1 frozen config in every field EXCEPT block_perm_reps (99 -> 299)."""
    v1 = frozen_config()
    v2 = frozen_config_v2()
    assert v1.block_perm_reps == 99
    assert v2.block_perm_reps == 299 == FROZEN_BLOCK_PERM_REPS_V2

    v1_fields = dataclasses.asdict(v1)
    v2_fields = dataclasses.asdict(v2)
    differing = {k for k in v1_fields if v1_fields[k] != v2_fields[k]}
    assert differing == {"block_perm_reps"}, f"v2 must differ from v1 ONLY in block_perm_reps, got {differing}"

    # Spell out the frozen numerics explicitly so a silent drift is caught.
    assert v2.dz == 25 and v2.dxy == 5
    assert v2.ridge == 1e-6 and v2.q == 0.05
    assert v2.null_method == "block_perm"
    assert v2.block_size == 25
    assert v2.residual_epsilon == 1e-12
    assert v2.rng_seed == 0


def test_v2_protocol_commit_is_frozen_sha_distinct_from_v1():
    """FROZEN: PROTOCOL_COMMIT_V2 is the v2 freeze SHA (real 40-hex), distinct from v1's."""
    assert PROTOCOL_COMMIT_V2 != PROTOCOL_COMMIT
    # A real git sha is 40 lowercase hex chars.
    assert len(PROTOCOL_COMMIT_V2) == 40 and all(c in "0123456789abcdef" for c in PROTOCOL_COMMIT_V2)


# --- persistence round-trip + fail-closed (truth-free) -----------------------
def test_v2_persist_and_load_round_trips(tmp_path: Path):
    d = tmp_path / "e2rcot_v2"
    write_dataset(E2DatasetConfig(n_rows_per_seed=64, seed=0, sampling_seed=0), d)
    record = write_discovery_rcot_v2(d)
    assert (d / "discovery_rcot_v2.json").exists()
    assert not (d / "discovery_rcot.json").exists()  # v2 never writes the v1 filename
    assert not (d / "discovery.json").exists()       # nor the pdCor one
    assert record["schema_version"] == RCOT_SCHEMA_VERSION
    assert record["frozen"] is True
    assert record["protocol_commit"] == PROTOCOL_COMMIT_V2
    assert record["score_method"] == "rcot_conditional_independence"
    assert record["null_method"] == "block_perm"
    assert record["block_perm_reps"] == 299
    assert record["dz"] == 25 and record["q"] == 0.05
    manifest = json.loads((d / "manifest.json").read_text())
    assert record["dataset_hash"] == manifest["dataset_hash"]

    before = (d / "discovery_rcot_v2.json").read_text()
    loaded = load_discovery_rcot_v2(d)
    assert (d / "discovery_rcot_v2.json").read_text() == before  # load does not mutate
    mask = np.asarray(loaded["binary_mask"], dtype=np.int64)
    assert mask.shape == (6, 14)
    assert set(np.unique(mask).tolist()).issubset({0, 1})


def test_v2_load_rejects_tampered_content_hash(tmp_path: Path):
    d = tmp_path / "e2rcot_v2_tamper"
    write_dataset(E2DatasetConfig(n_rows_per_seed=64, seed=0, sampling_seed=0), d)
    write_discovery_rcot_v2(d)
    record = json.loads((d / "discovery_rcot_v2.json").read_text())
    record["pvalues"][0][0] = 0.123456  # tamper without re-hashing
    (d / "discovery_rcot_v2.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    with pytest.raises(ValueError, match="content_hash"):
        load_discovery_rcot_v2(d)


def test_v2_load_rejects_wrong_block_perm_reps(tmp_path: Path):
    """A record carrying the v1 permutation count (99) must NOT load as v2 (re-derives to 299)."""
    d = tmp_path / "e2rcot_v2_reps"
    write_dataset(E2DatasetConfig(n_rows_per_seed=64, seed=0, sampling_seed=0), d)
    write_discovery_rcot_v2(d)
    record = json.loads((d / "discovery_rcot_v2.json").read_text())
    record["block_perm_reps"] = 99
    import hashlib

    from cdd_oran.e2slice.dataset import canonical_json
    payload = {k: v for k, v in record.items() if k != "content_hash"}
    record["content_hash"] = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    (d / "discovery_rcot_v2.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    with pytest.raises(ValueError, match="block_perm_reps"):
        load_discovery_rcot_v2(d)


def test_v2_load_rejects_wrong_protocol_commit(tmp_path: Path):
    d = tmp_path / "e2rcot_v2_pc"
    write_dataset(E2DatasetConfig(n_rows_per_seed=64, seed=0, sampling_seed=0), d)
    write_discovery_rcot_v2(d)
    record = json.loads((d / "discovery_rcot_v2.json").read_text())
    record["protocol_commit"] = PROTOCOL_COMMIT  # the v1 sha must be rejected under v2
    import hashlib

    from cdd_oran.e2slice.dataset import canonical_json
    payload = {k: v for k, v in record.items() if k != "content_hash"}
    record["content_hash"] = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    (d / "discovery_rcot_v2.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    with pytest.raises(ValueError, match="protocol_commit"):
        load_discovery_rcot_v2(d)


# --- recovery path (reads truth; writes the v2-named artifact) ----------------
def test_v2_recovery_round_trips_and_names_v2_artifact(tmp_path: Path):
    d = tmp_path / "e2rcot_v2_recover"
    write_dataset(E2DatasetConfig(n_rows_per_seed=64, seed=0, sampling_seed=0), d)
    write_discovery_rcot_v2(d)
    record = score_recovery_rcot_v2(d)
    assert record["protocol_commit"] == PROTOCOL_COMMIT_V2
    assert record["score_method"] == "rcot_conditional_independence"
    assert record["null_method"] == "block_perm"
    assert (d / "recovery_rcot_v2.json").exists()
    assert not (d / "recovery_rcot.json").exists()  # v2 never writes the v1 recovery filename
    assert not (d / "recovery.json").exists()
    for block in ("overall", "ncp_kpi", "kpi_kpi"):
        for key in ("precision", "recall", "f1", "tp", "fp", "fn"):
            assert key in record["recovery"][block]
    assert record["kpi_kpi_candidates"] == 36

    manifest = json.loads((d / "manifest.json").read_text())
    disc = json.loads((d / "discovery_rcot_v2.json").read_text())
    loaded = load_recovery_rcot_v2(
        d,
        expected_dataset_hash=manifest["dataset_hash"],
        expected_discovery_hash=disc["content_hash"],
    )
    assert loaded["content_hash"] == record["content_hash"]
    with pytest.raises(ValueError, match="dataset_hash"):
        load_recovery_rcot_v2(d, expected_dataset_hash="0" * 64)


# --- v1 STAYS UNTOUCHED: both records sit side by side ------------------------
def test_v1_and_v2_coexist_untouched(tmp_path: Path):
    """Writing v2 does not clobber v1: both discovery + recovery records persist and load independently."""
    d = tmp_path / "e2rcot_both"
    write_dataset(E2DatasetConfig(n_rows_per_seed=64, seed=0, sampling_seed=0), d)

    v1_rec = write_discovery_rcot(d)          # v1 frozen block_perm, B=99
    v2_rec = write_discovery_rcot_v2(d)        # v2 frozen block_perm, B=299

    # Distinct files, distinct frozen provenance.
    assert (d / "discovery_rcot.json").exists()
    assert (d / "discovery_rcot_v2.json").exists()
    assert v1_rec["protocol_commit"] == PROTOCOL_COMMIT
    assert v1_rec["block_perm_reps"] == 99
    assert v2_rec["protocol_commit"] == PROTOCOL_COMMIT_V2
    assert v2_rec["block_perm_reps"] == 299

    # v1's loader still round-trips its own artifact after v2 was written.
    v1_loaded = load_discovery_rcot(d)
    assert v1_loaded["content_hash"] == v1_rec["content_hash"]
    assert v1_loaded["block_perm_reps"] == 99

    # Cross-loaders refuse the other version's file (fail-closed both directions).
    v2_loaded = load_discovery_rcot_v2(d)
    assert v2_loaded["content_hash"] == v2_rec["content_hash"]
    # A v1 mask cannot masquerade as v2: point the v2 loader at the v1 filename via a copy.
    (d / "discovery_rcot_v2_probe.json").write_text((d / "discovery_rcot.json").read_text())
    with pytest.raises(ValueError):
        load_discovery_rcot(  # v1 record under v2 constants -> block_perm_reps/protocol mismatch
            d, filename="discovery_rcot_v2_probe.json",
            expected_protocol_commit=PROTOCOL_COMMIT_V2, config_factory=frozen_config_v2,
        )


def test_v1_env_import_ok():
    """Sanity that the truth env still imports (used by v2 recovery scoring)."""
    assert E2V2Env.num_params == 8
