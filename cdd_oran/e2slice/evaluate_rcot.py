"""POST-FREEZE E2 RCoT recovery scoring (§9). Sibling to ``evaluate.py`` for the frozen RCoT method.

This is the ONLY RCoT module that reads ground truth. It loads the already-persisted, hash-bound
``discovery_rcot.json`` FIRST (via ``load_discovery_rcot``, which re-derives every frozen §11 constant
and the ``protocol_commit`` fail-closed), maps its ``(6, 14)`` mask into the full ``(14, 14)`` graph,
then compares it to ``E2V2Env().true_adj_matrix()`` -- read here for the first time -- into a SEPARATE
``recovery_rcot.json`` record (NEVER ``recovery.json``) that can never feed back into selection.

It mirrors ``evaluate.py::score_recovery`` EXACTLY, changing only what the RCoT sibling requires:
the mask loader (``load_discovery_rcot`` instead of ``load_discovery``), the frozen ``protocol_commit``
(the RCoT ``PROTOCOL_COMMIT``), and the recorded method fields (``null_method`` instead of pdCor's
``threshold_method``). Every metric helper -- ``full_graph_from_discovered_mask``,
``_per_target_metrics``, ``recovery_by_edge_type`` / ``_prf`` -- is REUSED from the pdCor evaluator, so
there is no new discretion and no free parameter in the scoring.

Firewall: ``E2V2Env`` is imported here for recovery scoring only (§2, permitted use (b)); the RCoT
discovery module imports no env truth.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from cdd_oran.analysis.recovery_metrics import recovery_by_edge_type
from cdd_oran.e2slice import SCHEMA_VERSION
from cdd_oran.e2slice.dataset import _atomic_write_text, _git_sha, canonical_json, load_dataset
from cdd_oran.e2slice.discovery_rcot import PROTOCOL_COMMIT, load_discovery_rcot

# Reuse the shared, method-agnostic recovery machinery from the pdCor evaluator verbatim -- no new
# metric code, no free parameters. ``recovery_by_edge_type`` / ``_prf`` are reached through these.
from cdd_oran.e2slice.evaluate import (
    _N_KPI_KPI_CANDIDATES,
    _NODE_NAMES,
    _per_target_metrics,
    full_graph_from_discovered_mask,
)
from cdd_oran.envs.v2.e2 import E2V2Env


def score_recovery_rcot(
    dataset_dir: str | Path,
    *,
    discovery_loader: Callable[..., dict[str, Any]] = load_discovery_rcot,
    out_filename: str = "recovery_rcot.json",
) -> dict[str, Any]:
    """POST-FREEZE RCoT recovery scoring: score the persisted RCoT graph against E2 truth (§9).

    Loads the hash-bound ``discovery_rcot.json`` FIRST, then reads ``E2V2Env().true_adj_matrix()`` and
    writes a separate ``recovery_rcot.json`` so recovery numbers can never alter discovery.

    The keyword-only ``discovery_loader`` / ``out_filename`` default to v1; the RCoT-v2 sibling passes
    ``load_discovery_rcot_v2`` + ``recovery_rcot_v2.json`` so the v2 record sits beside the v1 one. The
    recovery record's ``protocol_commit`` flows through from the loaded discovery record, so it is
    automatically ``PROTOCOL_COMMIT_V2`` for a v2 mask.
    """
    out = Path(dataset_dir)
    _rows, manifest = load_dataset(out)
    disc = discovery_loader(out)
    if disc["dataset_hash"] != manifest["dataset_hash"]:
        raise ValueError(
            f"{out}: discovery_rcot dataset_hash {disc['dataset_hash']} != dataset "
            f"{manifest['dataset_hash']}"
        )

    mask = np.asarray(disc["binary_mask"], dtype=int)              # (6, 14)
    pred_full = full_graph_from_discovered_mask(mask)              # (14, 14)
    gt_full = E2V2Env(env_seed=0).true_adj_matrix().astype(int)   # (14, 14), decoy OFF true SCM
    num_params = E2V2Env.num_params

    recovery = recovery_by_edge_type(pred_full, gt_full, num_params, node_names=_NODE_NAMES)

    # KPI->KPI false positives (of 36) and rejection rate (§9, §10). The lagged KPI candidates are
    # the pred mask's columns num_params.. ; all are true-negatives, so every prediction is an FP.
    kpi_kpi_fp = int(mask[:, num_params:].sum())
    rejection_rate = 1.0 - kpi_kpi_fp / _N_KPI_KPI_CANDIDATES

    gt_mask = gt_full[num_params:, :]  # (6, 14): the KPI child rows of the true graph
    per_target = _per_target_metrics(mask, gt_mask, num_params)

    sha, dirty = _git_sha()
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_hash": manifest["dataset_hash"],
        "discovery_hash": disc["content_hash"],
        "protocol_commit": disc["protocol_commit"],
        "score_method": disc["score_method"],
        "null_method": disc["null_method"],
        "gt_edge_count": int(gt_full.sum()),
        "recovery": recovery,
        "kpi_kpi_fp": kpi_kpi_fp,
        "kpi_kpi_candidates": _N_KPI_KPI_CANDIDATES,
        "kpi_kpi_rejection_rate": float(rejection_rate),
        "per_target": per_target,
        "git_sha": sha,
        "git_dirty": dirty,
    }
    record["content_hash"] = hashlib.sha256(canonical_json(record).encode()).hexdigest()
    _atomic_write_text(out / out_filename, json.dumps(record, indent=2, sort_keys=True))
    return record


def load_recovery_rcot(
    dataset_dir: str | Path,
    *,
    expected_dataset_hash: str | None = None,
    expected_discovery_hash: str | None = None,
    filename: str = "recovery_rcot.json",
    expected_protocol_commit: str = PROTOCOL_COMMIT,
) -> dict[str, Any]:
    """Load ``recovery_rcot.json`` and fail closed on any parent-hash / shape / numeric / commit mismatch.

    The keyword-only ``filename`` / ``expected_protocol_commit`` default to v1; the RCoT-v2 sibling
    passes ``recovery_rcot_v2.json`` + ``PROTOCOL_COMMIT_V2``.
    """
    rec_file = Path(dataset_dir) / filename
    record = json.loads(rec_file.read_text())

    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{rec_file}: schema_version {record.get('schema_version')!r} != {SCHEMA_VERSION!r}"
        )
    for key in ("dataset_hash", "discovery_hash", "protocol_commit", "null_method", "recovery",
                "kpi_kpi_fp", "kpi_kpi_rejection_rate", "content_hash"):
        if key not in record:
            raise ValueError(f"{rec_file}: missing required field '{key}'")

    stored = record["content_hash"]
    payload = {key: value for key, value in record.items() if key != "content_hash"}
    recomputed = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    if recomputed != stored:
        raise ValueError(f"{rec_file}: content_hash {stored} != recomputed {recomputed}")

    if record["protocol_commit"] != expected_protocol_commit:
        raise ValueError(
            f"{rec_file}: protocol_commit {record['protocol_commit']!r} != frozen "
            f"{expected_protocol_commit!r}"
        )
    fp = int(record["kpi_kpi_fp"])
    expected_rate = 1.0 - fp / _N_KPI_KPI_CANDIDATES
    if abs(float(record["kpi_kpi_rejection_rate"]) - expected_rate) > 1e-12:
        raise ValueError(
            f"{rec_file}: kpi_kpi_rejection_rate {record['kpi_kpi_rejection_rate']} != "
            f"1 - {fp}/{_N_KPI_KPI_CANDIDATES} = {expected_rate}"
        )
    if expected_dataset_hash is not None and record["dataset_hash"] != expected_dataset_hash:
        raise ValueError(
            f"{rec_file}: dataset_hash {record['dataset_hash']} != dataset {expected_dataset_hash}"
        )
    if expected_discovery_hash is not None and record["discovery_hash"] != expected_discovery_hash:
        raise ValueError(
            f"{rec_file}: discovery_hash {record['discovery_hash']} != discovery {expected_discovery_hash}"
        )
    return record
