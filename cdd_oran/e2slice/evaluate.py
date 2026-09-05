"""POST-FREEZE E2 recovery scoring (§9). Implemented but NOT executed in the impl phase.

This is the ONLY module in the E2 slice that reads ground truth. It loads the already-persisted,
hash-bound ``discovery.json`` FIRST, maps its ``(6, 14)`` mask into the full ``(14, 14)`` graph
(child KPI rows ``8..13`` carry the predicted parents; the 8 param-child rows are empty), then
compares it to ``E2V2Env().true_adj_matrix()`` -- read here for the first time -- into a SEPARATE
``recovery.json`` record that can never feed back into selection.

Reports precision / recall / F1 overall and NCP->KPI, the KPI->KPI false-positive count (of 36) and
rejection rate ``1 - FP/36``, and per-target metrics. The recovery record loads fail-closed on any
parent-hash / shape / numeric / ``protocol_commit`` mismatch.

Firewall: ``E2V2Env`` is imported here for recovery scoring only (§2, permitted use (b)); the
discovery module imports no env truth. This scoring is not run in the implementation phase.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from cdd_oran.analysis.recovery_metrics import recovery_by_edge_type
from cdd_oran.analysis.threshold_sweep import _prf
from cdd_oran.e2slice import SCHEMA_VERSION
from cdd_oran.e2slice.dataset import _atomic_write_text, _git_sha, canonical_json, load_dataset
from cdd_oran.e2slice.discovery import PROTOCOL_COMMIT, load_discovery
from cdd_oran.envs.v2.e2 import E2V2Env

_NODE_NAMES = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "K0", "K1", "K2", "K3", "K4", "K5"]
_N_KPI_KPI_CANDIDATES = 36  # 6 targets x 6 lagged KPI candidates (§5), all true-negatives


def full_graph_from_discovered_mask(mask: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Embed a ``(6, 14)`` discovered parent mask into the full ``(14, 14)`` child/parent graph (§9).

    Only the KPI child rows ``P..P+K-1`` (= ``8..13``) carry predicted edges; the earlier param-child
    rows are empty. This is the layout ``recovery_by_edge_type`` expects. The mask's 14 candidate
    columns ``[P0..P7, K0..K5(lagged)]`` map directly onto the 14 full-graph parent columns.
    """
    mask = np.asarray(mask, dtype=int)
    p, k = E2V2Env.num_params, E2V2Env.num_kpis
    if mask.shape != (k, p + k):
        raise ValueError(
            f"discovered mask shape {mask.shape} != (num_kpis, num_params+num_kpis)={(k, p + k)}"
        )
    full = np.zeros((p + k, p + k), dtype=int)
    full[p : p + k, :] = mask
    return full


def _per_target_metrics(
    pred_mask: npt.NDArray[np.int64], gt_mask: npt.NDArray[np.int64], num_params: int
) -> list[dict[str, Any]]:
    """Per-target (per child KPI) PRF over the 14 candidate columns, split by candidate type."""
    out: list[dict[str, Any]] = []
    for j in range(pred_mask.shape[0]):
        pj = pred_mask[j : j + 1, :]
        gj = gt_mask[j : j + 1, :]
        out.append({
            "target": f"K{j}",
            "overall": _prf(pj, gj),
            "ncp_kpi": _prf(pj[:, :num_params], gj[:, :num_params]),
            "kpi_kpi": _prf(pj[:, num_params:], gj[:, num_params:]),
        })
    return out


def score_recovery(dataset_dir: str | Path) -> dict[str, Any]:
    """POST-FREEZE recovery scoring: score the persisted discovered graph against E2 truth (§9).

    Loads the hash-bound ``discovery.json`` FIRST, then reads ``E2V2Env().true_adj_matrix()`` and
    writes a separate ``recovery.json`` so recovery numbers can never alter discovery. NOT run in
    the implementation phase.
    """
    out = Path(dataset_dir)
    _rows, manifest = load_dataset(out)
    disc = load_discovery(out, expected_dataset_hash=manifest["dataset_hash"])

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
        "threshold_method": disc["threshold_method"],
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
    _atomic_write_text(out / "recovery.json", json.dumps(record, indent=2, sort_keys=True))
    return record


def load_recovery(
    dataset_dir: str | Path,
    *,
    expected_dataset_hash: str | None = None,
    expected_discovery_hash: str | None = None,
) -> dict[str, Any]:
    """Load ``recovery.json`` and fail closed on any parent-hash / shape / numeric / commit mismatch."""
    rec_file = Path(dataset_dir) / "recovery.json"
    record = json.loads(rec_file.read_text())

    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{rec_file}: schema_version {record.get('schema_version')!r} != {SCHEMA_VERSION!r}"
        )
    for key in ("dataset_hash", "discovery_hash", "protocol_commit", "recovery", "kpi_kpi_fp",
                "kpi_kpi_rejection_rate", "content_hash"):
        if key not in record:
            raise ValueError(f"{rec_file}: missing required field '{key}'")

    stored = record["content_hash"]
    payload = {key: value for key, value in record.items() if key != "content_hash"}
    recomputed = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    if recomputed != stored:
        raise ValueError(f"{rec_file}: content_hash {stored} != recomputed {recomputed}")

    if record["protocol_commit"] != PROTOCOL_COMMIT:
        raise ValueError(
            f"{rec_file}: protocol_commit {record['protocol_commit']!r} != frozen {PROTOCOL_COMMIT!r}"
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
