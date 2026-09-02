"""Label-free linear graph discovery for the E1 slice (frozen protocol).

This module implements the contract frozen in ``docs/benchmark/E1_DISCOVERY_PROTOCOL.md``
(committed as ``PROTOCOL_COMMIT`` below, BEFORE any E1 recovery result was computed). It selects
a directed candidate graph ``s_t -> k_{t+1}`` from TRAINING ROWS ONLY, with no labels and no
environment truth:

- Standardize every input/output column with training-row moments (population std); reject any
  zero-variance column.
- Fit one OLS linear transition per output KPI with an intercept via ``numpy.linalg.lstsq``;
  require the augmented design to be full column rank.
- Edge score = absolute standardized coefficient. Threshold = ``largest_gap(scores.ravel(),
  floor=1e-3)``; keep an edge iff ``score >= threshold`` (tie rule).

The module deliberately does NOT import ``E1V2Env`` or any true-adjacency symbol: dimensions come
from the persisted row shapes. Ground truth is read only elsewhere, and only after
``discovery.json`` is persisted and hashed (see ``evaluate.score_recovery``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from cdd_oran.analysis.auto_threshold import largest_gap
from cdd_oran.e1slice import SCHEMA_VERSION
from cdd_oran.e1slice.dataset import (
    E1Rows,
    _atomic_write_text,
    _git_sha,
    canonical_json,
    guard_descendants,
    load_dataset,
)
from cdd_oran.e1slice.split import load_split, row_indices_for

# SHA of the ``docs: freeze E1 discovery protocol`` commit. Recorded in every discovery.json so
# the frozen contract provably predates any recovery result (anti-p-hacking).
PROTOCOL_COMMIT = "e5312a696a0ed5f9d590e9ad766bdbfdad53a16c"

# Downstream stages that must not survive a re-``discover`` unless ``force`` is given.
_DISCOVER_DESCENDANTS = ("arms", "metrics.json", "recovery.json")

# The frozen primary contract (docs/benchmark/E1_DISCOVERY_PROTOCOL.md). These are NOT tunable:
# the persisted discovery.json the discovered arm consumes must always use exactly these.
FROZEN_FLOOR = 1e-3
FROZEN_METHOD = "largest_gap"

# The frozen E1 candidate-graph shape (num_kpis, num_params + num_kpis). This is the protocol's
# fixed layout, not environment truth, so it is a local constant (discovery imports no env symbol).
_NUM_KPIS = 4
_NUM_PARAMS = 4
_CANDIDATE_SHAPE = (_NUM_KPIS, _NUM_PARAMS + _NUM_KPIS)


@dataclass(frozen=True)
class DiscoveryConfig:
    """Frozen discovery contract: ``floor`` and ``method`` are fixed and non-overridable.

    Any value other than the frozen ``FROZEN_FLOOR`` / ``FROZEN_METHOD`` is rejected, so a rerun
    can never persist a differently-thresholded primary ``discovery.json`` while still stamping
    the original ``protocol_commit`` (the anti-p-hacking guarantee).
    """

    floor: float = FROZEN_FLOOR
    method: str = FROZEN_METHOD

    def __post_init__(self) -> None:
        if self.floor != FROZEN_FLOOR:
            raise ValueError(
                f"DiscoveryConfig: floor is frozen at {FROZEN_FLOOR}, got {self.floor}. The "
                "primary discovery threshold is not tunable; a floor sweep may only be run as a "
                "labelled diagnostic that never writes the primary discovery.json."
            )
        if self.method != FROZEN_METHOD:
            raise ValueError(
                f"DiscoveryConfig: method is frozen at {FROZEN_METHOD!r}, got {self.method!r}"
            )


@dataclass(frozen=True)
class DiscoveryResult:
    coefficients: npt.NDArray[np.float64]  # (K, P+K) signed standardized coefficients
    scores: npt.NDArray[np.float64]        # (K, P+K) absolute standardized coefficients
    binary_mask: npt.NDArray[np.int64]     # (K, P+K) {0,1}
    threshold: float
    method: str
    floor: float
    n_train_rows: int
    mean_x: npt.NDArray[np.float64]
    std_x: npt.NDArray[np.float64]
    mean_y: npt.NDArray[np.float64]
    std_y: npt.NDArray[np.float64]


def discover_graph(
    rows: E1Rows, train_episodes: list[int], cfg: DiscoveryConfig | None = None
) -> DiscoveryResult:
    """Select the E1 candidate graph from TRAINING ROWS ONLY (no labels, no env truth)."""
    cfg = cfg if cfg is not None else DiscoveryConfig()
    idx = row_indices_for(rows, train_episodes)
    if idx.size == 0:
        raise ValueError("discovery: no training rows for the given train episodes")

    x = np.concatenate([rows.x_params[idx], rows.x_kpis[idx]], axis=1).astype(np.float64)
    y = rows.y_kpis[idx].astype(np.float64)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("discovery: training rows contain non-finite values")

    n, d = x.shape
    k = y.shape[1]
    mean_x, std_x = x.mean(axis=0), x.std(axis=0)  # population std (ddof=0)
    mean_y, std_y = y.mean(axis=0), y.std(axis=0)
    if (std_x == 0.0).any():
        raise ValueError(f"discovery: zero-variance input feature(s) at {np.where(std_x == 0)[0]}")
    if (std_y == 0.0).any():
        raise ValueError(f"discovery: zero-variance (degenerate) output(s) at {np.where(std_y == 0)[0]}")

    xs = (x - mean_x) / std_x
    ys = (y - mean_y) / std_y
    design = np.concatenate([xs, np.ones((n, 1))], axis=1)  # (n, d+1) with intercept
    if np.linalg.matrix_rank(design) < d + 1:
        raise ValueError("discovery: rank-deficient design; coefficients are non-identifiable")

    coefficients = np.zeros((k, d), dtype=np.float64)
    for j in range(k):
        solution, *_ = np.linalg.lstsq(design, ys[:, j], rcond=None)
        coefficients[j] = solution[:d]  # drop the intercept term; it is not an edge
    scores = np.abs(coefficients)

    flat = scores.ravel()
    if not np.isfinite(flat).all():
        raise ValueError("discovery: non-finite edge scores; no valid threshold")
    if float(flat.max()) == float(flat.min()):
        raise ValueError(
            "discovery: all edge scores are equal, so largest_gap has no valid split. "
            "This is a STOP diagnostic; do not substitute a truth-informed threshold."
        )
    threshold = float(largest_gap(flat, floor=cfg.floor))
    binary_mask = (scores >= threshold).astype(np.int64)

    return DiscoveryResult(
        coefficients=coefficients,
        scores=scores,
        binary_mask=binary_mask,
        threshold=threshold,
        method=cfg.method,
        floor=float(cfg.floor),
        n_train_rows=int(idx.size),
        mean_x=mean_x,
        std_x=std_x,
        mean_y=mean_y,
        std_y=std_y,
    )


def build_discovery_record(
    result: DiscoveryResult,
    dataset_hash: str,
    split_hash: str,
    train_episodes: list[int],
) -> dict[str, Any]:
    """Assemble the persisted, hash-bound discovery record (content_hash added last)."""
    sha, dirty = _git_sha()
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_hash": dataset_hash,
        "split_hash": split_hash,
        "protocol_commit": PROTOCOL_COMMIT,
        "method": result.method,
        "floor": result.floor,
        "threshold": result.threshold,
        "tie_rule": "score >= threshold",
        "train_episodes": [int(e) for e in train_episodes],
        "n_train_rows": int(result.n_train_rows),
        "moments": {
            "mean_x": [float(v) for v in result.mean_x],
            "std_x": [float(v) for v in result.std_x],
            "mean_y": [float(v) for v in result.mean_y],
            "std_y": [float(v) for v in result.std_y],
        },
        "coefficients": result.coefficients.tolist(),
        "scores": result.scores.tolist(),
        "binary_mask": result.binary_mask.astype(int).tolist(),
        "candidate_shape": list(result.binary_mask.shape),
        "git_sha": sha,
        "git_dirty": dirty,
    }
    record["content_hash"] = hashlib.sha256(canonical_json(record).encode()).hexdigest()
    return record


def write_discovery(
    dataset_dir: str | Path, cfg: DiscoveryConfig | None = None, force: bool = False
) -> dict[str, Any]:
    """Discover the graph from the persisted dataset's training rows and write discovery.json.

    Binds to the current dataset + split, refuses to clobber downstream arms/metrics/recovery
    without ``force``, and publishes discovery.json atomically. This file is written BEFORE any
    recovery score is computed.
    """
    cfg = cfg if cfg is not None else DiscoveryConfig()
    out = Path(dataset_dir)
    rows, manifest = load_dataset(out)
    split = load_split(
        out,
        expected_dataset_hash=manifest["dataset_hash"],
        episode_ids=set(int(e) for e in rows.episode.tolist()),
    )
    result = discover_graph(rows, split["train_episodes"], cfg)
    record = build_discovery_record(result, manifest["dataset_hash"], split["split_hash"], split["train_episodes"])
    guard_descendants(out, _DISCOVER_DESCENDANTS, force, "discover")
    _atomic_write_text(out / "discovery.json", json.dumps(record, indent=2, sort_keys=True))
    return record


def load_discovery(
    dataset_dir: str | Path,
    *,
    expected_dataset_hash: str | None = None,
    expected_split_hash: str | None = None,
) -> dict[str, Any]:
    """Load discovery.json and fail closed unless it is internally consistent and hash-bound."""
    disc_file = Path(dataset_dir) / "discovery.json"
    record = json.loads(disc_file.read_text())

    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{disc_file}: schema_version {record.get('schema_version')!r} != {SCHEMA_VERSION!r}"
        )
    for key in (
        "dataset_hash", "split_hash", "protocol_commit", "method", "threshold",
        "binary_mask", "coefficients", "scores", "content_hash",
    ):
        if key not in record:
            raise ValueError(f"{disc_file}: missing required field '{key}'")

    stored = record["content_hash"]
    payload = {key: value for key, value in record.items() if key != "content_hash"}
    recomputed = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    if recomputed != stored:
        raise ValueError(f"{disc_file}: content_hash {stored} != recomputed {recomputed}")

    mask = np.asarray(record["binary_mask"], dtype=np.int64)
    coefs = np.asarray(record["coefficients"], dtype=np.float64)
    scores = np.asarray(record["scores"], dtype=np.float64)
    if mask.shape != _CANDIDATE_SHAPE:
        raise ValueError(
            f"{disc_file}: binary_mask shape {mask.shape} != frozen candidate shape {_CANDIDATE_SHAPE}"
        )
    if coefs.shape != mask.shape or scores.shape != mask.shape:
        raise ValueError(f"{disc_file}: coefficients/scores shapes disagree with binary_mask")
    if not set(np.unique(mask).tolist()).issubset({0, 1}):
        raise ValueError(f"{disc_file}: binary_mask has non-binary entries")
    if not (np.isfinite(coefs).all() and np.isfinite(scores).all()):
        raise ValueError(f"{disc_file}: coefficients/scores contain non-finite values")
    threshold = float(record["threshold"])
    if not np.array_equal(mask, (scores >= threshold).astype(np.int64)):
        raise ValueError(f"{disc_file}: binary_mask is inconsistent with scores >= threshold")

    if expected_dataset_hash is not None and record["dataset_hash"] != expected_dataset_hash:
        raise ValueError(
            f"{disc_file}: dataset_hash {record['dataset_hash']} != dataset {expected_dataset_hash}"
        )
    if expected_split_hash is not None and record["split_hash"] != expected_split_hash:
        raise ValueError(
            f"{disc_file}: split_hash {record['split_hash']} != split {expected_split_hash}"
        )
    return record


def discovered_mask_array(record: dict[str, Any]) -> npt.NDArray[np.float32]:
    """The frozen (K, P+K) binary mask as a float32 array, for building the discovered arm."""
    return np.asarray(record["binary_mask"], dtype=np.float32)
