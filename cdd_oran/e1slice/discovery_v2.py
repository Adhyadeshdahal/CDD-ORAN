"""Label-free per-target partial-correlation graph discovery for the E1 slice (frozen v2).

This module implements the contract frozen in ``docs/benchmark/E1_DISCOVERY_PROTOCOL_V2.md``
(committed as ``PROTOCOL_COMMIT`` below, BEFORE any E1 recovery result was computed). It is a
sibling of the v1 ``discovery.py`` (whose frozen behaviour is left untouched); v2 changes ONLY the
edge score and the threshold rule.

- Standardize every input/output column with training-row moments (population std); reject any
  zero-variance column; require the augmented input design ``[X | 1]`` to be full column rank.
- **Edge score (§3)** for ``(child KPI j, input i)`` = the magnitude of the *partial correlation*
  between input ``i`` and target ``j``, controlling for ALL other candidate inputs. Both the input
  and the target are residualized on the other inputs (+ intercept); the score is ``|corr(r_i,
  r_{j|i})|``. Per §3.4, resolved per edge in a fixed branch order: a vanishing TARGET residual is a
  *definition* ``rho := 0`` (input i adds no unique signal -> edge absent), NOT a stop; a vanishing
  FEATURE residual (genuine input collinearity) IS a recorded STOP; otherwise the §3.1 ratio.
- **Threshold (§4)**: per-target ``largest_gap(scores[j, :], floor=FROZEN_FLOOR)`` with
  ``FROZEN_FLOOR = 0.0`` (pinned for v2 so the ``0 -> signal`` separating gap is eligible), applied
  within each target's own row and never pooled. Keep an edge iff ``score >= threshold[j]``.

The module deliberately does NOT import ``E1V2Env`` or any true-adjacency symbol: dimensions come
from the persisted row shapes. Ground truth is read only elsewhere, and only after
``discovery_v2.json`` is persisted and hashed (see ``evaluate.score_recovery_v2``).
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

# SHA of the ``re-freeze protocol v2: pin FROZEN_FLOOR=0.0 for per-target cut`` commit. Recorded in
# every discovery_v2.json so the frozen contract provably predates any recovery result.
PROTOCOL_COMMIT = "c66b81d3244a0395266d4410640a0065a0816102"

# Downstream stages that must not survive a re-``discover-v2`` unless ``force`` is given.
_DISCOVER_V2_DESCENDANTS = ("recovery_v2.json",)

# The frozen v2 primary contract. These are NOT tunable: the persisted discovery_v2.json the
# discovered arm consumes must always use exactly these.
FROZEN_FLOOR = 0.0  # pinned to 0.0 for v2 (§4): makes the 0->signal separating gap eligible.
FROZEN_SCORE_METHOD = "partial_correlation"
FROZEN_THRESHOLD_METHOD = "per_target_largest_gap"

# Numerical near-singularity tolerance (§3.4). A rank/conditioning guard, not a score threshold:
# branch 1 only ever assigns score 0 (never selects); branch 2 only ever rejects (STOP).
EPS_COND = 1e-8

# The frozen E1 candidate-graph shape (num_kpis, num_params + num_kpis). This is the protocol's
# fixed layout, not environment truth, so it is a local constant (discovery imports no env symbol).
_NUM_KPIS = 4
_NUM_PARAMS = 4
_CANDIDATE_SHAPE = (_NUM_KPIS, _NUM_PARAMS + _NUM_KPIS)


@dataclass(frozen=True)
class DiscoveryV2Config:
    """Frozen v2 discovery contract: ``floor``/``score_method``/``threshold_method`` are fixed.

    Any value other than the frozen constants is rejected, so a rerun can never persist a
    differently-scored or differently-thresholded primary ``discovery_v2.json`` while still stamping
    the original ``protocol_commit`` (the anti-p-hacking guarantee).
    """

    floor: float = FROZEN_FLOOR
    score_method: str = FROZEN_SCORE_METHOD
    threshold_method: str = FROZEN_THRESHOLD_METHOD

    def __post_init__(self) -> None:
        if self.floor != FROZEN_FLOOR:
            raise ValueError(
                f"DiscoveryV2Config: floor is frozen at {FROZEN_FLOOR}, got {self.floor}. The v2 "
                "per-target threshold floor is not tunable."
            )
        if self.score_method != FROZEN_SCORE_METHOD:
            raise ValueError(
                f"DiscoveryV2Config: score_method is frozen at {FROZEN_SCORE_METHOD!r}, got "
                f"{self.score_method!r}"
            )
        if self.threshold_method != FROZEN_THRESHOLD_METHOD:
            raise ValueError(
                f"DiscoveryV2Config: threshold_method is frozen at {FROZEN_THRESHOLD_METHOD!r}, got "
                f"{self.threshold_method!r}"
            )


@dataclass(frozen=True)
class DiscoveryV2Result:
    partial_corr: npt.NDArray[np.float64]  # (K, P+K) signed partial correlations
    scores: npt.NDArray[np.float64]        # (K, P+K) |partial_corr|
    binary_mask: npt.NDArray[np.int64]     # (K, P+K) {0,1}
    threshold: npt.NDArray[np.float64]     # (K,) per-target threshold
    score_method: str
    threshold_method: str
    floor: float
    n_train_rows: int
    mean_x: npt.NDArray[np.float64]
    std_x: npt.NDArray[np.float64]
    mean_y: npt.NDArray[np.float64]
    std_y: npt.NDArray[np.float64]


def partial_correlation_scores(
    xs: npt.NDArray[np.float64], ys: npt.NDArray[np.float64], eps_cond: float = EPS_COND
) -> npt.NDArray[np.float64]:
    """Signed per-target partial correlations ``(K, P+K)`` (§3.1 + §3.4 branch order).

    ``xs`` are the standardized inputs ``(n, d)`` and ``ys`` the standardized targets ``(n, k)``.
    For each input ``i`` the control set is all OTHER inputs (+ intercept); both ``x_i`` and every
    target are residualized on it in one least-squares solve. Each edge ``(j, i)`` is then resolved
    in the frozen branch order: (1) target residual ~0 -> ``rho := 0`` (edge absent); (2) else input
    residual ~0 -> STOP (genuine collinearity); (3) else the §3.1 ratio.
    """
    n, d = xs.shape
    k = ys.shape[1]
    partial = np.zeros((k, d), dtype=np.float64)
    for i in range(d):
        others = np.concatenate([np.delete(xs, i, axis=1), np.ones((n, 1))], axis=1)  # (n, d)
        rhs = np.concatenate([xs[:, [i]], ys], axis=1)  # (n, 1+k): input i, then every target
        solution, *_ = np.linalg.lstsq(others, rhs, rcond=None)
        resid = rhs - others @ solution  # (n, 1+k)
        r_i = resid[:, 0]
        feat_resid_var = float(r_i @ r_i) / n  # ||r_i||^2 / n == 1 - R^2_i (input i on the others)
        for j in range(k):
            r_yj = resid[:, 1 + j]
            targ_resid_var = float(r_yj @ r_yj) / n
            if targ_resid_var < eps_cond:
                # Branch 1 (§3.4): the other inputs already explain target j -> input i carries no
                # unique signal. Definition, NOT a STOP: rho := 0 (edge absent). In noiseless E1
                # this fires on every non-parent edge and is the correct answer.
                partial[j, i] = 0.0
            elif feat_resid_var < eps_cond:
                # Branch 2 (§3.4): genuine input collinearity -> unstable denominator -> STOP.
                raise ValueError(
                    f"discovery_v2: input column {i} is near-collinear with the other inputs "
                    f"(||r_i||^2/n = {feat_resid_var:.3e} < {eps_cond:.0e}); the partial correlation "
                    "is non-identifiable. This is a recorded STOP, not an absent edge."
                )
            else:
                partial[j, i] = (r_i @ r_yj) / (np.sqrt(r_i @ r_i) * np.sqrt(r_yj @ r_yj))
    return partial


def discover_graph_v2(
    rows: E1Rows, train_episodes: list[int], cfg: DiscoveryV2Config | None = None
) -> DiscoveryV2Result:
    """Select the E1 candidate graph from TRAINING ROWS ONLY (no labels, no env truth)."""
    cfg = cfg if cfg is not None else DiscoveryV2Config()
    idx = row_indices_for(rows, train_episodes)
    if idx.size == 0:
        raise ValueError("discovery_v2: no training rows for the given train episodes")

    x = np.concatenate([rows.x_params[idx], rows.x_kpis[idx]], axis=1).astype(np.float64)
    y = rows.y_kpis[idx].astype(np.float64)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("discovery_v2: training rows contain non-finite values")

    n, d = x.shape
    k = y.shape[1]
    mean_x, std_x = x.mean(axis=0), x.std(axis=0)  # population std (ddof=0)
    mean_y, std_y = y.mean(axis=0), y.std(axis=0)
    if (std_x == 0.0).any():
        raise ValueError(f"discovery_v2: zero-variance input feature(s) at {np.where(std_x == 0)[0]}")
    if (std_y == 0.0).any():
        raise ValueError(f"discovery_v2: zero-variance (degenerate) output(s) at {np.where(std_y == 0)[0]}")

    xs = (x - mean_x) / std_x
    ys = (y - mean_y) / std_y
    design = np.concatenate([xs, np.ones((n, 1))], axis=1)  # (n, d+1) with intercept
    if np.linalg.matrix_rank(design) < d + 1:
        raise ValueError("discovery_v2: rank-deficient design; partial correlations are non-identifiable")

    partial = partial_correlation_scores(xs, ys, EPS_COND)
    scores = np.abs(partial)

    threshold = np.zeros(k, dtype=np.float64)
    binary_mask = np.zeros((k, d), dtype=np.int64)
    for j in range(k):
        row = scores[j]
        if not np.isfinite(row).all():
            raise ValueError(f"discovery_v2: non-finite edge scores in target {j}; no valid threshold")
        if float(row.max()) == float(row.min()):
            raise ValueError(
                f"discovery_v2: target {j} scores are all equal, so largest_gap has no valid split. "
                "This is a STOP diagnostic; do not substitute a truth-informed threshold."
            )
        thr = float(largest_gap(row, floor=cfg.floor))
        threshold[j] = thr
        binary_mask[j] = (row >= thr).astype(np.int64)

    return DiscoveryV2Result(
        partial_corr=partial,
        scores=scores,
        binary_mask=binary_mask,
        threshold=threshold,
        score_method=cfg.score_method,
        threshold_method=cfg.threshold_method,
        floor=float(cfg.floor),
        n_train_rows=int(idx.size),
        mean_x=mean_x,
        std_x=std_x,
        mean_y=mean_y,
        std_y=std_y,
    )


def build_discovery_v2_record(
    result: DiscoveryV2Result,
    dataset_hash: str,
    split_hash: str,
    train_episodes: list[int],
) -> dict[str, Any]:
    """Assemble the persisted, hash-bound discovery_v2 record (content_hash added last)."""
    sha, dirty = _git_sha()
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_hash": dataset_hash,
        "split_hash": split_hash,
        "protocol_commit": PROTOCOL_COMMIT,
        "score_method": result.score_method,
        "threshold_method": result.threshold_method,
        "floor": result.floor,
        "threshold": [float(t) for t in result.threshold],
        "tie_rule": "score >= threshold[target]",
        "train_episodes": [int(e) for e in train_episodes],
        "n_train_rows": int(result.n_train_rows),
        "moments": {
            "mean_x": [float(v) for v in result.mean_x],
            "std_x": [float(v) for v in result.std_x],
            "mean_y": [float(v) for v in result.mean_y],
            "std_y": [float(v) for v in result.std_y],
        },
        "partial_corr": result.partial_corr.tolist(),
        "scores": result.scores.tolist(),
        "binary_mask": result.binary_mask.astype(int).tolist(),
        "candidate_shape": list(result.binary_mask.shape),
        "git_sha": sha,
        "git_dirty": dirty,
    }
    record["content_hash"] = hashlib.sha256(canonical_json(record).encode()).hexdigest()
    return record


def write_discovery_v2(
    dataset_dir: str | Path, cfg: DiscoveryV2Config | None = None, force: bool = False
) -> dict[str, Any]:
    """Discover the v2 graph from the dataset's training rows and write discovery_v2.json.

    Binds to the current dataset + split, refuses to clobber a downstream recovery_v2 without
    ``force``, and publishes discovery_v2.json atomically. Written BEFORE any recovery score.
    """
    cfg = cfg if cfg is not None else DiscoveryV2Config()
    out = Path(dataset_dir)
    rows, manifest = load_dataset(out)
    split = load_split(
        out,
        expected_dataset_hash=manifest["dataset_hash"],
        episode_ids=set(int(e) for e in rows.episode.tolist()),
    )
    result = discover_graph_v2(rows, split["train_episodes"], cfg)
    record = build_discovery_v2_record(
        result, manifest["dataset_hash"], split["split_hash"], split["train_episodes"]
    )
    guard_descendants(out, _DISCOVER_V2_DESCENDANTS, force, "discover-v2")
    _atomic_write_text(out / "discovery_v2.json", json.dumps(record, indent=2, sort_keys=True))
    return record


def load_discovery_v2(
    dataset_dir: str | Path,
    *,
    expected_dataset_hash: str | None = None,
    expected_split_hash: str | None = None,
) -> dict[str, Any]:
    """Load discovery_v2.json and fail closed unless it is internally consistent and hash-bound."""
    disc_file = Path(dataset_dir) / "discovery_v2.json"
    record = json.loads(disc_file.read_text())

    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{disc_file}: schema_version {record.get('schema_version')!r} != {SCHEMA_VERSION!r}"
        )
    for key in (
        "dataset_hash", "split_hash", "protocol_commit", "score_method", "threshold_method",
        "floor", "threshold", "binary_mask", "partial_corr", "scores", "content_hash",
    ):
        if key not in record:
            raise ValueError(f"{disc_file}: missing required field '{key}'")

    stored = record["content_hash"]
    payload = {key: value for key, value in record.items() if key != "content_hash"}
    recomputed = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    if recomputed != stored:
        raise ValueError(f"{disc_file}: content_hash {stored} != recomputed {recomputed}")

    mask = np.asarray(record["binary_mask"], dtype=np.int64)
    partial = np.asarray(record["partial_corr"], dtype=np.float64)
    scores = np.asarray(record["scores"], dtype=np.float64)
    if mask.shape != _CANDIDATE_SHAPE:
        raise ValueError(
            f"{disc_file}: binary_mask shape {mask.shape} != frozen candidate shape {_CANDIDATE_SHAPE}"
        )
    if partial.shape != mask.shape or scores.shape != mask.shape:
        raise ValueError(f"{disc_file}: partial_corr/scores shapes disagree with binary_mask")
    if not set(np.unique(mask).tolist()).issubset({0, 1}):
        raise ValueError(f"{disc_file}: binary_mask has non-binary entries")
    if not (np.isfinite(partial).all() and np.isfinite(scores).all()):
        raise ValueError(f"{disc_file}: partial_corr/scores contain non-finite values")

    # Enforce the frozen protocol constants and re-derive the score/threshold/mask relations from
    # the persisted arrays. These re-derive values already in the file, so a correctly generated
    # discovery_v2.json is byte-unaffected; a rerun that quietly retuned is rejected.
    if record["protocol_commit"] != PROTOCOL_COMMIT:
        raise ValueError(
            f"{disc_file}: protocol_commit {record['protocol_commit']!r} != frozen {PROTOCOL_COMMIT!r}"
        )
    if record["score_method"] != FROZEN_SCORE_METHOD:
        raise ValueError(
            f"{disc_file}: score_method {record['score_method']!r} != frozen {FROZEN_SCORE_METHOD!r}"
        )
    if record["threshold_method"] != FROZEN_THRESHOLD_METHOD:
        raise ValueError(
            f"{disc_file}: threshold_method {record['threshold_method']!r} != frozen "
            f"{FROZEN_THRESHOLD_METHOD!r}"
        )
    if float(record["floor"]) != FROZEN_FLOOR:
        raise ValueError(f"{disc_file}: floor {record['floor']} != frozen {FROZEN_FLOOR}")
    if not np.array_equal(scores, np.abs(partial)):
        raise ValueError(f"{disc_file}: scores are not the absolute partial correlations")

    threshold = np.asarray(record["threshold"], dtype=np.float64)
    if threshold.shape != (mask.shape[0],):
        raise ValueError(
            f"{disc_file}: threshold shape {threshold.shape} != (num_kpis,)=({mask.shape[0]},)"
        )
    for j in range(mask.shape[0]):
        expected_threshold = float(largest_gap(scores[j], FROZEN_FLOOR))
        if float(threshold[j]) != expected_threshold:
            raise ValueError(
                f"{disc_file}: threshold[{j}] {float(threshold[j])} != "
                f"largest_gap(scores[{j}], {FROZEN_FLOOR}) {expected_threshold}"
            )
        if not np.array_equal(mask[j], (scores[j] >= float(threshold[j])).astype(np.int64)):
            raise ValueError(
                f"{disc_file}: binary_mask row {j} is inconsistent with scores >= threshold[{j}]"
            )

    if expected_dataset_hash is not None and record["dataset_hash"] != expected_dataset_hash:
        raise ValueError(
            f"{disc_file}: dataset_hash {record['dataset_hash']} != dataset {expected_dataset_hash}"
        )
    if expected_split_hash is not None and record["split_hash"] != expected_split_hash:
        raise ValueError(
            f"{disc_file}: split_hash {record['split_hash']} != split {expected_split_hash}"
        )
    return record


def discovered_v2_mask_array(record: dict[str, Any]) -> npt.NDArray[np.float32]:
    """The frozen (K, P+K) binary mask as a float32 array, for building the discovered arm."""
    return np.asarray(record["binary_mask"], dtype=np.float32)
