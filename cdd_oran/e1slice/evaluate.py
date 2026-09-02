"""Held-out one-step-prediction evaluation and artifact-reload verification.

- ``evaluate_dataset`` reloads each trained arm, predicts the HELD-OUT test episodes, and
  writes an immutable ``metrics.json`` (per-KPI + aggregate MSE/MAE, full provenance, a
  metrics_hash). Deterministic: on the same code state (metrics.json embeds live git
  provenance) re-running reproduces the file byte-for-byte.
- ``verify_dataset`` reloads each arm and asserts its test predictions reproduce the
  reference predictions captured from the live model at train time, within a frozen
  tolerance -- the acceptance condition that the persisted artifact round-trips.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from cdd_oran.e1slice import SCHEMA_VERSION
from cdd_oran.e1slice.dataset import (
    E1Rows,
    _atomic_write_text,
    _git_sha,
    canonical_json,
    load_dataset,
)
from cdd_oran.e1slice.model import REF_FILE, Arm, OneStepPredictor, load_arm
from cdd_oran.e1slice.split import load_split, row_indices_for
from cdd_oran.envs.v2.e1 import E1V2Env

_ARMS: tuple[Arm, ...] = ("oracle", "dense")
VERIFY_TOL = 1e-6


def predict(model: OneStepPredictor, rows: E1Rows, episodes: list[int]) -> npt.NDArray[np.float64]:
    """Predict next-step KPIs for the rows of ``episodes`` (model must be in eval mode)."""
    idx = row_indices_for(rows, episodes)
    x = np.concatenate([rows.x_params[idx], rows.x_kpis[idx]], axis=1)
    with torch.no_grad():
        out = model(torch.as_tensor(x, dtype=torch.float32))
    return out.numpy().astype(np.float64)


def regression_metrics(
    pred: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.ndim != 2 or target.ndim != 2:
        raise ValueError(f"regression_metrics expects 2-D arrays, got {pred.ndim}-D and {target.ndim}-D")
    if pred.shape != target.shape:
        raise ValueError(f"regression_metrics shape mismatch: pred {pred.shape} vs target {target.shape}")
    if pred.size == 0:
        raise ValueError("regression_metrics received an empty array")
    if not (np.isfinite(pred).all() and np.isfinite(target).all()):
        raise ValueError("regression_metrics received non-finite values")
    err = pred - target
    return {
        "mse": float((err**2).mean()),
        "mae": float(np.abs(err).mean()),
        "per_kpi_mse": [float(v) for v in (err**2).mean(axis=0)],
        "per_kpi_mae": [float(v) for v in np.abs(err).mean(axis=0)],
        "n_test_rows": int(pred.shape[0]),
    }


def _load_reference(
    dataset_dir: str | Path, arm: Arm, expected_shape: tuple[int, int]
) -> npt.NDArray[np.float64]:
    """Load a validated reload reference (its byte digest is already checked in ``load_arm``)."""
    ref_file = Path(dataset_dir) / "arms" / arm / REF_FILE
    with np.load(ref_file) as data:
        pred = data["pred"].astype(np.float64)
    if pred.shape != expected_shape:
        raise ValueError(f"{ref_file}: reference shape {pred.shape} != expected {expected_shape}")
    if not np.isfinite(pred).all():
        raise ValueError(f"{ref_file}: reference contains non-finite values")
    return pred


def _bind_split(dataset_dir: Path, rows: E1Rows, manifest: dict[str, Any]) -> dict[str, Any]:
    """Load the split bound to this dataset's hash and exact episode coverage."""
    return load_split(
        dataset_dir,
        expected_dataset_hash=manifest["dataset_hash"],
        episode_ids=set(int(e) for e in rows.episode.tolist()),
    )


def _bind_arm(
    dataset_dir: Path, arm: Arm, manifest: dict[str, Any], split: dict[str, Any]
) -> tuple[OneStepPredictor, dict[str, Any]]:
    """Reload an arm and bind its metadata to the current dataset + split hashes."""
    model, arm_meta = load_arm(dataset_dir, arm)
    if arm_meta["dataset_hash"] != manifest["dataset_hash"]:
        raise ValueError(f"arm '{arm}' was trained on a different dataset (dataset_hash mismatch).")
    if arm_meta["split_hash"] != split["split_hash"]:
        raise ValueError(f"arm '{arm}' was trained on a different split (split_hash mismatch).")
    return model, arm_meta


def evaluate_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    """Reload both arms, score them on the held-out test episodes, write metrics.json.

    Validates the complete provenance chain (rows->manifest, split->dataset, arm->dataset+split,
    model/reference bytes->digests) and confirms reload fidelity against each captured reference
    BEFORE atomically publishing metrics.json.
    """
    out = Path(dataset_dir)
    rows, manifest = load_dataset(out)
    split = _bind_split(out, rows, manifest)

    test_episodes = split["test_episodes"]
    idx = row_indices_for(rows, test_episodes)
    y = rows.y_kpis[idx]
    expected_shape = (int(idx.shape[0]), E1V2Env.num_kpis)

    arm_metrics: dict[str, Any] = {}
    for arm in _ARMS:
        model, arm_meta = _bind_arm(out, arm, manifest, split)
        pred = predict(model, rows, test_episodes)
        reference = _load_reference(out, arm, expected_shape)
        max_abs = float(np.abs(pred - reference).max())
        if max_abs > VERIFY_TOL:
            raise ValueError(
                f"arm '{arm}' reload fidelity failed: max_abs_diff {max_abs:.3e} > tol {VERIFY_TOL:.0e}"
            )
        metrics = regression_metrics(pred, y)
        metrics["model_sha256"] = arm_meta["model_sha256"]
        metrics["ref_sha256"] = arm_meta["ref_sha256"]
        arm_metrics[arm] = metrics

    sha, dirty = _git_sha()
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_hash": manifest["dataset_hash"],
        "split_hash": split["split_hash"],
        "test_episodes": test_episodes,
        "arms": arm_metrics,
        "git_sha": sha,
        "git_dirty": dirty,
    }
    record["metrics_hash"] = hashlib.sha256(canonical_json(record).encode()).hexdigest()
    _atomic_write_text(out / "metrics.json", json.dumps(record, indent=2, sort_keys=True))
    return record


@dataclass(frozen=True)
class VerifyResult:
    arm: Arm
    max_abs_diff: float
    passed: bool


def verify_dataset(dataset_dir: str | Path, tol: float = VERIFY_TOL) -> list[VerifyResult]:
    """Assert each reloaded artifact reproduces its train-time reference within ``tol``.

    Validates the same provenance chain as evaluation (dataset, split binding, arm binding, and
    model/reference byte digests), so a tampered artifact is rejected with ``ValueError`` rather
    than silently reported as a soft mismatch.
    """
    if not (math.isfinite(tol) and tol >= 0.0):
        raise ValueError(f"verify tolerance must be finite and >= 0, got {tol}")
    out = Path(dataset_dir)
    rows, manifest = load_dataset(out)
    split = _bind_split(out, rows, manifest)
    test_episodes = split["test_episodes"]
    expected_shape = (int(row_indices_for(rows, test_episodes).shape[0]), E1V2Env.num_kpis)

    results: list[VerifyResult] = []
    for arm in _ARMS:
        model, _ = _bind_arm(out, arm, manifest, split)
        reloaded_pred = predict(model, rows, test_episodes)
        reference = _load_reference(out, arm, expected_shape)
        max_abs = float(np.abs(reloaded_pred - reference).max())
        results.append(VerifyResult(arm=arm, max_abs_diff=max_abs, passed=max_abs <= tol))
    return results
