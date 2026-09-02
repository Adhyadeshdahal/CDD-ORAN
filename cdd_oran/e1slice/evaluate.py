"""Held-out one-step-prediction evaluation and artifact-reload verification.

- ``evaluate_dataset`` reloads each trained arm, predicts the HELD-OUT test episodes, and
  writes an immutable ``metrics.json`` (per-KPI + aggregate MSE/MAE, full provenance, a
  metrics_hash). Deterministic: re-running reproduces the file byte-for-byte.
- ``verify_dataset`` reloads each arm and asserts its test predictions reproduce the
  reference predictions captured from the live model at train time, within a frozen
  tolerance -- the acceptance condition that the persisted artifact round-trips.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from cdd_oran.e1slice import SCHEMA_VERSION
from cdd_oran.e1slice.dataset import E1Rows, _git_sha, load_dataset
from cdd_oran.e1slice.model import Arm, OneStepPredictor, load_arm
from cdd_oran.e1slice.split import load_split, row_indices_for

_ARMS: tuple[Arm, ...] = ("oracle", "dense")
_REF_FILE = "test_pred_ref.npz"
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
    err = pred - target
    return {
        "mse": float((err**2).mean()),
        "mae": float(np.abs(err).mean()),
        "per_kpi_mse": [float(v) for v in (err**2).mean(axis=0)],
        "per_kpi_mae": [float(v) for v in np.abs(err).mean(axis=0)],
        "n_test_rows": int(pred.shape[0]),
    }


def save_reference_predictions(
    dataset_dir: str | Path, arm: Arm, pred: npt.NDArray[np.float64]
) -> None:
    """Persist the live model's test predictions as the reload reference (train time)."""
    arm_dir = Path(dataset_dir) / "arms" / arm
    np.savez(arm_dir / _REF_FILE, pred=pred)


def _load_reference(dataset_dir: str | Path, arm: Arm) -> npt.NDArray[np.float64]:
    with np.load(Path(dataset_dir) / "arms" / arm / _REF_FILE) as data:
        return data["pred"].astype(np.float64)


def evaluate_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    """Reload both arms, score them on the held-out test episodes, write metrics.json."""
    out = Path(dataset_dir)
    rows, manifest = load_dataset(out)
    split = load_split(out)
    if split["dataset_hash"] != manifest["dataset_hash"]:
        raise ValueError("split.json is not bound to this dataset (dataset_hash mismatch).")

    test_episodes = split["test_episodes"]
    idx = row_indices_for(rows, test_episodes)
    y = rows.y_kpis[idx]

    arm_metrics: dict[str, Any] = {}
    for arm in _ARMS:
        model, arm_meta = load_arm(out, arm)
        if arm_meta["dataset_hash"] != manifest["dataset_hash"]:
            raise ValueError(f"arm '{arm}' was trained on a different dataset.")
        arm_metrics[arm] = regression_metrics(predict(model, rows, test_episodes), y)

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
    record["metrics_hash"] = hashlib.sha256(
        json.dumps(record["arms"], sort_keys=True).encode()
    ).hexdigest()
    (out / "metrics.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    return record


@dataclass(frozen=True)
class VerifyResult:
    arm: Arm
    max_abs_diff: float
    passed: bool


def verify_dataset(dataset_dir: str | Path, tol: float = VERIFY_TOL) -> list[VerifyResult]:
    """Assert each reloaded artifact reproduces its train-time reference within ``tol``."""
    out = Path(dataset_dir)
    rows, _ = load_dataset(out)
    split = load_split(out)
    results: list[VerifyResult] = []
    for arm in _ARMS:
        model, _ = load_arm(out, arm)
        reloaded_pred = predict(model, rows, split["test_episodes"])
        reference = _load_reference(out, arm)
        max_abs = float(np.abs(reloaded_pred - reference).max())
        results.append(VerifyResult(arm=arm, max_abs_diff=max_abs, passed=max_abs <= tol))
    return results
