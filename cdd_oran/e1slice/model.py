"""Matched oracle/dense/discovered one-step predictor and its training, for the E1 slice.

The arms share ONE architecture -- a small per-output MLP head for each KPI -- and differ only
by a fixed, non-trainable input mask:

    oracle:     head j sees only x[parents(j)]  (the true-graph parents of KPI j)
    dense:      head j sees all inputs
    discovered: head j sees only x[discovered-parents(j)]  (the FROZEN learned graph)

The mask is applied by zeroing non-parent inputs before the head, so it is not a learnable
parameter and all arms carry an IDENTICAL parameter budget. The oracle's advantage is the true
structural prior; the discovered arm gets the label-free learned prior from ``discovery.json``
(never env truth). Capacity is matched by construction and recorded as explicit metadata. E1's
mechanism is linear, so a correctly-masked arm is expected to fit -- this is the recovery
control, a pipeline test, not a superiority claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from cdd_oran.e1slice import SCHEMA_VERSION
from cdd_oran.e1slice.dataset import E1Rows, _atomic_write_text, _git_sha, guard_descendants
from cdd_oran.e1slice.split import row_indices_for
from cdd_oran.envs.v2.e1 import E1V2Env

Arm = Literal["oracle", "dense", "discovered"]

# The trained arm's per-KPI test predictions captured at train time; the reload reference.
REF_FILE = "test_pred_ref.npz"

# Downstream stages that must not survive a re-``train`` unless ``force`` is given.
_TRAIN_DESCENDANTS = ("metrics.json",)


@dataclass(frozen=True)
class ModelConfig:
    hidden: tuple[int, ...] = (16,)
    lr: float = 1e-2
    epochs: int = 300
    batch_size: int = 64
    weight_seed: int = 0

    def validate(self) -> None:
        """Reject configurations that cannot produce a trainable arm."""
        if not self.hidden or any(int(w) <= 0 for w in self.hidden):
            raise ValueError(f"ModelConfig: hidden widths must be non-empty and positive, got {self.hidden}")
        if not (math.isfinite(self.lr) and self.lr > 0.0):
            raise ValueError(f"ModelConfig: lr must be finite and > 0, got {self.lr}")
        if self.epochs <= 0:
            raise ValueError(f"ModelConfig: epochs must be > 0, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"ModelConfig: batch_size must be > 0, got {self.batch_size}")


def arm_mask(arm: Arm) -> torch.Tensor:
    """(num_kpis, in_dim) 0/1 input mask for an arm, from the E1 true adjacency.

    Only ``oracle`` (true graph) and ``dense`` (all inputs) are derived here. The ``discovered``
    mask MUST come from the frozen ``discovery.json`` via ``validate_explicit_mask`` -- it is
    never derived from environment truth.

    in_dim = [params | kpis]; source indexing in true_adj_matrix already matches this layout
    (source s < num_params is param s; s >= num_params is KPI s-num_params).
    """
    p, k = E1V2Env.num_params, E1V2Env.num_kpis
    if arm == "oracle":
        adj = E1V2Env(env_seed=0).true_adj_matrix()
        mask = adj[p : p + k, : p + k]
    elif arm == "dense":
        mask = np.ones((k, p + k), dtype=np.float32)
    else:
        raise ValueError(f"arm_mask does not derive a '{arm}' mask; supply it explicitly")
    return torch.as_tensor(np.asarray(mask, dtype=np.float32))


def validate_explicit_mask(mask: npt.NDArray[Any] | torch.Tensor) -> torch.Tensor:
    """Validate and coerce an explicitly-supplied (num_kpis, in_dim) binary mask to a tensor."""
    arr = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    p, k = E1V2Env.num_params, E1V2Env.num_kpis
    if arr.shape != (k, p + k):
        raise ValueError(f"explicit mask shape {arr.shape} != expected {(k, p + k)}")
    if not np.isin(arr, (0, 1)).all():
        raise ValueError("explicit mask must be binary (0/1)")
    return torch.as_tensor(arr.astype(np.float32))


class OneStepPredictor(nn.Module):
    """Per-output masked MLP: y_j = head_j(x * mask_j)."""

    mask: torch.Tensor

    def __init__(self, mask: torch.Tensor, hidden: tuple[int, ...]) -> None:
        super().__init__()
        num_outputs, in_dim = mask.shape
        self.register_buffer("mask", mask)
        self.heads = nn.ModuleList(
            [self._make_head(in_dim, hidden) for _ in range(num_outputs)]
        )

    @staticmethod
    def _make_head(in_dim: int, hidden: tuple[int, ...]) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev = in_dim
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.ReLU()]
            prev = width
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cols = [head(x * self.mask[j]) for j, head in enumerate(self.heads)]
        return torch.cat(cols, dim=1)

    def num_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))


def _xy(rows: E1Rows, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    x = np.concatenate([rows.x_params[indices], rows.x_kpis[indices]], axis=1)
    y = rows.y_kpis[indices]
    return (
        torch.as_tensor(x, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
    )


def build_model(
    arm: Arm, cfg: ModelConfig, mask: npt.NDArray[Any] | torch.Tensor | None = None
) -> OneStepPredictor:
    """Build an arm. ``mask`` must be supplied for ``discovered`` (the frozen learned graph)."""
    cfg.validate()
    torch.manual_seed(cfg.weight_seed)
    resolved = validate_explicit_mask(mask) if mask is not None else arm_mask(arm)
    return OneStepPredictor(resolved, cfg.hidden)


def train_arm(
    rows: E1Rows,
    train_episodes: list[int],
    arm: Arm,
    cfg: ModelConfig,
    mask: npt.NDArray[Any] | torch.Tensor | None = None,
) -> tuple[OneStepPredictor, dict[str, Any]]:
    """Train one arm on the TRAIN episodes only. Returns the model and capacity/training meta.

    For ``discovered``, pass the frozen binary mask from ``discovery.json``; oracle/dense derive
    their masks internally. All arms share the identical architecture and parameter budget.
    """
    model = build_model(arm, cfg, mask=mask)
    idx = row_indices_for(rows, train_episodes)
    x, y = _xy(rows, idx)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(cfg.weight_seed)
    model.train()
    final_loss = float("nan")
    for _ in range(cfg.epochs):
        perm = torch.randperm(x.shape[0], generator=generator)
        for start in range(0, x.shape[0], cfg.batch_size):
            batch = perm[start : start + cfg.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(x[batch]), y[batch])
            loss.backward()
            opt.step()
        final_loss = float(loss_fn(model(x), y).item())

    meta = {
        "arm": arm,
        "config": asdict(cfg),
        "capacity": {
            "architecture": "per-output masked MLP",
            "hidden": list(cfg.hidden),
            "num_parameters": model.num_parameters(),
            "num_active_inputs_per_output": [int(v) for v in model.mask.sum(dim=1).tolist()],
        },
        "n_train_rows": int(idx.shape[0]),
        "final_train_mse": final_loss,
    }
    return model, meta


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_torch_save(state: dict[str, Any], path: Path) -> str:
    """Save a state dict to a same-dir temp file, publish atomically, return its byte digest."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    return _sha256_file(path)


def _atomic_savez_ref(pred: npt.NDArray[np.float64], path: Path) -> str:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, pred=pred)
    os.replace(tmp, path)
    return _sha256_file(path)


def save_arm(
    dataset_dir: str | Path,
    model: OneStepPredictor,
    meta: dict[str, Any],
    dataset_hash: str,
    split_hash: str,
    reference_pred: npt.NDArray[np.float64],
    force: bool = False,
    discovery_hash: str | None = None,
) -> Path:
    """Persist an arm's weights, reload reference, and metadata under ``arms/<arm>/``.

    The arm stage owns three files -- ``model.pt``, the reload reference ``test_pred_ref.npz``,
    and ``arm_meta.json`` -- and records a SHA-256 digest of the first two so any later reload
    can prove the bytes it reads are the bytes that were trained/captured. Each file is written
    atomically. Refuses to run if downstream ``metrics.json`` exists unless ``force`` clears it.
    For the ``discovered`` arm, ``discovery_hash`` (the frozen graph's content hash) is required
    and recorded so eval/verify can bind the arm to the exact frozen graph.
    """
    out = Path(dataset_dir)
    guard_descendants(out, _TRAIN_DESCENDANTS, force, "train")
    if str(meta["arm"]) == "discovered" and not discovery_hash:
        raise ValueError("save_arm: the 'discovered' arm requires a discovery_hash")

    reference_pred = np.ascontiguousarray(reference_pred, dtype=np.float64)
    if reference_pred.ndim != 2 or reference_pred.shape[1] != E1V2Env.num_kpis:
        raise ValueError(
            f"reference predictions must be (n_test_rows, {E1V2Env.num_kpis}), "
            f"got {reference_pred.shape}"
        )
    if not np.isfinite(reference_pred).all():
        raise ValueError("reference predictions contain non-finite values")

    arm_dir = out / "arms" / str(meta["arm"])
    arm_dir.mkdir(parents=True, exist_ok=True)
    model_digest = _atomic_torch_save(model.state_dict(), arm_dir / "model.pt")
    ref_digest = _atomic_savez_ref(reference_pred, arm_dir / REF_FILE)
    sha, dirty = _git_sha()
    record = {
        "schema_version": SCHEMA_VERSION,
        "dataset_hash": dataset_hash,
        "split_hash": split_hash,
        "model_sha256": model_digest,
        "ref_sha256": ref_digest,
        "git_sha": sha,
        "git_dirty": dirty,
        **meta,
    }
    if discovery_hash is not None:
        record["discovery_hash"] = discovery_hash
    _atomic_write_text(arm_dir / "arm_meta.json", json.dumps(record, indent=2, sort_keys=True))
    return arm_dir


def load_arm(dataset_dir: str | Path, arm: Arm) -> tuple[OneStepPredictor, dict[str, Any]]:
    """Rebuild an arm's architecture, verify its file digests, and load its trained weights.

    Fails closed if ``arm_meta.json`` is missing a required field, names a different arm, or if
    the bytes of ``model.pt`` / ``test_pred_ref.npz`` do not match the digests recorded at
    train time (tampered or truncated weights/reference).
    """
    arm_dir = Path(dataset_dir) / "arms" / arm
    meta_file = arm_dir / "arm_meta.json"
    meta = json.loads(meta_file.read_text())

    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{meta_file}: schema_version {meta.get('schema_version')!r} != {SCHEMA_VERSION!r}"
        )
    required = ["arm", "config", "dataset_hash", "split_hash", "capacity", "model_sha256", "ref_sha256"]
    if arm == "discovered":
        required.append("discovery_hash")
    for key in required:
        if key not in meta:
            raise ValueError(f"{meta_file}: missing required field '{key}'")
    if meta["arm"] != arm:
        raise ValueError(f"{meta_file}: arm {meta['arm']!r} != requested {arm!r}")

    model_file = arm_dir / "model.pt"
    ref_file = arm_dir / REF_FILE
    if _sha256_file(model_file) != meta["model_sha256"]:
        raise ValueError(f"{model_file}: bytes do not match arm_meta model_sha256")
    if _sha256_file(ref_file) != meta["ref_sha256"]:
        raise ValueError(f"{ref_file}: bytes do not match arm_meta ref_sha256")

    if arm == "discovered":
        # Rebuild the discovered mask from the FROZEN graph on disk and bind by content hash,
        # so a changed discovery.json (or a mismatched graph) fails the reload closed.
        from cdd_oran.e1slice.discovery import discovered_mask_array, load_discovery

        disc = load_discovery(arm_dir.parent.parent)
        if disc["content_hash"] != meta["discovery_hash"]:
            raise ValueError(
                f"{meta_file}: discovery_hash {meta['discovery_hash']} != discovery.json "
                f"content_hash {disc['content_hash']} (arm bound to a different frozen graph)"
            )
        resolved_mask = validate_explicit_mask(discovered_mask_array(disc))
    else:
        resolved_mask = arm_mask(arm)

    raw = dict(meta["config"])
    raw["hidden"] = tuple(raw["hidden"])
    cfg = ModelConfig(**raw)
    model = OneStepPredictor(resolved_mask, cfg.hidden)
    # register_buffer aliases resolved_mask, and load_state_dict copies the checkpoint buffer INTO
    # it in place -- so snapshot the intended mask BEFORE loading, or the check would compare the
    # buffer to itself.
    expected_mask = resolved_mask.detach().clone()
    state = torch.load(model_file, weights_only=True)
    model.load_state_dict(state)
    # ``mask`` is a registered buffer, so load_state_dict OVERWRITES it with the checkpoint's copy.
    # Require the loaded buffer to equal the mask we resolved (true graph for oracle/dense; the
    # FROZEN discovery graph for discovered) so a checkpoint carrying a different valid mask is
    # rejected and the arm is truly bound to its intended graph.
    if not torch.equal(model.mask, expected_mask):
        raise ValueError(
            f"{model_file}: checkpoint mask does not match the resolved '{arm}' arm mask"
        )
    model.eval()
    return model, meta
