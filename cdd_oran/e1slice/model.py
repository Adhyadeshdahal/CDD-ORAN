"""Matched oracle/dense one-step predictor and its training, for the E1 slice.

The two arms share ONE architecture -- a small per-output MLP head for each KPI -- and
differ only by a fixed, non-trainable input mask:

    oracle: head j sees only x[parents(j)]  (the true-graph parents of KPI j)
    dense:  head j sees all inputs

The mask is applied by zeroing non-parent inputs before the head, so it is not a learnable
parameter and both arms carry an IDENTICAL parameter budget. The oracle's only advantage is
the structural prior (correct sparsity); capacity is matched by construction and recorded as
explicit metadata. E1's mechanism is linear, so both arms are expected to fit -- this is the
recovery control, a pipeline test, not a superiority claim.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from cdd_oran.e1slice import SCHEMA_VERSION
from cdd_oran.e1slice.dataset import E1Rows, _git_sha
from cdd_oran.e1slice.split import row_indices_for
from cdd_oran.envs.v2.e1 import E1V2Env

Arm = Literal["oracle", "dense"]


@dataclass(frozen=True)
class ModelConfig:
    hidden: tuple[int, ...] = (16,)
    lr: float = 1e-2
    epochs: int = 300
    batch_size: int = 64
    weight_seed: int = 0


def arm_mask(arm: Arm) -> torch.Tensor:
    """(num_kpis, in_dim) 0/1 input mask for an arm, from the E1 true adjacency.

    in_dim = [params | kpis]; source indexing in true_adj_matrix already matches this layout
    (source s < num_params is param s; s >= num_params is KPI s-num_params).
    """
    p, k = E1V2Env.num_params, E1V2Env.num_kpis
    if arm == "oracle":
        adj = E1V2Env(env_seed=0).true_adj_matrix()  # (p+k, p+k) float32
        mask = adj[p : p + k, : p + k]
    else:
        mask = np.ones((k, p + k), dtype=np.float32)
    return torch.as_tensor(np.asarray(mask, dtype=np.float32))


class OneStepPredictor(nn.Module):
    """Per-output masked MLP: y_j = head_j(x * mask_j)."""

    mask: torch.Tensor  # registered buffer; annotated so the checker sees a Tensor

    def __init__(self, mask: torch.Tensor, hidden: tuple[int, ...]) -> None:
        super().__init__()
        num_outputs, in_dim = mask.shape
        self.register_buffer("mask", mask)  # (num_outputs, in_dim); saved, not trained
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


def build_model(arm: Arm, cfg: ModelConfig) -> OneStepPredictor:
    torch.manual_seed(cfg.weight_seed)
    return OneStepPredictor(arm_mask(arm), cfg.hidden)


def train_arm(
    rows: E1Rows, train_episodes: list[int], arm: Arm, cfg: ModelConfig
) -> tuple[OneStepPredictor, dict[str, Any]]:
    """Train one arm on the TRAIN episodes only. Returns the model and capacity/training meta."""
    model = build_model(arm, cfg)
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


def save_arm(
    dataset_dir: str | Path,
    model: OneStepPredictor,
    meta: dict[str, Any],
    dataset_hash: str,
    split_hash: str,
) -> Path:
    """Persist an arm's weights + metadata under ``<dataset_dir>/arms/<arm>/``."""
    arm_dir = Path(dataset_dir) / "arms" / str(meta["arm"])
    arm_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), arm_dir / "model.pt")
    sha, dirty = _git_sha()
    record = {
        "schema_version": SCHEMA_VERSION,
        "dataset_hash": dataset_hash,
        "split_hash": split_hash,
        "git_sha": sha,
        "git_dirty": dirty,
        **meta,
    }
    (arm_dir / "arm_meta.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    return arm_dir


def load_arm(dataset_dir: str | Path, arm: Arm) -> tuple[OneStepPredictor, dict[str, Any]]:
    """Rebuild an arm's architecture and load its trained weights."""
    arm_dir = Path(dataset_dir) / "arms" / arm
    meta = json.loads((arm_dir / "arm_meta.json").read_text())
    raw = dict(meta["config"])
    raw["hidden"] = tuple(raw["hidden"])  # JSON has no tuples; restore the arch shape
    cfg = ModelConfig(**raw)
    model = OneStepPredictor(arm_mask(arm), cfg.hidden)
    state = torch.load(arm_dir / "model.pt", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, meta
