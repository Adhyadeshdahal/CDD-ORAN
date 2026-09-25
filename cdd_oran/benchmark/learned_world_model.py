"""Learned world model: per-KPI MLP ensembles fitted on discovered parents (the deployable decision path).

Each KPI is predicted by a small ensemble of MLPs whose inputs are that KPI's parents in a graph (e.g. the
MSCR graph). The resulting object is an env subclass whose ``_update_kpis`` uses ONLY the learned models,
so any V2 planner (``cdd_oran.planners.sequence``) can roll it like the true simulator. No true mechanism is
used for any KPI.

Inputs are laid out as ``[lagged params (num_params) | lagged KPIs (num_kpis)]``; a parent index ``>=
num_params`` refers to a lagged KPI. Settings are the DEV "lean" configuration fixed for decision study D1
(3 members, 2x64 SiLU, 1500 full-batch Adam epochs with cosine decay).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch


class MLPMember:
    """One 2x64 SiLU MLP on standardized inputs/target; prediction runs in numpy for rollout speed."""

    def __init__(self, x: np.ndarray, y: np.ndarray, seed: int, epochs: int = 1500, lr: float = 3e-3,
                 group_l1: float = 0.0):
        torch.manual_seed(seed)
        self.mx, self.sx = x.mean(0), x.std(0) + 1e-12
        self.my, self.sy = float(y.mean()), float(y.std()) + 1e-12
        xt = torch.tensor((x - self.mx) / self.sx, dtype=torch.float32)
        yt = torch.tensor((y - self.my) / self.sy, dtype=torch.float32)[:, None]
        net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 64), torch.nn.SiLU(),
                                  torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1))
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        for _ in range(epochs):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(xt), yt)
            if group_l1:  # group lasso: one L2 norm per input over its first-layer weight column
                loss = loss + group_l1 * torch.linalg.vector_norm(net[0].weight, dim=0).sum()
            loss.backward()
            opt.step()
            sched.step()
        self.layers = [(m.weight.detach().numpy().T, m.bias.detach().numpy()) for m in net if hasattr(m, "weight")]

    def predict(self, x: np.ndarray) -> np.ndarray:
        h = (np.atleast_2d(x) - self.mx) / self.sx
        for i, (w, b) in enumerate(self.layers):
            h = h @ w + b
            if i < len(self.layers) - 1:
                h = h / (1.0 + np.exp(-h))  # SiLU
        return h[:, 0] * self.sy + self.my


def fit_kpi_models(
    x: np.ndarray,
    y: np.ndarray,
    parents: Mapping[int, Sequence[int]],
    members: int = 3,
    epochs: int = 1500,
    group_l1: Mapping[int, float] | float = 0.0,
) -> list:
    """One predictor per KPI column of ``y``: mean over ``members`` MLPs on ``x[:, parents[k]]``.
    A KPI with no parents is predicted as its corpus mean. ``group_l1`` (per KPI or global) adds a group-lasso
    penalty on each input's first-layer weights (0 = none, the D1 default)."""
    models = []
    for k in range(y.shape[1]):
        par = list(parents.get(k, []))
        if not par:
            mu = float(y[:, k].mean())
            models.append(lambda z, mu=mu: mu)
            continue
        lam = group_l1.get(k, 0.0) if isinstance(group_l1, Mapping) else group_l1
        ens = [MLPMember(x[:, par], y[:, k], seed=m, epochs=epochs, group_l1=lam) for m in range(members)]
        models.append(lambda z, e=ens, pa=par: float(np.mean([m.predict(z[pa][None, :])[0] for m in e])))
    return models


def learned_env_class(base_cls, models, num_params: int):
    """Subclass of ``base_cls`` whose KPI update uses only the learned ``models``."""

    class _Learned(base_cls):
        def _update_kpis(self, prev_params, prev_kpis):
            z = np.concatenate([np.asarray(prev_params, float)[:num_params], np.asarray(prev_kpis, float)])
            return np.array([m(z) for m in models], dtype=float)

    _Learned.__name__ = f"Learned{base_cls.__name__}"
    return _Learned
