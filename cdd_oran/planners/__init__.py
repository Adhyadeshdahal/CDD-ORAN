"""Planner package.

The legacy planners (QACM, CEM, MPPI, MCTS, receding-horizon CEM, joint multi-NCP) and the
ensemble aggregator are loaded LAZILY (PEP 562 ``__getattr__`` + imports inside the factory
functions). Their modules pull in the RETIRED legacy env stack (``cost.py`` ->
``cdd_oran.envs.legacy``), so an eager import here would drag legacy into every live V2 consumer of
a submodule such as ``cdd_oran.planners.sequence``. Public names and behavior are unchanged.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cdd_oran.config import ExperimentConfig
    from cdd_oran.planners.ensemble import EnsembleAggregator

_LAZY = {
    "ModelBasedCEM": "cdd_oran.planners.cem",
    "EnsembleAggregator": "cdd_oran.planners.ensemble",
    "RecedingHorizonCEM": "cdd_oran.planners.horizon",
    "JointMultiNCPPlanner": "cdd_oran.planners.joint",
    "ModelBasedMCTS": "cdd_oran.planners.mcts",
    "ModelBasedMPPI": "cdd_oran.planners.mppi",
    "QACM": "cdd_oran.planners.qacm",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def build_aggregator(cfg: ExperimentConfig) -> EnsembleAggregator:
    """Assemble the Phase 3 return-aggregator from config. ``ensemble_kappa`` defaults to
    the planner's ``risk_kappa`` so the "kappa" aggregator reuses the same risk knob."""
    from cdd_oran.planners.ensemble import EnsembleAggregator

    kappa = getattr(cfg.planner, "risk_kappa", 0.0)
    ens_kappa = getattr(cfg.planner, "ensemble_kappa", None)
    return EnsembleAggregator(
        method=getattr(cfg.planner, "ensemble_aggregator", "quantile"),
        quantile=getattr(cfg.planner, "ensemble_quantile", 0.9),
        kappa=kappa if ens_kappa is None else ens_kappa,
        disagreement_penalty=getattr(cfg.planner, "disagreement_penalty", 0.0),
        utility_weight=getattr(cfg.planner, "utility_weight", 0.0),
        ood_threshold=getattr(cfg.planner, "ood_threshold", None),
    )


def get_planners(cfg: ExperimentConfig, model, env):
    # Same import order as the former module-level imports (behavior-neutral).
    from cdd_oran.planners.cem import ModelBasedCEM
    from cdd_oran.planners.horizon import RecedingHorizonCEM
    from cdd_oran.planners.joint import JointMultiNCPPlanner
    from cdd_oran.planners.mcts import ModelBasedMCTS
    from cdd_oran.planners.mppi import ModelBasedMPPI
    from cdd_oran.planners.qacm import QACM

    kappa = getattr(cfg.planner, "risk_kappa", 0.0)
    aggregator = build_aggregator(cfg)
    qacm = QACM(model=model, env=env, risk_kappa=kappa, aggregator=aggregator)
    cem = ModelBasedCEM(
        model=model, env=env, risk_kappa=kappa, aggregator=aggregator, **vars(cfg.planner.cem)
    )
    mppi = ModelBasedMPPI(
        model=model, env=env, risk_kappa=kappa, aggregator=aggregator, **vars(cfg.planner.mppi)
    )
    mcts = ModelBasedMCTS(
        model=model, env=env, risk_kappa=kappa, aggregator=aggregator, **vars(cfg.planner.mcts)
    )
    planners = [qacm, cem, mppi, mcts]
    # New planners are appended AFTER the original four so the shared RNG stream the
    # four consume is unchanged -> their H=1 results stay bit-reproducible.
    if cfg.planner.n_horizon > 1:
        planners.append(
            RecedingHorizonCEM(
                model=model, env=env, n_horizon=cfg.planner.n_horizon,
                risk_kappa=kappa, aggregator=aggregator, **vars(cfg.planner.cem),
            )
        )
    if getattr(cfg.planner, "joint", False):
        planners.append(
            JointMultiNCPPlanner(
                model=model, env=env, risk_kappa=kappa, aggregator=aggregator,
                **vars(cfg.planner.cem),
            )
        )
    return planners


__all__ = [
    "get_planners",
    "ModelBasedCEM",
    "ModelBasedMCTS",
    "ModelBasedMPPI",
    "QACM",
    "RecedingHorizonCEM",
    "JointMultiNCPPlanner",
]
