from cdd_oran.config import ExperimentConfig
from cdd_oran.planners.cem import ModelBasedCEM
from cdd_oran.planners.ensemble import EnsembleAggregator
from cdd_oran.planners.horizon import RecedingHorizonCEM
from cdd_oran.planners.joint import JointMultiNCPPlanner
from cdd_oran.planners.mcts import ModelBasedMCTS
from cdd_oran.planners.mppi import ModelBasedMPPI
from cdd_oran.planners.qacm import QACM


def build_aggregator(cfg: ExperimentConfig) -> EnsembleAggregator:
    """Assemble the Phase 3 return-aggregator from config. ``ensemble_kappa`` defaults to
    the planner's ``risk_kappa`` so the "kappa" aggregator reuses the same risk knob."""
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
