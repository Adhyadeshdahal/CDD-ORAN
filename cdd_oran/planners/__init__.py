from cdd_oran.config import ExperimentConfig
from cdd_oran.planners.cem import ModelBasedCEM
from cdd_oran.planners.horizon import RecedingHorizonCEM
from cdd_oran.planners.joint import JointMultiNCPPlanner
from cdd_oran.planners.mcts import ModelBasedMCTS
from cdd_oran.planners.mppi import ModelBasedMPPI
from cdd_oran.planners.qacm import QACM


def get_planners(cfg: ExperimentConfig, model, env):
    qacm = QACM(model=model, env=env)
    cem = ModelBasedCEM(model=model, env=env, **vars(cfg.planner.cem))
    mppi = ModelBasedMPPI(model=model, env=env, **vars(cfg.planner.mppi))
    mcts = ModelBasedMCTS(model=model, env=env, **vars(cfg.planner.mcts))
    planners = [qacm, cem, mppi, mcts]
    # New planners are appended AFTER the original four so the shared RNG stream the
    # four consume is unchanged -> their H=1 results stay bit-reproducible.
    if cfg.planner.n_horizon > 1:
        planners.append(
            RecedingHorizonCEM(
                model=model, env=env, n_horizon=cfg.planner.n_horizon, **vars(cfg.planner.cem)
            )
        )
    if getattr(cfg.planner, "joint", False):
        planners.append(JointMultiNCPPlanner(model=model, env=env, **vars(cfg.planner.cem)))
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
