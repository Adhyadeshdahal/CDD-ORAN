"""LEGACY (v1) O-RAN environments — RETIRED; kept for archival / reproduction only.

These are Environment I–IV (``env_i..iv``), the ORIGINAL v1 benchmark. They are NOT the active
benchmark and must not be confused with the E-series. The live benchmark is the **E1–E5 redesign**
in :mod:`cdd_oran.envs.v2` (env classes ``E1V2Env..E4V2Env``), with discovery/decision pipelines in
:mod:`cdd_oran.e1slice` / :mod:`cdd_oran.e2slice` and the ``scripts/e*_gate.py`` harnesses. See
``cdd_oran/envs/README.md`` and ``docs/benchmark/`` for the current benchmark and its status.

Do NOT build new work on these modules or on ``get_env`` below. This package is scheduled for
quarantine (deprecation markers + import-seam decoupling now; full removal deferred until the
E-series fully supersedes it — see reports/2026-09-05-planner-redesign-and-env-boundary).
"""

from cdd_oran.config import ExperimentConfig
from cdd_oran.envs.base import BaseORANEnv
from cdd_oran.envs.env_i import ORANEnvironment1
from cdd_oran.envs.env_ii import ORANEnvironment2
from cdd_oran.envs.env_iii import ORANEnvironment3, get_env_iii_mean_std
from cdd_oran.envs.env_iv import ORANEnvironment4, get_env_iv_mean_std
from cdd_oran.envs.statistics import get_env_i_mean_std, get_env_ii_mean_std


def get_env(cfg: ExperimentConfig) -> BaseORANEnv:
    env: BaseORANEnv | None = None
    if cfg.environment == "EnvironmentII":
        env = ORANEnvironment2(cfg=cfg)
    elif cfg.environment == "EnvironmentI":
        env = ORANEnvironment1(cfg=cfg)
    elif cfg.environment == "EnvironmentIII":
        env = ORANEnvironment3(cfg=cfg)
    elif cfg.environment == "EnvironmentIV":
        env = ORANEnvironment4(cfg=cfg)
    else:
        raise NameError(f"env{env} is not valid")
    return env


__all__ = [
    "ORANEnvironment1",
    "ORANEnvironment2",
    "ORANEnvironment3",
    "ORANEnvironment4",
    "get_env",
    "get_env_i_mean_std",
    "get_env_ii_mean_std",
    "get_env_iii_mean_std",
    "get_env_iv_mean_std",
]
