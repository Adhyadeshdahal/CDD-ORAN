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
