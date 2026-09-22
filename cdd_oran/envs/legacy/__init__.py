"""LEGACY (v1) O-RAN environments — RETIRED; kept for archival / reproduction only.

These are Environment I–IV (``env_i..iv``), the ORIGINAL v1 benchmark. They are NOT the active
benchmark and must not be confused with the E-series. The live benchmark is the **E1–E5 redesign**
in :mod:`cdd_oran.envs.v2` (env classes ``E1V2Env..E4V2Env``), with discovery/decision pipelines in
:mod:`cdd_oran.e1slice` / :mod:`cdd_oran.e2slice` and the ``scripts/e*_gate.py`` harnesses.

**Do NOT build new work on these modules or on ``get_env`` below.** Quarantined under
``cdd_oran/envs/legacy/`` (moved 2026-09-22); full removal is gated on porting the kept baselines to
``V2Env`` — see ``plans/012`` (legacy retirement) and ``plans/014`` (baselines). Import legacy names
explicitly from here, e.g. ``from cdd_oran.envs.legacy import get_env``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cdd_oran.config import ExperimentConfig
from cdd_oran.envs.legacy.base import BaseORANEnv

if TYPE_CHECKING:
    # For type checkers / IDEs only — never executed at runtime, so it loads no legacy env.
    from cdd_oran.envs.legacy.env_i import ORANEnvironment1
    from cdd_oran.envs.legacy.env_ii import ORANEnvironment2
    from cdd_oran.envs.legacy.env_iii import ORANEnvironment3, get_env_iii_mean_std
    from cdd_oran.envs.legacy.env_iv import ORANEnvironment4, get_env_iv_mean_std
    from cdd_oran.envs.legacy.statistics import get_env_i_mean_std, get_env_ii_mean_std


def get_env(cfg: ExperimentConfig) -> BaseORANEnv:
    # Legacy env classes imported HERE (not at module top) so importing this package is env-free.
    from cdd_oran.envs.legacy.env_i import ORANEnvironment1
    from cdd_oran.envs.legacy.env_ii import ORANEnvironment2
    from cdd_oran.envs.legacy.env_iii import ORANEnvironment3
    from cdd_oran.envs.legacy.env_iv import ORANEnvironment4

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


# PEP 562 lazy re-exports: (module, attribute) each name resolves to on first access. Kept out of
# module top-level so this package imports env-free; legacy callers still get them on demand.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "ORANEnvironment1": ("cdd_oran.envs.legacy.env_i", "ORANEnvironment1"),
    "ORANEnvironment2": ("cdd_oran.envs.legacy.env_ii", "ORANEnvironment2"),
    "ORANEnvironment3": ("cdd_oran.envs.legacy.env_iii", "ORANEnvironment3"),
    "ORANEnvironment4": ("cdd_oran.envs.legacy.env_iv", "ORANEnvironment4"),
    "get_env_i_mean_std": ("cdd_oran.envs.legacy.statistics", "get_env_i_mean_std"),
    "get_env_ii_mean_std": ("cdd_oran.envs.legacy.statistics", "get_env_ii_mean_std"),
    "get_env_iii_mean_std": ("cdd_oran.envs.legacy.env_iii", "get_env_iii_mean_std"),
    "get_env_iv_mean_std": ("cdd_oran.envs.legacy.env_iv", "get_env_iv_mean_std"),
}


def __getattr__(name: str) -> object:  # PEP 562
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(target[0]), target[1])
    globals()[name] = value  # cache so subsequent access skips __getattr__
    return value


def __dir__() -> list[str]:
    return sorted([*globals().keys(), *_LAZY_ATTRS])


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
