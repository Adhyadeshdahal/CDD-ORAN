"""LEGACY (v1) O-RAN environments — RETIRED; kept for archival / reproduction only.

These are Environment I–IV (``env_i..iv``), the ORIGINAL v1 benchmark. They are NOT the active
benchmark and must not be confused with the E-series. The live benchmark is the **E1–E5 redesign**
in :mod:`cdd_oran.envs.v2` (env classes ``E1V2Env..E4V2Env``), with discovery/decision pipelines in
:mod:`cdd_oran.e1slice` / :mod:`cdd_oran.e2slice` and the ``scripts/e*_gate.py`` harnesses. See
``cdd_oran/envs/README.md`` and ``docs/benchmark/`` for the current benchmark and its status.

Do NOT build new work on these modules or on ``get_env`` below. This package is scheduled for
quarantine (deprecation markers + import-seam decoupling now; full removal deferred until the
E-series fully supersedes it — see reports/2026-09-05-planner-redesign-and-env-boundary).

Import-seam note (quarantine hygiene): the four legacy env classes are imported LAZILY — inside
``get_env`` and, for the re-exported ``get_env_*_mean_std`` names, via a PEP 562 module-level
``__getattr__``. Importing this package (which happens transitively whenever ANY
``cdd_oran.envs.v2.*`` submodule is imported — the live E-series env truth) therefore no longer
eager-loads ``env_i..iv``. Legacy callers of ``get_env`` / ``get_env_*_mean_std`` are unaffected;
they trigger the underlying import on first use exactly as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cdd_oran.config import ExperimentConfig
from cdd_oran.envs.base import BaseORANEnv

if TYPE_CHECKING:
    # For type checkers / IDEs only — never executed at runtime, so it loads no legacy env.
    from cdd_oran.envs.env_i import ORANEnvironment1
    from cdd_oran.envs.env_ii import ORANEnvironment2
    from cdd_oran.envs.env_iii import ORANEnvironment3, get_env_iii_mean_std
    from cdd_oran.envs.env_iv import ORANEnvironment4, get_env_iv_mean_std
    from cdd_oran.envs.statistics import get_env_i_mean_std, get_env_ii_mean_std


def get_env(cfg: ExperimentConfig) -> BaseORANEnv:
    # Legacy env classes imported HERE (not at module top) so importing this package is env-free.
    from cdd_oran.envs.env_i import ORANEnvironment1
    from cdd_oran.envs.env_ii import ORANEnvironment2
    from cdd_oran.envs.env_iii import ORANEnvironment3
    from cdd_oran.envs.env_iv import ORANEnvironment4

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
    "ORANEnvironment1": ("cdd_oran.envs.env_i", "ORANEnvironment1"),
    "ORANEnvironment2": ("cdd_oran.envs.env_ii", "ORANEnvironment2"),
    "ORANEnvironment3": ("cdd_oran.envs.env_iii", "ORANEnvironment3"),
    "ORANEnvironment4": ("cdd_oran.envs.env_iv", "ORANEnvironment4"),
    "get_env_i_mean_std": ("cdd_oran.envs.statistics", "get_env_i_mean_std"),
    "get_env_ii_mean_std": ("cdd_oran.envs.statistics", "get_env_ii_mean_std"),
    "get_env_iii_mean_std": ("cdd_oran.envs.env_iii", "get_env_iii_mean_std"),
    "get_env_iv_mean_std": ("cdd_oran.envs.env_iv", "get_env_iv_mean_std"),
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
