"""v2 benchmark execution substrate.

Currently exposes the open-loop episodic rollout kernel (``rollout``): the shared, env-agnostic
executor of the ``docs/benchmark/SEMANTICS.md`` §1 call list, its exhaustive open-loop oracle
(§3), and paired unclamped regret (§3). See ``cdd_oran/benchmark/rollout.py`` for the contract.
"""

from __future__ import annotations

from cdd_oran.benchmark.rollout import (
    Action,
    EnvFactory,
    OracleResult,
    RegretResult,
    RolloutResult,
    ScoreFn,
    enumerate_open_loop,
    paired_regret,
    rollout_open_loop,
)

__all__ = [
    "Action",
    "EnvFactory",
    "OracleResult",
    "RegretResult",
    "RolloutResult",
    "ScoreFn",
    "enumerate_open_loop",
    "paired_regret",
    "rollout_open_loop",
]
