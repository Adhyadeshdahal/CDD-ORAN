"""Shared interface of the cross-method discovery study (E1-E5 x data regimes x methods).

Contract: scratchpad/xmethod/CONTRACT.md. This file is owned by the orchestrator; workers code against it and
propose changes in their hand-back instead of editing it.

A ``Dataset`` is one (world, regime, n, seed) corpus of one-step rows. A ``Method`` maps a Dataset to a
``Result`` (one entry per candidate edge). Ground truth never enters a Dataset; it lives in ``Truth`` and is read
only by the scorer.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Protocol

import numpy as np

WORLDS = ("E1", "E2", "E3", "E4", "E5")
REGIMES = ("R1", "R2", "R3", "R4")          # R1 randomised rows, R2 setpoint + dither, R3/R4 E4 only


@dataclasses.dataclass(frozen=True)
class Design:
    """The known assignment design of one action column (what a design-based test may redraw).

    kind:
      "iid"      the whole column is drawn i.i.d. from ``dist`` (R1).
      "dither"   column = setpoint (NOT randomised, slowly varying) + dither (randomised, i.i.d. from ``dist``);
                 ``random_part`` holds the realised dither and ``fixed_part`` the setpoint (R2).
      "logged"   column drawn from a context-dependent policy whose per-row probabilities / densities are logged
                 in ``propensity`` (R3).
      "none"     no known design (R4 latent confounding, or a non-action column).
    dist: {"name": "uniform", "lo": .., "hi": ..} | {"name": "categorical", "values": [...], "p": [...]} | ...
    """
    kind: str
    dist: dict[str, Any] | None = None
    random_part: np.ndarray | None = None
    fixed_part: np.ndarray | None = None
    propensity: np.ndarray | None = None
    redraw_seed_tag: int | None = None


@dataclasses.dataclass(frozen=True)
class Dataset:
    world: str
    regime: str
    n: int
    seed: int
    action_names: tuple[str, ...]           # NCP columns (P0..), the treatments
    kpi_names: tuple[str, ...]              # KPI columns (K0..)
    X_action: np.ndarray                    # [n, p] actions at t
    X_kpi_lag: np.ndarray                   # [n, k] KPIs at t (lagged state), may be all-NaN if the world has none
    Y: np.ndarray                           # [n, k] KPIs at t+1 (targets)
    designs: tuple[Design, ...]             # one per action column
    candidates: tuple[tuple[str, str], ...] # (source, target) pairs scored; source in actions or lagged KPIs
    time_index: np.ndarray | None = None    # [n] row time (for temporal methods, E3)
    context: np.ndarray | None = None       # [n, c] observed context (R3), else None
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class Truth:
    world: str
    regime: str
    edges: frozenset[tuple[str, str]]               # true (source, target) edges among the candidates
    signs: dict[tuple[str, str], int]               # +1 / -1 for edges with a defined sign (else absent)
    null_edges: frozenset[tuple[str, str]]          # candidates with exactly zero effect (for FPR)


@dataclasses.dataclass(frozen=True)
class EdgeResult:
    source: str
    target: str
    score: float            # method's raw edge score (larger = stronger), NaN if not scorable
    p: float | None         # p-value if the method has one
    sign: int               # +1 / -1 / 0 (unsigned)
    declared: bool


@dataclasses.dataclass(frozen=True)
class Result:
    method: str
    version: str
    edges: tuple[EdgeResult, ...]
    cpu_s: float
    config: dict[str, Any]
    notes: dict[str, Any] = dataclasses.field(default_factory=dict)


class Method(Protocol):
    name: str
    version: str

    def tune(self, dev: list[Dataset], truth_free: bool = True) -> dict[str, Any]:
        """Return a frozen config from DEV datasets only (CONTRACT.md section 5 tuning rule)."""

    def run(self, data: Dataset, config: dict[str, Any]) -> Result:
        ...
