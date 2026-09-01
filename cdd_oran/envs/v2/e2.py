"""E2 — Shared-control conflict (anchored in the validated CFCP fan-out), v2 SCM.

Reimplements the Env II fan-out **structure** (``env_ii.py:12-105``) on the v2 ``V2Env`` base
(apply/advance split, one-step actuation latency, coordinate-keyed CRN tape, noise-off latent
scoring). Legacy ``env_ii.py`` is left byte-identical (PD4 / benchmark-version boundary).

Everything here is FROZEN by ``docs/benchmark/GATE_CONTRACT_E2.md`` (reviewer ruling
``review-e2-buildplan.md``); nothing is re-derived or tuned:

- **Mechanisms (Q3):** the six Env II Gaussian-bump equations VERBATIM, including ``safe_exp``,
  coefficients, thresholds, directions, ID ranges, and xApp maps.
- **Adjacency (Q2):** the code-authoritative edge list; TRUE ``P0 -> {K0,K1,K2,K5}`` (no
  ``P0->K3``).
- **Standardization (Q4):** ``get_env_ii_mean_std(seed=0, num_samples=1_000_000)`` computed once
  and pasted as literals (provenance in ``KPI_MEAN_STD``).
- **Decoy:** ``decoy_omit_p0_k5=True`` drops edge ``(5,0)`` from the adjacency and freezes the
  committed pre-decision ``P0`` **inside the K5 mechanism only** (P6/P7 stay live), so the decoy
  world model predicts *no* effect of moving ``P0`` on ``K5``. All other mechanisms stay TRUE;
  realized scoring always uses the TRUE SCM.

The shared control is the single param ``P0``. Fan-out into ``{K0,K1,K2,K5}`` reaches the
conflict panel ``{xApp0,xApp1,xApp2,xApp4}`` (xApp3 has no ``P0`` in its params).
"""

from __future__ import annotations

from math import exp

import numpy as np

from cdd_oran.envs.v2.base import V2Env


def safe_exp(x: float) -> float:
    """Env II ``safe_exp`` VERBATIM (``env_ii.py:8-9``): x if |x| > 0.1 else 0.1."""
    return x if abs(x) > 1e-1 else 1e-1


# Frozen per-KPI (mean, std) standardization constants (Q4). Provenance:
# get_env_ii_mean_std(param_ranges=ID, seed=0, num_samples=1_000_000) over compute_kpis_ii.
KPI_MEAN_STD: list[tuple[float, float]] = [
    (21.482606, 27.694787),  # K0
    (26.776967, 34.651755),  # K1
    (40.931657, 44.767996),  # K2
    (14.975340, 32.369335),  # K3
    (18.755966, 40.460422),  # K4
    (-10.140367, 12.583744),  # K5
]


class E2V2Env(V2Env):
    num_params = 8
    num_kpis = 6

    # ID param ranges (non-ood branch, env_ii.py:56-65).
    id_ranges = [
        (-100.0, 100.0),  # P0
        (-10.0, 50.0),    # P1
        (-20.0, 20.0),    # P2
        (-60.0, 60.0),    # P3
        (-20.0, 20.0),    # P4
        (-50.0, 150.0),   # P5
        (-60.0, 65.0),    # P6
        (-100.0, 150.0),  # P7
    ]

    # Frozen standardization (as literals; see KPI_MEAN_STD provenance).
    mean = np.array([m for m, _ in KPI_MEAN_STD], dtype=float)
    std = np.array([s for _, s in KPI_MEAN_STD], dtype=float)

    # Code-authoritative adjacency (env_ii.py:88-105); source_index < num_params is a param.
    _TRUE_ADJACENCY = [
        (0, 0), (0, 1),
        (1, 0), (1, 1), (1, 2),
        (2, 3), (2, 0),
        (3, 1), (3, 4), (3, 5),
        (4, 1), (4, 4), (4, 5),
        (5, 0), (5, 6), (5, 7),
    ]
    _DECOY_OMITTED_EDGE = (5, 0)  # P0 -> K5

    # xApp maps / directions / thresholds (env_ii.py:74-87).
    kpi_to_xapp = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4}
    xapp_kpi_indices = [(0,), (1,), (2,), (3, 4), (5,)]
    xapp_param_indices = [(0, 1), (0, 1, 2), (0, 3), (4, 5, 1), (0, 6, 7)]
    directions = (0, 0, 0, 0, 0, 1)
    kpi_thresholds = (55.0, 95.0, 85.0, 75.0, 80.0, -25.0)

    def __init__(
        self,
        env_seed: int = 0,
        obs_noise_scale: float = 0.0,
        decoy_omit_p0_k5: bool = False,
        episode: int = 0,
    ) -> None:
        self.decoy_omit_p0_k5 = bool(decoy_omit_p0_k5)
        # Committed pre-decision P0 used by the decoy K5 mechanism; captured at reset/restore.
        self._decoy_p0_ref = 0.0
        super().__init__(env_seed=env_seed, obs_noise_scale=obs_noise_scale, episode=episode)

    # --- decoy reference capture -------------------------------------------------------------
    def reset(self, episode: int | None = None):
        state = super().reset(episode=episode)
        self._decoy_p0_ref = float(self.prev_params[0])
        return state

    def restore(self, snap: tuple) -> None:
        super().restore(snap)
        # The decoy freezes P0 at the COMMITTED pre-decision value (the restored state's P0),
        # which stays fixed through the H=1 rollout even after apply_action moves the live P0.
        self._decoy_p0_ref = float(self.prev_params[0])

    # --- mechanism ---------------------------------------------------------------------------
    def _update_kpis(self, prev_params: np.ndarray, prev_kpis: np.ndarray) -> np.ndarray:
        p = np.asarray(prev_params, dtype=float)
        # Decoy: use the frozen committed P0 for the K5 term only (edge (5,0) omitted); the
        # true env uses the live p[0].
        p0_k5 = self._decoy_p0_ref if self.decoy_omit_p0_k5 else p[0]
        k0 = 80.0 * exp(-((p[0]) ** 2) / (2.0 * safe_exp(p[1]) ** 2))
        k1 = 100.0 * exp(-((p[0] + p[2]) ** 2) / (2.0 * safe_exp(p[1]) ** 2))
        k2 = 120.0 * exp(-((p[0] + 45.0) ** 2) / (2.0 * safe_exp(p[3]) ** 2))
        k3 = 120.0 * exp(-((p[5] + p[1] - 30.0) ** 2) / (2.0 * safe_exp(p[4]) ** 2))
        k4 = 150.0 * exp(-((p[5] + p[1] - 50.0) ** 2) / (2.0 * safe_exp(p[4]) ** 2))
        k5 = -35.0 * exp(-((p[7] + p0_k5 - 25.0) ** 2) / (2.0 * safe_exp(p[6]) ** 2))
        return np.array([k0, k1, k2, k3, k4, k5], dtype=float)

    # --- structure ---------------------------------------------------------------------------
    @property
    def adjacency_edges(self) -> list[tuple[int, int]]:
        """TRUE edges, or TRUE minus (5,0) when the decoy omits P0->K5."""
        if self.decoy_omit_p0_k5:
            return [e for e in self._TRUE_ADJACENCY if e != self._DECOY_OMITTED_EDGE]
        return list(self._TRUE_ADJACENCY)

    def true_adj_matrix(self) -> np.ndarray:
        size = self.num_params + self.num_kpis
        matrix = np.zeros((size, size), dtype=np.float32)
        for kpi_index, source_index in self.adjacency_edges:
            matrix[self.num_params + kpi_index, source_index] = 1.0
        return matrix
