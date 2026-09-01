"""E1 — Clean sanity (recovery control), v2 SCM.

Implements the LOCKED E1 SCM from ``docs/benchmark/SPEC.md`` ("Locked SCM (exact ...)")
verbatim. Nothing here is invented: the linear monotone mechanisms, the ID ranges, the
standardization constants, and the xApp/KPI maps are all taken from the spec.

Raw latent update equations (one-step lag, ``K*_prev`` is the previous-step raw latent KPI)::

    K0(t) = a0 * P0(t-1)                       # edge P0->K0   (a0 = 1)
    K1(t) = a1 * P1(t-1)                       # edge P1->K1   (a1 = 1)
    K2(t) = a2 * P2(t-1) + b2 * K0(t-1)        # edges P2->K2 (a2 = 1), K0->K2 (b2 = 0.5)
    K3(t) = a3 * P3(t-1) + b3 * K1(t-1)        # edges P3->K3 (a3 = 1), K1->K3 (b3 = 0.5)

The coefficients are exposed on the constructor so the SCM gate can build a *shadow* SCM
with a single KPI-parent coefficient zeroed (edge-ablation / direct-parentage check); the
defaults reproduce the locked SCM exactly.
"""

from __future__ import annotations

import numpy as np

from cdd_oran.envs.v2.base import V2Env

# Locked standardization constants (derived analytically under P.~U[0,1]; SPEC E1).
SIGMA_K01 = float(np.sqrt(1.0 / 12.0))  # 0.288675
SIGMA_K23 = float(np.sqrt(1.25 / 12.0))  # 0.322749


class E1V2Env(V2Env):
    num_params = 4
    num_kpis = 4
    id_ranges = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    # Locked standardization constants (used by observed/utility standardization).
    mu = np.array([0.5, 0.5, 0.75, 0.75], dtype=float)
    sigma = np.array([SIGMA_K01, SIGMA_K01, SIGMA_K23, SIGMA_K23], dtype=float)

    # SCM edges as (kpi_index, source_index); source_index >= num_params encodes a KPI source
    # (base.py:134-138 convention): source 4 == K0, source 5 == K1.
    adjacency_edges = [(0, 0), (1, 1), (2, 2), (3, 3), (2, 4), (3, 5)]

    # Constructor-only xApp constants (E1 has NO decision panel; not read by any E1 gate).
    kpi_to_xapp = {0: 0, 1: 1, 2: 2, 3: 3}
    xapp_kpi_indices = [(0,), (1,), (2,), (3,)]
    xapp_param_indices = [(0,), (1,), (2,), (3,)]
    kpi_thresholds = (0.5, 0.5, 0.75, 0.75)
    directions = (0, 0, 0, 0)

    def __init__(
        self,
        env_seed: int = 0,
        obs_noise_scale: float = 0.0,
        a: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        b2: float = 0.5,
        b3: float = 0.5,
        episode: int = 0,
    ) -> None:
        self.a = np.asarray(a, dtype=float)
        self.b2 = float(b2)
        self.b3 = float(b3)
        super().__init__(env_seed=env_seed, obs_noise_scale=obs_noise_scale, episode=episode)

    def _update_kpis(self, prev_params: np.ndarray, prev_kpis: np.ndarray) -> np.ndarray:
        p = np.asarray(prev_params, dtype=float)
        k = np.asarray(prev_kpis, dtype=float)
        k0 = self.a[0] * p[0]
        k1 = self.a[1] * p[1]
        k2 = self.a[2] * p[2] + self.b2 * k[0]
        k3 = self.a[3] * p[3] + self.b3 * k[1]
        return np.array([k0, k1, k2, k3], dtype=float)

    def true_adj_matrix(self) -> np.ndarray:
        size = self.num_params + self.num_kpis
        matrix = np.zeros((size, size), dtype=np.float32)
        for kpi_index, source_index in self.adjacency_edges:
            matrix[self.num_params + kpi_index, source_index] = 1.0
        return matrix
