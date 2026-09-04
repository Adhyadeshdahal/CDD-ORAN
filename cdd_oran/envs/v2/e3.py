"""E3 — Temporal depth / fan-out cascade (v2 SCM).

Implements the FROZEN E3 SCM from ``docs/benchmark/GATE_CONTRACT_E3.md`` (reviewer ruling
``review-e3-redesign.md``, "Frozen E3 contract value block") VERBATIM. Nothing here is invented
or tuned: the five unit-coefficient mechanisms, the depth-three cascade
``P0->K0->K1->{K2,K3}``, the K4 non-descendant control, the exact steady-state standardization,
and the xApp/KPI maps are all transcribed from the frozen contract.

Raw latent update equations (one-step lag; ``K*_prev`` is the previous-step raw latent KPI)::

    K0(t) = P0(t-1)                    # edge P0->K0        (coeff 1)
    K1(t) = K0(t-1)                    # edge K0->K1        (coeff 1, internal conduit)
    K2(t) = P1(t-1) + K1(t-1)          # edges P1->K2, K1->K2  (both coeff 1)
    K3(t) = P2(t-1) + K1(t-1)          # edges P2->K3, K1->K3  (both coeff 1)
    K4(t) = P3(t-1)                    # edge P3->K4        (coeff 1, non-descendant control)

The TRUE graph is ``P0->K0->K1->{K2,K3}``, plus ``P1->K2``, ``P2->K3``, ``P3->K4``. K1 is an
internal propagation conduit and K4 is a structural negative control; both are excluded from the
decision panel ``(0,2,3)``.

Coefficient knobs are exposed on the constructor so the SCM gate can build *single-edge shadows*
(one KPI-parent coefficient zeroed) for the direct-parentage / edge-ablation checks; the defaults
reproduce the frozen SCM exactly. ``truncate_fanout`` realizes the ratified decision decoy: it
zeroes K1's contribution to BOTH K2 and K3 and drops adjacency edges ``(2,5),(3,5)`` — the
truncated analytic model against which the horizon-value factor is removed.
"""

from __future__ import annotations

import math

import numpy as np

from cdd_oran.envs.v2.base import V2Env

# FROZEN steady-state ID moments after exactly three neutral advances (contract "Exact
# standardization"). Evaluated at full float precision — NOT six-decimal approximations.
SIGMA_K01 = math.sqrt(1.0 / 12.0)  # K0, K1, K4  (single uniform)
SIGMA_K23 = math.sqrt(2.0 / 12.0)  # K2, K3      (sum of two uniforms)


class E3V2Env(V2Env):
    num_params = 4
    num_kpis = 5
    id_ranges = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    # FROZEN exact standardization. Steady vars = (P0, P0, P1+P0, P2+P0, P3).
    mu = np.array([0.5, 0.5, 1.0, 1.0, 0.5], dtype=float)
    sigma = np.array([SIGMA_K01, SIGMA_K01, SIGMA_K23, SIGMA_K23, SIGMA_K01], dtype=float)

    # TRUE adjacency as (kpi_index, source_index); source_index >= num_params encodes a KPI
    # source (base.py convention): source 4 == K0, source 5 == K1.
    TRUE_ADJACENCY = [(0, 0), (1, 4), (2, 1), (2, 5), (3, 2), (3, 5), (4, 3)]
    # The ratified decision decoy drops exactly this pair (K1's fan-out into K2 and K3).
    TRUNCATED_OMITTED_EDGES = [(2, 5), (3, 5)]

    xapp_kpi_indices = [(0,), (1,), (2,), (3,), (4,)]
    kpi_to_xapp = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    xapp_param_indices = [(0,), (0,), (1,), (2,), (3,)]
    directions = (0, 0, 1, 1, 0)
    kpi_thresholds = (0.5, 0.5, 1.0, 1.0, 0.5)
    # Decision panel: monitored services K0, K2, K3. K1 (conduit) and K4 (non-descendant) excluded.
    panel_kpi_ids = (0, 2, 3)
    excluded_kpi_ids = (1, 4)

    def __init__(
        self,
        env_seed: int = 0,
        obs_noise_scale: float = 0.0,
        a: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        c10: float = 1.0,
        c25: float = 1.0,
        c35: float = 1.0,
        truncate_fanout: bool = False,
        episode: int = 0,
    ) -> None:
        # a = (P0->K0, P1->K2, P2->K3, P3->K4). c10 = K0->K1; c25 = K1->K2; c35 = K1->K3.
        self.a = np.asarray(a, dtype=float)
        self.c10 = float(c10)
        self.truncate_fanout = bool(truncate_fanout)
        # The decoy zeroes K1's contribution to BOTH K2 and K3.
        self.c25 = 0.0 if self.truncate_fanout else float(c25)
        self.c35 = 0.0 if self.truncate_fanout else float(c35)
        super().__init__(env_seed=env_seed, obs_noise_scale=obs_noise_scale, episode=episode)

    def _update_kpis(self, prev_params: np.ndarray, prev_kpis: np.ndarray) -> np.ndarray:
        p = np.asarray(prev_params, dtype=float)
        k = np.asarray(prev_kpis, dtype=float)
        k0 = self.a[0] * p[0]
        k1 = self.c10 * k[0]
        k2 = self.a[1] * p[1] + self.c25 * k[1]
        k3 = self.a[2] * p[2] + self.c35 * k[1]
        k4 = self.a[3] * p[3]
        return np.array([k0, k1, k2, k3, k4], dtype=float)

    @property
    def adjacency_edges(self) -> list[tuple[int, int]]:
        """Structural edges implied by the active coefficients (KPI-parent edges toggle with
        their coefficient; direct-param edges are always present)."""
        edges = [(0, 0)]  # P0 -> K0
        if self.c10 != 0.0:
            edges.append((1, 4))  # K0 -> K1
        edges.append((2, 1))  # P1 -> K2
        if self.c25 != 0.0:
            edges.append((2, 5))  # K1 -> K2
        edges.append((3, 2))  # P2 -> K3
        if self.c35 != 0.0:
            edges.append((3, 5))  # K1 -> K3
        edges.append((4, 3))  # P3 -> K4
        return edges

    def true_adj_matrix(self) -> np.ndarray:
        size = self.num_params + self.num_kpis
        matrix = np.zeros((size, size), dtype=np.float32)
        for kpi_index, source_index in self.adjacency_edges:
            matrix[self.num_params + kpi_index, source_index] = 1.0
        return matrix
