"""Environment III — a harder ORAN env that stresses causal recovery.

Answers the paper's "synthetic-only" limitation by raising dimensionality
(9 NCPs, 7 KPIs vs Env II's 8/6), adding observation NOISE on KPIs, a genuine
two-hop KPI->KPI dependency chain (K0 -> K4 -> K5), and several multi-xApp
shared-NCP conflicts (P0 shared by K0/K1/K6, P4 by K2/K3, P1 by K0/K1).

Matches the BaseORANEnv contract so train/evaluate/viz run unchanged.
"""

from math import exp

import numpy as np

from cdd_oran.config import DEFAULT_CONFIG
from cdd_oran.envs.legacy.base import BaseORANEnv
from cdd_oran.envs.legacy.stats_cache import get_cached_mean_std

# Default observation-noise std on the (normalised) KPI channels. CMI-robustness knob.
DEFAULT_NOISE_SCALE = 0.05


def safe(x):
    return x if abs(x) > 1e-1 else 1e-1


def update_kpi0(p, k):  # P0 (center), P1 (scale)
    return 80 * exp(-(p[0] ** 2) / (2 * safe(p[1]) ** 2))


def update_kpi1(p, k):  # P0+P2 (center), P1 (scale)  -> shares P0 & P1 with K0
    return 100 * exp(-((p[0] + p[2]) ** 2) / (2 * safe(p[1]) ** 2))


def update_kpi2(p, k):  # P3 (center), P4 (scale)
    return 120 * exp(-((p[3] + 45) ** 2) / (2 * safe(p[4]) ** 2))


def update_kpi3(p, k):  # P5 (center), P4 (scale)  -> shares P4 with K2
    return 90 * exp(-((p[5] - 40) ** 2) / (2 * safe(p[4]) ** 2))


def update_kpi4(p, k):  # P6 + K0  -> one-hop KPI->KPI dependency (reads K0)
    return 110 * exp(-((p[6] + k[0] - 20) ** 2) / (2 * (50.0**2)))


def update_kpi5(p, k):  # P7 + K4  -> second hop (K0 -> K4 -> K5)
    return -35 * exp(-((p[7] + k[4] - 40) ** 2) / (2 * (70.0**2)))


def update_kpi6(p, k):  # P8 + P0  -> another conflict on shared NCP P0
    return 70 * exp(-((p[8] + p[0] + 20) ** 2) / (2 * (60.0**2)))


UPDATE_FNS = [
    update_kpi0,
    update_kpi1,
    update_kpi2,
    update_kpi3,
    update_kpi4,
    update_kpi5,
    update_kpi6,
]

# Ground-truth causal graph as (kpi_index, source_index). source_index in [0..8] is a
# param; source_index >= 9 (= num_params) is a KPI source j at column 9+j.
NUM_PARAMS = 9
ADJACENCY_EDGES = [
    (0, 0), (0, 1),                 # K0 <- P0, P1
    (1, 0), (1, 2), (1, 1),         # K1 <- P0, P2, P1
    (2, 3), (2, 4),                 # K2 <- P3, P4
    (3, 5), (3, 4),                 # K3 <- P5, P4
    (4, 6), (4, NUM_PARAMS + 0),    # K4 <- P6, K0   (KPI->KPI)
    (5, 7), (5, NUM_PARAMS + 4),    # K5 <- P7, K4   (KPI->KPI, second hop)
    (6, 8), (6, 0),                 # K6 <- P8, P0
]


def _param_ranges(cfg):
    if cfg.param_ranges == "ood":
        return [
            (100, 150), (50, 100), (-30, 30), (-90, 90), (-30, -19),
            (-50, 150), (66, 87), (-200, 150), (-80, -40),
        ]
    return [
        (-100, 100), (-10, 50), (-20, 20), (-60, 60), (-20, 20),
        (-50, 150), (-60, 65), (-100, 150), (-40, 40),
    ]


def _compute_kpis_iii(params):
    """Vectorised mirror of the scalar update_fns, sequential so K4/K5 see K0/K4."""
    p = params

    def vsafe(x):
        return np.where(np.abs(x) > 1e-1, x, 1e-1)

    k0 = 80 * np.exp(-(p[0] ** 2) / (2 * vsafe(p[1]) ** 2))
    k1 = 100 * np.exp(-((p[0] + p[2]) ** 2) / (2 * vsafe(p[1]) ** 2))
    k2 = 120 * np.exp(-((p[3] + 45) ** 2) / (2 * vsafe(p[4]) ** 2))
    k3 = 90 * np.exp(-((p[5] - 40) ** 2) / (2 * vsafe(p[4]) ** 2))
    k4 = 110 * np.exp(-((p[6] + k0 - 20) ** 2) / (2 * (50.0**2)))
    k5 = -35 * np.exp(-((p[7] + k4 - 40) ** 2) / (2 * (70.0**2)))
    k6 = 70 * np.exp(-((p[8] + p[0] + 20) ** 2) / (2 * (60.0**2)))
    return [k0, k1, k2, k3, k4, k5, k6]


def get_env_iii_mean_std(param_ranges, seed, num_samples=1_000_000):
    def compute():
        rng = np.random.default_rng(seed)
        params = [rng.uniform(low, high, size=num_samples) for low, high in param_ranges]
        kpis = _compute_kpis_iii(params)
        return [(float(np.mean(k)), float(np.std(k))) for k in kpis]

    return get_cached_mean_std(
        "EnvironmentIII", param_ranges, seed, num_samples, compute,
        cache_version="environment-iii-v1",
    )


class ORANEnvironment3(BaseORANEnv):
    def __init__(
        self, num_bins=10, max_steps=50, cfg=DEFAULT_CONFIG,
        noise_scale=DEFAULT_NOISE_SCALE, drift=0.0,
    ):
        self.noise_scale = noise_scale
        self.drift = drift  # non-stationarity knob; OFF (0.0) keeps env A4-friendly
        param_ranges = _param_ranges(cfg)
        super().__init__(
            param_ranges=param_ranges,
            mean_std_kpis=get_env_iii_mean_std(param_ranges, seed=cfg.seed),
            kpi_thresholds=[55, 70, 85, 60, 75, -20, 45],
            kpi_names=["kpi0", "kpi1", "kpi2", "kpi3", "kpi4", "kpi5", "kpi6"],
            directions=[0, 0, 0, 0, 0, 1, 0],
            update_fns=UPDATE_FNS,
            xapp_param_indices=[(0, 1), (0, 2, 1), (3, 4), (5, 4), (6,), (7,), (8, 0)],
            xapp_kpi_indices=[(0,), (1,), (2,), (3,), (4,), (5,), (6,)],
            kpi_to_xapp={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
            adjacency_edges=ADJACENCY_EDGES,
            num_bins=num_bins,
            max_steps=max_steps,
        )

    def get_state(self):
        state = super().get_state()
        if self.noise_scale > 0:
            for kpi in self.kpis:
                noise = np.random.normal(0, self.noise_scale)
                state[kpi.name] = (state[kpi.name] + noise).astype(np.float32)
        if self.drift > 0:
            state["kpi0"] = (state["kpi0"] + np.float32(self.drift * self.cur_step)).astype(
                np.float32
            )
        return state


if __name__ == "__main__":
    env = ORANEnvironment3()
    assert env.num_params == 9 and env.num_kpis == 7, (env.num_params, env.num_kpis)

    state = env.reset()
    assert len(state) == env.get_state_dim() == 16, (len(state), env.get_state_dim())

    action = (0, 5, 0)
    next_state, reward, done, info = env.step(action)
    assert len(next_state) == 16, len(next_state)
    assert isinstance(reward, float), type(reward)

    n = env.num_params + env.num_kpis
    expected = np.zeros((n, n), dtype=np.float32)
    for kpi_index, source in ADJACENCY_EDGES:
        expected[env.num_params + kpi_index, source] = 1
    assert np.array_equal(env.true_adj_matrix, expected), "adjacency mismatch"
    assert env.true_adj_matrix.sum() == len(ADJACENCY_EDGES), env.true_adj_matrix.sum()

    print("env_iii self-check OK: dims 9NCP/7KPI, state=16, edges=", len(ADJACENCY_EDGES))
