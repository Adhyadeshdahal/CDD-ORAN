"""Environment IV — latent-confounder benchmark that separates causal from correlational.

Env I/II/III are "clean": every KPI-KPI correlation has a real edge behind it, so a
correlation graph and a causal/CMI method recover almost the same structure. Env IV
adds the case they must disagree on: a HIDDEN confounder.

Dimensionality (10 NCPs P0..P9, 8 KPIs K0..K7 -> state dim 18) is larger and denser
than Env II (8/6) and Env III (9/7). It keeps Env III's genuine two-hop KPI->KPI
chain (K0 -> K4 -> K5) and shared-NCP conflicts (P0 shared by K0/K1, P1 by K0/K1,
P4 by K2/K3), and ADDS:

LATENT CONFOUNDER Z (the point)
------------------------------
Z is a hidden scalar sampled per step (Z ~ Normal(0, z_scale)). It is NOT a param,
NOT an NCP, NOT a KPI, and NOT in the observed state. Each step the SAME draw Z is
injected into the observation of K6 and K7 (with fixed gains beta6, beta7 -- think a
shared interference / cell-load term measured into both KPIs). Because Z is unobserved
and shared, K6 and K7 are CORRELATED even after conditioning on every observed variable
(params + all other KPIs) -- but there is NO causal edge between them and Z is not a
param, so no NCP->K6/K7 confounding edge either.

Ground truth (true_adj_matrix) therefore has NO K6<->K7 edge. A correlation / partial-
correlation graph sees the residual K6-K7 correlation and ADDS a spurious edge; a causal
method that demands a mechanism (an intervention on some param must change the K6-K7
relationship) finds none -- changing any NCP does not create the K6-K7 link -- so it
should NOT add the edge. That gap is the metric Env IV exists to expose.

So the env carries BOTH a real KPI->KPI edge (K0->K4->K5) AND a confounded non-edge
(K6,K7): recovery must keep the former and reject the latter.

Knobs (both reproducible with a fixed seed):
  - noise_scale: independent observation noise per KPI (Env III parity knob).
  - z_scale:     latent-confounder strength on K6,K7.
  - drift:       OPTIONAL non-stationary observation drift, OFF (0.0) by default.

Matches the BaseORANEnv contract exactly so recovery_gate, ablations, train, evaluate,
and viz run unchanged.
"""

from math import exp

import numpy as np

from cdd_oran.config import DEFAULT_CONFIG
from cdd_oran.envs.base import BaseORANEnv
from cdd_oran.envs.stats_cache import get_cached_mean_std

DEFAULT_NOISE_SCALE = 0.05
# Latent-confounder strength (normalised KPI units) shared by K6 and K7.
DEFAULT_Z_SCALE = 0.30
# The two KPIs the hidden Z drives, and their gains. No edge exists between them.
CONFOUNDED_KPIS = (6, 7)
Z_GAINS = (1.0, 0.8)


def safe(x):
    return x if abs(x) > 1e-1 else 1e-1


def update_kpi0(p, k):  # P0 (center), P1 (scale)
    return 80 * exp(-(p[0] ** 2) / (2 * safe(p[1]) ** 2))


def update_kpi1(p, k):  # P0+P2 (center), P1 (scale) -> shares P0 & P1 with K0
    return 100 * exp(-((p[0] + p[2]) ** 2) / (2 * safe(p[1]) ** 2))


def update_kpi2(p, k):  # P3 (center), P4 (scale)
    return 120 * exp(-((p[3] + 45) ** 2) / (2 * safe(p[4]) ** 2))


def update_kpi3(p, k):  # P5 (center), P4 (scale) -> shares P4 with K2
    return 90 * exp(-((p[5] - 40) ** 2) / (2 * safe(p[4]) ** 2))


def update_kpi4(p, k):  # P6 + K0 -> one-hop KPI->KPI dependency (reads K0)
    return 110 * exp(-((p[6] + k[0] - 20) ** 2) / (2 * (50.0**2)))


def update_kpi5(p, k):  # P7 + K4 -> second hop (K0 -> K4 -> K5)
    return -35 * exp(-((p[7] + k[4] - 40) ** 2) / (2 * (70.0**2)))


def update_kpi6(p, k):  # P8 only. Its correlation with K7 comes from the latent Z, not p.
    return 70 * exp(-((p[8] + 20) ** 2) / (2 * (60.0**2)))


def update_kpi7(p, k):  # P9 only. Confounded with K6 through the hidden Z.
    return 60 * exp(-((p[9] - 10) ** 2) / (2 * (55.0**2)))


UPDATE_FNS = [
    update_kpi0,
    update_kpi1,
    update_kpi2,
    update_kpi3,
    update_kpi4,
    update_kpi5,
    update_kpi6,
    update_kpi7,
]

# Ground-truth causal graph as (kpi_index, source_index). source_index in [0..9] is a
# param; source_index >= 10 (= num_params) is a KPI source j at column 10+j. There is
# deliberately NO (6, 10+7) or (7, 10+6) edge: the K6-K7 correlation is confounded by Z.
NUM_PARAMS = 10
ADJACENCY_EDGES = [
    (0, 0), (0, 1),                 # K0 <- P0, P1
    (1, 0), (1, 2), (1, 1),         # K1 <- P0, P2, P1   (shared-NCP conflict on P0, P1)
    (2, 3), (2, 4),                 # K2 <- P3, P4
    (3, 5), (3, 4),                 # K3 <- P5, P4       (shared-NCP conflict on P4)
    (4, 6), (4, NUM_PARAMS + 0),    # K4 <- P6, K0       (KPI->KPI, one hop)
    (5, 7), (5, NUM_PARAMS + 4),    # K5 <- P7, K4       (KPI->KPI, second hop)
    (6, 8),                         # K6 <- P8           (confounded with K7 via Z: no edge)
    (7, 9),                         # K7 <- P9           (confounded with K6 via Z: no edge)
]


def _param_ranges(cfg):
    if cfg.param_ranges == "ood":
        return [
            (100, 150), (50, 100), (-30, 30), (-90, 90), (-30, -19),
            (-50, 150), (66, 87), (-200, 150), (-80, -40), (40, 80),
        ]
    return [
        (-100, 100), (-10, 50), (-20, 20), (-60, 60), (-20, 20),
        (-50, 150), (-60, 65), (-100, 150), (-40, 40), (-50, 50),
    ]


def _compute_kpis_iv(params):
    """Vectorised mirror of the scalar update_fns, sequential so K4/K5 see K0/K4.

    Z is deliberately absent here: it is a zero-mean observation-channel confounder,
    so the normalisation mean/std are computed on the clean KPI signal (Env III style).
    """
    p = params

    def vsafe(x):
        return np.where(np.abs(x) > 1e-1, x, 1e-1)

    k0 = 80 * np.exp(-(p[0] ** 2) / (2 * vsafe(p[1]) ** 2))
    k1 = 100 * np.exp(-((p[0] + p[2]) ** 2) / (2 * vsafe(p[1]) ** 2))
    k2 = 120 * np.exp(-((p[3] + 45) ** 2) / (2 * vsafe(p[4]) ** 2))
    k3 = 90 * np.exp(-((p[5] - 40) ** 2) / (2 * vsafe(p[4]) ** 2))
    k4 = 110 * np.exp(-((p[6] + k0 - 20) ** 2) / (2 * (50.0**2)))
    k5 = -35 * np.exp(-((p[7] + k4 - 40) ** 2) / (2 * (70.0**2)))
    k6 = 70 * np.exp(-((p[8] + 20) ** 2) / (2 * (60.0**2)))
    k7 = 60 * np.exp(-((p[9] - 10) ** 2) / (2 * (55.0**2)))
    return [k0, k1, k2, k3, k4, k5, k6, k7]


def get_env_iv_mean_std(param_ranges, seed, num_samples=1_000_000):
    def compute():
        rng = np.random.default_rng(seed)
        params = [rng.uniform(low, high, size=num_samples) for low, high in param_ranges]
        kpis = _compute_kpis_iv(params)
        return [(float(np.mean(k)), float(np.std(k))) for k in kpis]

    return get_cached_mean_std(
        "EnvironmentIV", param_ranges, seed, num_samples, compute,
        cache_version="environment-iv-v1",
    )


class ORANEnvironment4(BaseORANEnv):
    def __init__(
        self, num_bins=10, max_steps=50, cfg=DEFAULT_CONFIG,
        noise_scale=DEFAULT_NOISE_SCALE, z_scale=DEFAULT_Z_SCALE, drift=0.0,
    ):
        # Set observation knobs BEFORE super().__init__ -- BaseORANEnv.__init__ calls
        # reset() -> get_state(), which reads them.
        self.noise_scale = noise_scale
        self.z_scale = z_scale
        self.drift = drift  # non-stationarity knob; OFF (0.0) keeps env A4-friendly
        self._forced_z = None  # test hook: pin the latent Z draw (see self-check)
        param_ranges = _param_ranges(cfg)
        super().__init__(
            param_ranges=param_ranges,
            mean_std_kpis=get_env_iv_mean_std(param_ranges, seed=cfg.seed),
            kpi_thresholds=[55, 70, 85, 60, 75, -20, 45, 40],
            kpi_names=["kpi0", "kpi1", "kpi2", "kpi3", "kpi4", "kpi5", "kpi6", "kpi7"],
            directions=[0, 0, 0, 0, 0, 1, 0, 0],
            update_fns=UPDATE_FNS,
            xapp_param_indices=[
                (0, 1), (0, 2, 1), (3, 4), (5, 4), (6,), (7,), (8,), (9,),
            ],
            xapp_kpi_indices=[(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,)],
            kpi_to_xapp={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7},
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
        # Latent confounder: ONE hidden draw Z shared by K6 and K7 -> correlated
        # observations with no edge between them and no param driving the link.
        if self.z_scale > 0 or self._forced_z is not None:
            z = self._forced_z if self._forced_z is not None else np.random.normal(0, self.z_scale)
            for kpi_index, gain in zip(CONFOUNDED_KPIS, Z_GAINS, strict=True):
                name = self.kpis[kpi_index].name
                state[name] = (state[name] + np.float32(gain * z)).astype(np.float32)
        if self.drift > 0:
            state["kpi0"] = (state["kpi0"] + np.float32(self.drift * self.cur_step)).astype(
                np.float32
            )
        return state


if __name__ == "__main__":
    env = ORANEnvironment4()
    assert env.num_params == 10 and env.num_kpis == 8, (env.num_params, env.num_kpis)

    state = env.reset()
    assert len(state) == env.get_state_dim() == 18, (len(state), env.get_state_dim())
    next_state, reward, done, info = env.step((0, 5, 0))
    assert len(next_state) == 18, len(next_state)
    assert isinstance(reward, float), type(reward)
    assert env.get_action_dim() == 3, env.get_action_dim()

    n = env.num_params + env.num_kpis
    expected = np.zeros((n, n), dtype=np.float32)
    for kpi_index, source in ADJACENCY_EDGES:
        expected[env.num_params + kpi_index, source] = 1
    assert np.array_equal(env.true_adj_matrix, expected), "adjacency mismatch"
    assert env.true_adj_matrix.sum() == len(ADJACENCY_EDGES), env.true_adj_matrix.sum()

    # The real KPI->KPI chain edges ARE present (K0->K4, K4->K5).
    assert env.true_adj_matrix[env.num_params + 4, env.num_params + 0] == 1  # K4 <- K0
    assert env.true_adj_matrix[env.num_params + 5, env.num_params + 4] == 1  # K5 <- K4
    # The confounded pair has NO edge in EITHER direction (K6 <-> K7).
    a = env.num_params + CONFOUNDED_KPIS[0]
    b = env.num_params + CONFOUNDED_KPIS[1]
    assert env.true_adj_matrix[a, b] == 0 and env.true_adj_matrix[b, a] == 0, "spurious Z edge"

    # Z influences its target KPIs: same seed + same params, only Z changes ->
    # K6 and K7 shift, a non-target KPI (K0) does not. noise_scale=0 isolates Z.
    envz = ORANEnvironment4(noise_scale=0.0)
    np.random.seed(1)
    envz.reset()
    np.random.seed(1)
    envz._forced_z = 0.0
    s_lo = envz.get_state()
    np.random.seed(1)
    envz._forced_z = 5.0
    s_hi = envz.get_state()
    assert float(s_lo["kpi6"][0]) != float(s_hi["kpi6"][0]), (s_lo["kpi6"], s_hi["kpi6"])
    assert float(s_lo["kpi7"][0]) != float(s_hi["kpi7"][0]), (s_lo["kpi7"], s_hi["kpi7"])
    assert float(s_lo["kpi0"][0]) == float(s_hi["kpi0"][0]), "Z leaked into a non-target KPI"

    def rollout():
        e = ORANEnvironment4()
        np.random.seed(7)
        e.reset()
        trace = []
        for _ in range(5):
            st, _, _, _ = e.step((0, 5, 0))
            trace.append([float(st[k.name][0]) for k in e.kpis])
        return trace

    assert rollout() == rollout(), "env not deterministic under a fixed seed with drift off"

    print(
        "env_iv self-check OK: dims 10NCP/8KPI, state=18, edges=", len(ADJACENCY_EDGES),
        "| real K-K chain kept, confounded K6-K7 has no edge",
    )
