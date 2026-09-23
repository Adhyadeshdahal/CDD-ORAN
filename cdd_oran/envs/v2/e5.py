"""E5 — Composed shared-knob conflict (v2 SCM). DEV / UNFROZEN (plan 015).

The centerpiece benchmark: a shared knob whose harmful edge is invisible to variance-based discovery
(SHAP-DAG / GNN) yet decision-critical, composed with the E2 fan-out, an E3-style chain, an E4-style
latent confounder, and noise. Only stratified-CI discovery → do-propagation → action-selection avoids
the trap; SHAP/GNN miss the subdominant edge, RCoT/pdCor miss the gated edge, and a marginal
correlational effect estimate fit on the OBSERVATIONAL corpus gets the confounded (sign-reversed) P0→K_harm
relationship.

Confound sign (E4 pattern; sol review #6, 2026-09-23). The true gated effect ``subdom·(P0−C0)`` is
POSITIVE (high P0 tips K_harm). For the latent confounder to MASK that association rather than reinforce
it, the confound path product must be NEGATIVE: ``Z → P0_behavior`` is positive (``lam>0``, natural) and
``Z → K_harm`` is NEGATIVE (``theta<0``, default ``theta=-2.5``). The confound-layer demonstration
(``scripts/e5_confound.py``) is the OBS-vs-DO contrast of an ORACLE-GATE IN-STRATUM OLS estimator: on the
observational corpus its in-gate slope is confound-BIASED (~+0.117 vs true +0.20 → the planner walks into
the trap); on the randomized ``do(P0) ⟂ Z`` corpus it IDENTIFIES the true gated effect (~+0.201). The
winning arm REQUIRES interventional data — this is the value of intervention capability under a
fully-latent confounder, not a superiority claim over observational methods. HONESTY (sol lanes): a
MARGINAL pooled OLS slope sign-reverses on OBS (−0.073) but is already decision-broken on DO by gate
dilution (+0.011), so its confound contribution does not add DECISION contrast; a covariate-aware GBDT
fit ATTENUATES (~50%) and is decision-broken even on DO (intrinsic shrinkage). Do NOT generalize to
"correlational methods" as a category, and do NOT read those arms' OBS-DO gaps as confounding wins.

This module is NOT yet frozen (no GATE_CONTRACT_E5.md). Constants below are the DESIGN values from the
validated probes (`scratchpad/e5_design/`); standardization moments are computed, not yet frozen. Every
stressor is an individually-toggleable LAYER (plan 015 §6.1, "incremental within composed"):

  layer      | constructor knob        | off value        | defeats
  -----------|-------------------------|------------------|--------------------------------
  core-gated | subdom                  | 0.0              | SHAP-DAG / GNN (subdominance) + RCoT/pdCor (gating)
  chain      | chain_gamma             | 0.0              | direct-edge-only discovery (2-hop path)
  confound   | theta (+ mode, lam)     | 0.0 / mode='do'  | observational (non-deconfounding) fits
  noise      | obs_noise_scale, proc   | 0.0 / False      | noise-fragile discovery+decision

Default constructor = the FULLY COMPOSED env (all layers on, mode='do' for the identifiable corpus).

Mechanisms (one-step lag, ``k_t = f(p_{t-1}, k_{t-1})``; ``K_mid`` is the previous-step conduit):

    K_ben  = A_ben  * exp(-((P0 - MU_BEN)^2) / (2 W_BEN^2))                      # benign local pull
    K_mid  = C_CHAIN * P0                                                        # chain conduit (excluded)
    K_harm = base(G1,G2)                                                         # DOMINANT benign parent
             + subdom      * (P0 - C0)      * gate(G1,G2)                        # subdominant gated DIRECT edge
             + chain_gamma * K_mid_prev     * gate(G1,G2)                        # gated 2-hop chain edge
             + theta       * Z_prev                                             # latent confounder
    K_dist = A_DIST * exp(-((P3 - MU_DIST)^2) / (2 W_DIST^2))                    # benign distractor

    base(G1,G2) = A_BASE * exp(-((G2 - MU_BASE)^2) / (2 W_BASE^2)) + B1 * G1     # dominant, P0-free
    gate(G1,G2) = 1[ G1 <= TAU1  AND  |G2 - C2| <= W_GATE ]                      # thin operating band

Panel = {K_ben (satisfy-above), K_harm (satisfy-below), K_dist (satisfy-above)}; K_mid is an excluded
conduit. Confounder Z is latent (E4 pattern): Z -> P0_behavior (obs mode only) and Z -> K_harm.
"""

from __future__ import annotations

from math import exp

import numpy as np

from cdd_oran.envs.v2.base import V2Env


def _gate(g1: float, g2: float, tau1: float, c2: float, w_gate: float) -> bool:
    return (g1 <= tau1) and (abs(g2 - c2) <= w_gate)


class E5V2Env(V2Env):
    num_params = 4  # [P0 (shared knob), G1 (gate axis), G2 (gate axis), P3 (distractor)]
    num_kpis = 4    # [K_ben, K_harm, K_mid (conduit), K_dist]

    # Param index roles.
    P0, G1, G2, P3 = 0, 1, 2, 3
    # KPI index roles.
    K_BEN, K_HARM, K_MID, K_DIST = 0, 1, 2, 3

    id_ranges = [
        (-100.0, 100.0),  # P0 shared knob (wide, randomized in the do-corpus)
        (1.5, 4.0),       # G1 gate axis (E2 P6-like; TAU1 ~ median splits ~half)
        (-100.0, 150.0),  # G2 gate axis (E2 P7-like; |G2-C2|<=W_GATE sets occupancy)
        (-20.0, 20.0),    # P3 distractor
    ]

    # --- DESIGN constants (validated probes; NOT frozen) --------------------------------------
    # Benign pull: bump peaked in the harmful P0 zone, peak below its satisfy-above threshold (55) so
    # it is unsatisfiable → the planner chases the peak (into the gate). Small amplitude keeps the
    # reward's P0-variation (D(s)) small so the K_harm satisfied-count flip is a large fraction of it →
    # a clearly non-trivial missed-edge regret. A_BEN does NOT enter K_harm, so SHAP-invisibility holds.
    A_BEN, MU_BEN, W_BEN = 12.0, 80.0, 40.0
    # Chain conduit coefficient (K_mid = C_CHAIN * P0).
    C_CHAIN = 1.0
    # Dominant benign parent of K_harm (large amplitude -> subdominance of the P0 term). Centered on
    # the gate (MU_BASE = C2) and amplitude tuned so in-gate base sits JUST BELOW the K_harm threshold
    # (80), so the subdominant P0 term tips a fraction of in-gate states over it (decision-critical).
    # A_BASE so in-gate base_max (A_BASE + B1*G1_max = 72 + 5.5 = 77.5) stays < K_harm threshold 80 —
    # the subdominant P0 term, not base alone, tips K_harm over. W_BASE=25 keeps base HIGH-VARIANCE over
    # the full G2 range (a strong bump) so it DOMINATES P0's SHAP importance (subdominance); the gate is
    # then aligned to the near-threshold sub-band where the edge is decision-critical.
    A_BASE, MU_BASE, W_BASE, B1 = 72.0, 25.0, 25.0, 2.0
    # Gate = the near-threshold operating band. W_GATE=12.6 gives occupancy
    # P(G1<=TAU1)*P(|G2-C2|<=W_GATE) = 0.5 * (25.2/250) ≈ 0.05. HONEST framing (sol #5): the harmful
    # edge is BOTH RARE (~5% occupancy, aligned with where base is near threshold) AND amplitude-
    # SUBDOMINANT. Both suppress its pooled variance footprint; see the occupancy×amplitude sweep. Do
    # NOT claim occupancy 0.2 or attribute the discovery miss to amplitude alone.
    TAU1, C2, W_GATE = 2.75, 25.0, 12.6
    C0 = 0.0
    # Distractor (satisfiable: peak 70 >= threshold 55 near P3=0).
    A_DIST, MU_DIST, W_DIST = 70.0, 0.0, 10.0
    # Confounder behavior coupling default (obs mode).
    LAM_DEFAULT = 1.0
    ETA_SCALE = 0.5

    # Panel thresholds / directions (satisfy-above=0, satisfy-below=1).
    kpi_thresholds = (55.0, 80.0, 0.0, 55.0)   # K_ben>=55, K_harm<=80, (K_mid n/a), K_dist>=55
    directions = (0, 1, 0, 0)
    xapp_kpi_indices = [(0,), (1,), (2,), (3,)]
    kpi_to_xapp = {0: 0, 1: 1, 2: 2, 3: 3}
    panel_kpi_ids = (0, 1, 3)     # K_ben, K_harm, K_dist
    excluded_kpi_ids = (2,)       # K_mid conduit

    MODE_OBS = "obs"
    MODE_DO = "do"

    # Dedicated non-aliasing tape slots (base: 0=init,1=proc,2=obs), E4 pattern.
    _NS_LATENT = 3   # Z
    _NS_ETA = 4      # behavior-policy noise
    _NS_DOACT = 5    # randomized do(P0)
    _DO_GRID = np.linspace(-100.0, 100.0, 201)

    def __init__(
        self,
        env_seed: int = 0,
        obs_noise_scale: float = 0.0,
        *,
        subdom: float = 0.20,        # core gated-subdominant DIRECT edge (0 -> off), POSITIVE
        chain_gamma: float = 0.20,   # gated 2-hop chain edge (0 -> off)
        theta: float = -2.5,         # latent confounder Z -> K_harm; NEGATIVE so it MASKS the +subdom
                                     # association (E4 pattern, sol #6). lam>0, theta<0 => confound
                                     # bias < 0 opposes true +0.20 -> marginal OBS fit sign-reverses.
        lam: float = LAM_DEFAULT,    # Z -> P0_behavior coupling (obs mode)
        mode: str = MODE_DO,         # 'do' = identifiable corpus; 'obs' = Z-confounded behavior
        process_noise: bool = False,
        episode: int = 0,
    ) -> None:
        self.subdom = float(subdom)
        self.chain_gamma = float(chain_gamma)
        self.theta = float(theta)
        self.lam = float(lam)
        if mode not in (self.MODE_OBS, self.MODE_DO):
            raise ValueError(f"mode must be {self.MODE_OBS!r} or {self.MODE_DO!r}, got {mode!r}")
        self.mode = mode
        self.process_noise = bool(process_noise)
        self.Z: float
        self.prev_Z: float
        super().__init__(env_seed=env_seed, obs_noise_scale=obs_noise_scale, episode=episode)

    # --- latent confounder tape (E4 pattern) --------------------------------------------------
    def _draw_Z(self, time: int) -> float:
        return float(self._coord_rng(self._NS_LATENT, 0, int(time)).standard_normal())

    def _draw_eta(self, time: int) -> float:
        return float(self._coord_rng(self._NS_ETA, 0, int(time)).standard_normal()) * self.ETA_SCALE

    def _draw_do_p0(self, time: int) -> float:
        idx = int(self._coord_rng(self._NS_DOACT, 0, int(time)).integers(0, len(self._DO_GRID)))
        return float(self._DO_GRID[idx])

    def reset(self, episode: int | None = None):
        state = super().reset(episode=episode)
        self.Z = self._draw_Z(self.time)
        self.prev_Z = self.Z
        return state

    def behavior_p0(self) -> float:
        """Z-confounded data-collection P0 (obs mode): 0 + lam*scale*Z + eta, clipped to range."""
        eta = self._draw_eta(self.time)
        raw = 0.0 + self.lam * 30.0 * self.Z + eta   # 30 = P0-scale so Z meaningfully shifts P0
        lo, hi = self.id_ranges[self.P0]
        return float(np.clip(raw, lo, hi))

    def generate_p0(self) -> float:
        """Corpus P0 for the current coordinate: Z-confounded (obs) or randomized do(P0) (do)."""
        return self.behavior_p0() if self.mode == self.MODE_OBS else self._draw_do_p0(self.time)

    # --- mechanism ----------------------------------------------------------------------------
    def _base(self, g1: float, g2: float) -> float:
        return self.A_BASE * exp(-((g2 - self.MU_BASE) ** 2) / (2.0 * self.W_BASE ** 2)) + self.B1 * g1

    def _update_kpis(self, prev_params: np.ndarray, prev_kpis: np.ndarray) -> np.ndarray:
        p = np.asarray(prev_params, dtype=float)
        k = np.asarray(prev_kpis, dtype=float)
        p0, g1, g2, p3 = p[self.P0], p[self.G1], p[self.G2], p[self.P3]
        in_gate = _gate(g1, g2, self.TAU1, self.C2, self.W_GATE)

        k_ben = self.A_BEN * exp(-((p0 - self.MU_BEN) ** 2) / (2.0 * self.W_BEN ** 2))
        k_mid = self.C_CHAIN * p0
        k_harm = self._base(g1, g2)
        if in_gate:
            k_harm += self.subdom * (p0 - self.C0)            # core gated-subdominant direct edge
            k_harm += self.chain_gamma * k[self.K_MID]        # gated 2-hop chain (prev K_mid)
        k_harm += self.theta * self.prev_Z                    # latent confounder
        k_dist = self.A_DIST * exp(-((p3 - self.MU_DIST) ** 2) / (2.0 * self.W_DIST ** 2))

        out = np.array([k_ben, k_harm, k_mid, k_dist], dtype=float)
        if self.process_noise:
            out = out + np.array(
                [self._process_noise(v, self.time) for v in range(self.num_kpis)], dtype=float
            )
        return out

    def advance(self) -> np.ndarray:
        """Roll one step; commit the latent Z alongside params (E4 pattern)."""
        new_kpis = np.asarray(self._update_kpis(self.prev_params, self.prev_kpis), dtype=float)
        self.prev_params = self.params.copy()
        self.prev_Z = self.Z
        self.prev_kpis = new_kpis
        self.time += 1
        self.Z = self._draw_Z(self.time)
        return self.prev_kpis.copy()

    def snapshot(self) -> tuple:
        base = super().snapshot()
        return (*base, float(self.Z), float(self.prev_Z))

    def restore(self, snap: tuple) -> None:
        super().restore(snap)
        if len(snap) >= 7:
            self.Z = float(snap[5])
            self.prev_Z = float(snap[6])

    # --- structure ----------------------------------------------------------------------------
    @property
    def adjacency_edges(self) -> list[tuple[int, int]]:
        """Active (kpi_index, source_index) edges; source>=num_params encodes a KPI source.

        Source index for a KPI j is num_params + j. Layer edges toggle with their coefficient.
        """
        np_ = self.num_params
        edges = [
            (self.K_BEN, self.P0),           # P0 -> K_ben
            (self.K_MID, self.P0),           # P0 -> K_mid (chain conduit)
            (self.K_HARM, self.G1),          # G1 -> K_harm (base)
            (self.K_HARM, self.G2),          # G2 -> K_harm (base)
            (self.K_DIST, self.P3),          # P3 -> K_dist
        ]
        if self.subdom != 0.0:
            edges.append((self.K_HARM, self.P0))            # subdominant gated direct edge
        if self.chain_gamma != 0.0:
            edges.append((self.K_HARM, np_ + self.K_MID))   # K_mid -> K_harm (chain)
        return edges

    # Latent-edge SCM metadata (Z is not a param/KPI); present only when the confound layer is on.
    @property
    def latent_edges(self) -> list[tuple[str, str]]:
        if self.theta == 0.0:
            return []
        e = [("Z", "K_harm")]
        if self.mode == self.MODE_OBS and self.lam != 0.0:
            e.append(("Z", "P0_behavior"))
        return e

    def true_adj_matrix(self) -> np.ndarray:
        size = self.num_params + self.num_kpis
        matrix = np.zeros((size, size), dtype=np.float32)
        for kpi_index, source_index in self.adjacency_edges:
            matrix[self.num_params + kpi_index, source_index] = 1.0
        return matrix
