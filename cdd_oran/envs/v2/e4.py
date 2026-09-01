"""E4 — Confounded decision / action-relevant confounding (v2 SCM).

Implements the FROZEN E4 SCM from ``docs/benchmark/GATE_CONTRACT_E4.md`` (reviewer ruling
``.herdr/reports/rule-e4-design.md``, "Frozen E4 Contract Value Block") VERBATIM. Nothing here
is invented or tuned: the single-outcome linear-Gaussian mechanism, the latent confounder ``Z``,
the clipped behavior policy, the observational/interventional protocol, and the exact
standardization are all transcribed from the frozen contract.

This is the **minimal single-outcome** confounded SCM — deliberately NOT the legacy Env IV
backbone (`cdd_oran/envs/env_iv.py`, which puts ``Z`` only in observations and adds irrelevant
graph depth/KPIs; ruling Q6). One acted param ``A = P0``, one KPI ``K_out = K0``::

    K_out(q+1) = alpha*A_q + theta*Z_q + c        # one-step lag (SEMANTICS PD2)
    alpha = -1.0,  theta = +2.5,  c = 0.0

The graph carries three edges: ``Z -> A_behavior`` (the confounder biases data-collection
actions), ``Z -> K_out`` (the confounder moves the outcome), and ``A -> K_out`` (the true causal
effect). ``Z`` is **latent**: neither a parameter nor a KPI, never exposed to a planner. The
observed env adjacency therefore contains only ``(K0, P0) = (0, 0)``; the two latent edges are
mandatory SCM metadata (``LATENT_EDGES``) and gate assertions.

Two data-collection modes (ruling Q7, "Behavior and matched obs/do protocol"):

* **OBSERVATIONAL** (``mode="obs"``): the action is the clipped behavior policy
  ``A_behavior = clip(0.5 + lambda*Z + eta, 0, 1)`` — the SAME ``Z`` that then enters ``K_out``,
  inducing the spurious action--outcome association. ``eta ~ Normal(0, 0.5^2)``.
* **INTERVENTIONAL** (``mode="do"``): the acted coordinate is set by ``do(A=v)`` with assignment
  **independent** of ``Z`` (either an externally applied ``apply_action`` or a randomized do-grid
  draw). ``Z`` still enters ``K_out``.

``Z`` is drawn from a DEDICATED coordinate-keyed tape slot (``_NS_LATENT``) that does not alias
the behavior-noise (``_NS_ETA``), randomized do-action (``_NS_DOACT``), process-noise
(``_NS_PROC``), or observation-noise (``_NS_OBS``) slots. Pending (``Z``) and committed
(``prev_Z``) latents are both carried in ``snapshot``/``restore`` so gate rollouts replay exactly.

``lambda`` (behavior coupling), ``alpha``, ``theta`` are constructor args so the structural gate
can build the ``lambda=0`` factor-removal control (removes ``Z -> A_behavior`` only) and the
deferred positive sweep ``{0, 0.5, 1.0, 1.5}``. Defaults reproduce the frozen primary SCM exactly.

Authoritative scoring (ruling Q10, "Exact standardization and objective") integrates ``Z`` to its
mean, giving the latent-noiseless interventional reduction ``K_score(a) = E_Z[K_out|do(A=a)] =
alpha*a + theta*E[Z] + c = -a`` (``E[Z] = 0``), exposed for the gate via
``score_interventional_mean``.
"""

from __future__ import annotations

import numpy as np

from cdd_oran.envs.v2.base import V2Env

# FROZEN standardization (ruling "Exact standardization and objective"). Evaluated at full float
# precision — NOT six-decimal approximations. K_score(a) = -a is Uniform[0,1]-referenced: mean
# -0.5, variance 1/12.
MU_OUT = -0.5
SIGMA_OUT = float(np.sqrt(1.0 / 12.0))  # 0.28867513459481287


class E4V2Env(V2Env):
    num_params = 1  # P0 = A
    num_kpis = 1    # K0 = K_out
    id_ranges = [(0.0, 1.0)]

    # FROZEN standardization of the interventional-mean outcome K_score(a) = -a over A~U[0,1].
    mu = np.array([MU_OUT], dtype=float)
    sigma = np.array([SIGMA_OUT], dtype=float)

    # Observed env adjacency as (kpi_index, source_index): ONLY the observed A -> K_out edge.
    # Z is latent, so Z-incident edges are NOT part of the observed adjacency (ruling "SCM and
    # timeline"). Kept as an instance-independent class constant.
    TRUE_ADJACENCY = [(0, 0)]  # (K0, P0)
    # Mandatory latent-edge SCM metadata / gate assertions (ruling Q7): the two Z-incident edges.
    LATENT_EDGES = [("Z", "A_behavior"), ("Z", "K_out")]

    # Objective constants (single-KPI panel; ruling "Exact standardization and objective").
    kpi_thresholds = (0.0,)     # threshold_raw = 0.0
    directions = (0,)           # satisfy-above
    xapp_kpi_indices = [(0,)]
    kpi_to_xapp = {0: 0}
    xapp_param_indices = [(0,)]
    panel_kpi_ids = (0,)

    # Data-collection modes.
    MODE_OBS = "obs"
    MODE_DO = "do"

    # Dedicated, NON-ALIASING tape namespaces (base uses 0=init, 1=proc, 2=obs). Each exogenous
    # purpose owns its own slot so a latent draw can never collide with behavior-noise or the
    # randomized do-action (ruling "Behavior and matched obs/do protocol").
    _NS_LATENT = 3  # Z_q
    _NS_ETA = 4     # eta_q (behavior-policy noise)
    _NS_DOACT = 5   # randomized do-action (interventional mode)

    # Randomized do-grid V = {0, 0.01, ..., 1.00} (101 inclusive points).
    _DO_GRID_N = 101

    def __init__(
        self,
        env_seed: int = 0,
        obs_noise_scale: float = 0.0,
        lam: float = 1.0,
        alpha: float = -1.0,
        theta: float = 2.5,
        c: float = 0.0,
        eta_scale: float = 0.5,
        mode: str = "do",
        episode: int = 0,
    ) -> None:
        # lam = lambda (Z -> A_behavior coupling; keyword avoids the Python reserved word).
        self.lam = float(lam)
        self.alpha = float(alpha)
        self.theta = float(theta)
        self.c = float(c)
        self.eta_scale = float(eta_scale)
        if mode not in (self.MODE_OBS, self.MODE_DO):
            raise ValueError(f"mode must be {self.MODE_OBS!r} or {self.MODE_DO!r}, got {mode!r}")
        self.mode = mode
        # Pending / committed latent Z (parallel to params / prev_params).
        self.Z: float
        self.prev_Z: float
        super().__init__(env_seed=env_seed, obs_noise_scale=obs_noise_scale, episode=episode)

    # --- dedicated exogenous draws (non-aliasing tape slots) ------------------
    def _draw_Z(self, time: int) -> float:
        """Latent ``Z_q ~ Normal(0,1)`` from the dedicated latent slot at coordinate ``time``."""
        return float(self._coord_rng(self._NS_LATENT, 0, int(time)).standard_normal())

    def _draw_eta(self, time: int) -> float:
        """Behavior-policy noise ``eta_q ~ Normal(0, eta_scale^2)`` from the dedicated eta slot."""
        return float(self._coord_rng(self._NS_ETA, 0, int(time)).standard_normal()) * self.eta_scale

    def _draw_do_action(self, time: int) -> float:
        """Randomized ``do(A)`` sampled uniformly over V, INDEPENDENT of ``Z`` (its own slot)."""
        idx = int(self._coord_rng(self._NS_DOACT, 0, int(time)).integers(0, self._DO_GRID_N))
        return idx / (self._DO_GRID_N - 1)

    # --- lifecycle (extends base with the latent Z bookkeeping) ---------------
    def reset(self, episode: int | None = None) -> dict[str, np.ndarray]:
        super().reset(episode)  # time=0, params=init, prev_params=init, prev_kpis=zeros
        # Z for the current coordinate (time 0); prev_Z pairs with the committed prev_params.
        self.Z = self._draw_Z(self.time)
        self.prev_Z = self.Z
        return self.latent_state()

    def behavior_action(self) -> float:
        """The clipped behavior policy at the CURRENT pending coordinate.

        ``A_behavior = clip(0.5 + lambda*Z + eta, 0, 1)`` using the current pending ``Z`` and the
        eta draw for the current time — the SAME ``Z`` that will enter ``K_out``.
        """
        eta = self._draw_eta(self.time)
        return float(np.clip(0.5 + self.lam * self.Z + eta, 0.0, 1.0))

    def generate_action(self) -> float:
        """The mode-appropriate action for the current coordinate.

        OBS: the ``Z``-confounded behavior policy. DO: a randomized do-grid draw independent of
        ``Z``. Corpus generation applies this via ``apply_action`` before ``advance`` (deferred).
        """
        if self.mode == self.MODE_OBS:
            return self.behavior_action()
        return self._draw_do_action(self.time)

    # --- SEMANTICS §1.1 apply/advance split (Z committed alongside params) ----
    def _update_kpis(self, prev_params: np.ndarray, prev_kpis: np.ndarray) -> np.ndarray:
        """``K_out = alpha*A_prev + theta*Z_prev + c`` on the COMMITTED (prev) action and latent.

        ``self.prev_Z`` is the latent committed alongside ``prev_params`` at the previous advance,
        so the delayed outcome uses the SAME ``Z_q`` that was paired with the action ``A_q`` —
        under obs (``A = A_behavior``) and do (``A`` independent of ``Z``) alike.
        """
        p = np.asarray(prev_params, dtype=float)
        k_out = self.alpha * p[0] + self.theta * self.prev_Z + self.c
        return np.array([k_out], dtype=float)

    def advance(self) -> np.ndarray:
        """Roll one step: outcome from committed (prev_params, prev_Z), then commit Z with params.

        Mirrors ``V2Env.advance`` but commits the pending latent ``self.Z -> self.prev_Z`` at the
        same instant the pending params commit, and draws a fresh ``Z`` for the new coordinate.
        """
        new_kpis = np.asarray(self._update_kpis(self.prev_params, self.prev_kpis), dtype=float)
        self.prev_params = self.params.copy()
        self.prev_Z = self.Z            # commit the current latent alongside the acted params
        self.prev_kpis = new_kpis
        self.time += 1
        self.Z = self._draw_Z(self.time)  # dedicated-slot draw for the new coordinate
        return self.prev_kpis.copy()

    # --- interventional-mean scoring (ruling Q10, choice B) -------------------
    def score_interventional_mean(self, a: float) -> float:
        """Latent-noiseless interventional-mean outcome ``E_Z[K_out | do(A=a)] = alpha*a + c``.

        ``Z`` is integrated to its analytic mean ``E[Z] = 0`` (the frozen reduction ``K_score = -a``
        at the default coefficients). Independent of ``lambda`` — the true causal effect of ``A``
        does not change when ``Z -> A_behavior`` is removed, which is exactly why the lambda=0
        control keeps the same scoring while the naive pooled fit collapses to ``-a``.
        """
        return self.alpha * float(a) + self.theta * 0.0 + self.c

    # --- deterministic snapshot/restore (carry pending + committed Z) ---------
    def snapshot(self) -> tuple:
        base = super().snapshot()  # (params, prev_params, prev_kpis, episode, time)
        return (*base, float(self.Z), float(self.prev_Z))

    def restore(self, snap: tuple) -> None:
        super().restore(snap)  # params/prev_params/prev_kpis + (episode,time) when present
        if len(snap) >= 7:
            self.Z = float(snap[5])
            self.prev_Z = float(snap[6])

    # --- adjacency ------------------------------------------------------------
    @property
    def adjacency_edges(self) -> list[tuple[int, int]]:
        """Observed structural edges: the single ``A -> K_out`` edge (Z is latent)."""
        return [(0, 0)]

    def true_adj_matrix(self) -> np.ndarray:
        size = self.num_params + self.num_kpis
        matrix = np.zeros((size, size), dtype=np.float32)
        for kpi_index, source_index in self.adjacency_edges:
            matrix[self.num_params + kpi_index, source_index] = 1.0
        return matrix
