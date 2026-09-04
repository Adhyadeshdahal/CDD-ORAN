"""Minimal v2 environment base.

Implements the SEMANTICS.md contract that the v2 suite needs, and *only* that:

- **Coordinate-keyed exogenous tape (SEMANTICS §4/§5).** Every exogenous value — initial
  params, per-step process noise, observation noise — is a PURE function of the coordinate
  ``(env_seed, episode, time, variable)`` and nothing else. A per-coordinate
  ``np.random.Generator`` is *derived* from a stable ``SeedSequence`` over those coordinates
  and consumed for that one draw; there is **no** long-lived sequential stream whose mutable
  draw order pairs arms. Consequences the contract requires and the tests assert:
    * repeated reads at one coordinate return the SAME value (stable tape);
    * two arms that take DIFFERENT actions still index the same coordinates and therefore
      receive BYTE-IDENTICAL exogenous draws, regardless of how many draws each makes;
    * distinct variables get independent draws;
    * no module-level ``np.random.*`` is ever touched, so the legacy global RNG stream is
      never perturbed.
- **apply/advance split with one-step latency (SEMANTICS §1.1).** ``apply_action`` writes a
  param coordinate into the *pending* param vector; ``advance`` computes the next latent KPI
  from the *previously committed* params/KPIs, commits the pending params, and advances the
  logical ``time`` coordinate. An action applied at step ``t`` first influences the KPI
  produced one ``advance`` later.
- **Noise-off knob + latent accessor (SPEC E1, SEMANTICS §2).** ``latent_kpis`` returns the
  raw noiseless KPI ``k_t``; ``observed_kpis`` returns ``k_t + eps`` with ``eps`` drawn from
  the observation tape only when ``obs_noise_scale > 0``. The latent state is never mutated
  by an observation. With the knob at ``0`` latent and observed outcomes are identical.

``snapshot()``/``restore()`` save/restore the committed + pending param/KPI state AND the
logical coordinates ``(episode, time)``. Because the tape is coordinate-keyed, replay after a
restore is exact without ever serializing a mutable generator's internal state.

Subclasses provide ``num_params``, ``num_kpis``, ``id_ranges`` and ``_update_kpis``.
"""

from __future__ import annotations

import numpy as np


class V2Env:
    num_params: int
    num_kpis: int
    id_ranges: list[tuple[float, float]]

    # Tape namespaces keep exogenous variable slots disjoint across purposes so an init draw
    # for param i can never alias the observation-noise draw for KPI i (§4 "own variable slot").
    _NS_INIT = 0  # initial parameters, keyed at time = 0
    _NS_PROC = 1  # per-step process/transition noise
    _NS_OBS = 2   # per-step observation noise eps_t

    def __init__(
        self, env_seed: int = 0, obs_noise_scale: float = 0.0, episode: int = 0
    ) -> None:
        self.env_seed = int(env_seed)
        self.obs_noise_scale = float(obs_noise_scale)
        self.episode = int(episode)
        self.time = 0
        self.params: np.ndarray
        self.prev_params: np.ndarray
        self.prev_kpis: np.ndarray
        self.reset()

    def _coord_rng(self, namespace: int, variable: int, time: int) -> np.random.Generator:
        """Derive the per-coordinate generator for ``(env_seed, episode, time, ns, variable)``.

        A fresh ``Generator`` is spawned from a ``SeedSequence`` over the (non-negative)
        coordinate tuple, so the value is a pure function of the coordinate — stable on
        repeated reads and independent of action history or draw order.
        """
        entropy = tuple(
            int(c) & 0xFFFFFFFFFFFFFFFF
            for c in (self.env_seed, self.episode, int(time), int(namespace), int(variable))
        )
        return np.random.default_rng(np.random.SeedSequence(entropy))

    def _process_noise(self, variable: int, time: int) -> float:
        """Standard-normal transition noise at ``(variable, time)`` from the tape.

        Provided for mechanisms that inject per-step process noise; E1 uses none. Subclasses
        call this from ``_update_kpis`` with the time of the KPI being produced.
        """
        return float(self._coord_rng(self._NS_PROC, variable, time).standard_normal())

    def _update_kpis(self, prev_params: np.ndarray, prev_kpis: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset(self, episode: int | None = None) -> dict[str, np.ndarray]:
        """Re-initialize at ``time = 0``; initial params come from the tape (§4)."""
        if episode is not None:
            self.episode = int(episode)
        self.time = 0
        self.params = np.array(
            [
                self._coord_rng(self._NS_INIT, i, 0).uniform(low, high)
                for i, (low, high) in enumerate(self.id_ranges)
            ],
            dtype=float,
        )
        self.prev_params = self.params.copy()
        self.prev_kpis = np.zeros(self.num_kpis, dtype=float)
        return self.latent_state()

    def apply_action(self, param_id: int, value: float) -> None:
        """Write one param coordinate into the pending param vector (no KPI advance)."""
        self.params[int(param_id)] = float(value)

    def advance(self) -> np.ndarray:
        """Roll one step: KPI from the *committed* prev state, commit pending params, tick time."""
        new_kpis = np.asarray(self._update_kpis(self.prev_params, self.prev_kpis), dtype=float)
        self.prev_params = self.params.copy()
        self.prev_kpis = new_kpis
        self.time += 1
        return self.prev_kpis.copy()

    def step(self, param_id: int, value: float) -> np.ndarray:
        """Legacy-equivalent composition ``apply_action(a); advance()``."""
        self.apply_action(param_id, value)
        return self.advance()

    def neutral_step(self) -> np.ndarray:
        """A no-op step: advance with no action applied (params held)."""
        return self.advance()

    def latent_kpis(self) -> np.ndarray:
        """Raw noiseless latent KPI vector ``k_t`` (SEMANTICS §2, PD3)."""
        return self.prev_kpis.copy()

    def observed_kpis(self) -> np.ndarray:
        """Observed KPI ``y_t = k_t + eps``; ``eps`` for KPI ``v`` is ``tape(seed, ep, t, v)``.

        The draw is keyed on the current ``time`` coordinate, so repeated reads at one state
        return identical noise, and arms at matching coordinates see byte-identical eps. The
        latent ``prev_kpis`` is never mutated.
        """
        kpis = self.prev_kpis.copy()
        if self.obs_noise_scale > 0.0:
            eps = np.array(
                [
                    self._coord_rng(self._NS_OBS, v, self.time).normal(0.0, self.obs_noise_scale)
                    for v in range(self.num_kpis)
                ],
                dtype=float,
            )
            kpis = kpis + eps
        return kpis

    def latent_state(self) -> dict[str, np.ndarray]:
        return {"params": self.prev_params.copy(), "kpis": self.prev_kpis.copy()}

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        """Save committed + pending state AND the logical coordinates (episode, time).

        No mutable generator state is serialized: the tape is coordinate-keyed, so restoring
        the coordinates is sufficient for exact replay (§4).
        """
        return (
            self.params.copy(),
            self.prev_params.copy(),
            self.prev_kpis.copy(),
            self.episode,
            self.time,
        )

    def restore(self, snap: tuple) -> None:
        """Restore state from a ``snapshot()``. A 3-tuple ``(params, prev_params, prev_kpis)``
        is also accepted for callers that rebuild only the arrays; the logical coordinates
        (episode/time) are then left unchanged."""
        params, prev_params, prev_kpis = snap[0], snap[1], snap[2]
        self.params = params.copy()
        self.prev_params = prev_params.copy()
        self.prev_kpis = prev_kpis.copy()
        if len(snap) >= 5:
            self.episode = int(snap[3])
            self.time = int(snap[4])
