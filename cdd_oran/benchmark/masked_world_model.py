"""Structure-masked E2 world model — the discovery→do-propagation→decision spine primitive.

This is the productized core of the live causal-conflict-mitigation pipeline: it turns a **discovered
structure** (a set of param→KPI edges) into a world model whose ``do()``-propagation only uses the
edges the discovery declared. Feeding that model through the frozen H=1 selection oracle
(``cdd_oran.analysis.v2_regret.score_grid`` / ``select_action``) yields the action a planner would take
*if it believed the discovered structure*; scoring that action under the TRUE simulator gives the
decision regret. A discovery method that recovers the harmful gated edge P0→K5 avoids the shared-knob
trap (regret 0); one that misses it walks into it (regret ≈ the decoy value).

Semantics (a strict generalization of ``E2V2Env(decoy_omit_p0_k5=True)``)
------------------------------------------------------------------------
A live edge set ``param_edges`` (a set of ``(kpi_index, param_index)`` pairs) says which of each KPI's
TRUE parents are "live". Inside KPI ``k``'s mechanism, parent param ``j`` uses the live
``prev_params[j]`` iff ``(k, j) ∈ param_edges``; otherwise the **committed reference** value (the param
vector captured at ``reset``/``restore``, exactly as the decoy freezes ``_decoy_p0_ref``) is substituted
for that parent — and only inside that KPI's term. Consequences:

- ``param_edges = TRUE_PARAM_EDGES`` reproduces the true env exactly (every parent live).
- ``param_edges = TRUE_PARAM_EDGES - {(5, 0)}`` reproduces ``decoy_omit_p0_k5=True`` exactly
  (only P0 inside the K5 term is frozen).
- **False edges are inert by construction**: a declared ``(k, j)`` whose true mechanism for KPI ``k``
  has no ``j`` term never changes KPI ``k`` — masking a non-parent to its committed value is a no-op
  because the true equation never reads it. So the decision is robust to discovery false-positives and
  fragile only to a MISSING true parent (the property the E2 shared-knob trap exploits).
- **KPI→KPI declarations are inert**: E2 KPIs are pure functions of params (``candidate >= num_params``
  are lagged KPIs), so they carry no param and cannot enter a mechanism. Callers should drop them
  upstream; ``param_edges_from_binary_mask`` does.

Drift safety: the six mechanism equations live in exactly ONE place — ``E2V2Env._update_kpis`` (frozen
by ``GATE_CONTRACT_E2.md``). This class never copies them. It computes each KPI by evaluating the TRUE
mechanism on a per-KPI masked param vector, so if the env's equations ever change, this model tracks
them automatically.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from cdd_oran.envs.v2.e2 import E2V2Env

# TRUE param→KPI edges, derived from the frozen adjacency so it can never drift from the env.
# ``_TRUE_ADJACENCY`` is a list of ``(kpi_index, source_index)``; a source below ``num_params`` is a
# param. Every E2 true edge is param→KPI (E2 has no true KPI→KPI edge), so this is all 16 edges.
TRUE_PARAM_EDGES: frozenset[tuple[int, int]] = frozenset(
    (int(kpi), int(src))
    for (kpi, src) in E2V2Env._TRUE_ADJACENCY
    if int(src) < E2V2Env.num_params
)


def param_edges_from_binary_mask(
    binary_mask, num_params: int = E2V2Env.num_params
) -> frozenset[tuple[int, int]]:
    """Decode a discovery ``binary_mask`` (KPI × candidate) into live param→KPI edges.

    ``binary_mask[kpi, candidate]`` is 1 where the method declared ``candidate`` a parent of KPI
    ``kpi``. Candidates ``0 .. num_params-1`` are params (kept); ``>= num_params`` are lagged KPIs
    (dropped — inert in this world model). Matches the candidate indexing of the e2slice recovery
    scorers (``evaluate_rcot`` / ``E2V2Env.true_adj_matrix``).
    """
    mask = np.asarray(binary_mask)
    if mask.ndim != 2:
        raise ValueError(f"binary_mask must be 2-D (KPI × candidate), got shape {mask.shape}")
    return frozenset(
        (kpi, cand)
        for kpi in range(mask.shape[0])
        for cand in range(mask.shape[1])
        if cand < num_params and mask[kpi, cand]
    )


class MaskedE2WorldModel(E2V2Env):
    """``E2V2Env`` whose per-parent live-vs-committed use is governed by a discovered edge set."""

    def __init__(
        self,
        param_edges: Iterable[tuple[int, int]],
        env_seed: int = 0,
        obs_noise_scale: float = 0.0,
        episode: int = 0,
    ) -> None:
        self.param_edges: frozenset[tuple[int, int]] = frozenset(
            (int(k), int(p)) for (k, p) in param_edges
        )
        self._committed_ref: np.ndarray | None = None
        # decoy_omit_p0_k5 stays False: the mask fully governs which parents are live, and the
        # per-KPI true-mechanism call below reads the live p[0] for the K5 term.
        super().__init__(env_seed=env_seed, obs_noise_scale=obs_noise_scale, episode=episode)

    def reset(self, episode: int | None = None):
        state = super().reset(episode=episode)
        self._committed_ref = self.prev_params.copy()
        return state

    def restore(self, snap: tuple) -> None:
        super().restore(snap)
        # Committed reference = the restored committed state's params; stays fixed through the H=1
        # rollout even after apply_action moves live params (mirrors the decoy's _decoy_p0_ref).
        self._committed_ref = self.prev_params.copy()

    def _masked_params_for_kpi(self, p: np.ndarray, kpi: int) -> np.ndarray:
        """``p`` with every param NOT a live parent of ``kpi`` replaced by the committed reference.

        Non-parents of ``kpi`` are masked too, but that is a no-op: KPI ``kpi``'s true equation never
        reads a param that is not its parent, so their value cannot affect the result.
        """
        ref = self._committed_ref if self._committed_ref is not None else p
        out = np.array(ref, dtype=float, copy=True)
        for j in range(len(p)):
            if (kpi, j) in self.param_edges:
                out[j] = p[j]
        return out

    def _update_kpis(self, prev_params: np.ndarray, prev_kpis: np.ndarray) -> np.ndarray:
        p = np.asarray(prev_params, dtype=float)
        # Evaluate the TRUE mechanism once per KPI on that KPI's masked param vector, keeping only
        # that KPI's value. The mechanisms are never copied here — E2V2Env owns them.
        out = np.empty(self.num_kpis, dtype=float)
        for kpi in range(self.num_kpis):
            p_k = self._masked_params_for_kpi(p, kpi)
            out[kpi] = E2V2Env._update_kpis(self, p_k, prev_kpis)[kpi]
        return out


def make_masked_factory(
    param_edges: Iterable[tuple[int, int]],
) -> Callable[[int], MaskedE2WorldModel]:
    """Return an ``env_factory(seed) -> MaskedE2WorldModel`` with a fixed live edge set."""
    edges = frozenset((int(k), int(p)) for (k, p) in param_edges)

    def factory(seed: int) -> MaskedE2WorldModel:
        return MaskedE2WorldModel(param_edges=edges, env_seed=seed)

    return factory


# --------------------------------------------------------------------------------------------------
# E5 masked world model (composed env; adds KPI→KPI chain edges and a latent confounder).
# --------------------------------------------------------------------------------------------------

def edges_from_binary_mask_e5(binary_mask, num_params: int):
    """Decode an E5 discovery ``binary_mask`` (KPI × candidate) into (param_edges, kpi_edges).

    Candidates ``0..num_params-1`` are params → ``param_edges`` as ``(kpi, param)``; candidates
    ``>= num_params`` are KPI sources → ``kpi_edges`` as ``(kpi, kpi_source)`` (the chain edges).
    """
    mask = np.asarray(binary_mask)
    param_edges = set()
    kpi_edges = set()
    for kpi in range(mask.shape[0]):
        for cand in range(mask.shape[1]):
            if not mask[kpi, cand]:
                continue
            if cand < num_params:
                param_edges.add((kpi, cand))
            else:
                kpi_edges.add((kpi, cand - num_params))
    return frozenset(param_edges), frozenset(kpi_edges)


class MaskedE5WorldModel:
    """Structure-masked E5 world model — the E5 analogue of ``MaskedE2WorldModel``.

    A discovered structure (``param_edges`` as ``(kpi, param)`` + ``kpi_edges`` as ``(kpi, kpi_source)``
    for the chain) governs which of each KPI's TRUE parents are live; non-declared parents are frozen
    at their committed reference (params) or committed previous KPI (chain), exactly as the E2 decoy
    freezes P0. The latent confounder ``Z`` is integrated to its mean (``E[Z]=0``) for planning — the
    do-propagation scoring uses the expected outcome, so the ``theta·Z`` term drops (it corrupts
    *observational discovery*, not the interventional decision). Mechanisms are never copied: each KPI
    is computed by the TRUE ``E5V2Env._update_kpis`` on a per-KPI masked (param, prev-KPI) pair with
    ``theta`` forced to 0, so the model tracks the frozen env automatically.

    This is not a ``V2Env`` subclass wrapper of arbitrary envs — it wraps ``E5V2Env`` directly (imported
    lazily to avoid a benchmark→env import cycle at module load).
    """

    def __init__(
        self,
        param_edges: Iterable[tuple[int, int]],
        kpi_edges: Iterable[tuple[int, int]] = (),
        env_seed: int = 0,
        obs_noise_scale: float = 0.0,
        *,
        subdom: float | None = None,
        chain_gamma: float | None = None,
    ) -> None:
        from cdd_oran.envs.v2.e5 import E5V2Env  # lazy: avoid import cycle

        self.param_edges = frozenset((int(k), int(p)) for (k, p) in param_edges)
        self.kpi_edges = frozenset((int(k), int(s)) for (k, s) in kpi_edges)
        # theta=0 integrates the latent Z out for planning (mean-KPI surrogate). subdom/chain_gamma
        # MUST match the TRUE env's layer config so the world model differs from truth ONLY by the
        # mask, not by the mechanism coefficients (sol finding #4 — corpus/spine/model consistency).
        kw = {}
        if subdom is not None:
            kw["subdom"] = float(subdom)
        if chain_gamma is not None:
            kw["chain_gamma"] = float(chain_gamma)
        self._env = E5V2Env(env_seed=env_seed, obs_noise_scale=obs_noise_scale, theta=0.0, **kw)
        self.num_params = self._env.num_params
        self.num_kpis = self._env.num_kpis
        self._committed_params: np.ndarray | None = None
        self._committed_kpis: np.ndarray | None = None

    # rollout interface (SEMANTICS §4): restore / apply_action / advance / snapshot
    def restore(self, snap: tuple) -> None:
        self._env.restore(snap)
        self._committed_params = self._env.prev_params.copy()
        self._committed_kpis = self._env.prev_kpis.copy()

    def snapshot(self):
        return self._env.snapshot()

    def apply_action(self, param_id: int, value: float) -> None:
        self._env.apply_action(param_id, value)

    def _masked_inputs(self, p: np.ndarray, kprev: np.ndarray, kpi: int):
        """Per-KPI: params frozen to committed except declared param-parents; prev-KPIs frozen except
        declared chain-parents."""
        ref_p = self._committed_params if self._committed_params is not None else p
        ref_k = self._committed_kpis if self._committed_kpis is not None else kprev
        p_k = np.array(ref_p, dtype=float, copy=True)
        k_k = np.array(ref_k, dtype=float, copy=True)
        for j in range(len(p)):
            if (kpi, j) in self.param_edges:
                p_k[j] = p[j]
        for s in range(len(kprev)):
            if (kpi, s) in self.kpi_edges:
                k_k[s] = kprev[s]
        return p_k, k_k

    def _update_masked(self, prev_params: np.ndarray, prev_kpis: np.ndarray) -> np.ndarray:
        from cdd_oran.envs.v2.e5 import E5V2Env  # lazy

        p = np.asarray(prev_params, dtype=float)
        kprev = np.asarray(prev_kpis, dtype=float)
        out = np.empty(self.num_kpis, dtype=float)
        for kpi in range(self.num_kpis):
            p_k, k_k = self._masked_inputs(p, kprev, kpi)
            out[kpi] = E5V2Env._update_kpis(self._env, p_k, k_k)[kpi]
        return out

    def advance(self) -> np.ndarray:
        env = self._env
        new_kpis = self._update_masked(env.prev_params, env.prev_kpis)
        env.prev_params = env.params.copy()
        env.prev_Z = env.Z
        env.prev_kpis = new_kpis
        env.time += 1
        env.Z = env._draw_Z(env.time)
        return env.prev_kpis.copy()
