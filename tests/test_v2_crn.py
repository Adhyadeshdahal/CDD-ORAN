"""Coordinate-keyed exogenous tape (CRN) tests for the v2 env base.

Covers the SEMANTICS §4/§5 contract (review findings 2 + 6):

- repeated reads at one coordinate are stable;
- arms with DIFFERENT action sequences get byte-identical exogenous draws at matching
  coordinates;
- ``noise_scale=0`` leaves latent == observed; ``noise_scale>0`` leaves latent unchanged and
  applies the declared scale;
- distinct variables draw independently;
- module-level NumPy state is untouched across a rollout;
- snapshot -> diverge -> restore reproduces the exact continuation.
"""

from __future__ import annotations

import numpy as np

from cdd_oran.envs.v2 import E1V2Env


def _make(noise: float = 0.0, seed: int = 7, episode: int = 0) -> E1V2Env:
    return E1V2Env(env_seed=seed, obs_noise_scale=noise, episode=episode)


# --- stable tape ------------------------------------------------------------

def test_repeated_observation_at_one_coordinate_is_identical():
    env = _make(noise=0.5)
    env.step(0, 0.3)
    env.step(1, 0.6)
    first = env.observed_kpis()
    for _ in range(5):
        assert np.array_equal(env.observed_kpis(), first)


def test_initial_params_are_coordinate_keyed_and_stable():
    a = _make(noise=0.0, seed=3, episode=2)
    b = _make(noise=0.0, seed=3, episode=2)
    assert np.array_equal(a.params, b.params)
    # different episode -> different init draw (own coordinate)
    c = _make(noise=0.0, seed=3, episode=5)
    assert not np.array_equal(a.params, c.params)


# --- cross-arm byte-identical exogenous draws (the BLOCKER acceptance) -------

def test_cross_arm_byte_identical_noise_under_different_actions():
    """Two arms that apply DIFFERENT actions still index the same (episode,time,var)
    coordinates and therefore receive byte-identical observation noise."""
    scale = 0.75
    arm_a = _make(noise=scale, seed=11)
    arm_b = _make(noise=scale, seed=11)

    def tape_eps(env: E1V2Env) -> np.ndarray:
        return np.array(
            [
                env._coord_rng(env._NS_OBS, v, env.time).normal(0.0, scale)
                for v in range(env.num_kpis)
            ]
        )

    # Arm A and arm B take entirely different action sequences and different call counts.
    for t in range(4):
        arm_a.step(0, 0.1 * t)
        arm_b.step(1, 0.9 - 0.1 * t)
        # arm B reads the tape extra times; that must not desynchronize anything.
        _ = arm_b.observed_kpis()

        assert arm_a.time == arm_b.time  # matching logical coordinate
        eps_a, eps_b = tape_eps(arm_a), tape_eps(arm_b)
        # the exogenous draws are byte-identical across arms despite different actions
        assert np.array_equal(eps_a, eps_b)
        # and each env actually applies exactly its coordinate's draw to its own latent state
        assert np.array_equal(arm_a.observed_kpis(), arm_a.latent_kpis() + eps_a)
        assert np.array_equal(arm_b.observed_kpis(), arm_b.latent_kpis() + eps_b)


# --- noise semantics --------------------------------------------------------

def test_noise_off_latent_equals_observed():
    env = _make(noise=0.0)
    env.step(0, 0.4)
    env.step(2, 0.8)
    assert np.array_equal(env.latent_kpis(), env.observed_kpis())


def test_noise_on_leaves_latent_unchanged_and_uses_declared_scale():
    scale = 0.6
    env = _make(noise=scale)
    env.step(0, 0.4)
    env.step(1, 0.7)
    latent_before = env.latent_kpis()
    # repeated noisy observations must never mutate the latent state
    for _ in range(10):
        obs = env.observed_kpis()
    assert np.array_equal(env.latent_kpis(), latent_before)

    # the applied eps must be exactly the tape draw at this coordinate and scale
    expected = np.array(
        [
            env._coord_rng(env._NS_OBS, v, env.time).normal(0.0, scale)
            for v in range(env.num_kpis)
        ]
    )
    assert np.array_equal(obs, latent_before + expected)
    # non-degenerate: a positive scale actually perturbs the observation
    assert np.any(expected != 0.0)


def test_distinct_variables_draw_independently():
    env = _make(noise=0.5)
    env.step(0, 0.5)
    eps = env.observed_kpis() - env.latent_kpis()
    # four KPIs, four independent variable slots -> not all equal
    assert len(set(np.round(eps, 12))) > 1


# --- global RNG isolation ---------------------------------------------------

def test_module_level_numpy_state_unchanged_across_rollout():
    before = np.random.get_state()
    env = _make(noise=0.5)
    for t in range(6):
        env.step(t % env.num_params, 0.1 * t)
        env.observed_kpis()
    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


# --- snapshot / diverge / restore -------------------------------------------

def test_snapshot_diverge_restore_reproduces_continuation():
    scale = 0.4
    env = _make(noise=scale, seed=21)
    env.step(0, 0.2)
    env.step(1, 0.5)
    snap = env.snapshot()

    # canonical continuation from the snapshot
    ref = _make(noise=scale, seed=21)
    ref.restore(snap)
    ref.step(2, 0.3)
    ref.step(3, 0.9)
    ref_latent = ref.latent_kpis()
    ref_obs = ref.observed_kpis()

    # diverge on the ORIGINAL env with different actions and extra tape reads
    env.step(2, 0.99)
    env.observed_kpis()
    env.step(3, 0.01)
    env.observed_kpis()

    # restore and replay the canonical continuation -> byte-identical outcome
    env.restore(snap)
    env.step(2, 0.3)
    env.step(3, 0.9)
    assert np.array_equal(env.latent_kpis(), ref_latent)
    assert np.array_equal(env.observed_kpis(), ref_obs)
