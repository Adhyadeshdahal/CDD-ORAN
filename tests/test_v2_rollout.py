"""Characterization tests for the open-loop episodic rollout kernel (Plan 005).

Anchors the kernel against ``docs/benchmark/SEMANTICS.md``:

- Step 1: exact H=1 (§1.4) and H=2 (§1.5) worked-example timelines, scored window, and returns.
- Step 2: exhaustive oracle optimum, lexicographic tie-break, and the ``max_sequences`` guard
  firing before any rollout.
- Step 3: paired cumulative regret (=1 for the H=1 example, =2 for the H=2 planner), unclamped.
- Step 4: CRN pairing under DIVERGENT actions (byte-identical exogenous tape draws) and caller
  env/snapshot non-mutation.
- Step 5: cross-check the shared kernel's return against the existing E2/E3 gate-local rollout
  arithmetic (no gate migration).

The kernel bakes in no environment; tiny local SCMs reproduce the semantics arithmetic exactly.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import cast

import numpy as np
import pytest

import cdd_oran.benchmark.rollout as rollout_mod
from cdd_oran.benchmark.rollout import (
    OracleResult,
    enumerate_open_loop,
    paired_regret,
    rollout_open_loop,
)
from cdd_oran.envs.v2.base import V2Env

# --- minimal worked-example SCMs (SEMANTICS §1.4 / §1.5) ------------------------------------


class _H1Env(V2Env):
    """§1.4: ``k_t = g(p_{t-1})``, ``g(p) = 1 − (p − 2)²``; reset ``p_0 = 0``, ``k_0 = 0``."""

    num_params = 1
    num_kpis = 1
    id_ranges = [(0.0, 0.0)]  # uniform(0, 0) → p_0 = 0 deterministically

    def _update_kpis(self, prev_params, prev_kpis):
        p = float(prev_params[0])
        return np.array([1.0 - (p - 2.0) ** 2], dtype=float)


class _H2Env(V2Env):
    """§1.5 chain: ``k^A_t = p_{t-1}``, ``k^B_t = p_{t-1} − 2·k^A_{t-1}``; reset all zero."""

    num_params = 1
    num_kpis = 2
    id_ranges = [(0.0, 0.0)]

    def _update_kpis(self, prev_params, prev_kpis):
        p = float(prev_params[0])
        ka_prev = float(prev_kpis[0])
        return np.array([p, p - 2.0 * ka_prev], dtype=float)


class _ConstEnv(V2Env):
    """A constant mechanism: every action yields the same return (forces an all-ties oracle)."""

    num_params = 1
    num_kpis = 1
    id_ranges = [(0.0, 0.0)]

    def _update_kpis(self, prev_params, prev_kpis):
        return np.array([7.0], dtype=float)


def _r_k0(k):
    return float(k[0])  # §1.4 R(k) = k


def _r_kb(k):
    return float(k[1])  # §1.5 R(k) = k^B (second KPI)


def test_h1_timeline_scored_window_and_warmup():
    """§1.4: warm-up k_1 = g(0) = −3 (unscored); single scored k_2 = g(a_1); return = k_2."""
    env = _H1Env()
    snap = env.snapshot()

    # planner a_1 = 1 → k_2 = 1 − (1−2)² = 0
    res = rollout_open_loop(lambda: _H1Env(), snap, [(0, 1.0)], _r_k0)
    assert res.warmup_kpi.item() == pytest.approx(-3.0)  # warm-up reflects only p_0
    assert len(res.scored_kpis) == 1  # H apply-actions → exactly H scored KPIs
    assert res.scored_kpis[0].item() == pytest.approx(0.0)
    assert res.step_returns == pytest.approx((0.0,))
    assert res.cumulative_return == pytest.approx(0.0)

    # oracle a_1 = 2 → k_2 = 1
    res2 = rollout_open_loop(lambda: _H1Env(), snap, [(0, 2.0)], _r_k0)
    assert res2.cumulative_return == pytest.approx(1.0)


def test_h1_returned_arrays_are_readonly_copies():
    env = _H1Env()
    res = rollout_open_loop(lambda: _H1Env(), env.snapshot(), [(0, 1.0)], _r_k0)
    assert not res.warmup_kpi.flags.writeable
    assert all(not k.flags.writeable for k in res.scored_kpis)


def test_h2_timeline_two_step_chain():
    """§1.5: planner (p_1,p_2)=(1,0): k_2=[1,1], k_3=[0,−2]; G = 1 + (−2) = −1."""
    env = _H2Env()
    snap = env.snapshot()

    res = rollout_open_loop(lambda: _H2Env(), snap, [(0, 1.0), (0, 0.0)], _r_kb)
    assert np.array_equal(res.warmup_kpi, np.array([0.0, 0.0]))  # k_1 = [0,0]
    assert len(res.scored_kpis) == 2
    assert np.array_equal(res.scored_kpis[0], np.array([1.0, 1.0]))  # k_2
    assert np.array_equal(res.scored_kpis[1], np.array([0.0, -2.0]))  # k_3
    assert res.step_returns == pytest.approx((1.0, -2.0))
    assert res.cumulative_return == pytest.approx(-1.0)


def test_h1_oracle_finds_grid_optimum():
    """§1.4 oracle over p ∈ {0,1,2,3}: argmax at a_1 = 2 with return 1."""
    env = _H1Env()
    options = [(0, 0.0), (0, 1.0), (0, 2.0), (0, 3.0)]
    oracle = enumerate_open_loop(lambda: _H1Env(), env.snapshot(), options, 1, _r_k0)
    assert oracle.best_actions == ((0, 2.0),)
    assert oracle.best_return == pytest.approx(1.0)
    assert oracle.num_sequences == 4


def test_h2_oracle_optimum():
    """§1.5 oracle over p ∈ {0,1}, H=2: best (0,1) return 1."""
    env = _H2Env()
    options = [(0, 0.0), (0, 1.0)]
    oracle = enumerate_open_loop(lambda: _H2Env(), env.snapshot(), options, 2, _r_kb)
    assert oracle.best_actions == ((0, 0.0), (0, 1.0))
    assert oracle.best_return == pytest.approx(1.0)
    assert oracle.num_sequences == 4


def test_oracle_ties_break_lexicographically_smallest():
    """All sequences tie under a constant mechanism → the first (lex-smallest) is kept."""
    env = _ConstEnv()
    options = [(0, 0.0), (0, 1.0), (0, 2.0)]
    oracle = enumerate_open_loop(lambda: _ConstEnv(), env.snapshot(), options, 2, _r_k0)
    assert oracle.best_actions == ((0, 0.0), (0, 0.0))  # lexicographically smallest tie-break


def test_oracle_tiebreak_lex_smallest_with_unsorted_options():
    """Ties break to the lex-smallest sequence even when caller options are UNSORTED.

    The kernel sorts options by (param_id, value) internally, so the tie-break is intrinsic and
    does not depend on the order the caller passed."""
    env = _ConstEnv()
    options = [(0, 2.0), (0, 0.0), (0, 1.0)]  # deliberately unsorted
    oracle = enumerate_open_loop(lambda: _ConstEnv(), env.snapshot(), options, 2, _r_k0)
    assert oracle.best_actions == ((0, 0.0), (0, 0.0))  # lex-smallest, not caller-first (0,2.0)


def test_max_sequences_guard_fires_before_any_rollout():
    """The guard raises BEFORE minting a single env / rollout."""
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return _H1Env()

    options = [(0, 0.0), (0, 1.0), (0, 2.0), (0, 3.0)]  # 4 ** 3 = 64 sequences
    with pytest.raises(ValueError, match="exceeds max_sequences"):
        enumerate_open_loop(factory, _H1Env().snapshot(), options, 3, _r_k0, max_sequences=10)
    assert calls["n"] == 0  # no rollout executed


class _GuardProbeOptions(Sequence):
    """A Sequence whose length is known but which raises if anything MATERIALIZES/iterates it."""

    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx):  # list()/iteration goes through here → must not run pre-guard
        raise AssertionError("option set was materialized/iterated before the max_sequences guard")


def test_max_sequences_guard_fires_before_materialization():
    """An oversized option set is rejected from ``len ** horizon`` alone, before it is listed."""
    probe = cast("Sequence", _GuardProbeOptions(4))  # 4 ** 3 = 64 > 10
    with pytest.raises(ValueError, match="exceeds max_sequences"):
        enumerate_open_loop(lambda: _H1Env(), _H1Env().snapshot(), probe, 3, _r_k0, max_sequences=10)


def test_non_finite_cumulative_return_rejected():
    """Finite per-step scores that overflow to a non-finite SUM are rejected."""
    env = _H2Env()
    with pytest.raises(ValueError, match="cumulative_return is non-finite"):
        # two scored steps, each a finite 1e308 → sum overflows to +inf
        rollout_open_loop(lambda: _H2Env(), env.snapshot(), [(0, 1.0), (0, 0.0)], lambda k: 1e308)


def test_h1_paired_regret_is_one():
    """§1.4: planner a_1=1 (return 0) vs oracle a_1=2 (return 1) → regret 1."""
    env = _H1Env()
    options = [(0, 0.0), (0, 1.0), (0, 2.0), (0, 3.0)]
    reg = paired_regret(lambda: _H1Env(), env.snapshot(), [(0, 1.0)], options, _r_k0)
    assert reg.planner_return == pytest.approx(0.0)
    assert reg.oracle_return == pytest.approx(1.0)
    assert reg.regret == pytest.approx(1.0)
    assert reg.oracle_actions == ((0, 2.0),)


def test_h2_paired_regret_is_two():
    """§1.5: planner (1,0) return −1 vs oracle (0,1) return 1 → cumulative regret 2."""
    env = _H2Env()
    options = [(0, 0.0), (0, 1.0)]
    reg = paired_regret(lambda: _H2Env(), env.snapshot(), [(0, 1.0), (0, 0.0)], options, _r_kb)
    assert reg.planner_return == pytest.approx(-1.0)
    assert reg.oracle_return == pytest.approx(1.0)
    assert reg.regret == pytest.approx(2.0)


def test_regret_is_unclamped_when_oracle_is_suboptimal(monkeypatch):
    """Regret is the raw ``oracle.best_return − planner.cumulative_return`` with NO ``max(0, ·)``
    clamp, so a genuinely SUB-OPTIMAL / buggy oracle surfaces as ``regret < 0`` rather than being
    hidden (SEMANTICS §3 "Non-negativity is a property to verify, not to clamp").

    The SAME pure objective ``_r_k0`` scores BOTH arms (SEMANTICS §2 identical-R contract) and the
    planner acts IN the grid (no out-of-domain escape). Negativity comes solely from an ORACLE that
    is genuinely sub-optimal on its own grid: the exhaustive search is monkeypatched to report an
    in-grid sequence whose return (0) is below the planner's realized return (1)."""
    env = _H1Env()
    snap = env.snapshot()
    grid = [(0, 0.0), (0, 1.0), (0, 2.0)]

    subopt_rollout = rollout_open_loop(lambda: _H1Env(), snap, [(0, 1.0)], _r_k0)  # k_2=0 → return 0
    assert subopt_rollout.cumulative_return == pytest.approx(0.0)

    def _suboptimal_oracle(*args, **kwargs):
        return OracleResult(
            best_actions=subopt_rollout.actions,
            best_return=subopt_rollout.cumulative_return,  # 0.0, below the planner's 1.0
            best_rollout=subopt_rollout,
            num_sequences=1,
        )

    monkeypatch.setattr(rollout_mod, "enumerate_open_loop", _suboptimal_oracle)

    reg = paired_regret(lambda: _H1Env(), snap, [(0, 2.0)], grid, _r_k0)
    assert reg.planner_return == pytest.approx(1.0)  # planner (0,2.0): k_2 = 1 under pure _r_k0
    assert reg.oracle_return == pytest.approx(0.0)   # injected sub-optimal oracle, same R
    assert reg.regret == pytest.approx(-1.0)         # raw, UNCLAMPED: negative regret surfaces the bug


def test_paired_regret_rejects_out_of_domain_planner_action():
    """An out-of-domain planner action (not on the oracle grid) is rejected up front with
    ``ValueError`` — regret is only defined when the planner and the oracle share one action grid
    (SEMANTICS §3). This closes the escape that used to manufacture spurious negative regret."""
    env = _H1Env()
    grid = [(0, 0.0), (0, 1.0)]  # oracle grid; planner action a=2 is NOT on it
    with pytest.raises(ValueError, match="outside the oracle action grid"):
        paired_regret(lambda: _H1Env(), env.snapshot(), [(0, 2.0)], grid, _r_k0)


def test_empty_actions_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        rollout_open_loop(lambda: _H1Env(), _H1Env().snapshot(), [], _r_k0)


def test_non_finite_action_value_rejected():
    with pytest.raises(ValueError, match="finite"):
        rollout_open_loop(lambda: _H1Env(), _H1Env().snapshot(), [(0, float("nan"))], _r_k0)


def test_out_of_range_param_id_rejected():
    with pytest.raises(ValueError, match="out of range"):
        rollout_open_loop(lambda: _H1Env(), _H1Env().snapshot(), [(5, 1.0)], _r_k0)


def test_non_finite_score_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        rollout_open_loop(lambda: _H1Env(), _H1Env().snapshot(), [(0, 1.0)], lambda k: float("inf"))


def test_empty_option_set_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        enumerate_open_loop(lambda: _H1Env(), _H1Env().snapshot(), [], 1, _r_k0)


class _ProcProbeEnv(V2Env):
    """A v2 test env exposing per-step process/observation exogenous draws it consumes.

    ``_update_kpis`` pulls a process-noise value per KPI from the coordinate-keyed tape and logs
    ``(time, eps0, eps1)`` so a test can prove two arms with DIFFERENT actions still consume
    byte-identical exogenous draws at matching coordinates (SEMANTICS §4)."""

    num_params = 2
    num_kpis = 2
    id_ranges = [(0.0, 0.0), (0.0, 0.0)]

    def __init__(self, env_seed: int = 0, obs_noise_scale: float = 0.0, episode: int = 0) -> None:
        self.proc_log: list[tuple[int, float, float]] = []
        super().__init__(env_seed=env_seed, obs_noise_scale=obs_noise_scale, episode=episode)

    def _update_kpis(self, prev_params, prev_kpis):
        eps0 = self._process_noise(0, self.time)
        eps1 = self._process_noise(1, self.time)
        self.proc_log.append((int(self.time), eps0, eps1))
        p = np.asarray(prev_params, dtype=float)
        return np.array([p[0] + eps0, p[1] + eps1], dtype=float)


def test_crn_paired_under_divergent_actions():
    """Two rollouts with entirely different actions consume byte-identical exogenous draws."""
    base = _ProcProbeEnv(env_seed=13, obs_noise_scale=0.5, episode=2)
    snap = base.snapshot()

    created: list[_ProcProbeEnv] = []

    def factory():
        env = _ProcProbeEnv(env_seed=13, obs_noise_scale=0.5, episode=2)
        created.append(env)
        return env

    rollout_open_loop(factory, snap, [(0, 1.0), (1, 2.0), (0, 3.0)], _r_k0)
    env_a = created[-1]
    rollout_open_loop(factory, snap, [(1, -9.0), (0, 8.0), (1, -7.0)], _r_k0)
    env_b = created[-1]

    # process-noise tape: identical coordinates → byte-identical draws despite different actions
    assert env_a.proc_log == env_b.proc_log
    assert env_a.time == env_b.time

    # observation-noise tape at the (shared) final coordinate is byte-identical too. Compare the
    # tape draws directly (reconstructing via observed − latent would fold in each arm's own,
    # action-dependent, latent rounding and is not a byte-equality of the exogenous draw).
    def obs_eps(env):
        return np.array(
            [
                env._coord_rng(env._NS_OBS, v, env.time).normal(0.0, env.obs_noise_scale)
                for v in range(env.num_kpis)
            ]
        )

    eps_a, eps_b = obs_eps(env_a), obs_eps(env_b)
    assert np.array_equal(eps_a, eps_b)
    assert np.any(eps_a != 0.0)  # non-degenerate: the noise is actually present
    # and each arm applies exactly its coordinate's draw to its own latent state
    assert np.array_equal(env_a.observed_kpis(), env_a.latent_kpis() + eps_a)
    assert np.array_equal(env_b.observed_kpis(), env_b.latent_kpis() + eps_b)


def test_caller_env_and_snapshot_unchanged_after_paired_eval():
    """Paired evaluation never mutates the caller's env or the snapshot it passed in."""
    caller = _H2Env()
    snap = caller.snapshot()
    snap_before = copy.deepcopy(snap)
    committed_before = caller.snapshot()

    paired_regret(lambda: _H2Env(), snap, [(0, 1.0), (0, 0.0)], [(0, 0.0), (0, 1.0)], _r_kb)

    for a, b in zip(snap, snap_before, strict=True):
        assert np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
    for a, b in zip(caller.snapshot(), committed_before, strict=True):
        assert np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b


def test_cross_check_e3_gate_local_rollout():
    """Kernel return == the E3 gate's ``sequence.py`` clone rollout arithmetic to 1e-12."""
    from cdd_oran.envs.v2.e3 import E3V2Env
    from cdd_oran.planners.sequence import _rollout_sequence_scored
    from scripts.e3_decision_gate import _R

    seed_env = E3V2Env(env_seed=16)
    seed_env.reset(episode=0)
    for _ in range(3):
        seed_env.neutral_step()
    snap = seed_env.snapshot()

    def factory():
        return E3V2Env(truncate_fanout=False)

    values = [0.3, 0.7, 0.1]
    kernel = rollout_open_loop(factory, snap, [(0, v) for v in values], _R)
    gate_scored = _rollout_sequence_scored(factory, snap, 0, values)
    gate_return = float(sum(_R(k) for k in gate_scored))

    assert len(kernel.scored_kpis) == len(gate_scored)
    for a, b in zip(kernel.scored_kpis, gate_scored, strict=True):
        assert np.allclose(a, b, atol=1e-12, rtol=0.0)
    assert kernel.cumulative_return == pytest.approx(gate_return, abs=1e-12)


def test_cross_check_e2_gate_local_rollout_h1():
    """Kernel H=1 return == the E2 gate's ``v2_regret`` rollout arithmetic to 1e-12."""
    from cdd_oran.analysis.v2_regret import reward, rollout_scored_latent
    from cdd_oran.envs.v2.e2 import E2V2Env
    from scripts.e2_decision_gate import FULL_PANEL_IDS, build_panel

    seed_env = E2V2Env(env_seed=5)
    seed_env.reset(episode=0)
    for _ in range(3):
        seed_env.neutral_step()
    snap = seed_env.snapshot()

    panel = build_panel(E2V2Env(env_seed=0), FULL_PANEL_IDS)

    def score_fn(k):
        return reward(np.asarray(k, dtype=float), panel)

    kernel = rollout_open_loop(lambda: E2V2Env(env_seed=5), snap, [(0, 12.0)], score_fn)

    gate_env = E2V2Env(env_seed=5)
    gate_env.restore(snap)
    gate_k2 = rollout_scored_latent(gate_env, 0, 12.0)
    gate_return = reward(np.asarray(gate_k2, dtype=float), panel)

    assert np.allclose(kernel.scored_kpis[0], gate_k2, atol=1e-12, rtol=0.0)
    assert kernel.cumulative_return == pytest.approx(gate_return, abs=1e-12)
