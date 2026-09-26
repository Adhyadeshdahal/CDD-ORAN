"""E2 decision-gap gate pytest (sim-level, NO training).

Real assertions against the FROZEN contract (``docs/benchmark/GATE_CONTRACT_E2.md``). Both sides
of every inequality are computed from the gate, never hard-coded:

(a) positive-control mean gap >= tau_E2 (0.10);
(b) negative-control mean gap <= tol (0.01);
(c) REJECTION POWER — a non-decoy world model that KEEPS P0->K5 yields gap <= tol on positives,
    so the gate would NOT fire: proves the gap is caused by the omitted edge, not an artifact;
(d) §3.1 factor-removal (drop xApp4/K5 from the panel) collapses the positive gap to <= tol;
(e) H=1 call-list latency + latent == observed at noise=0 on E2V2Env;
(f) bank feasibility — exactly 32 pos + 32 neg found in seeds 0..4095.
"""

from __future__ import annotations

import numpy as np

from cdd_oran.envs.v2.e2 import E2V2Env, safe_exp
from scripts.e2_decision_gate import (
    N_PER_CLASS,
    TAU_E2,
    TOL,
    build_bank,
    run_e2_decision_gate,
)


def _true_factory(seed: int) -> E2V2Env:
    return E2V2Env(env_seed=seed)


def _decoy_factory(seed: int) -> E2V2Env:
    return E2V2Env(env_seed=seed, decoy_omit_p0_k5=True)


class _P0BlindEnv(E2V2Env):
    """Test-only decoy that freezes the committed P0 in EVERY KPI (not just K5), so it
    mispredicts control-panel KPIs (K0/K1/K2) too. Used to force ``factor_removal_pos > TOL`` and
    prove the acceptance logic gates on the factor-removal control."""

    def _update_kpis(self, prev_params, prev_kpis):
        p = np.asarray(prev_params, dtype=float).copy()
        p[0] = self._decoy_p0_ref
        return E2V2Env._update_kpis(self, p, prev_kpis)


def _p0_blind_factory(seed: int) -> E2V2Env:
    return _P0BlindEnv(env_seed=seed, decoy_omit_p0_k5=True)


def test_positive_control_meets_preregistered_target():
    """(a): the FROZEN PASS criterion mean_positive(gap_norm) >= tau_E2.

    Now an expected PASS after the E2 operating-point redesign (protocol_commit 8ed31bf,
    docs/benchmark/E2_OPERATING_POINT_REDESIGN.md): narrowing P6 to (1.5, 4.0) makes the
    shared-knob trap live, so mean_positive(gap_norm) ~= 0.19 >= tau_E2 = 0.10. tau_E2 is
    UNCHANGED — the env was repaired, not the threshold."""
    result = run_e2_decision_gate(_true_factory, _decoy_factory)
    assert result.feasible
    assert result.mean_pos >= TAU_E2, f"mean_positive {result.mean_pos} < {TAU_E2}"


def test_negative_controls_are_inert():
    """(b): negative-control mean gap <= tol — the decoy is inert where do(P0) cannot move K5."""
    result = run_e2_decision_gate(_true_factory, _decoy_factory)
    assert result.feasible
    assert result.mean_neg <= TOL, f"mean_negative {result.mean_neg} > {TOL}"


def test_factor_removal_control_collapses_gap():
    """(d): removing xApp4/K5 from the panel drops the positive gap to <= tol."""
    result = run_e2_decision_gate(_true_factory, _decoy_factory)
    assert result.feasible
    assert result.factor_removal_pos <= TOL, (
        f"factor-removal positive gap {result.factor_removal_pos} > {TOL} — the gap does not "
        f"localize to the shared-control factor"
    )


def test_acceptance_gates_on_factor_removal_control():
    """Regression: gate.passed must be False whenever the factor-removal control exceeds TOL.

    Inject a P0-blind decoy that mispredicts the control-panel KPIs (K0/K1/K2), so dropping
    xApp4/K5 does NOT collapse the gap. The gap then fails to localize to the shared-control
    factor and the gate must reject — even if the positive/negative criteria were met.
    """
    result = run_e2_decision_gate(_true_factory, _p0_blind_factory)
    assert result.feasible
    assert result.factor_removal_pos > TOL, (
        "test setup: the P0-blind decoy should leave a control-panel gap above tol"
    )
    assert result.passed is False
    assert any("factor_removal_pos" in m for m in result.failures)


def test_rejection_power_non_decoy_does_not_fire():
    """(c): a world model that KEEPS P0->K5 (decoy == true) yields no positive gap.

    This proves the measured positive gap is caused by omitting (5,0), not by an artifact of the
    harness. The 'decoy' here is a second TRUE env, so a_decoy == a_oracle and gap == 0.
    """
    result = run_e2_decision_gate(_true_factory, _true_factory)
    assert result.feasible
    assert result.mean_pos <= TOL, (
        f"non-decoy control positive gap {result.mean_pos} > {TOL} — the gate fires without the "
        f"omitted edge, so the gap is an artifact"
    )


def test_bank_feasibility_32_plus_32():
    """(f): exactly 32 positive + 32 negative controls exist in seeds 0..4095."""
    bank = build_bank(_true_factory)
    assert bank.feasible
    assert len(bank.positives) == N_PER_CLASS
    assert len(bank.negatives) == N_PER_CLASS
    pos_seeds = {s for s, _ in bank.positives}
    neg_seeds = {s for s, _ in bank.negatives}
    assert pos_seeds.isdisjoint(neg_seeds)


def test_h1_call_list_latency():
    """(e): apply_action's effect surfaces one advance later (one-step latency, SEMANTICS §1.1).

    From a committed state, the warm-up advance (k1) still reflects the pre-decision P0, and only
    the terminal advance (k2) reflects the applied P0.
    """
    env = E2V2Env(env_seed=3)
    env.reset(episode=0)
    for _ in range(3):
        env.neutral_step()
    snap = env.snapshot()
    committed_p0 = float(snap[1][0])

    env.restore(snap)
    env.apply_action(0, committed_p0 + 60.0)
    k1 = env.advance()
    k2_moved = env.advance()

    env.restore(snap)
    env.apply_action(0, committed_p0)
    env.advance()
    k2_same = env.advance()

    env.restore(snap)
    k1_neutral = env.advance()
    assert np.allclose(k1, k1_neutral), "warm-up k1 must not reflect the applied action"
    assert not np.allclose(k2_moved, k2_same), "terminal k2 must reflect the applied P0"


def test_latent_equals_observed_at_noise_off():
    """(e): with obs_noise_scale=0, observed == latent (SEMANTICS §2 / PD3)."""
    env = E2V2Env(env_seed=7, obs_noise_scale=0.0)
    env.reset(episode=0)
    env.neutral_step()
    assert np.array_equal(env.latent_kpis(), env.observed_kpis())


def test_decoy_freezes_p0_in_k5_only():
    """The decoy's terminal K5 is invariant to the applied P0, while true K5 responds."""
    seed = 5
    true_env, decoy_env = _true_factory(seed), _decoy_factory(seed)
    true_env.reset(episode=0)
    for _ in range(3):
        true_env.neutral_step()
    snap = true_env.snapshot()

    def terminal_k5(env, v):
        env.restore(snap)
        env.apply_action(0, v)
        env.advance()
        return float(env.advance()[5])

    committed_p0 = float(snap[1][0])
    v_lo, v_hi = committed_p0 - 40.0, committed_p0 + 40.0
    assert not np.isclose(terminal_k5(true_env, v_lo), terminal_k5(true_env, v_hi))
    assert np.isclose(terminal_k5(decoy_env, v_lo), terminal_k5(decoy_env, v_hi))
    # Sanity: decoy K5 equals the TRUE K5 evaluated at the committed P0.
    p6, p7 = float(snap[1][6]), float(snap[1][7])
    frozen = -35.0 * np.exp(-((p7 + committed_p0 - 25.0) ** 2) / (2.0 * safe_exp(p6) ** 2))
    assert np.isclose(terminal_k5(decoy_env, v_hi), frozen)
