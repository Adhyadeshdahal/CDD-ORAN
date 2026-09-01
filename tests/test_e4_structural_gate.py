"""E4 structural design-validation gate pytest.

Runs the FROZEN deterministic structural gate from ``scripts/e4_structural_gate.py`` and asserts
the frozen moments/lines, the graded gap, the exact ``lambda=0`` factor-removal control, the bank
seeds, the two-clause acceptance (with a REJECTED bad control), and focused invariants on the v2
E4 env (obs/do modes, the confounded ``Z``, H=1 latency, snapshot/restore). Deterministic, noise
OFF; no training — the trained-arm decision-value evaluation is DEFERRED.

**Rejection power.** The gate is injectable over a control-env factory. A bad control that does
NOT actually remove ``Z -> A_behavior`` (``lambda != 0``) leaves the pooled line sign-reversed, so
the control gap stays large and the gate FAILS on the control clause — proving BOTH acceptance
clauses drive ``passed`` and the process exit (the E2 lesson).
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from cdd_oran.envs.v2.e4 import MU_OUT, SIGMA_OUT, E4V2Env
from scripts.e4_structural_gate import (
    CONTROL_LAM,
    FROZEN_A_NAIVE,
    FROZEN_A_ORACLE,
    FROZEN_B_POOL,
    FROZEN_C_POOL,
    FROZEN_COV_AZ,
    FROZEN_D_STATE,
    FROZEN_GAP_NORM,
    FROZEN_P,
    FROZEN_VAR_A,
    PRIMARY_LAM,
    STAT_TOL,
    TOL_ZERO,
    V_GRID,
    _gap_norm,
    build_bank,
    clipped_normal_moments,
    default_control_env_factory,
    default_env_factory,
    generate_corpus,
    pooled_line,
    run_e4_structural_gate,
    validate_env_corpus,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Small env-corpus for the fast rejection / bad-control tests. A broken env fails the mechanism
# identity (exact, tol_zero) and the moment checks at ANY N, so these do not need the frozen N.
_FAST = dict(corpus_n_obs=4000, corpus_n_do=1000, corpus_pool_do=1000)


# --------------------------------------------------------------------------------------
# True generator: full PASS on both acceptance clauses.
# --------------------------------------------------------------------------------------
def test_e4_structural_gate_passes():
    # Runs the FULL frozen env-corpus (BLOCKER 1 fix): the true env's GENERATED transitions must
    # match the frozen closed-form within stat_tol, in addition to the analytic clauses.
    result = run_e4_structural_gate()
    assert result.passed, "E4 structural gate failed:\n" + "\n".join(result.failures)
    assert result.feasible and result.seeds_ok
    assert result.adjacency_ok and result.latent_edges_ok
    assert result.env_corpus_ok and result.control_corpus_ok
    # Empirical ≈ frozen for every checked quantity; mechanism identity is exact.
    ec = result.env_corpus
    for key, frozen in ec.frozen.items():
        assert abs(ec.empirical[key] - frozen) <= STAT_TOL, f"{key} drifted"
    assert ec.empirical["mech_err_obs"] == 0.0 and ec.empirical["mech_err_do"] == 0.0


# --- (a) exact clipped-normal moments and the naive 90:10 pooled line -------------------
def test_frozen_moments_and_pooled_line_to_tol_zero():
    p, var_a, cov_az = clipped_normal_moments(PRIMARY_LAM)
    assert abs(p - FROZEN_P) <= TOL_ZERO
    assert abs(var_a - FROZEN_VAR_A) <= TOL_ZERO
    assert abs(cov_az - FROZEN_COV_AZ) <= TOL_ZERO

    b_pool, c_pool = pooled_line(PRIMARY_LAM, alpha=-1.0, theta=2.5)
    assert abs(b_pool - FROZEN_B_POOL) <= TOL_ZERO
    assert abs(c_pool - FROZEN_C_POOL) <= TOL_ZERO
    # A robust sign reversal: the naive slope is strongly POSITIVE while the true effect is -1.
    assert b_pool > 3.0


# --- (b) a_oracle=0, a_naive=0.66, gap_norm == frozen (single-state, exact to tol_zero) --
def test_frozen_actions_and_gap_norm():
    true_env = default_env_factory(0)
    gap_norm, d, a_oracle_idx, a_naive_idx = _gap_norm(true_env, FROZEN_B_POOL, FROZEN_C_POOL)
    assert float(V_GRID[a_oracle_idx]) == FROZEN_A_ORACLE
    assert float(V_GRID[a_naive_idx]) == FROZEN_A_NAIVE
    assert abs(gap_norm - FROZEN_GAP_NORM) <= TOL_ZERO
    assert abs(d - FROZEN_D_STATE) <= TOL_ZERO


# --- (c) lambda=0 factor-removal control collapses the gap to exactly 0 -----------------
def test_lambda_zero_control_gap_is_zero():
    b_pool_c, c_pool_c = pooled_line(CONTROL_LAM, alpha=-1.0, theta=2.5)
    assert abs(b_pool_c - (-1.0)) <= TOL_ZERO  # confounding removed ⇒ true slope recovered
    assert abs(c_pool_c - 0.0) <= TOL_ZERO
    control_env = default_control_env_factory(0)
    gap_c, _d, _ao, a_naive_c_idx = _gap_norm(control_env, b_pool_c, c_pool_c)
    assert float(V_GRID[a_naive_c_idx]) == 0.0  # naive control agrees with the oracle
    assert abs(gap_c) <= TOL_ZERO


# --- (d) geometry bank is exactly seeds 0..63 -------------------------------------------
def test_bank_is_seeds_0_to_63():
    bank = build_bank()
    assert bank.feasible
    assert bank.seeds == list(range(64))
    assert len(bank.snaps) == 64


# --- (e) BOTH clauses gate: a bad control (lambda != 0) is REJECTED ----------------------
def test_bad_control_is_rejected_in_process():
    # Control env that does NOT remove Z->A_behavior (lambda stays at the primary coupling):
    # the pooled line is still sign-reversed, so the control gap is large and the gate FAILS.
    def bad_control_factory(seed: int) -> E4V2Env:
        return E4V2Env(env_seed=seed, obs_noise_scale=0.0, lam=PRIMARY_LAM, mode=E4V2Env.MODE_DO)

    result = run_e4_structural_gate(control_env_factory=bad_control_factory, **_FAST)
    assert not result.passed
    assert result.mean_control > 0.01
    assert any("gap_norm_control" in f for f in result.failures)


def test_bad_control_nonzero_exit_subprocess():
    # SUBPROCESS check: a bad control injected into a real process yields a NONZERO exit status,
    # so the AND-of-both-clauses is wired to process exit (E2 BLOCKER 1 lesson).
    code = (
        "import sys; "
        "from scripts.e4_structural_gate import run_e4_structural_gate; "
        "from cdd_oran.envs.v2.e4 import E4V2Env; "
        "bad=lambda s: E4V2Env(env_seed=s, obs_noise_scale=0.0, lam=1.0, mode=E4V2Env.MODE_DO); "
        "r=run_e4_structural_gate(control_env_factory=bad, "
        "corpus_n_obs=4000, corpus_n_do=1000, corpus_pool_do=1000); "
        "sys.exit(0 if r.passed else 1)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=str(_REPO_ROOT), capture_output=True, text=True
    )
    assert proc.returncode == 1, f"bad control should fail:\n{proc.stdout}\n{proc.stderr}"


def test_script_runs_as_documented_command():
    proc = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / "e4_structural_gate.py")],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"gate exited {proc.returncode}:\n{proc.stdout}\n{proc.stderr}"
    assert "RESULT: PASS" in proc.stdout
    assert "DEFERRED" in proc.stdout
    assert "env-corpus (primary" in proc.stdout  # empirical-vs-frozen table is printed


# --------------------------------------------------------------------------------------
# ENV-CORPUS REJECTION POWER (circularity fix, review-e4-gate.md BLOCKER 1).
# The analytic clauses re-derive the frozen algebra and cannot see the env; these broken envs
# keep the DECLARED metadata/attrs but GENERATE data that violates the design, so ONLY the
# env-corpus validation can reject them.
# --------------------------------------------------------------------------------------
class _BrokenNoConfounding(E4V2Env):
    """Both functional latent edges deleted: behavior ≡ 0.5 (no Z->A_behavior) and K_out=-A
    (no Z->K_out), while the declared attrs (lam/alpha/theta) and LATENT_EDGES metadata stay."""

    def behavior_action(self) -> float:
        return 0.5

    def _update_kpis(self, prev_params, prev_kpis):
        return np.array([self.alpha * float(prev_params[0])], dtype=float)  # K_out = -A


class _ThetaMisscale(E4V2Env):
    """Mis-scaled outcome mechanism: generates K_out = alpha*A + (2*theta)*Z + c while the
    declared attr `theta` stays 2.5 — the env-corpus mechanism identity and slopes must catch it."""

    def _update_kpis(self, prev_params, prev_kpis):
        k = self.alpha * float(prev_params[0]) + 2.0 * self.theta * self.prev_Z + self.c
        return np.array([k], dtype=float)


def test_env_corpus_rejects_broken_no_confounding():
    def broken_factory(seed: int) -> _BrokenNoConfounding:
        return _BrokenNoConfounding(env_seed=seed, obs_noise_scale=0.0, mode=E4V2Env.MODE_DO)

    result = run_e4_structural_gate(env_factory=broken_factory, **_FAST)
    assert not result.passed, "gate must REJECT an env with no action-relevant confounding"
    assert not result.env_corpus_ok
    # It fails on the env-GENERATED corpus, not on any analytic clause: the frozen analytic bank
    # gap is state-independent (score_interventional_mean is untouched), so those still hold.
    assert result.mean_gap >= 0.10  # analytic gap unchanged — proves the check is env-level
    assert any("env-corpus:primary" in f for f in result.failures)


def test_env_corpus_rejects_theta_misscale():
    def misscale_factory(seed: int) -> _ThetaMisscale:
        return _ThetaMisscale(env_seed=seed, obs_noise_scale=0.0, mode=E4V2Env.MODE_DO)

    result = run_e4_structural_gate(env_factory=misscale_factory, **_FAST)
    assert not result.passed
    assert not result.env_corpus_ok
    # Mechanism identity K_out == alpha*A + theta*Z + c is violated on every transition.
    assert any("mechanism" in f for f in result.failures)


def test_env_corpus_true_env_generates_frozen_moments():
    # Direct corpus check on the true env. Mechanism is EXACT and the low-variance moments track
    # the frozen closed-form (the do OLS slope is a high-variance estimator that needs the frozen
    # N to tighten — the full-N pass is covered by test_e4_structural_gate_passes).
    res = validate_env_corpus(default_env_factory, "primary", n_obs=8000, n_do=8000, pool_do=1000)
    assert res.empirical["mech_err_obs"] == 0.0 and res.empirical["mech_err_do"] == 0.0
    for key in ("obs_E[A]", "obs_Var(A)", "obs_Cov(A,Z)", "obs_slope", "pooled_slope"):
        assert abs(res.empirical[key] - res.frozen[key]) <= STAT_TOL, f"{key} drifted"
    # The naive observational association is strongly POSITIVE (sign-reversed from the true -1).
    assert res.empirical["obs_slope"] > 3.0
    assert abs(res.empirical["obs_Cov(A,Z)"] - FROZEN_COV_AZ) <= STAT_TOL


def test_generate_corpus_pairs_committed_action_and_Z():
    # Every generated transition satisfies the exact mechanism against the committed (A, Z).
    probe = default_env_factory(0)
    a, z, k = generate_corpus(probe, E4V2Env.MODE_OBS, 500)
    assert np.max(np.abs(k - (probe.alpha * a + probe.theta * z + probe.c))) == 0.0
    assert a.min() >= 0.0 and a.max() <= 1.0  # clipped behavior policy stays in the action space


# --------------------------------------------------------------------------------------
# (f) obs vs do modes: same Z enters K_out both ways, A ⊥ Z under do, A=A_behavior under obs.
# --------------------------------------------------------------------------------------
def test_behavior_action_is_clipped_policy_of_Z():
    env = E4V2Env(env_seed=3, lam=1.0, mode=E4V2Env.MODE_OBS)
    eta = env._draw_eta(env.time)
    expected = float(np.clip(0.5 + env.lam * env.Z + eta, 0.0, 1.0))
    assert env.behavior_action() == expected
    # Under obs mode the generated action IS the behavior policy.
    assert env.generate_action() == expected


def test_do_action_is_independent_of_Z():
    env = E4V2Env(env_seed=3, lam=1.0, mode=E4V2Env.MODE_DO)
    a1 = env.generate_action()
    env.Z = env.Z + 123.456  # perturb the latent
    a2 = env.generate_action()
    assert a1 == a2  # do-action drawn from its own tape slot, never reads Z
    assert a1 in set(np.round(V_GRID, 2))


def test_same_Z_enters_K_out_under_both_modes():
    # Same seed ⇒ same coordinate-keyed Z. Applying the SAME action under obs and do and advancing
    # gives the identical K_out, because the outcome uses the same latent Z either way.
    do_env = E4V2Env(env_seed=7, mode=E4V2Env.MODE_DO)
    obs_env = E4V2Env(env_seed=7, mode=E4V2Env.MODE_OBS)
    assert do_env.Z == obs_env.Z

    def roll(env, v):
        env.reset(episode=0)
        env.apply_action(0, v)
        env.advance()  # warm-up
        return float(env.advance()[0])  # terminal K_out

    k_do = roll(do_env, 0.3)
    k_obs = roll(obs_env, 0.3)
    assert k_do == k_obs
    # And it equals the closed-form alpha*A + theta*Z0 + c with the committed Z0.
    z0 = E4V2Env(env_seed=7).Z
    assert math.isclose(k_do, -1.0 * 0.3 + 2.5 * z0 + 0.0, rel_tol=0.0, abs_tol=TOL_ZERO)


# --------------------------------------------------------------------------------------
# (g) H=1 one-step latency; latent == observed at noise 0.
# --------------------------------------------------------------------------------------
def test_h1_latency_action_hits_second_advance():
    env = E4V2Env(env_seed=1, mode=E4V2Env.MODE_DO)
    init_a = float(env.prev_params[0])
    z0 = env.Z
    env.reset(episode=0)
    v = init_a + 0.4  # clearly different from the initial action
    env.apply_action(0, v)
    k1 = float(env.advance()[0])  # warm-up: reflects the PRE-decision (init) action, not v
    k2 = float(env.advance()[0])  # terminal: reflects v
    assert math.isclose(k1, -1.0 * init_a + 2.5 * z0, abs_tol=TOL_ZERO)
    assert math.isclose(k2, -1.0 * v + 2.5 * z0, abs_tol=TOL_ZERO)
    assert k1 != k2


def test_noise_off_latent_equals_observed():
    env = E4V2Env(env_seed=2, obs_noise_scale=0.0, mode=E4V2Env.MODE_DO)
    env.step(0, 0.5)
    assert np.allclose(env.latent_kpis(), env.observed_kpis(), atol=0.0)


# --------------------------------------------------------------------------------------
# (h) snapshot/restore carries pending + committed Z.
# --------------------------------------------------------------------------------------
def test_snapshot_restore_carries_Z():
    env = E4V2Env(env_seed=5, mode=E4V2Env.MODE_DO)
    for _ in range(3):
        env.neutral_step()
    snap = env.snapshot()
    z_pending, z_committed = float(env.Z), float(env.prev_Z)
    assert snap[5] == z_pending and snap[6] == z_committed

    # Roll forward on a clone, then restore and confirm exact replay of the scored outcome.
    env.apply_action(0, 0.8)
    env.advance()
    k_after_first = float(env.advance()[0])

    env.restore(snap)
    assert float(env.Z) == z_pending and float(env.prev_Z) == z_committed
    env.apply_action(0, 0.8)
    env.advance()
    assert float(env.advance()[0]) == k_after_first


# --------------------------------------------------------------------------------------
# Focused frozen-constant invariants.
# --------------------------------------------------------------------------------------
def test_standardization_and_adjacency_constants():
    assert E4V2Env.mu[0] == MU_OUT == -0.5
    assert E4V2Env.sigma[0] == SIGMA_OUT == math.sqrt(1.0 / 12.0)
    assert list(E4V2Env(env_seed=0).adjacency_edges) == [(0, 0)]
    assert list(E4V2Env.LATENT_EDGES) == [("Z", "A_behavior"), ("Z", "K_out")]
    m = E4V2Env(env_seed=0).true_adj_matrix()
    assert m.shape == (2, 2)
    assert int(m.sum()) == 1 and m[1, 0] == 1.0  # K0 (row 1) reads P0 (col 0)
