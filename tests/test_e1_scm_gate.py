"""E1 SCM-gate pytest (WP3).

Runs the LOCKED deterministic SCM gate from ``scripts/e1_scm_gate.py`` and asserts every
clause passes on the true generator, plus a few focused invariants on the v2 E1 env
(latency, noise-off latent accessor, analytic mechanism). Deterministic, noise OFF; no
training, no evaluator wiring.

**Rejection power (review BLOCKER 1 / MAJOR 3).** The gate is injectable over an
env/shadow factory. Each mutation below feeds a *structurally wrong* generator whose
adjacency is still declared as the locked E1 graph, and asserts the gate FAILS with the
right clause flagged — closing the demonstrated ``K2 = P2 + b2*P0_prev`` false-pass.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from cdd_oran.envs.v2.e1 import E1V2Env
from scripts.e1_scm_gate import (
    DO_HIGH,
    DO_LOW,
    MIN_EFF,
    TOL_ZERO,
    GateResult,
    run_e1_scm_gate,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_e1_scm_gate_all_clauses_pass():
    result = run_e1_scm_gate()
    assert result.passed, "E1 SCM gate failed:\n" + "\n".join(result.failures)
    assert result.clauses == {"a": True, "b": True, "c": True, "d": True}


def _factories(cls: type[E1V2Env]) -> tuple[Callable[[int], E1V2Env], Callable[[int, int], E1V2Env]]:
    def env_factory(seed: int) -> E1V2Env:
        return cls(env_seed=seed, obs_noise_scale=0.0)

    def shadow_factory(seed: int, dst_kpi: int) -> E1V2Env:
        if dst_kpi == 2:
            return cls(env_seed=seed, obs_noise_scale=0.0, b2=0.0)
        return cls(env_seed=seed, obs_noise_scale=0.0, b3=0.0)

    return env_factory, shadow_factory


def _run(cls: type[E1V2Env]) -> GateResult:
    env_factory, shadow_factory = _factories(cls)
    return run_e1_scm_gate(env_factory=env_factory, shadow_factory=shadow_factory)


def _failures_for(result: GateResult, tag: str) -> list[str]:
    return [f for f in result.failures if f.lstrip().startswith(tag)]


# --- (a) undeclared direct P0 -> K2 (reads prev_params[0], not the latent KPI) ----------
class _ImpostorDirectParam(E1V2Env):
    """K2 = P2 + b2*P0_prev, adjacency still declared K0 -> K2 (the demonstrated false-pass)."""

    def _update_kpis(self, prev_params, prev_kpis):
        p = np.asarray(prev_params, dtype=float)
        k = np.asarray(prev_kpis, dtype=float)
        k0 = self.a[0] * p[0]
        k1 = self.a[1] * p[1]
        k2 = self.a[2] * p[2] + self.b2 * p[0]  # WRONG: reads param P0, not latent K0
        k3 = self.a[3] * p[3] + self.b3 * k[1]
        return np.array([k0, k1, k2, k3], dtype=float)


def test_gate_rejects_p0_to_k2_impostor():
    result = _run(_ImpostorDirectParam)
    assert not result.passed
    assert result.clauses["c"] is False, "off-manifold direct-source check must catch it"
    assert _failures_for(result, "[c.i]"), "expected c.i (declared edge inert off manifold)"


# --- (b) wrong lag: child reads the current-step K0 instead of prev_kpis[0] --------------
class _WrongLag(E1V2Env):
    def _update_kpis(self, prev_params, prev_kpis):
        p = np.asarray(prev_params, dtype=float)
        k = np.asarray(prev_kpis, dtype=float)
        k0 = self.a[0] * p[0]
        k1 = self.a[1] * p[1]
        k2 = self.a[2] * p[2] + self.b2 * k0  # WRONG lag: current k0, not prev_kpis[0]
        k3 = self.a[3] * p[3] + self.b3 * k[1]
        return np.array([k0, k1, k2, k3], dtype=float)


def test_gate_rejects_wrong_lag():
    result = _run(_WrongLag)
    assert not result.passed
    assert result.clauses["c"] is False
    assert _failures_for(result, "[c.i]")


# --- (c) cross-KPI parent: K2 reads K1_prev instead of K0_prev --------------------------
class _CrossKpiParent(E1V2Env):
    def _update_kpis(self, prev_params, prev_kpis):
        p = np.asarray(prev_params, dtype=float)
        k = np.asarray(prev_kpis, dtype=float)
        k0 = self.a[0] * p[0]
        k1 = self.a[1] * p[1]
        k2 = self.a[2] * p[2] + self.b2 * k[1]  # WRONG parent: K1 instead of K0
        k3 = self.a[3] * p[3] + self.b3 * k[1]
        return np.array([k0, k1, k2, k3], dtype=float)


def test_gate_rejects_cross_kpi_parent():
    result = _run(_CrossKpiParent)
    assert not result.passed
    assert result.clauses["c"] is False
    # Declared K0->K2 is inert (c.i) AND non-parent K1 moves K2 (c.neg).
    assert _failures_for(result, "[c.i]")
    assert _failures_for(result, "[c.neg]")


# --- (d) changed direct param coefficient: a2 = 2.0 -------------------------------------
class _ChangedDirectCoeff(E1V2Env):
    def _update_kpis(self, prev_params, prev_kpis):
        p = np.asarray(prev_params, dtype=float)
        k = np.asarray(prev_kpis, dtype=float)
        k0 = self.a[0] * p[0]
        k1 = self.a[1] * p[1]
        k2 = 2.0 * p[2] + self.b2 * k[0]  # WRONG direct coefficient (locked a2 = 1)
        k3 = self.a[3] * p[3] + self.b3 * k[1]
        return np.array([k0, k1, k2, k3], dtype=float)


def test_gate_rejects_changed_direct_coefficient():
    result = _run(_ChangedDirectCoeff)
    assert not result.passed
    assert result.clauses["d"] is False, "analytic clause must pin the locked coefficient"
    assert _failures_for(result, "[d]")


# --- (e) spurious non-descendant edge: P0 -> K3 ------------------------------------------
class _SpuriousEdge(E1V2Env):
    def _update_kpis(self, prev_params, prev_kpis):
        p = np.asarray(prev_params, dtype=float)
        k = np.asarray(prev_kpis, dtype=float)
        k0 = self.a[0] * p[0]
        k1 = self.a[1] * p[1]
        k2 = self.a[2] * p[2] + self.b2 * k[0]
        k3 = self.a[3] * p[3] + self.b3 * k[1] + 0.3 * p[0]  # spurious P0 -> K3
        return np.array([k0, k1, k2, k3], dtype=float)


def test_gate_rejects_spurious_nondescendant_edge():
    result = _run(_SpuriousEdge)
    assert not result.passed
    assert result.clauses["b"] is False, "non-descendant control must catch do(P0)->K3"
    assert _failures_for(result, "[b]")


def test_one_step_latency_action_delayed():
    # An action applied now must NOT affect the immediately-next latent KPI; it affects the
    # one after (SEMANTICS one-step latency).
    env = E1V2Env(env_seed=3, obs_noise_scale=0.0)
    for _ in range(3):
        env.neutral_step()
    baseline = env.snapshot()

    env.restore(baseline)
    env.apply_action(0, 0.9)
    k_after_apply = env.advance()  # warm-up: reflects pre-action params
    env.restore(baseline)
    k_neutral = env.advance()
    assert np.allclose(k_after_apply, k_neutral, atol=TOL_ZERO)

    k_next = env.advance()  # would be reached from k_after_apply; check via held rollout
    env.restore(baseline)
    env.apply_action(0, 0.9)
    env.advance()
    k_delayed = env.advance()
    assert abs(k_delayed[0] - 0.9) <= TOL_ZERO  # K0 now reflects the applied P0
    assert not np.allclose(k_delayed, k_next, atol=TOL_ZERO)


def test_noise_off_latent_equals_observed():
    env = E1V2Env(env_seed=1, obs_noise_scale=0.0)
    for _ in range(3):
        env.neutral_step()
    assert np.allclose(env.latent_kpis(), env.observed_kpis(), atol=0.0)


def test_analytic_mechanism_matches_rollout():
    env = E1V2Env(env_seed=5, obs_noise_scale=0.0)
    for _ in range(3):
        env.neutral_step()
    p, k = env.prev_params.copy(), env.prev_kpis.copy()
    expected = np.array([p[0], p[1], p[2] + 0.5 * k[0], p[3] + 0.5 * k[1]])
    assert np.allclose(env.advance(), expected, atol=TOL_ZERO)


def test_direct_param_effects_match_locked_analytic_values():
    # Direct param edges give 0.5/sigma standardized units; do(high)-do(low) = 0.5 raw.
    env = E1V2Env(env_seed=0, obs_noise_scale=0.0)
    for _ in range(3):
        env.neutral_step()
    snap = env.snapshot()

    def do_first_latent(param_id, value):
        env.restore(snap)
        env.apply_action(param_id, value)
        env.advance()
        return env.advance()

    d_p0 = abs(do_first_latent(0, DO_HIGH)[0] - do_first_latent(0, DO_LOW)[0])
    assert abs(d_p0 - 0.5) <= TOL_ZERO
    assert d_p0 / float(E1V2Env.sigma[0]) >= MIN_EFF


# Invocation smoke test (review MINOR 7): the documented direct-path command runs green.
def test_script_runs_as_documented_command():
    proc = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / "e1_scm_gate.py")],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"gate script exited {proc.returncode}:\n{proc.stdout}\n{proc.stderr}"
    assert "RESULT: PASS" in proc.stdout
