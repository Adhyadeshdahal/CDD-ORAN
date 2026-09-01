"""E3 SCM-gate pytest.

Runs the FROZEN deterministic SCM gate from ``scripts/e3_scm_gate.py`` and asserts every clause
passes on the true generator, plus focused invariants on the v2 E3 env (exact-sqrt
standardization, cascade timing, adjacency/decoy structure). Deterministic, noise OFF; no
training, no evaluator wiring.

**Rejection power.** The gate is injectable over an env/shadow factory. Each mutation below feeds
a *structurally wrong* generator whose adjacency is still declared as the frozen E3 graph, and
asserts the gate FAILS with the right clause flagged.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from cdd_oran.envs.v2.e3 import E3V2Env
from scripts.e3_scm_gate import (
    DO_HIGH,
    DO_LOW,
    MIN_EFF,
    TOL_ZERO,
    GateResult,
    run_e3_scm_gate,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------------------
# True generator: full 5-clause PASS.
# --------------------------------------------------------------------------------------
def test_e3_scm_gate_all_clauses_pass():
    result = run_e3_scm_gate()
    assert result.passed, "E3 SCM gate failed:\n" + "\n".join(result.failures)
    assert result.clauses == {"a": True, "b": True, "c": True, "d": True, "e": True}


# --------------------------------------------------------------------------------------
# Mutation harness: injectable factories over a wrong-generator subclass.
# --------------------------------------------------------------------------------------
def _factories(
    cls: type[E3V2Env],
) -> tuple[Callable[[int], E3V2Env], Callable[[int, int], E3V2Env]]:
    def env_factory(seed: int) -> E3V2Env:
        return cls(env_seed=seed, obs_noise_scale=0.0)

    def shadow_factory(seed: int, dst_kpi: int) -> E3V2Env:
        kw = {1: {"c10": 0.0}, 2: {"c25": 0.0}, 3: {"c35": 0.0}}[dst_kpi]
        return cls(env_seed=seed, obs_noise_scale=0.0, **kw)

    return env_factory, shadow_factory


def _run(cls: type[E3V2Env]) -> GateResult:
    env_factory, shadow_factory = _factories(cls)
    return run_e3_scm_gate(env_factory=env_factory, shadow_factory=shadow_factory)


def _failures_for(result: GateResult, tag: str) -> list[str]:
    return [f for f in result.failures if f.lstrip().startswith(tag)]


# --- (1) wrong lag: K1 reads the current-step K0 instead of prev_kpis[0] ----------------
class _WrongLag(E3V2Env):
    def _update_kpis(self, prev_params, prev_kpis):
        p = np.asarray(prev_params, dtype=float)
        k = np.asarray(prev_kpis, dtype=float)
        k0 = self.a[0] * p[0]
        k1 = self.c10 * k0  # WRONG lag: current k0, not prev_kpis[0]
        k2 = self.a[1] * p[1] + self.c25 * k[1]
        k3 = self.a[2] * p[2] + self.c35 * k[1]
        k4 = self.a[3] * p[3]
        return np.array([k0, k1, k2, k3, k4], dtype=float)


def test_gate_rejects_wrong_lag():
    result = _run(_WrongLag)
    assert not result.passed
    # A same-step K0 read collapses the K0->K1 chain depth: closed form and off-manifold
    # direct-source parentage both break.
    assert result.clauses["c"] is False or result.clauses["d"] is False


# --- (2) cross-KPI parent: K2 reads K0_prev instead of K1_prev --------------------------
class _CrossKpiParent(E3V2Env):
    def _update_kpis(self, prev_params, prev_kpis):
        p = np.asarray(prev_params, dtype=float)
        k = np.asarray(prev_kpis, dtype=float)
        k0 = self.a[0] * p[0]
        k1 = self.c10 * k[0]
        k2 = self.a[1] * p[1] + self.c25 * k[0]  # WRONG parent: K0 instead of K1
        k3 = self.a[2] * p[2] + self.c35 * k[1]
        k4 = self.a[3] * p[3]
        return np.array([k0, k1, k2, k3, k4], dtype=float)


def test_gate_rejects_cross_kpi_parent():
    result = _run(_CrossKpiParent)
    assert not result.passed
    assert result.clauses["c"] is False
    # Declared K1->K2 inert off-manifold (c.i) AND non-parent K0 moves K2 (c.neg).
    assert _failures_for(result, "[c.i]")
    assert _failures_for(result, "[c.neg]")


# --- (3) changed direct coefficient: a(P1->K2) = 2.0 ------------------------------------
class _ChangedDirectCoeff(E3V2Env):
    def _update_kpis(self, prev_params, prev_kpis):
        p = np.asarray(prev_params, dtype=float)
        k = np.asarray(prev_kpis, dtype=float)
        k0 = self.a[0] * p[0]
        k1 = self.c10 * k[0]
        k2 = 2.0 * p[1] + self.c25 * k[1]  # WRONG direct coefficient (frozen P1->K2 = 1)
        k3 = self.a[2] * p[2] + self.c35 * k[1]
        k4 = self.a[3] * p[3]
        return np.array([k0, k1, k2, k3, k4], dtype=float)


def test_gate_rejects_changed_direct_coefficient():
    result = _run(_ChangedDirectCoeff)
    assert not result.passed
    assert result.clauses["d"] is False, "analytic clause must pin the frozen coefficient"
    assert _failures_for(result, "[d]")


# --- (4) spurious non-descendant edge: P0 -> K4 -----------------------------------------
class _SpuriousEdge(E3V2Env):
    def _update_kpis(self, prev_params, prev_kpis):
        p = np.asarray(prev_params, dtype=float)
        k = np.asarray(prev_kpis, dtype=float)
        k0 = self.a[0] * p[0]
        k1 = self.c10 * k[0]
        k2 = self.a[1] * p[1] + self.c25 * k[1]
        k3 = self.a[2] * p[2] + self.c35 * k[1]
        k4 = self.a[3] * p[3] + 0.3 * p[0]  # spurious P0 -> K4 (K4 must be P0's neg control)
        return np.array([k0, k1, k2, k3, k4], dtype=float)


def test_gate_rejects_spurious_p0_to_k4():
    result = _run(_SpuriousEdge)
    assert not result.passed
    assert result.clauses["b"] is False, "K4 must be inert under do(P0)"
    assert _failures_for(result, "[b]")


# --- (5) impostor: K1 reads param P0 directly instead of the K0 conduit -----------------
class _ImpostorDirectParam(E3V2Env):
    """K1 = c10*P0_prev, adjacency still declared K0->K1 (reads the param, not the conduit)."""

    def _update_kpis(self, prev_params, prev_kpis):
        p = np.asarray(prev_params, dtype=float)
        k = np.asarray(prev_kpis, dtype=float)
        k0 = self.a[0] * p[0]
        k1 = self.c10 * p[0]  # WRONG: reads param P0, not latent K0
        k2 = self.a[1] * p[1] + self.c25 * k[1]
        k3 = self.a[2] * p[2] + self.c35 * k[1]
        k4 = self.a[3] * p[3]
        return np.array([k0, k1, k2, k3, k4], dtype=float)


def test_gate_rejects_p0_to_k1_impostor():
    result = _run(_ImpostorDirectParam)
    assert not result.passed
    assert result.clauses["c"] is False, "off-manifold direct-source check must catch it"
    assert _failures_for(result, "[c.i]"), "declared K0->K1 edge inert off manifold"


# --------------------------------------------------------------------------------------
# Focused env invariants.
# --------------------------------------------------------------------------------------
def test_standardization_equals_exact_sqrt():
    expected_sigma = np.array(
        [
            math.sqrt(1.0 / 12.0),
            math.sqrt(1.0 / 12.0),
            math.sqrt(2.0 / 12.0),
            math.sqrt(2.0 / 12.0),
            math.sqrt(1.0 / 12.0),
        ]
    )
    assert np.array_equal(E3V2Env.sigma, expected_sigma), "sigma must be exact sqrt, not rounded"
    assert np.array_equal(E3V2Env.mu, np.array([0.5, 0.5, 1.0, 1.0, 0.5]))
    # Exact float identity of L used by the (future) state bank.
    assert E3V2Env.kpi_thresholds == (0.5, 0.5, 1.0, 1.0, 0.5)


def test_steady_state_after_three_neutral_advances():
    # (K0,K1,K2,K3,K4) = (P0, P0, P1+P0, P2+P0, P3) after exactly 3 neutral advances.
    env = E3V2Env(env_seed=7, obs_noise_scale=0.0)
    p = env.prev_params.copy()
    for _ in range(3):
        env.neutral_step()
    k = env.prev_kpis
    expected = np.array([p[0], p[0], p[1] + p[0], p[2] + p[0], p[3]])
    assert np.allclose(k, expected, atol=TOL_ZERO)


def test_cascade_timing_p0_reaches_k2_k3_at_t_plus_3():
    # do(P0) reaches K0 at t+1, K1 at t+2, K2 & K3 at t+3, K4 never.
    env = E3V2Env(env_seed=0, obs_noise_scale=0.0)
    for _ in range(3):
        env.neutral_step()
    snap = env.snapshot()

    def rollout(value):
        env.restore(snap)
        env.apply_action(0, value)
        env.advance()  # warm-up
        return [env.advance() for _ in range(3)]

    hi, lo = rollout(DO_HIGH), rollout(DO_LOW)

    def moved(lag, kpi):
        return abs(hi[lag][kpi] - lo[lag][kpi]) > TOL_ZERO

    assert moved(0, 0) and not moved(0, 1)  # K0 at t+1; K1 not yet
    assert moved(1, 1) and not moved(1, 2) and not moved(1, 3)  # K1 at t+2; K2/K3 not yet
    assert moved(2, 2) and moved(2, 3)  # K2 & K3 at t+3
    for lag in range(3):
        assert not moved(lag, 4)  # K4 non-descendant of P0 at every lag


def test_direct_param_effects_meet_min_eff():
    env = E3V2Env(env_seed=0, obs_noise_scale=0.0)
    for _ in range(3):
        env.neutral_step()
    snap = env.snapshot()

    def do_first_latent(param_id, value):
        env.restore(snap)
        env.apply_action(param_id, value)
        env.advance()  # warm-up
        return env.advance()

    for param_id, child in {0: 0, 1: 2, 2: 3, 3: 4}.items():
        d = abs(do_first_latent(param_id, DO_HIGH)[child] - do_first_latent(param_id, DO_LOW)[child])
        assert abs(d - 0.5) <= TOL_ZERO  # do(0.75)-do(0.25) = 0.5 raw
        assert d / float(E3V2Env.sigma[child]) >= MIN_EFF


def test_true_adjacency_and_decoy_structure():
    true_env = E3V2Env(env_seed=0)
    assert set(true_env.adjacency_edges) == {
        (0, 0), (1, 4), (2, 1), (2, 5), (3, 2), (3, 5), (4, 3)
    }
    decoy = E3V2Env(env_seed=0, truncate_fanout=True)
    dropped = set(true_env.adjacency_edges) - set(decoy.adjacency_edges)
    assert dropped == {(2, 5), (3, 5)}, "decoy drops exactly the K1 fan-out pair"
    # Truncation zeroes K1's contribution to BOTH K2 and K3.
    assert decoy.c25 == 0.0 and decoy.c35 == 0.0


def test_true_adj_matrix_encodes_seven_edges():
    m = E3V2Env(env_seed=0).true_adj_matrix()
    assert m.shape == (9, 9)
    assert int(m.sum()) == 7
    # K1 (row 4+1=5) has its source at K0 (col 4).
    assert m[5, 4] == 1.0
    # K2 (row 6) reads P1 (col 1) and K1 (col 5).
    assert m[6, 1] == 1.0 and m[6, 5] == 1.0


def test_noise_off_latent_equals_observed():
    env = E3V2Env(env_seed=1, obs_noise_scale=0.0)
    for _ in range(3):
        env.neutral_step()
    assert np.allclose(env.latent_kpis(), env.observed_kpis(), atol=0.0)


# --------------------------------------------------------------------------------------
# Invocation smoke test: the documented direct-path command runs green.
# --------------------------------------------------------------------------------------
def test_script_runs_as_documented_command():
    proc = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / "e3_scm_gate.py")],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"gate exited {proc.returncode}:\n{proc.stdout}\n{proc.stderr}"
    assert "RESULT: PASS" in proc.stdout
