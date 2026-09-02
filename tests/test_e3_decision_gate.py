"""E3 decision-gap gate pytest (sim-level, NO training).

Real computed assertions against the FROZEN contract (``docs/benchmark/GATE_CONTRACT_E3.md``):

(a) the bank reproduces the exact 32 reviewer seeds;
(b) the vectorized 101^3 oracle equals actual clone rollouts on a random sample (tol_zero) — the
    correctness anchor for the array shortcut;
(c) the truncated control gap ``gap_T_norm`` is 0 structurally (greedy == exhaustive on the
    coupling-free truncated model);
(d) FH > FM in RAW cumulative R on the anchor seeds (horizon value is real);
(e) acceptance gates on BOTH clauses — injecting a non-truncated "truncated" model makes
    ``gap_T_norm > tol`` and forces ``passed`` False and a nonzero exit;
(f) H=3 one-step latency + latent == observed at noise 0 on E3V2Env.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cdd_oran.envs.v2.e3 import E3V2Env
from cdd_oran.planners.sequence import _rollout_sequence_scored
from scripts.e3_decision_gate import (
    _R,
    GRID,
    N_BANK,
    TOL,
    TOL_ZERO,
    build_bank,
    run_e3_decision_gate,
    vectorized_G,
)

_EXPECTED_SEEDS = [
    16, 62, 81, 99, 101, 117, 121, 132, 133, 145, 175, 176, 186, 196, 208, 217,
    223, 224, 238, 252, 270, 312, 355, 365, 370, 373, 375, 386, 401, 414, 415, 436,
]


@pytest.fixture(scope="module")
def gate_result():
    return run_e3_decision_gate()


# --- (a) bank -------------------------------------------------------------------------------
def test_bank_reproduces_exact_seeds():
    bank = build_bank(E3V2Env)
    assert bank.feasible
    assert len(bank.seeds) == N_BANK
    assert bank.seeds == _EXPECTED_SEEDS


# --- (b) vectorized oracle == clone rollout -------------------------------------------------
def test_vectorized_oracle_matches_clone_rollout():
    bank = build_bank(E3V2Env)
    snap = bank.snaps[0]  # seed 16
    G = vectorized_G(snap, 1.0, 1.0, 1.0)  # full model coeffs

    def clone_G(indices):
        actions = [float(GRID[j]) for j in indices]
        factory = lambda: E3V2Env(truncate_fanout=False)  # noqa: E731
        scored = _rollout_sequence_scored(factory, snap, 0, actions)
        return float(sum(_R(k) for k in scored))

    rng = np.random.default_rng(0)
    n = 250
    max_err = 0.0
    for _ in range(n):
        idx = tuple(int(x) for x in rng.integers(0, len(GRID), size=3))
        err = abs(float(G[idx]) - clone_G(idx))
        max_err = max(max_err, err)
    assert max_err <= TOL_ZERO, f"vectorized vs clone max abs err {max_err:.3e} > {TOL_ZERO}"


# --- (c) truncated control is structurally zero ---------------------------------------------
def test_truncated_control_gap_is_zero(gate_result):
    assert gate_result.feasible
    # every per-state gap_T_norm must be ~0 (greedy == exhaustive on the separable truncated model)
    per_state_gapT = [abs(s["gap_T_norm"]) for s in gate_result.per_state]
    assert max(per_state_gapT) <= TOL_ZERO
    assert gate_result.mean_abs_gap_T_norm <= TOL_ZERO


# --- (d) FH > FM raw cumulative R on anchor seeds -------------------------------------------
def test_fh_beats_fm_raw_on_anchor_seeds(gate_result):
    by_seed = {s["seed"]: s for s in gate_result.per_state}
    anchors = [16, 62, 81]
    positive = [a for a in anchors if by_seed[a]["gap_H"] > 0.0]
    assert len(positive) >= 2, f"FH did not beat FM on >=2 anchors: {[(a, by_seed[a]['gap_H']) for a in anchors]}"
    # raw gap is a genuine cumulative-R difference, not a normalization artifact
    for a in positive:
        assert by_seed[a]["gap_H"] == pytest.approx(
            by_seed[a]["G_FH"] - by_seed[a]["G_FM"], abs=TOL_ZERO
        )


# --- per-state diagnostics completeness (GATE_CONTRACT_E3.md:171-174) ------------------------
def test_per_state_records_all_mandated_diagnostics(gate_result):
    assert len(gate_result.per_state) == N_BANK
    required = {
        "seed", "G_star", "D3",
        "seq_FH", "seq_FM", "seq_TH", "seq_TM",
        "G_FH", "G_FM", "G_TH", "G_TM",
        "regret_FH", "regret_FM", "regret_TH", "regret_TM",
        "gap_H", "gap_H_norm", "gap_T", "gap_T_norm",
        "fh_vs_th", "fh_vs_th_norm",
    }
    for s in gate_result.per_state:
        assert required.issubset(s.keys()), f"missing {required - set(s.keys())}"
        for arm in ("FH", "FM", "TH", "TM"):
            assert len(s[f"seq_{arm}"]) == 3  # H=3 emitted P0 values
            # regret is the unclamped G_star - G_arm
            assert s[f"regret_{arm}"] == pytest.approx(s["G_star"] - s[f"G_{arm}"], abs=TOL_ZERO)
        # normalized quantities consistent with their raw + D3
        assert s["gap_H_norm"] == pytest.approx(s["gap_H"] / s["D3"], abs=TOL_ZERO)
        assert s["gap_T_norm"] == pytest.approx(s["gap_T"] / s["D3"], abs=TOL_ZERO)
        assert s["fh_vs_th"] == pytest.approx(s["G_FH"] - s["G_TH"], abs=TOL_ZERO)


# --- (e) acceptance gates on BOTH clauses ---------------------------------------------------
def test_acceptance_gates_on_truncated_control():
    """Inject a non-truncated 'truncated' model: gap_T_norm becomes gap_H_norm > tol, so the
    factor-removal clause must fail and the whole gate must reject (passed False)."""
    full = lambda: E3V2Env(truncate_fanout=False)  # noqa: E731
    result = run_e3_decision_gate(E3V2Env, full_factory=full, trunc_factory=full)
    assert result.feasible
    assert result.mean_abs_gap_T_norm > TOL, "test setup: injected control should exceed tol"
    assert result.passed is False
    assert any("gap_T_norm" in m for m in result.failures)


def test_cli_exit_status_gates_on_both_clauses():
    """SUBPROCESS check: the true gate exits 0; the injected-bad-control path exits nonzero.

    Proves the BOTH-clause acceptance is wired into the process exit status, not just `passed`."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "e3_decision_gate.py"

    ok = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert ok.returncode == 0, f"true gate should exit 0, got {ok.returncode}\n{ok.stdout[-500:]}"
    assert "RESULT: PASS" in ok.stdout

    bad = subprocess.run(
        [sys.executable, str(script), "--inject-bad-control"], capture_output=True, text=True
    )
    assert bad.returncode != 0, f"injected-bad-control should exit nonzero, got {bad.returncode}"
    assert "RESULT: FAIL" in bad.stdout


# --- (f) env invariants ---------------------------------------------------------------------
def test_h3_latency():
    """apply_action's effect surfaces one advance later (SEMANTICS §1.1)."""
    env = E3V2Env(env_seed=16)
    env.reset(episode=0)
    for _ in range(3):
        env.neutral_step()
    snap = env.snapshot()
    committed_p0 = float(snap[1][0])

    env.restore(snap)
    env.apply_action(0, committed_p0 + 0.5)
    k1 = env.advance()  # warm-up: reflects pre-decision P0
    k2_moved = env.advance()  # terminal: reflects the moved P0 in K0

    env.restore(snap)
    k1_neutral = env.advance()

    assert np.allclose(k1, k1_neutral), "warm-up k1 must not reflect the applied action"
    assert np.isclose(k1[0], committed_p0), "K0 at warm-up = pre-decision P0"
    assert np.isclose(k2_moved[0], committed_p0 + 0.5), "K0 at terminal = moved P0"


def test_latent_equals_observed_at_noise_off():
    env = E3V2Env(env_seed=81, obs_noise_scale=0.0)
    env.reset(episode=0)
    env.neutral_step()
    assert np.array_equal(env.latent_kpis(), env.observed_kpis())


def test_scm_prerequisite_blocks_decision(monkeypatch):
    """The authoritative decision-gate entry point runs the SCM-correctness gate FIRST and refuses
    to report a decision PASS when the prerequisite fails (GATE_CONTRACT_E3.md). A user cannot run
    the decision gate alone and receive PASS on a structurally wrong env."""
    import scripts.e3_decision_gate as e3d

    class _FailingSCM:
        passed = False
        failures = ["injected SCM failure"]

    monkeypatch.setattr(e3d, "run_e3_scm_gate", lambda *a, **k: _FailingSCM())
    assert e3d.main([]) == 1
