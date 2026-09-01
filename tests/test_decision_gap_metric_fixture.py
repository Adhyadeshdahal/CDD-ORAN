"""Synthetic decision-gap metric fixture (WP4, review B4).

A self-contained toy dynamic system with a **known closed-form finite-horizon oracle**
where cumulative regret is analytically computable.  No real env, no harness, no evaluator
touching.  Reproduces the SEMANTICS §1.5 point that summed per-step myopic regret does
NOT equal cumulative regret.

Toy system (SEMANTICS §1.5 two-KPI chain, R(k)=k^B, direction=maximize):

    k^A_t = p_{t-1}
    k^B_t = p_{t-1} − 2 · k^A_{t-1}
    reset: p_0 = 0, k^A_0 = k^B_0 = 0
    actions: p ∈ {0, 1}
    horizon H = 2

Analytical results (undiscounted, scored window k_2, k_3):

    Oracle action sequence:  (0, 1)  →  G_oracle = 1
    Planner (bad) sequence:  (1, 0)  →  G_planner = −1
    Cumulative regret:       G_oracle − G_planner = 2

    Myopic per-step oracle:  (1, 1)  →  G_myopic = 0
    Summed myopic regret:    0       (each step individually optimal)
    Actual myopic regret:    G_oracle − G_myopic = 1   (≠ 0)

This demonstrates that per-step decision_regret summed across steps is NOT cumulative
regret (SEMANTICS §1.5, B2).
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Toy dynamics  (no imports from cdd_oran — fully self-contained)
# ---------------------------------------------------------------------------

ACTION_GRID = [0.0, 1.0]
HORIZON = 2


def _step(kA_prev: float, kB_prev: float, p: float) -> tuple[float, float]:
    """One-step latent transition: k_t = f(p_{t-1}, k_{t-1})."""
    kA = p
    kB = p - 2.0 * kA_prev
    return kA, kB


def _rollout(actions: tuple[float, ...], p0: float = 0.0, kA0: float = 0.0, kB0: float = 0.0):
    """Return (scored_kB_window, G) for a given action sequence.

    Follows SEMANTICS §1.1 canonical call list for H=2:
        apply_action(a1); advance()  →  warm-up (k1, UNSCORED)
        apply_action(a2); advance()  →  k2 (SCORED)
        advance()          terminal  →  k3 (SCORED)

    With the toy dynamics the warm-up k1 = g(p0) is deterministic and equal
    to the pre-decision state, so the scored window is k2, k3 produced by the
    two apply_action+advance pairs and the terminal advance.
    """
    p1, p2 = actions[0], actions[1]

    # warm-up advance (k1, UNSCORED)
    _kA1, _kB1 = _step(kA0, kB0, p0)

    # scored advance 1  (k2 = g(p1, k1))
    kA2, kB2 = _step(kA_prev=_kA1, kB_prev=_kB1, p=p1)

    # scored advance 2  (k3 = g(p2, k2))
    kA3, kB3 = _step(kA_prev=kA2, kB_prev=kB2, p=p2)

    G = kB2 + kB3  # R(k) = k^B, γ = 1
    return (kB2, kB3), G


# ---------------------------------------------------------------------------
# Analytical ground truth
# ---------------------------------------------------------------------------

def _oracle_return() -> float:
    """Enumerate all action pairs, return max G."""
    return max(_rollout((a1, a2))[1] for a1 in ACTION_GRID for a2 in ACTION_GRID)


def _myopic_greedy_walk() -> tuple[tuple[float, ...], float]:
    """Independently compute the myopic per-step policy AND its summed static regret.

    Walks the toy system one scored decision at a time. At each decision state we
    ENUMERATE the feasible actions, compute each action's local (H=1 static) return
    ``k^B`` for that step, take the local best and the greedily-selected action's return,
    accumulate ``local_regret = local_best − selected_return``, then commit the chosen
    action and advance. The enumerated action sequence is returned rather than asserted
    against a constant. Summed static regret is 0 by construction (greedy selects the local
    argmax) — the point of §1.5 is that this 0 is NOT the cumulative dynamic regret.
    """
    # warm-up advance (k1, UNSCORED): committed pre-decision state
    kA, kB = _step(0.0, 0.0, 0.0)
    actions: list[float] = []
    summed_regret = 0.0
    for _ in range(HORIZON):
        local_returns = {a: _step(kA, kB, a)[1] for a in ACTION_GRID}
        local_best = max(local_returns.values())
        chosen = max(ACTION_GRID, key=lambda a: local_returns[a])
        summed_regret += local_best - local_returns[chosen]
        actions.append(chosen)
        kA, kB = _step(kA, kB, chosen)  # commit chosen action, advance scored step
    return tuple(actions), summed_regret


# Known values
ORACLE_ACTION = (0.0, 1.0)
PLANNER_ACTION = (1.0, 0.0)

G_ORACLE = 1.0
G_PLANNER = -1.0

CUMULATIVE_REGRET = G_ORACLE - G_PLANNER  # 2.0
MYOPIC_REGRET_ACTUAL = 1.0  # G_oracle − G_myopic, verified independently below


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReferenceMetricMatchesKnownValues:
    """The reference implementation must match the analytical oracle."""

    def test_oracle_return(self):
        """Oracle return over H=2 on the toy system equals the known value."""
        _, G = _rollout(ORACLE_ACTION)
        assert G == pytest.approx(G_ORACLE)

    def test_planner_return(self):
        _, G = _rollout(PLANNER_ACTION)
        assert G == pytest.approx(G_PLANNER)

    def test_cumulative_regret_matches_known(self):
        """Cumulative regret (§3) = G_oracle − G_planner = 2."""
        _, G_oracle = _rollout(ORACLE_ACTION)
        _, G_planner = _rollout(PLANNER_ACTION)
        regret = G_oracle - G_planner
        assert regret == pytest.approx(CUMULATIVE_REGRET)

    def test_oracle_enumeration(self):
        """Exhaustive enumeration finds the same oracle return."""
        assert _oracle_return() == pytest.approx(G_ORACLE)

    def test_myopic_per_step_action(self):
        """Myopic greedy walk enumerates (1, 1) — each step individually optimal."""
        myopic_actions, _ = _myopic_greedy_walk()
        assert myopic_actions == (1.0, 1.0)
        # each chosen action is the local argmax, so its scored kB is the per-step best
        (kB2, kB3), _ = _rollout(myopic_actions)
        assert kB2 == pytest.approx(1.0)  # p1=1 maximizes kB2
        assert kB3 == pytest.approx(-1.0)  # p2=1 maximizes kB3 given p1=1


class TestMyopicRegretMismatch:
    """SEMANTICS §1.5: summed per-step myopic regret ≠ cumulative regret.

    The per-step decision_regret is 0 at each step (myopic oracle is locally
    optimal), but the true finite-horizon regret of the myopic sequence is 1.
    This proves summed per-step regret cannot stand in for cumulative regret.
    """

    def test_myopic_regret_is_not_zero(self):
        """Dynamic regret of the enumerated myopic sequence = G_oracle − G_myopic = 1."""
        myopic_actions, _ = _myopic_greedy_walk()
        _, G_myopic = _rollout(myopic_actions)
        actual_regret = G_ORACLE - G_myopic
        assert actual_regret == pytest.approx(MYOPIC_REGRET_ACTUAL)

    def test_summed_myopic_regret_is_zero_but_dynamic_regret_is_not(self):
        """Independently computed summed static regret == 0 WHILE the independently
        computed cumulative (dynamic) regret is 2 (planner) / 1 (myopic vs oracle)."""
        myopic_actions, summed_static_regret = _myopic_greedy_walk()

        # summed per-step static regret, computed by enumeration, is exactly 0
        assert summed_static_regret == pytest.approx(0.0)

        # dynamic regrets, independently computed via full rollouts, are NOT 0
        _, G_oracle = _rollout(ORACLE_ACTION)
        _, G_planner = _rollout(PLANNER_ACTION)
        _, G_myopic = _rollout(myopic_actions)
        planner_dynamic_regret = G_oracle - G_planner
        myopic_dynamic_regret = G_oracle - G_myopic

        assert planner_dynamic_regret == pytest.approx(2.0)
        assert myopic_dynamic_regret == pytest.approx(1.0)

        # the crux (§1.5, B2): summed static regret cannot stand in for dynamic regret
        assert summed_static_regret != pytest.approx(myopic_dynamic_regret)

    def test_myopic_action_is_not_oracle(self):
        """Enumerated myopic sequence (1,1) differs from oracle sequence (0,1)."""
        myopic_actions, _ = _myopic_greedy_walk()
        assert myopic_actions != ORACLE_ACTION
