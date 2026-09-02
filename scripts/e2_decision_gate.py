"""E2 decision-gap gate — shared-control fan-out (SPEC E2, GATES §2.3/§3.1).

Self-contained sim-level gate (like ``scripts/e1_scm_gate.py``); NO training. Every threshold,
bank rule, decoy rule, margin, and normalization is FROZEN in
``docs/benchmark/GATE_CONTRACT_E2.md`` (reviewer ruling ``review-e2-buildplan.md``) and is not
re-derived here.

Procedure:

1. **Bank (Q5).** For ``env_seed = 0..4095`` (``episode=0``, noise OFF, 3 neutral advances), take
   the committed state and label it by TRUE-K5 response geometry over the 101-point P0 grid:
   positive if ``z_min <= theta5-0.5`` and ``z_max >= theta5+0.5``; negative if ``S5 <= 0.05``.
   Retain the first 32 positive and first 32 negative (bank size exactly 64). Fewer than 32 of
   either class ⇒ preregistration-feasibility FAIL.
2. **Gap (Q6).** With the locked full panel ``{xApp0,xApp1,xApp2,xApp4}`` and objective ``R``,
   select ``a_oracle = argmax_v R_true`` and ``a_decoy = argmax_v R_decoy`` (decoy freezes P0 in
   K5). ``regret_true(a) = max_v R_true − R_true(a)``; ``gap_raw = regret_true(a_decoy) −
   regret_true(a_oracle)``; ``gap_norm = gap_raw / D(s)`` with ``D(s) = max_v R_true − min_v
   R_true``. ``D(s) <= tol_zero`` ⇒ degenerate FAIL.
3. **Accept.** PASS iff ``mean_positive(gap_norm) >= tau_E2`` AND ``mean_negative(gap_norm) <=
   tol``.
4. **Factor-removal control (§3.1).** Drop xApp4/K5 from the panel; the positive-control gap must
   fall to ``<= tol``.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cdd_oran.analysis.v2_regret import (  # noqa: E402
    P0_GRID,
    PanelXApp,
    score_grid,
)
from cdd_oran.envs.v2.e2 import E2V2Env, safe_exp  # noqa: E402

# --- FROZEN protocol constants (GATE_CONTRACT_E2.md) ------------------------------------------
SEED_POOL = range(4096)  # env_seed = 0..4095
EPISODE = 0
NEUTRAL_STEPS = 3
N_PER_CLASS = 32
SHARED_PARAM = 0  # P0
MU5, SIGMA5 = E2V2Env.mean[5], E2V2Env.std[5]
THETA5 = (-25.0 - MU5) / SIGMA5
POS_MARGIN = 0.5
NEG_S5_MAX = 0.05
TAU_E2 = 0.10
TOL = 0.01
TOL_ZERO = 1e-12
FULL_PANEL_IDS = (0, 1, 2, 4)
CONTROL_PANEL_IDS = (0, 1, 2)  # §3.1: xApp4/K5 removed


def build_panel(env: E2V2Env, xapp_ids) -> list[PanelXApp]:
    """Locked panel from the env's xApp maps (single-KPI members reduce to the KPI's stats)."""
    panel = []
    for xid in xapp_ids:
        kpis = env.xapp_kpi_indices[xid]
        mean = float(np.mean([env.mean[k] for k in kpis]))
        std = float(np.mean([env.std[k] for k in kpis]))
        threshold = float(np.mean([env.kpi_thresholds[k] for k in kpis]))
        direction = int(env.directions[kpis[0]])
        panel.append(PanelXApp(tuple(kpis), mean, std, threshold, direction))
    return panel


def _committed_state(env_factory: Callable[[int], E2V2Env], seed: int):
    env = env_factory(seed)
    env.reset(episode=EPISODE)
    for _ in range(NEUTRAL_STEPS):
        env.neutral_step()
    return env.snapshot()


def _k5_true(prev_params: np.ndarray, v: float) -> float:
    """TRUE noiseless K5 mechanism at committed params with do(P0=v) (holds P6,P7)."""
    p6, p7 = float(prev_params[6]), float(prev_params[7])
    return -35.0 * np.exp(-((p7 + v - 25.0) ** 2) / (2.0 * safe_exp(p6) ** 2))


def _label(prev_params: np.ndarray) -> str:
    """'positive' / 'negative' / 'grey' by K5-response geometry over the P0 grid (Q5)."""
    z = np.array([(_k5_true(prev_params, float(v)) - MU5) / SIGMA5 for v in P0_GRID])
    z_min, z_max = float(z.min()), float(z.max())
    if z_min <= THETA5 - POS_MARGIN and z_max >= THETA5 + POS_MARGIN:
        return "positive"
    if (z_max - z_min) <= NEG_S5_MAX:
        return "negative"
    return "grey"


@dataclass
class BankResult:
    positives: list = field(default_factory=list)  # (seed, snap)
    negatives: list = field(default_factory=list)
    feasible: bool = False


def build_bank(env_factory: Callable[[int], E2V2Env] = E2V2Env) -> BankResult:
    res = BankResult()
    for seed in SEED_POOL:
        snap = _committed_state(env_factory, seed)
        prev_params = snap[1]
        label = _label(prev_params)
        if label == "positive" and len(res.positives) < N_PER_CLASS:
            res.positives.append((seed, snap))
        elif label == "negative" and len(res.negatives) < N_PER_CLASS:
            res.negatives.append((seed, snap))
        if len(res.positives) == N_PER_CLASS and len(res.negatives) == N_PER_CLASS:
            break
    res.feasible = len(res.positives) == N_PER_CLASS and len(res.negatives) == N_PER_CLASS
    return res


def _gap_norm_for_state(
    true_env: E2V2Env,
    decoy_env: E2V2Env,
    snap,
    panel: list[PanelXApp],
) -> tuple[float, float]:
    """Return (gap_norm, D) for one state. Raises on degenerate D (<= tol_zero)."""
    scores_true = score_grid(true_env, snap, SHARED_PARAM, panel)
    scores_decoy = score_grid(decoy_env, snap, SHARED_PARAM, panel)

    g_star = float(scores_true.max())
    d = g_star - float(scores_true.min())
    if d <= TOL_ZERO:
        raise ValueError(f"degenerate D(s)={d:.3e} <= tol_zero {TOL_ZERO}")

    a_decoy_idx = int(np.argmax(scores_decoy))  # argmax R_decoy
    # a_oracle = argmax R_true ⇒ regret_true(a_oracle) = 0 by construction.
    regret_decoy = g_star - float(scores_true[a_decoy_idx])
    gap_raw = regret_decoy - 0.0
    return gap_raw / d, d


@dataclass
class GateResult:
    passed: bool = False
    feasible: bool = False
    mean_pos: float = float("nan")
    mean_neg: float = float("nan")
    separation: float = float("nan")
    factor_removal_pos: float = float("nan")
    n_pos: int = 0
    n_neg: int = 0
    failures: list[str] = field(default_factory=list)


def run_e2_decision_gate(
    env_factory: Callable[[int], E2V2Env] = E2V2Env,
    decoy_factory: Callable[[int], E2V2Env] | None = None,
) -> GateResult:
    if decoy_factory is None:
        def decoy_factory(seed: int) -> E2V2Env:
            return E2V2Env(env_seed=seed, decoy_omit_p0_k5=True)

    result = GateResult()
    bank = build_bank(env_factory)
    result.feasible = bank.feasible
    result.n_pos = len(bank.positives)
    result.n_neg = len(bank.negatives)
    if not bank.feasible:
        result.failures.append(
            f"preregistration feasibility: found {result.n_pos} pos / {result.n_neg} neg "
            f"in seed pool (need {N_PER_CLASS}+{N_PER_CLASS})"
        )
        return result

    full_panel = build_panel(env_factory(0), FULL_PANEL_IDS)
    control_panel = build_panel(env_factory(0), CONTROL_PANEL_IDS)

    pos_gaps, neg_gaps, ctrl_pos_gaps = [], [], []
    for seed, snap in bank.positives:
        te, de = env_factory(seed), decoy_factory(seed)
        pos_gaps.append(_gap_norm_for_state(te, de, snap, full_panel)[0])
        ctrl_pos_gaps.append(_gap_norm_for_state(te, de, snap, control_panel)[0])
    for seed, snap in bank.negatives:
        te, de = env_factory(seed), decoy_factory(seed)
        neg_gaps.append(_gap_norm_for_state(te, de, snap, full_panel)[0])

    result.mean_pos = float(np.mean(pos_gaps))
    result.mean_neg = float(np.mean(neg_gaps))
    result.separation = result.mean_pos - result.mean_neg
    result.factor_removal_pos = float(np.mean(ctrl_pos_gaps))

    pass_pos = result.mean_pos >= TAU_E2
    pass_neg = result.mean_neg <= TOL
    # Factor-removal control (GATES §3; GATE_CONTRACT_E2 §7) is MANDATORY: dropping xApp4/K5 from
    # the panel must collapse the positive gap to <= TOL, else the gap does not localize to the
    # shared-control factor. Gate on it, not just compute it (review-e2-impl.md BLOCKER 1).
    pass_factor_removal = result.factor_removal_pos <= TOL
    if not pass_pos:
        result.failures.append(f"mean_positive {result.mean_pos:.4f} < tau_E2 {TAU_E2}")
    if not pass_neg:
        result.failures.append(f"mean_negative {result.mean_neg:.4f} > tol {TOL}")
    if not pass_factor_removal:
        result.failures.append(
            f"factor_removal_pos {result.factor_removal_pos:.4f} > tol {TOL}"
        )
    result.passed = pass_pos and pass_neg and pass_factor_removal
    return result


def main() -> int:
    result = run_e2_decision_gate()
    print("E2 decision-gap gate")
    print(f"  bank feasible: {result.feasible} ({result.n_pos} pos + {result.n_neg} neg)")
    print(f"  mean_positive(gap_norm): {result.mean_pos:.6f}   (tau_E2 = {TAU_E2})")
    print(f"  mean_negative(gap_norm): {result.mean_neg:.6f}   (tol    = {TOL})")
    print(f"  separation (pos - neg):  {result.separation:.6f}   (must be >= 0.09)")
    print(f"  factor-removal pos gap:  {result.factor_removal_pos:.6f}   (must be <= {TOL})")
    if result.failures:
        print("  failures:")
        for msg in result.failures:
            print(f"    - {msg}")
    print(f"RESULT: {'PASS' if result.passed else 'FAIL'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
