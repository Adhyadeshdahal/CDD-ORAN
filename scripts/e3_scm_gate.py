"""E3 SCM gate — deterministic structural-correctness check.

Implements the FROZEN "Standalone SCM gate" protocol from ``docs/benchmark/GATE_CONTRACT_E3.md``
on the **latent noiseless** state (observation noise excluded entirely). It runs BEFORE the E3
decision gate and verifies that the generated latent transitions match the declared depth-three
fan-out cascade ``P0->K0->K1->{K2,K3}`` plus ``P1->K2``, ``P2->K3``, ``P3->K4``.

Mirrors ``scripts/e1_scm_gate.py``. The direct-parentage repair (E1 BLOCKER 1) is applied to
every KPI->KPI edge: a param-ancestor drive alone cannot tell a declared ``K_src -> K_dst`` from
an undeclared ``P_src`` read, so each KPI-parent edge is probed by intervening **directly** on the
latent source KPI ``prev_kpis[src]`` — pushing it OFF the manifold — while holding every param and
every other KPI fixed, and reading the child one ``advance`` later.

Clauses (FROZEN contract "Required checks at all eight states"):

* (a) TIMING/MIN-EFFECT: direct param children at t+1 (P0->K0, P1->K2, P2->K3, P3->K4); and the
  P0 descendant timing K0@t+1, K1@t+2, K2&K3@t+3 — each descendant inert before its first-move
  lag and moving by >= ``min_eff`` at it.
* (b) NON-DESCENDANT controls: every non-descendant KPI inert (<= ``tol_zero``) through
  ``H_desc``; in particular K4 is inert under do(P0).
* (c) DIRECT off-manifold KPI-source parentage for each edge K0->K1, K1->K2, K1->K3: TRUE child
  moves >= ``min_eff``; single-edge shadow inert; direct child-param path invariant where present
  (P1->K2 / P2->K3; K0->K1 has none); identically moved non-parent KPI leaves the child inert.
* (d) CLOSED-FORM: independently computed analytic latent transition matches simulator output on
  neutral, intervention, and off-manifold states.
* (e) ADJACENCY: TRUE == the 7-edge list; each single-edge shadow differs by exactly its one
  intended edge; the decision decoy (truncate_fanout) differs from TRUE by exactly (2,5),(3,5).

The gate is injectable over ``env_factory`` (true env under test) and ``shadow_factory``
(single-edge-ablated env), so mutation tests can feed structurally wrong generators and confirm
the gate REJECTS them (see ``tests/test_e3_scm_gate.py``).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

# Allow direct execution (``python scripts/e3_scm_gate.py``) as well as ``-m``: put the repo root
# on the path before importing the package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cdd_oran.envs.v2.e3 import E3V2Env  # noqa: E402

# --- FROZEN protocol constants (GATE_CONTRACT_E3.md "Standalone SCM gate") ---------------
N_STATES = 8
NEUTRAL_STEPS = 3
DO_LOW = 0.25  # 25th pct of ID range [0, 1]
DO_HIGH = 0.75  # 75th pct of ID range [0, 1]
H_DESC = 3
TOL_ZERO = 1e-12
MIN_EFF = 0.05

# Preregistered OFF-manifold latent-KPI intervention offsets (added to the on-manifold source
# value so the intervened K_src != on-manifold value). Distinct and nonzero.
KPI_DO_LOW = -0.3
KPI_DO_HIGH = 0.3

# Preregistered OFF-manifold (prev_params, prev_kpis) offset vectors for the analytic clause.
# Every entry nonzero, so at each bank state K_i departs its manifold value for all i.
OFF_MANIFOLD_KPI_OFFSETS = [
    np.array([0.30, -0.30, 0.20, -0.20, 0.10], dtype=float),
    np.array([-0.25, 0.35, -0.15, 0.25, -0.20], dtype=float),
]

# Declared structure (contract SCM).
DIRECT_PARAM_CHILD = {0: 0, 1: 2, 2: 3, 3: 4}  # P0->K0, P1->K2, P2->K3, P3->K4

# Held-do descendant first-move lag (index into out; out[j] is the t+1+j response) per param.
# do(P0): K0@t+1, K1@t+2, K2&K3@t+3. do(P1/P2/P3): their single direct child at t+1.
PARAM_DESCENDANT_LAG = {
    0: {0: 0, 1: 1, 2: 2, 3: 2},
    1: {2: 0},
    2: {3: 0},
    3: {4: 0},
}
# Permanent non-descendants (never move within H_desc) per param.
PARAM_NONDESC = {
    0: {4},              # K4 is P0's structural negative control
    1: {0, 1, 3, 4},
    2: {0, 1, 2, 4},
    3: {0, 1, 2, 3},
}

# KPI->KPI edges (src_kpi, dst_kpi).
KPI_EDGES = [(0, 1), (1, 2), (1, 3)]  # K0->K1, K1->K2, K1->K3
# Direct param parent of each KPI-edge child (for the c.iii invariance subcheck); None where the
# child has no direct param path (K1). Values are param indices.
CHILD_PARAM_PARENT = {1: None, 2: 1, 3: 2}
# On-manifold value of each source KPI is the value of this param after 3 neutral advances:
# K0 == P0 and K1 == P0, so both map to param 0.
SRC_ONMANIFOLD_PARAM = {0: 0, 1: 0}
# A non-parent KPI source (identically moved, must leave the child inert) per edge dst.
NONPARENT_SRC = {1: 1, 2: 0, 3: 0}  # dst K1: move K1; dst K2/K3: move K0


@dataclass
class GateResult:
    passed: bool
    clauses: dict[str, bool] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)


def default_env_factory(seed: int) -> E3V2Env:
    """True env under test: the frozen E3 SCM, noise OFF."""
    return E3V2Env(env_seed=seed, obs_noise_scale=0.0)


def default_shadow_factory(seed: int, dst_kpi: int) -> E3V2Env:
    """Single-edge-ablated shadow: zero the one KPI-parent coefficient feeding ``dst_kpi``."""
    if dst_kpi == 1:
        return E3V2Env(env_seed=seed, obs_noise_scale=0.0, c10=0.0)  # drop K0->K1
    if dst_kpi == 2:
        return E3V2Env(env_seed=seed, obs_noise_scale=0.0, c25=0.0)  # drop K1->K2
    if dst_kpi == 3:
        return E3V2Env(env_seed=seed, obs_noise_scale=0.0, c35=0.0)  # drop K1->K3
    raise ValueError(f"no ablatable KPI edge into K{dst_kpi}")


def _build_bank(
    env_factory: Callable[[int], E3V2Env],
) -> list[tuple]:
    bank = []
    for seed in range(N_STATES):
        env = env_factory(seed)
        for _ in range(NEUTRAL_STEPS):
            env.neutral_step()
        bank.append(env.snapshot())
    return bank


def _do_rollout(
    env: E3V2Env, snap: tuple, param_id: int, value: float, horizon: int
) -> list[np.ndarray]:
    """Held-``do(P_i = value)`` from a bank state; latent KPIs at lags t+1..t+horizon.

    One warm-up ``advance`` absorbs the one-step actuation latency, so ``out[j]`` is the ``t+1+j``
    response.
    """
    env.restore(snap)
    env.apply_action(param_id, value)
    env.advance()  # warm-up: absorbs one-step latency
    out = []
    for _ in range(horizon):
        out.append(env.advance())
    return out


def _do_kpi_source(env: E3V2Env, snap: tuple, src_kpi: int, value: float) -> np.ndarray:
    """DIRECT off-manifold intervention on the latent source KPI.

    Sets ``prev_kpis[src_kpi] = value`` while holding params, prev_params, and every OTHER latent
    KPI fixed, then advances exactly one step. No warm-up: a child of a KPI->KPI edge reads
    ``prev_kpis`` directly, so it responds at the very next ``advance``. This distinguishes a real
    ``K_src -> K_dst`` edge (child moves) from a ``P_src``-reading impostor (child inert).
    """
    params, prev_params, prev_kpis = snap[0], snap[1], snap[2]
    pk = prev_kpis.copy()
    pk[src_kpi] = float(value)
    env.restore((params.copy(), prev_params.copy(), pk, *snap[3:]))
    return env.advance()


def _raw_delta(vec_high: np.ndarray, vec_low: np.ndarray, kpi_index: int) -> float:
    return abs(float(vec_high[kpi_index]) - float(vec_low[kpi_index]))


def _std_delta(vec_high: np.ndarray, vec_low: np.ndarray, kpi_index: int) -> float:
    return _raw_delta(vec_high, vec_low, kpi_index) / float(E3V2Env.sigma[kpi_index])


def _analytic_next(prev_params: np.ndarray, prev_kpis: np.ndarray) -> np.ndarray:
    """Closed-form next latent KPI from the FROZEN equations, independent of the env code."""
    p = np.asarray(prev_params, dtype=float)
    k = np.asarray(prev_kpis, dtype=float)
    return np.array([p[0], k[0], p[1] + k[1], p[2] + k[1], p[3]], dtype=float)


def _edges_set(env: E3V2Env) -> set[tuple[int, int]]:
    return set(env.adjacency_edges)


def run_e3_scm_gate(
    env_factory: Callable[[int], E3V2Env] = default_env_factory,
    shadow_factory: Callable[[int, int], E3V2Env] = default_shadow_factory,
) -> GateResult:
    """Run the 5-clause E3 SCM gate.

    ``env_factory(seed)`` builds the true env under test (noise OFF). ``shadow_factory(seed,
    dst_kpi)`` builds the single-edge-ablated shadow with the KPI-parent coefficient feeding
    ``dst_kpi`` zeroed. Both injectable so mutation tests can prove REJECTION of structurally
    wrong generators.
    """
    result = GateResult(passed=True)

    bank = _build_bank(env_factory)
    env = env_factory(0)

    clause_a = True  # direct-param + P0 descendant timing (min-effect at first-move lag)
    clause_b = True  # non-descendant negative controls
    clause_c = True  # DIRECT off-manifold KPI-source parentage + controls
    clause_d = True  # analytic closed-form == rolled-out latent (neutral/intervention/off)
    clause_e = True  # adjacency structural identity (true / shadows / decoy)

    for si, snap in enumerate(bank):
        params, prev_params, prev_kpis = snap[0], snap[1], snap[2]
        roll_high = {p: _do_rollout(env, snap, p, DO_HIGH, H_DESC) for p in range(4)}
        roll_low = {p: _do_rollout(env, snap, p, DO_LOW, H_DESC) for p in range(4)}

        # (a) descendant timing + min-effect: inert before first-move lag, >= min_eff at it.
        for param_id, lags in PARAM_DESCENDANT_LAG.items():
            for kpi, first in lags.items():
                for lag in range(first):
                    pre = _raw_delta(roll_high[param_id][lag], roll_low[param_id][lag], kpi)
                    if pre > TOL_ZERO:
                        clause_a = False
                        result.failures.append(
                            f"[a] state {si}: do(P{param_id}) moved descendant K{kpi} EARLY at "
                            f"lag {lag + 1} by {pre:.3e} (> tol; first-move is t+{first + 1})"
                        )
                eff = _std_delta(roll_high[param_id][first], roll_low[param_id][first], kpi)
                if eff < MIN_EFF:
                    clause_a = False
                    result.failures.append(
                        f"[a] state {si}: do(P{param_id}) -> K{kpi} effect {eff:.4f} < min_eff "
                        f"{MIN_EFF} at first-move lag t+{first + 1}"
                    )

        # (b) non-descendant negative controls: no movement at any lag up to H_desc.
        for param_id in range(4):
            for lag in range(H_DESC):
                for kpi in PARAM_NONDESC[param_id]:
                    delta = _raw_delta(roll_high[param_id][lag], roll_low[param_id][lag], kpi)
                    if delta > TOL_ZERO:
                        clause_b = False
                        result.failures.append(
                            f"[b] state {si}: do(P{param_id}) moved non-descendant K{kpi} by "
                            f"{delta:.3e} at lag {lag + 1} (> tol {TOL_ZERO})"
                        )

        # (c) DIRECT off-manifold KPI-source parentage for each KPI->KPI edge.
        for src_kpi, dst_kpi in KPI_EDGES:
            shadow = shadow_factory(0, dst_kpi)
            src_on = float(prev_params[SRC_ONMANIFOLD_PARAM[src_kpi]])
            v_lo = src_on + KPI_DO_LOW
            v_hi = src_on + KPI_DO_HIGH

            # (c.i) TRUE SCM: direct off-manifold source move drives the child at the next step.
            child_hi = _do_kpi_source(env, snap, src_kpi, v_hi)
            child_lo = _do_kpi_source(env, snap, src_kpi, v_lo)
            eff_true = _std_delta(child_hi, child_lo, dst_kpi)
            # (c.ii) SHADOW: same direct source move leaves the child inert (edge coeff gone).
            s_hi = _do_kpi_source(shadow, snap, src_kpi, v_hi)
            s_lo = _do_kpi_source(shadow, snap, src_kpi, v_lo)
            eff_shadow = _raw_delta(s_hi, s_lo, dst_kpi)

            if eff_true < MIN_EFF:
                clause_c = False
                result.failures.append(
                    f"[c.i] state {si}: direct K{src_kpi}->K{dst_kpi} off-manifold effect "
                    f"{eff_true:.4f} < min_eff {MIN_EFF} (declared edge absent or reads a param)"
                )
            if eff_shadow > TOL_ZERO:
                clause_c = False
                result.failures.append(
                    f"[c.ii] state {si}: K{src_kpi}->K{dst_kpi} shadow (ablated) effect "
                    f"{eff_shadow:.3e} > tol {TOL_ZERO}"
                )

            # (c.iii) direct child-param path invariant under ablation, where present.
            parent_param = CHILD_PARAM_PARENT[dst_kpi]
            if parent_param is not None:
                direct_true = _raw_delta(
                    roll_high[parent_param][0], roll_low[parent_param][0], dst_kpi
                )
                sd_high = _do_rollout(shadow, snap, parent_param, DO_HIGH, H_DESC)
                sd_low = _do_rollout(shadow, snap, parent_param, DO_LOW, H_DESC)
                direct_shadow = _raw_delta(sd_high[0], sd_low[0], dst_kpi)
                if abs(direct_shadow - direct_true) > TOL_ZERO:
                    clause_c = False
                    result.failures.append(
                        f"[c.iii] state {si}: P{parent_param}->K{dst_kpi} direct effect changed "
                        f"by ablation |{direct_shadow:.6f}-{direct_true:.6f}| > tol"
                    )

            # (c.neg) identically moved non-parent KPI leaves the child inert (TRUE SCM).
            other_src = NONPARENT_SRC[dst_kpi]
            o_on = float(prev_params[SRC_ONMANIFOLD_PARAM[other_src]])
            o_hi = _do_kpi_source(env, snap, other_src, o_on + KPI_DO_HIGH)
            o_lo = _do_kpi_source(env, snap, other_src, o_on + KPI_DO_LOW)
            delta = _raw_delta(o_hi, o_lo, dst_kpi)
            if delta > TOL_ZERO:
                clause_c = False
                result.failures.append(
                    f"[c.neg] state {si}: direct move of non-parent K{other_src} moved "
                    f"K{dst_kpi} by {delta:.3e} (> tol) — undeclared cross-KPI edge"
                )

        # (d) analytic closed-form == rolled-out latent — neutral, intervention, off-manifold.
        analytic = _analytic_next(prev_params, prev_kpis)
        env.restore(snap)
        rolled = env.advance()
        if np.max(np.abs(analytic - rolled)) > TOL_ZERO:
            clause_d = False
            result.failures.append(
                f"[d] state {si} (neutral): analytic {analytic} != rolled {rolled}"
            )
        # intervention state: one committed do(P0) then compare the closed form to the simulator.
        env.restore(snap)
        env.apply_action(0, DO_HIGH)
        env.advance()  # commit the acted param
        iv_params, iv_kpis = env.prev_params.copy(), env.prev_kpis.copy()
        analytic_iv = _analytic_next(iv_params, iv_kpis)
        rolled_iv = env.advance()
        if np.max(np.abs(analytic_iv - rolled_iv)) > TOL_ZERO:
            clause_d = False
            result.failures.append(
                f"[d] state {si} (intervention): analytic {analytic_iv} != rolled {rolled_iv}"
            )
        for oi, offset in enumerate(OFF_MANIFOLD_KPI_OFFSETS):
            off_kpis = prev_kpis + offset
            off_snap = (params.copy(), prev_params.copy(), off_kpis, *snap[3:])
            analytic_off = _analytic_next(prev_params, off_kpis)
            env.restore(off_snap)
            rolled_off = env.advance()
            if np.max(np.abs(analytic_off - rolled_off)) > TOL_ZERO:
                clause_d = False
                result.failures.append(
                    f"[d] state {si} (off-manifold #{oi}): analytic {analytic_off} != "
                    f"rolled {rolled_off}"
                )

    # (e) adjacency structural identity — env-level, not per-state.
    true_edges = _edges_set(env_factory(0))
    expected_true = set(E3V2Env.TRUE_ADJACENCY)
    if true_edges != expected_true:
        clause_e = False
        result.failures.append(
            f"[e] TRUE adjacency {sorted(true_edges)} != frozen 7-edge list "
            f"{sorted(expected_true)}"
        )
    # Each single-edge shadow must differ from TRUE by exactly its one intended edge.
    shadow_expected = {1: (1, 4), 2: (2, 5), 3: (3, 5)}
    for dst_kpi, edge in shadow_expected.items():
        diff = true_edges ^ _edges_set(shadow_factory(0, dst_kpi))
        if diff != {edge}:
            clause_e = False
            result.failures.append(
                f"[e] shadow(dst K{dst_kpi}) differs from TRUE by {sorted(diff)}, expected "
                f"exactly {{{edge}}}"
            )
    # The decision decoy (truncate_fanout) must differ from TRUE by exactly (2,5),(3,5).
    decoy_diff = expected_true ^ _edges_set(E3V2Env(env_seed=0, truncate_fanout=True))
    if decoy_diff != set(E3V2Env.TRUNCATED_OMITTED_EDGES):
        clause_e = False
        result.failures.append(
            f"[e] decision decoy differs from TRUE by {sorted(decoy_diff)}, expected exactly "
            f"{sorted(E3V2Env.TRUNCATED_OMITTED_EDGES)}"
        )

    result.clauses = {"a": clause_a, "b": clause_b, "c": clause_c, "d": clause_d, "e": clause_e}
    result.passed = all(result.clauses.values())
    return result


def main() -> int:
    result = run_e3_scm_gate()
    print("E3 SCM gate")
    for clause, ok in result.clauses.items():
        print(f"  clause ({clause}): {'PASS' if ok else 'FAIL'}")
    if result.failures:
        print("  failures:")
        for msg in result.failures[:50]:
            print(f"    - {msg}")
    print(f"RESULT: {'PASS' if result.passed else 'FAIL'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
