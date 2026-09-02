"""E1 SCM gate — deterministic structural-correctness check (review MAJOR 8).

Implements the LOCKED "E1 SCM-gate protocol" from ``docs/benchmark/SPEC.md`` on the
**latent noiseless** state (observation noise excluded entirely). E1 is exempt from the
decision-gap and factor-removal gates; this gate only verifies that the generated latent
transitions match the declared graph.

**Direct-parentage repair (review BLOCKER 1).** A param-ancestor drive alone cannot tell a
declared ``K0 -> K2`` from an undeclared direct ``P0 -> K2``: after the bank equilibrates on
the manifold ``K0 = P0``, both move ``K2`` when ``P0`` moves, and a shadow that zeroes the
KPI coefficient removes both paths. The repaired clause (c) therefore intervenes **directly**
on the latent source KPI ``prev_kpis[src]`` — pushing it OFF the manifold (``K_src != P_src``)
while holding every param and every other KPI fixed — and reads the child one ``advance``
later. A real KPI->KPI edge moves the child; a ``P_src``-reading impostor does not, because
its params are unchanged. Clause (d) likewise evaluates the closed form on preregistered
OFF-manifold states, not only neutral on-manifold bank states.

Protocol quantities (all preregistered in SPEC.md, nothing tuned here):

* State bank: 8 latent states from ``env_seed in {0..7}``, 3 neutral (no-op) steps, noise OFF.
* do-values: 25th / 75th percentile of each param's ID range (0.25 / 0.75).
* Direct-child effect at lag ``t+1``; KPI->KPI descendant at lag ``t+2`` (``H_desc = 2``).
* ``tol_zero = 1e-6`` (raw equality tolerance); ``min_eff = 0.05`` (standardized units).
* Direct off-manifold KPI-source intervention for every KPI->KPI edge, plus its negative
  control and the true-vs-shadow direct-param-path invariance.
* Non-parent / non-descendant negative controls.
* 4-clause acceptance rule (a)-(d).

The gate is injectable over ``env_factory`` (true env under test) and ``shadow_factory``
(edge-ablated env), so mutation tests can feed structurally wrong generators and confirm the
gate REJECTS them (see ``tests/test_e1_scm_gate.py``).

One-step latency (SEMANTICS §1.1): an intervention applied via ``apply_action`` is absorbed
by one warm-up ``advance`` before the responding KPI appears, so a direct child of the
intervened param shows at the first post-warm-up ``advance`` (``t+1``) and a KPI->KPI
descendant at the second (``t+2``). A **direct** latent-KPI intervention needs no warm-up:
it sets ``prev_kpis[src]`` in place, so the child responds at the very next ``advance``.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

# Allow direct execution (``python scripts/e1_scm_gate.py``) as well as ``-m`` (review
# MINOR 7): put the repo root on the path before importing the package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cdd_oran.envs.v2.e1 import E1V2Env  # noqa: E402

# --- LOCKED protocol constants (SPEC.md E1 SCM-gate protocol) ---------------------
N_STATES = 8
NEUTRAL_STEPS = 3
DO_LOW = 0.25  # 25th pct of ID range [0, 1]
DO_HIGH = 0.75  # 75th pct of ID range [0, 1]
H_DESC = 2
TOL_ZERO = 1e-6
MIN_EFF = 0.05

# Preregistered OFF-manifold latent-KPI intervention offsets (added to the on-manifold
# source value ``P_src`` so the intervened ``K_src != P_src``). Distinct and nonzero so the
# lo/hi source values are off manifold AND differ from each other.
KPI_DO_LOW = -0.3
KPI_DO_HIGH = 0.3

# Preregistered OFF-manifold ``(prev_params, prev_kpis)`` offset vectors for the analytic
# clause. Every entry is nonzero, so at each bank state ``K_i != P_i`` for all i (in
# particular K0 != P0 and K1 != P1), off the manifold the neutral bank sits on.
OFF_MANIFOLD_KPI_OFFSETS = [
    np.array([0.30, -0.30, 0.20, -0.20], dtype=float),
    np.array([-0.25, 0.35, -0.15, 0.25], dtype=float),
]

# Declared structure (SPEC.md E1).
DIRECT_PARAM_CHILD = {0: 0, 1: 1, 2: 2, 3: 3}  # P_i -> K_i
KPI_EDGES = {0: 2, 1: 3}  # K0 -> K2, K1 -> K3  (source_kpi -> dst_kpi)
KPI_ANCESTOR_PARAM = {0: 0, 1: 1}  # to drive source KPI K_s, intervene on P_{ancestor}
# Descendants of each param (direct child at lag 1, KPI-child at lag 2).
PARAM_DESCENDANTS = {0: {0, 2}, 1: {1, 3}, 2: {2}, 3: {3}}


@dataclass
class GateResult:
    passed: bool
    clauses: dict[str, bool] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)


def default_env_factory(seed: int) -> E1V2Env:
    """True env under test: the locked E1 SCM, noise OFF."""
    return E1V2Env(env_seed=seed, obs_noise_scale=0.0)


def default_shadow_factory(seed: int, dst_kpi: int) -> E1V2Env:
    """Edge-ablated shadow: zero the KPI-parent coefficient feeding ``dst_kpi``."""
    if dst_kpi == 2:
        return E1V2Env(env_seed=seed, obs_noise_scale=0.0, b2=0.0)
    if dst_kpi == 3:
        return E1V2Env(env_seed=seed, obs_noise_scale=0.0, b3=0.0)
    raise ValueError(f"no ablatable KPI edge into K{dst_kpi}")


# A committed env snapshot: (params, prev_params, prev_kpis, episode, time). See E1V2Env.snapshot.
Snapshot = tuple[np.ndarray, np.ndarray, np.ndarray, int, int]


def _build_bank(
    env_factory: Callable[[int], E1V2Env],
) -> list[Snapshot]:
    bank = []
    for seed in range(N_STATES):
        env = env_factory(seed)
        for _ in range(NEUTRAL_STEPS):
            env.neutral_step()
        bank.append(env.snapshot())
    return bank


def _do_rollout(
    env: E1V2Env,
    snap: Snapshot,
    param_id: int,
    value: float,
    horizon: int,
) -> list[np.ndarray]:
    """Held-``do(P_i = value)`` from a bank state; returns latent KPIs at lags 1..horizon.

    One warm-up ``advance`` absorbs the one-step actuation latency, so ``out[0]`` is the
    ``t+1`` (direct-child) response and ``out[1]`` the ``t+2`` (descendant) response.
    """
    env.restore(snap)
    env.apply_action(param_id, value)
    env.advance()  # warm-up: absorbs one-step latency
    out = []
    for _ in range(horizon):
        out.append(env.advance())
    return out


def _do_kpi_source(
    env: E1V2Env,
    snap: Snapshot,
    src_kpi: int,
    value: float,
) -> np.ndarray:
    """DIRECT off-manifold intervention on the latent source KPI.

    Sets ``prev_kpis[src_kpi] = value`` while holding params, prev_params, and every OTHER
    latent KPI fixed, then advances exactly one step and returns the resulting latent KPIs.
    No warm-up: the child of a KPI->KPI edge reads ``prev_kpis`` directly, so it responds at
    the very next ``advance``. This is what distinguishes a real ``K_src -> K_dst`` edge
    (child moves) from a ``P_src``-reading impostor (child inert, params unchanged).
    """
    params, prev_params, prev_kpis = snap[0], snap[1], snap[2]
    pk = prev_kpis.copy()
    pk[src_kpi] = float(value)
    # rebuild the full snapshot (preserving the logical coordinates) with only prev_kpis swapped
    env.restore((params.copy(), prev_params.copy(), pk, *snap[3:]))
    return env.advance()


def _raw_delta(vec_high: np.ndarray, vec_low: np.ndarray, kpi_index: int) -> float:
    return abs(float(vec_high[kpi_index]) - float(vec_low[kpi_index]))


def _std_delta(vec_high: np.ndarray, vec_low: np.ndarray, kpi_index: int) -> float:
    return _raw_delta(vec_high, vec_low, kpi_index) / float(E1V2Env.sigma[kpi_index])


def _analytic_next(prev_params: np.ndarray, prev_kpis: np.ndarray) -> np.ndarray:
    """Closed-form next latent KPI from the LOCKED equations, independent of the env code."""
    p = np.asarray(prev_params, dtype=float)
    k = np.asarray(prev_kpis, dtype=float)
    return np.array([p[0], p[1], p[2] + 0.5 * k[0], p[3] + 0.5 * k[1]], dtype=float)


def run_e1_scm_gate(
    env_factory: Callable[[int], E1V2Env] = default_env_factory,
    shadow_factory: Callable[[int, int], E1V2Env] = default_shadow_factory,
) -> GateResult:
    """Run the 4-clause E1 SCM gate.

    ``env_factory(seed)`` builds the true env under test (noise OFF). ``shadow_factory(seed,
    dst_kpi)`` builds the edge-ablated shadow with the KPI-parent coefficient feeding
    ``dst_kpi`` zeroed. Both are injectable so mutation tests can prove REJECTION of
    structurally wrong generators.
    """
    result = GateResult(passed=True)

    bank = _build_bank(env_factory)
    env = env_factory(0)

    clause_a = True  # declared edges pass min-effect at correct lag, all states
    clause_b = True  # non-descendant negative controls, all states
    clause_c = True  # DIRECT off-manifold KPI-source parentage + controls, all states
    clause_d = True  # analytical closed-form == rolled-out latent (on + off manifold)

    for si, snap in enumerate(bank):
        params, prev_params, prev_kpis = snap[0], snap[1], snap[2]
        # Precompute do(high)/do(low) rollouts for every param at this state.
        roll_high = {p: _do_rollout(env, snap, p, DO_HIGH, H_DESC) for p in range(4)}
        roll_low = {p: _do_rollout(env, snap, p, DO_LOW, H_DESC) for p in range(4)}

        # (a) direct param edges P_i -> K_i at lag t+1 (out index 0).
        for param_id, child in DIRECT_PARAM_CHILD.items():
            eff = _std_delta(roll_high[param_id][0], roll_low[param_id][0], child)
            if eff < MIN_EFF:
                clause_a = False
                result.failures.append(
                    f"[a] state {si}: direct edge P{param_id}->K{child} effect "
                    f"{eff:.4f} < min_eff {MIN_EFF}"
                )

        # (a) KPI->KPI edges at descendant lag t+2 (out index 1), driven via param-ancestor
        #     (ancestry: the child must move when the source's ancestor param is moved).
        for src_kpi, dst_kpi in KPI_EDGES.items():
            anc = KPI_ANCESTOR_PARAM[src_kpi]
            eff = _std_delta(roll_high[anc][1], roll_low[anc][1], dst_kpi)
            if eff < MIN_EFF:
                clause_a = False
                result.failures.append(
                    f"[a] state {si}: KPI edge K{src_kpi}->K{dst_kpi} ancestry effect "
                    f"{eff:.4f} < min_eff {MIN_EFF}"
                )

        # (b) non-descendant negative controls: every non-descendant KPI must not move at any
        #     lag up to H_desc under do(P_i).
        for param_id in range(4):
            non_desc = set(range(4)) - PARAM_DESCENDANTS[param_id]
            for lag in range(H_DESC):
                for kpi in non_desc:
                    delta = _raw_delta(roll_high[param_id][lag], roll_low[param_id][lag], kpi)
                    if delta > TOL_ZERO:
                        clause_b = False
                        result.failures.append(
                            f"[b] state {si}: do(P{param_id}) moved non-descendant K{kpi} "
                            f"by {delta:.3e} at lag {lag + 1} (> tol {TOL_ZERO})"
                        )

        # (c) DIRECT off-manifold KPI-source parentage check for each declared KPI->KPI edge.
        for src_kpi, dst_kpi in KPI_EDGES.items():
            shadow = shadow_factory(0, dst_kpi)
            dst_param_parent = dst_kpi  # P_dst -> K_dst (direct param parent of the child)

            # OFF-manifold source values: K_src = P_src + offset, so K_src != P_src.
            src_on = float(prev_params[src_kpi])
            v_lo = src_on + KPI_DO_LOW
            v_hi = src_on + KPI_DO_HIGH

            # (i) TRUE SCM: pushing the latent source K_src off-manifold moves the child at
            #     the next step by >= min_eff (a real K_src -> K_dst edge is present).
            child_hi = _do_kpi_source(env, snap, src_kpi, v_hi)
            child_lo = _do_kpi_source(env, snap, src_kpi, v_lo)
            eff_true = _std_delta(child_hi, child_lo, dst_kpi)
            # (ii) SHADOW SCM (KPI-parent coeff zeroed): same direct source move leaves the
            #      child inert (<= tol_zero) — the KPI-parent contribution is gone.
            s_hi = _do_kpi_source(shadow, snap, src_kpi, v_hi)
            s_lo = _do_kpi_source(shadow, snap, src_kpi, v_lo)
            eff_shadow = _raw_delta(s_hi, s_lo, dst_kpi)
            # (iii) direct param->child effect (do(P_dst)) unchanged between true and shadow.
            direct_true = _raw_delta(
                roll_high[dst_param_parent][0], roll_low[dst_param_parent][0], dst_kpi
            )
            sd_high = _do_rollout(shadow, snap, dst_param_parent, DO_HIGH, H_DESC)
            sd_low = _do_rollout(shadow, snap, dst_param_parent, DO_LOW, H_DESC)
            direct_shadow = _raw_delta(sd_high[0], sd_low[0], dst_kpi)

            if eff_true < MIN_EFF:
                clause_c = False
                result.failures.append(
                    f"[c.i] state {si}: direct K{src_kpi}->K{dst_kpi} off-manifold effect "
                    f"{eff_true:.4f} < min_eff {MIN_EFF} (declared edge absent or reads a "
                    f"param, not the latent KPI)"
                )
            if eff_shadow > TOL_ZERO:
                clause_c = False
                result.failures.append(
                    f"[c.ii] state {si}: K{src_kpi}->K{dst_kpi} shadow (ablated) effect "
                    f"{eff_shadow:.3e} > tol {TOL_ZERO}"
                )
            if abs(direct_shadow - direct_true) > TOL_ZERO:
                clause_c = False
                result.failures.append(
                    f"[c.iii] state {si}: P{dst_param_parent}->K{dst_kpi} direct effect "
                    f"changed by ablation |{direct_shadow:.6f}-{direct_true:.6f}| > tol"
                )

            # (c) negative control: a DIRECT off-manifold move of a NON-parent source KPI
            #     must leave the child inert in the true SCM (no undeclared KPI->KPI edge).
            for other_src in KPI_ANCESTOR_PARAM:
                if other_src == src_kpi:
                    continue
                o_on = float(prev_params[other_src])
                o_hi = _do_kpi_source(env, snap, other_src, o_on + KPI_DO_HIGH)
                o_lo = _do_kpi_source(env, snap, other_src, o_on + KPI_DO_LOW)
                delta = _raw_delta(o_hi, o_lo, dst_kpi)
                if delta > TOL_ZERO:
                    clause_c = False
                    result.failures.append(
                        f"[c.neg] state {si}: direct move of non-parent K{other_src} moved "
                        f"K{dst_kpi} by {delta:.3e} (> tol) — undeclared cross-KPI edge"
                    )

        # (d) analytical closed-form next latent KPI matches the rolled-out latent value —
        #     on the neutral (on-manifold) bank state AND on preregistered OFF-manifold states.
        analytic = _analytic_next(prev_params, prev_kpis)
        env.restore(snap)
        rolled = env.advance()  # neutral advance -> next latent
        if np.max(np.abs(analytic - rolled)) > TOL_ZERO:
            clause_d = False
            result.failures.append(
                f"[d] state {si} (on-manifold): analytic {analytic} != rolled {rolled}"
            )
        for oi, offset in enumerate(OFF_MANIFOLD_KPI_OFFSETS):
            off_kpis = prev_kpis + offset  # K_i != P_i for all i
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

    result.clauses = {"a": clause_a, "b": clause_b, "c": clause_c, "d": clause_d}
    result.passed = all(result.clauses.values())
    return result


def main() -> int:
    result = run_e1_scm_gate()
    print("E1 SCM gate")
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
