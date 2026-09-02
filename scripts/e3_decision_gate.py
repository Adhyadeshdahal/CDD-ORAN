"""E3 decision-gap gate — temporal-depth fan-out cascade (SPEC E3, GATE_CONTRACT_E3.md).

Self-contained sim-level gate (like ``scripts/e2_decision_gate.py``); NO training. Every value is
FROZEN in ``docs/benchmark/GATE_CONTRACT_E3.md`` (reviewer ruling ``review-e3-redesign.md``) and is
not re-derived or tuned here.

Procedure:

1. **Bank (geometry only).** For ``env_seed = 0..4095`` (``episode=0``, noise OFF, reset + 3
   neutral advances), take the committed TRUE params and retain the first 32 with
   ``P1 >= L AND P2 >= L``, ``L = 0.5 + 0.5*sqrt(2/12)``. Fewer than 32 ⇒ feasibility FAIL. The
   reviewer's exact 32 retained seeds are reproduced by construction.
2. **TRUE oracle (vectorized).** Over all ``101^3 = 1,030,301`` P0 sequences ``a in V^3``,
   ``V = {0,0.01,...,1}``, compute ``G_true(s,a)`` on the latent scored window ``{k2,k3,k4}`` with
   the locked panel R, via analytic array broadcast (NOT 1M Python clones). ``G_star = max``,
   ``D3 = max - min``. ``sequence.py`` clone rollouts are the correctness anchor (tested).
3. **Arms.** FH = vectorized exhaustive argmax on the FULL model; TH = same on the TRUNCATED model
   (``(2,5),(3,5)`` dropped); FM = ``greedy_fm`` on the FULL model; TM = ``greedy_fm`` on the
   TRUNCATED model. Lexicographic tie-break. All four are realized on the FULL TRUE env (i.e.
   scored via the full-model ``G`` array). γ=1, latent noiseless.
4. **Metrics.** ``regret_true(X) = G_star - G_true(X)``; ``gap_H = regret(FM) - regret(FH) =
   G_true(FH) - G_true(FM)``; ``gap_H_norm = gap_H/D3``; ``gap_T_norm = (G_true(TH) -
   G_true(TM))/D3``. ``D3 <= tol_zero`` ⇒ degenerate FAIL.
5. **Accept.** PASS iff ``mean(gap_H_norm) >= tau_E3`` AND ``mean(abs(gap_T_norm)) <= tol``. BOTH
   clauses are wired into ``passed`` and the process exit status.
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cdd_oran.analysis.v2_regret import PanelXApp, reward  # noqa: E402
from cdd_oran.envs.v2.e3 import E3V2Env  # noqa: E402
from cdd_oran.planners.sequence import greedy_fm  # noqa: E402
from scripts.e3_scm_gate import run_e3_scm_gate  # noqa: E402

# --- FROZEN protocol constants (GATE_CONTRACT_E3.md) ------------------------------------------
SEED_POOL = range(4096)
EPISODE = 0
NEUTRAL_STEPS = 3
N_BANK = 32
SHARED_PARAM = 0  # P0
H = 3
GRID = np.linspace(0.0, 1.0, 101)  # V = {0, 0.01, ..., 1.00}
M_STD = 0.50
L = 0.5 + M_STD * math.sqrt(2.0 / 12.0)  # 0.7041241452319316...
TAU_E3 = 0.10
TOL = 0.01
TOL_ZERO = 1e-12
PANEL_KPI_IDS = (0, 2, 3)

# Standardization (exact sqrt; contract "Exact standardization").
SIG01 = math.sqrt(1.0 / 12.0)
SIG23 = math.sqrt(2.0 / 12.0)


def build_panel() -> list[PanelXApp]:
    """Locked R panel (0,2,3); single-KPI xApps with exact-sqrt standardization."""
    panel = []
    for i in PANEL_KPI_IDS:
        panel.append(
            PanelXApp(
                kpi_indices=(i,),
                mean=float(E3V2Env.mu[i]),
                std=float(E3V2Env.sigma[i]),
                threshold=float(E3V2Env.kpi_thresholds[i]),
                direction=int(E3V2Env.directions[i]),
            )
        )
    return panel


_PANEL = build_panel()


def _R(k: np.ndarray) -> float:
    return reward(np.asarray(k, dtype=float), _PANEL)


# --- state bank ------------------------------------------------------------------------------
def _committed_state(env_factory: Callable[[int], E3V2Env], seed: int):
    env = env_factory(seed)
    env.reset(episode=EPISODE)
    for _ in range(NEUTRAL_STEPS):
        env.neutral_step()
    return env.snapshot()


@dataclass
class BankResult:
    seeds: list[int] = field(default_factory=list)
    snaps: list = field(default_factory=list)
    feasible: bool = False


def build_bank(env_factory: Callable[[int], E3V2Env] = E3V2Env) -> BankResult:
    res = BankResult()
    for seed in SEED_POOL:
        snap = _committed_state(env_factory, seed)
        p = snap[1]  # committed prev_params
        if p[1] >= L and p[2] >= L:
            res.seeds.append(seed)
            res.snaps.append(snap)
        if len(res.seeds) == N_BANK:
            break
    res.feasible = len(res.seeds) == N_BANK
    return res


# --- vectorized analytic cumulative return over V^3 ------------------------------------------
def _panel_step_R(k0, k2, k3):
    """R for one scored step over panel {K0(dir0), K2(dir1), K3(dir1)}, broadcast over arrays."""
    z0 = (k0 - 0.5) / SIG01
    z2 = (k2 - 1.0) / SIG23
    z3 = (k3 - 1.0) / SIG23
    ok = (z0 >= 0).astype(float) + (z2 <= 0).astype(float) + (z3 <= 0).astype(float)
    dist = np.maximum(-z0, 0.0) + np.maximum(z2, 0.0) + np.maximum(z3, 0.0)
    return ok**2 - 10.0 * dist


def vectorized_G(snap, c10: float, c25: float, c35: float) -> np.ndarray:
    """Cumulative latent return G(a1,a2,a3) over the full V^3 grid for one analytic model.

    Closed form of the SEMANTICS §1.1 call list on the E3 mechanisms from committed state ``snap``
    (P0..P3 committed, P0 acted). Returns a (101,101,101) array indexed by grid indices of
    (a1,a2,a3). Scored window {k2,k3,k4}; k1 warm-up excluded.
    """
    P = np.asarray(snap[1], dtype=float)  # committed prev_params
    K = np.asarray(snap[2], dtype=float)  # committed prev_kpis
    P0c, P1c, P2c = P[0], P[1], P[2]  # P3 (K4) is excluded from the E3 panel
    K0c = K[0]

    a1 = GRID[:, None, None]
    a2 = GRID[None, :, None]
    a3 = GRID[None, None, :]

    # k_2 (SCORED): k0=a1; K2/K3 carry the committed K0c through the K1 conduit.
    R2 = _panel_step_R(a1, P1c + c25 * c10 * K0c, P2c + c35 * c10 * K0c)
    # k_3 (SCORED): k0=a2; K2/K3 carry committed P0c through the conduit.
    R3 = _panel_step_R(a2, P1c + c25 * c10 * P0c, P2c + c35 * c10 * P0c)
    # k_4 (SCORED): k0=a3; K2/K3 carry a1 (the delayed harm surfaces two steps later).
    R4 = _panel_step_R(a3, P1c + c25 * c10 * a1, P2c + c35 * c10 * a1)

    return R2 + R3 + R4  # broadcasts to (101,101,101)


def _argmax_lex(G: np.ndarray) -> tuple[int, int, int]:
    """Lexicographically smallest argmax: np.argmax on C-order returns the first max (row-major)."""
    idx = np.unravel_index(int(np.argmax(G)), G.shape)
    return int(idx[0]), int(idx[1]), int(idx[2])


def _seq_to_indices(values) -> tuple[int, ...]:
    return tuple(int(np.argmin(np.abs(GRID - float(v)))) for v in values)


# --- gate ------------------------------------------------------------------------------------
@dataclass
class GateResult:
    passed: bool = False
    feasible: bool = False
    mean_gap_H_norm: float = float("nan")
    mean_abs_gap_T_norm: float = float("nan")
    separation: float = float("nan")
    raw_gap_H: list[float] = field(default_factory=list)  # per-state RAW G_true(FH)-G_true(FM)
    per_state: list[dict] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


def run_e3_decision_gate(
    env_factory: Callable[[int], E3V2Env] = E3V2Env,
    full_factory: Callable[[], E3V2Env] | None = None,
    trunc_factory: Callable[[], E3V2Env] | None = None,
) -> GateResult:
    if full_factory is None:
        def full_factory() -> E3V2Env:
            return E3V2Env(truncate_fanout=False)
    if trunc_factory is None:
        def trunc_factory() -> E3V2Env:
            return E3V2Env(truncate_fanout=True)

    fm = full_factory()
    tm = trunc_factory()
    full_c = (fm.c10, fm.c25, fm.c35)
    trunc_c = (tm.c10, tm.c25, tm.c35)

    result = GateResult()
    bank = build_bank(env_factory)
    result.feasible = bank.feasible
    if not bank.feasible:
        result.failures.append(
            f"feasibility: only {len(bank.seeds)} valid states in pool (need {N_BANK})"
        )
        return result

    gap_H_norms, gap_T_norms = [], []
    for seed, snap in zip(bank.seeds, bank.snaps, strict=True):
        G_full = vectorized_G(snap, *full_c)
        G_trunc = vectorized_G(snap, *trunc_c)

        g_star = float(G_full.max())
        d3 = g_star - float(G_full.min())
        if d3 <= TOL_ZERO:
            result.failures.append(f"degenerate: seed {seed} D3={d3:.3e} <= tol_zero")
            result.passed = False
            return result

        fh_idx = _argmax_lex(G_full)
        th_idx = _argmax_lex(G_trunc)
        fm_idx = _seq_to_indices(
            greedy_fm(full_factory, snap, SHARED_PARAM, GRID.tolist(), H, _R)
        )
        tm_idx = _seq_to_indices(
            greedy_fm(trunc_factory, snap, SHARED_PARAM, GRID.tolist(), H, _R)
        )

        # All arms realized on the FULL TRUE env (the full-model G array).
        G_FH = float(G_full[fh_idx])
        G_FM = float(G_full[fm_idx])
        G_TH = float(G_full[th_idx])
        G_TM = float(G_full[tm_idx])

        gap_H = G_FH - G_FM  # = regret(FM) - regret(FH)
        gap_T = G_TH - G_TM
        gap_H_norms.append(gap_H / d3)
        gap_T_norms.append(gap_T / d3)
        result.raw_gap_H.append(gap_H)

        # Contract "Decision gate" reporting (GATE_CONTRACT_E3.md:171-174): all four arm
        # sequences, TRUE returns/regrets, D3, raw+normalized gap_H and gap_T, and FH-vs-TH.
        def _seq(idx):
            return [float(GRID[j]) for j in idx]

        fh_vs_th = G_FH - G_TH  # secondary structure diagnostic (NOT gated)
        result.per_state.append(
            {
                "seed": seed,
                "G_star": g_star,
                "D3": d3,
                # arm emitted sequences (P0 values, length H=3)
                "seq_FH": _seq(fh_idx),
                "seq_FM": _seq(fm_idx),
                "seq_TH": _seq(th_idx),
                "seq_TM": _seq(tm_idx),
                # realized TRUE cumulative returns
                "G_FH": G_FH,
                "G_FM": G_FM,
                "G_TH": G_TH,
                "G_TM": G_TM,
                # realized TRUE regrets (G_star - G_arm), unclamped
                "regret_FH": g_star - G_FH,
                "regret_FM": g_star - G_FM,
                "regret_TH": g_star - G_TH,
                "regret_TM": g_star - G_TM,
                # primary horizon gap (raw + normalized)
                "gap_H": gap_H,
                "gap_H_norm": gap_H / d3,
                # truncated factor-removal control (raw + normalized)
                "gap_T": gap_T,
                "gap_T_norm": gap_T / d3,
                # FH-vs-TH secondary structure diagnostic (raw + normalized)
                "fh_vs_th": fh_vs_th,
                "fh_vs_th_norm": fh_vs_th / d3,
            }
        )

    result.mean_gap_H_norm = float(np.mean(gap_H_norms))
    result.mean_abs_gap_T_norm = float(np.mean(np.abs(gap_T_norms)))
    result.separation = result.mean_gap_H_norm - result.mean_abs_gap_T_norm

    pass_primary = result.mean_gap_H_norm >= TAU_E3
    pass_control = result.mean_abs_gap_T_norm <= TOL
    if not pass_primary:
        result.failures.append(
            f"mean(gap_H_norm) {result.mean_gap_H_norm:.4f} < tau_E3 {TAU_E3}"
        )
    if not pass_control:
        result.failures.append(
            f"mean(|gap_T_norm|) {result.mean_abs_gap_T_norm:.4f} > tol {TOL}"
        )
    result.passed = pass_primary and pass_control
    return result


def _print_per_state_table(per_state: list[dict]) -> None:
    """Contract-mandated per-state diagnostics (GATE_CONTRACT_E3.md:171-174). DIAGNOSTIC ONLY."""
    print("  per-state diagnostics (DIAGNOSTIC - not gated):")
    header = (
        f"    {'seed':>4} {'D3':>8} {'G*':>8} "
        f"{'G_FH':>8} {'G_FM':>8} {'G_TH':>8} {'G_TM':>8} "
        f"{'regFH':>7} {'regFM':>7} {'regTH':>7} {'regTM':>7} "
        f"{'gapH':>8} {'gapH_n':>7} {'gapT':>6} {'gapT_n':>7} {'FHvTH_n':>8}"
    )
    print(header)
    for s in per_state:
        print(
            f"    {s['seed']:>4} {s['D3']:>8.3f} {s['G_star']:>8.3f} "
            f"{s['G_FH']:>8.3f} {s['G_FM']:>8.3f} {s['G_TH']:>8.3f} {s['G_TM']:>8.3f} "
            f"{s['regret_FH']:>7.3f} {s['regret_FM']:>7.3f} {s['regret_TH']:>7.3f} "
            f"{s['regret_TM']:>7.3f} "
            f"{s['gap_H']:>8.3f} {s['gap_H_norm']:>7.4f} {s['gap_T']:>6.3f} "
            f"{s['gap_T_norm']:>7.4f} {s['fh_vs_th_norm']:>8.4f}"
        )
        print(
            f"         seqs FH={s['seq_FH']} FM={s['seq_FM']} "
            f"TH={s['seq_TH']} TM={s['seq_TM']}"
        )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="E3 decision-gap gate")
    parser.add_argument(
        "--inject-bad-control",
        action="store_true",
        help="TEST-ONLY: use the FULL model as the 'truncated' control so the factor-removal "
        "clause fails; verifies acceptance gates on BOTH clauses (exit nonzero).",
    )
    parser.add_argument(
        "--skip-scm",
        action="store_true",
        help="TEST-ONLY: skip the SCM-correctness prerequisite and run the decision gate in "
        "isolation. The authoritative entry point runs the SCM gate first (contract prerequisite).",
    )
    args = parser.parse_args(argv)

    # GATE_CONTRACT_E3.md prerequisite: the canonical SCM-correctness gate MUST pass before the
    # decision gate. This CLI default couples them — the authoritative entry point runs the
    # CANONICAL SCM gate (default factories) and refuses a decision PASS if it fails. Scope note:
    # this is NOT a universal coupling of arbitrary factories. run_e3_decision_gate() stays
    # independently injectable, and --skip-scm bypasses the prerequisite — both for unit tests.
    if not args.skip_scm:
        scm = run_e3_scm_gate()
        print(f"E3 SCM prerequisite: {'PASS' if scm.passed else 'FAIL'}")
        if not scm.passed:
            for msg in getattr(scm, "failures", []):
                print(f"    - {msg}")
            print("  SCM prerequisite FAILED — decision gate not run.")
            print("RESULT: FAIL")
            return 1

    if args.inject_bad_control:
        def _full() -> E3V2Env:
            return E3V2Env(truncate_fanout=False)

        result = run_e3_decision_gate(E3V2Env, full_factory=_full, trunc_factory=_full)
    else:
        result = run_e3_decision_gate()
    print("E3 decision-gap gate")
    print(f"  bank feasible: {result.feasible} ({len(result.per_state)} states)")
    if result.per_state:
        _print_per_state_table(result.per_state)
    print("  aggregate:")
    print(f"  mean(gap_H_norm):    {result.mean_gap_H_norm:.6f}   (tau_E3 = {TAU_E3})")
    print(f"  mean(|gap_T_norm|):  {result.mean_abs_gap_T_norm:.6f}   (tol    = {TOL})")
    print(f"  separation:          {result.separation:.6f}")
    if result.raw_gap_H:
        print(
            f"  raw FH-FM cumulative-R range: "
            f"[{min(result.raw_gap_H):.6f}, {max(result.raw_gap_H):.6f}]"
        )
    if result.failures:
        print("  failures:")
        for msg in result.failures:
            print(f"    - {msg}")
    print(f"RESULT: {'PASS' if result.passed else 'FAIL'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
