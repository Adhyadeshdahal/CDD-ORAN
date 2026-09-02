"""E4 structural design-validation gate — action-relevant confounding (SPEC E4, GATES §2.5/§3.1).

Self-contained sim-level gate (like ``scripts/e2_decision_gate.py`` / ``scripts/e3_scm_gate.py``);
NO training. Every equation, coefficient, policy, mixture, bank rule, action, threshold, and
tolerance is FROZEN in ``docs/benchmark/GATE_CONTRACT_E4.md`` (reviewer ruling
``.herdr/reports/rule-e4-design.md``, "Frozen E4 Contract Value Block") and is not re-derived here.

This is DESIGN VALIDATION: deterministic success says the benchmark *expresses* action-relevant
confounding (the same latent ``Z`` drives the behavior action and the outcome, and a naive pooler
on the declared 90:10 obs/do mixture is sign-reversed relative to the true ``A -> K_out`` effect).
It does NOT claim a learned causal arm wins — that trained-arm evaluation is DEFERRED.

Procedure (contract):

1. **Exact clipped-normal moments.** Compute ``p``, ``Var(A_behavior)``, ``Cov(A_behavior,Z)`` at
   the primary ``lambda`` and assert they equal the frozen numbers to ``tol_zero``.
2. **Naive 90:10 pooled line.** Compute ``b_pool`` / ``c_pool`` for the OLS pooler that ignores
   ``D`` and omits latent ``Z``, and assert they equal the frozen numbers to ``tol_zero``.
3. **Geometry bank.** Ascending ``env_seed = 0..4095`` (``episode=0``, noise OFF, 3 neutral
   advances), retain the first ``N=64`` states whose declared TRUE interventional geometry over
   ``V`` satisfies the frozen validity rule (expected seeds ``0..63``). Fewer than 64 ⇒ feasibility
   FAIL.
4. **Per-state structural gap.** Oracle/aware model ``K_hat=-a`` and naive model
   ``K_hat=c_pool+b_pool*a`` each select ``argmax`` of the locked hinge over ``V`` (smallest index
   on ties); realize both on the true interventional-mean SCM; ``gap_norm = (regret_true(naive) −
   regret_true(oracle)) / D(s)``. ``D(s) <= tol_zero`` ⇒ degenerate FAIL.
5. **λ=0 factor-removal control.** Remove exactly ``Z -> A_behavior`` (``lambda=0``); recompute the
   exact pooler (slope ``-1``); the naive control action is ``0`` and ``gap_norm_control = 0``.
6. **Accept.** PASS iff ``mean_bank(gap_norm) >= tau_E4`` AND ``mean_bank(abs(gap_norm_control)) <=
   tol``. BOTH clauses drive ``passed`` and the process exit status (the E2 lesson — a computed but
   ungated control silently passes a broken benchmark).

Injectable over ``env_factory`` (true env, primary ``lambda``) and ``control_env_factory`` (the
``lambda=0`` factor-removal env), so mutation tests can feed a *wrong* control (a ``lambda != 0``
env that does NOT remove the confounding edge) and confirm the gate REJECTS it.
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

# Allow direct execution (``python scripts/e4_structural_gate.py``) as well as ``-m``: put the repo
# root on the path before importing the package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cdd_oran.analysis.v2_regret import PanelXApp, reward  # noqa: E402
from cdd_oran.envs.v2.e4 import MU_OUT, SIGMA_OUT, E4V2Env  # noqa: E402

# --- FROZEN protocol constants (GATE_CONTRACT_E4.md) ------------------------------------------
SEED_POOL = range(4096)  # env_seed = 0..4095
EPISODE = 0
NEUTRAL_STEPS = 3
N_STATES = 64
M_STD = 0.50
RHO_OBS, RHO_DO = 0.9, 0.1
VAR_A_DO = 17.0 / 200.0  # 0.085 (uniform do-grid variance)
ETA_SCALE = 0.5
PRIMARY_LAM = 1.0
CONTROL_LAM = 0.0

# Action grid V = {0, 0.01, ..., 1.00} (101 inclusive points).
V_GRID = np.linspace(0.0, 1.0, 101)

# Frozen exact moments / lines at the primary lambda (assert == to tol_zero).
FROZEN_P = 0.34527915398142284
FROZEN_VAR_A = 0.1916935893025823
FROZEN_COV_AZ = 0.34527915398142284
FROZEN_B_POOL = 3.291569669211391
FROZEN_C_POOL = -2.145784834605695
FROZEN_D_STATE = 35.64101615137755
FROZEN_GAP_NORM = 0.6695395708852947
FROZEN_A_ORACLE = 0.00
FROZEN_A_NAIVE = 0.66
EXPECTED_SEEDS = list(range(N_STATES))

TAU_E4 = 0.10
TOL = 0.01
TOL_ZERO = 1e-12

# --- FROZEN env-corpus validation coordinates (circularity fix, review-e4-gate.md BLOCKER 1) ---
# The analytic checks above re-derive the frozen algebra and never touch env-generated data — a
# broken env still passed. These constants pin a FROZEN obs/do corpus GENERATED FROM THE REAL
# ``E4V2Env`` (via generate_action/behavior_action/apply_action/advance) whose EMPIRICAL moments,
# OLS slopes, pooled line, mechanism identity, do-independence, and lambda-zero collapse are
# asserted against the frozen closed-form. ``STAT_TOL`` is a STATISTICAL tolerance (distinct from
# ``tol_zero``), sized from N so the TRUE env passes comfortably and a broken env (no confounding,
# pooled slope ~= -1, or a mis-scaled/absent mechanism) FAILS by a wide margin.
CORPUS_SEED = 0            # single frozen env_seed; a walk over distinct time coordinates is i.i.d.
CORPUS_N_OBS = 90000       # observational transitions
CORPUS_N_DO = 20000        # interventional transitions (do-slope / do-independence sample)
CORPUS_POOL_DO = 10000     # do rows entering the 90:10 pool (90000:10000 == 90:10)
STAT_TOL = 0.15            # statistical tolerance; TRUE-env max error ~0.10 (do slope), broken >=0.19
MECH_TOL = TOL_ZERO        # mechanism identity K_out == alpha*A + theta*Z + c is EXACT


# --- exact clipped-normal moments (contract "Behavior and matched obs/do protocol") -----------
def clipped_normal_moments(lam: float, eta_scale: float = ETA_SCALE) -> tuple[float, float, float]:
    """Exact moments of ``A_behavior = clip(0.5 + lambda*Z + eta, 0, 1)``.

    Returns ``(p, Var(A_behavior), Cov(A_behavior, Z))`` with ``p = 2*Phi(r) - 1``,
    ``r = 0.5/sqrt(s2)``, ``s2 = lambda^2 + eta_scale^2``, standard-normal CDF/pdf ``Phi``/``phi``.
    """
    s2 = lam * lam + eta_scale * eta_scale
    r = 0.5 / math.sqrt(s2)
    phi_r = math.exp(-r * r / 2.0) / math.sqrt(2.0 * math.pi)
    Phi_r = 0.5 * (1.0 + math.erf(r / math.sqrt(2.0)))
    p = 2.0 * Phi_r - 1.0
    var_a = s2 * (p - 2.0 * r * phi_r) + 0.25 * 2.0 * (1.0 - Phi_r)
    cov_az = lam * p
    return p, var_a, cov_az


def pooled_line(lam: float, alpha: float, theta: float) -> tuple[float, float]:
    """Exact 90:10 OLS pooler slope/intercept that ignores ``D`` and omits latent ``Z``.

    ``E[A] = 0.5`` (obs and do), ``E[K] = alpha*0.5 + theta*E[Z] + c`` with ``E[Z]=0``; the do
    stream contributes ``Cov = alpha*Var(A_do)`` (no ``Z`` coupling under randomization).
    """
    _p, var_a, cov_az = clipped_normal_moments(lam)
    var_pool = RHO_OBS * var_a + RHO_DO * VAR_A_DO
    cov_obs = alpha * var_a + theta * cov_az
    cov_pool = RHO_OBS * cov_obs + RHO_DO * alpha * VAR_A_DO
    b_pool = cov_pool / var_pool
    e_k = alpha * 0.5 + theta * 0.0  # + c; c folded in by the caller's env (default c=0)
    c_pool = e_k - b_pool * 0.5
    return b_pool, c_pool


# --- locked single-KPI hinge panel (contract "Exact standardization and objective") -----------
def _panel() -> list[PanelXApp]:
    return [PanelXApp(kpi_indices=(0,), mean=MU_OUT, std=SIGMA_OUT, threshold=0.0, direction=0)]


def _hinge_scores(k_hat: np.ndarray) -> np.ndarray:
    """Locked hinge return over the grid for a predicted-outcome vector ``k_hat`` (one per action)."""
    panel = _panel()
    return np.array([reward(np.array([kh]), panel) for kh in k_hat], dtype=float)


def _select_argmax(k_hat: np.ndarray) -> int:
    """``argmax`` of the locked hinge, smallest grid index on ties (numpy argmax default)."""
    return int(np.argmax(_hinge_scores(k_hat)))


# --- default injectable factories -------------------------------------------------------------
def default_env_factory(seed: int) -> E4V2Env:
    """True env under test: frozen primary E4 SCM, interventional mode, noise OFF."""
    return E4V2Env(env_seed=seed, obs_noise_scale=0.0, lam=PRIMARY_LAM, mode=E4V2Env.MODE_DO)


def default_control_env_factory(seed: int) -> E4V2Env:
    """Factor-removal control: identical SCM with ONLY ``Z -> A_behavior`` removed (lambda=0)."""
    return E4V2Env(env_seed=seed, obs_noise_scale=0.0, lam=CONTROL_LAM, mode=E4V2Env.MODE_DO)


# --- geometry bank (contract "Geometry bank") -------------------------------------------------
def _state_valid(env: E4V2Env) -> bool:
    """Declared TRUE interventional geometry validity over ``V`` (state-independent by design)."""
    mu, sig = float(env.mu[0]), float(env.sigma[0])
    theta_std = (0.0 - mu) / sig

    def z_true(a: float) -> float:
        return (env.score_interventional_mean(a) - mu) / sig

    z0, z1 = z_true(0.0), z_true(1.0)
    return z0 >= theta_std and z1 <= theta_std - M_STD and (z0 - z1) >= 2.0 * M_STD


@dataclass
class BankResult:
    seeds: list = field(default_factory=list)
    snaps: list = field(default_factory=list)
    feasible: bool = False


def build_bank(env_factory: Callable[[int], E4V2Env] = default_env_factory) -> BankResult:
    res = BankResult()
    for seed in SEED_POOL:
        env = env_factory(seed)
        env.reset(episode=EPISODE)
        for _ in range(NEUTRAL_STEPS):
            env.neutral_step()
        if _state_valid(env):
            res.seeds.append(seed)
            res.snaps.append(env.snapshot())
        if len(res.seeds) == N_STATES:
            break
    res.feasible = len(res.seeds) == N_STATES
    return res


# --- per-state structural gap -----------------------------------------------------------------
def _gap_norm(true_env: E4V2Env, b_pool: float, c_pool: float) -> tuple[float, float, int, int]:
    """Return ``(gap_norm, D, a_oracle_idx, a_naive_idx)`` for one state.

    Oracle/aware model ``K_hat=-a`` and naive model ``K_hat=c_pool+b_pool*a`` each select argmax
    of the locked hinge; both are realized on the TRUE interventional-mean SCM. Raises on
    degenerate ``D <= tol_zero``.
    """
    k_true = np.array([true_env.score_interventional_mean(v) for v in V_GRID], dtype=float)
    g_true = _hinge_scores(k_true)  # realized true return per action
    g_star = float(g_true.max())
    d = g_star - float(g_true.min())
    if d <= TOL_ZERO:
        raise ValueError(f"degenerate D(s)={d:.3e} <= tol_zero {TOL_ZERO}")

    k_oracle = np.array([-v for v in V_GRID], dtype=float)  # aware model = true interventional line
    k_naive = np.array([c_pool + b_pool * v for v in V_GRID], dtype=float)
    a_oracle_idx = _select_argmax(k_oracle)
    a_naive_idx = _select_argmax(k_naive)

    regret_oracle = g_star - float(g_true[a_oracle_idx])
    regret_naive = g_star - float(g_true[a_naive_idx])
    gap_norm = (regret_naive - regret_oracle) / d
    return gap_norm, d, a_oracle_idx, a_naive_idx


# --- ENVIRONMENT-CORPUS validation (circularity fix, review-e4-gate.md BLOCKER 1) -------------
def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Ordinary least squares ``y ~ slope*x + intercept`` (empirical).

    A degenerate zero-variance regressor (e.g. a broken constant behavior policy) returns ``nan``
    rather than dividing by zero; the accompanying moment checks (``Var(A)``, mechanism) flag it.
    """
    xm = float(x.mean())
    var = float(((x - xm) ** 2).mean())
    if var <= 0.0:
        return float("nan"), float("nan")
    cov = float(((x - xm) * (y - y.mean())).mean())
    slope = cov / var
    return slope, float(y.mean() - slope * xm)


def _cov(x: np.ndarray, y: np.ndarray) -> float:
    return float(((x - x.mean()) * (y - y.mean())).mean())


def generate_corpus(probe: E4V2Env, mode: str, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate ``n`` delayed transitions ``(A_committed, Z_committed, K_out)`` from the REAL env.

    Walks a rebuilt clone of ``probe`` (same lambda/alpha/theta/c/eta, given ``mode``) over distinct
    time coordinates via ``generate_action -> apply_action -> advance``. Each coordinate draws an
    independent ``Z``/``eta``/do-action from the coordinate-keyed tape, so the transitions are
    i.i.d. The outcome collected one ``advance`` after an action is committed is paired with that
    action's committed ``Z`` — exactly the confounding the design claims. ``type(probe)`` preserves
    a broken subclass so its generated behavior/mechanism (not its declared attrs) is what is tested.
    """
    env = type(probe)(
        env_seed=CORPUS_SEED,
        obs_noise_scale=0.0,
        lam=probe.lam,
        alpha=probe.alpha,
        theta=probe.theta,
        c=probe.c,
        eta_scale=probe.eta_scale,
        mode=mode,
    )
    a_arr = np.empty(n, dtype=float)
    z_arr = np.empty(n, dtype=float)
    k_arr = np.empty(n, dtype=float)
    prev: tuple[float, float] | None = None
    i = 0
    while i < n:
        a = env.generate_action()
        z = float(env.Z)
        env.apply_action(0, a)
        k = float(env.advance()[0])  # outcome for the PREVIOUSLY committed (a, z)
        if prev is not None:
            a_arr[i], z_arr[i], k_arr[i] = prev[0], prev[1], k
            i += 1
        prev = (a, z)
    return a_arr, z_arr, k_arr


@dataclass
class EnvCorpusResult:
    ok: bool = True
    empirical: dict = field(default_factory=dict)
    frozen: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)


def validate_env_corpus(
    env_factory: Callable[[int], E4V2Env],
    label: str,
    n_obs: int = CORPUS_N_OBS,
    n_do: int = CORPUS_N_DO,
    pool_do: int = CORPUS_POOL_DO,
) -> EnvCorpusResult:
    """Assert an env-GENERATED 90:10 corpus matches the frozen closed-form of the env's declared SCM.

    Frozen targets are derived from the env's OWN ``(lam, alpha, theta)`` attrs; the empirical
    values come from transitions the env actually produced. A broken env whose generated behavior
    or mechanism does not match its declared design fails — this is the anti-circularity property.
    Checked (each within ``STAT_TOL``, mechanism within ``MECH_TOL``): observational ``E[A]``,
    ``Var(A)``, ``Cov(A,Z)``, OLS slope; do ``Var(A)``, ``Cov(A,Z)`` (~0), OLS slope; the 90:10
    pooled slope + intercept; and the exact ``K_out == alpha*A + theta*Z + c`` mechanism identity
    on every sampled transition, in both modes.
    """
    res = EnvCorpusResult()
    probe = env_factory(CORPUS_SEED)
    alpha, theta = probe.alpha, probe.theta

    # frozen closed-form targets from the env's DECLARED parameters.
    _p, var_a, cov_az = clipped_normal_moments(probe.lam)
    obs_slope_f = (alpha * var_a + theta * cov_az) / var_a
    b_pool_f, c_pool_f = pooled_line(probe.lam, alpha, theta)
    res.frozen = {
        "obs_E[A]": 0.5,
        "obs_Var(A)": var_a,
        "obs_Cov(A,Z)": cov_az,
        "obs_slope": obs_slope_f,
        "do_Var(A)": VAR_A_DO,
        "do_Cov(A,Z)": 0.0,
        "do_slope": alpha,
        "pooled_slope": b_pool_f,
        "pooled_intercept": c_pool_f,
    }

    # env-generated corpora.
    ao, zo, ko = generate_corpus(probe, E4V2Env.MODE_OBS, n_obs)
    ad, zd, kd = generate_corpus(probe, E4V2Env.MODE_DO, n_do)

    obs_slope, _obs_int = _ols(ao, ko)
    do_slope, _do_int = _ols(ad, kd)
    pool_do = min(pool_do, n_do)
    a_pool = np.concatenate([ao, ad[:pool_do]])
    k_pool = np.concatenate([ko, kd[:pool_do]])
    pooled_slope, pooled_int = _ols(a_pool, k_pool)
    mech_obs = float(np.max(np.abs(ko - (alpha * ao + theta * zo + probe.c))))
    mech_do = float(np.max(np.abs(kd - (alpha * ad + theta * zd + probe.c))))

    res.empirical = {
        "obs_E[A]": float(ao.mean()),
        "obs_Var(A)": float(ao.var()),
        "obs_Cov(A,Z)": _cov(ao, zo),
        "obs_slope": obs_slope,
        "do_Var(A)": float(ad.var()),
        "do_Cov(A,Z)": _cov(ad, zd),
        "do_slope": do_slope,
        "pooled_slope": pooled_slope,
        "pooled_intercept": pooled_int,
        "mech_err_obs": mech_obs,
        "mech_err_do": mech_do,
    }

    # statistical-tolerance checks against the frozen closed-form.
    for key in res.frozen:
        got, frozen = res.empirical[key], res.frozen[key]
        if abs(got - frozen) > STAT_TOL:
            res.ok = False
            res.failures.append(
                f"[env-corpus:{label}] {key}={got:.6f} != frozen {frozen:.6f} "
                f"(|diff|={abs(got - frozen):.6f} > stat_tol {STAT_TOL})"
            )
    # exact mechanism identity on every sampled transition.
    for mode, err in (("obs", mech_obs), ("do", mech_do)):
        if err > MECH_TOL:
            res.ok = False
            res.failures.append(
                f"[env-corpus:{label}] {mode} mechanism K_out != alpha*A+theta*Z+c "
                f"(max err {err:.3e} > mech_tol {MECH_TOL})"
            )
    return res


@dataclass
class GateResult:
    passed: bool = False
    feasible: bool = False
    n_states: int = 0
    seeds_ok: bool = False
    mean_gap: float = float("nan")
    mean_control: float = float("nan")
    b_pool: float = float("nan")
    c_pool: float = float("nan")
    b_pool_control: float = float("nan")
    a_naive: float = float("nan")
    a_oracle: float = float("nan")
    a_naive_control: float = float("nan")
    d_state: float = float("nan")
    adjacency_ok: bool = False
    latent_edges_ok: bool = False
    env_corpus_ok: bool = False
    control_corpus_ok: bool = False
    single_factor_ok: bool = False
    env_corpus: EnvCorpusResult | None = None
    control_corpus: EnvCorpusResult | None = None
    failures: list = field(default_factory=list)


def run_e4_structural_gate(
    env_factory: Callable[[int], E4V2Env] = default_env_factory,
    control_env_factory: Callable[[int], E4V2Env] = default_control_env_factory,
    corpus_n_obs: int = CORPUS_N_OBS,
    corpus_n_do: int = CORPUS_N_DO,
    corpus_pool_do: int = CORPUS_POOL_DO,
) -> GateResult:
    """Run the E4 structural design-validation gate + λ=0 factor-removal control.

    ``env_factory(seed)`` builds the true env (primary lambda); ``control_env_factory(seed)`` the
    factor-removal env (lambda read from the env, so injecting a lambda != 0 control REJECTS).
    ``corpus_n_obs``/``corpus_n_do``/``corpus_pool_do`` default to the FROZEN sample coordinates;
    they are exposed only so tests can shrink the env-corpus for speed (a broken env fails the
    mechanism identity at any N). All acceptance clauses — the analytic primary + control gaps,
    the adjacency/latent-edge structure, AND the env-generated corpus checks — are ANDed into
    ``passed`` and, via ``main``, the process exit status.
    """
    result = GateResult()

    probe = env_factory(0)
    control_probe = control_env_factory(0)
    alpha, theta = probe.alpha, probe.theta

    # (1b) single-factor control invariant (GATE_CONTRACT_E4.md: the control removes ONLY
    # Z->A_behavior, i.e. differs from the primary env in lambda alone; every other frozen SCM
    # parameter is identical). Without this the control could silently mutate alpha/theta/c and
    # validate against its own declarations. Assert the invariant executably.
    # Every contract-relevant field except lambda must be identical between primary and control.
    _SINGLE_FACTOR_FIELDS = ("alpha", "theta", "c", "eta_scale", "obs_noise_scale", "mode")
    _field_mismatch = [
        f for f in _SINGLE_FACTOR_FIELDS if getattr(control_probe, f) != getattr(probe, f)
    ]
    result.single_factor_ok = bool(
        probe.lam == PRIMARY_LAM and control_probe.lam == CONTROL_LAM and not _field_mismatch
    )
    if not result.single_factor_ok:
        details = ", ".join(
            f"{f}: primary={getattr(probe, f)!r} vs control={getattr(control_probe, f)!r}"
            for f in (["lam"] if probe.lam != PRIMARY_LAM or control_probe.lam != CONTROL_LAM else [])
            + _field_mismatch
        )
        result.failures.append(
            "control is not a single-factor (lambda-only) removal — mismatched: " + details
        )

    # (1) exact moments at the primary lambda.
    p, var_a, cov_az = clipped_normal_moments(probe.lam)
    for name, got, frozen in (
        ("p", p, FROZEN_P),
        ("Var(A_behavior)", var_a, FROZEN_VAR_A),
        ("Cov(A_behavior,Z)", cov_az, FROZEN_COV_AZ),
    ):
        if abs(got - frozen) > TOL_ZERO:
            result.failures.append(f"moment {name}={got!r} != frozen {frozen!r} (>tol_zero)")

    # (2) naive 90:10 pooled line (primary).
    b_pool, c_pool = pooled_line(probe.lam, alpha, theta)
    result.b_pool, result.c_pool = b_pool, c_pool
    if abs(b_pool - FROZEN_B_POOL) > TOL_ZERO:
        result.failures.append(f"b_pool={b_pool!r} != frozen {FROZEN_B_POOL!r} (>tol_zero)")
    if abs(c_pool - FROZEN_C_POOL) > TOL_ZERO:
        result.failures.append(f"c_pool={c_pool!r} != frozen {FROZEN_C_POOL!r} (>tol_zero)")

    # control pooled line (lambda read from the control env — a wrong lambda!=0 fails to remove
    # the confounding edge and is REJECTED downstream).
    b_pool_c, c_pool_c = pooled_line(control_probe.lam, control_probe.alpha, control_probe.theta)
    result.b_pool_control = b_pool_c

    # (8) adjacency / latent-edge assertions.
    adj = probe.true_adj_matrix()
    expected_adj = np.zeros_like(adj)
    for ki, si in E4V2Env.TRUE_ADJACENCY:
        expected_adj[probe.num_params + ki, si] = 1.0
    result.adjacency_ok = (
        list(probe.adjacency_edges) == list(E4V2Env.TRUE_ADJACENCY)
        and np.array_equal(adj, expected_adj)
    )
    result.latent_edges_ok = list(E4V2Env.LATENT_EDGES) == [
        ("Z", "A_behavior"),
        ("Z", "K_out"),
    ]
    if not result.adjacency_ok:
        result.failures.append(
            f"observed adjacency {sorted(probe.adjacency_edges)} != frozen "
            f"{sorted(E4V2Env.TRUE_ADJACENCY)}"
        )
    if not result.latent_edges_ok:
        result.failures.append(f"latent edges {E4V2Env.LATENT_EDGES} != frozen 2 Z-incident edges")

    # (env-corpus) validate the ACTUAL env-generated obs/do corpus against the frozen closed-form
    # (circularity fix, review-e4-gate.md BLOCKER 1). The primary corpus pins behavior moments,
    # OLS slopes, the 90:10 pooled line, mechanism identity, and do-independence; the control
    # corpus additionally pins the lambda-zero pooled-slope collapse to -1. A broken env that does
    # not GENERATE the declared confounding fails here even though the analytic algebra above holds.
    env_corpus = validate_env_corpus(
        env_factory, "primary", corpus_n_obs, corpus_n_do, corpus_pool_do
    )
    control_corpus = validate_env_corpus(
        control_env_factory, "control", corpus_n_obs, corpus_n_do, corpus_pool_do
    )
    result.env_corpus = env_corpus
    result.control_corpus = control_corpus
    result.env_corpus_ok = env_corpus.ok
    result.control_corpus_ok = control_corpus.ok
    result.failures.extend(env_corpus.failures)
    result.failures.extend(control_corpus.failures)

    # (3) geometry bank.
    bank = build_bank(env_factory)
    result.feasible = bank.feasible
    result.n_states = len(bank.seeds)
    result.seeds_ok = bank.seeds == EXPECTED_SEEDS
    if not bank.feasible:
        result.failures.append(
            f"bank feasibility: retained {result.n_states}/{N_STATES} valid states"
        )
        return result
    if not result.seeds_ok:
        result.failures.append(f"bank seeds {bank.seeds[:8]}... != expected 0..{N_STATES - 1}")

    # (4)+(5) per-state primary gap and lambda=0 control gap.
    gaps, ctrl_gaps = [], []
    for seed in bank.seeds:
        te, ce = env_factory(seed), control_env_factory(seed)
        g, d, ao, an = _gap_norm(te, b_pool, c_pool)
        gc, _dc, _aoc, anc = _gap_norm(ce, b_pool_c, c_pool_c)
        gaps.append(g)
        ctrl_gaps.append(abs(gc))
        result.d_state, result.a_oracle = d, float(V_GRID[ao])
        result.a_naive, result.a_naive_control = float(V_GRID[an]), float(V_GRID[anc])

    result.mean_gap = float(np.mean(gaps))
    result.mean_control = float(np.mean(ctrl_gaps))

    # (6) acceptance — BOTH clauses gate.
    pass_primary = result.mean_gap >= TAU_E4
    pass_control = result.mean_control <= TOL
    if not pass_primary:
        result.failures.append(f"mean(gap_norm)={result.mean_gap:.6f} < tau_E4 {TAU_E4}")
    if not pass_control:
        result.failures.append(
            f"mean(|gap_norm_control|)={result.mean_control:.6f} > tol {TOL} "
            f"(Z->A_behavior not removed?)"
        )
    result.passed = (
        pass_primary
        and pass_control
        and result.adjacency_ok
        and result.latent_edges_ok
        and result.seeds_ok
        and result.env_corpus_ok
        and result.control_corpus_ok
        and result.single_factor_ok
        and not any("moment" in f or "b_pool" in f or "c_pool" in f for f in result.failures)
    )
    return result


def main() -> int:
    result = run_e4_structural_gate()
    print("E4 structural design-validation gate")
    print(f"  bank feasible:            {result.feasible} ({result.n_states}/{N_STATES}; "
          f"seeds 0..{N_STATES - 1}: {result.seeds_ok})")
    print(f"  b_pool (naive):           {result.b_pool:.12f}   (frozen {FROZEN_B_POOL})")
    print(f"  c_pool (naive):           {result.c_pool:.12f}   (frozen {FROZEN_C_POOL})")
    print(f"  b_pool (control, lam=0):  {result.b_pool_control:.12f}   (expect -1.0)")
    print(f"  a_oracle / a_naive:       {result.a_oracle:.2f} / {result.a_naive:.2f}   "
          f"(frozen {FROZEN_A_ORACLE:.2f} / {FROZEN_A_NAIVE:.2f})")
    print(f"  a_naive (control):        {result.a_naive_control:.2f}   (expect 0.00)")
    print(f"  D(s):                     {result.d_state:.12f}   (frozen {FROZEN_D_STATE})")
    print(f"  mean(gap_norm):           {result.mean_gap:.16f}   (tau_E4 = {TAU_E4})")
    print(f"  mean(|gap_norm_control|): {result.mean_control:.16f}   (tol    = {TOL})")
    print(f"  observed adjacency [(0,0)]: {result.adjacency_ok}   latent edges: "
          f"{result.latent_edges_ok}")
    if result.env_corpus is not None:
        ec = result.env_corpus
        print(f"  env-corpus (primary, N={CORPUS_N_OBS}+{CORPUS_N_DO}, stat_tol={STAT_TOL}): "
              f"{result.env_corpus_ok}")
        print(f"    {'quantity':<18}{'empirical':>16}{'frozen':>16}{'|diff|':>12}")
        for key in ec.frozen:
            emp, fro = ec.empirical[key], ec.frozen[key]
            print(f"    {key:<18}{emp:>16.6f}{fro:>16.6f}{abs(emp - fro):>12.6f}")
        print(f"    {'mech_err_obs':<18}{ec.empirical['mech_err_obs']:>16.2e}"
              f"{0.0:>16.2e}{ec.empirical['mech_err_obs']:>12.2e}")
        print(f"    {'mech_err_do':<18}{ec.empirical['mech_err_do']:>16.2e}"
              f"{0.0:>16.2e}{ec.empirical['mech_err_do']:>12.2e}")
    if result.control_corpus is not None:
        cc = result.control_corpus
        print(f"  env-corpus (control lam=0): {result.control_corpus_ok}   "
              f"pooled_slope emp={cc.empirical['pooled_slope']:.6f} (frozen -1.0)")
    if result.failures:
        print("  failures:")
        for msg in result.failures:
            print(f"    - {msg}")
    print("  NOTE: design validation only; trained-arm decision-value evaluation is DEFERRED.")
    print(f"RESULT: {'PASS' if result.passed else 'FAIL'}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
