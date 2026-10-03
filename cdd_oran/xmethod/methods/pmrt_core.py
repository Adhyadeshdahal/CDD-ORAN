"""PMRT core (``pmrt_core``): the design-based part of PMRT (docs/benchmark/METHOD_NAMES.md) for the one-step rows of
the cross-method study (E1-E5 x R1-R4; scratchpad/xmethod/CONTRACT.md, cdd_oran/xmethod/api.py).

Hypothesis (a, k), a an action column with a known design: "the random part of a's assignment at row t has no effect
on the KPI k at t + 1" (sharp in the random part; the design-based analogue of PMRT's H0(f, rel, kpi)). Candidates
whose source has no known design (a lagged KPI, a regime-R4 action) are NOT scorable: they are reported "not
applicable" (score NaN, p None) and never fall back to a row permutation.

Statistic, per action a (all targets on ONE set of B draws, as pmrt.family_tests_pmrt):
  assignment  v_t = A_t - E_design[A_t]: iid -> the centred action; dither -> the centred random part (the setpoint is
              not randomised and is never redrawn); logged -> the propensity-centred action (per-row categorical
              table, or per-row distribution parameters); none -> not applicable. Var_t = Var_design(A_t) (known).
  adjust      the rows in information order (``time_index`` if given, else row order). For every row t the
              predictable covariates X_t = [lagged KPIs (pre-state), observed context, the setpoints (fixed parts) of
              every dither action, the actions of rows t-1 .. t-L (L = hist_lags), the CONCURRENT values at t of the
              other designed actions (orchestrator ruling Q3: their designs are independent of a's given the observed
              context, true in R1-R3 as built)]; none of them is affected by a's draw at t. The covariates come from
              the shared builder ``cdd_oran.xmethod.covariates.design_covariates`` (ruling R-17: the same Z_eq for
              every equal-information method). Option ``covariates="r3"`` (ruling R-19 ablation) drops the
              setpoints and the lagged actions, leaving the R-3 set (lagged KPIs, context, concurrent actions).
              Per target k: ridge of
              Y_k on X fitted on the PAST rows only (an expanding window refitted at geometric block boundaries, lambda by GCV on those past rows),
              e_t = Y_tk - Yhat_tk (out of sample); the first ``burn`` rows get weight 0. w_t = e_t (default).
  robust      OFF by default (ruling R-14). Option huber_c = c: predictable Huber clip w_t = s_t clip(e_t / s_t, +- c),
              s_t = MAD scale of the past out-of-sample residuals (block-wise; the first block uses the in-sample
              residuals of its own past fit). Why off: with the clip, E2 R2 null rate .070 [.055, .087] at .05
              (n 1000, 40 seeds); the fixed-W CRT is only asymptotically valid when W depends on the focal action's
              PAST dithers (lagged-action / lagged-KPI covariates, ridge and clip scale refit on them), and the clip
              amplifies the finite-sample error. Without it .052 [.037, .067] (docs/xmethod/PMRT_CORE.md).
  test        S = sum_t v_t w_t, sd = sqrt(sum_t Var_t w_t^2), z = S / sd. Redraw ONLY a's random part from its
              known distribution (other columns fixed; W fixed), at most B = 9999 draws with Besag-Clifford
              sequential stopping (ruling R-9: stop at h = 20 exceedances, p = h / k; else
              p = (1 + #{|z_b| >= |z| - tol}) / (B + 1)); one-sided p_plus / p_minus likewise. seq_h = None gives
              pmrt's fixed-B p (used only for the equivalence gate). RNG
              ``default_rng([data.seed, 7801, action_index, split])`` (CONTRACT section 6), chunked, chunk-invariant.
  sign        sign(S) (the plain-kernel sign rule of pmrt).
  statistic   option (ruling R-42). "linear" (default; the secondary arm, version pmrt-core-v1, unchanged) is S above.
              "gbm" / "rff" / "poly" run the same redraw with a nonlinear statistic from
              ``cdd_oran.xmethod.methods.pmrt_nl`` (version pmrt-core-v2; option ``adjust`` = "ridge" | "gbm", None = the
              statistic's default). R-42 selection (2026-10-03, docs/xmethod/PMRT_CORE.md): "gbm" (the learned
              predictable matched filter, past-only gbm adjustment) is the primary PMRT arm pmrt_nl_eq.
  declare     fdr_layer.declare(..., "by") at q = .05 over every action -> KPI candidate incl. P_placebo (ruling
              R-6; a not-applicable one counts in m and is never declared; plain BY: the E-series has no
              independent prior data, so no weights and no one-sided directions). Secondary (CONTRACT section 5): tau on
              the score |z| from the shared placebo rule ``score.placebo_tau`` (R-29 conformal, ``tune``).
Validity: w_t is predictable (a function of rows < t, of pre-state and context at t, and of fixed parameters), v_t is
mean-zero given all of that under the design, so S is a martingale with predictable variance sum Var_t w_t^2 = the
variance of the CRT redraw; the CRT p-value is asymptotically valid (martingale CLT), exactly the PMRT argument
(pmrt.py docstring), not exact. It needs the design to be the TRUE assignment law of the random part (R1-R3).

PMRT components on E6-P -> here (the "PMRT core" text; also docs/xmethod/PMRT_CORE.md):
  assignment score (design_regressor, propensity-centred)          KEPT (generalised to iid / dither / logged)
  predictable adjustment (ridge on pre-window bins, ctx, hist)       KEPT, REDUCED: one time step, so pre-state = the
      lagged KPI vector; ridge fitted past-only on the same corpus (E6: on independent training episodes and frozen);
      hist = the actions of the previous L rows; no slots / time bins. EXTENDED by the concurrent values of the
      other actions (E6 excludes other families' units at t >= t0 because their existence can depend on earlier
      units; with independent row designs that cannot happen)
  running (past-only) stratum centres                                REDUCED to the intercept of the past-only ridge
      (no episode x sgn strata in a one-step row corpus)
  matched filter K = Sigma^-1 mu over receiving-cell slots x time bins, sign-constrained loadsp profile
                                                                     DROPPED (one target per row, no cells / bins: the
      kernel degenerates to the scalar 1, i.e. the "plain" arm)
  log-variance weights (_h)                                          DROPPED (not in the PMRT primary arm)
  predictable Huber clip (_c)                                        DROPPED by default (R-14: finite-sample over-
      rejection with the clip, see "robust"); kept as option huber_c (scale = past-residual MAD)
  CV arm selection (best), GBDT g-hat (best_gb), arm combos          DROPPED (no training episodes)
  CRT redraw of the logged propensities, B = 9999, asymptotic validity  KEPT (redraws only the random part)
  receiver table rho, support rule (>= 30 units, accept / reject counts, >= 3 episodes)
                                                                     DROPPED (E6 unit-table specific); replaced by
      "undetermined" for a non-finite / constant target or zero design variance
  declarations: weighted BY one-sided where the prior |z| >= 3 (wby1s)  REDUCED to plain BY (no independent prior)
"""
from __future__ import annotations

import dataclasses
import math
import time
from typing import Any

import numpy as np

from cdd_oran.decision import fdr_layer as FL
from cdd_oran.xmethod import api
from cdd_oran.xmethod.covariates import concurrent_indices, design_covariates, info_order

PMRT_CORE_VERSION = "pmrt-core-v1"                # statistic "linear" (default)
PMRT_CORE_V2_VERSION = "pmrt-core-v2"             # a nonlinear statistic (R-42, pmrt_nl)
STATISTICS = ("linear", "gbm", "rff", "poly")
RNG_TAG = 7801                                  # CONTRACT section 6 (pmrt_core stream)
NOT_APPLICABLE = "not_applicable"


@dataclasses.dataclass(frozen=True)
class PmrtCoreConfig:
    B: int = 9999                               # maximum B (ruling R-9)
    seq_h: int | None = 20                      # Besag-Clifford stopping at h exceedances (R-9); None = fixed B
    q: float = 0.05
    procedure: str = "by"
    huber_c: float | None = None                # R-14: no clip (pmrt "plain" arm); 2.5 = pmrt's "_c" clip (option)
    hist_lags: int = 2                          # previous rows' actions as predictable covariates
    concurrent_actions: bool = True             # other actions' values at t as covariates (orchestrator Q3)
    covariates: str = "eq"                      # "eq": Z_eq (R-17, default); "r3": R-3 set only (R-19 ablation)
    min_burn: int = 30                          # first rows with weight 0: max(min_burn, burn_per_cov x (d + 1))
    burn_per_cov: float = 1.0                   # ridge (GCV) is regularised, so n ~ d is usable
    growth: float = 1.25                        # expanding-window refit boundaries b_{i+1} = ceil(growth b_i)
    lams: tuple = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)   # ridge lambda x n grid (pmrt._ridge_gcv)
    chunk: int = 1000
    max_chunk_bytes: int = 64_000_000
    split: int = 0
    tau: float | None = None                    # secondary score threshold (placebo rule, from tune)
    statistic: str = "linear"                   # R-42: "linear" (S = sum v w) | "gbm" | "rff" | "poly" (pmrt_nl)
    adjust: str | None = None                   # nonlinear statistics only: "ridge" | "gbm" | None (their default)


# ============================================================================================ assignment (design)
@dataclasses.dataclass
class Assignment:
    """v (n,) observed centred assignment, var (n,) its design variance, draw(rng, b) -> (b, n) centred redraws.
    Arrays in the caller's row order."""
    v: np.ndarray
    var: np.ndarray
    draw: Any
    kind: str


def _rowparam(x, n):
    a = np.asarray(x, float)
    return np.full(n, float(a)) if a.ndim == 0 else a.reshape(n)


def _categorical(values, probs, n) -> tuple:
    """(values (m,) | (n, m), probs (n, m)) -> (mean (n,), var (n,), draw). Inverse-CDF draws exactly as
    crt_units.PiAssignment.mode_draws (cumsum, last edge +inf, U >= edge counting)."""
    P = np.asarray(probs, float)
    if P.ndim == 1:
        P = np.broadcast_to(P, (n, len(P)))
    vals = np.asarray(values, float)
    if vals.ndim == 1:
        mean = P @ vals
        var = P @ vals ** 2 - mean ** 2
    else:
        mean = (P * vals).sum(1)
        var = (P * vals ** 2).sum(1) - mean ** 2
    cum = np.cumsum(P, axis=1)
    cum[:, -1] = np.inf

    edges = np.ascontiguousarray(cum[:, :-1])
    shared = bool(np.all(edges == edges[:1]))

    def draw(rng, b):
        U = rng.random((int(b), n))
        if edges.shape[1] <= 8:                     # M = #{m < last: U >= cum_m}, as PiAssignment.mode_draws
            M = np.zeros(U.shape, np.int64)
            for m in range(edges.shape[1]):
                M += U >= edges[None, :, m]
        elif shared:                                # many values (E4 grid, 101): the same count by bisection
            M = np.searchsorted(edges[0], U, side="right")
        else:
            M = np.empty(U.shape, np.int64)
            for j in range(n):
                M[:, j] = np.searchsorted(edges[j], U[:, j], side="right")
        V = vals[M] if vals.ndim == 1 else np.take_along_axis(np.broadcast_to(vals, (int(b),) + vals.shape),
                                                               M[:, :, None], 2)[:, :, 0]
        V = V.astype(float)
        V -= mean[None, :]
        return V
    return mean, np.maximum(var, 0.0), draw


def _continuous(dist: dict, n: int) -> tuple:
    """Uniform / normal with scalar or per-row parameters -> (mean, var, draw)."""
    name = dist.get("name")
    if name == "uniform":
        lo, hi = _rowparam(dist["lo"], n), _rowparam(dist["hi"], n)
        mean, var = (lo + hi) / 2.0, (hi - lo) ** 2 / 12.0

        def draw(rng, b):
            return rng.random((int(b), n)) * (hi - lo)[None, :] + (lo - mean)[None, :]
        return mean, var, draw
    if name == "normal":
        loc, sc = _rowparam(dist.get("loc", dist.get("mean", 0.0)), n), _rowparam(dist.get("scale", dist.get("sd")), n)

        def draw(rng, b):
            return rng.standard_normal((int(b), n)) * sc[None, :]
        return loc, sc ** 2, draw
    if name == "clipped_normal":                    # clip(N(mean, sd^2), lo, hi): atoms at the bounds (harness J6)
        from scipy.stats import norm
        m, sd = _rowparam(dist["mean"], n), _rowparam(dist["sd"], n)
        lo, hi = _rowparam(dist.get("lo", -np.inf), n), _rowparam(dist.get("hi", np.inf), n)
        a, b = (lo - m) / sd, (hi - m) / sd
        Fa, Fb, fa, fb = norm.cdf(a), norm.cdf(b), norm.pdf(a), norm.pdf(b)
        lo0, hi0 = np.where(np.isfinite(lo), lo, 0.0), np.where(np.isfinite(hi), hi, 0.0)
        a0, b0 = np.where(np.isfinite(a), a, 0.0), np.where(np.isfinite(b), b, 0.0)
        mean = lo0 * Fa + hi0 * (1 - Fb) + m * (Fb - Fa) + sd * (fa - fb)
        ex2 = (lo0 ** 2 * Fa + hi0 ** 2 * (1 - Fb) + (m ** 2 + sd ** 2) * (Fb - Fa) + 2 * m * sd * (fa - fb)
               + sd ** 2 * (a0 * fa - b0 * fb))

        def draw(rng, b):
            return np.clip(rng.standard_normal((int(b), n)) * sd[None, :] + m[None, :], lo[None, :], hi[None, :])                 - mean[None, :]
        return mean, np.maximum(ex2 - mean ** 2, 0.0), draw
    raise ValueError(f"unsupported design distribution {name!r}")


def assignment(design: api.Design, a: np.ndarray) -> Assignment | str:
    """The centred assignment of one action column, or the reason it is not scorable."""
    n = len(a)
    kind = design.kind
    if kind == "none" or design.dist is None:
        return "no known design (kind 'none')" if kind == "none" else f"kind {kind!r} without a distribution"
    dist = dict(design.dist)
    if kind == "iid":
        x = np.asarray(a, float)
    elif kind == "dither":
        if design.random_part is None:
            return "dither design without random_part"
        x = np.asarray(design.random_part, float)
    elif kind == "logged":
        x = np.asarray(a, float)
        if dist.get("name") in ("categorical", "categorical_rows"):    # harness R3: "categorical_rows" (J4)
            if design.propensity is None:
                return "logged categorical design without propensity"
            mean, var, draw = _categorical(dist["values"], design.propensity, n)
            return Assignment(x - mean, var, draw, kind)
    else:
        return f"unknown design kind {kind!r}"
    if dist.get("name") == "categorical":
        mean, var, draw = _categorical(dist["values"], dist["p"], n)
    else:
        try:
            mean, var, draw = _continuous(dist, n)
        except (KeyError, ValueError) as ex:                 # unknown law: not scorable (never a permutation)
            return f"design distribution not redrawable: {ex}"
    return Assignment(x - mean, var, draw, kind)


# ============================================================================================ predictable adjustment
COVARIATE_SETS = ("eq", "r3")


def covariates(data: api.Dataset, order: np.ndarray, hist_lags: int, which: str = "eq") -> np.ndarray:
    """(n, d) predictable covariates in information order (module docstring, "adjust"), from the shared builder.
    ``which`` "eq": lagged KPIs, context, setpoints, lagged actions (Z_eq); "r3": lagged KPIs and context only."""
    if which not in COVARIATE_SETS:
        raise ValueError(f"covariates must be one of {COVARIATE_SETS}, got {which!r}")
    eq = which == "eq"
    _, M, _ = design_covariates(data, include_setpoints=eq, include_lagged_actions=eq, lags=hist_lags)
    return M[order]


def concurrent_columns(data: api.Dataset, ai: int) -> list:
    """Indices of the other action columns whose values at t may adjust the focal action ai's test: every designed
    column (kind iid / dither / logged). Valid when the designs are independent of the focal design given the
    observed context (true in R1, R2 and R3 as built: orchestrator ruling Q3). A kind-"none" column is never used
    (its assignment law is unknown) and a kind-"none" focal action is never tested. Shared rule:
    ``covariates.concurrent_indices(..., "designed")``."""
    return concurrent_indices(data, ai, "designed")


def _ridge_fit(X, Y, lams):
    """Per-target ridge on X (standardised here, intercept unpenalised), lambda (x n) by GCV per target column.
    Returns predict(Xnew) -> (m, k) and the in-sample residuals."""
    n, d = X.shape
    mx = X.mean(0) if n else np.zeros(d)
    sx = X.std(0) if n else np.ones(d)
    keep = sx > 1e-12
    my = Y.mean(0)
    Yc = Y - my
    if not keep.any():
        return (lambda Xn: np.broadcast_to(my, (len(Xn), Y.shape[1])).copy()), Yc
    Xs = (X[:, keep] - mx[keep]) / sx[keep]
    U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
    UtY = U.T @ Yc
    best_g = np.full(Y.shape[1], np.inf)
    coef = np.zeros((Xs.shape[1], Y.shape[1]))
    fit_best = np.zeros_like(Yc)
    for lam in lams:
        L = lam * n
        f = s ** 2 / (s ** 2 + L)
        fit = U @ (f[:, None] * UtY)
        rss = ((Yc - fit) ** 2).sum(0)
        g = rss / max(n - float(f.sum()), 1.0) ** 2
        better = g < best_g
        if better.any():
            best_g[better] = g[better]
            coef[:, better] = (Vt.T @ ((s / (s ** 2 + L))[:, None] * UtY))[:, better]
            fit_best[:, better] = fit[:, better]
    mxk, sxk = mx[keep], sx[keep]

    def predict(Xn):
        return my[None, :] + ((Xn[:, keep] - mxk) / sxk) @ coef
    return predict, Yc - fit_best


def _mad(r: np.ndarray) -> np.ndarray:
    med = np.median(r, 0)
    s = 1.4826 * np.median(np.abs(r - med), 0)
    rms = np.sqrt(np.mean(r ** 2, 0))
    return np.where(s > 0, s, np.where(rms > 0, rms, 1.0))


def predictable_weights(X: np.ndarray, Y: np.ndarray, cfg: PmrtCoreConfig) -> tuple[np.ndarray, dict]:
    """(W (n, k), info): predictable (past-only) adjusted and clipped residual weights, information order."""
    n, k = Y.shape
    d = X.shape[1]
    burn = int(max(cfg.min_burn, math.ceil(cfg.burn_per_cov * (d + 1))))
    W = np.zeros((n, k))
    E = np.zeros((n, k))
    if burn >= n:
        return W, {"burn": burn, "n_refits": 0, "d": d}
    bounds = [burn]
    while bounds[-1] < n:
        bounds.append(min(n, max(bounds[-1] + 1, int(math.ceil(bounds[-1] * cfg.growth)))))
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        predict, res_in = _ridge_fit(X[:a], Y[:a], cfg.lams)
        e = Y[a:b] - predict(X[a:b])
        E[a:b] = e
        if cfg.huber_c is None:
            W[a:b] = e
            continue
        scale = _mad(E[burn:a]) if a - burn >= 10 else _mad(res_in)
        W[a:b] = scale[None, :] * np.clip(e / scale[None, :], -cfg.huber_c, cfg.huber_c)
    return W, {"burn": burn, "n_refits": len(bounds) - 1, "d": d}


# ============================================================================================ CRT
def crt(v, var, W, draw, rng, B: int, chunk: int = 1000, max_chunk_bytes: int = 64_000_000,
        seq_h: int | None = None) -> dict:
    """Design-based CRT of sum v w per column of W (one set of redraws shared by every column). Exceedance arithmetic
    of pmrt_bench.integrated_family for an o = +1 (plain-type) arm. Returns z, p2, p_plus, p_minus (each (k,)) and
    the number of draws each p used.
    seq_h None: fixed B, p = (1 + count) / (B + 1) (pmrt native). seq_h = h: Besag-Clifford (1991) sequential
    stopping per statistic (orchestrator ruling R-9): a statistic stops at the draw k where its count reaches h,
    p = h / k; one that never does gets (1 + count) / (B + 1). Drawing ends when every statistic has stopped or at B.
    Draws are consumed one at a time in stream order, so both modes are chunk-invariant."""
    n = len(v)
    m = W.shape[1]
    sd = np.sqrt(var @ (W * W))
    sd_safe = np.where(sd > 0, sd, 1.0)
    z_obs = (v @ W) / sd_safe
    tol = 1e-9 * (1.0 + np.abs(z_obs))
    ch = int(max(1, min(chunk, max_chunk_bytes // (8 * max(n, 1)))))
    cnt = np.zeros(3 * m, np.int64)                 # [two-sided | plus | minus] x columns
    k_stop = np.zeros(3 * m, np.int64)              # 0 = not stopped
    done = 0
    while done < B and not (seq_h is not None and np.all(k_stop > 0)):
        b = min(ch, B - done)
        V = draw(rng, b)
        Zb = (V @ W) / sd_safe[None, :]
        del V
        E = np.concatenate([np.abs(Zb) >= (np.abs(z_obs) - tol)[None, :], Zb >= (z_obs - tol)[None, :],
                            Zb <= (z_obs + tol)[None, :]], axis=1)
        if seq_h is None:
            cnt += np.count_nonzero(E, axis=0)
        else:
            cs = cnt[None, :] + np.cumsum(E, axis=0, dtype=np.int64)
            hit = (k_stop == 0) & (cs[-1] >= seq_h)
            if hit.any():
                k_stop[hit] = done + 1 + np.argmax(cs[:, hit] >= seq_h, axis=0)
            cnt = cs[-1]
        done += b
    B1 = B + 1.0
    if seq_h is None:
        p = (1 + cnt) / B1
        used = np.full(3 * m, done)
    else:
        p = np.where(k_stop > 0, seq_h / np.maximum(k_stop, 1), (1 + cnt) / B1)
        used = np.where(k_stop > 0, k_stop, done)
    return {"z": z_obs, "sd": sd, "p2": p[:m], "p_plus": p[m:2 * m], "p_minus": p[2 * m:], "draws": used[:m],
            "draws_total": done}


# ============================================================================================ method
def config_from_dict(cfg: dict | None) -> PmrtCoreConfig:
    cfg = dict(cfg or {})
    fields = {f.name for f in dataclasses.fields(PmrtCoreConfig)}
    kw = {k: (tuple(v) if k == "lams" else v) for k, v in cfg.items() if k in fields}
    return PmrtCoreConfig(**kw)


class PmrtCore:
    """api.Method implementation of the PMRT core (module docstring)."""
    name = "pmrt_core"
    version = PMRT_CORE_VERSION

    def __init__(self, cfg: PmrtCoreConfig | None = None):
        self.cfg = cfg or PmrtCoreConfig()

    # ---------------------------------------------------------------------------------------- tune
    def tune(self, dev: list[api.Dataset], truth_free: bool = True) -> dict[str, Any]:
        """Frozen config = the defaults (no per-method tuning: CONTRACT section 5) + the secondary score threshold
        tau = the shared rule ``score.placebo_tau`` over ``dev`` (one (world, regime, n) cell): the R-29 conformal .05
        cutoff of the pooled ``P_placebo`` scores (R-38: replaced the 2nd-largest score). Truth is never read."""
        from cdd_oran.xmethod.score import placebo_tau, tau_rule_name
        cfg = dataclasses.asdict(self.cfg)
        results = [self.run(d, cfg) for d in dev]
        s = sorted((e.score for r in results for e in r.edges if e.source == "P_placebo" and np.isfinite(e.score)),
                   reverse=True)
        cfg["tau"] = float(placebo_tau(results))
        cfg["tau_rule"] = tau_rule_name()
        cfg["tune_info"] = {"n_dev": len(dev), "n_placebo_scores": len(s), "placebo_top3": s[:3], "truth_free": True}
        return cfg

    # ---------------------------------------------------------------------------------------- run
    def run(self, data: api.Dataset, config: dict[str, Any] | None = None, rngs: dict | None = None,
            W_override: np.ndarray | None = None) -> api.Result:
        """Score every candidate. ``rngs`` ({action: Generator}) and ``W_override`` ((n, k) weights in information
        order) exist only for the equivalence check against pmrt.py (scratchpad/xmethod/pmrt_equiv.py)."""
        t_cpu = time.process_time()
        cfg = config_from_dict(config) if config is not None else self.cfg
        if cfg.statistic not in STATISTICS:
            raise ValueError(f"statistic must be one of {STATISTICS}, got {cfg.statistic!r}")
        if cfg.statistic != "linear":
            return self._run_nonlinear(data, cfg, t_cpu, rngs, W_override)
        if cfg.adjust is not None:
            raise ValueError("adjust applies to the nonlinear statistics only (linear uses the ridge residual)")
        order = info_order(data)
        kpis = list(data.kpi_names)
        Y = np.asarray(data.Y, float)[order]
        bad_t = ~np.all(np.isfinite(Y), 0) | (np.nanstd(Y, 0) <= 0)
        Yf = np.where(np.isfinite(Y), Y, 0.0)
        if cfg.covariates not in COVARIATE_SETS:
            raise ValueError(f"covariates must be one of {COVARIATE_SETS}, got {cfg.covariates!r}")
        eq = cfg.covariates == "eq"
        cov_names, Xc = (), None
        if W_override is None:
            cov_names, M, _ = design_covariates(data, include_setpoints=eq, include_lagged_actions=eq,
                                                lags=cfg.hist_lags)
            Xc = M[order]
        A_ord = np.asarray(data.X_action, float)[order]
        W_shared, winfo = None, {}
        per_action: dict[str, dict] = {}
        na: dict[str, str] = {}
        for ai, a in enumerate(data.action_names):
            asg = assignment(data.designs[ai], np.asarray(data.X_action, float)[:, ai])
            if isinstance(asg, str):
                na[a] = asg
                continue
            if not np.any(asg.var > 0):
                na[a] = "zero design variance"
                continue
            if W_override is not None:
                W, winfo[a] = np.asarray(W_override, float), {"override": True}
            else:
                conc = concurrent_columns(data, ai) if cfg.concurrent_actions else []
                if conc:
                    W, winfo[a] = predictable_weights(np.column_stack([Xc, A_ord[:, conc]]), Yf, cfg)
                    winfo[a]["concurrent"] = [data.action_names[j] for j in conc]
                else:                                   # no concurrent covariates: one fit shared by such actions
                    if W_shared is None:
                        W_shared, info0 = predictable_weights(Xc, Yf, cfg)
                    W, winfo[a] = W_shared, dict(info0, concurrent=[])
            W = np.where(bad_t[None, :], 0.0, W)
            rng = (rngs or {}).get(a) or np.random.default_rng([int(data.seed), RNG_TAG, ai, int(cfg.split)])
            draw = asg.draw

            def draw_ordered(r, b, draw=draw):
                return draw(r, b)[:, order]
            per_action[a] = crt(asg.v[order], asg.var[order], W, draw_ordered, rng, cfg.B, cfg.chunk,
                                cfg.max_chunk_bytes, cfg.seq_h)
        # ---- per candidate
        stats, rows = {}, []
        for src, tgt in data.candidates:
            if src not in per_action:
                why = na.get(src, "source is not an action with a known design")
                rows.append((src, tgt, None, why))
                continue
            k = kpis.index(tgt)
            if bad_t[k]:
                rows.append((src, tgt, None, "degenerate target"))
                continue
            r = per_action[src]
            st = FL.HypStats(p2=float(r["p2"][k]), sign=int(np.sign(r["z"][k])), p_plus=float(r["p_plus"][k]),
                             p_minus=float(r["p_minus"][k]))
            stats[(src, tgt)] = st
            rows.append((src, tgt, (float(abs(r["z"][k])), st), None))
        # BY family (ruling R-6): the primary action -> KPI candidates incl. P_placebo (harness meta
        # "primary_candidates" when present; a not-applicable one counts in m). Diagnostic candidates (harness
        # P_placebo_conf) get their own separate BY so they never change a primary declaration.
        acts, kset = set(data.action_names), set(kpis)
        family = [tuple(h) for h in data.candidates if h[0] in acts and h[1] in kset]
        if data.meta.get("primary_candidates") is not None:
            prim = {tuple(h) for h in data.meta["primary_candidates"]}
            family, diag = [h for h in family if h in prim], [h for h in family if h not in prim]
        else:
            diag = []
        dec = FL.declare(stats, cfg.procedure, q=cfg.q, hyps=family) if stats and family else {}
        if diag and any(h in stats for h in diag):
            dec.update(FL.declare(stats, cfg.procedure, q=cfg.q, hyps=diag))
        edges, reasons, tau_decl = [], {}, []
        for src, tgt, val, why in rows:
            if val is None:
                edges.append(api.EdgeResult(src, tgt, float("nan"), None, 0, False))
                reasons[f"{src}->{tgt}"] = why
                continue
            score, st = val
            d = dec.get((src, tgt), {"declared": False})
            edges.append(api.EdgeResult(src, tgt, score, st.p2, int(st.sign), bool(d["declared"])))
            if cfg.tau is not None and score > cfg.tau:
                tau_decl.append(f"{src}->{tgt}")
        notes = {
            NOT_APPLICABLE: reasons,
            "z": {f"{s}->{t}": float(per_action[s]["z"][kpis.index(t)]) for s, t in data.candidates
                  if s in per_action},
            "p_plus": {f"{s}->{t}": float(per_action[s]["p_plus"][kpis.index(t)]) for s, t in data.candidates
                       if s in per_action},
            "p_minus": {f"{s}->{t}": float(per_action[s]["p_minus"][kpis.index(t)]) for s, t in data.candidates
                        if s in per_action},
            "adjust": winfo, "covariates": {"set": cfg.covariates, "base": list(cov_names)},
            "n_tested": len(stats), "by_family_m": len(family), "by_diag_m": len(diag), "declared_tau": tau_decl if cfg.tau is not None else None,
            "rng": f"default_rng([seed, {RNG_TAG}, action_index, {cfg.split}])",
        }
        return api.Result(method=self.name, version=self.version, edges=tuple(edges),
                          cpu_s=time.process_time() - t_cpu, config=dataclasses.asdict(cfg), notes=notes)

    def _run_nonlinear(self, data, cfg, t_cpu, rngs, W_override) -> api.Result:
        """statistic != "linear": the pmrt_nl statistic on the same redraw machinery and settings (B, Besag-Clifford,
        BY family, covariates, burn, refit blocks, RNG stream, tau); the Result is reported as pmrt_core v2."""
        from cdd_oran.xmethod.methods import pmrt_nl as NL
        if cfg.huber_c is not None or rngs is not None or W_override is not None:
            raise ValueError("huber_c / rngs / W_override apply to the linear statistic only")
        shared = {f.name for f in dataclasses.fields(NL.PmrtNlConfig)} & {f.name for f in dataclasses.fields(cfg)}
        res = NL.PmrtNl(NL.PmrtNlConfig(**{k: getattr(cfg, k) for k in shared})).run(data)
        notes = dict(res.notes, statistic_module=f"{NL.__name__} {NL.PMRT_NL_VERSION}")
        return api.Result(method=self.name, version=PMRT_CORE_V2_VERSION, edges=res.edges,
                          cpu_s=time.process_time() - t_cpu, config=dataclasses.asdict(cfg), notes=notes)


def make() -> PmrtCore:
    return PmrtCore()


__all__ = ["COVARIATE_SETS", "NOT_APPLICABLE", "PMRT_CORE_V2_VERSION", "PMRT_CORE_VERSION", "RNG_TAG", "STATISTICS",
           "Assignment", "PmrtCore", "PmrtCoreConfig", "assignment", "concurrent_columns", "config_from_dict",
           "covariates", "crt", "info_order", "make", "predictable_weights"]
