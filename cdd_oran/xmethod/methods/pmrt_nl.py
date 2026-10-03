"""Nonlinear PMRT statistics (ruling R-42): the PMRT core redraw (``pmrt_core``: the known random part of one action is
redrawn, every other column and every non-random part stay fixed; B = 9999 with Besag-Clifford stopping, R-9; BY,
R-2 / R-6; no clip, R-14) with a statistic that can see non-monotone and state-dependent effects.

Why (scratchpad/xmethod/results/pmrt_nl/DIAG.md): the linear S = sum v_t w_t misses (1) bumps centred in the design
range (Cov(v, Y) = 0; under setpoint + dither the local slopes cancel across setpoints), (2) effects that exist only in
interaction with the concurrent actions or a gate, and (3) loses power because the linear ridge leaves the other
actions' nonlinear effects in w.

Notation (rows in information order, focal action a, target k): v_t the centred random part (pmrt_core.assignment),
x_t the predictable covariates (pmrt_core's Z_eq: lagged KPIs, context, setpoints, lagged actions) and the concurrent
values of the other designed actions (R-8), y_t the target. "Predictable" = a function of rows < t and of x_t only.

Adjustment (``adjust``):
  ridge  w_t = the pmrt_core past-only ridge residual (pmrt_core.predictable_weights, unchanged).
  gbm    one gradient-boosted model g_k(A_t, x_t) per target, A_t = every designed action at t (raw values), fitted on
         the PAST rows only and refitted at pmrt_core's geometric block boundaries (expanding window); for focal a,
         m_t = E_design[g_k(A_t with a's random part redrawn, x_t)] and w_t = y_t - m_t (a nonlinear predictable
         adjustment that integrates a's own draw out). The first ``burn`` rows get weight 0 (pmrt_core's rule).
Statistic (``statistic``), each a function T(v) of the redrawn v with W and every other quantity fixed:
  poly   U = sum_t phi(u_t) (x) psi_t w_t, u_t = v_t / sd_t, phi = (u, u^2, u^3) centred by their design means, psi_t
         = [1] (+ [s_t, s_t^2 - mean] for a dither design, s_t its standardised setpoint); T = U' Sigma^+ U with Sigma =
         sum_t w_t^2 Cov_design(phi_t) (x) psi_t psi_t' (the redraw covariance, known): bumps and setpoint-dependent
         slopes.
  rff    the same U with phi = 10 random Fourier features of u (Gaussian kernel, length 1), centred by their design
         means, and psi_t = [1, 20 random Fourier features of the standardised state] (state = the concurrent designed
         actions, the focal setpoint, the context; Gaussian kernel, length sqrt(dim)); T = sum_j U_j^2 / Sigma_jj
         (an HSIC-type statistic of v and w given the state): interactions with the concurrent actions.
  gbm    the learned predictable matched filter: h_t(v) = g_k(A_t with a's random part = v, x_t) - m_t (the profile
         of the past-only model above, linearly interpolated on a grid of ``gbm_grid`` points over the design's
         support, or exact on a categorical support); T = sum_t h_t(v_t) w_t / sqrt(sum_t Var_design(h_t) w_t^2)
         (PMRT's matched filter learned from the past instead of a fixed kernel).
  Every T is one-sided (large = evidence); p = Besag-Clifford over at most B redraws, as pmrt_core.crt. The declared
  sign is sign(sum_t v_t w_t) (the linear PMRT sign rule on the same w).

Validity (one argument for every option). Conditional on everything except a's random parts, the CRT redraws v from
its known design with W, psi, h and Sigma held fixed. (i) Exact case: if W, h, psi do not depend on a's random parts
(no lagged focal action among the covariates, no KPI dynamics driven by a), T(v_obs) and T(v_b) have the same law
under H0 and the p-value is exact (finite-sample, any statistic). (ii) General case, as pmrt_core: W, h are
predictable but may depend on a's PAST draws (lagged actions, lagged KPIs a drives, model refits). Every option is
built from the martingale-difference arrays d_t = phi~(v_t) (x) psi_t w_t (poly, rff) or h_t(v_t) w_t (gbm): the
factor in v_t has design mean zero given F_{t-1} and x_t (it is centred by the exact design expectation, or by a
quadrature of it for continuous non-uniform designs), and v_t is independent of F_{t-1}, x_t and, under H0 (no effect
of v_t on y_t), of y_t. So U (or the scalar sum) is a martingale whose predictable covariance equals the covariance of
the redraw distribution; by the martingale CLT the observed and the redraw law of the (continuous) statistic agree
asymptotically, and the CRT p-value is asymptotically valid. Not exact in general (as pmrt_core). The gbm model is
fitted on past rows only, which is what keeps h and w predictable (a cross-fitted model would use future rows whose
KPIs can depend on v_t; orchestrator ruling on Q1, 2026-10-03).
"""
from __future__ import annotations

import dataclasses
import math
import time
from typing import Any

import numpy as np

from cdd_oran.decision import fdr_layer as FL
from cdd_oran.xmethod import api
from cdd_oran.xmethod.covariates import DESIGNED_KINDS, design_covariates, info_order
from cdd_oran.xmethod.methods import pmrt_core as PC

PMRT_NL_VERSION = "pmrt-nl-v1"
STATISTICS = ("poly", "rff", "gbm")
ADJUSTMENTS = ("ridge", "gbm")
RFF_TAG = 78011                                  # fixed feature stream (not data-seeded): the statistic is a constant


@dataclasses.dataclass(frozen=True)
class PmrtNlConfig:
    statistic: str = "gbm"
    adjust: str | None = None                   # None: "gbm" for statistic gbm, else "ridge"
    B: int = 9999
    seq_h: int | None = 20
    q: float = 0.05
    procedure: str = "by"
    hist_lags: int = 2
    concurrent_actions: bool = True
    covariates: str = "eq"
    min_burn: int = 30
    burn_per_cov: float = 1.0
    growth: float = 1.25
    lams: tuple = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    chunk: int = 1000
    max_chunk_bytes: int = 64_000_000
    split: int = 0
    tau: float | None = None
    poly_degree: int = 3
    rff_dv: int = 10
    rff_dx: int = 20
    gbm_trees: int = 200
    gbm_depth: int = 4
    gbm_eta: float = 0.1
    gbm_grid: int = 25
    n_quad: int = 48

    @property
    def adjustment(self) -> str:
        return self.adjust or ("gbm" if self.statistic == "gbm" else "ridge")


# ============================================================================================ design law
@dataclasses.dataclass
class Law:
    """Assignment (pmrt_core) + what the nonlinear statistics need, every array in INFORMATION order.
    raw value of the action = off + v. nodes / wq (R, Q), R in {1, n}: quadrature of the centred design (exact for
    categorical; Gauss-Legendre / -Hermite otherwise). Grid for the gbm profile: kind "cont" (centred-v grid from glo
    to ghi per row, G points) or "cat" (raw support values, probs (R, m))."""
    v: np.ndarray
    var: np.ndarray
    draw: Any
    off: np.ndarray
    nodes: np.ndarray
    wq: np.ndarray
    dist: str
    glo: np.ndarray | None = None
    ghi: np.ndarray | None = None
    support: np.ndarray | None = None
    probs: np.ndarray | None = None


def _row(x, n):
    return PC._rowparam(x, n)


def design_law(design: api.Design, col: np.ndarray, order: np.ndarray, n_quad: int) -> Law | str:
    n = len(col)
    asg = PC.assignment(design, col)
    if isinstance(asg, str):
        return asg
    dist = dict(design.dist)
    name = dist.get("name")
    off = np.asarray(col, float) - asg.v
    o = order
    if name in ("categorical", "categorical_rows"):
        vals = np.asarray(dist["values"], float)
        P = np.asarray(design.propensity if name == "categorical_rows" else dist["p"], float)
        P = P[None, :] if P.ndim == 1 else P[o]
        mean = off[o]
        if vals.ndim != 1:
            return "per-row categorical values are not supported by pmrt_nl"
        srt = np.argsort(vals, kind="stable")
        nodes = vals[None, :] - mean[:, None] if P.shape[0] > 1 or np.ptp(mean) > 0 else vals[None, :] - mean[:1, None]
        return Law(asg.v[o], asg.var[o], asg.draw, off[o], nodes, np.broadcast_to(P, (nodes.shape[0], len(vals))),
                   name, support=vals[srt], probs=P[:, srt])
    if name == "uniform":
        lo, hi = _row(dist["lo"], n)[o], _row(dist["hi"], n)[o]
        xi, w = np.polynomial.legendre.leggauss(n_quad)
        half = (hi - lo) / 2.0
        R = 1 if np.ptp(half) == 0 else n
        nodes = half[:R, None] * xi[None, :]
        return Law(asg.v[o], asg.var[o], asg.draw, off[o], nodes, np.broadcast_to(w / 2.0, nodes.shape), name,
                   glo=-half, ghi=half)
    if name == "normal":
        sc = np.sqrt(asg.var[o])
        xi, w = np.polynomial.hermite_e.hermegauss(n_quad)
        R = 1 if np.ptp(sc) == 0 else n
        nodes = sc[:R, None] * xi[None, :]
        return Law(asg.v[o], asg.var[o], asg.draw, off[o], nodes, np.broadcast_to(w / w.sum(), nodes.shape), name,
                   glo=-4 * sc, ghi=4 * sc)
    if name == "clipped_normal":
        from scipy.stats import norm
        m, sd = _row(dist["mean"], n)[o], _row(dist["sd"], n)[o]
        lo = _row(dist.get("lo", -np.inf), n)[o]
        hi = _row(dist.get("hi", np.inf), n)[o]
        lo_f, hi_f = np.maximum(lo, m - 8 * sd), np.minimum(hi, m + 8 * sd)
        xi, w = np.polynomial.legendre.leggauss(n_quad)
        mid, half = (lo_f + hi_f) / 2, (hi_f - lo_f) / 2
        x = mid[:, None] + half[:, None] * xi[None, :]
        dens = norm.pdf((x - m[:, None]) / sd[:, None]) / sd[:, None] * half[:, None] * w[None, :]
        Fa, Fb = norm.cdf((lo - m) / sd), norm.cdf((hi - m) / sd)
        xs = np.column_stack([lo_f, x, hi_f])
        ws = np.column_stack([np.where(np.isfinite(lo), Fa, 0.0), dens, np.where(np.isfinite(hi), 1 - Fb, 0.0)])
        ws = ws / ws.sum(1, keepdims=True)
        mean = off[o]
        R = 1 if (np.ptp(m) == 0 and np.ptp(sd) == 0) else n
        return Law(asg.v[o], asg.var[o], asg.draw, off[o], (xs - mean[:, None])[:R], ws[:R], name,
                   glo=lo_f - mean, ghi=hi_f - mean)
    return f"design distribution {name!r} not supported by pmrt_nl"


def _expect(law: Law, f_nodes: np.ndarray) -> np.ndarray:
    """E_design f per row from f evaluated at the nodes: f_nodes (R, Q, ...) -> (R, ...)."""
    wq = law.wq.reshape(law.wq.shape + (1,) * (f_nodes.ndim - 2))
    return (wq * f_nodes).sum(1)


# ============================================================================================ feature statistics
def _poly(u: np.ndarray, deg: int) -> np.ndarray:
    return np.stack([u ** j for j in range(1, deg + 1)], -1)


class _Features:
    """phi(v) for the poly / rff statistics: (rows, J) features of v, their design means and covariances."""

    def __init__(self, law: Law, kind: str, deg: int, dv: int, ai: int):
        self.law, self.kind = law, kind
        self.sd = np.sqrt(np.maximum(law.var, 0.0))
        self.sd_safe = np.where(self.sd > 0, self.sd, 1.0)
        if kind == "rff":
            r = np.random.default_rng([RFF_TAG, 1, ai])
            self.om, self.ph = r.standard_normal(dv), r.uniform(0, 2 * np.pi, dv)
        else:
            self.deg = deg
        R = law.nodes.shape[0]
        Fn = self.raw(law.nodes, self.sd_safe[:R, None])                     # (R, Q, J)
        self.mean = _expect(law, Fn)                                          # (R, J)
        self.cov = _expect(law, Fn[..., :, None] * Fn[..., None, :]) - self.mean[:, :, None] * self.mean[:, None, :]

    def raw(self, v, sd):
        u = v / sd
        if self.kind == "rff":
            return np.sqrt(2.0) * np.cos(u[..., None] * self.om + self.ph)
        return _poly(u, self.deg)

    def centred(self, V):                                                     # V (..., n) -> (..., n, J)
        F = self.raw(V, self.sd_safe)
        return (F - self.mean[None] if F.ndim == 3 else F - self.mean) * (self.sd > 0)[..., None]


def _psi_setpoint(design: api.Design, order) -> np.ndarray:
    n = len(order)
    if design.kind != "dither" or design.fixed_part is None:
        return np.ones((n, 1))
    s = np.asarray(design.fixed_part, float)[order]
    sd = s.std()
    if not np.isfinite(sd) or sd <= 0:
        return np.ones((n, 1))
    s = (s - s.mean()) / sd
    return np.column_stack([np.ones(n), s, s ** 2 - np.mean(s ** 2)])


def _psi_rff(state: np.ndarray, dx: int, ai: int) -> np.ndarray:
    n = state.shape[0]
    if state.shape[1] == 0:
        return np.ones((n, 1))
    sd = state.std(0)
    keep = sd > 0
    X = (state[:, keep] - state[:, keep].mean(0)) / sd[keep]
    if X.shape[1] == 0:
        return np.ones((n, 1))
    r = np.random.default_rng([RFF_TAG, 2, ai, X.shape[1]])
    Om = r.standard_normal((X.shape[1], dx)) / math.sqrt(X.shape[1])
    c = r.uniform(0, 2 * np.pi, dx)
    return np.column_stack([np.ones(n), np.sqrt(2.0) * np.cos(X @ Om + c)])


class FeatureStat:
    """T(V) for poly (whitened quadratic form) / rff (diagonal-standardised sum of squares)."""

    def __init__(self, feats: _Features, psi: np.ndarray, W: np.ndarray, whiten: bool):
        self.f, self.whiten = feats, whiten
        n, L = psi.shape
        k = W.shape[1]
        self.k, self.L = k, L
        self.M = (psi[:, :, None] * W[:, None, :]).reshape(n, L * k)               # (n, L k)
        J = feats.mean.shape[1]
        self.J = J
        cov = feats.cov                                                          # (R, J, J)
        W2 = W ** 2
        self.A = []
        if not whiten:                                                           # diagonal of Sigma only
            cd = np.einsum("tjj->tj", cov)
            D = (np.einsum("j,tl,tk->kjl", cd[0], psi ** 2, W2) if cov.shape[0] == 1
                 else np.einsum("tj,tl,tk->kjl", cd, psi ** 2, W2)).reshape(k, J * L)
            for kk in range(k):
                d = D[kk]
                self.A.append(np.where(d > 0, 1.0 / np.sqrt(np.where(d > 0, d, 1.0)), 0.0))
        else:                                                                    # Sigma_k = sum_t w^2 C_t (x) psi psi'
            if cov.shape[0] == 1:
                P = np.einsum("tl,tm,tk->klm", psi, psi, W2)
                Sig = np.einsum("ij,klm->kiljm", cov[0], P)
            else:
                Sig = np.einsum("tij,tl,tm,tk->kiljm", cov, psi, psi, W2)
            Sig = Sig.reshape(k, J * L, J * L)
            for kk in range(k):
                S = Sig[kk]
                ev, Q = np.linalg.eigh((S + S.T) / 2)
                ok = ev > 1e-10 * max(ev.max(), 1e-300)
                self.A.append((Q[:, ok] / np.sqrt(ev[ok])).T)
        self.df = [a.shape[0] if whiten else int((a > 0).sum()) for a in self.A]

    def __call__(self, V: np.ndarray) -> np.ndarray:                            # V (b, n) -> T (b, k)
        F = self.f.centred(V)                                                    # (b, n, J)
        U = np.matmul(F.transpose(0, 2, 1), self.M)                              # (b, J, L k)
        U = U.reshape(len(V), self.J, self.L, self.k).transpose(0, 3, 1, 2).reshape(len(V), self.k, self.J * self.L)
        out = np.empty((len(V), self.k))
        for kk in range(self.k):
            Z = U[:, kk] @ self.A[kk].T if self.whiten else U[:, kk] * self.A[kk]
            out[:, kk] = (Z * Z).sum(1)
        return out


# ============================================================================================ gbm (learned filter)
def _gbm_fit(X, y, cfg: PmrtNlConfig):
    import xgboost as xgb
    params = {"max_depth": cfg.gbm_depth, "eta": cfg.gbm_eta, "nthread": 1, "tree_method": "hist", "seed": 0,
              "objective": "reg:squarederror", "verbosity": 0}
    return xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=cfg.gbm_trees)


def _grid(law: Law, G: int) -> np.ndarray:
    """(n, G') centred-v grid values per row (cont: G equally spaced points; cat: the support minus the row mean)."""
    if law.support is not None:
        return law.support[None, :] - (law.off[:, None])
    s = np.linspace(0.0, 1.0, G)
    return law.glo[:, None] + (law.ghi - law.glo)[:, None] * s[None, :]


def _grid_moments(law: Law, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Design mean and variance per row of the interpolated profile H (n, G') (rows in info order)."""
    if law.support is not None:
        P = np.broadcast_to(law.probs, H.shape)
        m = (P * H).sum(1)
        return m, np.maximum((P * H * H).sum(1) - m * m, 0.0)
    if law.dist == "uniform":                                    # exact for a piecewise-linear profile
        h0, h1 = H[:, :-1], H[:, 1:]
        m = ((h0 + h1) / 2).mean(1)
        e2 = ((h0 * h0 + h0 * h1 + h1 * h1) / 3).mean(1)
        return m, np.maximum(e2 - m * m, 0.0)
    R = law.nodes.shape[0]
    nodes = np.broadcast_to(law.nodes, (len(H), law.nodes.shape[1])) if R == 1 else law.nodes
    Hn = _interp(law, H, nodes.T).T                              # (rows, Q)
    wq = np.broadcast_to(law.wq, Hn.shape)
    m = (wq * Hn).sum(1)
    return m, np.maximum((wq * Hn * Hn).sum(1) - m * m, 0.0)


def _interp(law: Law, H: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Profile values at centred draws V (..., n) (n = rows of H); cat: exact support lookup; cont: linear, clamped."""
    n, G = H.shape
    rows = np.arange(n)
    if law.support is not None:
        mids = (law.support[1:] + law.support[:-1]) / 2
        idx = np.searchsorted(mids, V + law.off)
        return H[rows, idx]
    span = law.ghi - law.glo
    s = np.clip((V - law.glo) / np.where(span > 0, span, 1.0), 0.0, 1.0) * (G - 1)
    i = np.minimum(s.astype(np.int64), G - 2)
    f = s - i
    i += rows * G                                                # flat index into H (n, G)
    h0 = np.take(H, i)
    return h0 + f * (np.take(H, i + 1) - h0)


def gbm_profiles(Xfull: np.ndarray, Y: np.ndarray, laws: dict[int, Law], cols: dict[int, int], cfg: PmrtNlConfig,
                 bad_t: np.ndarray) -> tuple[dict, dict, dict]:
    """Past-only gbm per target, refitted at the geometric block boundaries. Returns per focal action index:
    H (n, G', k) centred profile (0 in the burn-in), m (n, k) design mean (the adjustment), and info."""
    n, d = Xfull.shape
    k = Y.shape[1]
    burn = int(max(cfg.min_burn, math.ceil(cfg.burn_per_cov * (d + 1))))
    grids = {ai: _grid(law, cfg.gbm_grid) for ai, law in laws.items()}
    H = {ai: np.zeros((n, g.shape[1], k)) for ai, g in grids.items()}
    Mn = {ai: np.zeros((n, k)) for ai in laws}
    Vr = {ai: np.zeros((n, k)) for ai in laws}
    if burn >= n:
        return H, Mn, {"burn": burn, "n_refits": 0, "d": d, "Var": Vr}
    bounds = [burn]
    while bounds[-1] < n:
        bounds.append(min(n, max(bounds[-1] + 1, int(math.ceil(bounds[-1] * cfg.growth)))))
    for kk in range(k):
        if bad_t[kk]:
            continue
        for i in range(len(bounds) - 1):
            a, b = bounds[i], bounds[i + 1]
            model = _gbm_fit(Xfull[:a], Y[:a, kk], cfg)
            for ai, law in laws.items():
                g = grids[ai][a:b]                                              # (m, G')
                m_, G_ = g.shape
                Xr = np.repeat(Xfull[a:b], G_, axis=0)
                Xr[:, cols[ai]] = (law.off[a:b, None] + g).reshape(-1)
                Hb = model.inplace_predict(Xr).reshape(m_, G_)
                sub = Law(law.v[a:b], law.var[a:b], None, law.off[a:b],
                          law.nodes if law.nodes.shape[0] == 1 else law.nodes[a:b],
                          law.wq if law.wq.shape[0] == 1 else law.wq[a:b], law.dist,
                          None if law.glo is None else law.glo[a:b], None if law.ghi is None else law.ghi[a:b],
                          law.support, None if law.probs is None else
                          (law.probs if law.probs.shape[0] == 1 else law.probs[a:b]))
                mu, var = _grid_moments(sub, Hb)
                H[ai][a:b, :, kk] = Hb - mu[:, None]
                Mn[ai][a:b, kk] = mu
                Vr[ai][a:b, kk] = var
    return H, Mn, {"burn": burn, "n_refits": len(bounds) - 1, "d": d, "Var": Vr}


class GbmStat:
    """T(V) = sum_t h_t(v_t) w_t / sqrt(sum_t Var_design(h_t) w_t^2), per target."""

    def __init__(self, law: Law, H: np.ndarray, Var: np.ndarray, W: np.ndarray):
        self.law, self.H, self.W = law, H, W
        sd = np.sqrt((Var * W * W).sum(0))
        self.sd = np.where(sd > 0, sd, 1.0)
        self.k = W.shape[1]
        self.Hk = [np.ascontiguousarray(H[:, :, kk]) for kk in range(self.k)]

    def __call__(self, V: np.ndarray) -> np.ndarray:
        out = np.empty((len(V), self.k))
        for kk in range(self.k):
            out[:, kk] = (_interp(self.law, self.Hk[kk], V) @ self.W[:, kk]) / self.sd[kk]
        return out


# ============================================================================================ CRT (one-sided)
def crt_stat(stat, v_obs: np.ndarray, draw, rng, B: int, chunk: int, seq_h: int | None) -> dict:
    """One-sided CRT of a statistic T(V) -> (b, k) (large = evidence), Besag-Clifford stopping as pmrt_core.crt:
    a column stops at the draw k where its exceedance count reaches h (p = h / k), else p = (1 + count) / (B + 1).
    Draws are consumed in stream order, so the result is chunk-invariant."""
    t_obs = stat(v_obs[None, :])[0]
    m = len(t_obs)
    tol = 1e-9 * (1.0 + np.abs(t_obs))
    cnt = np.zeros(m, np.int64)
    k_stop = np.zeros(m, np.int64)
    done = 0
    while done < B and not (seq_h is not None and np.all(k_stop > 0)):
        b = min(chunk, B - done)
        E = stat(draw(rng, b)) >= (t_obs - tol)[None, :]
        if seq_h is None:
            cnt += E.sum(0)
        else:
            cs = cnt[None, :] + np.cumsum(E, axis=0, dtype=np.int64)
            hit = (k_stop == 0) & (cs[-1] >= seq_h)
            if hit.any():
                k_stop[hit] = done + 1 + np.argmax(cs[:, hit] >= seq_h, axis=0)
            cnt = cs[-1]
        done += b
    if seq_h is None:
        p, used = (1 + cnt) / (B + 1.0), np.full(m, done)
    else:
        p = np.where(k_stop > 0, seq_h / np.maximum(k_stop, 1), (1 + cnt) / (B + 1.0))
        used = np.where(k_stop > 0, k_stop, done)
    return {"t": t_obs, "p": p, "draws": used, "draws_total": done}


# ============================================================================================ method
def config_from_dict(cfg: dict | None) -> PmrtNlConfig:
    cfg = dict(cfg or {})
    fields = {f.name for f in dataclasses.fields(PmrtNlConfig)}
    kw = {k: (tuple(v) if k == "lams" else v) for k, v in cfg.items() if k in fields}
    return PmrtNlConfig(**kw)


class PmrtNl:
    """api.Method: PMRT core with a nonlinear statistic (module docstring)."""
    name = "pmrt_nl"
    version = PMRT_NL_VERSION

    def __init__(self, cfg: PmrtNlConfig | None = None):
        self.cfg = cfg or PmrtNlConfig()

    def tune(self, dev: list[api.Dataset], truth_free: bool = True) -> dict[str, Any]:
        from cdd_oran.xmethod.score import placebo_tau, tau_rule_name
        cfg = dataclasses.asdict(self.cfg)
        results = [self.run(d, cfg) for d in dev]
        cfg["tau"] = float(placebo_tau(results))
        cfg["tau_rule"] = tau_rule_name()
        return cfg

    def run(self, data: api.Dataset, config: dict[str, Any] | None = None) -> api.Result:
        t_cpu = time.process_time()
        cfg = config_from_dict(config) if config is not None else self.cfg
        if cfg.statistic not in STATISTICS or cfg.adjustment not in ADJUSTMENTS:
            raise ValueError(f"statistic {cfg.statistic!r} / adjust {cfg.adjustment!r} unknown")
        if cfg.covariates not in PC.COVARIATE_SETS:
            raise ValueError(f"covariates must be one of {PC.COVARIATE_SETS}, got {cfg.covariates!r}")
        order = info_order(data)
        kpis = list(data.kpi_names)
        Y = np.asarray(data.Y, float)[order]
        bad_t = ~np.all(np.isfinite(Y), 0) | (np.nanstd(Y, 0) <= 0)
        Yf = np.where(np.isfinite(Y), Y, 0.0)
        eq = cfg.covariates == "eq"
        cov_names, M, _ = design_covariates(data, include_setpoints=eq, include_lagged_actions=eq, lags=cfg.hist_lags)
        Xc = M[order]
        A_ord = np.asarray(data.X_action, float)[order]
        laws, na = {}, {}
        for ai, a in enumerate(data.action_names):
            law = design_law(data.designs[ai], np.asarray(data.X_action, float)[:, ai], order, cfg.n_quad)
            if isinstance(law, str):
                na[a] = law
            elif not np.any(law.var > 0):
                na[a] = "zero design variance"
            else:
                laws[ai] = law
        designed = [j for j, d in enumerate(data.designs) if d.kind in DESIGNED_KINDS]
        gb = None
        if laws and (cfg.adjustment == "gbm" or cfg.statistic == "gbm"):
            use = designed if cfg.concurrent_actions else []
            use = sorted(set(use) | set(laws))
            Xfull = np.column_stack([Xc, A_ord[:, use]])
            cols = {ai: Xc.shape[1] + use.index(ai) for ai in laws}
            gb = gbm_profiles(Xfull, Yf, laws, cols, cfg, bad_t)
        per_action, info = {}, {}
        for ai, law in laws.items():
            a = data.action_names[ai]
            if cfg.adjustment == "ridge":
                conc = PC.concurrent_columns(data, ai) if cfg.concurrent_actions else []
                X = np.column_stack([Xc, A_ord[:, conc]]) if conc else Xc
                W, info[a] = PC.predictable_weights(X, Yf, PC.PmrtCoreConfig(
                    min_burn=cfg.min_burn, burn_per_cov=cfg.burn_per_cov, growth=cfg.growth, lams=cfg.lams))
            else:
                burn = gb[2]["burn"]
                W = Yf - gb[1][ai]
                W[:burn] = 0.0
                info[a] = {k_: v_ for k_, v_ in gb[2].items() if k_ != "Var"}
            W = np.where(bad_t[None, :], 0.0, W)
            if cfg.statistic == "gbm":
                stat = GbmStat(law, gb[0][ai], gb[2]["Var"][ai], W)
                df = None
            else:
                feats = _Features(law, cfg.statistic, cfg.poly_degree, cfg.rff_dv, ai)
                if cfg.statistic == "poly":
                    psi = _psi_setpoint(data.designs[ai], order)
                else:
                    conc = PC.concurrent_columns(data, ai) if cfg.concurrent_actions else []
                    parts = [A_ord[:, conc]]
                    d = data.designs[ai]
                    if d.kind == "dither" and d.fixed_part is not None:
                        parts.append(np.asarray(d.fixed_part, float)[order][:, None])
                    if data.context is not None:
                        parts.append(np.asarray(data.context, float).reshape(data.n, -1)[order])
                    psi = _psi_rff(np.column_stack(parts) if parts else np.zeros((data.n, 0)), cfg.rff_dx, ai)
                stat = FeatureStat(feats, psi, W, whiten=cfg.statistic == "poly")
                df = stat.df
            per_byte = 8 * data.n * max(getattr(stat, "J", 1) * 2, len(kpis) + 2)
            chunk = int(max(1, min(cfg.chunk, cfg.max_chunk_bytes // per_byte)))
            rng = np.random.default_rng([int(data.seed), PC.RNG_TAG, ai, int(cfg.split)])
            r = crt_stat(stat, law.v, lambda g, b, dr=law.draw: dr(g, b)[:, order], rng, cfg.B, chunk, cfg.seq_h)
            r["S"] = law.v @ W
            r["df"] = df
            per_action[a] = r
        stats, rows = {}, []
        for src, tgt in data.candidates:
            if src not in per_action:
                rows.append((src, tgt, None, na.get(src, "source is not an action with a known design")))
                continue
            k = kpis.index(tgt)
            if bad_t[k]:
                rows.append((src, tgt, None, "degenerate target"))
                continue
            r = per_action[src]
            st = FL.HypStats(p2=float(r["p"][k]), sign=int(np.sign(r["S"][k])), p_plus=float(r["p"][k]),
                             p_minus=float(r["p"][k]))
            stats[(src, tgt)] = st
            rows.append((src, tgt, (float(r["t"][k]), st), None))
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
            PC.NOT_APPLICABLE: reasons,
            "statistic": cfg.statistic, "adjust": cfg.adjustment, "one_sided": True,
            "sign_rule": "sign(sum_t v_t w_t) (linear PMRT sign on the same w)",
            "T": {f"{s}->{t}": float(per_action[s]["t"][kpis.index(t)]) for s, t in data.candidates if s in per_action},
            "df": {s: per_action[s]["df"] for s in per_action},
            "draws": {s: int(per_action[s]["draws_total"]) for s in per_action},
            "adjust_info": info, "covariates": {"set": cfg.covariates, "base": list(cov_names)},
            "n_tested": len(stats), "by_family_m": len(family), "by_diag_m": len(diag),
            "declared_tau": tau_decl if cfg.tau is not None else None,
            "rng": f"default_rng([seed, {PC.RNG_TAG}, action_index, {cfg.split}])",
        }
        return api.Result(method=self.name, version=self.version, edges=tuple(edges),
                          cpu_s=time.process_time() - t_cpu, config=dataclasses.asdict(cfg), notes=notes)


def make(**kw) -> PmrtNl:
    return PmrtNl(PmrtNlConfig(**kw))


__all__ = ["ADJUSTMENTS", "PMRT_NL_VERSION", "STATISTICS", "GbmStat", "FeatureStat", "Law", "PmrtNl", "PmrtNlConfig",
           "config_from_dict", "crt_stat", "design_law", "gbm_profiles", "make"]
