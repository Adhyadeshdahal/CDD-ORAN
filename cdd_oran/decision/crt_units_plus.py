"""mscr-crt-units-plus (MSCR+, agent S, 2026-09-30, NOT frozen): a predictable, learned-kernel, design-based CRT
statistic family on the E6-P unit table. Research module: every learned ingredient is fitted on TRAINING episodes
(ev2 sub "v2" + DEV) and frozen before it touches the evaluation episodes.

Setting / validity (agent V, eprocess_units): a family f's units, taken in information order (episode, t0, c), have
modes W_u drawn from the logged pi0 given the past, so the design-centred regressor v_u = sgn_u (L(W_u) - E_pi0 L)
(L = crt_units_v2.LEVEL_V2: accept / half 1, reject 0) has E[v_u | F_{u-1}] = 0 whatever the skeleton does next.
Under the sharp null H0(f, rel, kpi) the KPI series of the target is invariant to f's uniforms. For ANY weights w_u
that are predictable (functions of F_{u-1}: the unit's own key / cell / t0 / sgn, its pre-window of ANY KPI, its
obs-only ctx at t0, past units' modes and outcomes, fixed-in-advance parameters) and of the null-invariant target
series, S = sum_u v_u w_u is a martingale with predictable variance sum_u Var_pi0(v_u) w_u^2 = the variance of the
CRT's i.i.d. re-draw on the realised skeleton; the CRT p-value is therefore asymptotically valid (martingale CLT) even
though the skeleton depends on past modes. What is NOT allowed (and not used here): stratum means / slopes over the
realised unit set (crt_units_v2's episode x sgn FE), later units' existence / modes, other families' units at
t >= t0_u (their existence may depend on W_u), n_req / n_applied.

Unit features (``PlusData``; ``plus_data`` from the disc_bench cache, which stores cumulative per-cell KPI sums
post_H / pre_H for H = 30, 60, 90, 150): per cell and KPI, POST bins [0,30) [30,60) [60,90) [90,150) s after t0 and PRE
bins [0,30) [30,60) [60,90) [90,150) s before t0 (NaN where the window does not fit the episode); the obs-only ctx;
``hist`` = the design-centred modes of every family's units in (t0 - 150, t0) at the unit's own cell / its exposure
neighbours (past only); t0 / T. The unit set is the crt_units H = 90 set (t0 - 90 >= 0, t0 + 90 <= T), so a unit
without the [90,150) post bin uses the kernel restricted to the observed bins (GLS sub-block, below).

Per hypothesis (f, rel, kpi) - the statistic:
  slots     the relation's cells ranked by a FIXED receiver score rho_f(c_u, c') (training: design-centred z of f's
            effect on the per-cell load, signed by f's pooled nbr-load direction); slots = top-(S-1) cells + the rest
            (own: 1 slot; nbr: 4; far: 2). P_u = post bins of the target KPI per slot (S x 4); Q_u = its pre bins.
  adjust    X_u = [Q_u (pre bin [90,150) imputed 2 x [60,90) + indicator), other KPIs' rel-sum pre (2 bins),
            own-cell pre of every KPI (rel != own), ctx, hist, t0 / T] standardised (training), and B = multi-output
            ridge of P on X within training (episode x sgn) strata (GCV lambda): E_u = P_u - B X_u.
  kernel    K = Sigma^-1 mu (GLS matched filter; Sigma = Ledoit-Wolf covariance of the training E), mu per VARIANT:
            "plain" K = 1 on bins [0,90) (the v2 target + learned adjustment), "gls1" mu = 1, "full" mu = design
            estimate v'E / v'v, "rank1" its (slot x time) rank-1 part, "loadsp" spatial profile of the (f, rel, load)
            hypothesis x the target's own time profile. Unit with bin [90,150) missing: K_A = Sigma_AA^-1 mu_A.
  score     k_u = K'E_u (optionally minus a GBDT g-hat(X_u) fitted on training, arm "best_gb"),
            r_u = k_u - m_u, m_u = running mean of past k in the unit's (episode, sgn) stratum shrunk (n0 pseudo-
            units) to the past pooled same-sgn mean, itself shrunk (n1) to the training intercept: PREDICTABLE.
  weights   modifier "" (w = r), "_h" (w = h_u r, h = sigma_hat^-h_pow from a training log-variance ridge model of r
            on X_u, median 1, clipped [0.1, 10]) or "_c" (Huber: w = s_u clip(r / s_u, +-huber_c), s_u the same
            model's predicted sd, calibrated on training). All predictable.
  arms      pred (agent V's predictable residual on the plain y, no learning); VARIANTS x MODS with the extra kernels
            "plain4" (1 on all bins 0-150 s) and "prof" (non-whitened rank-1 profile); best (variant x modifier chosen
            per hypothesis by repeated 2-fold CV of the held-out z on the training episodes; "full" excluded), best_gb;
            combos = exact max of |S_k| / sd_k over <= 3 arms on the SAME draws.
  bench     (ev2 + DEV fit, ev3 test; scratchpad/e6_dev/runs/mscrplus-s-2): loadsp_c / plain_c raise the premise z
            ~1.4x over v2; gls1, plain4, prof LOSE power (whitening / the 90-150 s bin hurt); GBDT adds nothing.
  test      S_k = sum v_u w_uk; sd_k = sqrt(sum Var_pi0(v_u) w_uk^2) (analytic, fixed); T = |S_k| / sd_k (or the max
            over a combo); the family's modes re-drawn from the logged tables (every other family fixed), B draws,
            p = (1 + #{T_b >= T_obs - tol}) / (B + 1). RNG ``default_rng([seed, 6616, PLUS_STREAM, family_idx,
            split])``; chunked, chunk-invariant. BY at q over the tested hypotheses per arm / combo (as v2).
  sign      pred / plain / gls1: sign(S). Kernel arms: sign(S) x sign(1_[0,90)' mu) when the training plain effect
            has |z| >= 2, else the plain arm's sign. Combos: the sign of the arm attaining the max.
Support rule as crt_units_v2 (>= 30 units, >= 5 accepted, >= 5 rejected, >= 3 episodes).
"""
from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import time

import numpy as np

from cdd_oran.discovery import mscr

from . import disc_bench as DB
from .crt_units import CRT_TAG, FAMILIES, LEVEL_ARR, RELATIONS, SPLIT_POOLED, PiAssignment, UnitData
from .crt_units_v2 import LEVEL_V2_ARR
from .eprocess_units import EProcConfig, predictable_residuals, unit_order

PLUS_VERSION = "mscr-crt-units-plus-v0"
PLUS_STREAM = 3
HS4 = (30, 60, 90, 150)
NSLOT = {"own": 1, "nbr": 4, "far": 2}
HIST_WIN = 150
VARIANTS = ("plain", "plain4", "gls1", "full", "rank1", "loadsp", "prof")
MODS = ("", "_h", "_c")
SIGN_FROM_S = ("pred", "plain", "plain4", "gls1")      # kernels whose own sign is the plain-effect sign
ARMS = ("pred",) + tuple(f"{v}{h}" for v in VARIANTS for h in MODS) + ("best", "best_gb")
COMBOS = {"max_pred_best": ("pred", "best"), "max_plain_best": ("plain", "best"),
          "max_pred_best_gb": ("pred", "best", "best_gb"), "max_plain_plain4_best": ("plain", "plain4", "best")}
F32 = np.float32


# ================================================================================================ data
@dataclasses.dataclass
class PlusData:
    ud: UnitData
    post: np.ndarray          # (n, K, C, 4) float32 per-bin sums; NaN where the bin does not fit
    pre: np.ndarray           # (n, K, C, 4)
    ctx: np.ndarray           # (n, J) float64, NaN = missing / non-numeric
    ctx_keys: list
    hist: np.ndarray          # (n, 2 * len(FAMILIES)) past design-centred modes: [own cell per family, nbr per family]
    tfrac: np.ndarray         # (n,) t0 / T

    @property
    def n(self) -> int:
        return self.ud.n


def load_plus_pool(paths, stages=None, H: int = 90) -> DB.Pool:
    """disc_bench.load_pool(H = H_pre = H) + pool.u["post_cum"] / ["pre_cum"] (N, K, C, 4) float32 = the cached
    cumulative sums for H in HS4 (same rows / dedupe as load_pool)."""
    if isinstance(paths, str):
        paths = [p for p in paths.split(",") if p]
    pool = DB.load_pool(paths, H=H, H_pre=H, stages=stages)
    posts, pres, keys = [], [], []
    for path in paths:
        z = np.load(path, allow_pickle=False)
        posts.append(np.stack([z[f"post_H{h}"] for h in HS4], -1).astype(F32))
        pres.append(np.stack([z[f"pre_H{h}"] for h in HS4], -1).astype(F32))
        ep = z["ep"]
        k = list(zip(z["ep_seed"].tolist(), z["ep_sub"].tolist(), z["ep_stage"].tolist(), z["ep_smoke"].tolist(),
                     strict=True))
        keys.append((k, ep))
        z.close()
    seen, keep_u = set(), []
    for k, ep in keys:                                    # load_pool's rule: first file wins, stage filter
        ke = np.array([(key not in seen) and (stages is None or key[2] in stages) for key in k], bool)
        seen.update(k)
        keep_u.append(ke[ep])
    keep_u = np.concatenate(keep_u)
    pool.u["post_cum"] = np.concatenate(posts)[keep_u]
    pool.u["pre_cum"] = np.concatenate(pres)[keep_u]
    if len(pool.u["post_cum"]) != len(pool.u["gep"]):
        raise AssertionError("row alignment with load_pool failed")
    return pool


def pool_rows(pool: DB.Pool, episodes=None):
    """(kept rows, all rows of the episodes) of pool.u in Pool.unit_data's order (same selection rule)."""
    E = pool.eps
    ids = np.arange(pool.n_eps) if episodes is None else np.asarray(episodes, int)
    ids = np.array(sorted(ids.tolist(), key=lambda e: (int(E["seed"][e]), str(E["sub"][e]))), int)
    pos = np.full(pool.n_eps, -1)
    pos[ids] = np.arange(len(ids))
    u = pool.u
    rows_all = np.nonzero(pos[u["gep"]] >= 0)[0]
    rows_all = rows_all[np.argsort(pos[u["gep"][rows_all]], kind="stable")]
    t0 = u["t0"][rows_all].astype(int)
    T = E["T"][u["gep"][rows_all]]
    keep = (t0 - pool.H_pre >= 0) & (t0 + pool.H <= T)
    return rows_all[keep], rows_all


def _bins(cum: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """cumulative (n, K, C, 4) -> per-bin sums, NaN where the cumulative window H does not fit."""
    b = np.empty_like(cum)
    b[..., 0] = cum[..., 0]
    b[..., 1:] = cum[..., 1:] - cum[..., :-1]
    b[~ok[:, None, None, :].repeat(cum.shape[1], 1).repeat(cum.shape[2], 2)] = np.nan
    return b


def v_design(mode, probs, sgn) -> np.ndarray:
    m = probs @ LEVEL_V2_ARR
    return sgn * (LEVEL_V2_ARR[mode] - m)


def v_var(probs) -> np.ndarray:
    m = probs @ LEVEL_V2_ARR
    return probs @ LEVEL_V2_ARR ** 2 - m ** 2


def _hist(pool: DB.Pool, rows, rows_all) -> np.ndarray:
    """Past (t0 - HIST_WIN <= t0' < t0) design-centred modes per family at the unit's own cell / exposure neighbours."""
    u = pool.u
    nf = len(FAMILIES)
    out = np.zeros((len(rows), 2 * nf))
    va = v_design(u["mode"][rows_all].astype(int), u["probs"][rows_all], u["sgn"][rows_all])
    gep_a = u["gep"][rows_all]
    t_a, c_a, f_a = u["t0"][rows_all].astype(int), u["c"][rows_all].astype(int), u["family"][rows_all].astype(int)
    gep_k = u["gep"][rows]
    starts = {}
    for e in np.unique(gep_a):
        starts[int(e)] = np.nonzero(gep_a == e)[0]
    pos_k = {}
    for i, e in enumerate(gep_k.tolist()):
        pos_k.setdefault(e, []).append(i)
    for e, ks in pos_k.items():
        ia = starts[e]
        ks = np.array(ks)
        tk = u["t0"][rows[ks]].astype(int)
        ck = u["c"][rows[ks]].astype(int)
        expk = u["exp"][rows[ks]]
        dt = tk[:, None] - t_a[ia][None, :]                            # (nk, na)
        past = (dt > 0) & (dt <= HIST_WIN)
        same = ck[:, None] == c_a[ia][None, :]
        nbr = expk[:, c_a[ia]] & ~same
        for fi in range(nf):
            wf = past & (f_a[ia] == fi)[None, :]
            out[ks, fi] = (wf & same) @ va[ia]
            out[ks, nf + fi] = (wf & nbr) @ va[ia]
    return out


def plus_data(pool: DB.Pool, episodes=None) -> PlusData:
    """PlusData of the given global episode ids (UnitData identical to pool.unit_data(episodes))."""
    ud = pool.unit_data(episodes)
    rows, rows_all = pool_rows(pool, episodes)
    u = pool.u
    if not (np.array_equal(u["t0"][rows], ud.t0) and np.array_equal(u["c"][rows].astype(int), ud.c)):
        raise AssertionError("PlusData rows do not align with the UnitData")
    t0 = u["t0"][rows].astype(int)
    T = pool.eps["T"][u["gep"][rows]]
    ok_post = np.stack([t0 + h <= T for h in HS4], 1)
    ok_pre = np.stack([t0 - h >= 0 for h in HS4], 1)
    post = _bins(u["post_cum"][rows], ok_post)
    pre = _bins(u["pre_cum"][rows], ok_pre)
    ctx = np.where(u["ctx_num"][rows], u["ctx"][rows], np.nan)
    return PlusData(ud=ud, post=post, pre=pre, ctx=ctx, ctx_keys=list(pool.ctx_keys),
                    hist=_hist(pool, rows, rows_all), tfrac=t0 / T.astype(float))


def subset_plus(pd: PlusData, rows) -> PlusData:
    """PlusData restricted to ``rows`` (UnitData.subset + the same rows of every per-unit array)."""
    rows = np.asarray(rows)
    if rows.dtype == bool:
        rows = np.nonzero(rows)[0]
    return PlusData(ud=pd.ud.subset(rows), post=pd.post[rows], pre=pd.pre[rows], ctx=pd.ctx[rows],
                    ctx_keys=list(pd.ctx_keys), hist=pd.hist[rows], tfrac=pd.tfrac[rows])


def cache_records(recs, out_path: str, src: str = "records") -> str:
    """disc_bench.build_cache for in-memory episode records (e.g. null-outcome variants); uncompressed npz."""
    parts = [DB.episode_arrays(r) for r in recs]
    keys = sorted({k for p in parts for c in p["_ctx"] for k in c})
    kx = {k: j for j, k in enumerate(keys)}
    arrs = {name: np.concatenate([p[name] for p in parts]) for name in parts[0] if not name.startswith("_")}
    arrs["ep"] = np.concatenate([np.full(len(p["c"]), e, np.int32) for e, p in enumerate(parts)])
    n = len(arrs["ep"])
    ctx = np.full((n, len(keys)), np.nan)
    num = np.zeros((n, len(keys)), bool)
    i = 0
    for p in parts:
        for c in p["_ctx"]:
            for k, v in c.items():
                if v is not None:
                    ctx[i, kx[k]] = v
                    num[i, kx[k]] = True
            i += 1
    arrs["ctx"], arrs["ctx_num"], arrs["ctx_keys"] = ctx, num, np.array(keys, dtype="<U64")
    for f in ("seed", "sub", "fold", "stage", "policy", "smoke", "n_units_logged", "T"):
        arrs[f"ep_{f}"] = np.array([p["_ep"][f] for p in parts])
    arrs["src"] = np.array(src)
    arrs["schema"] = np.array(DB.CACHE_SCHEMA)
    arrs["Hs"] = np.array(DB.HS, np.int32)
    arrs["n_cells"] = np.array(parts[0]["_ep"]["n_cells"], np.int32)
    np.savez(out_path, **arrs)
    return out_path


def plus_data_from_records(recs, tmpdir: str | None = None) -> PlusData:
    """PlusData of in-memory records (one temporary npz)."""
    d = tmpdir or tempfile.mkdtemp(prefix="mscrplus_")
    f = os.path.join(d, f"rec_{os.getpid()}_{time.time_ns()}.npz")
    cache_records(recs, f)
    try:
        return plus_data(load_plus_pool([f]))
    finally:
        try:
            os.remove(f)
        except OSError:
            pass


# ================================================================================================ features
def slot_matrix(c, rel_mask, score, S) -> np.ndarray:
    """(n, S, C) indicator: slot j < S-1 = the j-th relation cell by descending score[c_u, c'] (ties: lower cell
    index), slot S-1 = the remaining relation cells. S == 1: all relation cells."""
    n, C = rel_mask.shape
    A = np.zeros((n, S, C), F32)
    if S == 1:
        A[:, 0] = rel_mask
        return A
    sc = np.where(rel_mask, score[c] - 1e-9 * np.arange(C)[None, :], -np.inf)
    order = np.argsort(-sc, axis=1, kind="stable")
    ar = np.arange(n)
    for j in range(S - 1):
        cj = order[:, j]
        ok = rel_mask[ar, cj]
        A[ar[ok], j, cj[ok]] = 1.0
    A[:, S - 1] = rel_mask & (A[:, : S - 1].sum(1) == 0)
    return A


def hyp_design(pd: PlusData, rows, f, rel, kpi, rho, ctx_keys=None) -> tuple:
    """(P (n, S*4) post target bins per slot (NaN where missing), X (n, d) raw predictable covariates, okp (n, 4))."""
    ud = pd.ud
    ki = list(ud.kpis).index(kpi)
    S = NSLOT[rel]
    mask = ud.rel_mask[rel][rows]
    A = slot_matrix(ud.c[rows], mask, rho[f], S)
    post = pd.post[rows, ki]                                          # (n, C, 4)
    okp = np.isfinite(post[:, 0, :])
    P = np.einsum("usc,ucb->usb", A, np.nan_to_num(post)).astype(float)
    P[~np.repeat(okp[:, None, :], S, 1)] = np.nan
    pre = pd.pre[rows]                                                # (n, K, C, 4)
    okq = np.isfinite(pre[:, 0, 0, :])
    pre = np.nan_to_num(pre)
    Q = np.einsum("usc,ucb->usb", A, pre[:, ki]).astype(float)
    miss = ~okq[:, 3]
    Q[miss, :, 3] = 2.0 * Q[miss, :, 2]
    cols = [Q.reshape(len(rows), -1)]
    relm = mask.astype(F32)
    for j, k in enumerate(ud.kpis):
        if j == ki:
            continue
        s = np.einsum("uc,ucb->ub", relm, pre[:, j]).astype(float)
        cols.append(np.column_stack([s[:, 0], s[:, 1] + s[:, 2]]))
    if rel != "own":
        own = ud.rel_mask["own"][rows].astype(F32)
        cols.append(np.einsum("uc,ukcb->uk", own, pre[:, :, :, :3]).astype(float))
    cols += [align_ctx(pd, rows, ctx_keys), pd.hist[rows], pd.tfrac[rows, None], miss[:, None].astype(float)]
    return P.reshape(len(rows), -1), np.column_stack(cols), okp


def align_ctx(pd: PlusData, rows, keys=None) -> np.ndarray:
    """ctx columns of ``rows`` in the order of ``keys`` (training keys; a key absent here -> NaN)."""
    if keys is None or list(keys) == list(pd.ctx_keys):
        return pd.ctx[rows]
    out = np.full((len(rows), len(keys)), np.nan)
    ix = {k: j for j, k in enumerate(pd.ctx_keys)}
    for j, k in enumerate(keys):
        if k in ix:
            out[:, j] = pd.ctx[rows, ix[k]]
    return out


def strata(ud: UnitData, rows) -> np.ndarray:
    return ud.episode[rows].astype(np.int64) * 2 + (ud.sgn[rows] > 0)


def demean(a: np.ndarray, st: np.ndarray) -> np.ndarray:
    _, inv = np.unique(st, return_inverse=True)
    k = int(inv.max()) + 1
    n = np.bincount(inv, minlength=k).astype(float)
    if a.ndim == 1:
        return a - (np.bincount(inv, a, k) / n)[inv]
    out = np.empty_like(a)
    for j in range(a.shape[1]):
        out[:, j] = a[:, j] - (np.bincount(inv, a[:, j], k) / n)[inv]
    return out


def running_center(k: np.ndarray, episode, sgn, n0: float, n1: float, prior) -> np.ndarray:
    """k - m (predictable): m_u = (sum of past k in u's (episode, sgn) stratum + n0 * pm_u) / (count + n0),
    pm_u = (sum of past k with the same sgn, any episode + n1 * prior[sgn]) / (count + n1). Arrays in information
    order; k (n,) or (n, m); prior {+1: (m,), -1: (m,)}."""
    k2 = k[:, None] if k.ndim == 1 else k
    g = (np.asarray(sgn) > 0).astype(np.int64)
    pr = np.where(g[:, None] == 1, np.asarray(prior[1], float)[None, :], np.asarray(prior[-1], float)[None, :])

    def past(key):
        order = np.argsort(key, kind="stable")
        ks = key[order]
        v = k2[order]
        cs = np.empty_like(v)
        cnt = np.empty(len(ks))
        start = np.r_[0, np.nonzero(ks[1:] != ks[:-1])[0] + 1] if len(ks) else np.zeros(0, int)
        for a, b in zip(start, np.r_[start[1:], len(ks)], strict=True):   # per group: no cross-group round-off
            cs[a:b] = np.cumsum(v[a:b], 0) - v[a:b]
            cnt[a:b] = np.arange(b - a)
        s_out, c_out = np.empty_like(cs), np.empty_like(cnt)
        s_out[order], c_out[order] = cs, cnt
        return s_out, c_out

    ps, pc = past(g)
    pm = (ps + n1 * pr) / (pc[:, None] + n1)
    ss, sc = past(np.asarray(episode).astype(np.int64) * 2 + g)
    m = (ss + n0 * pm) / (sc[:, None] + n0)
    r = k2 - m
    return r[:, 0] if k.ndim == 1 else r


# ================================================================================================ fitting
@dataclasses.dataclass(frozen=True)
class PlusConfig:
    B: int = 9999
    q: float = 0.05
    alpha: float = 0.05
    chunk: int = 1000
    max_chunk_bytes: int = 64_000_000
    n0: float = 20.0
    n1: float = 50.0
    cv_reps: int = 3
    cv_penalty: float = 0.25         # CV z handicap of non-"plain" variants (prefer the simple target on ties)
    best_exclude: tuple = ("full",)  # variants never chosen as "best" (unconstrained mu: redistribution patterns)
    h_clip: float = 10.0
    h_lam: float = 30.0              # ridge on the log-variance model (relative to n)
    huber_c: float = 2.5             # "_c": clip r at +- huber_c predicted (training-model) sds
    h_pow: float = 1.0               # h = sigma_hat^-h_pow (1: effect ~ noise sd; 2: inverse variance)
    lw: bool = True
    gb: bool = True
    min_units: int = 30
    min_accept: int = 5
    min_reject: int = 5
    min_episodes: int = 3
    seed: int = 0


def _ridge_gcv(X, Y, lams=(1e-3, 1e-2, 1e-1, 1, 10, 100)):
    """Multi-output ridge on centred X (n, d), Y (n, p); lambda (x n) by GCV. Returns (Bmat (d, p), lam)."""
    n = len(X)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    UtY = U.T @ Y
    best = None
    for lam in lams:
        L = lam * n
        f = s ** 2 / (s ** 2 + L)
        fit = U @ (f[:, None] * UtY)
        rss = float(((Y - fit) ** 2).sum())
        df = float(f.sum())
        g = rss / max(n - df, 1.0) ** 2
        if best is None or g < best[0]:
            best = (g, lam, f)
    _, lam, _ = best
    L = lam * n
    Bm = Vt.T @ ((s / (s ** 2 + L))[:, None] * UtY)
    return Bm, lam


def _lw_cov(E):
    """Ledoit-Wolf shrinkage covariance (toward scaled identity on the standardised scale)."""
    n, p = E.shape
    Ec = E - E.mean(0)
    sd = Ec.std(0)
    sd[sd <= 0] = 1.0
    Z = Ec / sd
    Sm = Z.T @ Z / n
    mu = np.trace(Sm) / p
    d2 = float(((Sm - mu * np.eye(p)) ** 2).sum())
    b2 = float(sum(((np.outer(z, z) - Sm) ** 2).sum() for z in Z[:: max(1, n // 2000)])) * max(1, n // 2000) / n ** 2
    b2 = min(b2, d2)
    sh = b2 / d2 if d2 > 0 else 1.0
    C = sh * mu * np.eye(p) + (1 - sh) * Sm
    return C * np.outer(sd, sd), sh


def _kernels(mu, Sigma, p_cols_nobin3):
    """(K_full (p,), K_nobin3 (p,) zero on bin-3 columns): Sigma^-1 mu and the sub-block GLS without bin 3."""
    reg = 1e-8 * np.trace(Sigma) / len(Sigma)
    Kf = np.linalg.solve(Sigma + reg * np.eye(len(Sigma)), mu)
    A = p_cols_nobin3
    Ks = np.zeros_like(mu)
    Ks[A] = np.linalg.solve(Sigma[np.ix_(A, A)] + reg * np.eye(len(A)), mu[A])
    return Kf, Ks


def _rank1(M):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return s[0] * np.outer(U[:, 0], Vt[0])


def _pos_profile(a) -> np.ndarray:
    """Same-signed spatial profile (aggregate-effect cone): a oriented to a positive sum, negatives clipped to 0,
    normalised to unit sum (all-zero -> uniform)."""
    a = np.asarray(a, float)
    a = a * (1.0 if a.sum() >= 0 else -1.0)
    a = np.maximum(a, 0.0)
    return a / a.sum() if a.sum() > 0 else np.full(len(a), 1.0 / len(a))


def _rank1_pos(M, a=None) -> np.ndarray:
    """mu = a (x) t with a same-signed (``_pos_profile``; the rank-1 left vector of M unless given) and t the
    least-squares time profile given a."""
    if a is None:
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        a = U[:, 0]
    a = _pos_profile(a)
    t = a @ M / max(float(a @ a), 1e-300)
    return np.outer(a, t)


def fit_core(P, X, okp, v, st, S, cfg: PlusConfig, load_a=None) -> dict:
    """Learn (fill, standardisation, B, Sigma, mu variants, kernels, intercepts) on training rows."""
    p = P.shape[1]
    fill = np.nanmedian(X, 0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    Xf = np.where(np.isfinite(X), X, fill[None, :])
    mx, sx = Xf.mean(0), Xf.std(0)
    keep = sx > 1e-12
    Xs = (Xf[:, keep] - mx[keep]) / sx[keep]
    full = okp.all(1)
    Pd = demean(P[full], st[full])
    Xd = demean(Xs[full], st[full])
    Bm, lam = _ridge_gcv(Xd, Pd)
    E = Pd - Xd @ Bm
    vf = v[full]
    vv = float(vf @ vf)
    mu_full = E.T @ vf / vv
    if cfg.lw:
        Sigma, sh = _lw_cov(E)
    else:
        Sigma, sh = np.cov(E.T), 0.0
    se_mu = np.sqrt(np.diag(Sigma) / vv)
    bins = np.tile(np.arange(4), S)
    one03 = (bins < 3).astype(float)
    nob3 = np.nonzero(bins < 3)[0]
    Mf = mu_full.reshape(S, 4)
    mus = {"gls1": np.ones(p), "full": mu_full, "rank1": _rank1_pos(Mf).ravel()}
    mus["loadsp"] = _rank1_pos(Mf, load_a).ravel() if (load_a is not None and S > 1) else mus["rank1"]
    Ks = {"plain": (one03.copy(), one03.copy()), "plain4": (np.ones(p), one03.copy())}
    for nm, mu in mus.items():
        Ks[nm] = _kernels(mu, Sigma, nob3)
    Ks["prof"] = (mus["rank1"].copy(), mus["rank1"] * one03)          # non-whitened profile (identity metric)
    # plain-effect direction on training (bins [0, 90))
    z_plain = float(one03 @ mu_full / np.sqrt(max(one03 @ Sigma @ one03 / vv, 1e-300)))
    sdir = {nm: float(np.sign(one03 @ mu)) for nm, mu in mus.items()}
    sdir["prof"] = sdir["rank1"]
    for nm in SIGN_FROM_S:
        sdir[nm] = 1.0
    core = {"fill": fill, "mx": mx, "sx": sx, "keep": keep, "B": Bm, "lam": lam, "Sigma": Sigma, "lw_shrink": sh,
            "mu_full": mu_full, "se_mu": se_mu, "mus": mus, "K": Ks, "S": S, "z_plain": z_plain, "sdir": sdir,
            "load_a": _pos_profile(np.linalg.svd(Mf, full_matrices=False)[0][:, 0]) if S > 1 else np.ones(1)}
    # intercepts (per sgn, raw scale) of each variant's k on training (full-bin units)
    k = variant_scores(core, P, X, okp)
    sg = np.where(np.asarray(st) % 2 == 1, 1, -1)
    core["prior"] = {nm: {1: float(np.nanmean(k[nm][sg > 0])) if (sg > 0).any() else 0.0,
                          -1: float(np.nanmean(k[nm][sg < 0])) if (sg < 0).any() else 0.0} for nm in k}
    return core


def _xs(core, X):
    Xf = np.where(np.isfinite(X), X, core["fill"][None, :])
    kp = core["keep"]
    return (Xf[:, kp] - core["mx"][kp]) / core["sx"][kp]


def variant_scores(core, P, X, okp) -> dict:
    """k per variant: K_A'(P - B X) with K_A the sub-block kernel when bin 3 is missing."""
    E = np.nan_to_num(P) - _xs(core, X) @ core["B"]
    has3 = okp[:, 3]
    out = {}
    for nm, (Kf, Kn) in core["K"].items():
        out[nm] = np.where(has3, E @ Kf, E @ Kn)
    return out


def fit_hmodel(r, X, core, cfg: PlusConfig) -> dict:
    """log-variance ridge of r on the standardised covariates -> h = exp(-pred) / median (training), clipped."""
    Xs = _xs(core, X)
    y = np.log(r ** 2 + 1e-3 * float(np.mean(r ** 2)) + 1e-300)
    ym = y.mean()
    n, d = Xs.shape
    xm = Xs.mean(0)
    Xc = Xs - xm
    w = np.linalg.solve(Xc.T @ Xc + cfg.h_lam * n / 100.0 * np.eye(d), Xc.T @ (y - ym))
    pred = ym + (Xs - xm) @ w
    med = float(np.median(np.exp(-0.5 * cfg.h_pow * pred)))
    kappa = float(np.median(np.abs(r) / np.exp(0.5 * pred)) / 0.6745) or 1.0
    return {"w": w, "xm": xm, "ym": ym, "med": med, "kappa": kappa}


def h_weights(hm, core, X, cfg: PlusConfig) -> np.ndarray:
    pred = hm["ym"] + (_xs(core, X) - hm["xm"]) @ hm["w"]
    return np.clip(np.exp(-0.5 * cfg.h_pow * pred) / hm["med"], 1.0 / cfg.h_clip, cfg.h_clip)


def lin_z(v, w, var):
    sd = float(np.sqrt(var @ (w * w)))
    return float(v @ w / sd) if sd > 0 else 0.0


def split_arm(a: str) -> tuple:
    """"rank1_h" -> ("rank1", "_h")."""
    for m in ("_h", "_c"):
        if a.endswith(m):
            return a[: -len(m)], m
    return a, ""


def modified(r, mod, hm, core, X, cfg) -> np.ndarray:
    """w from the running residual r: "" -> r; "_h" -> h r; "_c" -> Huber s psi_c(r / s) with the predictable
    training scale s = kappa exp(pred / 2) (log-variance model)."""
    if mod == "":
        return r
    if mod == "_h":
        return h_weights(hm, core, X, cfg) * r
    pred = hm["ym"] + (_xs(core, X) - hm["xm"]) @ hm["w"]
    sc = hm["kappa"] * np.exp(0.5 * pred)
    return sc * np.clip(r / sc, -cfg.huber_c, cfg.huber_c)


def _center_all(core, k: dict, ep, sg, cfg) -> dict:
    return {nm: running_center(k[nm], ep, sg, cfg.n0, cfg.n1,
                               {1: [core["prior"][nm][1]], -1: [core["prior"][nm][-1]]}) for nm in k}


def cv_select(P, X, okp, v, var, ud_ep, sgn, st, S, cfg: PlusConfig, load_a=None, rng=None) -> dict:
    """Repeated 2-fold CV over training EPISODES: mean held-out oriented z per (variant, h)."""
    rng = rng or np.random.default_rng([cfg.seed, 6616, 31])
    eps = np.unique(ud_ep)
    scores = {}
    for rep in range(cfg.cv_reps):
        perm = rng.permutation(eps)
        halves = [np.isin(ud_ep, perm[: len(perm) // 2]), np.isin(ud_ep, perm[len(perm) // 2:])]
        for a, b in ((0, 1), (1, 0)):
            A, Bm = halves[a], halves[b]
            if Bm.sum() < 30 or A.sum() < 30 or okp[A].all(1).sum() < 30:
                continue
            core = fit_core(P[A], X[A], okp[A], v[A], st[A], S, cfg, load_a)
            kA = variant_scores(core, P[A], X[A], okp[A])
            rA = _center_all(core, kA, ud_ep[A], sgn[A], cfg)
            kB = variant_scores(core, P[Bm], X[Bm], okp[Bm])
            rB = _center_all(core, kB, ud_ep[Bm], sgn[Bm], cfg)
            for nm in rA:
                o = np.sign(lin_z(v[A], rA[nm], var[A])) if nm in SIGN_FROM_S else 1.0
                o = o or 1.0
                hm = fit_hmodel(rA[nm], X[A], core, cfg)
                for mod in MODS:
                    wB = modified(rB[nm], mod, hm, core, X[Bm], cfg)
                    scores.setdefault(nm + mod, []).append(o * lin_z(v[Bm], wB, var[Bm]))
    mean = {k: float(np.mean(s)) for k, s in scores.items()}
    if not mean:
        return {"cv": {}, "best": "plain"}
    pen = {k: mean[k] - (0.0 if k == "plain" else cfg.cv_penalty) for k in mean
           if not k.startswith(cfg.best_exclude)}
    return {"cv": mean, "best": max(pen, key=pen.get)}


def fit_plus(pd: PlusData, cfg: PlusConfig | None = None, families=FAMILIES, log=None) -> dict:
    """Learn every hypothesis' frozen parameters on TRAINING data ``pd`` (ev2 + DEV). Returns the params dict."""
    cfg = cfg or PlusConfig()
    ud = pd.ud
    params = {"version": PLUS_VERSION, "config": dataclasses.asdict(cfg), "rho": {}, "hyp": {},
              "n_train_units": int(ud.n), "n_train_eps": int(len(np.unique(ud.episode))), "ctx_keys": pd.ctx_keys}
    C = ud.n_cells
    li = list(ud.kpis).index("load")
    for f in families:                                   # receiver table from the per-cell load effect
        rows = ud.rows_of(f)
        v = v_design(ud.mode[rows], ud.probs[rows], ud.sgn[rows])
        y = np.nan_to_num(pd.post[rows, li, :, :3].sum(-1) - pd.pre[rows, li, :, :3].sum(-1)).astype(float)
        y = demean(y, strata(ud, rows))
        z = np.zeros((C, C))
        for c in range(C):
            m = ud.c[rows] == c
            if m.sum() < 5:
                continue
            vv = float(v[m] @ v[m])
            if vv <= 0:
                continue
            b = v[m] @ y[m] / vv
            res = y[m] - np.outer(v[m], b)
            se = np.sqrt((v[m] ** 2) @ (res ** 2)) / vv
            z[c] = np.where(se > 0, b / np.where(se > 0, se, 1), 0.0)
        tot = 0.0
        for c in range(C):
            m = ud.c[rows] == c
            if m.any():
                tot += float(z[c][ud.rel_mask["nbr"][rows][m].any(0)].sum())
        sgn_f = 1.0 if tot >= 0 else -1.0
        params["rho"][f] = (sgn_f * z).tolist()
    rho = {f: np.array(params["rho"][f]) for f in params["rho"]}
    for f in families:
        rows = unit_order(ud, ud.rows_of(f))
        if len(rows) < cfg.min_units:
            continue
        v = v_design(ud.mode[rows], ud.probs[rows], ud.sgn[rows])
        var = v_var(ud.probs[rows])
        st = strata(ud, rows)
        load_a = {}
        for rel in RELATIONS:
            kp_order = ["load"] + [k for k in ud.kpis if k != "load"]
            for kpi in kp_order:
                t = time.time()
                P, X, okp = hyp_design(pd, rows, f, rel, kpi, rho, params["ctx_keys"])
                S = NSLOT[rel]
                la = load_a.get(rel) if kpi != "load" else None
                sel = cv_select(P, X, okp, v, var, ud.episode[rows], ud.sgn[rows], st, S, cfg, la,
                                np.random.default_rng([cfg.seed, 6616, 31, FAMILIES.index(f), RELATIONS.index(rel),
                                                       list(ud.kpis).index(kpi)]))
                core = fit_core(P, X, okp, v, st, S, cfg, la)
                if kpi == "load":
                    load_a[rel] = core["load_a"]
                k = variant_scores(core, P, X, okp)
                r = _center_all(core, k, ud.episode[rows], ud.sgn[rows], cfg)
                hms = {nm: fit_hmodel(r[nm], X, core, cfg) for nm in r}
                zin = {nm: lin_z(v, r[nm], var) for nm in r}
                best = sel["best"]
                bvar, _ = split_arm(best)
                gb = None
                if cfg.gb:
                    gb = _fit_gb(k[bvar], X, st, core, rows_ok=okp.all(1))
                params["hyp"][f"{f}|{rel}|{kpi}"] = {"core": core, "h": hms, "cv": sel["cv"], "best": best,
                                                     "z_in": zin, "gb": gb}
                if log:
                    log(f"fit {f}|{rel}|{kpi}: n {len(rows)} lam {core['lam']:g} lw {core['lw_shrink']:.2f} "
                        f"z_plain {core['z_plain']:.2f} best {best} cv "
                        f"{ {k2: round(v2, 2) for k2, v2 in sel['cv'].items()} } ({time.time() - t:.1f} s)")
    return params


def _fit_gb(k, X, st, core, rows_ok):
    """GBDT g-hat of the within-stratum-demeaned score on the standardised covariates (training)."""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except Exception:                                   # pragma: no cover
        return None
    Xs = _xs(core, X)
    y = demean(k, st)
    m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100,
                                      l2_regularization=1.0, random_state=0)
    m.fit(Xs, y)
    return m


# ================================================================================================ test
def family_columns(pd: PlusData, f: str, params: dict, cfg: PlusConfig, arms=ARMS):
    """(rows (info order), W (n, n_targets * n_arms), meta per target) for family f: the predictable weighted
    residuals w = h r of every arm (column index t * n_arms + a)."""
    ud = pd.ud
    rows = unit_order(ud, ud.rows_of(f))
    rho = {g: np.array(params["rho"][g]) for g in params["rho"]}
    ep, sg = ud.episode[rows], ud.sgn[rows]
    targets = [(r, k) for r in ud.relations for k in ud.kpis]
    cols, meta = [], []
    ecfg = EProcConfig()
    for rel, kpi in targets:
        hp = params["hyp"].get(f"{f}|{rel}|{kpi}")
        y, pre = ud.y[(rel, kpi)][rows], ud.pre[(rel, kpi)][rows]
        if hp is None or not np.all(np.isfinite(y)):
            cols.append(np.zeros((len(rows), len(arms))))
            meta.append(None)
            continue
        core = hp["core"]
        P, X, okp = hyp_design(pd, rows, f, rel, kpi, rho, params["ctx_keys"])
        k = variant_scores(core, P, X, okp)
        r = _center_all(core, k, ep, sg, cfg)
        best = hp["best"]
        bvar, bmod = split_arm(best)
        W = {}
        rp, _ = predictable_residuals(y, pre if np.all(np.isfinite(pre)) else None, ep, sg, ecfg)
        rp = rp.copy()
        rp[: ecfg.burn] = 0.0
        W["pred"] = rp
        for nm in VARIANTS:
            for mod in MODS:
                W[nm + mod] = modified(r[nm], mod, hp["h"][nm], core, X, cfg)
        W["best"] = W[best]
        if hp.get("gb") is not None:
            kg = k[bvar] - hp["gb"].predict(_xs(core, X))
            prior = {1: [core["prior"][bvar][1]], -1: [core["prior"][bvar][-1]]}
            rg = running_center(kg, ep, sg, cfg.n0, cfg.n1, prior)
            W["best_gb"] = modified(rg, bmod, hp["h"][bvar], core, X, cfg)
        else:
            W["best_gb"] = W["best"]
        cols.append(np.column_stack([W[a] for a in arms]))
        sdir = {}
        for a in arms:
            base = split_arm(a)[0]
            if a in ("best", "best_gb"):
                base = bvar
            if base in ("pred",) + SIGN_FROM_S:
                sdir[a] = 1.0                            # the arm's own sign is the plain-effect sign
            elif abs(core["z_plain"]) < 2.0:
                sdir[a] = 0.0                            # 0: use the plain arm's sign
            else:
                sdir[a] = core["sdir"][base] or 0.0
        meta.append({"best": best, "sdir": sdir, "z_plain_train": core["z_plain"]})
    return rows, np.column_stack(cols), targets, meta


def _stream(cfg: PlusConfig, fi: int, split: int) -> np.random.Generator:
    return np.random.default_rng([int(cfg.seed), CRT_TAG, PLUS_STREAM, int(fi), int(split)])


def family_tests_plus(pd: PlusData, f: str, params: dict, cfg: PlusConfig | None = None, split: int = SPLIT_POOLED,
                      arms=ARMS, combos=COMBOS) -> list:
    """CRT of family f for every (target, arm) and combo on one set of B draws. Returns one dict per target."""
    cfg = cfg or PlusConfig()
    ud = pd.ud
    fi = FAMILIES.index(f)
    rows0 = ud.rows_of(f)
    lv = LEVEL_ARR[ud.mode[rows0]]
    n_acc, n_rej = int((lv == 1.0).sum()), int((lv == 0.0).sum())
    n_ep = len(np.unique(ud.episode[rows0]))
    targets = [(r, k) for r in ud.relations for k in ud.kpis]
    base = {"version": PLUS_VERSION, "family": f, "n": len(rows0), "n_accept": n_acc, "n_reject": n_rej,
            "n_episodes": n_ep, "B": cfg.B}
    if (len(rows0) < cfg.min_units or n_acc < cfg.min_accept or n_rej < cfg.min_reject
            or n_ep < cfg.min_episodes):
        return [dict(base, relation=r, kpi=k, status="undetermined", reason="support") for r, k in targets]
    rows, W, targets, meta = family_columns(pd, f, params, cfg, arms)
    na = len(arms)
    v = v_design(ud.mode[rows], ud.probs[rows], ud.sgn[rows])
    var = v_var(ud.probs[rows])
    sd = np.sqrt(var @ (W * W))
    sd_safe = np.where(sd > 0, sd, 1.0)
    s_obs = v @ W
    z_obs = s_obs / sd_safe
    t_obs = np.abs(z_obs)
    ai = {a: i for i, a in enumerate(arms)}
    nt = len(targets)
    cidx = {c: [ai[a] for a in arms_c] for c, arms_c in combos.items()}
    tc_obs = {c: np.array([t_obs[[t * na + i for i in ix]].max() for t in range(nt)]) for c, ix in cidx.items()}
    tol = 1e-9 * (1.0 + t_obs)
    pa = PiAssignment(ud)
    rng = _stream(cfg, fi, split)
    ch = int(max(1, min(cfg.chunk, cfg.max_chunk_bytes // (8 * max(len(rows), 1)))))
    cnt = np.zeros(W.shape[1], np.int64)
    cntc = {c: np.zeros(nt, np.int64) for c in combos}
    mu = ud.probs[rows] @ LEVEL_V2_ARR
    sgv = ud.sgn[rows]
    done = 0
    while done < cfg.B:
        b = min(ch, cfg.B - done)
        M = pa.conditional_draws(rng, f, b, rows=rows, what="mode")
        V = LEVEL_V2_ARR[M]
        del M
        V -= mu[None, :]
        V *= sgv[None, :]
        T = np.abs(V @ W) / sd_safe[None, :]
        del V
        cnt += np.count_nonzero(T >= (t_obs - tol)[None, :], axis=0)
        T3 = T.reshape(b, nt, na)
        for c, ix in cidx.items():
            tc = T3[:, :, ix].max(2)
            cntc[c] += np.count_nonzero(tc >= (tc_obs[c] - 1e-9 * (1 + tc_obs[c]))[None, :], axis=0)
        done += b
    out = []
    for t, (rel, kpi) in enumerate(targets):
        if meta[t] is None:
            out.append(dict(base, relation=rel, kpi=kpi, status="undetermined", reason="degenerate_outcome"))
            continue
        res = dict(base, relation=rel, kpi=kpi, status="tested", best=meta[t]["best"],
                   z_plain_train=meta[t]["z_plain_train"], arms={})
        plain_sign = float(np.sign(s_obs[t * na + ai["plain"]])) if "plain" in ai else 0.0
        for a, i in ai.items():
            j = t * na + i
            sdir = meta[t]["sdir"].get(a, 0.0)
            sgn_ = float(np.sign(s_obs[j])) * sdir if sdir else plain_sign
            res["arms"][a] = {"p": float((1 + cnt[j]) / (cfg.B + 1)), "z": float(z_obs[j]), "sign": int(sgn_)}
        for c, ix in cidx.items():
            jbest = ix[int(np.argmax(t_obs[[t * na + i for i in ix]]))]
            res["arms"][c] = {"p": float((1 + cntc[c][t]) / (cfg.B + 1)), "z": float(t_obs[t * na + jbest]),
                              "sign": res["arms"][arms[jbest]]["sign"], "via": arms[jbest]}
        out.append(res)
    return out


def run_crt_units_plus(pd: PlusData, params: dict, cfg: PlusConfig | None = None, split: int = SPLIT_POOLED,
                       families=FAMILIES) -> dict:
    """The families x relations x KPIs hypotheses for every arm / combo (one CRT per family)."""
    cfg = cfg or PlusConfig()
    t0 = time.time()
    res = []
    for f in families:
        res.extend(family_tests_plus(pd, f, params, cfg, split))
    names = sorted({a for r in res if r["status"] == "tested" for a in r["arms"]})
    return {"version": PLUS_VERSION, "results": res, "arms": names, "wall_s": round(time.time() - t0, 2),
            "n_units": int(pd.n)}


def edges_plus(run: dict, arm: str, q: float = 0.05) -> dict:
    """BY at q over the tested hypotheses for one arm / combo -> {(f, rel, kpi): {"declared", "sign", "p", ...}}."""
    tested = [r for r in run["results"] if r["status"] == "tested" and arm in r["arms"]]
    dec = mscr.by_declare([r["arms"][arm]["p"] for r in tested], q) if tested else []
    out = {}
    for r in run["results"]:
        out[(r["family"], r["relation"], r["kpi"])] = {"declared": False, "sign": 0, "p": float("nan"),
                                                       "status": r["status"], "z_approx": float("nan")}
    for r, d in zip(tested, dec, strict=True):
        a = r["arms"][arm]
        out[(r["family"], r["relation"], r["kpi"])] = {"declared": bool(d), "sign": int(a["sign"]), "p": a["p"],
                                                       "status": "declared" if d else "not_detected",
                                                       "z_approx": a["z"],
                                                       "score": float(-np.log10(a["p"]))}
    return out


# ================================================================================================ params io
def params_to_json(params: dict) -> dict:
    """JSON-safe summary (no GBDT objects) for reports."""
    out = {k: params[k] for k in ("version", "config", "n_train_units", "n_train_eps")}
    out["hyp"] = {h: {"best": p["best"], "cv": p["cv"], "z_in": p["z_in"], "z_plain_train": p["core"]["z_plain"],
                      "lam": p["core"]["lam"], "lw_shrink": p["core"]["lw_shrink"],
                      "mu_full": np.asarray(p["core"]["mu_full"]).tolist(),
                      "se_mu": np.asarray(p["core"]["se_mu"]).tolist()} for h, p in params["hyp"].items()}
    out["rho"] = params["rho"]
    return out


def save_params(params: dict, path: str) -> None:
    import pickle
    with open(path, "wb") as fh:
        pickle.dump(params, fh)
    with open(os.path.splitext(path)[0] + ".json", "w") as fh:
        json.dump(params_to_json(params), fh, indent=1, default=DB._js)


def load_params(path: str) -> dict:
    import pickle
    with open(path, "rb") as fh:
        return pickle.load(fh)


__all__ = ["ARMS", "COMBOS", "MODS", "SIGN_FROM_S", "split_arm", "modified", "HS4", "NSLOT", "PLUS_STREAM", "PLUS_VERSION", "VARIANTS", "PlusConfig", "PlusData",
           "cache_records", "edges_plus", "family_columns", "family_tests_plus", "fit_core", "fit_plus",
           "hyp_design", "load_params", "load_plus_pool", "plus_data", "plus_data_from_records", "pool_rows",
           "run_crt_units_plus", "running_center", "save_params", "slot_matrix", "v_design", "v_var",
           "variant_scores"]

