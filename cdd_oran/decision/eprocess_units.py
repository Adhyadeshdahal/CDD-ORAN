"""mscr-eproc-units-v1: design-based sequential e-process test on the E6-P unit table (the EXACT companion of
mscr-crt-units-v2; MSCR+ agent V, 2026-09-30, not frozen).

Why (the skeleton gap of crt_units / crt_units_v2): which units of a knob family exist, when, and with which request
direction sgn depends on the family's OWN past modes (a reject leaves the request pending -> a new unit ~60 s later; an
accept moves the knob and starts a dwell during which open_rule "feasible" opens no unit). The CRT re-draws every mode
i.i.d. from pi0 on the REALISED unit set, and v2's residual r_u is demeaned over the realised (episode x sgn) stratum, so
the CRT p-value is not exactly valid (asymptotically it is, by a martingale CLT, only for predictable weights). The
placebo cannot reveal this (accept is applied there: the skeleton never depends on the modes).

Setting / null. Episode e, family f. Z = everything except f's pi0 uniforms (plant seed, other families' uniforms).
collect_p draws unit u's mode from the uniform keyed [seed, 6612, c, x_idx, t0]: a fresh coordinate of an i.i.d.
field for every (c, x, t0), and the f-grid of keys is disjoint from the other families' grids. Order f's units in
time (episode by episode; within an episode by (t0, c)); F_u = sigma(Z, W_1..W_u). The key (c, t0), the request
(sgn), the obs-only ctx and the pre-window of unit u are functions of the trajectory before t0_u, i.e. of (Z, W_<u):
F_{u-1}-measurable; the number of units N is a stopping time. Hence W_u | F_{u-1} ~ pi0(x_f) and the DESIGN-CENTRED
assignment v_u = sgn_u (L(W_u) - E_pi0 L), L = crt_units_v2.LEVEL_V2 (accept / half = 1, reject = 0) has
E[v_u | F_{u-1}] = 0 whatever the skeleton does next.
Sharp null H0(f, rel, kpi): the KPI series Y of (rel, kpi) is invariant to f's uniforms, Y = Y(Z) in F_0. (The
knob, requests and later units MAY depend on f's modes: that is not an outcome.)

e-process. M_n = prod_{u <= n} (1 + lambda_u v_u s_u) with lambda_u, s_u F_{u-1}-measurable, s_u in [-1, 1] and
|lambda_u| max|v| <= 1 (so every factor is >= 0). Then E[1 + lambda_u v_u s_u | F_{u-1}] = 1 + lambda_u s_u
E[v_u | F_{u-1}] = 1: M is a nonnegative test martingale under H0, M_N (N bounded) is an e-value and
P(sup_n M_n >= 1/a) <= a (Ville). A mixture over a fixed lambda grid is a test martingale too. Episodes are
independent, so the product runs across episodes in a fixed order.

What s_u may use (the whole point): under H0 anything in F_{u-1}: the unit's OWN outcome window y_u = post - pre (Y is
in F_0, so its post-t0 part is "known" in information order although it is physically later), its pre-window, sgn_u,
ctx_u, the outcomes / modes / keys of the PAST units u' < u, any function of the Y series at any time and cell. NOT
allowed: the existence, timing, sgn or number of LATER units, strata means / slopes / scales computed over the realised
unit set (v2's residual is such a function), the unit's n_req / n_applied (post-assignment skeleton), W_u.
Implemented (``predictable_scores``): r_u = y_u - (a_s + b (pre_u - p_s)), where for u's stratum s = (episode, sgn)
a_s / p_s are the running means of y / pre over the PAST units of s, shrunk with ``prior_n`` pseudo-units to the running
cross-episode same-sgn means, and b is the running pooled within-stratum slope of y on pre (ridge ``ridge``) over past
units; s_u = clip(r_u / (clip sigma_u), -1, 1), sigma_u = sqrt(running mean of past r^2). The first ``burn`` units of
the family have s_u = 0 (factor 1) while the running state warms up.
lambda (``lam``): "grapa" (default, below) or "mix": uniform mixture over +-lam_max 2^-j, j = 0..n_grid-1 (two-sided; ``sign`` = +1 / -1
keeps one side, allowed only when the sign comes from data independent of the test data); "grapa": predictable
plug-in lambda_u = clip(sum_{i<u} v_i s_i / (sum_{i<u} (v_i s_i)^2 + c0), +-lam_max) (two-sided by construction).
Defaults (lam, clip, prior_n) were chosen on DEV (20 eps, e6p-disc-dev-1): to be frozen before any fresh test.
lam_max = lam_frac / max_u |v|_max, where |v|_max = max(m_u, 1 - m_u) over the unit's logged table.

Multiplicity: e-BH (Wang & Ramdas 2022) at q over the tested hypotheses: reject the k* largest e-values,
k* = max{k : e_(k) >= m / (q k)}; valid under arbitrary dependence. Untested hypotheses (support rule of v2) get e = 0
(so the selection can never inflate an e-value). Descriptive: z_equiv = sqrt(2 log max(e, 1)) (the z at which an oracle
Gaussian likelihood ratio reaches e) and the anytime p = min(1, 1 / max_n M_n).

Input: ``crt_units.UnitData`` (built by ``build_unit_data``; this module reads no record field).
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .crt_units import FAMILIES, LEVEL_ARR, UnitData
from .crt_units_v2 import LEVEL_V2_ARR

EPROC_VERSION = "mscr-eproc-units-v1"
LAMS = ("mix", "grapa")


@dataclasses.dataclass(frozen=True)
class EProcConfig:
    q: float = 0.05
    lam: str = "grapa"            # DEV (20 eps): grapa >= mix on every hypothesis checked (.tmp/mscr_plus/V)
    n_grid: int = 8               # mixture: |lambda| = lam_max * 2^-j, j = 0..n_grid-1, both signs
    lam_frac: float = 0.9         # lam_max = lam_frac / max |v|
    c0: float = 1.0               # grapa denominator pseudo-count
    clip: float = 2.0             # s = clip(r / (clip * sigma), -1, 1) (DEV: 1.0 favours strong, 3.0 weak effects)
    burn: int = 10                # first units of the family: s = 0
    prior_n: float = 20.0         # pseudo-units shrinking a stratum's running means to the same-sgn pooled means
    ridge: float = 1e-6           # ridge on the pooled within-stratum pre slope (relative to the pre SS scale)
    use_pre: bool = True
    min_units: int = 30
    min_accept: int = 5
    min_reject: int = 5
    min_episodes: int = 3


# ------------------------------------------------------------------------------------------------ order / pieces
def unit_order(data: UnitData, rows) -> np.ndarray:
    """``rows`` sorted by (episode, t0, c): the information order of the martingale (episode order = table order,
    fixed before the test)."""
    rows = np.asarray(rows, int)
    return rows[np.lexsort((data.c[rows], data.t0[rows], data.episode[rows]))]


def design_v(data: UnitData, rows) -> tuple[np.ndarray, np.ndarray]:
    """(v_u, |v|_max per unit) with v_u = sgn_u (LEVEL_V2[mode_u] - E_pi0u LEVEL_V2)."""
    rows = np.asarray(rows, int)
    m = data.probs[rows] @ LEVEL_V2_ARR
    v = data.sgn[rows] * (LEVEL_V2_ARR[data.mode[rows]] - m)
    return v, np.maximum(m, 1.0 - m)


def predictable_scores(y, pre, episode, sgn, cfg: EProcConfig) -> np.ndarray:
    """s_u in [-1, 1] from PAST units only (+ the unit's own y, pre, episode, sgn); arrays in information order.
    See the module docstring."""
    r, sig = predictable_residuals(y, pre, episode, sgn, cfg)
    s = np.zeros(len(r))
    ok = (np.arange(len(r)) >= cfg.burn) & np.isfinite(sig) & (sig > 0)
    s[ok] = np.clip(r[ok] / (cfg.clip * sig[ok]), -1.0, 1.0)
    return s


def predictable_residuals(y, pre, episode, sgn, cfg: EProcConfig) -> tuple[np.ndarray, np.ndarray]:
    """(r_u, sigma_u): r_u = y_u - a_s - b (pre_u - p_s) with the running (past-only) stratum means a_s / p_s (shrunk
    to the same-sgn pooled running means) and the running pooled within-stratum slope b; sigma_u = RMS of the past
    residuals (NaN before any). Arrays in information order; one O(n) pass."""
    y = np.asarray(y, float)
    pre = np.zeros_like(y) if (pre is None or not cfg.use_pre) else np.asarray(pre, float)
    n = len(y)
    res = np.zeros(n)
    sigs = np.full(n, np.nan)
    st = {}                                   # (episode, sgn) -> [n, mean_y, mean_p]
    pool = {1.0: [0, 0.0, 0.0], -1.0: [0, 0.0, 0.0]}
    cxy = cpp = 0.0                           # pooled within-stratum co-moments (Welford)
    r2, nr = 0.0, 0
    k = float(cfg.prior_n)
    for u in range(n):
        g = 1.0 if sgn[u] > 0 else -1.0
        key = (int(episode[u]), g)
        ns, my, mp = st.get(key, (0, 0.0, 0.0))
        pn, py, pp = pool[g]
        if pn == 0:
            a_y, a_p = (my, mp) if ns else (y[u], pre[u])  # no history at all: r would be 0
        elif ns + k <= 0:
            a_y, a_p = py, pp
        else:
            a_y = (ns * my + k * py) / (ns + k)
            a_p = (ns * mp + k * pp) / (ns + k)
        b = cxy / (cpp + cfg.ridge * max(cpp, 1e-300) + 1e-300) if cpp > 0 else 0.0
        r = y[u] - a_y - b * (pre[u] - a_p)
        res[u] = r
        if nr > 0:
            sigs[u] = np.sqrt(r2 / nr)
        # ---- update the running state with unit u (only now: it is past for u + 1)
        if pn >= cfg.burn:                    # scale from residuals with a warm pool, winsorised at 5 sigma
            rc = min(abs(r), 5.0 * np.sqrt(r2 / nr)) if nr >= 5 else abs(r)
            r2 += rc * rc
            nr += 1
        if ns > 0:
            f = ns / (ns + 1.0)
            cxy += f * (pre[u] - mp) * (y[u] - my)
            cpp += f * (pre[u] - mp) ** 2
        st[key] = (ns + 1, my + (y[u] - my) / (ns + 1), mp + (pre[u] - mp) / (ns + 1))
        pool[g] = [pn + 1, py + (y[u] - py) / (pn + 1), pp + (pre[u] - pp) / (pn + 1)]
    return res, sigs


def lambda_grid(lam_max: float, n_grid: int, sign: int = 0) -> np.ndarray:
    mag = lam_max * 2.0 ** -np.arange(int(n_grid))
    if sign > 0:
        return mag
    if sign < 0:
        return -mag
    return np.concatenate([mag, -mag])


def eprocess_path(v, s, vmax, cfg: EProcConfig, sign: int = 0) -> dict:
    """log M_n path of the test martingale for arrays in information order. Returns {"log_e": final log e-value,
    "log_e_max": log max_n M_n, "log_path": (n,) path}."""
    v, s = np.asarray(v, float), np.asarray(s, float)
    n = len(v)
    if n == 0:
        return {"log_e": 0.0, "log_e_max": 0.0, "log_path": np.zeros(0)}
    lam_max = cfg.lam_frac / float(np.max(vmax))
    x = v * s
    if cfg.lam == "mix":
        lams = lambda_grid(lam_max, cfg.n_grid, sign)
        fac = 1.0 + lams[:, None] * x[None, :]                 # (K, n), >= 1 - lam_frac > 0
        L = np.cumsum(np.log(fac), axis=1)                    # (K, n)
        mx = L.max(0)
        path = mx + np.log(np.exp(L - mx[None, :]).mean(0))
    elif cfg.lam == "grapa":
        num = np.concatenate([[0.0], np.cumsum(x)[:-1]])
        den = np.concatenate([[0.0], np.cumsum(x * x)[:-1]]) + cfg.c0
        lam = np.clip(num / den, -lam_max, lam_max)
        if sign > 0:
            lam = np.maximum(lam, 0.0)
        elif sign < 0:
            lam = np.minimum(lam, 0.0)
        path = np.cumsum(np.log1p(lam * x))
    else:
        raise ValueError(f"lam in {LAMS}")
    return {"log_e": float(path[-1]), "log_e_max": float(max(path.max(), 0.0)), "log_path": path}


def e_bh(e, q: float = 0.05) -> np.ndarray:
    """e-BH (Wang & Ramdas 2022): reject the k* largest e-values, k* = max{k: e_(k) >= m / (q k)}."""
    e = np.asarray(e, float)
    m = len(e)
    out = np.zeros(m, bool)
    if m == 0:
        return out
    order = np.argsort(-e, kind="stable")
    ks = np.arange(1, m + 1)
    ok = e[order] >= m / (q * ks)
    if ok.any():
        out[order[: np.nonzero(ok)[0].max() + 1]] = True
    return out


# ------------------------------------------------------------------------------------------------ test
def family_eprocess(data: UnitData, family: str, targets, cfg: EProcConfig | None = None, signs=None) -> list:
    """e-process of family ``family`` against each target [(name, (rel, kpi))]; ``signs`` optional {name: +1 / -1}
    (one-sided, only from independent data). Returns one dict per target (status "tested" | "undetermined")."""
    cfg = cfg or EProcConfig()
    rows = unit_order(data, data.rows_of(family))
    lv = LEVEL_ARR[data.mode[rows]]
    n_acc, n_rej = int((lv == 1.0).sum()), int((lv == 0.0).sum())
    n_ep = len(np.unique(data.episode[rows]))
    base = {"version": EPROC_VERSION, "family": family, "n": len(rows), "n_accept": n_acc, "n_reject": n_rej,
            "n_episodes": n_ep, "lam": cfg.lam, "e": 0.0, "log_e": float("-inf"), "log_e_max": 0.0, "p_anytime": 1.0,
            "z_equiv": 0.0, "sign": 0, "status": "undetermined", "reason": ""}
    if (len(rows) < cfg.min_units or n_acc < cfg.min_accept or n_rej < cfg.min_reject
            or n_ep < cfg.min_episodes):
        return [dict(base, target=nm, key=list(k), reason="support") for nm, k in targets]
    v, vmax = design_v(data, rows)
    ep, sg = data.episode[rows], data.sgn[rows]
    out = []
    for nm, key in targets:
        y = data.y[tuple(key)][rows]
        pre = data.pre[tuple(key)][rows]
        if not np.all(np.isfinite(y)):
            out.append(dict(base, target=nm, key=list(key), reason="degenerate_outcome"))
            continue
        s = predictable_scores(y, pre if np.all(np.isfinite(pre)) else None, ep, sg, cfg)
        sgn = 0 if signs is None else int(signs.get(nm, 0))
        r = eprocess_path(v, s, vmax, cfg, sgn)
        vs = float(v @ s)
        out.append(dict(base, target=nm, key=list(key), status="tested", log_e=r["log_e"],
                        e=float(np.exp(min(r["log_e"], 700.0))), log_e_max=r["log_e_max"],
                        p_anytime=float(min(1.0, np.exp(-r["log_e_max"]))),
                        z_equiv=float(np.sqrt(2.0 * max(r["log_e"], 0.0))), sign=int(np.sign(vs)),
                        vs=vs, z_s=float(vs / np.sqrt(max(float((v * v) @ (s * s)), 1e-300))),
                        n_scored=int(np.count_nonzero(s))))
    return out


def run_eprocess_units(data: UnitData, cfg: EProcConfig | None = None, families=FAMILIES, signs=None) -> dict:
    """The families x relations x KPIs hypotheses + e-BH at q over the tested ones."""
    cfg = cfg or EProcConfig()
    res = []
    for f in families:
        tg = [(f"{r}_{k}", (r, k)) for r in data.relations for k in data.kpis]
        for r in family_eprocess(data, f, tg, cfg, (signs or {}).get(f)):
            r["relation"], r["kpi"] = r["key"]
            res.append(r)
    tested = [r for r in res if r["status"] == "tested"]
    dec = e_bh([r["e"] for r in tested], cfg.q)
    for r, d in zip(tested, dec, strict=True):
        r["status"] = "declared" if d else "not_detected"
    m = len(tested)
    return {"version": EPROC_VERSION, "config": dataclasses.asdict(cfg), "n_tested": m, "n_declared": int(dec.sum()),
            "ebh_rank1_threshold": (m / cfg.q) if m else None, "results": res,
            "selection": f"e-BH at q={cfg.q:g} over the tested hypotheses (arbitrary dependence)"}


__all__ = ["EPROC_VERSION", "EProcConfig", "design_v", "e_bh", "eprocess_path", "family_eprocess", "lambda_grid",
           "predictable_residuals", "predictable_scores", "run_eprocess_units", "unit_order"]
